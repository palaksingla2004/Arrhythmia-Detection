from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
try:
    import wfdb
except Exception:  # pragma: no cover - optional dependency fallback
    wfdb = None

from app.core.config import settings
from app.data.labels import (
    AF_LABEL,
    FEATURE_CLASS_MAP,
    LABEL_TO_ID,
    NORMAL_LABEL,
    OTHER_LABEL,
    map_aux_note_to_label,
    map_feature_type_to_label,
    map_mitbih_numeric_to_label,
    map_ptbdb_numeric_to_label,
    map_ptbxl_codes_to_label,
    map_symbol_to_label,
    parse_scp_codes,
)
from app.data.preprocessing import (
    classify_signal_quality,
    preprocess_signal,
    resample_signal,
    segment_beats_around_r_peaks,
    zscore_normalize,
)


@dataclass
class UnifiedSignalSample:
    signal: np.ndarray
    sampling_rate: int
    label: str
    label_id: int
    binary_label: int
    quality_id: int
    source: str
    record_id: str


@dataclass
class TabularSampleBatch:
    X: np.ndarray
    y: np.ndarray
    source: np.ndarray
    feature_names: list[str]


def _parse_possible_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return []


def _binary_from_label(label: str) -> int:
    return 0 if label == NORMAL_LABEL else 1


def _label_to_id(label: str) -> int:
    return LABEL_TO_ID.get(label, LABEL_TO_ID[OTHER_LABEL])


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def _read_first_lead_signal(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    candidate_cols = [c for c in df.columns if c not in {"", "index", "symbol"}]
    if not candidate_cols:
        raise ValueError(f"No signal columns found in {csv_path}")
    lead = candidate_cols[0]
    return df[lead].astype(np.float32).to_numpy()


def _read_fs_from_record_json(json_path: Path) -> int:
    if not json_path.exists():
        return 360
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    fs = payload.get("fs", 360)
    try:
        return int(fs)
    except (TypeError, ValueError):
        return 360


def _load_annotation_aux_map(annotation_json_path: Path) -> dict[int, str]:
    if not annotation_json_path.exists():
        return {}
    with annotation_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = _parse_possible_list(payload.get("sample"))
    aux_notes = payload.get("aux_note", [])
    if isinstance(aux_notes, str):
        aux_notes = _parse_possible_list(aux_notes)
    if not isinstance(aux_notes, list):
        aux_notes = []

    aux_map: dict[int, str] = {}
    for idx, sample in enumerate(samples):
        if idx >= len(aux_notes):
            continue
        try:
            sample_int = int(sample)
        except (TypeError, ValueError):
            continue
        aux_note = str(aux_notes[idx]).strip() if aux_notes[idx] is not None else ""
        if aux_note:
            aux_map[sample_int] = aux_note
    return aux_map


def _build_af_intervals_from_aux(aux_map: dict[int, str], signal_length: int) -> list[tuple[int, int]]:
    if not aux_map:
        return []
    intervals: list[tuple[int, int]] = []
    in_af = False
    af_start = 0

    for sample_idx in sorted(aux_map.keys()):
        note = str(aux_map[sample_idx]).strip()
        af_label = map_aux_note_to_label(note)
        clean_note = note.rstrip("0123456789")
        if af_label == AF_LABEL and not in_af:
            in_af = True
            af_start = sample_idx
        elif clean_note.startswith("(N") and in_af:
            in_af = False
            intervals.append((af_start, sample_idx))

    if in_af:
        intervals.append((af_start, signal_length))
    return intervals


def _is_in_intervals(value: int, intervals: list[tuple[int, int]]) -> bool:
    for start, end in intervals:
        if start <= value < end:
            return True
    return False


def load_mitbih_raw_samples(
    datasets_dir: Path = settings.datasets_dir,
    max_records: int | None = None,
    max_samples: int | None = None,
) -> Iterator[UnifiedSignalSample]:
    ekg_files = sorted(datasets_dir.glob("*_ekg.csv"))
    if max_records is not None:
        ekg_files = ekg_files[:max_records]

    produced = 0
    for ekg_path in ekg_files:
        record_id = ekg_path.stem.split("_")[0]
        record_json = datasets_dir / f"{record_id}_ekg.json"
        fs = _read_fs_from_record_json(record_json)
        raw_signal = _read_first_lead_signal(ekg_path)

        prep = preprocess_signal(raw_signal, fs)

        # Annotation CSV files may contain multiple channels/extensions.
        annotation_files = sorted(datasets_dir.glob(f"{record_id}_annotations_*.csv"))
        if not annotation_files:
            continue

        aux_map = _load_annotation_aux_map(datasets_dir / f"{record_id}_annotations_1.json")
        af_intervals = _build_af_intervals_from_aux(aux_map, len(raw_signal))

        pre = int(settings.beat_pre_seconds * settings.target_sampling_rate)
        post = int(settings.beat_post_seconds * settings.target_sampling_rate)
        segment_len = pre + post
        if segment_len <= 0:
            continue

        for ann_path in annotation_files:
            ann_df = pd.read_csv(ann_path)
            if "index" not in ann_df.columns or "annotation_symbol" not in ann_df.columns:
                continue

            for row in ann_df.itertuples(index=False):
                sample_idx = int(getattr(row, "index"))
                symbol = str(getattr(row, "annotation_symbol"))
                aux_note = aux_map.get(sample_idx)

                label = map_symbol_to_label(symbol, aux_note=aux_note)
                if _is_in_intervals(sample_idx, af_intervals):
                    label = AF_LABEL

                target_idx = int(round(sample_idx * settings.target_sampling_rate / fs))
                start = target_idx - pre
                end = target_idx + post
                if start < 0 or end > len(prep.signal):
                    continue

                beat = prep.signal[start:end]
                if len(beat) != segment_len:
                    continue

                yield UnifiedSignalSample(
                    signal=beat.astype(np.float32),
                    sampling_rate=settings.target_sampling_rate,
                    label=label,
                    label_id=_label_to_id(label),
                    binary_label=_binary_from_label(label),
                    quality_id=prep.quality_id,
                    source="mitbih_raw",
                    record_id=record_id,
                )
                produced += 1
                if max_samples is not None and produced >= max_samples:
                    return


def _iter_numeric_label_csv_rows(
    csv_path: Path,
    label_mapper,
    source: str,
    assumed_sampling_rate: int = 125,
    max_rows: int | None = None,
) -> Iterator[UnifiedSignalSample]:
    df = pd.read_csv(csv_path, header=None)
    if max_rows is not None:
        df = df.iloc[:max_rows]

    for idx, row in enumerate(df.itertuples(index=False)):
        values = np.asarray(row, dtype=np.float32)
        signal = values[:-1]
        label_value = values[-1]
        label = label_mapper(label_value)

        resampled = resample_signal(signal, assumed_sampling_rate, settings.target_sampling_rate)
        normalized = zscore_normalize(resampled)
        quality_id, _ = classify_signal_quality(resampled, normalized)

        yield UnifiedSignalSample(
            signal=normalized.astype(np.float32),
            sampling_rate=settings.target_sampling_rate,
            label=label,
            label_id=_label_to_id(label),
            binary_label=_binary_from_label(label),
            quality_id=quality_id,
            source=source,
            record_id=f"{source}_{idx}",
        )


def load_mitbih_heartbeat_samples(
    datasets_dir: Path = settings.datasets_dir,
    max_train_rows: int | None = None,
    max_test_rows: int | None = None,
) -> Iterator[UnifiedSignalSample]:
    train_path = datasets_dir / "mitbih_train.csv"
    test_path = datasets_dir / "mitbih_test.csv"
    if train_path.exists():
        yield from _iter_numeric_label_csv_rows(
            train_path,
            label_mapper=map_mitbih_numeric_to_label,
            source="mitbih_train",
            max_rows=max_train_rows,
        )
    if test_path.exists():
        yield from _iter_numeric_label_csv_rows(
            test_path,
            label_mapper=map_mitbih_numeric_to_label,
            source="mitbih_test",
            max_rows=max_test_rows,
        )


def load_ptbdb_samples(
    datasets_dir: Path = settings.datasets_dir,
    max_normal_rows: int | None = None,
    max_abnormal_rows: int | None = None,
) -> Iterator[UnifiedSignalSample]:
    normal_path = datasets_dir / "ptbdb_normal.csv"
    abnormal_path = datasets_dir / "ptbdb_abnormal.csv"
    if normal_path.exists():
        yield from _iter_numeric_label_csv_rows(
            normal_path,
            label_mapper=map_ptbdb_numeric_to_label,
            source="ptbdb_normal",
            max_rows=max_normal_rows,
        )
    if abnormal_path.exists():
        yield from _iter_numeric_label_csv_rows(
            abnormal_path,
            label_mapper=map_ptbdb_numeric_to_label,
            source="ptbdb_abnormal",
            max_rows=max_abnormal_rows,
        )


def _load_ptbxl_arrhythmia_code_set(ptbxl_dir: Path) -> set[str]:
    scp_path = ptbxl_dir / "scp_statements.csv"
    if not scp_path.exists():
        return set()
    scp_df = pd.read_csv(scp_path)
    code_set: set[str] = set()
    for row in scp_df.itertuples(index=False):
        code = str(getattr(row, "Unnamed: 0", getattr(row, "Index", ""))).strip()
        # file has unnamed first column with code when loaded through pandas
        if not code:
            code = str(getattr(row, "_0", "")).strip()
        if not code:
            # fallback for properly read index column
            code = str(row[0]).strip()
        diagnostic = getattr(row, "diagnostic", np.nan)
        rhythm = getattr(row, "rhythm", np.nan)
        if (pd.notna(diagnostic) and float(diagnostic) == 1.0) or (
            pd.notna(rhythm) and float(rhythm) == 1.0
        ):
            code_set.add(code)
    return code_set


def load_ptbxl_samples(
    datasets_dir: Path = settings.datasets_dir,
    max_records: int = 2000,
    seed: int = 42,
) -> Iterator[UnifiedSignalSample]:
    if wfdb is None:
        return
    ptbxl_dir = datasets_dir / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    db_path = ptbxl_dir / "ptbxl_database.csv"
    if not db_path.exists():
        return

    db = pd.read_csv(db_path)
    arrhythmia_codes = _load_ptbxl_arrhythmia_code_set(ptbxl_dir)

    rng = _rng(seed)
    indices = list(range(len(db)))
    rng.shuffle(indices)
    indices = indices[: min(max_records, len(indices))]

    for idx in indices:
        row = db.iloc[idx]
        ecg_id = str(row["ecg_id"])
        codes = parse_scp_codes(row.get("scp_codes"))
        label = map_ptbxl_codes_to_label(codes, arrhythmia_code_set=arrhythmia_codes)

        record_rel_path = row.get("filename_hr") or row.get("filename_lr")
        if not isinstance(record_rel_path, str) or not record_rel_path:
            continue

        record_path = ptbxl_dir / record_rel_path
        try:
            signal_2d, meta = wfdb.rdsamp(str(record_path))
        except Exception:
            continue

        if signal_2d.size == 0:
            continue
        lead = signal_2d[:, 0].astype(np.float32)
        fs = int(meta.get("fs", 500))
        prep = preprocess_signal(lead, fs)

        for seg_idx, segment in enumerate(prep.fixed_segments):
            yield UnifiedSignalSample(
                signal=segment.astype(np.float32),
                sampling_rate=settings.target_sampling_rate,
                label=label,
                label_id=_label_to_id(label),
                binary_label=_binary_from_label(label),
                quality_id=prep.quality_id,
                source="ptbxl",
                record_id=f"{ecg_id}_{seg_idx}",
            )


def load_engineered_feature_batch(
    datasets_dir: Path = settings.datasets_dir,
    sample_per_dataset: int | None = None,
    seed: int = 42,
) -> TabularSampleBatch:
    files = [
        ("MIT-BIH Arrhythmia Database.csv", "mitbih_features"),
        ("MIT-BIH Supraventricular Arrhythmia Database.csv", "mitbih_supra_features"),
        ("INCART 2-lead Arrhythmia Database.csv", "incart_features"),
        ("Sudden Cardiac Death Holter Database.csv", "scd_features"),
    ]
    rng = _rng(seed)
    frames: list[pd.DataFrame] = []

    for filename, source in files:
        path = datasets_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        if "type" not in df.columns:
            continue
        df = df.dropna(subset=["type"])
        if sample_per_dataset is not None and len(df) > sample_per_dataset:
            idx = list(df.index)
            rng.shuffle(idx)
            idx = idx[:sample_per_dataset]
            df = df.loc[idx]
        df = df.copy()
        df["source"] = source
        df["label"] = df["type"].astype(str).map(map_feature_type_to_label)
        frames.append(df)

    if not frames:
        return TabularSampleBatch(
            X=np.empty((0, 0), dtype=np.float32),
            y=np.empty((0,), dtype=np.int64),
            source=np.empty((0,), dtype=object),
            feature_names=[],
        )

    merged = pd.concat(frames, axis=0, ignore_index=True)
    metadata_cols = {"record", "type", "source", "label"}
    feature_cols = [c for c in merged.columns if c not in metadata_cols]
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged["label"].map(_label_to_id).to_numpy(dtype=np.int64)
    source = merged["source"].to_numpy(dtype=object)
    return TabularSampleBatch(X=X, y=y, source=source, feature_names=feature_cols)


def load_all_signal_samples(
    include_ptbxl: bool = True,
    mitbih_raw_max_samples: int = 120_000,
    mitbih_train_max_rows: int | None = 50_000,
    mitbih_test_max_rows: int | None = 20_000,
    ptbdb_normal_max_rows: int | None = 10_000,
    ptbdb_abnormal_max_rows: int | None = 10_000,
    ptbxl_max_records: int = 3000,
) -> list[UnifiedSignalSample]:
    samples: list[UnifiedSignalSample] = []

    samples.extend(
        list(
            load_mitbih_raw_samples(
                max_samples=mitbih_raw_max_samples,
            )
        )
    )
    samples.extend(
        list(
            load_mitbih_heartbeat_samples(
                max_train_rows=mitbih_train_max_rows,
                max_test_rows=mitbih_test_max_rows,
            )
        )
    )
    samples.extend(
        list(
            load_ptbdb_samples(
                max_normal_rows=ptbdb_normal_max_rows,
                max_abnormal_rows=ptbdb_abnormal_max_rows,
            )
        )
    )

    if include_ptbxl:
        samples.extend(list(load_ptbxl_samples(max_records=ptbxl_max_records)))

    return samples
