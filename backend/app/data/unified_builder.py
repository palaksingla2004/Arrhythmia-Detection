from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from app.core.config import settings
from app.data.labels import MULTICLASS_LABELS
from app.data.loaders import UnifiedSignalSample, load_all_signal_samples


@dataclass
class UnifiedSignalDataset:
    X: np.ndarray
    y_multi: np.ndarray
    y_binary: np.ndarray
    y_quality: np.ndarray
    groups: np.ndarray
    source: np.ndarray
    record_id: np.ndarray
    sampling_rate: int
    input_length: int


def _pad_or_crop(signal: np.ndarray, target_len: int) -> np.ndarray:
    if len(signal) == target_len:
        return signal.astype(np.float32)
    if len(signal) > target_len:
        start = (len(signal) - target_len) // 2
        return signal[start : start + target_len].astype(np.float32)
    pad_left = (target_len - len(signal)) // 2
    pad_right = target_len - len(signal) - pad_left
    return np.pad(signal, (pad_left, pad_right), mode="edge").astype(np.float32)


def _to_numpy_dataset(samples: list[UnifiedSignalSample], input_length: int) -> UnifiedSignalDataset:
    X = np.stack([_pad_or_crop(s.signal, input_length) for s in samples], axis=0).astype(np.float32)
    y_multi = np.array([s.label_id for s in samples], dtype=np.int64)
    y_binary = np.array([s.binary_label for s in samples], dtype=np.int64)
    y_quality = np.array([s.quality_id for s in samples], dtype=np.int64)
    groups = np.array([s.record_id.split("_")[0] for s in samples], dtype=object)
    source = np.array([s.source for s in samples], dtype=object)
    record_id = np.array([s.record_id for s in samples], dtype=object)
    return UnifiedSignalDataset(
        X=X,
        y_multi=y_multi,
        y_binary=y_binary,
        y_quality=y_quality,
        groups=groups,
        source=source,
        record_id=record_id,
        sampling_rate=settings.target_sampling_rate,
        input_length=input_length,
    )


def build_unified_signal_dataset(
    output_npz: Path | None = None,
    output_meta_json: Path | None = None,
    include_ptbxl: bool = True,
    mitbih_raw_max_samples: int = 120_000,
    mitbih_train_max_rows: int | None = 50_000,
    mitbih_test_max_rows: int | None = 20_000,
    ptbdb_normal_max_rows: int | None = 10_000,
    ptbdb_abnormal_max_rows: int | None = 10_000,
    ptbxl_max_records: int = 3000,
) -> UnifiedSignalDataset:
    print("🔧 Building unified signal dataset...")
    sys.stdout.flush()
    
    input_len = int(settings.target_sampling_rate * settings.segment_seconds)
    if output_npz is None:
        output_npz = settings.processed_dir / "unified_signals.npz"
    if output_meta_json is None:
        output_meta_json = settings.processed_dir / "unified_signals_meta.json"

    print(f"📂 Output NPZ: {output_npz}")
    print(f"📂 Output Meta: {output_meta_json}")
    print(f"📏 Input length: {input_len} samples")
    print(f"🔢 Include PTB-XL: {include_ptbxl}")
    print("\n⏳ Loading signal samples (this may take several minutes)...")
    sys.stdout.flush()
    
    samples = load_all_signal_samples(
        include_ptbxl=include_ptbxl,
        mitbih_raw_max_samples=mitbih_raw_max_samples,
        mitbih_train_max_rows=mitbih_train_max_rows,
        mitbih_test_max_rows=mitbih_test_max_rows,
        ptbdb_normal_max_rows=ptbdb_normal_max_rows,
        ptbdb_abnormal_max_rows=ptbdb_abnormal_max_rows,
        ptbxl_max_records=ptbxl_max_records,
    )
    
    print(f"✅ Loaded {len(samples)} signal samples")
    sys.stdout.flush()
    
    if not samples:
        raise RuntimeError("No ECG samples were loaded. Check dataset paths and formats.")

    print("🔄 Converting to numpy dataset...")
    sys.stdout.flush()
    dataset = _to_numpy_dataset(samples, input_len)
    
    print(f"✅ Dataset shape: {dataset.X.shape}")
    sys.stdout.flush()
    
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving compressed dataset to {output_npz}...")
    sys.stdout.flush()
    
    np.savez_compressed(
        output_npz,
        X=dataset.X,
        y_multi=dataset.y_multi,
        y_binary=dataset.y_binary,
        y_quality=dataset.y_quality,
        groups=dataset.groups.astype(str),
        source=dataset.source.astype(str),
        record_id=dataset.record_id.astype(str),
        sampling_rate=np.array([dataset.sampling_rate], dtype=np.int64),
        input_length=np.array([dataset.input_length], dtype=np.int64),
    )
    
    print("✅ Dataset saved")
    print("📊 Computing class weights and metadata...")
    sys.stdout.flush()

    class_names = list(MULTICLASS_LABELS)
    present_classes = np.unique(dataset.y_multi)
    present_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=dataset.y_multi,
    )
    class_weights = np.ones(len(class_names), dtype=np.float32)
    for cls, weight in zip(present_classes, present_weights):
        class_weights[int(cls)] = float(weight)
    class_distribution = {
        class_names[i]: int((dataset.y_multi == i).sum()) for i in range(len(class_names))
    }
    source_distribution = {
        source_name: int((dataset.source == source_name).sum())
        for source_name in sorted(set(dataset.source.tolist()))
    }

    metadata: dict[str, Any] = {
        "num_samples": int(dataset.X.shape[0]),
        "input_length": int(dataset.input_length),
        "sampling_rate": int(dataset.sampling_rate),
        "class_names": class_names,
        "class_distribution": class_distribution,
        "source_distribution": source_distribution,
        "class_weights": class_weights.tolist(),
    }
    output_meta_json.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving metadata to {output_meta_json}...")
    sys.stdout.flush()
    output_meta_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("✅ Metadata saved")
    sys.stdout.flush()
    return dataset


def load_unified_signal_dataset(path: Path | None = None) -> UnifiedSignalDataset:
    if path is None:
        path = settings.processed_dir / "unified_signals.npz"
    payload = np.load(path, allow_pickle=True)
    return UnifiedSignalDataset(
        X=payload["X"].astype(np.float32),
        y_multi=payload["y_multi"].astype(np.int64),
        y_binary=payload["y_binary"].astype(np.int64),
        y_quality=payload["y_quality"].astype(np.int64),
        groups=payload["groups"],
        source=payload["source"],
        record_id=payload["record_id"],
        sampling_rate=int(payload["sampling_rate"][0]),
        input_length=int(payload["input_length"][0]),
    )


def cross_validation_splits(
    dataset: UnifiedSignalDataset,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, valid_idx in splitter.split(dataset.X, dataset.y_multi, groups=dataset.groups):
        splits.append((train_idx, valid_idx))
    return splits
