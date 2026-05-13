from __future__ import annotations

import ast
import re
from collections.abc import Iterable


NORMAL_LABEL = "Normal"
AF_LABEL = "Atrial Fibrillation (AF)"
PVC_LABEL = "Premature Ventricular Contraction (PVC)"
OTHER_LABEL = "Other Arrhythmia"

QUALITY_LABELS = ("Good", "Noisy", "Unusable")
MULTICLASS_LABELS = (NORMAL_LABEL, AF_LABEL, PVC_LABEL, OTHER_LABEL)

LABEL_TO_ID = {label: idx for idx, label in enumerate(MULTICLASS_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


PVC_SYMBOLS = {"V", "r", "E"}
AF_AUX_PATTERNS = (r"^\(AFIB", r"^\(AFL")

FEATURE_CLASS_MAP = {
    "N": NORMAL_LABEL,
    "VEB": PVC_LABEL,
    "SVEB": OTHER_LABEL,
    "F": OTHER_LABEL,
    "Q": OTHER_LABEL,
}

MITBIH_NUMERIC_LABEL_MAP = {
    0: NORMAL_LABEL,
    1: OTHER_LABEL,
    2: PVC_LABEL,
    3: OTHER_LABEL,
    4: OTHER_LABEL,
}

PTBDB_NUMERIC_LABEL_MAP = {
    0: NORMAL_LABEL,
    1: OTHER_LABEL,
}


def clean_aux_note(aux_note: str | None) -> str:
    if not aux_note:
        return ""
    note = aux_note.strip()
    note = re.sub(r"\d+$", "", note)
    return note


def parse_scp_codes(raw: str | dict[str, float] | None) -> dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
    except (ValueError, SyntaxError):
        pass
    return {}


def map_aux_note_to_label(aux_note: str | None) -> str | None:
    note = clean_aux_note(aux_note)
    if not note:
        return None
    for pattern in AF_AUX_PATTERNS:
        if re.match(pattern, note):
            return AF_LABEL
    return None


def map_symbol_to_label(symbol: str, aux_note: str | None = None) -> str:
    aux_label = map_aux_note_to_label(aux_note)
    if aux_label is not None:
        return aux_label

    sym = symbol.strip()
    if sym == "N":
        return NORMAL_LABEL
    if sym in PVC_SYMBOLS:
        return PVC_LABEL
    return OTHER_LABEL


def map_feature_type_to_label(feature_type: str) -> str:
    return FEATURE_CLASS_MAP.get(feature_type.strip(), OTHER_LABEL)


def map_mitbih_numeric_to_label(label_value: int | float) -> str:
    return MITBIH_NUMERIC_LABEL_MAP.get(int(label_value), OTHER_LABEL)


def map_ptbdb_numeric_to_label(label_value: int | float) -> str:
    return PTBDB_NUMERIC_LABEL_MAP.get(int(label_value), OTHER_LABEL)


def map_ptbxl_codes_to_label(
    codes: dict[str, float],
    arrhythmia_code_set: set[str] | None = None,
) -> str:
    keys = set(codes.keys())
    if {"AFIB", "AFLT"} & keys:
        return AF_LABEL
    if "PVC" in keys:
        return PVC_LABEL

    if arrhythmia_code_set is None:
        arrhythmia_code_set = set()
    present_arrhythmia = (keys & arrhythmia_code_set) - {"NORM", "SR"}
    if present_arrhythmia:
        return OTHER_LABEL

    if "NORM" in keys and present_arrhythmia == set():
        return NORMAL_LABEL
    return OTHER_LABEL


def label_to_id(label: str) -> int:
    return LABEL_TO_ID[label]


def id_to_label(label_id: int) -> str:
    return ID_TO_LABEL[label_id]


def binary_from_multiclass(label_id: int) -> int:
    return 0 if label_id == LABEL_TO_ID[NORMAL_LABEL] else 1


def quality_label_from_id(quality_id: int) -> str:
    if quality_id < 0 or quality_id >= len(QUALITY_LABELS):
        return QUALITY_LABELS[-1]
    return QUALITY_LABELS[quality_id]


def quality_id_from_label(quality_label: str) -> int:
    label = quality_label.strip().capitalize()
    try:
        return QUALITY_LABELS.index(label)
    except ValueError:
        return QUALITY_LABELS.index("Unusable")


def normalize_label_distribution(labels: Iterable[str]) -> dict[str, int]:
    counts = {label: 0 for label in MULTICLASS_LABELS}
    for label in labels:
        if label in counts:
            counts[label] += 1
    return counts

