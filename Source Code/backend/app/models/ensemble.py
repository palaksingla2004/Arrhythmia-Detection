from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from app.data.labels import id_to_label, quality_label_from_id


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def weighted_average_probs(probs: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if len(probs) != len(weights):
        raise ValueError("probs and weights must have the same length")
    w = np.asarray(weights, dtype=np.float32)
    w = w / np.sum(w)
    stacked = np.stack(probs, axis=0)
    return np.tensordot(w, stacked, axes=(0, 0))


def aggregate_model_outputs(
    model_outputs: list[dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> dict[str, np.ndarray]:
    binary_probs: list[np.ndarray] = []
    multi_probs: list[np.ndarray] = []
    quality_probs: list[np.ndarray] = []
    for out in model_outputs:
        binary_probs.append(torch.softmax(out["binary_logits"], dim=-1).detach().cpu().numpy())
        multi_probs.append(torch.softmax(out["multiclass_logits"], dim=-1).detach().cpu().numpy())
        quality_probs.append(torch.softmax(out["quality_logits"], dim=-1).detach().cpu().numpy())

    return {
        "binary_probs": weighted_average_probs(binary_probs, weights),
        "multiclass_probs": weighted_average_probs(multi_probs, weights),
        "quality_probs": weighted_average_probs(quality_probs, weights),
    }


def top_k_multiclass(prob_vec: np.ndarray, k: int = 3) -> list[dict[str, float]]:
    idx = np.argsort(prob_vec)[::-1][:k]
    return [{"label": id_to_label(int(i)), "probability": float(prob_vec[i])} for i in idx]


def build_prediction_summary(aggregated: dict[str, np.ndarray]) -> dict[str, object]:
    binary = aggregated["binary_probs"][0]
    multi = aggregated["multiclass_probs"][0]
    quality = aggregated["quality_probs"][0]

    binary_class = int(np.argmax(binary))
    multi_class = int(np.argmax(multi))
    quality_class = int(np.argmax(quality))

    return {
        "arrhythmia": bool(binary_class == 1),
        "binary_probability": float(binary[1]),
        "risk_score": float(1.0 - multi[0]),
        "confidence": float(np.max(multi)),
        "predicted_label": id_to_label(multi_class),
        "top_classes": top_k_multiclass(multi, k=4),
        "signal_quality": quality_label_from_id(quality_class),
    }

