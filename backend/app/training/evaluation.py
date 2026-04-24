from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from app.data.unified_builder import UnifiedSignalDataset


class _EvalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def evaluate_model_on_indices(
    model: torch.nn.Module,
    dataset: UnifiedSignalDataset,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    ds = _EvalDataset(dataset.X[indices], dataset.y_multi[indices])
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()

    probs = []
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = torch.softmax(out["multiclass_logits"], dim=-1).cpu().numpy()
            probs.append(p)
            preds.append(np.argmax(p, axis=1))
            targets.append(y.numpy())

    probs_np = np.concatenate(probs, axis=0)
    preds_np = np.concatenate(preds, axis=0)
    targets_np = np.concatenate(targets, axis=0)

    precision, recall, _, _ = precision_recall_fscore_support(
        targets_np, preds_np, average="weighted", zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(targets_np, preds_np)),
        "f1_weighted": float(f1_score(targets_np, preds_np, average="weighted")),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "confusion_matrix": confusion_matrix(targets_np, preds_np).tolist(),
    }
    try:
        result["auroc_ovr"] = float(roc_auc_score(targets_np, probs_np, multi_class="ovr"))
    except ValueError:
        result["auroc_ovr"] = 0.0
    return result


def external_source_evaluation(
    model: torch.nn.Module,
    dataset: UnifiedSignalDataset,
    device: torch.device,
    output_json: Path | None = None,
) -> dict[str, Any]:
    sources = sorted(set(dataset.source.tolist()))
    report: dict[str, Any] = {}
    for source in sources:
        idx = np.where(dataset.source == source)[0]
        if len(idx) < 50:
            continue
        report[source] = evaluate_model_on_indices(model, dataset, idx, device=device)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

