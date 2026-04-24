from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from app.data.unified_builder import UnifiedSignalDataset, cross_validation_splits
from app.models.calibration import CalibrationBundle, TemperatureScaler
from app.models.deep_models import build_deep_model, model_registry
from app.training.evaluation import external_source_evaluation
from app.training.augmentations import ECGAugmenter
from app.training.losses import MultiTaskECGLoss


@dataclass
class DeepModelArtifact:
    model_name: str
    checkpoint_path: Path
    calibration_path: Path
    metrics: dict[str, float]


class ECGTorchDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y_multi: np.ndarray,
        y_binary: np.ndarray,
        y_quality: np.ndarray,
        augmenter: ECGAugmenter | None = None,
    ) -> None:
        self.X = X
        self.y_multi = y_multi
        self.y_binary = y_binary
        self.y_quality = y_quality
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        signal = self.X[idx]
        if self.augmenter is not None:
            signal = self.augmenter(signal)
        return {
            "x": torch.tensor(signal, dtype=torch.float32),
            "y_multi": torch.tensor(self.y_multi[idx], dtype=torch.long),
            "y_binary": torch.tensor(self.y_binary[idx], dtype=torch.long),
            "y_quality": torch.tensor(self.y_quality[idx], dtype=torch.long),
        }


def _build_sampler(y_multi: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(y_multi)
    weights = np.array([1.0 / max(counts[c], 1) for c in y_multi], dtype=np.float32)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model.eval()
    logits_binary = []
    logits_multi = []
    logits_quality = []
    y_binary = []
    y_multi = []
    y_quality = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            outputs = model(x)
            logits_binary.append(outputs["binary_logits"].cpu())
            logits_multi.append(outputs["multiclass_logits"].cpu())
            logits_quality.append(outputs["quality_logits"].cpu())
            y_binary.append(batch["y_binary"])
            y_multi.append(batch["y_multi"])
            y_quality.append(batch["y_quality"])
    logits = {
        "binary": torch.cat(logits_binary, dim=0),
        "multi": torch.cat(logits_multi, dim=0),
        "quality": torch.cat(logits_quality, dim=0),
    }
    labels = {
        "binary": torch.cat(y_binary, dim=0),
        "multi": torch.cat(y_multi, dim=0),
        "quality": torch.cat(y_quality, dim=0),
    }
    return logits, labels


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    calibration: CalibrationBundle | None = None,
) -> dict[str, float]:
    model.eval()
    probs_multi = []
    pred_multi = []
    gt_multi = []
    pred_binary = []
    gt_binary = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = model(x)
            multi_logits = out["multiclass_logits"]
            binary_logits = out["binary_logits"]
            if calibration is not None:
                multi_logits = multi_logits / calibration.multiclass_temperature
                binary_logits = binary_logits / calibration.binary_temperature
            p_multi = torch.softmax(multi_logits, dim=-1).cpu().numpy()
            p_binary = torch.softmax(binary_logits, dim=-1).cpu().numpy()

            probs_multi.append(p_multi)
            pred_multi.append(np.argmax(p_multi, axis=1))
            gt_multi.append(batch["y_multi"].numpy())
            pred_binary.append(np.argmax(p_binary, axis=1))
            gt_binary.append(batch["y_binary"].numpy())

    probs_multi_np = np.concatenate(probs_multi, axis=0)
    pred_multi_np = np.concatenate(pred_multi, axis=0)
    gt_multi_np = np.concatenate(gt_multi, axis=0)
    pred_binary_np = np.concatenate(pred_binary, axis=0)
    gt_binary_np = np.concatenate(gt_binary, axis=0)

    metrics: dict[str, float | list[list[int]]] = {}
    metrics["accuracy"] = float(accuracy_score(gt_multi_np, pred_multi_np))
    metrics["f1_weighted"] = float(f1_score(gt_multi_np, pred_multi_np, average="weighted"))
    precision, recall, _, _ = precision_recall_fscore_support(
        gt_multi_np, pred_multi_np, average="weighted", zero_division=0
    )
    metrics["precision_weighted"] = float(precision)
    metrics["recall_weighted"] = float(recall)

    try:
        metrics["auroc_ovr"] = float(roc_auc_score(gt_multi_np, probs_multi_np, multi_class="ovr"))
    except ValueError:
        metrics["auroc_ovr"] = 0.0

    metrics["binary_accuracy"] = float(accuracy_score(gt_binary_np, pred_binary_np))
    metrics["confusion_matrix"] = confusion_matrix(gt_multi_np, pred_multi_np).tolist()
    return metrics


def _fit_temperature_scalers(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
) -> CalibrationBundle:
    binary_scaler = TemperatureScaler()
    multi_scaler = TemperatureScaler()
    quality_scaler = TemperatureScaler()
    binary_scaler.fit(logits["binary"], labels["binary"])
    multi_scaler.fit(logits["multi"], labels["multi"])
    quality_scaler.fit(logits["quality"], labels["quality"])
    return CalibrationBundle(
        binary_temperature=float(binary_scaler.temperature.item()),
        multiclass_temperature=float(multi_scaler.temperature.item()),
        quality_temperature=float(quality_scaler.temperature.item()),
    )


def train_one_model(
    model_name: str,
    dataset: UnifiedSignalDataset,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    output_dir: Path,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 6,
) -> DeepModelArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = dataset.X[train_idx]
    y_train_multi = dataset.y_multi[train_idx]
    y_train_binary = dataset.y_binary[train_idx]
    y_train_quality = dataset.y_quality[train_idx]

    X_valid = dataset.X[valid_idx]
    y_valid_multi = dataset.y_multi[valid_idx]
    y_valid_binary = dataset.y_binary[valid_idx]
    y_valid_quality = dataset.y_quality[valid_idx]

    train_ds = ECGTorchDataset(
        X_train,
        y_train_multi,
        y_train_binary,
        y_train_quality,
        augmenter=ECGAugmenter(),
    )
    valid_ds = ECGTorchDataset(
        X_valid,
        y_valid_multi,
        y_valid_binary,
        y_valid_quality,
        augmenter=None,
    )

    sampler = _build_sampler(y_train_multi)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_deep_model(model_name).to(device)

    class_counts = np.bincount(y_train_multi, minlength=4)
    multi_weights = class_counts.sum() / np.maximum(class_counts, 1)
    multi_weights = multi_weights / multi_weights.sum() * len(multi_weights)
    multi_weights_t = torch.tensor(multi_weights, dtype=torch.float32, device=device)

    quality_counts = np.bincount(y_train_quality, minlength=3)
    quality_weights = quality_counts.sum() / np.maximum(quality_counts, 1)
    quality_weights = quality_weights / quality_weights.sum() * len(quality_weights)
    quality_weights_t = torch.tensor(quality_weights, dtype=torch.float32, device=device)

    criterion = MultiTaskECGLoss(
        multi_class_weights=multi_weights_t,
        quality_class_weights=quality_weights_t,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    best_f1 = -1.0
    best_state = None
    patience = 0

    for _epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y_multi = batch["y_multi"].to(device)
            y_binary = batch["y_binary"].to(device)
            y_quality = batch["y_quality"].to(device)

            out = model(x)
            loss, _ = criterion(out, y_binary=y_binary, y_multi=y_multi, y_quality=y_quality)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        metrics = _evaluate(model, valid_loader, device)
        val_f1 = metrics["f1_weighted"]
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    logits, labels = _collect_logits_and_labels(model, valid_loader, device)
    calibration = _fit_temperature_scalers(logits, labels)
    final_metrics = _evaluate(model, valid_loader, device, calibration=calibration)

    checkpoint_path = output_dir / f"{model_name}.pt"
    calibration_path = output_dir / f"{model_name}_calibration.json"
    torch.save({"model_name": model_name, "state_dict": model.state_dict()}, checkpoint_path)
    calibration.save(calibration_path)

    metrics_path = output_dir / f"{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    return DeepModelArtifact(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        calibration_path=calibration_path,
        metrics=final_metrics,
    )


def train_deep_ensemble(
    dataset: UnifiedSignalDataset,
    output_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> list[DeepModelArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = cross_validation_splits(dataset, n_splits=n_splits, random_state=random_state)
    train_idx, valid_idx = splits[0]

    artifacts: list[DeepModelArtifact] = []
    summary: dict[str, Any] = {}
    for model_name in ["model1", "model2", "model3"]:
        artifact = train_one_model(
            model_name=model_name,
            dataset=dataset,
            train_idx=train_idx,
            valid_idx=valid_idx,
            output_dir=output_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        artifacts.append(artifact)
        summary[model_name] = artifact.metrics

    # External dataset testing across source domains using the final transformer model.
    try:
        final_model_name = artifacts[-1].model_name
        final_ckpt = torch.load(artifacts[-1].checkpoint_path, map_location="cpu")
        eval_model = build_deep_model(final_model_name)
        eval_model.load_state_dict(final_ckpt["state_dict"])
        eval_model = eval_model.to(device)
        external_report = external_source_evaluation(
            model=eval_model,
            dataset=dataset,
            device=device,
            output_json=output_dir / "external_source_metrics.json",
        )
        summary["external_source_evaluation"] = external_report
    except Exception:
        summary["external_source_evaluation"] = {}

    (output_dir / "deep_metrics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return artifacts
