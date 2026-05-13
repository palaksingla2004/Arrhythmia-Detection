from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from app.core.config import settings
from app.data.labels import MULTICLASS_LABELS, id_to_label
from app.data.preprocessing import PreprocessingResult
from app.models.calibration import CalibrationBundle
from app.models.classical import classical_ensemble_predict_proba, load_classical_suite
from app.models.deep_models import build_deep_model
from app.models.ensemble import build_prediction_summary, top_k_multiclass
from app.models.explainability import GradCAM1D, find_last_conv1d_layer


@dataclass
class LoadedDeepModel:
    name: str
    model: torch.nn.Module
    calibration: CalibrationBundle
    weight: float


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


class DeepEnsembleInference:
    def __init__(self, model_dir: Path | None = None, device: str | None = None) -> None:
        self.model_dir = model_dir or (settings.models_dir / "deep")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models: list[LoadedDeepModel] = self._load_models()
        self.classical_models = self._load_classical_models()
        self.weights = np.array([m.weight for m in self.models], dtype=np.float32)
        if self.weights.size > 0:
            self.weights = self.weights / self.weights.sum()

    def _load_models(self) -> list[LoadedDeepModel]:
        loaded: list[LoadedDeepModel] = []
        names = ["model1", "model2", "model3"]
        default_weights = settings.ensemble_weights

        for i, name in enumerate(names):
            ckpt = self.model_dir / f"{name}.pt"
            calib = self.model_dir / f"{name}_calibration.json"
            if not ckpt.exists():
                continue
            payload = torch.load(ckpt, map_location="cpu")
            model = build_deep_model(name).to(self.device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            loaded.append(
                LoadedDeepModel(
                    name=name,
                    model=model,
                    calibration=CalibrationBundle.load(calib),
                    weight=default_weights[i] if i < len(default_weights) else 1.0,
                )
            )
        return loaded

    def _load_classical_models(self):
        classical_dir = settings.models_dir / "classical"
        if not classical_dir.exists():
            return []
        try:
            return load_classical_suite(classical_dir)
        except Exception:
            return []

    def available(self) -> bool:
        return len(self.models) > 0

    def _segment_for_inference(self, prep: PreprocessingResult) -> torch.Tensor:
        if prep.fixed_segments.shape[0] > 0:
            seg = prep.fixed_segments[0]
        else:
            target_len = int(settings.target_sampling_rate * settings.segment_seconds)
            if len(prep.signal) >= target_len:
                seg = prep.signal[:target_len]
            else:
                seg = np.pad(prep.signal, (0, target_len - len(prep.signal)), mode="edge")
        x = torch.tensor(seg, dtype=torch.float32, device=self.device).unsqueeze(0)
        return x

    @staticmethod
    def _enable_dropout_only(model: torch.nn.Module) -> None:
        model.eval()
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()

    def _heuristic_fallback(self, prep: PreprocessingResult) -> dict[str, Any]:
        variability = float(np.std(np.diff(prep.signal))) if len(prep.signal) > 1 else 0.0
        irregularity = min(1.0, variability * 2.0)
        probs = np.array([1.0 - irregularity, 0.15 * irregularity, 0.35 * irregularity, 0.5 * irregularity])
        probs = probs / probs.sum()
        top_classes = [{"label": id_to_label(int(i)), "probability": float(probs[i])} for i in np.argsort(probs)[::-1]]
        return {
            "arrhythmia": bool(np.argmax(probs) != 0),
            "risk_score": float(1.0 - probs[0]),
            "confidence": float(np.max(probs)),
            "top_classes": top_classes,
            "signal_quality": prep.quality_label,
            "uncertainty": float(0.25),
            "binary_probability": float(1.0 - probs[0]),
            "explanation_map": np.zeros_like(prep.signal, dtype=np.float32).tolist(),
            "explanation_text": (
                "Model checkpoints are not available yet. This result is a statistical fallback estimate "
                "based on waveform variability. Train the models for research-grade predictions."
            ),
        }

    def _extract_classical_features(self, signal: np.ndarray, target_dim: int) -> np.ndarray:
        diffs = np.diff(signal) if len(signal) > 1 else np.array([0.0], dtype=np.float32)
        abs_diff = np.abs(diffs)
        base = np.array(
            [
                float(np.mean(signal)),
                float(np.std(signal)),
                float(np.min(signal)),
                float(np.max(signal)),
                float(np.percentile(signal, 10)),
                float(np.percentile(signal, 25)),
                float(np.percentile(signal, 50)),
                float(np.percentile(signal, 75)),
                float(np.percentile(signal, 90)),
                float(np.mean(abs_diff)),
                float(np.std(abs_diff)),
                float(np.max(abs_diff)),
                float(np.mean(signal**2)),
                float(np.mean(np.abs(signal))),
                float(np.sum(signal > 0) / max(len(signal), 1)),
                float(np.sum(signal < 0) / max(len(signal), 1)),
                float(np.mean(np.signbit(signal))),
                float(np.var(diffs)),
                float(np.percentile(abs_diff, 95)),
                float(np.percentile(abs_diff, 99)),
            ],
            dtype=np.float32,
        )
        if target_dim <= len(base):
            return base[:target_dim]
        out = np.zeros((target_dim,), dtype=np.float32)
        out[: len(base)] = base
        # Repeat informative stats to fill expected shape for classical feature models.
        for i in range(len(base), target_dim):
            out[i] = base[i % len(base)]
        return out

    def predict(self, prep: PreprocessingResult) -> dict[str, Any]:
        if not self.available():
            return self._heuristic_fallback(prep)

        x = self._segment_for_inference(prep)

        binary_prob_runs = []
        multi_prob_runs = []
        quality_prob_runs = []
        model_outputs: list[dict[str, torch.Tensor]] = []

        for model_idx, loaded in enumerate(self.models):
            model = loaded.model
            self._enable_dropout_only(model)

            run_binary = []
            run_multi = []
            run_quality = []
            for _ in range(settings.mc_dropout_passes):
                with torch.no_grad():
                    out = model(x)
                    b_logits = out["binary_logits"] / loaded.calibration.binary_temperature
                    m_logits = out["multiclass_logits"] / loaded.calibration.multiclass_temperature
                    q_logits = out["quality_logits"] / loaded.calibration.quality_temperature
                    run_binary.append(_softmax(b_logits).cpu().numpy())
                    run_multi.append(_softmax(m_logits).cpu().numpy())
                    run_quality.append(_softmax(q_logits).cpu().numpy())

            run_binary_np = np.concatenate(run_binary, axis=0)  # [T, 2]
            run_multi_np = np.concatenate(run_multi, axis=0)  # [T, 4]
            run_quality_np = np.concatenate(run_quality, axis=0)  # [T, 3]

            binary_prob_runs.append(run_binary_np.mean(axis=0))
            multi_prob_runs.append(run_multi_np.mean(axis=0))
            quality_prob_runs.append(run_quality_np.mean(axis=0))

            # keep deterministic output for Grad-CAM/inspection
            model.eval()
            with torch.no_grad():
                model_outputs.append(model(x))

        binary_probs = np.average(np.stack(binary_prob_runs, axis=0), axis=0, weights=self.weights)
        multi_probs = np.average(np.stack(multi_prob_runs, axis=0), axis=0, weights=self.weights)
        quality_probs = np.average(np.stack(quality_prob_runs, axis=0), axis=0, weights=self.weights)

        # Optional hybrid blending with traditional models.
        if self.classical_models:
            try:
                first_model = self.classical_models[0][1]
                if hasattr(first_model, "n_features_in_"):
                    feat_dim = int(first_model.n_features_in_)
                elif hasattr(first_model, "steps"):
                    feat_dim = int(first_model.steps[-1][1].n_features_in_)
                else:
                    feat_dim = 34
                feats = self._extract_classical_features(prep.signal, feat_dim).reshape(1, -1)
                classical_probs = classical_ensemble_predict_proba(self.classical_models, feats)[0]
                if classical_probs.shape[0] == multi_probs.shape[0]:
                    multi_probs = 0.85 * multi_probs + 0.15 * classical_probs
                    multi_probs = multi_probs / np.sum(multi_probs)
            except Exception:
                pass

        per_model_stack = np.stack(multi_prob_runs, axis=0)
        model_uncertainty = float(np.mean(np.std(per_model_stack, axis=0)))

        top_classes = top_k_multiclass(multi_probs, k=4)
        quality_idx = int(np.argmax(quality_probs))
        predicted_idx = int(np.argmax(multi_probs))
        arrhythmia_flag = bool(np.argmax(binary_probs) == 1)

        explanation_map = self._generate_gradcam(
            x=x,
            predicted_class=predicted_idx,
            signal_len=len(prep.signal),
        )

        explanation_text = _build_explanation_text(
            label=id_to_label(predicted_idx),
            confidence=float(np.max(multi_probs)),
            quality_label=prep.quality_label,
            uncertainty=model_uncertainty,
        )

        return {
            "arrhythmia": arrhythmia_flag,
            "risk_score": float(1.0 - multi_probs[0]),
            "confidence": float(np.max(multi_probs)),
            "top_classes": top_classes,
            "signal_quality": prep.quality_label if quality_idx <= 1 else "Unusable",
            "uncertainty": model_uncertainty,
            "binary_probability": float(binary_probs[1]),
            "explanation_map": explanation_map.tolist(),
            "explanation_text": explanation_text,
        }

    def _generate_gradcam(self, x: torch.Tensor, predicted_class: int, signal_len: int) -> np.ndarray:
        # Use the first model for Grad-CAM visualization.
        primary = self.models[0]
        primary.model.eval()
        target_layer = find_last_conv1d_layer(primary.model)
        gradcam = GradCAM1D(primary.model, target_layer)
        result = gradcam.generate(x, target_class=predicted_class, head_name="multiclass_logits")
        cam = result.heatmap
        if len(cam) != signal_len:
            cam_t = torch.tensor(cam, dtype=torch.float32).view(1, 1, -1)
            cam_resized = F.interpolate(cam_t, size=signal_len, mode="linear", align_corners=False)
            cam = cam_resized.view(-1).cpu().numpy()
        return cam.astype(np.float32)


def _build_explanation_text(label: str, confidence: float, quality_label: str, uncertainty: float) -> str:
    confidence_pct = confidence * 100.0
    uncertainty_pct = min(100.0, uncertainty * 100.0)
    base = (
        f"Predicted class: {label}. Confidence: {confidence_pct:.1f}%. "
        f"Signal quality: {quality_label}. Estimated uncertainty: {uncertainty_pct:.1f}%."
    )
    if confidence < 0.6 or quality_label != "Good":
        base += " Confidence is limited; follow up with a clinician is recommended."
    return base
