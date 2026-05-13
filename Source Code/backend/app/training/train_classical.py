from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.data.loaders import load_engineered_feature_batch
from app.models.classical import ClassicalModelArtifact, train_classical_suite


@dataclass
class ClassicalTrainingSummary:
    num_samples: int
    num_features: int
    model_rankings: list[dict[str, Any]]


def train_classical_models(
    output_dir: Path,
    sample_per_dataset: int | None = 120_000,
) -> ClassicalTrainingSummary:
    feature_batch = load_engineered_feature_batch(sample_per_dataset=sample_per_dataset)
    X = feature_batch.X
    y = feature_batch.y
    if X.size == 0:
        raise RuntimeError("Engineered feature datasets are empty or missing.")

    artifacts = train_classical_suite(X=X, y=y, output_dir=output_dir)
    rankings = [
        {
            "name": art.name,
            "path": str(art.path),
            "validation_f1": art.validation_f1,
        }
        for art in artifacts
    ]
    summary = ClassicalTrainingSummary(
        num_samples=int(X.shape[0]),
        num_features=int(X.shape[1]),
        model_rankings=rankings,
    )
    (output_dir / "classical_summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )
    return summary

