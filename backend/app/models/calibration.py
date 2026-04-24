from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    def __init__(self, init_temp: float = 1.0) -> None:
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor([init_temp], dtype=torch.float32)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp).clamp(min=1e-3, max=100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.log_temp], lr=0.05, max_iter=max_iter)

        logits = logits.detach()
        labels = labels.detach()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        final_loss = float(criterion(self(logits), labels).detach().cpu())
        return final_loss


@dataclass
class CalibrationBundle:
    binary_temperature: float = 1.0
    multiclass_temperature: float = 1.0
    quality_temperature: float = 1.0

    def apply(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "binary_logits": outputs["binary_logits"] / self.binary_temperature,
            "multiclass_logits": outputs["multiclass_logits"] / self.multiclass_temperature,
            "quality_logits": outputs["quality_logits"] / self.quality_temperature,
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "binary_temperature": self.binary_temperature,
                "multiclass_temperature": self.multiclass_temperature,
                "quality_temperature": self.quality_temperature,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, payload: str) -> "CalibrationBundle":
        data = json.loads(payload)
        return cls(
            binary_temperature=float(data.get("binary_temperature", 1.0)),
            multiclass_temperature=float(data.get("multiclass_temperature", 1.0)),
            quality_temperature=float(data.get("quality_temperature", 1.0)),
        )

    @classmethod
    def load(cls, path: Path) -> "CalibrationBundle":
        if not path.exists():
            return cls()
        return cls.from_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

