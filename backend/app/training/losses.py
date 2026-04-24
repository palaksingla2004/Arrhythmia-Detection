from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskECGLoss(nn.Module):
    def __init__(
        self,
        multi_class_weights: torch.Tensor | None = None,
        quality_class_weights: torch.Tensor | None = None,
        alpha_binary: float = 1.0,
        alpha_multi: float = 1.5,
        alpha_quality: float = 0.5,
    ) -> None:
        super().__init__()
        self.binary_loss = nn.CrossEntropyLoss()
        self.multi_loss = nn.CrossEntropyLoss(weight=multi_class_weights)
        self.quality_loss = nn.CrossEntropyLoss(weight=quality_class_weights)
        self.alpha_binary = alpha_binary
        self.alpha_multi = alpha_multi
        self.alpha_quality = alpha_quality

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        y_binary: torch.Tensor,
        y_multi: torch.Tensor,
        y_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        binary_ce = self.binary_loss(outputs["binary_logits"], y_binary)
        multi_ce = self.multi_loss(outputs["multiclass_logits"], y_multi)
        quality_ce = self.quality_loss(outputs["quality_logits"], y_quality)
        total = (
            self.alpha_binary * binary_ce
            + self.alpha_multi * multi_ce
            + self.alpha_quality * quality_ce
        )
        metrics = {
            "binary_ce": float(binary_ce.detach().cpu()),
            "multi_ce": float(multi_ce.detach().cpu()),
            "quality_ce": float(quality_ce.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }
        return total, metrics

