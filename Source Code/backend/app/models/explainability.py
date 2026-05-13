from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GradCAMResult:
    heatmap: np.ndarray
    target_class: int
    score: float


class GradCAM1D:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module, _input, output):
            self.activations = output

        def backward_hook(_module, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    @torch.no_grad()
    def _normalize(self, cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.min(dim=-1, keepdim=True)[0]
        denom = cam.max(dim=-1, keepdim=True)[0] + 1e-8
        return cam / denom

    def generate(
        self,
        x: torch.Tensor,
        target_class: int,
        head_name: str = "multiclass_logits",
    ) -> GradCAMResult:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(x)
        logits = outputs[head_name]
        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            seq_len = x.shape[-1]
            return GradCAMResult(heatmap=np.zeros(seq_len, dtype=np.float32), target_class=target_class, score=0.0)

        gradients = self.gradients  # [B, C, T]
        activations = self.activations  # [B, C, T]
        weights = gradients.mean(dim=-1, keepdim=True)  # [B, C, 1]
        cam = torch.sum(weights * activations, dim=1)  # [B, T]
        cam = F.relu(cam)
        cam = cam.unsqueeze(1)
        cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False).squeeze(1)
        cam = self._normalize(cam)

        return GradCAMResult(
            heatmap=cam[0].detach().cpu().numpy().astype(np.float32),
            target_class=target_class,
            score=float(score.detach().cpu()),
        )


def find_last_conv1d_layer(model: nn.Module) -> nn.Module:
    candidate = None
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            candidate = module
    if candidate is None:
        raise ValueError("No Conv1d layer found in model for Grad-CAM.")
    return candidate

