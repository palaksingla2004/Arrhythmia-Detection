from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_MULTI_CLASSES = 4
NUM_BINARY_CLASSES = 2
NUM_QUALITY_CLASSES = 3


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 7)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out = F.silu(out + x, inplace=True)
        return out


class InceptionModule1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        branch = out_ch // 4
        self.b1 = ConvBNAct(in_ch, branch, 1)
        self.b2 = nn.Sequential(ConvBNAct(in_ch, branch, 1), ConvBNAct(branch, branch, 3))
        self.b3 = nn.Sequential(ConvBNAct(in_ch, branch, 1), ConvBNAct(branch, branch, 5))
        self.b4 = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1, padding=1), ConvBNAct(in_ch, branch, 1))
        self.mix = ConvBNAct(branch * 4, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.mix(x)


class MultiHeadClassifier(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(256, NUM_BINARY_CLASSES)
        self.multiclass_head = nn.Linear(256, NUM_MULTI_CLASSES)
        self.quality_head = nn.Linear(256, NUM_QUALITY_CLASSES)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.backbone(features)
        return {
            "binary_logits": self.binary_head(z),
            "multiclass_logits": self.multiclass_head(z),
            "quality_logits": self.quality_head(z),
            "embedding": z,
        }


class InceptionResNet1D(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 64, 7, stride=2),
            ConvBNAct(64, 128, 5, stride=2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.block1 = InceptionModule1D(128, 128)
        self.block2 = ResidualBlock1D(128, dropout=dropout)
        self.block3 = InceptionModule1D(128, 256)
        self.block4 = ResidualBlock1D(256, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = MultiHeadClassifier(256, dropout=dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x).squeeze(-1)
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        features = self.forward_features(x)
        outputs = self.head(features)
        outputs["feature_map"] = x
        return outputs


class CNNBiLSTM(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBNAct(in_channels, 64, 7, stride=2),
            ConvBNAct(64, 128, 5, stride=2),
            ConvBNAct(128, 128, 3, stride=1),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.head = MultiHeadClassifier(256, dropout=dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)  # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        features = self.forward_features(x)
        outputs = self.head(features)
        outputs["feature_map"] = x
        return outputs


class CNNTransformer(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.3, n_heads: int = 8) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBNAct(in_channels, 64, 7, stride=2),
            ConvBNAct(64, 128, 5, stride=2),
            ConvBNAct(128, 192, 3, stride=2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=n_heads,
            dim_feedforward=384,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.norm = nn.LayerNorm(192)
        self.dropout = nn.Dropout(dropout)
        self.head = MultiHeadClassifier(192, dropout=dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)  # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        features = self.forward_features(x)
        outputs = self.head(features)
        outputs["feature_map"] = x
        return outputs


@dataclass
class ModelBundle:
    model_name: str
    model: nn.Module


def build_deep_model(model_name: str, in_channels: int = 1, dropout: float = 0.3) -> nn.Module:
    name = model_name.lower()
    if name in {"inception", "resnet", "inception_resnet", "model1"}:
        return InceptionResNet1D(in_channels=in_channels, dropout=dropout)
    if name in {"cnn_lstm", "lstm", "model2"}:
        return CNNBiLSTM(in_channels=in_channels, dropout=dropout)
    if name in {"cnn_transformer", "transformer", "model3"}:
        return CNNTransformer(in_channels=in_channels, dropout=dropout)
    raise ValueError(f"Unknown model_name '{model_name}'")


def model_registry() -> list[str]:
    return ["model1_inception_resnet", "model2_cnn_lstm", "model3_cnn_transformer"]

