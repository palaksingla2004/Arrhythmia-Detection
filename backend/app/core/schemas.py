from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    signal: list[float] | None = Field(
        default=None,
        description="Single-lead ECG samples as a 1D float array.",
    )
    signal_2d: list[list[float]] | None = Field(
        default=None,
        description="Optional multi-lead ECG with shape [lead, time].",
    )
    sampling_rate: int = Field(default=250, ge=50, le=1000)
    metadata: dict[str, Any] | None = None

    @field_validator("signal")
    @classmethod
    def validate_signal(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) < 100:
            raise ValueError("signal must contain at least 100 samples")
        return value

    @field_validator("signal_2d")
    @classmethod
    def validate_signal_2d(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        if not value or not value[0]:
            raise ValueError("signal_2d cannot be empty")
        base_len = len(value[0])
        if any(len(lead) != base_len for lead in value):
            raise ValueError("all leads in signal_2d must have the same length")
        return value


class TopClass(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    arrhythmia: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    top_classes: list[TopClass]
    signal_quality: str
    uncertainty: float = Field(ge=0.0)
    explanation_map: list[float]
    binary_probability: float = Field(ge=0.0, le=1.0)
    timestamp_utc: datetime
    explanation_text: str


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    content_type: str
    bytes_received: int
    parsed_points: int


class ReportRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=128)
    prediction: PredictionResponse
    notes: str | None = None


class ReportResponse(BaseModel):
    report_id: str
    report_path: str
    created_at_utc: datetime

