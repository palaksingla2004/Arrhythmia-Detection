from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.schemas import (
    PredictRequest,
    PredictionResponse,
    ReportRequest,
    ReportResponse,
    TopClass,
    UploadResponse,
)
from app.data.preprocessing import preprocess_signal
from app.models.inference import DeepEnsembleInference
from app.utils.reporting import generate_prediction_report_pdf

router = APIRouter()

_inference_engine = DeepEnsembleInference()


def _parse_csv_signal(content: str) -> np.ndarray:
    reader = csv.reader(io.StringIO(content))
    values: list[float] = []
    for row in reader:
        if not row:
            continue
        for cell in row:
            cell = cell.strip()
            if not cell:
                continue
            try:
                values.append(float(cell))
            except ValueError:
                # ignore headers/non-numeric tokens
                continue
    if not values:
        raise ValueError("no numeric ECG values found in CSV")
    return np.asarray(values, dtype=np.float32)


def _parse_json_signal(content: str) -> tuple[np.ndarray | list[list[float]], int]:
    payload = json.loads(content)
    if isinstance(payload, list):
        return np.asarray(payload, dtype=np.float32), settings.target_sampling_rate
    if not isinstance(payload, dict):
        raise ValueError("JSON must be an object or numeric array")
    sampling_rate = int(payload.get("sampling_rate", settings.target_sampling_rate))
    if "signal_2d" in payload and payload["signal_2d"] is not None:
        return payload["signal_2d"], sampling_rate
    if "signal" in payload and payload["signal"] is not None:
        return np.asarray(payload["signal"], dtype=np.float32), sampling_rate
    raise ValueError("JSON payload must include 'signal' or 'signal_2d'")


def _build_prediction_response(raw_result: dict) -> PredictionResponse:
    top_classes = [TopClass(label=c["label"], probability=float(c["probability"])) for c in raw_result["top_classes"]]
    return PredictionResponse(
        arrhythmia=bool(raw_result["arrhythmia"]),
        risk_score=float(raw_result["risk_score"]),
        confidence=float(raw_result["confidence"]),
        top_classes=top_classes,
        signal_quality=str(raw_result["signal_quality"]),
        uncertainty=float(raw_result["uncertainty"]),
        explanation_map=[float(v) for v in raw_result["explanation_map"]],
        binary_probability=float(raw_result["binary_probability"]),
        timestamp_utc=datetime.now(timezone.utc),
        explanation_text=str(raw_result["explanation_text"]),
    )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest) -> PredictionResponse:
    if payload.signal is None and payload.signal_2d is None:
        raise HTTPException(status_code=400, detail="Either 'signal' or 'signal_2d' is required.")

    signal = payload.signal_2d if payload.signal_2d is not None else payload.signal
    try:
        prep = preprocess_signal(signal=signal, sampling_rate=payload.sampling_rate)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {exc}") from exc

    result = _inference_engine.predict(prep)
    return _build_prediction_response(result)


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    content = await file.read()
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")

    upload_id = f"upload-{uuid4().hex[:12]}"
    suffix = Path(file.filename or "ecg_signal").suffix or ".dat"
    save_path = settings.uploads_dir / f"{upload_id}{suffix}"
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)

    decoded = content.decode("utf-8", errors="ignore")
    parsed_points = 0
    try:
        if file.content_type and "json" in file.content_type:
            signal, _ = _parse_json_signal(decoded)
            parsed_points = int(np.asarray(signal).size)
        else:
            signal = _parse_csv_signal(decoded)
            parsed_points = int(signal.size)
    except Exception:
        parsed_points = 0

    return UploadResponse(
        upload_id=upload_id,
        filename=file.filename or save_path.name,
        content_type=file.content_type or "application/octet-stream",
        bytes_received=len(content),
        parsed_points=parsed_points,
    )


@router.post("/report", response_model=ReportResponse)
def report(payload: ReportRequest) -> ReportResponse:
    report_id, report_path = generate_prediction_report_pdf(
        patient_id=payload.patient_id,
        prediction=payload.prediction,
        notes=payload.notes,
    )
    return ReportResponse(
        report_id=report_id,
        report_path=str(report_path),
        created_at_utc=datetime.now(timezone.utc),
    )


@router.get("/report/{report_id}")
def download_report(report_id: str) -> FileResponse:
    report_path = settings.reports_dir / f"{report_id}.pdf"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(path=report_path, filename=report_path.name, media_type="application/pdf")

