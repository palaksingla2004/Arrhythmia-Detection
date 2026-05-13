from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[3]
    datasets_dir: Path = Path(__file__).resolve().parents[3] / "Datasets"
    data_dir: Path = Path(__file__).resolve().parents[3] / "data"
    processed_dir: Path = Path(__file__).resolve().parents[3] / "data" / "processed"
    cache_dir: Path = Path(__file__).resolve().parents[3] / "data" / "cache"
    models_dir: Path = Path(__file__).resolve().parents[3] / "models" / "artifacts"
    reports_dir: Path = Path(__file__).resolve().parents[3] / "backend" / "reports"
    uploads_dir: Path = Path(__file__).resolve().parents[3] / "backend" / "uploads"

    target_sampling_rate: int = 250
    segment_seconds: float = 2.5
    beat_pre_seconds: float = 0.2
    beat_post_seconds: float = 0.4

    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    bandpass_order: int = 4

    quality_good_snr_db: float = 10.0
    quality_noisy_snr_db: float = 3.0
    quality_unusable_snr_db: float = 0.0

    ensemble_weights: tuple[float, float, float] = (0.34, 0.33, 0.33)
    mc_dropout_passes: int = 20
    max_upload_size_bytes: int = 50 * 1024 * 1024


settings = Settings()


def ensure_runtime_dirs() -> None:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)

