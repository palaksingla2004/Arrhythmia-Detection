from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import ensure_runtime_dirs


def create_app() -> FastAPI:
    ensure_runtime_dirs()
    app = FastAPI(
        title="ECG Arrhythmia Detection API",
        version="1.0.0",
        description=(
            "Research-grade ECG arrhythmia detection backend with multi-head deep ensemble, "
            "uncertainty estimation, calibration, and explainability."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()

