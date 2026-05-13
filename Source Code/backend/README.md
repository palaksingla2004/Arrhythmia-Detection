# Backend API

## Start server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Main endpoints

- `POST /predict`: ECG signal JSON input -> arrhythmia/risk/confidence/top classes/quality/explanation map
- `POST /upload`: Upload ECG file (CSV/JSON)
- `POST /report`: Generate PDF report from prediction payload
- `GET /report/{report_id}`: Download generated PDF
- `GET /health`: Health check

## Train models

```bash
python -m app.training.train_all --rebuild-dataset --epochs 30 --batch-size 256
```

