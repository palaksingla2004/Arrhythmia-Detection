# Full-Stack ECG Arrhythmia Detection System

This project provides an end-to-end arrhythmia detection stack with:

- Unified ingestion across multiple ECG datasets in `Datasets/`
- Research-style ECG preprocessing (0.5-40 Hz filtering, baseline denoise, z-score, Pan-Tompkins R-peaks, SQI)
- Deep ensemble (Inception/ResNet 1D, CNN+LSTM, CNN+Transformer) with multi-head outputs
- Traditional ML suite (SVM, RF, GB, KNN, Logistic, Bagging, Boosting, Stacking, optional XGBoost/LightGBM)
- Calibrated inference (temperature scaling), Monte Carlo dropout uncertainty, and 1D Grad-CAM
- FastAPI backend (`/predict`, `/upload`, `/report`)
- React medical dashboard

## Project Structure

```text
backend/
frontend/
models/
data/
notebooks/
Datasets/
```

## Backend

```bash
cd backend
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Train models

```bash
cd backend
python -m app.training.train_all --rebuild-dataset --epochs 30 --batch-size 256
```

Training outputs are saved under `models/artifacts/`.

Quick smoke training (reduced subset):

```bash
python -m app.training.train_all --rebuild-dataset --quick-mode --epochs 3 --batch-size 128
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Optional API URL override:

```bash
set VITE_API_BASE_URL=http://localhost:8000
```

## One-Command Launcher (Windows)

Use the root PowerShell launcher:

```powershell
.\startup.ps1
```

Direct modes:

```powershell
.\startup.ps1 -Mode setup
.\startup.ps1 -Mode start
.\startup.ps1 -Mode train-lite
.\startup.ps1 -Mode train-balanced
.\startup.ps1 -Mode train-full
```

`train-lite` is optimized for laptops (tiny dataset, low thread count, below-normal process priority).

## Notes

- Dataset unification includes:
  - MIT-BIH raw CSV+annotations
  - MIT-BIH heartbeat categorization (`mitbih_train/test.csv`)
  - PTBDB (`ptbdb_normal/abnormal.csv`)
  - PTB-XL WFDB records
  - Combined engineered feature datasets (`MIT-BIH Arrhythmia Database.csv`, `INCART`, `Sudden Cardiac Death`, etc.)
- Mandatory dataset-name mapping used in this repo:
  - `ECG Heartbeat Categorization Dataset` -> `mitbih_train.csv` + `mitbih_test.csv`
  - `ECG Arrhythmia Classification Dataset (combined datasets)` -> engineered feature CSVs (`MIT-BIH Arrhythmia Database.csv`, `MIT-BIH Supraventricular Arrhythmia Database.csv`, `INCART 2-lead Arrhythmia Database.csv`, `Sudden Cardiac Death Holter Database.csv`)
- Clinical deployment requires external validation, compliance review, and human oversight.
