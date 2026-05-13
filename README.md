Palak Singla(2210990994), Rajat Sharma(2210990707), Pareen Khirbat(2210990639)
Project Title:- ECG Arrhythmia Detection System Using CNN
Research Paper Status:- Pending


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

## 📂 Dataset Setup (Required)

This project requires external ECG datasets.  
You must **download them manually** and place them in a `Datasets/` folder.

---

### 🔽 Download Datasets

- ECG Arrhythmia Classification Dataset  
  https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset/data  

- PTB-XL ECG Dataset  
  https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset/suggestions  

- ECG Heartbeat Categorization Dataset (MIT-BIH)  
  https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data?select=mitbih_train.csv  

- MIT-BIH Arrhythmia Database (Modern 2023)  
  https://www.kaggle.com/datasets/protobioengineering/mit-bih-arrhythmia-database-modern-2023  

---

### 📁 Folder Setup

Create this folder:


Datasets/


Place all dataset files inside it.

Example structure:


Datasets/
├── mitbih_train.csv
├── mitbih_test.csv
├── ptbdb_normal.csv
├── ptbdb_abnormal.csv
├── PTB-XL/
├── MIT-BIH Arrhythmia Database.csv
├── INCART 2-lead Arrhythmia Database.csv
├── Sudden Cardiac Death Holter Database.csv


---

### ⚠️ Important Notes

- Do **NOT rename files** unless required
- Extract all `.zip` files
- Keep PTB-XL folder structure intact
- First run may take time (dataset preprocessing)
---

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
