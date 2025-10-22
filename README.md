# Credit Card Default Risk Prediction (API + Frontend)

Production-ready credit risk scoring service with:
- FastAPI backend (`algo/`) serving predictions from an ensemble (LightGBM + MLP + optional CNN1D/LSTM)
- Beautiful Vite/React frontend (`card/`) consuming the API
- Leak-free training with CV stacking, holdout threshold selection (Youden's J)
- Offline experiment suite to compare multiple models and update decision threshold

---

## Project Structure

```
AI-Hack1/
├─ algo/                       # Backend (FastAPI) + training code
│  ├─ main.py                  # FastAPI app (POST /predict)
│  ├─ train_ensemble.py        # CV stacking trainer (leakage-free)
│  ├─ dl_models.py             # MLP/CNN1D/LSTM + scaler utils
│  ├─ artifacts.json           # Decision threshold + summary metrics (hot-loaded)
│  ├─ models/                  # Saved models + ensemble config (created after training)
│  ├─ experiments/             # Offline model/threshold comparison
│  │  ├─ stacking_rf.py        # RF meta-learner over XGB/Cat/LGBM (Youden's J)
│  │  ├─ stacking_logistic.py  # Logistic meta-learner over XGB/Cat/LGBM (Youden's J)
│  │  ├─ lightgbm_eval.py      # LGBM only (ADASYN, Youden's J)
│  │  ├─ xgboost_eval.py       # XGBoost only (ADASYN, calibrated, Youden's J)
│  │  ├─ catboost_eval.py      # CatBoost only (ADASYN, calibrated, Youden's J)
│  │  └─ select_threshold.py   # Pick best threshold and update artifacts.json
│  ├─ final_model_ready_data.csv  # Training data (you provide)
│  └─ requirements.txt
├─ card/                       # Frontend (Vite + React)
│  ├─ src/
│  └─ package.json
└─ README.md
```

---

## Quick Start

Prerequisites:
- Python 3.10+
- Node 18+
- PowerShell (Windows)

### 1) Backend setup
```powershell
# From repo root
pip install -r algo/requirements.txt
# Optional (to run all experiments):
pip install xgboost catboost imbalanced-learn
```

### 2) Train the ensemble
Choose one mode:

- FAST (LightGBM + MLP only; quick CPU training)
```powershell
# In AI-Hack1/algo
Remove-Item -Recurse -Force .\models -ErrorAction SilentlyContinue
$env:FAST_ENS="1"
python train_ensemble.py
```

- FULL (LightGBM + MLP + CNN1D + LSTM)
```powershell
# In AI-Hack1/algo
Remove-Item -Recurse -Force .\models -ErrorAction SilentlyContinue
$env:FAST_ENS="0"
python train_ensemble.py
```

Expected output:
- "Ensemble (API schema, CV stacking) ROC-AUC: … | AP: …"
- "Selected decision_threshold (holdout): …"
- Artifacts saved to `algo/models/` and `algo/artifacts.json`

### 3) Start the API
```powershell
# In AI-Hack1/algo
python -m uvicorn main:app --reload
```
You should see:
- "Ensemble artifacts loaded: using models ['lgbm','mlp']" (and ['cnn1d','lstm'] in FULL mode)
- "Optimized decision threshold (in-use): …" (from `artifacts.json`)

### 4) Test the API (two contrasting applicants)
```powershell
# High risk
$body = @{
  gender='Male'; age=23; maritalStatus='Single / not married'; children=1; familyMembers=4; income=20000;
  incomeSource='Student'; education='Lower secondary'; employmentStatus='Unemployed'; yearsEmployed=0;
  occupation='Unknown'; ownCar='No'; ownRealty='No'; housingSituation='Rented apartment'
} | ConvertTo-Json

$res = Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Body $body -ContentType 'application/json'
$res._debug | ConvertTo-Json -Depth 5

# Low risk
$body2 = @{
  gender='Female'; age=44; maritalStatus='Married'; children=0; familyMembers=2; income=180000;
  incomeSource='Working'; education='Higher education'; employmentStatus='Employed'; yearsEmployed=15;
  occupation='Managers'; ownCar='Yes'; ownRealty='Yes'; housingSituation='House / apartment'
} | ConvertTo-Json

$res2 = Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Body $body2 -ContentType 'application/json'
$res2._debug | ConvertTo-Json -Depth 5
```
Check:
- `_debug.threshold` matches `algo/artifacts.json`
- `_debug.ensemble` is `true`
- Probabilities differ; decision flips around the threshold

---

## Offline Experiments (pick best decision threshold)
Run multiple approaches; automatically choose the best threshold and update `artifacts.json`.

```powershell
# In AI-Hack1/algo
python experiments\stacking_rf.py
python experiments\stacking_logistic.py
python experiments\lightgbm_eval.py
python experiments\xgboost_eval.py
python experiments\catboost_eval.py
python experiments\credit_pipeline_eval.py

# Select the best (highest ROC_AUC, tie-break AP) and update artifacts.json
python experiments\select_threshold.py
```
The API hot‑loads `artifacts.json` per request; no server restart required.

---

## Frontend (Vite + React)
```powershell
# In AI-Hack1/card
npm install
npm run dev
```
Open the printed URL (typically http://localhost:5173/) and submit the form. The UI calls the FastAPI backend at http://127.0.0.1:8000/predict.

---

## Important Notes
- **Leakage control**: training excludes post‑outcome credit history fields. The API inference uses only form fields + engineered features.
- **Threshold strategy**: default is **Youden’s J** computed on a holdout set (or the experiment test split). You can re-run experiments to update it or manually edit `algo/artifacts.json`.
- **Fast vs Full mode**: control via env var `FAST_ENS` (1 = skip CNN/LSTM; 0 = include). Artifacts record included models in `models/ensemble.json`.
- **Hot reload**: the API hot‑loads `artifacts.json` every request. Model files are loaded on server start.

---

## Troubleshooting
- `_debug.ensemble` is false:
  - Restart API so it reloads saved models.
  - Ensure `algo/models/ensemble.json`, `feature_columns.json`, `scaler.json`, and required model files exist.
- Approvals too lenient/strict:
  - Re-run experiments and `select_threshold.py`, or set `decision_threshold` manually in `algo/artifacts.json` (hot‑loads).
- Slow training on CPU:
  - Use FAST mode (`$env:FAST_ENS="1"`), reduce `EPOCHS`/`N_SPLITS` in `train_ensemble.py`.

---

## GitHub: Push this project
```powershell
# From repo root (AI-Hack1)
git init
git remote add origin https://github.com/Tarun-03/credit-card-prediction-AI
git add .
git commit -m "Initial commit: FastAPI ensemble, experiments, frontend, README"
# If the GitHub repo is empty and you want this as the main branch
git branch -M main
git push -u origin main
```
If the remote already has history, pull first or create a new branch:
```powershell
git fetch origin
git checkout -b feature/setup
git add .
git commit -m "Add backend, experiments, frontend, README"
git push -u origin feature/setup
```
