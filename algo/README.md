# Credit Card Approval - ML Pipeline

This project builds a credit default classifier and produces per-applicant approval decisions using a pre-combined dataset `final_model_ready_data.csv`.

## Overview
- Loads model-ready data from `final_model_ready_data.csv`
- Engineers features, encodes categoricals, handles imbalance, and trains a LightGBM model
- Evaluates on a holdout set and generates visualizations
- Produces a narrative markdown report and a CSV of per-row approval decisions

## Getting Started

### 1) Python environment
```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

### 2) Data
Place your dataset as `final_model_ready_data.csv` in the project root. The script auto-detects the target column (`target` or `RISK_ASSESSMENT`).

## Run
```bash
python credit_pipeline.py
```

## Outputs
- `credit_approvals.csv` — per-applicant outputs:
  - `ID` (or `ROW_ID` if no ID column)
  - `risk_probability_default` (0..1)
  - `decision` (Approved/Rejected)
- `credit_analysis_model.png` — ROC, confusion matrix, top features, income-by-target
- `credit_analysis_demographics.png` — age, years employed, employment ratio by target
- `credit_report.md` — narrative overview, metrics, top features, and plot references

## Approval Logic
- The model outputs a probability of default per applicant.
- An approval threshold is selected via ROC Youden's J on the validation split (fallback 0.5).
- Decision rule: if `prob_default < threshold` → `Approved`, else `Rejected`.

## Notes
- Extremely high metrics can indicate data leakage; consider adding time-based splits and leakage checks for production use.
- You can adjust `.gitignore` to include or exclude generated outputs.

## License
MIT (or choose your own)
