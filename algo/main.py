import os
import sys
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
from fastapi.staticfiles import StaticFiles
import joblib
import torch
from dl_models import TabMLP, TabCNN1D, TabLSTM, SimpleScaler

# -------------------------------
# Define request schema
# -------------------------------
class CreditData(BaseModel):
    gender: str
    age: int
    maritalStatus: str
    children: int
    familyMembers: int
    income: int
    incomeSource: str
    education: str
    employmentStatus: str
    yearsEmployed: int
    occupation: str
    ownCar: str
    ownRealty: str
    housingSituation: str
    creditHistoryLength: int = 0
    countLatePayments: int = 0
    percentageOnTimePayments: float = 100.0
    monthsSinceLastDelinquency: int = 999

# -------------------------------
# Initialize FastAPI
# -------------------------------
app = FastAPI(title="Credit Approval Prediction API")

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Serve static files
# -------------------------------
# Create the 'static' directory if it doesn't exist
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount the static directory to a URL path
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------------------------------
# Load CSV & Train Model
# -------------------------------
try:
    data_path = os.path.join(os.path.dirname(__file__), 'final_model_ready_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV file not found at {data_path}")
    data = pd.read_csv(data_path)

    # Identify target
    target_col = 'target' if 'target' in data.columns else 'RISK_ASSESSMENT'
    if target_col not in data.columns:
        raise ValueError("Target column not found in CSV")

    # Feature engineering
    if 'AGE_YEARS' not in data.columns:
        if 'DAYS_BIRTH' in data.columns:
            data['AGE_YEARS'] = abs(data['DAYS_BIRTH'] / 365)
        elif 'AGE' in data.columns:
            data['AGE_YEARS'] = data['AGE']

    if 'EMPLOYMENT_YEARS' not in data.columns:
        if 'DAYS_EMPLOYED' in data.columns:
            data['EMPLOYMENT_YEARS'] = abs(data['DAYS_EMPLOYED'] / 365)
        elif 'YEARS_EMPLOYED' in data.columns:
            data['EMPLOYMENT_YEARS'] = data['YEARS_EMPLOYED']

    # Ratios
    if 'AMT_INCOME_TOTAL' in data.columns and 'CNT_FAM_MEMBERS' in data.columns:
        denom = data['CNT_FAM_MEMBERS'].replace(0, np.nan)
        data['INCOME_PER_FAMILY'] = data['AMT_INCOME_TOTAL'] / denom

    if 'EMPLOYMENT_YEARS' in data.columns and 'AGE_YEARS' in data.columns:
        denom = data['AGE_YEARS'].replace(0, np.nan)
        data['EMPLOYMENT_RATIO'] = data['EMPLOYMENT_YEARS'] / denom

    if 'AMT_INCOME_TOTAL' in data.columns and 'EMPLOYMENT_YEARS' in data.columns:
        data['INCOME_PER_EMPLOYMENT_YEAR'] = data['AMT_INCOME_TOTAL'] / (data['EMPLOYMENT_YEARS'] + 1)

    # Encode categorical and boolean columns WITHOUT leaking target
    feature_base_cols = [c for c in data.columns if c not in [target_col]]
    features_df = data[feature_base_cols].copy()
    if 'ID' in features_df.columns:
        features_df = features_df.drop(columns=['ID'])

    categorical_cols = features_df.select_dtypes(include='object').columns
    X = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)

    bool_cols = X.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    # Target vector
    y = data[target_col]
    try:
        y = y.astype(int)
    except Exception:
        y = y.map({True: 1, False: 0, '1': 1, '0': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0}).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Drop zero-variance columns
    constant_cols = X.columns[X.nunique() <= 1]
    X = X.drop(columns=constant_cols)

    # Train final model
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        scale_pos_weight=10,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    model.fit(X, y)
    model_columns = X.columns.tolist()

    # Compute optimal threshold based on training/validation split
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train_val, y_train_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
    youden = tpr - fpr
    opt_idx = np.argmax(youden)
    optimal_threshold = thresholds[opt_idx] if thresholds is not None and len(thresholds) > 0 else 0.5
    if np.isnan(optimal_threshold):
        optimal_threshold = 0.5

    # Attempt to load decision threshold from artifacts.json (generated by credit_pipeline)
    artifacts_path = os.path.join(os.path.dirname(__file__), 'artifacts.json')
    if os.path.exists(artifacts_path):
        try:
            with open(artifacts_path, 'r', encoding='utf-8') as f:
                artifacts = json.load(f)
                if isinstance(artifacts, dict) and 'decision_threshold' in artifacts:
                    optimal_threshold = float(artifacts['decision_threshold'])
                    print(f"Loaded decision threshold from artifacts.json: {optimal_threshold:.4f}")
        except Exception as _:
            pass
    print(f"Optimized decision threshold (in-use): {optimal_threshold:.4f}")

    # Attempt to load ensemble artifacts (if available)
    ENSEMBLE_DIR = os.path.join(os.path.dirname(__file__), 'models')
    ensemble_enabled = False
    lgbm_ens = None
    mlp_ens = None
    cnn_ens = None
    lstm_ens = None
    scaler_ens = None
    ens_weights = None
    ens_bias = None
    ens_feature_columns = None
    ens_model_names = None

    try:
        feature_cols_path = os.path.join(ENSEMBLE_DIR, 'feature_columns.json')
        ensemble_json_path = os.path.join(ENSEMBLE_DIR, 'ensemble.json')
        scaler_path = os.path.join(ENSEMBLE_DIR, 'scaler.json')
        lgbm_path = os.path.join(ENSEMBLE_DIR, 'lgbm.pkl')
        mlp_path = os.path.join(ENSEMBLE_DIR, 'mlp.pt')
        cnn_path = os.path.join(ENSEMBLE_DIR, 'cnn1d.pt')
        lstm_path = os.path.join(ENSEMBLE_DIR, 'lstm.pt')

        # Load config first to know which models are required
        if os.path.exists(feature_cols_path) and os.path.exists(ensemble_json_path) and os.path.exists(scaler_path):
            with open(feature_cols_path, 'r', encoding='utf-8') as f:
                ens_feature_columns = json.load(f)['feature_columns']
            with open(ensemble_json_path, 'r', encoding='utf-8') as f:
                ens_cfg = json.load(f)
                ens_weights = np.array(ens_cfg['weights'], dtype=float)
                ens_bias = float(ens_cfg['bias'])
                ens_model_names = ens_cfg.get('models', ["lgbm", "mlp", "cnn1d", "lstm"])
            with open(scaler_path, 'r', encoding='utf-8') as f:
                scaler_ens = SimpleScaler.from_dict(json.load(f))

            # Check required model files dynamically
            required = []
            if "lgbm" in ens_model_names:
                required.append(lgbm_path)
            if "mlp" in ens_model_names:
                required.append(mlp_path)
            if "cnn1d" in ens_model_names:
                required.append(cnn_path)
            if "lstm" in ens_model_names:
                required.append(lstm_path)

            if all(os.path.exists(p) for p in required):
                in_dim = len(ens_feature_columns)
                if "lgbm" in ens_model_names:
                    lgbm_ens = joblib.load(lgbm_path)
                if "mlp" in ens_model_names:
                    mlp_ens = TabMLP(in_dim=in_dim)
                    mlp_ens.load_state_dict(torch.load(mlp_path, map_location='cpu'))
                    mlp_ens.eval()
                if "cnn1d" in ens_model_names:
                    cnn_ens = TabCNN1D(in_dim=in_dim)
                    cnn_ens.load_state_dict(torch.load(cnn_path, map_location='cpu'))
                    cnn_ens.eval()
                if "lstm" in ens_model_names:
                    lstm_ens = TabLSTM(in_dim=in_dim)
                    lstm_ens.load_state_dict(torch.load(lstm_path, map_location='cpu'))
                    lstm_ens.eval()

                ensemble_enabled = True
                print(f"Ensemble artifacts loaded: using models {ens_model_names}")
    except Exception as _e:
        print("Ensemble load failed or missing; falling back to LightGBM-only. Reason:", _e)

except Exception as e:
    print("Error loading/training model:", e)
    model = None
    model_columns = None
    optimal_threshold = 0.5
    ensemble_enabled = False

if model is not None and model_columns is not None:
    print(f"Model trained successfully with {len(model_columns)} features")
else:
    print("Model training failed. Check CSV path and preprocessing.")

# -------------------------------
# API endpoints
# -------------------------------
@app.get("/")
def home():
    return {"message": "Credit Card Approval Prediction API is running!"}

@app.post("/predict")
def predict_credit_approval(data: CreditData):
    if model is None or model_columns is None:
        return {"error": "ML model not loaded."}

    # Hot-reload decision threshold from artifacts.json if present
    try:
        artifacts_path = os.path.join(os.path.dirname(__file__), 'artifacts.json')
        if os.path.exists(artifacts_path):
            with open(artifacts_path, 'r', encoding='utf-8') as f:
                artifacts = json.load(f)
                if isinstance(artifacts, dict) and 'decision_threshold' in artifacts:
                    current_threshold = float(artifacts['decision_threshold'])
                else:
                    current_threshold = optimal_threshold
        else:
            current_threshold = optimal_threshold
    except Exception:
        current_threshold = optimal_threshold

    input_data = data.dict()

    # Feature engineering
    years_employed_raw = input_data['yearsEmployed']
    engineered_features = {
        'AGE_YEARS': input_data['age'],
        'EMPLOYMENT_YEARS': years_employed_raw,
        'INCOME_PER_FAMILY': input_data['income'] / input_data['familyMembers'] if input_data['familyMembers'] > 0 else 0,
        'INCOME_PER_EMPLOYMENT_YEAR': input_data['income'] / years_employed_raw if years_employed_raw > 0 else 0,
        'EMPLOYMENT_RATIO': years_employed_raw / input_data['age'] if input_data['age'] > 0 else 0,
    }

    # IMPORTANT: Do NOT include post-outcome credit-history variables to avoid leakage
    # (creditHistoryLength, countLatePayments, percentageOnTimePayments, monthsSinceLastDelinquency)
    processed_data = {**input_data, **engineered_features}
    processed_df = pd.DataFrame([processed_data])

    # Apply one-hot encoding to categorical (object) columns to match training
    cat_cols = processed_df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        processed_df = pd.get_dummies(processed_df, columns=cat_cols, drop_first=True)

    # Cast booleans to integers
    bool_cols = processed_df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        processed_df[bool_cols] = processed_df[bool_cols].astype(int)

    # Coerce non-numeric columns to numeric when possible
    for col in processed_df.columns:
        if not np.issubdtype(processed_df[col].dtype, np.number):
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

    # Replace inf and fill NaNs with medians
    processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
    if processed_df.isna().any().any():
        processed_df = processed_df.fillna(processed_df.median(numeric_only=True))

    # Align columns for LightGBM-only model
    final_df = pd.DataFrame(columns=model_columns)
    for col in model_columns:
        final_df[col] = processed_df[col] if col in processed_df.columns else 0

    # Predict (default)
    prob_default = float(model.predict_proba(final_df)[:, 1][0])
    p_lgbm_only = prob_default

    # If ensemble is available, compute ensemble probability with safeguards
    ens_used = False
    ens_prob = None
    ens_components = {}
    if 'ensemble_enabled' in globals() and ensemble_enabled:
        # Build ensemble feature vector aligned to ens_feature_columns
        ens_df = pd.DataFrame(columns=ens_feature_columns)
        for col in ens_feature_columns:
            ens_df[col] = processed_df[col] if col in processed_df.columns else 0
        X_np = ens_df.values.astype(np.float32)
        X_np = scaler_ens.transform(X_np)
        with torch.no_grad():
            comps_list = []
            ens_components = {}
            # LGBM prob (if included)
            if "lgbm" in ens_model_names and lgbm_ens is not None:
                p = float(lgbm_ens.predict_proba(ens_df)[:, 1][0])
                ens_components["p_lgbm"] = p
                comps_list.append(p)
            # Torch models
            x_tensor = torch.tensor(X_np, dtype=torch.float32)
            if "mlp" in ens_model_names and mlp_ens is not None:
                p = float(torch.sigmoid(mlp_ens(x_tensor)).cpu().numpy()[0])
                ens_components["p_mlp"] = p
                comps_list.append(p)
            if "cnn1d" in ens_model_names and cnn_ens is not None:
                p = float(torch.sigmoid(cnn_ens(x_tensor)).cpu().numpy()[0])
                ens_components["p_cnn"] = p
                comps_list.append(p)
            if "lstm" in ens_model_names and lstm_ens is not None:
                p = float(torch.sigmoid(lstm_ens(x_tensor)).cpu().numpy()[0])
                ens_components["p_lstm"] = p
                comps_list.append(p)

        # Validate components
        comps = np.array(comps_list, dtype=float)
        if np.all(np.isfinite(comps)):
            probs_vec = comps
            logit = float(ens_bias + np.dot(ens_weights, probs_vec))
            ens_prob = float(1.0 / (1.0 + np.exp(-logit)))
            if np.isfinite(ens_prob):
                prob_default = ens_prob
                ens_used = True
            else:
                # fall back to LightGBM-only
                prob_default = p_lgbm_only
        else:
            # fall back to LightGBM-only
            prob_default = p_lgbm_only

    # Sanitize outputs to avoid NaN/Inf in JSON
    def _safe01(x: float, fallback: float = 0.5) -> float:
        try:
            if not np.isfinite(x):
                return float(fallback)
            # clamp to [0,1]
            if x < 0.0:
                return 0.0
            if x > 1.0:
                return 1.0
            return float(x)
        except Exception:
            return float(fallback)

    prob_default = _safe01(prob_default, 0.5)
    current_threshold = _safe01(current_threshold, optimal_threshold if np.isfinite(optimal_threshold) else 0.5)

    is_approved = bool(prob_default < current_threshold)
    message = "Congratulations! Your application has been approved." if is_approved else "We regret to inform you that your application has been denied."

    debug_block = {
        "prob_default": prob_default,
        "threshold": current_threshold,
        "ensemble": bool(ens_used),
        "p_lgbm_only": _safe01(p_lgbm_only, 0.5)
    }
    # attach component probs if available
    if ens_components:
        for k, v in ens_components.items():
            debug_block[k] = _safe01(v, 0.5)

    return {
        "isApproved": is_approved,
        "message": message,
        "risk_probability_default": prob_default,
        "decision_threshold": current_threshold,
        "_debug": debug_block
    }

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
