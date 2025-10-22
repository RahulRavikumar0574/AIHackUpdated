import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

from lightgbm import LGBMClassifier

from dl_models import TabMLP, TabCNN1D, TabLSTM, SimpleScaler, train_torch_model, save_state

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# Fast mode controls (set FAST_ENS=1 to enable)
# -----------------------------
FAST = os.getenv('FAST_ENS', '0') == '1'
N_SPLITS = 3 if FAST else 5
EPOCHS = 2 if FAST else 10
INCLUDE_CNN_LSTM = not FAST  # when FAST, skip CNN/LSTM to speed up


def _map_dataset_to_form_fields(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a DataFrame using the SAME schema the API expects from the form.
    Fields: gender, age, maritalStatus, children, familyMembers, income, incomeSource, education,
            employmentStatus, yearsEmployed, occupation, ownCar, ownRealty, housingSituation
    Then compute engineered features: AGE_YEARS, EMPLOYMENT_YEARS, INCOME_PER_FAMILY,
            INCOME_PER_EMPLOYMENT_YEAR, EMPLOYMENT_RATIO
    Return a wide one-hot encoded feature matrix consistent with API preprocessing.
    """
    data = df.copy()

    # Target vector
    y = data[target_col]
    try:
        y = y.astype(int)
    except Exception:
        y = y.map({True: 1, False: 0, '1': 1, '0': 0, 'Y': 1, 'N': 0, 'yes': 1, 'no': 0}).astype(int)

    # Map dataset columns to form-like raw inputs
    # Gender
    if 'CODE_GENDER' in data.columns:
        data['gender'] = data['CODE_GENDER'].map({'M': 'Male', 'F': 'Female'}).fillna('Unknown')
    elif 'gender' not in data.columns:
        data['gender'] = 'Unknown'

    # Age (years)
    if 'DAYS_BIRTH' in data.columns:
        data['AGE_YEARS'] = (data['DAYS_BIRTH'].abs() / 365.0)
    elif 'AGE' in data.columns:
        data['AGE_YEARS'] = data['AGE']
    else:
        data['AGE_YEARS'] = np.nan
    data['age'] = data['AGE_YEARS']  # for parity with API input

    # Employment years & status
    if 'DAYS_EMPLOYED' in data.columns:
        data['EMPLOYMENT_YEARS'] = (data['DAYS_EMPLOYED'].abs() / 365.0)
    elif 'YEARS_EMPLOYED' in data.columns:
        data['EMPLOYMENT_YEARS'] = data['YEARS_EMPLOYED']
    else:
        data['EMPLOYMENT_YEARS'] = 0.0
    data['yearsEmployed'] = data['EMPLOYMENT_YEARS']
    data['employmentStatus'] = np.where(data['EMPLOYMENT_YEARS'] > 0, 'Employed', 'Unemployed')

    # Income and ratios
    if 'AMT_INCOME_TOTAL' in data.columns:
        data['income'] = data['AMT_INCOME_TOTAL']
    else:
        data['income'] = 0.0

    if 'CNT_FAM_MEMBERS' in data.columns:
        data['familyMembers'] = data['CNT_FAM_MEMBERS'].replace(0, 1)
    else:
        data['familyMembers'] = 1

    if 'CNT_CHILDREN' in data.columns:
        data['children'] = data['CNT_CHILDREN']
    else:
        data['children'] = 0

    data['INCOME_PER_FAMILY'] = data['income'] / data['familyMembers'].replace(0, np.nan)
    data['INCOME_PER_EMPLOYMENT_YEAR'] = data['income'] / (data['EMPLOYMENT_YEARS'] + 1)
    data['EMPLOYMENT_RATIO'] = data['EMPLOYMENT_YEARS'] / data['AGE_YEARS'].replace(0, np.nan)

    # Categorical fields mapping
    data['maritalStatus'] = data.get('NAME_FAMILY_STATUS', pd.Series(['Unknown'] * len(data))).astype(str)
    data['incomeSource'] = data.get('NAME_INCOME_TYPE', pd.Series(['Unknown'] * len(data))).astype(str)
    data['education'] = data.get('NAME_EDUCATION_TYPE', pd.Series(['Unknown'] * len(data))).astype(str)
    data['housingSituation'] = data.get('NAME_HOUSING_TYPE', pd.Series(['Unknown'] * len(data))).astype(str)
    data['occupation'] = data.get('OCCUPATION_TYPE', pd.Series(['Unknown'] * len(data))).astype(str)

    # Ownership flags -> Yes/No strings (to match form)
    if 'FLAG_OWN_CAR' in data.columns:
        data['ownCar'] = data['FLAG_OWN_CAR'].map({1: 'Yes', 0: 'No', 'Y': 'Yes', 'N': 'No'}).fillna('No').astype(str)
    else:
        data['ownCar'] = 'No'
    if 'FLAG_OWN_REALTY' in data.columns:
        data['ownRealty'] = data['FLAG_OWN_REALTY'].map({1: 'Yes', 0: 'No', 'Y': 'Yes', 'N': 'No'}).fillna('No').astype(str)
    else:
        data['ownRealty'] = 'No'

    # Compose the API-like row inputs (same keys main.py expects)
    api_cols = [
        'gender', 'age', 'maritalStatus', 'children', 'familyMembers', 'income', 'incomeSource', 'education',
        'employmentStatus', 'yearsEmployed', 'occupation', 'ownCar', 'ownRealty', 'housingSituation'
    ]
    # Keep engineered features as well since API includes them prior to encoding
    engineered = ['AGE_YEARS', 'EMPLOYMENT_YEARS', 'INCOME_PER_FAMILY', 'INCOME_PER_EMPLOYMENT_YEAR', 'EMPLOYMENT_RATIO']

    # IMPORTANT: Do NOT include post-outcome credit-history variables to avoid leakage
    # (creditHistoryLength, countLatePayments, percentageOnTimePayments, monthsSinceLastDelinquency)
    base_df = data[api_cols + engineered].copy()

    # One-hot encode on object columns (same as API)
    cat_cols = base_df.select_dtypes(include='object').columns
    X = pd.get_dummies(base_df, columns=cat_cols, drop_first=True)

    # Cast bools to int, coerce non-numeric
    bool_cols = X.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors='coerce')

    # Clean inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    # Drop constant columns
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    return X, y


def main():
    data_path = os.path.join(BASE_DIR, 'final_model_ready_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    if 'target' in df.columns:
        target_col = 'target'
    elif 'RISK_ASSESSMENT' in df.columns:
        target_col = 'RISK_ASSESSMENT'
    else:
        raise ValueError("No target column found. Expected 'target' or 'RISK_ASSESSMENT'.")

    # Build features STRICTLY with the API schema
    X_all, y_all = _map_dataset_to_form_fields(df, target_col)
    feature_columns = X_all.columns.tolist()

    # Split once into train_dev and holdout to evaluate threshold without leakage
    X_trd, X_hold, y_trd, y_hold = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # Scaler fit on train_dev
    scaler_trd = SimpleScaler()
    X_trd_np = scaler_trd.fit_transform(X_trd.values.astype(np.float32))

    # Prepare OOF container for meta-learner on train_dev
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    base_count = 2 + (2 if INCLUDE_CNN_LSTM else 0)  # lgbm + mlp + (cnn + lstm)
    P_oof = np.zeros((len(X_trd), base_count), dtype=np.float32)
    oof_idx_all = np.zeros(len(X_trd), dtype=bool)

    X_trd_array = X_trd.values
    y_trd_array = y_trd.values

    # OOF predictions for base models
    for fold, (idx_tr, idx_val) in enumerate(skf.split(X_trd_array, y_trd_array), 1):
        X_tr, X_val = X_trd.iloc[idx_tr], X_trd.iloc[idx_val]
        y_tr_, y_val_ = y_trd.iloc[idx_tr], y_trd.iloc[idx_val]

        # Fit scaler within train_dev fold to avoid leakage
        scaler_fold = SimpleScaler()
        X_tr_np = scaler_fold.fit_transform(X_tr.values.astype(np.float32))
        X_val_np = scaler_fold.transform(X_val.values.astype(np.float32))

        # LightGBM
        lgbm = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            class_weight='balanced',
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        lgbm.fit(X_tr, y_tr_)
        p_lgbm = lgbm.predict_proba(X_val)[:, 1]

        # MLP
        mlp = TabMLP(in_dim=X_tr_np.shape[1])
        info_mlp = train_torch_model(mlp, X_tr_np, y_tr_.values, X_val_np, y_val_.values, epochs=EPOCHS)
        p_mlp = info_mlp['val_probs']

        P_oof[idx_val, 0] = p_lgbm
        P_oof[idx_val, 1] = p_mlp

        col_ptr = 2
        if INCLUDE_CNN_LSTM:
            # CNN1D
            cnn = TabCNN1D(in_dim=X_tr_np.shape[1])
            info_cnn = train_torch_model(cnn, X_tr_np, y_tr_.values, X_val_np, y_val_.values, epochs=EPOCHS)
            p_cnn = info_cnn['val_probs']
            P_oof[idx_val, col_ptr] = p_cnn
            col_ptr += 1

            # LSTM
            lstm = TabLSTM(in_dim=X_tr_np.shape[1])
            info_lstm = train_torch_model(lstm, X_tr_np, y_tr_.values, X_val_np, y_val_.values, epochs=EPOCHS)
            p_lstm = info_lstm['val_probs']
            P_oof[idx_val, col_ptr] = p_lstm
        oof_idx_all[idx_val] = True

    # Ensure OOF filled
    assert oof_idx_all.all(), "OOF predictions missing for some rows"

    # Train meta-learner on OOF
    meta = LogisticRegression(max_iter=1000)
    meta.fit(P_oof, y_trd.values)

    # Evaluate on holdout: train base models on full train_dev, predict holdout
    # Fit scaler on train_dev to use for holdout
    X_hold_np = scaler_trd.transform(X_hold.values.astype(np.float32))

    lgbm_full = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgbm_full.fit(X_trd, y_trd)
    p_lgbm_hold = lgbm_full.predict_proba(X_hold)[:, 1]

    mlp_full = TabMLP(in_dim=X_trd_np.shape[1])
    info_mlp_full = train_torch_model(mlp_full, X_trd_np, y_trd.values, X_hold_np, y_hold.values, epochs=EPOCHS)
    p_mlp_hold = info_mlp_full['val_probs']

    probs_hold_list = [p_lgbm_hold, p_mlp_hold]
    if INCLUDE_CNN_LSTM:
        cnn_full = TabCNN1D(in_dim=X_trd_np.shape[1])
        info_cnn_full = train_torch_model(cnn_full, X_trd_np, y_trd.values, X_hold_np, y_hold.values, epochs=EPOCHS)
        p_cnn_hold = info_cnn_full['val_probs']
        probs_hold_list.append(p_cnn_hold)

        lstm_full = TabLSTM(in_dim=X_trd_np.shape[1])
        info_lstm_full = train_torch_model(lstm_full, X_trd_np, y_trd.values, X_hold_np, y_hold.values, epochs=EPOCHS)
        p_lstm_hold = info_lstm_full['val_probs']
        probs_hold_list.append(p_lstm_hold)

    P_hold = np.vstack(probs_hold_list).T
    p_meta_hold = meta.predict_proba(P_hold)[:, 1]

    roc = roc_auc_score(y_hold, p_meta_hold)
    ap = average_precision_score(y_hold, p_meta_hold)
    print(f"Ensemble (API schema, CV stacking) ROC-AUC: {roc:.4f} | AP: {ap:.4f}")

    # Choose threshold by Youden's J on holdout
    fpr, tpr, th = roc_curve(y_hold, p_meta_hold)
    youden = tpr - fpr
    idx = np.argmax(youden)
    threshold = float(th[idx]) if len(th) > 0 else 0.5
    if np.isnan(threshold):
        threshold = 0.5
    print(f"Selected decision_threshold (holdout): {threshold:.4f}")

    # FINAL: Fit base models on ALL data for inference artifacts
    scaler_all = SimpleScaler()
    X_all_np = scaler_all.fit_transform(X_all.values.astype(np.float32))

    lgbm_all = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgbm_all.fit(X_all, y_all)

    mlp_all = TabMLP(in_dim=X_all_np.shape[1])
    _ = train_torch_model(mlp_all, X_all_np, y_all.values, X_all_np, y_all.values, epochs=EPOCHS)

    if INCLUDE_CNN_LSTM:
        cnn_all = TabCNN1D(in_dim=X_all_np.shape[1])
        _ = train_torch_model(cnn_all, X_all_np, y_all.values, X_all_np, y_all.values, epochs=EPOCHS)

        lstm_all = TabLSTM(in_dim=X_all_np.shape[1])
        _ = train_torch_model(lstm_all, X_all_np, y_all.values, X_all_np, y_all.values, epochs=EPOCHS)

    # Save artifacts
    joblib.dump(lgbm_all, os.path.join(MODELS_DIR, 'lgbm.pkl'))
    save_state(mlp_all, os.path.join(MODELS_DIR, 'mlp.pt'))
    if INCLUDE_CNN_LSTM:
        save_state(cnn_all, os.path.join(MODELS_DIR, 'cnn1d.pt'))
        save_state(lstm_all, os.path.join(MODELS_DIR, 'lstm.pt'))

    with open(os.path.join(MODELS_DIR, 'feature_columns.json'), 'w', encoding='utf-8') as f:
        json.dump({"feature_columns": feature_columns}, f)

    with open(os.path.join(MODELS_DIR, 'scaler.json'), 'w', encoding='utf-8') as f:
        json.dump(scaler_all.to_dict(), f)

    # Save meta-learner coefficients as weights
    weights = meta.coef_.ravel().tolist()
    bias = float(meta.intercept_[0])
    model_list = ["lgbm", "mlp"] + (["cnn1d", "lstm"] if INCLUDE_CNN_LSTM else [])
    with open(os.path.join(MODELS_DIR, 'ensemble.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "type": "logistic",
            "weights": weights,
            "bias": bias,
            "models": model_list,
            "decision_threshold": threshold
        }, f, indent=2)

    # Also update top-level artifacts.json for FastAPI
    with open(os.path.join(BASE_DIR, 'artifacts.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "best_model_name": "ensemble",
            "decision_threshold": threshold,
            "metrics": {"ROC_AUC": roc, "AP": ap}
        }, f, indent=2)

    print("Saved ensemble artifacts to:")
    print(f" - {MODELS_DIR}")
    print(" - artifacts.json (decision threshold for API)")


if __name__ == '__main__':
    main()
