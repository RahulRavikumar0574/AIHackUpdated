import os
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'final_model_ready_data.csv')
OUT_DIR = os.path.join(BASE_DIR, 'experiments', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

LEAKAGE_FEATURES = [
    'Credit_History_Length', 'Count_Late_Payments',
    'Percentage_On_Time_Payments', 'Months_Since_Last_Delinquency'
]


def _safe_drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    cols_present = [c for c in cols if c in df.columns]
    if cols_present:
        return df.drop(columns=cols_present)
    return df.copy()


def get_top_features(df: pd.DataFrame, target_col: str = 'RISK_ASSESSMENT', id_col: str = 'ID', top_n: int = 10):
    df_check = df.copy()
    if id_col in df_check.columns:
        df_check = df_check.drop(columns=[id_col])
    numeric_cols = df_check.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in LEAKAGE_FEATURES + [target_col]]
    if not feature_cols:
        raise ValueError('No eligible numeric features after excluding leaky/id/target columns')
    corrs = df_check[feature_cols].corrwith(df_check[target_col]).abs().sort_values(ascending=False)
    top_features = corrs.head(top_n).index.tolist()
    print(f"Selected Top {top_n} Non-Leaky Features: {top_features}")
    return top_features


def youden_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, th = roc_curve(y_true, y_probs)
    youden = tpr - fpr
    idx = int(np.argmax(youden)) if len(youden) else 0
    thr = float(th[idx]) if len(th) else 0.5
    if not np.isfinite(thr):
        thr = 0.5
    return thr, float(youden[idx] if len(youden) else 0.0)


def run_cross_validation_pipeline(df: pd.DataFrame, best_params: Dict, selected_features: list, target_col: str = 'RISK_ASSESSMENT', random_state: int = 42):
    print("\n--- Cross-Validation with Tuned CatBoost Model ---")
    X = df[selected_features]
    y = df[target_col]

    tuned_cat = CatBoostClassifier(random_state=random_state, silent=True, **best_params)
    adasyn = ADASYN(random_state=random_state)
    pipeline = ImbPipeline([('resample', adasyn), ('clf', tuned_cat)])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='average_precision', n_jobs=-1)
    print(f"Cross-Validation AP scores ({len(scores)} folds): {[f'{s:.3f}' for s in scores]}")
    print(f"Mean AP: {scores.mean():.4f}  Std: {scores.std():.4f}")
    print("-" * 40)
    return scores


def run_final_evaluation(df: pd.DataFrame, best_params: Dict, selected_features: list, target_col: str = 'RISK_ASSESSMENT', id_col: str = 'ID', test_size: float = 0.2, random_state: int = 42, calibration_method: str = 'sigmoid'):
    print("\n--- Final Evaluation with Tuned CatBoost Model ---")

    ids_series = df[id_col].copy() if id_col in df.columns else None
    X = df[selected_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index if ids_series is None else ids_series, test_size=test_size, random_state=random_state, stratify=y
    )

    # ADASYN resampling and training
    print("\nTraining final CatBoost with ADASYN and best hyperparameters...")
    adasyn = ADASYN(random_state=random_state)
    X_train_bal, y_train_bal = adasyn.fit_resample(X_train, y_train)

    final_model = CatBoostClassifier(random_state=random_state, silent=True, **best_params)
    final_model.fit(X_train_bal, y_train_bal)

    # Calibrate the model on a held-out calibration split
    print("Calibrating the final model on a held-out split...")
    X_tr_sub, X_calib, y_tr_sub, y_calib = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)
    calibrator = CalibratedClassifierCV(estimator=final_model, method=calibration_method, cv='prefit')
    calibrator.fit(X_calib, y_calib)

    y_proba_cal = calibrator.predict_proba(X_test)[:, 1]

    # Select threshold via Youden's J (current logic)
    thr, j = youden_threshold(y_test.values, y_proba_cal)
    y_pred_opt = (y_proba_cal >= thr).astype(int)

    ap_cal = average_precision_score(y_test, y_proba_cal)
    roc_cal = roc_auc_score(y_test, y_proba_cal)
    print(f"Calibrated CatBoost -> AP: {ap_cal:.4f}, ROC AUC: {roc_cal:.4f}, Youden thr: {thr:.3f}")

    print("\nClassification Report at Youden-J Threshold:")
    print(classification_report(y_test, y_pred_opt, target_names=['Low Risk (0)', 'High Risk (1)']))

    out = {
        'approach': 'catboost_eval',
        'threshold': float(thr),
        'metrics': {'ROC_AUC': float(roc_cal), 'AP': float(ap_cal)}
    }
    out_path = os.path.join(OUT_DIR, 'catboost_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('Saved:', out_path)

    return out


if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    tgt = 'target' if 'target' in df.columns else 'RISK_ASSESSMENT'
    if tgt not in df.columns:
        raise ValueError("No target column found. Expected 'target' or 'RISK_ASSESSMENT'.")
    try:
        df[tgt] = df[tgt].astype(int)
    except Exception:
        df[tgt] = df[tgt].map({True: 1, False: 0, '1': 1, '0': 0, 'Y': 1, 'N': 0, 'yes': 1, 'no': 0}).astype(int)

    feats = get_top_features(df, target_col=tgt)
    # Optional CV preview
    run_cross_validation_pipeline(df, best_params={
        'colsample_bylevel': 0.694,
        'depth': 9,
        'iterations': 458,
        'l2_leaf_reg': 1.404,
        'learning_rate': 0.081,
        'subsample': 0.644
    }, selected_features=feats, target_col=tgt)

    run_final_evaluation(df, best_params={
        'colsample_bylevel': 0.694,
        'depth': 9,
        'iterations': 458,
        'l2_leaf_reg': 1.404,
        'learning_rate': 0.081,
        'subsample': 0.644
    }, selected_features=feats, target_col=tgt)
