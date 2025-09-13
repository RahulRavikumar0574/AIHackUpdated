import os
import json
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import ADASYN

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


def youden_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, th = roc_curve(y_true, y_probs)
    youden = tpr - fpr
    idx = int(np.argmax(youden)) if len(youden) else 0
    thr = float(th[idx]) if len(th) else 0.5
    if not np.isfinite(thr):
        thr = 0.5
    return thr, float(youden[idx] if len(youden) else 0.0)


def run_credit_pipeline_eval(df: pd.DataFrame, target_col: str = 'RISK_ASSESSMENT', id_col: str = 'ID', random_state: int = 42):
    """Mirror core of credit_pipeline: compare ADASYN vs scale_pos_weight, calibrate best,
    then select decision threshold via Youden's J on the final test set.
    Writes a result JSON compatible with select_threshold.py
    """
    drop_candidates = [c for c in LEAKAGE_FEATURES if c in df.columns]
    ids_series = df[id_col].copy() if id_col in df.columns else None
    if ids_series is not None:
        drop_candidates.append(id_col)

    X = _safe_drop_columns(df, drop_candidates + [target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index if ids_series is None else ids_series, test_size=0.2, random_state=random_state, stratify=y
    )

    # 1) ADASYN -> LGBM
    adasyn = ADASYN(random_state=random_state)
    X_train_bal, y_train_bal = adasyn.fit_resample(X_train, y_train)

    model_adasyn = lgb.LGBMClassifier(random_state=random_state, n_jobs=-1, verbosity=-1,
                                      n_estimators=800, learning_rate=0.05)
    model_adasyn.fit(X_train_bal, y_train_bal,
                     eval_set=[(X_test, y_test)],
                     eval_metric='auc',
                     callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    y_proba_adasyn = model_adasyn.predict_proba(X_test)[:, 1]
    ap_adasyn = average_precision_score(y_test, y_proba_adasyn)
    roc_adasyn = roc_auc_score(y_test, y_proba_adasyn)

    # 2) scale_pos_weight -> LGBM
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / (n_pos + 1e-9)

    model_spw = lgb.LGBMClassifier(random_state=random_state, n_jobs=-1, verbosity=-1,
                                   n_estimators=800, learning_rate=0.05,
                                   scale_pos_weight=scale)
    model_spw.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    y_proba_spw = model_spw.predict_proba(X_test)[:, 1]
    ap_spw = average_precision_score(y_test, y_proba_spw)
    roc_spw = roc_auc_score(y_test, y_proba_spw)

    if ap_adasyn >= ap_spw:
        best_model = model_adasyn
        best_name = 'ADASYN'
    else:
        best_model = model_spw
        best_name = 'scale_pos_weight'

    # Calibration on a held-out calibration split from training
    X_tr_sub, X_calib, y_tr_sub, y_calib = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)
    calibrator = CalibratedClassifierCV(estimator=best_model, method='sigmoid', cv='prefit')
    calibrator.fit(X_calib, y_calib)

    # Evaluate calibrated probs on the test set
    y_proba_cal = calibrator.predict_proba(X_test)[:, 1]
    ap_cal = average_precision_score(y_test, y_proba_cal)
    roc_cal = roc_auc_score(y_test, y_proba_cal)

    thr, j = youden_threshold(y_test.values, y_proba_cal)
    y_pred = (y_proba_cal >= thr).astype(int)

    print(f"credit_pipeline (calibrated LGBM best={best_name}) -> AP: {ap_cal:.4f}, ROC AUC: {roc_cal:.4f}, Youden thr: {thr:.3f}")
    print("\nClassification Report (Youden threshold):")
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, target_names=['Low Risk (0)', 'High Risk (1)']))

    out = {
        'approach': 'credit_pipeline',
        'threshold': float(thr),
        'metrics': {'ROC_AUC': float(roc_cal), 'AP': float(ap_cal)}
    }
    out_path = os.path.join(OUT_DIR, 'credit_pipeline_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('Saved:', out_path)

    return out


if __name__ == '__main__':
    data_path = DATA_PATH
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV not found: {data_path}")
    df = pd.read_csv(data_path)
    tgt = 'target' if 'target' in df.columns else 'RISK_ASSESSMENT'
    if tgt not in df.columns:
        raise ValueError("No target column found. Expected 'target' or 'RISK_ASSESSMENT'.")
    try:
        df[tgt] = df[tgt].astype(int)
    except Exception:
        df[tgt] = df[tgt].map({True: 1, False: 0, '1': 1, '0': 0, 'Y': 1, 'N': 0, 'yes': 1, 'no': 0}).astype(int)

    run_credit_pipeline_eval(df, target_col=tgt)
