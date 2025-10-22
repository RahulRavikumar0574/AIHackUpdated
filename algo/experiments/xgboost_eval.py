import os
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import ADASYN
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'final_model_ready_data.csv')
OUT_DIR = os.path.join(BASE_DIR, 'experiments', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def get_top_features(df: pd.DataFrame, target_col='RISK_ASSESSMENT', id_col='ID', top_n=10):
    leakage_features = [
        'Credit_History_Length', 'Count_Late_Payments',
        'Percentage_On_Time_Payments', 'Months_Since_Last_Delinquency'
    ]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in leakage_features + [target_col, id_col]]
    corrs = df[feature_cols].corrwith(df[target_col]).abs()
    top_features = corrs.sort_values(ascending=False).head(top_n).index.tolist()
    print(f"Top {top_n} non-leaky numeric features: {top_features}")
    return top_features


def youden_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, th = roc_curve(y_true, y_probs)
    youden = tpr - fpr
    idx = int(np.argmax(youden)) if len(youden) else 0
    thr = float(th[idx]) if len(th) else 0.5
    if not np.isfinite(thr):
        thr = 0.5
    return thr, float(youden[idx] if len(youden) else 0.0)


def run_xgb_eval(df: pd.DataFrame, best_params: Dict, target_col='RISK_ASSESSMENT', random_state=42):
    top_features = get_top_features(df, target_col=target_col)
    X = df[top_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    adasyn = ADASYN(random_state=random_state)
    X_train_bal, y_train_bal = adasyn.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(random_state=random_state, n_jobs=-1, eval_metric='logloss', **best_params)
    model.fit(X_train_bal, y_train_bal)

    # Calibrate
    cal_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
    cal_model.fit(X_train_bal, y_train_bal)

    y_proba_cal = cal_model.predict_proba(X_test)[:, 1]

    thr, j = youden_threshold(y_test.values, y_proba_cal)
    y_pred = (y_proba_cal >= thr).astype(int)

    ap_cal = average_precision_score(y_test, y_proba_cal)
    roc_cal = roc_auc_score(y_test, y_proba_cal)
    
    # Additional metrics
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    print(f"Calibrated XGBoost -> AP: {ap_cal:.4f}, ROC AUC: {roc_cal:.4f}, Youden thr: {thr:.3f}")
    print("\nClassification Report (Youden threshold):")
    print(classification_report(y_test, y_pred, target_names=['Low Risk (0)', 'High Risk (1)']))

    out = {
        'approach': 'xgboost_eval',
        'threshold': float(thr),
        'metrics': {
            'ROC_AUC': float(roc_cal),
            'AP': float(ap_cal),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'accuracy': float(acc)
        },
    }

    out_path = os.path.join(OUT_DIR, 'xgboost_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")

    return out


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    target_col = 'RISK_ASSESSMENT' if 'RISK_ASSESSMENT' in df.columns else 'target'
    
    best_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'reg_alpha': 0.0
    }
    
    run_xgb_eval(df, best_params, target_col=target_col)
