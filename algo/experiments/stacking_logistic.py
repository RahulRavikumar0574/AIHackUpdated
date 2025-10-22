import os
import json
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, roc_curve
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

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


def run_stacking_logistic(df: pd.DataFrame, target_col='RISK_ASSESSMENT', random_state=42):
    print("=== Stacking Ensemble with Logistic Regression Meta-Learner ===")
    
    top_features = get_top_features(df, target_col=target_col)
    X = df[top_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Base models
    base_models = [
        ('xgb', xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=random_state, eval_metric='logloss')),
        ('lgbm', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=random_state, verbosity=-1)),
    ]
    
    if CATBOOST_AVAILABLE:
        base_models.append(('cat', CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_state=random_state, verbose=False)))

    # Generate meta-features using cross-validation
    print("Generating meta-features from base models...")
    meta_features_train = []
    meta_features_test = []
    
    for name, model in base_models:
        print(f"  Training {name}...")
        # Cross-val predictions for train set
        train_preds = cross_val_predict(model, X_train, y_train, cv=3, method='predict_proba')[:, 1]
        meta_features_train.append(train_preds)
        
        # Fit on full train and predict test
        model.fit(X_train, y_train)
        test_preds = model.predict_proba(X_test)[:, 1]
        meta_features_test.append(test_preds)
    
    # Stack meta-features
    X_meta_train = np.column_stack(meta_features_train)
    X_meta_test = np.column_stack(meta_features_test)
    
    # Train meta-learner (Logistic Regression)
    print("Training Logistic Regression meta-learner...")
    meta_model = LogisticRegression(max_iter=1000, random_state=random_state)
    meta_model.fit(X_meta_train, y_train)
    
    # Predict
    y_test_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    
    # Find optimal threshold
    best_threshold, _ = youden_threshold(y_test.values, y_test_probs)
    y_test_pred = (y_test_probs >= best_threshold).astype(int)

    # Metrics
    ap = average_precision_score(y_test, y_test_probs)
    roc = roc_auc_score(y_test, y_test_probs)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nStacking Logistic Meta -> AP: {ap:.4f}, ROC AUC: {roc:.4f}, Youden thr: {best_threshold:.3f}")
    print("\nClassification Report (Youden threshold):")
    print(classification_report(y_test, y_test_pred, target_names=['Low Risk (0)', 'High Risk (1)']))

    # Save result
    out = {
        'approach': 'stacking_logistic',
        'threshold': float(best_threshold),
        'metrics': {
            'ROC_AUC': float(roc),
            'AP': float(ap),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'accuracy': float(acc)
        },
    }

    out_path = os.path.join(OUT_DIR, 'stacking_logistic_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    return out


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    target_col = 'RISK_ASSESSMENT' if 'RISK_ASSESSMENT' in df.columns else 'target'
    run_stacking_logistic(df, target_col=target_col)
