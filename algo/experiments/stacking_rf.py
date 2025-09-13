import os
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, roc_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'final_model_ready_data.csv')
OUT_DIR = os.path.join(BASE_DIR, 'experiments', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# Utility: Top N non-leaky numeric features
# -------------------------
def get_top_features(df: pd.DataFrame, target_col='RISK_ASSESSMENT', id_col='ID', top_n=10):
    leakage_features = ['Credit_History_Length', 'Count_Late_Payments',
                        'Percentage_On_Time_Payments', 'Months_Since_Last_Delinquency']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in leakage_features + [target_col, id_col]]
    if not feature_cols:
        raise ValueError('No eligible numeric features after excluding leaky/id/target columns')
    corrs = df[feature_cols].corrwith(df[target_col]).abs()
    top_features = corrs.sort_values(ascending=False).head(top_n).index.tolist()
    print(f"Top {top_n} features (excluding leaky): {top_features}")
    return top_features


# -------------------------
# Utility: Youden's J threshold (our current logic)
# -------------------------
def youden_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, th = roc_curve(y_true, y_probs)
    youden = tpr - fpr
    idx = int(np.argmax(youden)) if len(youden) else 0
    thr = float(th[idx]) if len(th) else 0.5
    if not np.isfinite(thr):
        thr = 0.5
    return thr, float(youden[idx] if len(youden) else 0.0)


# -------------------------
# Stacking ensemble with Random Forest meta-learner
# -------------------------
def run_stacking_ensemble_rf(df: pd.DataFrame,
                             xgb_params: Dict,
                             cat_params: Dict,
                             lgbm_params: Dict,
                             target_col='RISK_ASSESSMENT',
                             id_col='ID',
                             test_size=0.2,
                             val_size=0.2,
                             random_state=42):

    # 1) Select top features (non-leaky numeric)
    top_features = get_top_features(df, target_col=target_col, id_col=id_col)
    X = df[top_features]
    y = df[target_col]

    # 2) Split train / validation / test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, stratify=y_train_full, random_state=random_state
    )

    # 3) Apply ADASYN on training set
    adasyn = ADASYN(random_state=random_state)
    X_train_bal, y_train_bal = adasyn.fit_resample(X_train, y_train)

    # 4) Train base models
    print('Training XGBoost...')
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=-1, **xgb_params)
    xgb_model.fit(X_train_bal, y_train_bal)

    print('Training CatBoost...')
    cat_model = CatBoostClassifier(silent=True, random_state=random_state, **cat_params)
    cat_model.fit(X_train_bal, y_train_bal)

    print('Training LightGBM...')
    lgb_model = lgb.LGBMClassifier(random_state=random_state, n_jobs=-1, **lgbm_params)
    lgb_model.fit(X_train_bal, y_train_bal)

    # 5) Generate meta-features on validation set
    X_val_meta = pd.DataFrame({
        'xgb': xgb_model.predict_proba(X_val)[:, 1],
        'cat': cat_model.predict_proba(X_val)[:, 1],
        'lgb': lgb_model.predict_proba(X_val)[:, 1]
    }, index=X_val.index)

    # 6) Train Random Forest meta-learner on validation meta-features
    meta_learner = RandomForestClassifier(n_estimators=200, random_state=random_state)
    meta_learner.fit(X_val_meta, y_val)

    # 7) Generate test meta-features
    X_test_meta = pd.DataFrame({
        'xgb': xgb_model.predict_proba(X_test)[:, 1],
        'cat': cat_model.predict_proba(X_test)[:, 1],
        'lgb': lgb_model.predict_proba(X_test)[:, 1]
    }, index=X_test.index)

    # 8) Predict probabilities on test set
    y_test_probs = meta_learner.predict_proba(X_test_meta)[:, 1]

    # 9) Calculate threshold by Youden's J (current logic)
    best_threshold, youden_j = youden_threshold(y_test.values, y_test_probs)
    print(f"Youden-J optimal threshold: {best_threshold:.3f} (J: {youden_j:.3f})")

    # 10) Final predictions using this threshold
    y_test_pred = (y_test_probs >= best_threshold).astype(int)

    # 11) Metrics
    ap = average_precision_score(y_test, y_test_probs)
    roc = roc_auc_score(y_test, y_test_probs)
    print(f"\nStacking RF Meta -> AP: {ap:.4f}, ROC AUC: {roc:.4f}")
    print("\nClassification Report (Youden threshold):")
    print(classification_report(y_test, y_test_pred, target_names=['Low Risk (0)', 'High Risk (1)']))

    # 12) Save result JSON
    out = {
        'approach': 'stacking_rf',
        'threshold': float(best_threshold),
        'metrics': {'ROC_AUC': float(roc), 'AP': float(ap)},
    }
    out_path = os.path.join(OUT_DIR, 'stacking_rf_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")

    return out


if __name__ == '__main__':
    df_fmod = pd.read_csv(DATA_PATH)

    xgb_params = {
        'colsample_bytree': 0.9464704583099741,
        'gamma': 0.3005575058716044,
        'learning_rate': 0.08080725777960454,
        'max_depth': 8,
        'n_estimators': 408,
        'reg_alpha': 0.9699098521619943,
        'reg_lambda': 4.329770563201687,
        'subsample': 0.6849356442713105
    }

    cat_params = {
        'colsample_bylevel': 0.694,
        'depth': 9,
        'iterations': 458,
        'l2_leaf_reg': 1.404,
        'learning_rate': 0.081,
        'subsample': 0.644
    }

    lgbm_params = {
        'colsample_bytree': 0.9468,
        'learning_rate': 0.1013,
        'max_depth': 10,
        'n_estimators': 490,
        'num_leaves': 44,
        'reg_alpha': 0.6279,
        'reg_lambda': 0.1943,
        'subsample': 0.6284
    }

    run_stacking_ensemble_rf(df_fmod, xgb_params, cat_params, lgbm_params)
