import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

# -------------------------
# Utility: safe drop list
# -------------------------
def _safe_drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Drop only the columns that actually exist (avoid KeyError)."""
    cols_present = [c for c in cols if c in df.columns]
    if cols_present:
        return df.drop(columns=cols_present)
    return df.copy()

# -------------------------
# 1) Leakage check (handles ID)
# -------------------------
def check_for_leakage(df: pd.DataFrame, target_col: str = 'RISK_ASSESSMENT', id_col: str = 'ID', threshold: float = 0.8):
    """
    Compute absolute Pearson correlations between numeric features and the target.
    Excludes the ID column from the analysis.
    """
    print("--- Checking for Feature-Target Leakage ---")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    
    df_check = df.copy()
    if id_col in df_check.columns:
        df_check = df_check.drop(columns=[id_col])
    
    numeric_cols = df_check.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        raise ValueError(f"Target column '{target_col}' must be numeric (0/1).")
    
    feature_cols = [c for c in numeric_cols if c != target_col]
    if not feature_cols:
        print("[INFO] No numeric features to check.")
        return
    
    corrs = df_check[feature_cols].corrwith(df_check[target_col]).abs().sort_values(ascending=False)
    print("Top 10 most correlated features with the target (abs Pearson):")
    print(corrs.head(10))
    
    high_corr = corrs[corrs > threshold]
    if not high_corr.empty:
        print(f"\n[WARNING] Features with correlation > {threshold}:")
        print(high_corr)
    else:
        print(f"\n[INFO] No features found with correlation > {threshold}.")
    print("-" * 40)

# -------------------------
# 2) Cross-validation pipeline (drops leakage + ID)
# -------------------------
def run_cross_validation_pipeline(df: pd.DataFrame, target_col: str = 'RISK_ASSESSMENT', id_col: str = 'ID',
                                  n_splits: int = 5, n_repeats: int = 2, random_state: int = 42):
    """
    Run repeated stratified CV with ADASYN inside each fold.
    Removes leakage features and ID if present.
    """
    print("\n--- Cross-Validation (ImbPipeline + ADASYN) ---")
    leakage_features = [
        'Credit_History_Length', 'Count_Late_Payments',
        'Percentage_On_Time_Payments', 'Months_Since_Last_Delinquency'
    ]
    
    # Build drop list, ensuring we only drop existing columns
    drop_candidates = [c for c in leakage_features if c in df.columns]
    if id_col in df.columns:
        drop_candidates.append(id_col)
    if target_col in drop_candidates:
        drop_candidates.remove(target_col)
    
    X = _safe_drop_columns(df, drop_candidates + [target_col])
    y = df[target_col]
    
    print(f"Running CV with {X.shape[1]} features (leakage features removed if present).")
    
    lgbm = lgb.LGBMClassifier(random_state=random_state, verbosity=-1, n_jobs=-1)
    adasyn = ADASYN(random_state=random_state)
    pipeline = ImbPipeline([('resample', adasyn), ('clf', lgbm)])
    
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='average_precision', n_jobs=-1)
    print(f"Cross-Validation AP scores ({len(scores)} folds): {[f'{s:.3f}' for s in scores]}")
    print(f"Mean AP: {scores.mean():.4f}  Std: {scores.std():.4f}")
    print("-" * 40)
    return scores

# -------------------------
# 3) Training experiments + safe calibration (returns IDs with predictions)
# -------------------------
def run_training_experiments(df: pd.DataFrame, target_col: str = 'RISK_ASSESSMENT', id_col: str = 'ID',
                             test_size: float = 0.2, random_state: int = 42, calibration_method: str = 'sigmoid'):
    """
    Compare ADASYN vs scale_pos_weight, then calibrate best model using a held-out calibration split.
    Returns final_report with calibrated metrics and a test_predictions DataFrame containing IDs.
    """
    print("\n--- Running Training Experiments (ID preserved) ---")
    leakage_features = [
        'Credit_History_Length', 'Count_Late_Payments',
        'Percentage_On_Time_Payments', 'Months_Since_Last_Delinquency'
    ]
    drop_candidates = [c for c in leakage_features if c in df.columns]
    if id_col in df.columns:
        ids_series = df[id_col].copy()
        drop_candidates.append(id_col)
    else:
        ids_series = None

    X = _safe_drop_columns(df, drop_candidates + [target_col])
    y = df[target_col]

    print(f"Training with {X.shape[1]} features (leakage dropped if present).")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index if ids_series is None else ids_series, test_size=test_size, random_state=random_state, stratify=y
    )

    # If ids_series was provided, idx_test contains the IDs; else idx_test contains indices
    test_ids = pd.Series(idx_test, index=X_test.index, name=id_col) if ids_series is not None else pd.Series(X_test.index, index=X_test.index, name='index')

    # 1) ADASYN -> LGBM
    print("\n[Experiment 1] ADASYN resample -> LGBM")
    adasyn = ADASYN(random_state=random_state)
    X_train_bal, y_train_bal = adasyn.fit_resample(X_train, y_train)
    
    model_adasyn = lgb.LGBMClassifier(random_state=random_state, n_jobs=-1, verbosity=-1,
                                      n_estimators=1000, learning_rate=0.05)
    model_adasyn.fit(X_train_bal, y_train_bal,
                     eval_set=[(X_test, y_test)],
                     eval_metric='auc',
                     callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    y_proba_adasyn = model_adasyn.predict_proba(X_test)[:, 1]
    ap_adasyn = average_precision_score(y_test, y_proba_adasyn)
    roc_adasyn = roc_auc_score(y_test, y_proba_adasyn)
    print(f"ADASYN -> AP: {ap_adasyn:.4f}, ROC AUC: {roc_adasyn:.4f}")

    # 2) scale_pos_weight -> LGBM
    print("\n[Experiment 2] scale_pos_weight -> LGBM")
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / (n_pos + 1e-9)
    print(f"Computed scale_pos_weight = {scale:.2f}")
    
    model_spw = lgb.LGBMClassifier(random_state=random_state, n_jobs=-1, verbosity=-1,
                                   n_estimators=1000, learning_rate=0.05,
                                   scale_pos_weight=scale)
    model_spw.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    y_proba_spw = model_spw.predict_proba(X_test)[:, 1]
    ap_spw = average_precision_score(y_test, y_proba_spw)
    roc_spw = roc_auc_score(y_test, y_proba_spw)
    print(f"scale_pos_weight -> AP: {ap_spw:.4f}, ROC AUC: {roc_spw:.4f}")

    # Choose best
    if ap_adasyn >= ap_spw:
        best_model = model_adasyn
        best_name = "ADASYN"
        best_uncal_proba = y_proba_adasyn
    else:
        best_model = model_spw
        best_name = "scale_pos_weight"
        best_uncal_proba = y_proba_spw

    print(f"\nSelected best model: {best_name} (by AP). Proceeding to safe calibration using held-out calibration split.")

    # Create a small calibration holdout from the training data (20% of training)
    X_tr_sub, X_calib, y_tr_sub, y_calib = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)

    # Refit best_model on the training SUBSET (respecting chosen strategy)
    if best_name == "ADASYN":
        X_tr_sub_bal, y_tr_sub_bal = adasyn.fit_resample(X_tr_sub, y_tr_sub)
        best_model.fit(X_tr_sub_bal, y_tr_sub_bal,
                       eval_set=[(X_calib, y_calib)],
                       eval_metric='auc',
                       callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    else:
        # scale_pos_weight model
        best_model.fit(X_tr_sub, y_tr_sub,
                       eval_set=[(X_calib, y_calib)],
                       eval_metric='auc',
                       callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

    # FIX: Changed 'base_estimator' to 'estimator' for compatibility with newer scikit-learn versions
    calibrator = CalibratedClassifierCV(estimator=best_model, method=calibration_method, cv='prefit')
    calibrator.fit(X_calib, y_calib)  # calibrator uses predictions from prefit best_model on X_calib

    # Preds on final test set
    y_proba_uncal = best_model.predict_proba(X_test)[:, 1]
    y_proba_cal = calibrator.predict_proba(X_test)[:, 1]
    ap_cal = average_precision_score(y_test, y_proba_cal)
    roc_cal = roc_auc_score(y_test, y_proba_cal)
    print(f"Calibrated model -> AP: {ap_cal:.4f}, ROC AUC: {roc_cal:.4f}")

    # Build test_predictions DataFrame with IDs (or index if ID missing)
    test_predictions = pd.DataFrame({
        id_col if ids_series is not None else 'index': test_ids,
        'y_true': y_test,
        'proba_uncalibrated': y_proba_uncal,
        'proba_calibrated': y_proba_cal
    }, index=X_test.index)

    test_predictions['y_pred_calibrated'] = (test_predictions['proba_calibrated'] >= 0.3).astype(int)

    # Calibration plot
    fig, ax = plt.subplots(figsize=(7, 7))
    try:
        CalibrationDisplay.from_predictions(y_test, y_proba_uncal, n_bins=10, ax=ax, name=f"Uncalibrated ({best_name})")
    except Exception:
        pass
    CalibrationDisplay.from_predictions(y_test, y_proba_cal, n_bins=10, ax=ax, name=f"Calibrated ({calibration_method})")
    plt.title("Calibration (Reliability) Diagram")
    plt.legend()
    plt.tight_layout()
    plt.show()

    final_report = {
        "best_model_name": best_name,
        "best_model_uncalibrated": best_model,
        "final_calibrator": calibrator,
        "metrics": {
            "AP_calibrated": ap_cal,
            "ROC_AUC_calibrated": roc_cal
        },
        "test_predictions": test_predictions
    }
    return final_report


# --- Main Execution Block ---

if __name__ == '__main__':
    try:
        # --- Load data from CSV ---
        data_path = os.path.join(os.path.dirname(__file__), 'final_model_ready_data.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV file not found at: {data_path}")
        df_fmod = pd.read_csv(data_path)

        # Detect target column
        if 'target' in df_fmod.columns:
            target_col = 'target'
        elif 'RISK_ASSESSMENT' in df_fmod.columns:
            target_col = 'RISK_ASSESSMENT'
        else:
            raise ValueError("No target column found. Expected 'target' or 'RISK_ASSESSMENT'.")

        # Ensure target is numeric 0/1
        try:
            df_fmod[target_col] = df_fmod[target_col].astype(int)
        except Exception:
            df_fmod[target_col] = df_fmod[target_col].map({True:1, False:0, '1':1, '0':0, 'Y':1, 'N':0, 'yes':1, 'no':0}).astype(int)
        
        # --- Experiment 1: Check for obvious data leakage ---
        check_for_leakage(df=df_fmod, target_col=target_col)

        # --- Experiment 2: Get a robust performance estimate with CV ---
        run_cross_validation_pipeline(df=df_fmod, target_col=target_col)

        # --- Experiment 3: Compare imbalance techniques and calibrate the best model ---
        experiment_results = run_training_experiments(df=df_fmod, target_col=target_col)
        
        print("\n--- Final Experiment Summary ---")
        print(f"The best performing model type was: {experiment_results['best_model_name']}")
        print("Final Calibrated Metrics:")
        for metric, value in experiment_results['metrics'].items():
            print(f"  - {metric}: {value:.4f}")
        
        # The final, ready-to-use model is now in this variable
        final_model = experiment_results['final_calibrator']
        print(f"\nThe final calibrated model is ready for use: {type(final_model)}")
        
        # You can now analyze the test predictions
        test_preds_df = experiment_results['test_predictions']
        print("\nTest predictions with IDs and probabilities are available in 'test_preds_df'.")
        print(test_preds_df.head())

        # ADDITION: Print the full classification report for the final calibrated model
        print("\nFinal Classification Report (at 0.5 threshold):")
        y_true = test_preds_df['y_true']
        y_pred = test_preds_df['y_pred_calibrated']
        print(classification_report(y_true, y_pred, target_names=['Low Risk (0)', 'High Risk (1)']))

        # Persist lightweight artifacts for FastAPI consumption (e.g., decision threshold)
        # We used a calibrated probability and a 0.3 threshold for y_pred_calibrated above.
        import json
        artifacts = {
            'best_model_name': experiment_results['best_model_name'],
            'decision_threshold': 0.3,
            'metrics': experiment_results['metrics']
        }
        with open('artifacts.json', 'w', encoding='utf-8') as f:
            json.dump(artifacts, f, indent=2)
        print("\nSaved artifacts.json with decision_threshold and summary metrics for the API.")


    except NameError:
        print("\nERROR: The DataFrame 'df_fmod' was not found.")
        print("Please ensure your final merged DataFrame is loaded and named 'df_fmod' before running.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")