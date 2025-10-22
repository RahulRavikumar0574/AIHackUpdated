import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from lightgbm import LGBMClassifier

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # algo/
STATIC_DIR = os.path.join(BASE_DIR, 'static')
DATA_PATH = os.path.join(BASE_DIR, 'final_model_ready_data.csv')
ARTIFACTS_PATH = os.path.join(BASE_DIR, 'artifacts.json')

os.makedirs(STATIC_DIR, exist_ok=True)


def _detect_target(df: pd.DataFrame) -> str:
    if 'target' in df.columns:
        return 'target'
    if 'RISK_ASSESSMENT' in df.columns:
        return 'RISK_ASSESSMENT'
    raise ValueError("No target column found. Expected 'target' or 'RISK_ASSESSMENT'.")


def _prepare_features(df: pd.DataFrame, target_col: str):
    data = df.copy()

    # Feature engineering similar to algo/main.py
    if 'AGE_YEARS' not in data.columns:
        if 'DAYS_BIRTH' in data.columns:
            data['AGE_YEARS'] = (data['DAYS_BIRTH'].abs() / 365.0)
        elif 'AGE' in data.columns:
            data['AGE_YEARS'] = data['AGE']

    if 'EMPLOYMENT_YEARS' not in data.columns:
        if 'DAYS_EMPLOYED' in data.columns:
            data['EMPLOYMENT_YEARS'] = (data['DAYS_EMPLOYED'].abs() / 365.0)
        elif 'YEARS_EMPLOYED' in data.columns:
            data['EMPLOYMENT_YEARS'] = data['YEARS_EMPLOYED']
        else:
            data['EMPLOYMENT_YEARS'] = 0.0

    if 'AMT_INCOME_TOTAL' in data.columns and 'CNT_FAM_MEMBERS' in data.columns:
        denom = data['CNT_FAM_MEMBERS'].replace(0, np.nan)
        data['INCOME_PER_FAMILY'] = data['AMT_INCOME_TOTAL'] / denom

    if 'EMPLOYMENT_YEARS' in data.columns and 'AGE_YEARS' in data.columns:
        denom = data['AGE_YEARS'].replace(0, np.nan)
        data['EMPLOYMENT_RATIO'] = data['EMPLOYMENT_YEARS'] / denom

    if 'AMT_INCOME_TOTAL' in data.columns and 'EMPLOYMENT_YEARS' in data.columns:
        data['INCOME_PER_EMPLOYMENT_YEAR'] = data['AMT_INCOME_TOTAL'] / (data['EMPLOYMENT_YEARS'] + 1)

    feature_base_cols = [c for c in data.columns if c != target_col]
    features_df = data[feature_base_cols].copy()
    if 'ID' in features_df.columns:
        features_df = features_df.drop(columns=['ID'])

    categorical_cols = features_df.select_dtypes(include='object').columns
    X = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)

    bool_cols = X.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    y = data[target_col]
    try:
        y = y.astype(int)
    except Exception:
        y = y.map({True: 1, False: 0, '1': 1, '0': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0}).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    # Drop zero-variance columns
    constant_cols = X.columns[X.nunique() <= 1]
    if len(constant_cols) > 0:
        X = X.drop(columns=constant_cols)

    return X, y


def generate_model_charts():
    if not os.path.exists(DATA_PATH):
        print(f"[report_exporter] Data not found: {DATA_PATH}")
        return
    df = pd.read_csv(DATA_PATH)
    target_col = _detect_target(df)

    X, y = _prepare_features(df, target_col)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Threshold from artifacts if available
    threshold = 0.5
    if os.path.exists(ARTIFACTS_PATH):
        try:
            with open(ARTIFACTS_PATH, 'r', encoding='utf-8') as f:
                artifacts = json.load(f)
                threshold = float(artifacts.get('decision_threshold', threshold))
        except Exception:
            pass

    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    ap = average_precision_score(y_val, y_prob)

    plt.figure(figsize=(12, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}', color='#2563eb')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Validation)')
    plt.legend(loc='lower right')

    # PR Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'AP = {ap:.3f}', color='#10b981')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Validation)')
    plt.legend(loc='lower left')

    out_path = os.path.join(STATIC_DIR, 'credit_analysis_model.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"[report_exporter] Saved model chart -> {out_path}")


def generate_demographics_charts():
    if not os.path.exists(DATA_PATH):
        print(f"[report_exporter] Data not found: {DATA_PATH}")
        return
    df = pd.read_csv(DATA_PATH)

    # Derive easy-to-read fields if available
    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = (df['DAYS_BIRTH'].abs() / 365.0)
    elif 'AGE' in df.columns and 'AGE_YEARS' not in df.columns:
        df['AGE_YEARS'] = df['AGE']

    if 'AMT_INCOME_TOTAL' in df.columns:
        df['INCOME'] = df['AMT_INCOME_TOTAL']

    if 'CNT_FAM_MEMBERS' in df.columns:
        df['FAMILY_MEMBERS'] = df['CNT_FAM_MEMBERS']

    plt.figure(figsize=(12, 5))
    # Age distribution
    plt.subplot(1, 2, 1)
    if 'AGE_YEARS' in df.columns:
        sns.histplot(df['AGE_YEARS'].dropna(), kde=True, bins=30, color='#6366f1')
        plt.title('Age Distribution (Years)')
        plt.xlabel('Age (years)')
    else:
        plt.text(0.5, 0.5, 'AGE not available', ha='center')

    # Income distribution
    plt.subplot(1, 2, 2)
    if 'INCOME' in df.columns:
        sns.histplot(df['INCOME'].dropna(), kde=True, bins=30, color='#f59e0b')
        plt.title('Income Distribution')
        plt.xlabel('Annual Income')
    else:
        plt.text(0.5, 0.5, 'INCOME not available', ha='center')

    out_path = os.path.join(STATIC_DIR, 'credit_analysis_demographics.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"[report_exporter] Saved demographics chart -> {out_path}")


if __name__ == '__main__':
    generate_model_charts()
    generate_demographics_charts()
