import os
import json
import warnings
import sys
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional models
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

# Torch models
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Local models: ensure parent (algo/) is on sys.path so we can import dl_models when run as a script
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from dl_models import TabMLP, TabTransformer, LSTMClassifier, SimpleScaler

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # algo/
DATA_PATH = os.path.join(BASE_DIR, 'final_model_ready_data.csv')
OUT_DIR = os.path.join(BASE_DIR, 'experiments', 'outputs')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

warnings.filterwarnings('ignore')

# Known leakage features (align with credit_pipeline.py)
LEAKAGE_FEATURES = [
    'Credit_History_Length',
    'Count_Late_Payments',
    'Percentage_On_Time_Payments',
    'Months_Since_Last_Delinquency',
]


def _detect_target(df: pd.DataFrame) -> str:
    if 'target' in df.columns:
        return 'target'
    if 'RISK_ASSESSMENT' in df.columns:
        return 'RISK_ASSESSMENT'
    raise ValueError("No target column found. Expected 'target' or 'RISK_ASSESSMENT'.")


def _prepare_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.copy()

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

    constant_cols = X.columns[X.nunique() <= 1]
    if len(constant_cols) > 0:
        X = X.drop(columns=constant_cols)

    return X, y


def youden_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    fpr, tpr, th = roc_curve(y_true, y_probs)
    if len(th) == 0:
        return 0.5
    j = tpr - fpr
    idx = int(np.argmax(j))
    thr = float(th[idx])
    if not np.isfinite(thr):
        thr = 0.5
    return thr


def _metric_row(name: str, y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    return {
        'model': name,
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'ap': float(average_precision_score(y_true, y_prob)),
        'threshold': float(thr),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_classical_models(X_train, X_test, y_train, y_test) -> List[Dict[str, float]]:
    results = []

    # Logistic Regression
    try:
        lr = LogisticRegression(max_iter=500, class_weight='balanced')
        lr.fit(X_train, y_train)
        prob = lr.predict_proba(X_test)[:, 1]
        thr = youden_threshold(y_test, prob)
        results.append(_metric_row('LogisticRegression', y_test, prob, thr))
    except Exception:
        pass

    # RandomForest
    try:
        rf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced_subsample', n_jobs=-1)
        rf.fit(X_train, y_train)
        prob = rf.predict_proba(X_test)[:, 1]
        thr = youden_threshold(y_test, prob)
        results.append(_metric_row('RandomForest', y_test, prob, thr))
    except Exception:
        pass

    # XGBoost
    if xgb is not None:
        try:
            xgbc = xgb.XGBClassifier(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                tree_method='hist',
                n_jobs=-1,
                random_state=42,
            )
            xgbc.fit(X_train, y_train)
            prob = xgbc.predict_proba(X_test)[:, 1]
            thr = youden_threshold(y_test, prob)
            results.append(_metric_row('XGBoost', y_test, prob, thr))
        except Exception:
            pass

    # LightGBM
    if lgb is not None:
        try:
            lgbm = lgb.LGBMClassifier(
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            )
            lgbm.fit(X_train, y_train)
            prob = lgbm.predict_proba(X_test)[:, 1]
            thr = youden_threshold(y_test, prob)
            results.append(_metric_row('LightGBM', y_test, prob, thr))
        except Exception:
            pass

    # CatBoost
    if CatBoostClassifier is not None:
        try:
            cat = CatBoostClassifier(
                iterations=800,
                learning_rate=0.05,
                depth=6,
                loss_function='Logloss',
                eval_metric='AUC',
                verbose=False,
                random_state=42,
            )
            cat.fit(X_train, y_train)
            prob = cat.predict_proba(X_test)[:, 1]
            thr = youden_threshold(y_test, prob)
            results.append(_metric_row('CatBoost', y_test, prob, thr))
        except Exception:
            pass

    return results


class TorchModelWrapper:
    def __init__(self, model_cls, input_dim: int, epochs: int = 10, batch_size: int = 512, lr: float = 1e-3):
        self.model_cls = model_cls
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None

    def fit(self, X, y):
        self.scaler = SimpleScaler()
        X_scaled = self.scaler.fit_transform(X.values if hasattr(X, 'values') else X)
        X_scaled = np.asarray(X_scaled, dtype=np.float32)
        y_array = np.asarray(y.values if hasattr(y, 'values') else y, dtype=np.float32)

        if self.model_cls is TabTransformer:
            self.model = self.model_cls(in_dim=self.input_dim)
        elif self.model_cls is LSTMClassifier:
            self.model = self.model_cls(input_dim=self.input_dim)
        else:
            self.model = self.model_cls(in_dim=self.input_dim) if self.model_cls is TabMLP else self.model_cls(self.input_dim)
        self.model = self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        dataset = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_array, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X.values if hasattr(X, 'values') else X)
        X_scaled = np.asarray(X_scaled, dtype=np.float32)
        with torch.no_grad():
            xt = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            out = self.model(xt)
            prob = torch.sigmoid(out).cpu().numpy()
        return np.column_stack([1 - prob, prob])


def run_deep_learning_models(X_train, X_test, y_train, y_test) -> List[Dict[str, float]]:
    results = []
    input_dim = X_train.shape[1]

    dl_configs = [
        ('TabMLP', TorchModelWrapper(TabMLP, input_dim, epochs=10, batch_size=512, lr=1e-3)),
        ('LSTM', TorchModelWrapper(LSTMClassifier, input_dim, epochs=10, batch_size=512, lr=1e-3)),
        ('Transformer', TorchModelWrapper(TabTransformer, input_dim, epochs=10, batch_size=512, lr=1e-3)),
    ]

    for name, model in dl_configs:
        try:
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            thr = youden_threshold(y_test, y_proba)
            results.append(_metric_row(name, y_test, y_proba, thr))
        except Exception:
            continue

    return results


def save_tables(name: str, rows: List[Dict[str, float]]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(by=['roc_auc', 'ap'], ascending=False)
    csv_path = os.path.join(OUT_DIR, f'{name}_comparison.csv')
    md_path = os.path.join(OUT_DIR, f'{name}_comparison.md')
    # Write CSV with Windows-friendly fallback if file is locked
    try:
        df_sorted.to_csv(csv_path, index=False)
    except PermissionError:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(OUT_DIR, f'{name}_comparison_{ts}.csv')
        df_sorted.to_csv(csv_path, index=False)
    try:
        md_content = df_sorted.to_markdown(index=False)
    except Exception:
        # Fallback simple pipe table if tabulate is not installed
        headers = list(df_sorted.columns)
        rows = [headers] + df_sorted.astype(str).values.tolist()
        widths = [max(len(str(x)) for x in col) for col in zip(*rows)]
        lines = []
        # Header
        header_line = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
        sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
        lines.append(header_line)
        lines.append(sep_line)
        # Rows
        for r in df_sorted.values:
            lines.append("| " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths)) + " |")
        md_content = "\n".join(lines)
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    except PermissionError:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_path = os.path.join(OUT_DIR, f'{name}_comparison_{ts}.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    print(f"Saved: {csv_path}\nSaved: {md_path}")


def run_pca_and_heatmaps(X_train: pd.DataFrame, name_prefix: str = 'all_features'):
    num_X = X_train.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num_X)

    pca = PCA(n_components=min(20, X_scaled.shape[1]))
    _ = pca.fit_transform(X_scaled)

    exp_var = pca.explained_variance_ratio_
    cum_exp_var = np.cumsum(exp_var)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(exp_var) + 1), exp_var, marker='o', label='Explained variance')
    plt.plot(range(1, len(cum_exp_var) + 1), cum_exp_var, marker='s', label='Cumulative')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend()
    pca_var_path = os.path.join(STATIC_DIR, f'{name_prefix}_pca_variance.png')
    plt.tight_layout()
    plt.savefig(pca_var_path, dpi=140)
    plt.close()

    corr = pd.DataFrame(X_scaled, columns=num_X.columns).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    corr_path = os.path.join(STATIC_DIR, f'{name_prefix}_corr_heatmap.png')
    plt.tight_layout()
    plt.savefig(corr_path, dpi=140)
    plt.close()

    loadings = pd.DataFrame(
        pca.components_.T,
        index=num_X.columns,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])]
    )
    plt.figure(figsize=(12, max(6, 0.25 * loadings.shape[0])))
    sns.heatmap(loadings.iloc[:, :10], cmap='vlag', center=0)
    plt.title('PCA Loadings (Top 10 PCs)')
    load_path = os.path.join(STATIC_DIR, f'{name_prefix}_pca_loadings_heatmap.png')
    plt.tight_layout()
    plt.savefig(load_path, dpi=140)
    plt.close()

    metrics = {
        'explained_variance_ratio': exp_var.tolist(),
        'cumulative_explained_variance_ratio': cum_exp_var.tolist(),
        'n_components': int(pca.n_components_),
    }
    with open(os.path.join(OUT_DIR, f'{name_prefix}_pca_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved PCA plots -> {pca_var_path}, {load_path}, {corr_path}")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    target_col = _detect_target(df)
    X, y = _prepare_features(df, target_col)

    # Drop known leakage features if present (and any lingering ID-like cols just in case)
    drop_candidates = [c for c in LEAKAGE_FEATURES if c in X.columns]
    for id_like in ['ID', 'Id', 'id']:
        if id_like in X.columns and id_like not in drop_candidates:
            drop_candidates.append(id_like)
    if drop_candidates:
        X = X.drop(columns=drop_candidates)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train-only correlation guard: drop features with |corr| >= 0.98 to target
    train_numeric = X_train.select_dtypes(include=[np.number])
    common_cols = train_numeric.columns
    if len(common_cols) > 0:
        corrs = train_numeric[common_cols].corrwith(y_train).abs()
        high_corr_feats = corrs[corrs >= 0.98].index.tolist()
        if len(high_corr_feats) > 0:
            X_train = X_train.drop(columns=high_corr_feats, errors='ignore')
            X_test = X_test.drop(columns=high_corr_feats, errors='ignore')

    classical_results = run_classical_models(X_train, X_test, y_train, y_test)
    save_tables('classical', classical_results)

    dl_results = run_deep_learning_models(X_train, X_test, y_train, y_test)
    save_tables('deep_learning', dl_results)

    combined = sorted(classical_results + dl_results, key=lambda r: (r['roc_auc'], r['ap']), reverse=True)
    save_tables('combined', combined)

    run_pca_and_heatmaps(X_train, name_prefix='features')


if __name__ == '__main__':
    main()
