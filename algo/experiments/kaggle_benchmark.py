import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import sys
    # Add parent directory to path to import dl_models
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from dl_models import TabTransformer, LSTMClassifier, SimpleScaler
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    TabTransformer = None
    LSTMClassifier = None

try:
    import shap
except Exception:
    shap = None

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    SMOTE = None
    ImbPipeline = None

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DEFAULT = os.path.join(BASE_DIR, 'creditcard.csv')  # Expect Kaggle dataset if provided
OUT_DIR = os.path.join(BASE_DIR, 'experiments', 'kaggle_outputs')
os.makedirs(OUT_DIR, exist_ok=True)

warnings.filterwarnings('ignore')


@dataclass
class ModelSpec:
    name: str
    model: object
    is_tree: bool = False


# =============================
# PyTorch Model Wrapper for sklearn compatibility
# =============================
class TorchModelWrapper:
    """Wrapper to make PyTorch models compatible with sklearn interface."""
    def __init__(self, model_class, input_dim, epochs=20, batch_size=256, lr=0.001, device='cpu'):
        self.model_class = model_class
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = None
        self.scaler = None
    
    def fit(self, X, y):
        """Fit the PyTorch model."""
        # Initialize scaler
        self.scaler = SimpleScaler()
        X_scaled = self.scaler.fit_transform(X.values if hasattr(X, 'values') else X)
        
        # Ensure proper type conversion to float32
        X_scaled = np.asarray(X_scaled, dtype=np.float32)
        y_array = np.asarray(y.values if hasattr(y, 'values') else y, dtype=np.float32)
        
        # Initialize model - use correct parameter name for each model
        if 'Transformer' in str(self.model_class):
            self.model = self.model_class(in_dim=self.input_dim)
        else:
            self.model = self.model_class(input_dim=self.input_dim)
        self.model = self.model.to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        
        # Create dataset
        dataset = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_array, dtype=torch.float32)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not fitted yet")
        
        self.model.eval()
        X_scaled = self.scaler.transform(X.values if hasattr(X, 'values') else X)
        
        # Ensure proper type conversion to float32
        X_scaled = np.asarray(X_scaled, dtype=np.float32)
        
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
        
        # Return in sklearn format: [prob_class_0, prob_class_1]
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


# =============================
# Preprocessing per baseline appendix (arXiv:2412.07437)
# =============================

def load_kaggle_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Kaggle dataset not found at: {path}. Place 'creditcard.csv' there or pass --data PATH.")
    df = pd.read_csv(path)
    if 'Class' not in df.columns:
        raise ValueError("Expected Kaggle fraud dataset with 'Class' target column.")
    return df


def preprocess_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    # Scale Amount and Time to [-1, 1] (as shown in appendix)
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    data['Amount_Scaled'] = scaler_amount.fit_transform(data['Amount'].values.reshape(-1, 1)).reshape(-1)
    data['Time_Scaled'] = scaler_time.fit_transform(data['Time'].values.reshape(-1, 1)).reshape(-1)

    # Create Hour feature from Time
    data['Hour'] = (data['Time'] // 3600) % 24

    # Create Day_Segment feature and one-hot
    def assign_day_segment(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 24:
            return 'Evening'
        else:
            return 'Night'

    data['Day_Segment'] = data['Hour'].apply(assign_day_segment)
    data = pd.get_dummies(data, columns=['Day_Segment'], drop_first=True)

    # Drop original Amount/Time to avoid leakage of raw scale
    X = data.drop(columns=['Class'])
    y = data['Class'].astype(int)

    # Optionally drop original Time/Amount keeping scaled ones + engineered
    drop_cols = []
    if 'Amount' in X.columns:
        drop_cols.append('Amount')
    if 'Time' in X.columns:
        drop_cols.append('Time')
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # Ensure no NaNs or infs
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    # Drop zero-variance
    nunique = X.nunique(dropna=False)
    zero_var = nunique[nunique <= 1].index.tolist()
    if zero_var:
        X = X.drop(columns=zero_var)

    return X, y


# =============================
# Threshold selection (Youden's J default if baseline not explicit)
# =============================

def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    if len(j) == 0 or len(thr) == 0:
        return 0.5
    idx = int(np.argmax(j))
    t = float(thr[idx])
    if not np.isfinite(t):
        t = 0.5
    return t


# =============================
# Cross-validation evaluation helper
# =============================

def evaluate_model_cv(X: pd.DataFrame, y: pd.Series, spec: ModelSpec, cv, use_smote: bool = False) -> Dict:
    metrics_list = []

    # Build pipeline
    if use_smote:
        if SMOTE is None or ImbPipeline is None:
            raise RuntimeError("imblearn not available for SMOTE. Install imbalanced-learn.")
        pipe = ImbPipeline([
            ('clf', spec.model)
        ])
        sampler = SMOTE(random_state=42)
    else:
        pipe = Pipeline([
            ('clf', spec.model)
        ])
        sampler = None

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        if use_smote and sampler is not None:
            X_res, y_res = sampler.fit_resample(X_tr, y_tr)
        else:
            X_res, y_res = X_tr, y_tr

        pipe.fit(X_res, y_res)
        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
            y_prob = pipe.named_steps['clf'].predict_proba(X_te)[:, 1]
        else:
            # fallback decision_function -> prob via sigmoid-ish scaling
            if hasattr(pipe.named_steps['clf'], 'decision_function'):
                s = pipe.named_steps['clf'].decision_function(X_te)
                # scale to 0-1
                y_prob = (s - s.min()) / (s.max() - s.min() + 1e-9)
            else:
                y_prob = pipe.named_steps['clf'].predict(X_te)

        # Threshold (use Youden by default)
        thr = youden_threshold(y_te.values, y_prob)
        y_pred = (y_prob >= thr).astype(int)

        metrics_list.append({
            'fold': fold,
            'threshold': thr,
            'roc_auc': roc_auc_score(y_te, y_prob),
            'ap': average_precision_score(y_te, y_prob),
            'accuracy': accuracy_score(y_te, y_pred),
            'precision': precision_score(y_te, y_pred, zero_division=0),
            'recall': recall_score(y_te, y_pred, zero_division=0),
            'f1': f1_score(y_te, y_pred, zero_division=0),
        })

    dfm = pd.DataFrame(metrics_list)
    summary = dfm.mean(numeric_only=True).to_dict()
    # Keep also std for context
    stds = dfm.std(numeric_only=True).add_suffix('_std').to_dict()
    summary.update(stds)
    summary['model'] = spec.name
    summary['smote'] = use_smote

    # Save per-fold metrics
    dfm.to_csv(os.path.join(OUT_DIR, f"{spec.name}_{'smote' if use_smote else 'nosmote'}_cv.csv"), index=False)

    return summary


# =============================
# SHAP explanation
# =============================

def compute_shap(spec: ModelSpec, X_sample: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series, use_smote: bool = False):
    if shap is None:
        print("[WARN] shap not installed; skipping SHAP plots for", spec.name)
        return

    # Fit a fresh model on X_train (optionally SMOTE) for SHAP explainer
    model = spec.model
    X_fit, y_fit = X_train, y_train
    if use_smote and SMOTE is not None:
        sm = SMOTE(random_state=42)
        X_fit, y_fit = sm.fit_resample(X_train, y_train)
    model.fit(X_fit, y_fit)

    try:
        if spec.is_tree and hasattr(shap, 'TreeExplainer'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            # shap_values can be list for multiclass; for binary pick index 1 if list
            if isinstance(shap_values, list) and len(shap_values) > 1:
                sv = shap_values[1]
            else:
                sv = shap_values
        else:
            # KernelExplainer as fallback (slower)
            background = shap.sample(X_fit, 200) if hasattr(shap, 'sample') else X_fit.sample(min(200, len(X_fit)), random_state=42)
            explainer = shap.KernelExplainer(model.predict_proba, background) if hasattr(model, 'predict_proba') else shap.KernelExplainer(model.predict, background)
            sv = explainer.shap_values(X_sample, nsamples=100)
            if isinstance(sv, list) and len(sv) > 1:
                sv = sv[1]
        # Summary bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_sample, plot_type='bar', show=False, max_display=20)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"shap_{spec.name}_{'smote' if use_smote else 'nosmote'}_bar.png")
        plt.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close()
        print("Saved SHAP bar plot ->", out_path)

        # Beeswarm
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_sample, show=False, max_display=20)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"shap_{spec.name}_{'smote' if use_smote else 'nosmote'}_beeswarm.png")
        plt.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close()
        print("Saved SHAP beeswarm plot ->", out_path)
    except Exception as e:
        print(f"[WARN] SHAP failed for {spec.name}: {e}")


# =============================
# Comparison plotting
# =============================

def plot_comparison(results: List[Dict]):
    df = pd.DataFrame(results)
    # Select key metrics
    metrics = ['roc_auc', 'accuracy', 'f1', 'recall', 'precision', 'ap']
    melted = df.melt(id_vars=['model', 'smote'], value_vars=metrics, var_name='metric', value_name='value')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='metric', y='value', hue='model')
    plt.title('Model comparison (SMOTE and No-SMOTE combined averages)')
    plt.ylim(0, 1.0)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'comparison_metrics.png')
    plt.savefig(out_path, dpi=140)
    plt.close()
    print('Saved comparison plot ->', out_path)


# =============================
# Main
# =============================

def main(data_path: str = DATA_DEFAULT, use_repeated_cv: bool = True, skip_shap: bool = False):
    # Load & preprocess
    df = load_kaggle_dataset(data_path)
    X, y = preprocess_df(df)

    # Holdout split for SHAP reference and potential threshold reference if needed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cross-validation config per request (include cross-fold)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42) if use_repeated_cv else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model list (without SMOTE first, then with SMOTE)
    specs: List[ModelSpec] = []
    specs.append(ModelSpec('LogReg', LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=None)))
    specs.append(ModelSpec('RandomForest', RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced_subsample'), is_tree=True))
    if xgb is not None:
        specs.append(ModelSpec('XGBoost', xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, reg_alpha=0.0, n_jobs=-1, random_state=42, eval_metric='auc'
        ), is_tree=True))
    if lgb is not None:
        specs.append(ModelSpec('LightGBM', lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, verbosity=-1
        ), is_tree=True))
    
    # Add deep learning models if available
    if TORCH_AVAILABLE:
        # Note: These will be wrapped in a sklearn-compatible wrapper (10 epochs for faster training)
        specs.append(ModelSpec('Transformer', TorchModelWrapper(TabTransformer, X.shape[1], epochs=10), is_tree=False))
        specs.append(ModelSpec('LSTM', TorchModelWrapper(LSTMClassifier, X.shape[1], epochs=10), is_tree=False))

    results: List[Dict] = []

    # Evaluate without SMOTE
    for spec in specs:
        print(f"\n=== Evaluating {spec.name} (NO SMOTE) ===")
        res = evaluate_model_cv(X, y, spec, cv, use_smote=False)
        results.append(res)
        # SHAP on a sample of train set (skip if requested)
        if not skip_shap:
            X_sample = X_train.sample(min(2000, len(X_train)), random_state=42)
            compute_shap(spec, X_sample, X_train, y_train, use_smote=False)

    # Evaluate with SMOTE (optional per user instruction to compare)
    if SMOTE is not None and ImbPipeline is not None:
        for spec in specs:
            print(f"\n=== Evaluating {spec.name} (WITH SMOTE) ===")
            res = evaluate_model_cv(X, y, spec, cv, use_smote=True)
            results.append(res)
            if not skip_shap:
                X_sample = X_train.sample(min(2000, len(X_train)), random_state=42)
                compute_shap(spec, X_sample, X_train, y_train, use_smote=True)
    else:
        print("[INFO] imbalanced-learn not available; skipping SMOTE runs.")

    # Save results and plot
    res_df = pd.DataFrame(results)
    res_path = os.path.join(OUT_DIR, 'kaggle_benchmark_results.csv')
    res_df.to_csv(res_path, index=False)
    print('Saved results ->', res_path)

    plot_comparison(results)

    # Also save a JSON summary
    with open(os.path.join(OUT_DIR, 'kaggle_benchmark_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Display best model selection
    print("\n" + "="*80)
    print("BEST MODEL SELECTION SUMMARY")
    print("="*80)
    
    # Find best models by different metrics
    metrics_to_check = ['roc_auc', 'ap', 'accuracy', 'precision', 'recall', 'f1']
    for metric in metrics_to_check:
        best_result = max(results, key=lambda x: x.get(metric, 0))
        smote_str = "WITH SMOTE" if best_result['smote'] else "NO SMOTE"
        print(f"\nBest by {metric.upper()}: {best_result['model']} ({smote_str})")
        print(f"  - {metric}: {best_result[metric]:.4f}")
        print(f"  - ROC-AUC: {best_result['roc_auc']:.4f}")
        print(f"  - AP: {best_result['ap']:.4f}")
        print(f"  - Accuracy: {best_result['accuracy']:.4f}")
        print(f"  - Precision: {best_result['precision']:.4f}")
        print(f"  - Recall: {best_result['recall']:.4f}")
        print(f"  - F1: {best_result['f1']:.4f}")
    
    # Overall best model (by ROC-AUC as primary metric)
    print("\n" + "="*80)
    print("OVERALL BEST MODEL (by ROC-AUC):")
    best_overall = max(results, key=lambda x: x.get('roc_auc', 0))
    smote_str = "WITH SMOTE" if best_overall['smote'] else "NO SMOTE"
    print(f"{best_overall['model']} ({smote_str})")
    print(f"  - ROC-AUC: {best_overall['roc_auc']:.4f}")
    print(f"  - AP: {best_overall['ap']:.4f}")
    print(f"  - Accuracy: {best_overall['accuracy']:.4f}")
    print(f"  - F1: {best_overall['f1']:.4f}")
    print("="*80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Kaggle Fraud Benchmark (2024–2025 baseline + our pipeline)')
    parser.add_argument('--data', type=str, default=DATA_DEFAULT, help='Path to Kaggle creditcard.csv')
    parser.add_argument('--no-repeated', action='store_true', help='Use plain 5-fold instead of 5x2 repeated CV')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP analysis for faster results')
    args = parser.parse_args()
    main(data_path=args.data, use_repeated_cv=not args.no_repeated, skip_shap=args.skip_shap)
