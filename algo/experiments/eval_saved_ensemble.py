import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from typing import Dict
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure we can import modules from the parent 'algo' directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Reuse feature mapping from training and scaler/model classes
from dl_models import TabMLP, TabCNN1D, TabLSTM, SimpleScaler
from train_ensemble import _map_dataset_to_form_fields

DATA_PATH = os.path.join(BASE_DIR, 'final_model_ready_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def load_artifacts():
    feature_cols_path = os.path.join(MODELS_DIR, 'feature_columns.json')
    ensemble_json_path = os.path.join(MODELS_DIR, 'ensemble.json')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.json')
    lgbm_path = os.path.join(MODELS_DIR, 'lgbm.pkl')
    mlp_path = os.path.join(MODELS_DIR, 'mlp.pt')
    cnn_path = os.path.join(MODELS_DIR, 'cnn1d.pt')
    lstm_path = os.path.join(MODELS_DIR, 'lstm.pt')

    with open(feature_cols_path, 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)['feature_columns']
    with open(ensemble_json_path, 'r', encoding='utf-8') as f:
        ens_cfg = json.load(f)
        weights = np.array(ens_cfg['weights'], dtype=float)
        bias = float(ens_cfg['bias'])
        model_names = ens_cfg.get('models', ["lgbm", "mlp", "cnn1d", "lstm"])  # order matters
    with open(scaler_path, 'r', encoding='utf-8') as f:
        scaler = SimpleScaler.from_dict(json.load(f))

    models: Dict[str, object] = {}
    if "lgbm" in model_names and os.path.exists(lgbm_path):
        models['lgbm'] = joblib.load(lgbm_path)
    if "mlp" in model_names and os.path.exists(mlp_path):
        mlp = TabMLP(in_dim=len(feature_columns))
        mlp.load_state_dict(torch.load(mlp_path, map_location='cpu'))
        mlp.eval()
        models['mlp'] = mlp
    if "cnn1d" in model_names and os.path.exists(cnn_path):
        cnn = TabCNN1D(in_dim=len(feature_columns))
        cnn.load_state_dict(torch.load(cnn_path, map_location='cpu'))
        cnn.eval()
        models['cnn1d'] = cnn
    if "lstm" in model_names and os.path.exists(lstm_path):
        lstm = TabLSTM(in_dim=len(feature_columns))
        lstm.load_state_dict(torch.load(lstm_path, map_location='cpu'))
        lstm.eval()
        models['lstm'] = lstm

    return feature_columns, scaler, models, model_names, weights, bias


def build_feature_matrix(df: pd.DataFrame, feature_columns):
    """Map raw dataset to API-like schema and align columns to feature_columns."""
    X_all, y_all = _map_dataset_to_form_fields(df, target_col=('target' if 'target' in df.columns else 'RISK_ASSESSMENT'))
    # Align to saved feature column order
    X = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        X[col] = X_all[col] if col in X_all.columns else 0
    return X, y_all


def evaluate(models, model_names, weights, bias, scaler, X, y):
    metrics = {}

    # Scale for torch models
    X_np = scaler.transform(X.values.astype(np.float32))
    x_tensor = torch.tensor(X_np, dtype=torch.float32)

    # Per-model probs
    probs: Dict[str, np.ndarray] = {}
    if 'lgbm' in models:
        probs['lgbm'] = models['lgbm'].predict_proba(X)[:, 1]
    if 'mlp' in models:
        with torch.no_grad():
            probs['mlp'] = torch.sigmoid(models['mlp'](x_tensor)).cpu().numpy().ravel()
    if 'cnn1d' in models:
        with torch.no_grad():
            probs['cnn1d'] = torch.sigmoid(models['cnn1d'](x_tensor)).cpu().numpy().ravel()
    if 'lstm' in models:
        with torch.no_grad():
            probs['lstm'] = torch.sigmoid(models['lstm'](x_tensor)).cpu().numpy().ravel()

    # Compute metrics per available model
    for name, p in probs.items():
        try:
            roc = roc_auc_score(y, p)
            ap = average_precision_score(y, p)
        except Exception:
            roc, ap = float('nan'), float('nan')
        metrics[f'{name}_ROC_AUC'] = float(roc)
        metrics[f'{name}_AP'] = float(ap)

    # Ensemble combine (respect order in model_names that are present)
    stacked = []
    used_names = []
    for n in model_names:
        if n in probs:
            stacked.append(probs[n])
            used_names.append(n)
    if stacked:
        P = np.vstack(stacked).T  # shape (N, k)
        # If weight length differs (e.g., fewer models in FAST mode), slice
        w = weights[: P.shape[1]]
        logits = bias + np.dot(P, w)
        p_ens = 1.0 / (1.0 + np.exp(-logits))
        try:
            metrics['ensemble_ROC_AUC'] = float(roc_auc_score(y, p_ens))
            metrics['ensemble_AP'] = float(average_precision_score(y, p_ens))
        except Exception:
            metrics['ensemble_ROC_AUC'] = float('nan')
            metrics['ensemble_AP'] = float('nan')
        metrics['ensemble_models_used'] = used_names

    return metrics


if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    feature_columns, scaler, models, model_names, weights, bias = load_artifacts()

    # Build features and split a small holdout for evaluation
    X_all, y_all = build_feature_matrix(df, feature_columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # Evaluate on holdout
    results = evaluate(models, model_names, weights, bias, scaler, X_test, y_test.values)

    # Print nicely
    print("\nPer-model and ensemble metrics on holdout:")
    for k in sorted(results.keys()):
        if k.endswith('_ROC_AUC') or k.endswith('_AP'):
            print(f" - {k}: {results[k]:.4f}")
    if 'ensemble_models_used' in results:
        print(" - ensemble_models_used:", results['ensemble_models_used'])

    # Save to models/metrics.json
    out_path = os.path.join(MODELS_DIR, 'metrics.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", out_path)
