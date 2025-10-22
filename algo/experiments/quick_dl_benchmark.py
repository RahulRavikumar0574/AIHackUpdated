"""
Quick benchmark for deep learning models (Transformer & LSTM) on Kaggle fraud dataset.
No SHAP analysis - just metrics comparison.
"""
import os
import json
import sys
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dl_models import TabTransformer, LSTMClassifier, SimpleScaler

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'creditcard.csv')
OUT_DIR = os.path.join(BASE_DIR, 'experiments', 'kaggle_outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def load_kaggle_dataset(path: str) -> pd.DataFrame:
    """Load Kaggle fraud dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if 'Class' not in df.columns:
        raise ValueError("Expected 'Class' target column")
    return df


def preprocess_df(df: pd.DataFrame) -> tuple:
    """Preprocess Kaggle dataset."""
    X = df.drop(columns=['Class'])
    y = df['Class']
    # Drop Time column if exists
    if 'Time' in X.columns:
        X = X.drop(columns=['Time'])
    return X, y


class TorchModelWrapper:
    """Wrapper to make PyTorch models sklearn-compatible."""
    
    def __init__(self, model_class, input_dim, epochs=10, batch_size=512, lr=0.001):
        self.model_class = model_class
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
    
    def fit(self, X, y):
        """Fit the model."""
        # Scale data
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
            total_loss = 0
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        self.model.eval()
        X_scaled = self.scaler.transform(X.values if hasattr(X, 'values') else X)
        
        # Ensure proper type conversion to float32
        X_scaled = np.asarray(X_scaled, dtype=np.float32)
        
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
        
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def find_optimal_threshold(y_true, y_proba):
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    threshold = find_optimal_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics (convert to Python native types for JSON serialization)
    results = {
        'model': model_name,
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'ap': float(average_precision_score(y_test, y_proba)),
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
    }
    
    print(f"\nResults for {model_name}:")
    print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    print(f"  AP: {results['ap']:.4f}")
    print(f"  Threshold: {results['threshold']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    
    return results


def main():
    print("Loading Kaggle fraud dataset...")
    df = load_kaggle_dataset(DATA_PATH)
    X, y = preprocess_df(df)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Fraud rate: {y.mean():.4f}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Evaluate models
    results = []
    
    # Transformer
    print("\n" + "="*60)
    print("TRANSFORMER MODEL")
    print("="*60)
    transformer = TorchModelWrapper(TabTransformer, X.shape[1], epochs=10, batch_size=512)
    res_transformer = evaluate_model(transformer, X_train, X_test, y_train, y_test, 'Transformer')
    results.append(res_transformer)
    
    # LSTM
    print("\n" + "="*60)
    print("LSTM MODEL")
    print("="*60)
    lstm = TorchModelWrapper(LSTMClassifier, X.shape[1], epochs=10, batch_size=512)
    res_lstm = evaluate_model(lstm, X_train, X_test, y_train, y_test, 'LSTM')
    results.append(res_lstm)
    
    # Save results
    output_file = os.path.join(OUT_DIR, 'dl_models_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_file}")
    
    # Display comparison
    print("\nModel Comparison:")
    print(f"{'Model':<15} {'ROC-AUC':<10} {'AP':<10} {'F1':<10} {'Accuracy':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<15} {r['roc_auc']:<10.4f} {r['ap']:<10.4f} {r['f1']:<10.4f} {r['accuracy']:<10.4f}")
    
    # Best model
    best = max(results, key=lambda x: x['roc_auc'])
    print(f"\n🏆 Best Model: {best['model']} (ROC-AUC: {best['roc_auc']:.4f})")


if __name__ == '__main__':
    main()
