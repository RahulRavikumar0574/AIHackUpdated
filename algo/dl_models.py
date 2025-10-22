import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import json

# =============================
# Scaler for normalization
# =============================
class SimpleScaler:
    """Simple min-max scaler that can be serialized to JSON."""
    def __init__(self):
        self.min_ = None
        self.max_ = None
    
    def fit_transform(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        denom = self.max_ - self.min_
        denom[denom == 0] = 1.0
        return (X - self.min_) / denom
    
    def transform(self, X):
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler not fitted yet")
        denom = self.max_ - self.min_
        denom[denom == 0] = 1.0
        return (X - self.min_) / denom
    
    def to_dict(self):
        return {
            'min': self.min_.tolist() if self.min_ is not None else None,
            'max': self.max_.tolist() if self.max_ is not None else None
        }
    
    @classmethod
    def from_dict(cls, d):
        scaler = cls()
        if d['min'] is not None:
            scaler.min_ = np.array(d['min'], dtype=np.float32)
        if d['max'] is not None:
            scaler.max_ = np.array(d['max'], dtype=np.float32)
        return scaler


# =============================
# Model Architectures
# =============================
class TabMLP(nn.Module):
    """Simple MLP for tabular data."""
    def __init__(self, in_dim, hidden_dims=[128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        x = self.layers(x)
        return self.head(x).squeeze(-1)


class TabCNN1D(nn.Module):
    """1D CNN for tabular data."""
    def __init__(self, in_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (batch, hidden_dim*2)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


class TabLSTM(nn.Module):
    """LSTM for tabular data."""
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features) - treat as sequence of length 1
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # Take last timestep
        return self.fc(out).squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x.unsqueeze(1))  # Add sequence dimension
        # Take the output from the final time step
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out.squeeze(-1)

class TabTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 64, nhead: int = 8, 
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(-1)          # (batch, features, 1)
        x = self.embed(x)            # (batch, features, d_model)
        x = self.dropout(x)
        h = self.encoder(x)          # (batch, features, d_model)
        h = self.norm(h)
        pooled = h.mean(dim=1)       # mean pool across features
        return self.head(pooled).squeeze(-1)

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # Calculate metrics
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Calculate AUPRC (Area Under Precision-Recall Curve)
    precision_curve, recall_curve, _ = precision_recall_curve(all_targets, all_probs)
    auprc = auc(recall_curve, precision_curve)
    
    # Calculate AUROC
    try:
        auroc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auroc = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc
    }
    
    return metrics, all_probs, all_targets

def find_optimal_threshold(model, dataloader, device, metric='f1'):
    """Find the optimal threshold based on the specified metric"""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # Try different thresholds and find the one that maximizes the target metric
    thresholds = np.linspace(0.1, 0.9, 50)
    best_metric = -1
    best_threshold = 0.5
    
    for thresh in thresholds:
        preds = (np.array(all_probs) >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(all_targets, preds, zero_division=0)
        elif metric == 'precision':
            score = precision_score(all_targets, preds, zero_division=0)
        elif metric == 'recall':
            score = recall_score(all_targets, preds, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        if score > best_metric:
            best_metric = score
            best_threshold = thresh
    
    return best_threshold, best_metric


def find_optimal_threshold_multi_metric(model, dataloader, device, metrics=['f1', 'precision', 'recall', 'accuracy']):
    """Find the optimal threshold based on multiple metrics (returns dict with threshold for each metric)"""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # Try different thresholds and find the one that maximizes each target metric
    thresholds = np.linspace(0.1, 0.9, 50)
    results = {}
    
    for metric in metrics:
        best_metric_value = -1
        best_threshold = 0.5
        
        for thresh in thresholds:
            preds = (np.array(all_probs) >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(all_targets, preds, zero_division=0)
            elif metric == 'precision':
                score = precision_score(all_targets, preds, zero_division=0)
            elif metric == 'recall':
                score = recall_score(all_targets, preds, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(all_targets, preds)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            if score > best_metric_value:
                best_metric_value = score
                best_threshold = thresh
        
        results[metric] = {'threshold': best_threshold, 'score': best_metric_value}
    
    return results


# =============================
# Training utilities
# =============================
def train_torch_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=256, lr=0.001, device='cpu'):
    """Train a PyTorch model and return training info."""
    if isinstance(device, str):
        device = torch.device(device)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * len(x_batch)
                
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    # Return validation probabilities and info
    return {
        'val_probs': np.array(all_probs),
        'val_targets': np.array(all_targets),
        'best_val_loss': best_val_loss
    }


def save_state(model, path):
    """Save model state dict."""
    torch.save(model.state_dict(), path)