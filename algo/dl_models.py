import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn


# -----------------------------
# Simple Standard Scaler (fit on train, apply to arrays)
# -----------------------------
@dataclass
class SimpleScaler:
    mean_: np.ndarray = None
    std_: np.ndarray = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        sc = cls()
        sc.mean_ = np.array(d["mean"], dtype=np.float32)
        sc.std_ = np.array(d["std"], dtype=np.float32)
        return sc


# -----------------------------
# Models
# -----------------------------
class TabMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TabCNN1D(nn.Module):
    def __init__(self, in_dim: int, channels: int = 32, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.head = nn.Linear(in_dim * channels, 1)

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = x.flatten(1)
        return self.head(x).squeeze(-1)


class TabLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
        self.in_dim = in_dim

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(-1)  # (batch, features, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# -----------------------------
# Utilities
# -----------------------------

def save_state(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_state(model: nn.Module, path: str, map_location: str = "cpu") -> nn.Module:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    return model


def train_torch_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      epochs: int = 10, batch_size: int = 256, lr: float = 1e-3,
                      device: str = None) -> Dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_ds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # return validation logits -> probs
    with torch.no_grad():
        all_logits = []
        for xb, _ in val_loader:
            xb = xb.to(device)
            all_logits.append(model(xb).cpu())
        logits = torch.cat(all_logits)
        probs = torch.sigmoid(logits).numpy()
    return {"val_loss": best_val_loss, "val_probs": probs}