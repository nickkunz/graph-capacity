## libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin

## base neural network
class TinyNet(nn.Module):
    """ Simple two-layer MLP with 8 hidden units. """
    def __init__(self, p):
        super().__init__()
        self.fc1 = nn.Linear(p, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

## sklearn wrapper for TinyNet
class TinyNetRegressor(BaseEstimator, RegressorMixin):
    """ Scikit-learn compatible wrapper of TinyNet. """
    def __init__(self, input_dim, lr = 0.01, weight_decay = 0.01, epochs = 2000, device = None):
        self.input_dim = input_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device
    def fit(self, X, y):
        X_arr = np.asarray(X, dtype = np.float32)
        y_arr = np.asarray(y, dtype = np.float32).reshape(-1, 1)
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = TinyNet(self.input_dim).to(self.device_)
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr = self.lr,
            weight_decay = self.weight_decay,
        )
        loss_fn = nn.MSELoss()
        X_tensor = torch.from_numpy(X_arr).to(self.device_)
        y_tensor = torch.from_numpy(y_arr).to(self.device_)
        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model_(X_tensor)
            loss = loss_fn(preds, y_tensor)
            loss.backward()
            optimizer.step()
        return self
    def predict(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("TinyNetRegressor must be fit before calling predict().")
        X_arr = np.asarray(X, dtype = np.float32)
        X_tensor = torch.from_numpy(X_arr).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(X_tensor).cpu().numpy().ravel()
        return preds
