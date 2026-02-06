## libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


## neural network sklearn regressors
class NeuralQuantileRegressors:
    def __init__(self, quantile_c = 0.99, quantile_r = 0.5, input_dims = None, **kwargs):
        self.estimator_c = NeuralBaseRegressor(
            net_cls = QuantileNet,
            loss_fn = quantile_loss,
            quantile = quantile_c,
            input_dims = input_dims,
            **kwargs
        )
        self.estimator_r = NeuralBaseRegressor(
            net_cls = QuantileNet,
            loss_fn = quantile_loss,
            quantile = quantile_r,
            input_dims = input_dims,
            **kwargs
        )

class NeuralExpectileRegressors:
    def __init__(self, quantile_c = 0.99, quantile_r = 0.5, input_dims = None, **kwargs):
        self.estimator_c = NeuralBaseRegressor(
            net_cls = ExpectileNet,
            loss_fn = expectile_loss,
            quantile = quantile_c,
            input_dims = input_dims,
            **kwargs
        )
        self.estimator_r = NeuralBaseRegressor(
            net_cls = ExpectileNet,
            loss_fn = expectile_loss,
            quantile = quantile_r,
            input_dims = input_dims,
            **kwargs
        )

class NeuralConvexRegressors:
    def __init__(self, quantile_c = 0.99, quantile_r = 0.5, input_dims = None, **kwargs):
        self.estimator_c = NeuralBaseRegressor(
            net_cls = ConvexNet,
            loss_fn = quantile_loss,
            quantile = quantile_c,
            input_dims = input_dims,
            **kwargs
        )
        self.estimator_r = NeuralBaseRegressor(
            net_cls = ConvexNet,
            loss_fn = quantile_loss,
            quantile = quantile_r,
            input_dims = input_dims,
            **kwargs
        )


## neural network sklearn framework
class NeuralBaseRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, net_cls, loss_fn, quantile, input_dims, hidden_dims = [8, 4], lr = 0.1, epochs = 5000, dropout = 0.1, weight_decay = 0.01):
        self.net_cls = net_cls
        self.loss_fn = loss_fn
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.quantile = quantile 
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_ = None

    ## sklearn fit interface
    def fit(self, X, y):

        ## data prep
        X_train = np.array(X)
        y_train = np.array(y)

        ## train model
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        
        ## init model
        infer_dims = X_train.shape[1]
        if (self.input_dims is None or self.input_dims != infer_dims):
            input_dims = infer_dims
        else:
            input_dims = self.input_dims
        self.model_ = self.net_cls(input_dims, self.hidden_dims, dropout = self.dropout).to(
            self.device
        )
        
        ## init bias to quantile of training data for faster convergence
        bias_init = np.quantile(y_train, self.quantile)
        self.model_.mlp.net[-1].bias.data.fill_(bias_init)

        ## init optimizer
        optimizer = optim.AdamW(
            params = self.model_.parameters(), 
            lr = self.lr, 
            weight_decay = self.weight_decay
        )
        
        ## conduct training
        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model_(X_train_t)
            loss = self.loss_fn(y_train_t, preds, self.quantile, weights = None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm = 5.0)
            optimizer.step()
            
        return self
    
    ## sklearn predict interface
    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.FloatTensor(X).to(self.device)).cpu().numpy()


## quantile neural network
class QuantileNet(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.mlp = BaseNet(input_dims, hidden_dims, output_dims = 1, dropout = dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

## expectile neural network
class ExpectileNet(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.mlp = BaseNet(input_dims, hidden_dims, output_dims = 1, dropout = dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

## convex neural network
class ConvexNet(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: list[int], dropout: float = 0.0):
        super().__init__()
        self.mlp = BaseNet(input_dims, hidden_dims, output_dims = 1, dropout = dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.mlp.net):
            if isinstance(layer, nn.Linear) and i > 0:
                x = F.linear(x, layer.weight.abs(), layer.bias)
            else:
                x = layer(x)
        return x.squeeze(-1)

## base neural network (shared by quantile / expectile)
class BaseNet(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: list[int], output_dims: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev_dim = input_dims
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dims))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


## training loss functions
def quantile_loss(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float, weights: torch.Tensor = None) -> torch.Tensor:
    diff = y_true - y_pred
    loss = torch.maximum(alpha * diff, (alpha - 1.0) * diff)
    if weights is not None:
        loss = loss * weights
    return torch.mean(loss)

def expectile_loss(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float, weights: torch.Tensor = None) -> torch.Tensor:
    diff = y_true - y_pred
    weight_alpha = torch.where(diff > 0, alpha, 1.0 - alpha)
    loss = diff ** 2
    if weights is not None:
        weight_alpha = weight_alpha * weights
    return torch.mean(weight_alpha * loss)
