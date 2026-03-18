## libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Any, Callable
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin

## modules
from src.estimators.config import ASYMMETRY_C, ASYMMETRY_R

## neural network sklearn regressors
class NeuralQuantile(BaseEstimator):
    def __init__(
        self, 
        quantile_c: float = ASYMMETRY_C,
        quantile_r: float = ASYMMETRY_R,
        input_dims: int | None = None, 
        **kwargs: Any
        ) -> None:

        self.quantile_c = quantile_c
        self.quantile_r = quantile_r
        self.input_dims = input_dims
        self.kwargs: dict[str, Any] = kwargs

        self.estimator_c = NeuralBase(
            net_cls = QuantileNet,
            loss_fn = quantile_loss,
            quantile = quantile_c,
            input_dims = input_dims,
            **kwargs
        )
        self.estimator_r = NeuralBase(
            net_cls = QuantileNet,
            loss_fn = quantile_loss,
            quantile = quantile_r,
            input_dims = input_dims,
            **kwargs
        )

class NeuralExpectile(BaseEstimator):
    def __init__(
        self, 
        quantile_c: float = ASYMMETRY_C, 
        quantile_r: float = ASYMMETRY_R, 
        input_dims: int | None = None, 
        **kwargs: Any
        ) -> None:

        self.quantile_c = quantile_c
        self.quantile_r = quantile_r
        self.input_dims = input_dims
        self.kwargs: dict[str, Any] = kwargs
        
        self.estimator_c = NeuralBase(
            net_cls = ExpectileNet,
            loss_fn = expectile_loss,
            quantile = quantile_c,
            input_dims = input_dims,
            **kwargs
        )
        self.estimator_r = NeuralBase(
            net_cls = ExpectileNet,
            loss_fn = expectile_loss,
            quantile = quantile_r,
            input_dims = input_dims,
            **kwargs
        )

class NeuralConvex(BaseEstimator):
    def __init__(
        self, 
        quantile_c: float = ASYMMETRY_C, 
        quantile_r: float = ASYMMETRY_R, 
        input_dims: int | None = None, 
        **kwargs: Any
        ) -> None:

        self.quantile_c = quantile_c
        self.quantile_r = quantile_r
        self.input_dims = input_dims
        self.kwargs: dict[str, Any] = kwargs

        self.estimator_c = NeuralBase(
            net_cls = ConvexNet,
            loss_fn = quantile_loss,
            quantile = quantile_c,
            input_dims = input_dims,
            **kwargs
        )
        self.estimator_r = NeuralBase(
            net_cls = ConvexNet,
            loss_fn = quantile_loss,
            quantile = quantile_r,
            input_dims = input_dims,
            **kwargs
        )


## neural network sklearn framework
class NeuralBase(BaseEstimator, RegressorMixin):
    def __init__(
        self, 
        net_cls: type[nn.Module],
        loss_fn: Callable[[torch.Tensor, torch.Tensor, float, torch.Tensor | None], torch.Tensor],
        quantile: float,
        input_dims: int | None,
        hidden_dims: list[int] = [8, 4],
        lr: float = 0.1,
        epochs: int = 1000,
        dropout: float = 0.1,
        weight_decay: float = 0.01,
        random_state: int | None = None
        ) -> None:

        self.net_cls = net_cls
        self.loss_fn = loss_fn
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.quantile = quantile 
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_ = None

    ## sklearn fit interface
    def fit(self, X: ArrayLike, y: ArrayLike) -> "NeuralBase":

        ## seed torch for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        ## data prep
        X_train = np.array(X)
        y_train = np.array(y)

        ## train model
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        
        ## init model input dimensions based on training data
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
            
        ## store training target range for prediction clipping
        self.y_min_ = float(y_train.min())
        self.y_max_ = float(y_train.max())

        return self
    
    ## sklearn predict interface
    def predict(self, X: ArrayLike) -> NDArray[np.float32]:
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(torch.FloatTensor(X).to(self.device)).cpu().numpy()
            return np.clip(preds, self.y_min_, self.y_max_)


## quantile neural network
class QuantileNet(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.mlp = BaseNet(input_dims, hidden_dims, output_dims = 1, dropout = dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

## expectile neural network
class ExpectileNet(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.mlp = BaseNet(input_dims, hidden_dims, output_dims = 1, dropout = dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

## convex neural network
class ConvexNet(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: list[int], dropout: float = 0.0) -> None:
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
    def __init__(
        self, 
        input_dims: int, 
        hidden_dims: list[int], 
        output_dims: int, 
        dropout: float = 0.0
        ) -> None:

        super().__init__()

        layers = list()
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
def quantile_loss(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
    alpha: float, 
    weights: torch.Tensor | None = None
    ) -> torch.Tensor:
    
    diff = y_true - y_pred
    loss = torch.maximum(alpha * diff, (alpha - 1.0) * diff)
    if weights is not None:
        loss = loss * weights
    return torch.mean(loss)

def expectile_loss(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
    alpha: float, 
    weights: torch.Tensor | None = None
    ) -> torch.Tensor:

    diff = y_true - y_pred
    weight_alpha = torch.where(diff > 0, alpha, 1.0 - alpha)
    loss = diff ** 2
    if weights is not None:
        weight_alpha = weight_alpha * weights
    return torch.mean(weight_alpha * loss)
