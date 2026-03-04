## libraries
import numpy as np
from typing import Any
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn_quantile import RandomForestQuantileRegressor

## modules
from src.estimators.config import ASYMMETRY_C, ASYMMETRY_R

## random forest sklearn regressors
class ForestQuantile:
    def __init__(
        self,
        quantile_c: float = ASYMMETRY_C,
        quantile_r: float = ASYMMETRY_R,
        **kwargs: Any
        ) -> None:
        self.estimator_c = ForestBase(quantile = quantile_c, **kwargs)
        self.estimator_r = ForestBase(quantile = quantile_r, **kwargs)

## random forest sklearn framework
class ForestBase(BaseEstimator, RegressorMixin):
    def __init__(self, quantile: float, **kwargs: Any) -> None:
        self.quantile = quantile
        self.kwargs: dict[str, Any] = kwargs

    ## sklearn fit interface
    def fit(self, X: ArrayLike, y: ArrayLike) -> "ForestBase":
        self.model_ = RandomForestQuantileRegressor(
            q = [self.quantile],  ## quantile must be passed at construction time
            **self.kwargs
        )
        self.model_.fit(X, y)
        return self

    ## sklearn predict interface
    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        preds = self.model_.predict(X)
        if preds.ndim == 1:
            return preds
        return preds[:, 0]  ## squeeze to single-quantile dimension

