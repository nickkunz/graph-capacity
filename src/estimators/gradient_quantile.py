## libraries
from typing import Any
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor

## modules
from src.estimators.config import ASYMMETRY_C, ASYMMETRY_R

## gradient boosting sklearn regressors
class BoostingQuantile(BaseEstimator):
    def __init__(
        self,
        quantile_c: float = ASYMMETRY_C,
        quantile_r: float = ASYMMETRY_R,
        **kwargs: Any
        ) -> None:

        self.quantile_c = quantile_c
        self.quantile_r = quantile_r
        self.kwargs: dict[str, Any] = kwargs
        self.estimator_c = GradientBoostingRegressor(
            loss = "quantile", 
            alpha = quantile_c,  ## quantile, not regularization
            **kwargs
        )
        self.estimator_r = GradientBoostingRegressor(
            loss = "quantile", 
            alpha = quantile_r,  ## quantile, not regularization
            **kwargs
        )
