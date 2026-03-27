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
        random_state: int = 42,
        **kwargs: Any
        ) -> None:

        self.quantile_c = quantile_c
        self.quantile_r = quantile_r
        self.random_state = random_state
        self.kwargs: dict[str, Any] = kwargs
        self.estimator_c = GradientBoostingRegressor(
            loss = "quantile", 
            alpha = quantile_c,  ## quantile, not regularization
            random_state = random_state,
            **kwargs
        )
        self.estimator_r = GradientBoostingRegressor(
            loss = "quantile", 
            alpha = quantile_r,  ## quantile, not regularization
            random_state = random_state,
            **kwargs
        )
