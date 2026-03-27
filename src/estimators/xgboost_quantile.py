## libraries
from typing import Any
from sklearn.base import BaseEstimator
import xgboost as xgb

## modules
from src.estimators.config import ASYMMETRY_C, ASYMMETRY_R

## xgboost sklearn regressors
class XGBoostQuantile(BaseEstimator):
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
        self.estimator_c = xgb.XGBRegressor(
            objective = "reg:quantileerror",
            quantile_alpha = quantile_c,  ## quantile, not reguarization
            random_state = random_state,
            **kwargs
        )
        self.estimator_r = xgb.XGBRegressor(
            objective = "reg:quantileerror",
            quantile_alpha = quantile_r,  ## quantile, not reguarization
            random_state = random_state,
            **kwargs
        )
