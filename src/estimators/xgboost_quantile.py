## libraries
from sklearn.base import BaseEstimator
import xgboost as xgb

## modules
from src.estimators.config import ASYMMETRY_C, ASYMMETRY_R

## xgboost sklearn regressors
class XGBoostQuantile(BaseEstimator):
    def __init__(self, quantile_c = ASYMMETRY_C, quantile_r = ASYMMETRY_R, **kwargs):
        self.quantile_c = quantile_c
        self.quantile_r = quantile_r
        self.kwargs = kwargs
        self.estimator_c = xgb.XGBRegressor(
            objective = "reg:quantileerror",
            quantile_alpha = quantile_c,
            **kwargs
        )
        self.estimator_r = xgb.XGBRegressor(
            objective = "reg:quantileerror",
            quantile_alpha = quantile_r,
            **kwargs
        )
