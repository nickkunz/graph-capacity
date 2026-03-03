## libraries
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor

## modules
from src.estimators.config import ASYMMETRY_C, ASYMMETRY_R

## gradient boosting sklearn regressors
class BoostingQuantile(BaseEstimator):
    def __init__(self, quantile_c = ASYMMETRY_C, quantile_r = ASYMMETRY_R, **kwargs):
        self.quantile_c = quantile_c
        self.quantile_r = quantile_r
        self.kwargs = kwargs
        self.estimator_c = GradientBoostingRegressor(
            loss = "quantile", 
            alpha = quantile_c, 
            **kwargs
        )
        self.estimator_r = GradientBoostingRegressor(
            loss = "quantile", 
            alpha = quantile_r, 
            **kwargs
        )
