## libraries
from sklearn.base import BaseEstimator
from sklearn.linear_model import QuantileRegressor

## modules
from src.estimators.config import ASYMMETRY_C, ASYMMETRY_R

## linear quantile sklearn regressors
class LinearQuantile(BaseEstimator):
    def __init__(self, quantile_c = ASYMMETRY_C, quantile_r = ASYMMETRY_R, alpha = 0.1, solver = "highs"):
        self.quantile_c = quantile_c
        self.quantile_r = quantile_r
        self.alpha = alpha
        self.solver = solver
        self.estimator_c = QuantileRegressor(
            quantile = quantile_c, 
            alpha = alpha, 
            solver = solver
        )
        self.estimator_r = QuantileRegressor(
            quantile = quantile_r, 
            alpha = alpha, 
            solver = solver
        )
