## libraries
from sklearn.base import BaseEstimator
from sklearn.linear_model import QuantileRegressor

## linear quantile regressors wrapper
class LinearQuantile(BaseEstimator):
    def __init__(self, quantile_c = 0.99, quantile_r = 0.5, alpha = 0.1, solver = "highs"):
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
