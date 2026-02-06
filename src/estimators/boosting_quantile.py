## libraries
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor

## gradient boosting sklearn regressors
class BoostingQuantile(BaseEstimator):
    def __init__(self, quantile_c = 0.99, quantile_r = 0.5, **kwargs):
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
