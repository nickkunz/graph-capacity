## libraries
from sklearn.base import BaseEstimator
from sklearn.linear_model import QuantileRegressor

## modules
from src.estimators.config import (
    ASYMMETRY_C,
    ASYMMETRY_R,
    ALPHA_C,
    ALPHA_R
)

## linear quantile sklearn regressors
class LinearQuantile(BaseEstimator):
    def __init__(self, 
        quantile_c: float = ASYMMETRY_C, 
        quantile_r: float = ASYMMETRY_R, 
        alpha_c: float = ALPHA_C, 
        alpha_r: float = ALPHA_R,
        solver: str = "highs"
        ) -> None:

        self.quantile_c = quantile_c
        self.quantile_r = quantile_r
        self.alpha_c = alpha_c
        self.alpha_r = alpha_r
        self.solver = solver
        self.estimator_c = QuantileRegressor(
            quantile = quantile_c, 
            alpha = alpha_c, 
            solver = solver
        )
        self.estimator_r = QuantileRegressor(
            quantile = quantile_r, 
            alpha = alpha_r, 
            solver = solver
        )
