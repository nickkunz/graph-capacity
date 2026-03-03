## libraries
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn_quantile import RandomForestQuantileRegressor

## modules
from src.estimators.config import ASYMMETRY_C, ASYMMETRY_R

## random forest sklearn regressors
class ForestQuantile:
    def __init__(self, quantile_c = ASYMMETRY_C, quantile_r = ASYMMETRY_R, **kwargs):
        self.estimator_c = ForestBase(quantile = quantile_c, **kwargs)
        self.estimator_r = ForestBase(quantile = quantile_r, **kwargs)

## random forest sklearn framework
class ForestBase(BaseEstimator, RegressorMixin):
    def __init__(self, quantile, **kwargs):
        self.quantile = quantile
        self.kwargs = kwargs

    ## sklearn fit interface
    def fit(self, X, y):
        self.model_ = RandomForestQuantileRegressor(
            q = [self.quantile],  ## quantile must be passed at construction time
            **self.kwargs
        )
        self.model_.fit(X, y)
        return self

    ## sklearn predict interface
    def predict(self, X):
        preds = self.model_.predict(X)
        if preds.ndim == 1:
            return preds
        return preds[:, 0]  ## squeeze to single-quantile dimension

