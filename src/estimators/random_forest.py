## libraries
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn_quantile import RandomForestQuantileRegressor

class ForestQuantile:
    def __init__(self, quantile_c = 0.99, quantile_r = 0.5, **kwargs):
        self.estimator_c = ForestBase(quantile = quantile_c, **kwargs)
        self.estimator_r = ForestBase(quantile = quantile_r, **kwargs)

class ForestBase(BaseEstimator, RegressorMixin):
    def __init__(self, quantile, **kwargs):
        self.quantile = quantile
        self.kwargs = kwargs

    def fit(self, X, y):
        # quantile(s) must be passed at construction time
        self.model_ = RandomForestQuantileRegressor(
            q = [self.quantile],
            **self.kwargs
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        preds = self.model_.predict(X)
        if preds.ndim == 1:
            return preds
        return preds[:, 0]  # squeeze the single-quantile dimension

