## libraries
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn_quantile import RandomForestQuantileRegressor

class UpperQuantileRF(BaseEstimator, RegressorMixin):
    def __init__(self, quantile = 0.99, **kwargs):
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
