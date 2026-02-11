## libraries
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import QuantileRegressor

## convex hull sklearn regressors
class LinearConvex(BaseEstimator):
    def __init__(self, quantile_c = 0.99, quantile_r = 0.5, alpha = 0.1, beta = None):
        self.estimator_c = BaseConvex(quantile = quantile_c, alpha = alpha, beta = beta)
        self.estimator_r = BaseConvex(quantile = quantile_r, alpha = alpha, beta = beta)

## convex hull sklearn framework
class BaseConvex(BaseEstimator, RegressorMixin):
    def __init__(self, quantile, alpha, beta):
        self.beta = beta
        self.quantile = quantile
        self.alpha = alpha

        # fitted state
        self.beta_ = None
        self.s_hull_ = None
        self.y_hull_ = None

    ## sklearn fit interfaces
    def fit(self, X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float).reshape(-1)
        if X.size == 0:
            raise ValueError("Empty X passed to ConvexHullRegressors.fit")

        # infer a linear score if beta not provided
        if self.beta is None:
            qr = QuantileRegressor(quantile = self.quantile, alpha = self.alpha, solver = "highs")
            qr.fit(X, y)
            self.beta_ = np.asarray(qr.coef_)
        else:
            self.beta_ = np.asarray(self.beta)

        # 1-d structural scores
        s = X @ self.beta_

        uniq_s, inv = np.unique(s, return_inverse = True)
        y_max = np.full(len(uniq_s), -np.inf, dtype = float)
        np.maximum.at(y_max, inv, y)

        # compute concave upper envelope on aggregated points
        self.s_hull_, self.y_hull_ = _concave_upper_envelope(uniq_s, y_max)
        return self

    ## sklearn predict interface
    def predict(self, X):
        if self.s_hull_ is None or self.y_hull_ is None:
            raise RuntimeError("ConvexHullRegressor must be fit before calling predict().")
        X = np.asarray(X, dtype = float)
        s = X @ self.beta_
        if self.s_hull_.size == 1:
            return np.full(len(s), float(self.y_hull_[0]))
        return np.interp(s, self.s_hull_, self.y_hull_, left = float(self.y_hull_[0]), right = float(self.y_hull_[-1]))


## compute concave upper envelope
def _concave_upper_envelope(x, y):
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)

    # filter non-finite
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.array([0.0]), np.array([0.0])

    # if only one point, return it
    if x.size == 1:
        return x.copy(), y.copy()

    # ensure x is strictly increasing by keeping first occurrence of ties
    uniq_x, idx = np.unique(x, return_index=True)
    x = uniq_x
    y = y[idx]

    ## compute upper envelope using a stack-based algorithm
    x_hull = []
    y_hull = []
    for xi, yi in zip(x, y):
        while len(x_hull) >= 2:
            x1, y1 = x_hull[-2], y_hull[-2]
            x2, y2 = x_hull[-1], y_hull[-1]
            
            # slopes
            slope1 = (y2 - y1) / (x2 - x1)
            slope2 = (yi - y2) / (xi - x2)
            if slope2 >= slope1:
                x_hull.pop(); y_hull.pop()
            else:
                break
        x_hull.append(xi); y_hull.append(yi)

    return np.asarray(x_hull), np.asarray(y_hull)
