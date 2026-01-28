## libraries
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import QuantileRegressor


class ConvexHullRegressors(BaseEstimator):
    """Simple convex-hull (max-affine in 1D score) wrapper.

    - If `beta` is None, a linear upper score is inferred via linear quantile regressor.
    - The hull is computed on the 1D score s = X @ beta using the upper concave envelope
      of points (s, y) (duplicates aggregated by max y).

    The wrapper exposes `estimator_c` and `estimator_r` to match other models so it
    can be plugged into the LOSO pipeline.
    """
    def __init__(self, beta = None, quantile = 0.99, alpha = 0.1):
        self.beta = beta
        self.quantile = quantile
        self.alpha = alpha

        # expose same attributes as other wrappers
        self.estimator_c = self
        self.estimator_r = DummyRegressor(strategy = "constant", constant = 0.0)

        # fitted state
        self.beta_ = None
        self.s_hull_ = None
        self.y_hull_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

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

        # aggregate duplicate s values: take max y for each unique s
        uniq_s, inv = np.unique(s, return_inverse = True)
        y_max = np.full(len(uniq_s), -np.inf, dtype=float)
        for i, yi in enumerate(y):
            y_max[inv[i]] = max(y_max[inv[i]], float(yi))

        # compute concave upper envelope on aggregated points
        self.s_hull_, self.y_hull_ = _concave_upper_envelope(uniq_s, y_max)

        return self

    def predict(self, X):
        if self.s_hull_ is None or self.y_hull_ is None:
            raise RuntimeError("ConvexHullRegressors must be fit before calling predict().")
        X = np.asarray(X)
        s = X @ self.beta_
        if self.s_hull_.size == 1:
            return np.full(len(s), float(self.y_hull_[0]))
        return np.interp(s, self.s_hull_, self.y_hull_, left = float(self.y_hull_[0]), right = float(self.y_hull_[-1]))


def _concave_upper_envelope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

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

    hull_x = []
    hull_y = []

    for xi, yi in zip(x, y):
        while len(hull_x) >= 2:
            x1, y1 = hull_x[-2], hull_y[-2]
            x2, y2 = hull_x[-1], hull_y[-1]
            # slopes
            slope1 = (y2 - y1) / (x2 - x1)
            slope2 = (yi - y2) / (xi - x2)
            # if slope2 >= slope1 then last point is not part of concave hull
            if slope2 >= slope1:
                hull_x.pop(); hull_y.pop()
            else:
                break
        hull_x.append(xi); hull_y.append(yi)

    return np.asarray(hull_x), np.asarray(hull_y)
