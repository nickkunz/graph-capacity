## libraries
import numpy as np
from scipy.optimize import linprog
from sklearn.base import BaseEstimator, RegressorMixin

## linear constrained sklearn regressors
class LinearConstrained(BaseEstimator):
    def __init__(self, quantile_c = 0.99, quantile_r = 0.5):
        self.quantile_c = quantile_c
        self.quantile_r = quantile_r
        self.estimator_c = BaseConstrained(quantile = quantile_c, fit_intercept = True)
        self.estimator_r = BaseConstrained(quantile = quantile_r, fit_intercept = False)

## linear constrained sklearn framework
class BaseConstrained(BaseEstimator, RegressorMixin):
    def __init__(self, quantile = 0.95, fit_intercept = True):
        self.quantile = quantile
        self.fit_intercept = fit_intercept

    ## sklearn fit interface
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        n, p = X.shape

        if self.fit_intercept:
            X_ = np.c_[np.ones(n), X]
            p_ = p + 1
            nonneg_idx = list(range(1, p_))
        else:
            X_ = X
            p_ = p
            nonneg_idx = list(range(p_))

        ## objective coefficients
        c = np.concatenate([
            np.zeros(p_),
            self.quantile * np.ones(n),
            (1 - self.quantile) * np.ones(n)
        ])

        ## base equality constraints
        A_eq = np.hstack([X_, np.eye(n), -np.eye(n)])
        b_eq = y

        ## variable bounds
        bounds = [(None, None)] * p_
        for j in nonneg_idx:
            bounds[j] = (0.0, None)
            
        bounds += [(0.0, None)] * (2 * n)

        ## solve linear program
        res = linprog(c, A_eq = A_eq, b_eq = b_eq, bounds = bounds, method = "highs")
        if not res.success:
            raise RuntimeError("Constrained quantile failed")

        ## extract coefficients
        beta = res.x[:p_]
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        return self

    ## sklearn predict interface
    def predict(self, X):
        return X @ self.coef_ + self.intercept_
