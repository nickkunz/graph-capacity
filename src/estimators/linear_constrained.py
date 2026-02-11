## libraries
import numpy as np
from scipy.optimize import linprog
from sklearn.base import BaseEstimator, RegressorMixin

## linear constrained sklearn regressors
class LinearConstrained(BaseEstimator):
    def __init__(
        self, 
        quantile_c = 0.99, 
        quantile_r = 0.5, 
        fit_intercept = True, 
        constraint = "nonneg"
    ):
        self.quantile_c = quantile_c
        self.quantile_r = quantile_r
        self.fit_intercept = fit_intercept
        self.constraint = constraint
        self.estimator_c = BaseConstrainedQuantile(
            quantile = quantile_c, 
            fit_intercept = fit_intercept,
            constraint = constraint
        )
        self.estimator_r = BaseConstrainedQuantile(
            quantile = quantile_r, 
            fit_intercept = fit_intercept,
            constraint = constraint
        )

## sc-lqr implementation
class BaseConstrainedQuantile(BaseEstimator, RegressorMixin):
    """
    Linear Quantile Regression with constraints (SC-LQR).
    
    Parameters
    ----------
    quantile : float, default=0.95
        Quantile to estimate.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    constraint : {"nonneg", "simplex"}, default="nonneg"
        - "nonneg": beta_j >= 0 (Type A)
        - "simplex": beta_j >= 0 and sum(beta_j) = 1 (Type C)
    """
    def __init__(self, quantile = 0.95, fit_intercept = True, constraint = "nonneg"):
        self.quantile = quantile
        self.fit_intercept = fit_intercept
        self.constraint = constraint

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        n, p = X.shape

        if self.fit_intercept:
            X_ = np.c_[np.ones(n), X]
            p_ = p + 1
            # Constraints apply to features, usually not intercept
            feat_idx = list(range(1, p_))
        else:
            X_ = X
            p_ = p
            feat_idx = list(range(p_))

        ## variables: beta (p_), u (n), v (n)
        c = np.concatenate([
            np.zeros(p_),
            self.quantile * np.ones(n),
            (1 - self.quantile) * np.ones(n)
        ])

        ## base equality constraints: X beta + u - v = y
        A_eq = np.hstack([X_, np.eye(n), -np.eye(n)])
        b_eq = y

        ## add simplex constraint: sum(beta_features) = 1
        if self.constraint == "simplex":
            simplex_row = np.zeros(p_ + 2 * n)
            simplex_row[feat_idx] = 1.0
            
            A_eq = np.vstack([A_eq, simplex_row])
            b_eq = np.concatenate([b_eq, [1.0]])

        ## bounds: beta_features >= 0, u >= 0, v >= 0
        bounds = [(None, None)] * p_
        for j in feat_idx:
            bounds[j] = (0.0, None)
            
        bounds += [(0.0, None)] * (2 * n)

        res = linprog(c, A_eq = A_eq, b_eq = b_eq, bounds = bounds, method = "highs")
        if not res.success:
            raise RuntimeError(f"Constrained quantile ({self.constraint}) failed")

        beta = res.x[:p_]
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
