## libraries
import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import linprog

class ChebyshevRegressors(BaseEstimator):
    def __init__(self):
        ## capacity: one-sided upper envelope
        self.estimator_c = ChebyshevRegressor(fit_intercept = True)

        ## process: usually NOT chebyshev — keep central
        self.estimator_r = ChebyshevRegressor(fit_intercept = False)

class ChebyshevRegressor(BaseEstimator):
    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        n, p = X.shape

        ## add intercept if requested
        if self.fit_intercept:
            X_ = np.c_[np.ones(n), X]
            p_ = p + 1
        else:
            X_ = X
            p_ = p

        ## decision variables: [beta_0, beta_1, ..., beta_p, epsilon]
        c = np.zeros(p_ + 1)
        c[-1] = 1.0  ## minimize epsilon

        ## constraints: X beta + epsilon >= y
        ## rewritten as: -X beta - epsilon <= -y
        A_ub = np.hstack([
            -X_,
            -np.ones((n, 1))
        ])
        b_ub = -y

        ## no bounds on beta, epsilon >= 0
        bounds = [(None, None)] * p_ + [(0.0, None)]

        res = linprog(
            c = c,
            A_ub = A_ub,
            b_ub = b_ub,
            bounds = bounds,
            method = "highs"
        )

        if not res.success:
            raise RuntimeError("Chebyshev regression failed to converge")

        beta = res.x[:-1]
        self.epsilon_ = res.x[-1]

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

