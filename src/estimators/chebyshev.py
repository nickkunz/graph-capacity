## libraries
import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import linprog

## chebyshev regression sklearn regressors
class ChebyshevRegressors(BaseEstimator):
    def __init__(self):
        self.estimator_c = ChebyshevRegressor(fit_intercept = True)
        self.estimator_r = ChebyshevRegressor(fit_intercept = False)

## chebyshev regression sklearn framework
class ChebyshevRegressor(BaseEstimator):
    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept

    ## sklearn fit interface
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

        ## decision variables
        c = np.zeros(p_ + 1)
        c[-1] = 1.0  ## minimize epsilon

        ## constraints
        A_ub = np.hstack([
            -X_,
            -np.ones((n, 1))
        ])
        b_ub = -y

        ## no bounds on coefficients
        bounds = [(None, None)] * p_ + [(0.0, None)]

        ## solve linear program
        res = linprog(
            c = c,
            A_ub = A_ub,
            b_ub = b_ub,
            bounds = bounds,
            method = "highs"
        )
        if not res.success:
            raise RuntimeError("Chebyshev regression failed to converge")

        ## extract coefficients and epsilon
        beta = res.x[:-1]
        self.epsilon_ = res.x[-1]
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

