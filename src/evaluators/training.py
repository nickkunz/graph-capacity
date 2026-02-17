## libraries
import numpy as np
import pandas as pd
from sklearn.base import clone

## modules
from src.vectorizers.scalers import _log_transformer, _standardizer

## ------------------------
## fit and predict frontier
## ------------------------
def fit_predict_frontier(
    data: pd.DataFrame,
    feat_x: list[str],
    feat_z: list[str],
    estimator_c,
    estimator_r,
    target: str = "target"
	) -> np.ndarray:

    ## init variables
    X = data[feat_x].apply(pd.to_numeric, errors = "coerce")
    Z = data[feat_z].apply(pd.to_numeric, errors = "coerce")
    y_star = _log_transformer(data[target]).astype(float)

    ## standardize graph invariants
    X_scaled, _ = _standardizer(X, feat_x)
    X_scaled = X_scaled[feat_x].values.astype(float)
    
	## standardize process signatures
    Z_scaled, _ = _standardizer(Z, feat_z)
    Z_scaled = Z_scaled[feat_z].values.astype(float)

	## ensure fresh parameters for each model
    model_c = clone(estimator_c)
    model_r = clone(estimator_r)

	## train C on graph invariants
    model_c.fit(X_scaled, y_star)
    c_hat = model_c.predict(X_scaled).astype(float)

    ## train R on process signatures and residualized target
    slack = (y_star - c_hat).astype(float)
    model_r.fit(Z_scaled, slack)
    r_hat = model_r.predict(Z_scaled).astype(float)

    ## identifiability constraint: zero-mean log-factor
    r_hat = (r_hat - np.mean(r_hat)).astype(float)

    ## final prediction: y* = log C + log R + epsilon
    y_pred = (c_hat + r_hat).astype(float)
    return y_pred, model_c, model_r

