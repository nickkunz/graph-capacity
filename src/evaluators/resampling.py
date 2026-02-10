## libraries
import numpy as np
import pandas as pd
from typing import Sequence
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import LeaveOneGroupOut, KFold

## modules
from src.vectorizers.scalers import _log_transformer, _standardizer
from src.evaluators.metrics import frontier_metrics, central_metrics

## ------------------------------------
## leave-one-group-out cross validation
## ------------------------------------
def logo_cross_valid(
    data: pd.DataFrame,
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    estimator_c: BaseEstimator,
    estimator_r: BaseEstimator,
    target: str = "target",
    group: str = "domain",
    ) -> pd.DataFrame:

    ## validate inputs
    if not feat_x:
        raise ValueError("feat_x must contain at least one column name.")
    if not feat_z:
        raise ValueError("feat_z must contain at least one column name.")

    ## init variables
    X = data[feat_x].apply(pd.to_numeric, errors = "coerce")
    Z = data[feat_z].apply(pd.to_numeric, errors = "coerce")
    y_star = _log_transformer(data[target]).astype(float)
    groups = data[group].values

    ## cross-validation
    logo = LeaveOneGroupOut()
    results = list()
    for train_idx, test_idx in logo.split(X = X.values, y = y_star.values, groups = groups):

        ## split training data
        X_train = X.iloc[train_idx]
        Z_train = Z.iloc[train_idx]
        y_train = y_star.iloc[train_idx].values.astype(float)

        ## split test data
        X_test = X.iloc[test_idx]
        Z_test = Z.iloc[test_idx]
        y_true = y_star.iloc[test_idx].values.astype(float)

        group_name = groups[test_idx][0]


        X_train_scaled, x_scaler = _standardizer(X_train, feat_x)
        X_train_scaled = X_train_scaled[feat_x].values.astype(float)
        X_test_scaled = x_scaler.transform(X_test.astype(float))

        ## ensure fresh parameters for each fold
        model_c = clone(estimator_c)
        model_r = clone(estimator_r)

        ## stage 1: graph invariants C(x')
        model_c.fit(X_train_scaled, y_train)
        c_hat_train = model_c.predict(X_train_scaled).astype(float)
        c_hat_test = model_c.predict(X_test_scaled).astype(float)

        ## stage 2: process signatures R(z')
        Z_train_scaled, z_scaler = _standardizer(Z_train, feat_z)
        Z_train_scaled = Z_train_scaled[feat_z].values.astype(float)
        Z_test_scaled = z_scaler.transform(Z_test.astype(float))

        ## signed slack in log-space (interpretable as log R)
        slack_train = (y_train - c_hat_train).astype(float)

        ## fit R on all observations
        model_r.fit(Z_train_scaled, slack_train)
        r_hat_train = model_r.predict(Z_train_scaled).astype(float)
        r_hat_test = model_r.predict(Z_test_scaled).astype(float)

        ## identifiability: force zero-mean log-factor
        r_hat_test = (r_hat_test - np.mean(r_hat_train)).astype(float)

        ## final prediction (log-space): y* = log C + log R
        y_pred_frontier = (c_hat_test + r_hat_test).astype(float)
        frontier = frontier_metrics(y_true = y_true, y_pred = y_pred_frontier)
        results.append({"group": group_name, **frontier})
    return pd.DataFrame(results)


## -----------------------
## k-fold cross validation
## -----------------------
def kfold_cross_valid(
    data: pd.DataFrame,
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    estimator_c: BaseEstimator,
    estimator_r: BaseEstimator,
    target: str = "target",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = None
    ) -> pd.DataFrame:

    ## validate inputs
    if not feat_x:
        raise ValueError("feat_x must contain at least one column name.")
    if not feat_z:
        raise ValueError("feat_z must contain at least one column name.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    ## extract variables
    X = data[feat_x].apply(pd.to_numeric, errors = "coerce")
    Z = data[feat_z].apply(pd.to_numeric, errors = "coerce")
    y_star = _log_transformer(data[target]).astype(float)

    ## cross-validation
    kfold = KFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
    results = list()
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X.values), start = 1):

        ## split training data
        X_train = X.iloc[train_idx]
        Z_train = Z.iloc[train_idx]
        y_train = y_star.iloc[train_idx].values.astype(float)

        ## split test data
        X_test = X.iloc[test_idx]
        Z_test = Z.iloc[test_idx]
        y_true = y_star.iloc[test_idx].values.astype(float)

        X_train_scaled, x_scaler = _standardizer(X_train, feat_x)
        X_train_scaled = X_train_scaled[feat_x].values.astype(float)
        X_test_scaled = x_scaler.transform(X_test.astype(float))

        ## ensure fresh parameters for each fold
        model_c = clone(estimator_c)
        model_r = clone(estimator_r)

        ## stage 1: graph invariants C(x')
        model_c.fit(X_train_scaled, y_train)
        c_hat_train = model_c.predict(X_train_scaled).astype(float)
        c_hat_test = model_c.predict(X_test_scaled).astype(float)

        ## stage 2: process signatures R(z')
        Z_train_scaled, z_scaler = _standardizer(Z_train, feat_z)
        Z_train_scaled = Z_train_scaled[feat_z].values.astype(float)
        Z_test_scaled = z_scaler.transform(Z_test.astype(float))

        ## signed slack in log-space (interpretable as log R)
        slack_train = (y_train - c_hat_train).astype(float)

        ## fit R on all observations
        model_r.fit(Z_train_scaled, slack_train)
        r_hat_train = model_r.predict(Z_train_scaled).astype(float)
        r_hat_test = model_r.predict(Z_test_scaled).astype(float)

        ## identifiability: force zero-mean log-factor
        r_hat_test = (r_hat_test - np.mean(r_hat_train)).astype(float)

        ## final prediction (log-space): y* = log C + log R
        y_pred_frontier = (c_hat_test + r_hat_test).astype(float)
        frontier = frontier_metrics(y_true = y_true, y_pred = y_pred_frontier)
        results.append({"fold": fold_idx, **frontier})
    return pd.DataFrame(results)

