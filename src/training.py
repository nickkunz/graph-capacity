## libraries
import numpy as np
import pandas as pd
from typing import Sequence
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import LeaveOneGroupOut, KFold

## modules
from src.scalers import _log_transform_target, _standardize_invariants
from src.metrics import frontier_metrics, central_metrics


## ------------------------------------------------------------------
## leave-one-group-out cross validation (two-stage orthogonalized)
## ------------------------------------------------------------------
def logo_cross_valid(
    estimator_c: BaseEstimator,
    estimator_r: BaseEstimator,
    data: pd.DataFrame,
    struct_feat: Sequence[str],
    proc_feat: Sequence[str],
    target: str = "target",
    group: str = "domain",
    alpha: float = 0.01,
) -> pd.DataFrame:

    ## validate inputs
    if not struct_feat:
        raise ValueError("struct_feat must contain at least one column name.")
    if not proc_feat:
        raise ValueError("proc_feat must contain at least one column name.")

    ## extract variables
    X = data[struct_feat].apply(pd.to_numeric, errors = "coerce")
    Z = data[proc_feat].apply(pd.to_numeric, errors = "coerce")
    y = _log_transform_target(data[target]).astype(float)
    groups = data[group].values

    ## calibration level
    alpha = float(alpha)

    ## cross-validation
    logo = LeaveOneGroupOut()
    central_results = []
    frontier_results = []

    for train_idx, test_idx in logo.split(X = X.values, y = y.values, groups = groups):

        ## split data
        X_train = X.iloc[train_idx]
        Z_train = Z.iloc[train_idx]
        y_train = y.iloc[train_idx].values.astype(float)

        X_test = X.iloc[test_idx]
        Z_test = Z.iloc[test_idx]
        y_true = y.iloc[test_idx].values.astype(float)

        group_name = groups[test_idx][0]

        ## stage 1: graph invariants c(x')
        X_train_scaled_df, x_scaler = _standardize_invariants(X_train, struct_feat)
        X_train_scaled = X_train_scaled_df[struct_feat].values.astype(float)
        X_test_scaled = x_scaler.transform(X_test.astype(float))

        model_c = clone(estimator_c)
        model_c.fit(X_train_scaled, y_train)

        c_hat_train = model_c.predict(X_train_scaled).astype(float)
        c_hat_test = model_c.predict(X_test_scaled).astype(float)

        ## stage 2: process signatures r(z')
        Z_train_scaled_df, z_scaler = _standardize_invariants(Z_train, proc_feat)
        Z_train_scaled = Z_train_scaled_df[proc_feat].values.astype(float)
        Z_test_scaled = z_scaler.transform(Z_test.astype(float))

        ## r* = y* - c_hat
        r_star_train = (y_train - c_hat_train).astype(float)

        model_r = clone(estimator_r)
        model_r.fit(Z_train_scaled, r_star_train)

        r_hat_train = model_r.predict(Z_train_scaled).astype(float)
        r_hat_test = model_r.predict(Z_test_scaled).astype(float)

        ## post-fit residuals: eps = y* - (c_hat + r_hat) = r* - r_hat
        e_resid = (r_star_train - r_hat_train).astype(float)

        ## quantile calibration constant q_alpha
        q_alpha = float(np.quantile(e_resid, 1.0 - alpha))

        ## final prediction
        y_pred = (c_hat_test + r_hat_test + q_alpha).astype(float)

        ## metrics
        central = central_metrics(y_true = y_true, y_pred = y_pred)
        frontier = frontier_metrics(y_true = y_true, y_pred = y_pred)

        central_results.append({"group": group_name, **central})
        frontier_results.append({"group": group_name, **frontier})

    return pd.DataFrame(frontier_results), pd.DataFrame(central_results)


## ------------------------------------------------------------------
## standard k-fold cross validation (two-stage orthogonalized)
## ------------------------------------------------------------------
def kfold_cross_valid(
    estimator_c: BaseEstimator,
    estimator_r: BaseEstimator,
    data: pd.DataFrame,
    struct_feat: Sequence[str],
    proc_feat: Sequence[str],
    target: str = "target",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = None,
    alpha: float = 0.01,
) -> pd.DataFrame:

    ## validate inputs
    if not struct_feat:
        raise ValueError("struct_feat must contain at least one column name.")
    if not proc_feat:
        raise ValueError("proc_feat must contain at least one column name.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    ## extract variables
    X = data[struct_feat].apply(pd.to_numeric, errors = "coerce")
    Z = data[proc_feat].apply(pd.to_numeric, errors = "coerce")
    y = _log_transform_target(data[target]).astype(float)

    ## calibration level
    alpha = float(alpha)

    ## cross-validation
    kfold = KFold(
        n_splits = n_splits,
        shuffle = shuffle,
        random_state = random_state,
    )

    central_results = []
    frontier_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X.values), start = 1):

        ## split data
        X_train = X.iloc[train_idx]
        Z_train = Z.iloc[train_idx]
        y_train = y.iloc[train_idx].values.astype(float)

        X_test = X.iloc[test_idx]
        Z_test = Z.iloc[test_idx]
        y_true = y.iloc[test_idx].values.astype(float)

        ## stage 1: graph invariants c(x')
        X_train_scaled_df, x_scaler = _standardize_invariants(X_train, struct_feat)
        X_train_scaled = X_train_scaled_df[struct_feat].values.astype(float)
        X_test_scaled = x_scaler.transform(X_test.astype(float))

        model_c = clone(estimator_c)
        model_c.fit(X_train_scaled, y_train)

        c_hat_train = model_c.predict(X_train_scaled).astype(float)
        c_hat_test = model_c.predict(X_test_scaled).astype(float)

        ## stage 2: process signatures r(z')
        Z_train_scaled_df, z_scaler = _standardize_invariants(Z_train, proc_feat)
        Z_train_scaled = Z_train_scaled_df[proc_feat].values.astype(float)
        Z_test_scaled = z_scaler.transform(Z_test.astype(float))

        r_star_train = (y_train - c_hat_train).astype(float)

        model_r = clone(estimator_r)
        model_r.fit(Z_train_scaled, r_star_train)

        r_hat_train = model_r.predict(Z_train_scaled).astype(float)
        r_hat_test = model_r.predict(Z_test_scaled).astype(float)

        e_resid = (r_star_train - r_hat_train).astype(float)
        q_alpha = float(np.quantile(e_resid, 1.0 - alpha))

        ## final prediction
        y_pred = (c_hat_test + r_hat_test + q_alpha).astype(float)

        ## metrics
        central = central_metrics(y_true = y_true, y_pred = y_pred)
        frontier = frontier_metrics(y_true = y_true, y_pred = y_pred)

        central_results.append({"fold": fold_idx, **central})
        frontier_results.append({"fold": fold_idx, **frontier})

    return pd.DataFrame(frontier_results), pd.DataFrame(central_results)
