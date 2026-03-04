## libraries
import numpy as np
import pandas as pd
from typing import Sequence
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import LeaveOneGroupOut, KFold

## modules
from src.vectorizers.scalers import _log_transformer, _standardizer
from src.evaluators.metrics import frontier_metrics


## ----------------------------------------------------------------------------
## fold-local helpers
## ----------------------------------------------------------------------------
def _drop_nan_rows(
    X: pd.DataFrame,
    Z: pd.DataFrame,
    y: np.ndarray,
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:

    """
    Desc: drop rows containing nans in any feature or target column.
          returns the cleaned arrays and a boolean mask of kept rows.
    Args:
        X: graph invariant features.
        Z: process signature features.
        y: target array (same length as X/Z).
        feat_x: graph invariant column names.
        feat_z: process signature column names.
    Returns:
        tuple of (X_clean, Z_clean, y_clean, kept_mask).
    """

    mask = (
        X[feat_x].notna().all(axis = 1)
        & Z[feat_z].notna().all(axis = 1)
        & ~np.isnan(y)
    )
    return X.loc[mask], Z.loc[mask], y[mask.values], mask.values


## ----------------------------------------------------------------------------
## leave-one-group-out cross validation
## ----------------------------------------------------------------------------
def logo_cross_valid(
    data: pd.DataFrame,
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    estimator_c: BaseEstimator,
    estimator_r: BaseEstimator,
    target: str = "target",
    group: str = "domain",
    random_state: int | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:

    """
    Desc: leave-one-group-out cross validation (retrain). trains and evaluates
          on data using standard logo cv. transforms are always fit on the
          training fold only. estimators are cloned per fold. rows with nans
          are dropped per fold using training-fold-only statistics.
    Args:
        data: training data with features, target, and group columns.
        feat_x: graph invariant feature column names.
        feat_z: process signature feature column names.
        estimator_c: capacity estimator (cloned per fold).
        estimator_r: residual estimator (cloned per fold).
        target: target column name.
        group: group column name for logo splitting.
        random_state: random seed forwarded to cloned estimators when supported.
    Returns:
        tuple of (frontier results dataframe, predicted values array).
    """

    ## validate inputs
    if not feat_x:
        raise ValueError("feat_x must contain at least one column name.")
    if not feat_z:
        raise ValueError("feat_z must contain at least one column name.")

    ## init variables from training data
    X = data[feat_x].apply(pd.to_numeric, errors = "coerce")
    Z = data[feat_z].apply(pd.to_numeric, errors = "coerce")
    y_star = _log_transformer(data[target]).astype(float)
    groups = data[group].values

    ## output array
    y_pred_out = np.full(shape = len(data), fill_value = np.nan)

    ## cross-validation
    logo = LeaveOneGroupOut()
    frontier_results = list()
    for train_idx, test_idx in logo.split(X = X.values, y = y_star.values, groups = groups):

        ## assert single group per fold (logo invariant)
        assert len(np.unique(groups[test_idx])) == 1, "logo fold contains multiple groups"

        ## split training data
        X_train = X.iloc[train_idx]
        Z_train = Z.iloc[train_idx]
        y_train = y_star.iloc[train_idx].values.astype(float)

        ## split test data (same source)
        X_test = X.iloc[test_idx]
        Z_test = Z.iloc[test_idx]
        y_true = y_star.iloc[test_idx].values.astype(float)
        group_name = groups[test_idx][0]

        ## drop rows with nans (fold-local)
        X_train, Z_train, y_train, _ = _drop_nan_rows(
            X = X_train, Z = Z_train, y = y_train,
            feat_x = feat_x, feat_z = feat_z
        )
        X_test, Z_test, y_true, kept_test = _drop_nan_rows(
            X = X_test, Z = Z_test, y = y_true,
            feat_x = feat_x, feat_z = feat_z
        )

        if len(X_train) < 2 or len(X_test) == 0:
            continue

        ## enforce column order before scaling
        X_test = X_test[feat_x]
        Z_test = Z_test[feat_z]

        ## standardize graph invariants (fit on train only)
        X_train_scaled, x_scaler = _standardizer(X_train, feat_x)
        X_train_scaled = X_train_scaled[feat_x].values.astype(float)
        X_test_scaled = x_scaler.transform(X_test.astype(float))

        ## ensure fresh parameters for each fold
        model_c = clone(estimator_c)
        model_r = clone(estimator_r)

        ## set random_state on estimators if supported
        if random_state is not None:
            for m in (model_c, model_r):
                if hasattr(m, "random_state"):
                    m.set_params(random_state = random_state)

        ## train C on graph invariants
        model_c.fit(X_train_scaled, y_train)
        c_hat_train = model_c.predict(X_train_scaled).astype(float)
        c_hat_test = model_c.predict(X_test_scaled).astype(float)

        ## standardize process signatures (fit on train only)
        Z_train_scaled, z_scaler = _standardizer(Z_train, feat_z)
        Z_train_scaled = Z_train_scaled[feat_z].values.astype(float)
        Z_test_scaled = z_scaler.transform(Z_test.astype(float))

        ## signed slack in log-space (interpretable as log R)
        slack_train = (y_train - c_hat_train).astype(float)

        ## train R on process signatures and residualized target
        model_r.fit(Z_train_scaled, slack_train)
        r_hat_train = model_r.predict(Z_train_scaled).astype(float)
        r_hat_test = model_r.predict(Z_test_scaled).astype(float)

        ## identifiability: force zero-mean log-factor
        r_hat_test = (r_hat_test - np.mean(r_hat_train)).astype(float)

        ## final prediction: y* = log C + log R + epsilon
        y_pred = (c_hat_test + r_hat_test).astype(float)

        ## map predictions back to original indices (accounting for nan drops)
        kept_indices = test_idx[kept_test]
        y_pred_out[kept_indices] = y_pred

        ## compute frontier metrics for this fold
        frontier = frontier_metrics(y_true = y_true, y_pred = y_pred)
        frontier_results.append({"group": group_name, **frontier})

    return pd.DataFrame(frontier_results), y_pred_out


## ----------------------------------------------------------------------------
## frozen-manifold leave-one-group-out cv
## ----------------------------------------------------------------------------
def logo_cross_valid_frozen(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    estimator_c: BaseEstimator,
    estimator_r: BaseEstimator,
    target: str = "target",
    group: str = "domain",
    ) -> tuple[pd.DataFrame, np.ndarray, dict]:

    """
    Desc: frozen-manifold leave-one-group-out cross validation. trains on
          data_train and evaluates on data_test, isolating geometric sensitivity
          from estimator adaptation. all preprocessing (log transform,
          standardization) is fit exclusively on the training fold and applied
          to data_test without refitting. estimators are cloned per fold.
          rows with nans are dropped per fold using training-fold-only
          statistics.
    Args:
        data_train: clean training data with features, target, and group columns.
        data_test: perturbed evaluation data with the same schema.
        feat_x: graph invariant feature column names.
        feat_z: process signature feature column names.
        estimator_c: capacity estimator (cloned per fold).
        estimator_r: residual estimator (cloned per fold).
        target: target column name.
        group: group column name for logo splitting.
    Returns:
        tuple of (frontier results dataframe, predicted values array,
        metadata dict with n_groups_evaluated, n_groups_skipped, and
        groups_evaluated).
    """

    ## validate inputs
    if not feat_x:
        raise ValueError("feat_x must contain at least one column name.")
    if not feat_z:
        raise ValueError("feat_z must contain at least one column name.")

    ## init variables from training data
    X = data_train[feat_x].apply(pd.to_numeric, errors = "coerce")
    Z = data_train[feat_z].apply(pd.to_numeric, errors = "coerce")
    y_star = _log_transformer(data_train[target]).astype(float)
    groups = data_train[group].values

    ## output array sized to evaluation data
    y_pred_out = np.full(shape = len(data_test), fill_value = np.nan)

    ## cross-validation
    logo = LeaveOneGroupOut()
    frontier_results = list()
    groups_evaluated = list()
    groups_skipped = list()
    for train_idx, test_idx in logo.split(X = X.values, y = y_star.values, groups = groups):

        ## assert single group per fold (logo invariant)
        assert len(np.unique(groups[test_idx])) == 1, "logo fold contains multiple groups"

        ## split training data (always from clean data)
        X_train = X.iloc[train_idx]
        Z_train = Z.iloc[train_idx]
        y_train = y_star.iloc[train_idx].values.astype(float)

        ## identify held-out group
        held_out_group = groups[test_idx][0]

        ## locate held-out group in perturbed test data
        test_mask = data_test[group] == held_out_group
        if test_mask.sum() == 0:
            groups_skipped.append(held_out_group)
            continue

        ## split test data (perturbed)
        X_test = data_test[feat_x].loc[test_mask].apply(pd.to_numeric, errors = "coerce")
        Z_test = data_test[feat_z].loc[test_mask].apply(pd.to_numeric, errors = "coerce")
        y_test_raw = _log_transformer(data_test[target].loc[test_mask]).astype(float).values
        group_name = held_out_group

        ## drop nan rows from training fold (fold-local)
        X_train, Z_train, y_train, _ = _drop_nan_rows(
            X = X_train, Z = Z_train, y = y_train,
            feat_x = feat_x, feat_z = feat_z
        )

        ## drop nan rows from test fold (perturbed data)
        test_mask_indices = np.where(test_mask.values)[0]
        X_test_df = X_test.copy()
        Z_test_df = Z_test.copy()
        X_test_df, Z_test_df, y_true, kept_test = _drop_nan_rows(
            X = X_test_df, Z = Z_test_df, y = y_test_raw,
            feat_x = feat_x, feat_z = feat_z
        )

        if len(X_train) < 2 or len(X_test_df) == 0:
            groups_skipped.append(held_out_group)
            continue

        ## enforce column order before scaling
        X_test_df = X_test_df[feat_x]
        Z_test_df = Z_test_df[feat_z]

        ## standardize graph invariants (fit on train only)
        X_train_scaled, x_scaler = _standardizer(X_train, feat_x)
        X_train_scaled = X_train_scaled[feat_x].values.astype(float)
        X_test_scaled = x_scaler.transform(X_test_df.astype(float))

        ## ensure fresh parameters for each fold
        model_c = clone(estimator_c)
        model_r = clone(estimator_r)

        ## train C on graph invariants
        model_c.fit(X_train_scaled, y_train)
        c_hat_train = model_c.predict(X_train_scaled).astype(float)
        c_hat_test = model_c.predict(X_test_scaled).astype(float)

        ## standardize process signatures (fit on train only)
        Z_train_scaled, z_scaler = _standardizer(Z_train, feat_z)
        Z_train_scaled = Z_train_scaled[feat_z].values.astype(float)
        Z_test_scaled = z_scaler.transform(Z_test_df.astype(float))

        ## signed slack in log-space (interpretable as log R)
        slack_train = (y_train - c_hat_train).astype(float)

        ## train R on process signatures and residualized target
        model_r.fit(Z_train_scaled, slack_train)
        r_hat_train = model_r.predict(Z_train_scaled).astype(float)
        r_hat_test = model_r.predict(Z_test_scaled).astype(float)

        ## identifiability: force zero-mean log-factor
        r_hat_test = (r_hat_test - np.mean(r_hat_train)).astype(float)

        ## final prediction: y* = log C + log R + epsilon
        y_pred = (c_hat_test + r_hat_test).astype(float)

        ## map predictions back to original indices (accounting for nan drops)
        kept_global = test_mask_indices[kept_test]
        y_pred_out[kept_global] = y_pred

        ## compute frontier metrics for this fold
        frontier = frontier_metrics(y_true = y_true, y_pred = y_pred)
        frontier_results.append({"group": group_name, **frontier})
        groups_evaluated.append(group_name)

    ## metadata for group coverage tracking
    metadata = {
        "n_groups_evaluated": len(groups_evaluated),
        "n_groups_skipped": len(groups_skipped),
        "groups_evaluated": sorted(groups_evaluated),
        "groups_skipped": sorted(groups_skipped),
    }

    return pd.DataFrame(frontier_results), y_pred_out, metadata


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
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int | None = None
    ) -> tuple[pd.DataFrame, np.ndarray]:

    """
    Desc: k-fold cross validation. transforms are fit on the training fold
          only. estimators are cloned per fold. rows with nans are dropped
          per fold using training-fold-only statistics.
    Args:
        data: training data with features, target, and group columns.
        feat_x: graph invariant feature column names.
        feat_z: process signature feature column names.
        estimator_c: capacity estimator (cloned per fold).
        estimator_r: residual estimator (cloned per fold).
        target: target column name.
        n_splits: number of folds.
        shuffle: whether to shuffle before splitting.
        random_state: random seed for reproducibility.
    Returns:
        tuple of (frontier results dataframe, predicted values array).
    """

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
    frontier_results = list()
    y_pred_test = np.full(shape = len(data), fill_value = np.nan)
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X.values), start = 1):

        ## split training data
        X_train = X.iloc[train_idx]
        Z_train = Z.iloc[train_idx]
        y_train = y_star.iloc[train_idx].values.astype(float)

        ## split test data
        X_test = X.iloc[test_idx]
        Z_test = Z.iloc[test_idx]
        y_true = y_star.iloc[test_idx].values.astype(float)

        ## drop rows with nans (fold-local)
        X_train, Z_train, y_train, _ = _drop_nan_rows(
            X = X_train, Z = Z_train, y = y_train,
            feat_x = feat_x, feat_z = feat_z
        )
        X_test, Z_test, y_true, kept_test = _drop_nan_rows(
            X = X_test, Z = Z_test, y = y_true,
            feat_x = feat_x, feat_z = feat_z
        )

        if len(X_train) < 2 or len(X_test) == 0:
            continue

        ## enforce column order before scaling
        X_test = X_test[feat_x]
        Z_test = Z_test[feat_z]

        ## standardize graph invariants (fit on train only)
        X_train_scaled, x_scaler = _standardizer(X_train, feat_x)
        X_train_scaled = X_train_scaled[feat_x].values.astype(float)
        X_test_scaled = x_scaler.transform(X_test.astype(float))

        ## ensure fresh parameters for each fold
        model_c = clone(estimator_c)
        model_r = clone(estimator_r)

        ## train C on graph invariants
        model_c.fit(X_train_scaled, y_train)
        c_hat_train = model_c.predict(X_train_scaled).astype(float)
        c_hat_test = model_c.predict(X_test_scaled).astype(float)

        ## standardize process signatures (fit on train only)
        Z_train_scaled, z_scaler = _standardizer(Z_train, feat_z)
        Z_train_scaled = Z_train_scaled[feat_z].values.astype(float)
        Z_test_scaled = z_scaler.transform(Z_test.astype(float))

        ## signed slack in log-space (interpretable as log R)
        slack_train = (y_train - c_hat_train).astype(float)

        ## train R on all observations
        model_r.fit(Z_train_scaled, slack_train)
        r_hat_train = model_r.predict(Z_train_scaled).astype(float)
        r_hat_test = model_r.predict(Z_test_scaled).astype(float)

        ## identifiability: force zero-mean log-factor
        r_hat_test = (r_hat_test - np.mean(r_hat_train)).astype(float)

        ## final prediction: y* = log C + log R + epsilon
        y_pred = (c_hat_test + r_hat_test).astype(float)

        ## map predictions back to original indices (accounting for nan drops)
        kept_indices = test_idx[kept_test]
        y_pred_test[kept_indices] = y_pred

        ## compute frontier metrics for this fold
        frontier = frontier_metrics(y_true = y_true, y_pred = y_pred)
        frontier_results.append(frontier)

    return pd.DataFrame(frontier_results), y_pred_test
