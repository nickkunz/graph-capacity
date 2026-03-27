## libraries
import numpy as np
import pandas as pd
from typing import Sequence
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import LeaveOneGroupOut, KFold, RepeatedKFold

## modules
from src.evaluators.metrics import frontier_metrics
from src.vectorizers.scalers import _log_transformer, _standardizer

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
    Desc:
        Omits rows containing nans in any feature or target column.
        Returns the cleaned arrays and a boolean mask of kept rows.

    Args:
        X: graph invariant features.
        Z: process signature features.
        y: target array (same length as X/Z).
        feat_x: graph invariant column names.
        feat_z: process signature column names.
    
    Returns:
        tuple of (X_clean, Z_clean, y_clean, kept_mask).

    Raises:
        ValueError if feature lists are empty or fold-local inputs are inconsistent.
    """

    ## validate inputs
    if not feat_x:
        raise ValueError("feat_x must contain at least one column name.")
    if not feat_z:
        raise ValueError("feat_z must contain at least one column name.")
    if len(X) != len(Z) or len(X) != len(y):
        raise ValueError("X, Z, and y must have the same length.")

    ## create mask of rows without nans in any feature or target column
    mask = (
        X[feat_x].notna().all(axis = 1)
        & Z[feat_z].notna().all(axis = 1)
        & ~np.isnan(y)
    )
    return X.loc[mask], Z.loc[mask], y[mask.values], mask.values


## ----------------------------------------------------------------------------
## retrained-manifold worker
## ----------------------------------------------------------------------------
def _run_retrain_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: pd.DataFrame,
    Z: pd.DataFrame,
    y_star: pd.Series,
    feat_x: list[str],
    feat_z: list[str],
    estimator_c: BaseEstimator,
    estimator_r: BaseEstimator,
    random_state: int = 42,
    group_name: str | None = None,
    ) -> dict | None:

    """
    Desc: 
        Execute a single retrained f_C(X) + f_R(Z) cross-validation fold.
        Retrains capacity and residual estimators on the training split and 
        evaluates frontier metrics on the test split.

    Args:
        train_idx: training indices.
        test_idx: test indices.
        X: graph invariant feature dataframe.
        Z: process signature feature dataframe.
        y_star: log-transformed target series.
        feat_x: graph invariant column names.
        feat_z: process signature column names.
        estimator_c: capacity estimator (cloned internally).
        estimator_r: residual estimator (cloned internally).
        random_state: random seed forwarded to estimators when supported.
        group_name: optional group label for the fold.
    
    Returns:
        dict with frontier metrics, predictions, and index mapping,
        or none if the fold is skipped due to insufficient data.

    Raises:
        ValueError propagated from _drop_nan_rows if fold-local inputs are
        inconsistent or if feat_x/feat_z are empty.
    """

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
        return None

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

    ## compute frontier metrics for this fold
    kept_indices = test_idx[kept_test]
    frontier = frontier_metrics(y_true = y_true, y_pred = y_pred)

    return {
        "group_name": group_name,
        "frontier": frontier,
        "kept_indices": kept_indices,
        "y_pred": y_pred,
    }

## ----------------------------------------------------------------------------
## frozen-manifold worker
## ----------------------------------------------------------------------------
def _run_frozen_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: pd.DataFrame,
    Z: pd.DataFrame,
    y_star: pd.Series,
    groups: np.ndarray,
    data_test: pd.DataFrame,
    feat_x: list[str],
    feat_z: list[str],
    estimator_c: BaseEstimator,
    estimator_r: BaseEstimator,
    target: str = "target",
    group: str = "domain",
    random_state: int = 42,
    ) -> dict:

    """
    Desc: Execute a single frozen-manifold logo fold. Trains on clean
          training data and evaluates on perturbed test data for the
          held-out group.

    Args:
        train_idx: Training indices into X/Z/y_star.
        test_idx: Test indices (used for group identification).
        X: Graph invariant features from training data.
        Z: Process signature features from training data.
        y_star: Log-transformed target from training data.
        groups: Group labels from training data.
        data_test: Perturbed evaluation dataframe with same schema.
        feat_x: Graph invariant column names.
        feat_z: Process signature column names.
        estimator_c: Capacity estimator (cloned internally).
        estimator_r: Residual estimator (cloned internally).
        target: Target column name.
        group: Group column name.

    Returns:
        dict with frontier metrics, predictions, and index mapping,
        or dict with skipped group name if fold cannot be evaluated.

    Raises:
        ValueError propagated from _drop_nan_rows if fold-local inputs are
        inconsistent or if feat_x/feat_z are empty.
    """

    ## split training data (always from clean data)
    X_train = X.iloc[train_idx]
    Z_train = Z.iloc[train_idx]
    y_train = y_star.iloc[train_idx].values.astype(float)

    ## identify held-out group
    held_out_group = groups[test_idx][0]

    ## locate held-out group in perturbed test data
    test_mask = data_test[group] == held_out_group
    if test_mask.sum() == 0:
        return {"skipped": held_out_group}

    ## split test data (perturbed)
    X_test = data_test[feat_x].loc[test_mask].apply(pd.to_numeric, errors = "coerce")
    Z_test = data_test[feat_z].loc[test_mask].apply(pd.to_numeric, errors = "coerce")
    y_test_raw = _log_transformer(data_test[target].loc[test_mask]).astype(float).values

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
        return {"skipped": held_out_group}

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

    ## compute frontier metrics for this fold
    kept_global = test_mask_indices[kept_test]
    frontier = frontier_metrics(y_true = y_true, y_pred = y_pred)

    return {
        "group_name": held_out_group,
        "frontier": frontier,
        "kept_indices": kept_global,
        "y_pred": y_pred,
    }

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
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
    ) -> tuple[pd.DataFrame, np.ndarray]:

    """
    Desc: Leave-one-group-out cross validation (retrain). Trains and evaluates
          on data using standard logo cv. When n_repeats > 1, the full logo
          procedure is repeated with different random seeds to average out
          stochastic learner variance; frontier metrics are averaged per group
          across repeats. Transforms are always fit on the training fold only.
          Estimators are cloned per fold. Rows with nans are dropped per fold
          using training-fold-only statistics.
    Args:
        data: Training data with features, target, and group columns.
        feat_x: Graph invariant feature column names.
        feat_z: Process signature feature column names.
        estimator_c: Capacity estimator (cloned per fold).
        estimator_r: Residual estimator (cloned per fold).
        target: Target column name.
        group: Group column name for logo splitting.
        n_repeats: Number of times to repeat the full logo procedure with
            different random seeds. Defaults to 1 (single pass).
        random_state: Random seed forwarded to cloned estimators when supported.
            Each repeat offsets the seed by the repeat index.
        n_jobs: Number of parallel jobs for fold execution (-1 for all cores).
    Returns:
        Tuple of (frontier results dataframe, predicted values array).
        Frontier metrics are averaged across repeats per group.

    Raises:
        ValueError if feat_x or feat_z is empty.
        ValueError if n_repeats < 1.
        AssertionError if any logo test fold contains multiple groups.
        ValueError propagated from _drop_nan_rows if fold-local inputs are
        inconsistent.
    """

    ## validate inputs
    if not feat_x:
        raise ValueError("feat_x must contain at least one column name.")
    if not feat_z:
        raise ValueError("feat_z must contain at least one column name.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    ## init variables from training data
    feat_x = list(feat_x)
    feat_z = list(feat_z)
    X = data[feat_x].apply(pd.to_numeric, errors = "coerce")
    Z = data[feat_z].apply(pd.to_numeric, errors = "coerce")
    y_star = _log_transformer(data[target]).astype(float)
    groups = data[group].values

    ## cross-validation splits (fixed across repeats)
    logo = LeaveOneGroupOut()
    fold_splits = list(logo.split(X = X.values, y = y_star.values, groups = groups))

    ## validate groups
    for _, test_idx in fold_splits:
        assert len(np.unique(groups[test_idx])) == 1, "logo fold contains multiple groups"

    ## repeat seed per iteration
    def _repeat_seed(repeat: int) -> int | None:
        if random_state is None:
            return None
        return random_state + repeat

    ## parallel fold execution across all repeats
    fold_results = Parallel(n_jobs = n_jobs)(
        delayed(_run_retrain_fold)(
            train_idx = train_idx,
            test_idx = test_idx,
            X = X,
            Z = Z,
            y_star = y_star,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = estimator_c,
            estimator_r = estimator_r,
            random_state = _repeat_seed(repeat),
            group_name = groups[test_idx][0],
        )
        for repeat in range(n_repeats)
        for train_idx, test_idx in fold_splits
    )

    ## accumulate predictions and frontier metrics across repeats
    y_pred_sum = np.zeros(shape = len(data), dtype = float)
    y_pred_count = np.zeros(shape = len(data), dtype = int)
    group_frontiers: dict[str, list[dict]] = dict()

    for result in fold_results:
        if result is None:
            continue
        y_pred_sum[result["kept_indices"]] += result["y_pred"]
        y_pred_count[result["kept_indices"]] += 1
        grp = result["group_name"]
        if grp not in group_frontiers:
            group_frontiers[grp] = list()
        group_frontiers[grp].append(result["frontier"])

    ## average frontier metrics per group across repeats
    frontier_results = list()
    for grp in sorted(group_frontiers.keys()):
        metrics_avg = pd.DataFrame(group_frontiers[grp]).mean().to_dict()
        frontier_results.append({"group": grp, **metrics_avg})

    ## average predictions across repeats
    y_pred_out = np.full(shape = len(data), fill_value = np.nan)
    valid_pred = y_pred_count > 0
    y_pred_out[valid_pred] = y_pred_sum[valid_pred] / y_pred_count[valid_pred]

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
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
    ) -> tuple[pd.DataFrame, np.ndarray, dict]:

    """
    Desc: Frozen-manifold leave-one-group-out cross validation. Trains on
          data_train and evaluates on data_test, isolating geometric sensitivity
          from estimator adaptation. When n_repeats > 1, the full logo
          procedure is repeated with different random seeds to average out
          stochastic learner variance; frontier metrics are averaged per group
          across repeats. All preprocessing (log transform, standardization)
          is fit exclusively on the training fold and applied to data_test
          without refitting. Estimators are cloned per fold. Rows with nans
          are dropped per fold using training-fold-only statistics.
    
    Args:
        data_train: Clean training data with features, target, and group columns.
        data_test: Perturbed evaluation data with the same schema.
        feat_x: Graph invariant feature column names.
        feat_z: Process signature feature column names.
        estimator_c: Capacity estimator (cloned per fold).
        estimator_r: Residual estimator (cloned per fold).
        target: Target column name.
        group: Group column name for logo splitting.
        n_repeats: Number of times to repeat the full logo procedure with
            different random seeds. Defaults to 10.
        random_state: Random seed forwarded to cloned estimators when supported.
            Each repeat offsets the seed by the repeat index.
        n_jobs: Number of parallel jobs for fold execution (-1 for all cores).
    
    Returns:
        Tuple of (frontier results dataframe, predicted values array,
        metadata dict with n_groups_evaluated, n_groups_skipped, and
        groups_evaluated). Frontier metrics are averaged across repeats
        per group.

    Raises:
        ValueError if feat_x or feat_z is empty.
        ValueError if n_repeats < 1.
        AssertionError if any logo test fold contains multiple groups.
        ValueError propagated from _drop_nan_rows if fold-local inputs are
        inconsistent.
    """

    ## validate inputs
    if not feat_x:
        raise ValueError("feat_x must contain at least one column name.")
    if not feat_z:
        raise ValueError("feat_z must contain at least one column name.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    ## init variables from training data
    feat_x = list(feat_x)
    feat_z = list(feat_z)
    X = data_train[feat_x].apply(pd.to_numeric, errors = "coerce")
    Z = data_train[feat_z].apply(pd.to_numeric, errors = "coerce")
    y_star = _log_transformer(data_train[target]).astype(float)
    groups = data_train[group].values

    ## cross-validation splits (fixed across repeats)
    logo = LeaveOneGroupOut()
    fold_splits = list(logo.split(X = X.values, y = y_star.values, groups = groups))

    ## validate groups
    for _, test_idx in fold_splits:
        assert len(np.unique(groups[test_idx])) == 1, "logo fold contains multiple groups"

    ## repeat seed per iteration
    def _repeat_seed(repeat: int) -> int | None:
        if random_state is None:
            return None
        return random_state + repeat

    ## parallel fold execution across all repeats
    fold_results = Parallel(n_jobs = n_jobs)(
        delayed(_run_frozen_fold)(
            train_idx = train_idx,
            test_idx = test_idx,
            X = X,
            Z = Z,
            y_star = y_star,
            groups = groups,
            data_test = data_test,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = estimator_c,
            estimator_r = estimator_r,
            target = target,
            group = group,
            random_state = _repeat_seed(repeat),
        )
        for repeat in range(n_repeats)
        for train_idx, test_idx in fold_splits
    )

    ## accumulate predictions and frontier metrics across repeats
    y_pred_sum = np.zeros(shape = len(data_test), dtype = float)
    y_pred_count = np.zeros(shape = len(data_test), dtype = int)
    group_frontiers: dict[str, list[dict]] = dict()
    groups_evaluated_set: set[str] = set()
    groups_skipped_set: set[str] = set()

    for result in fold_results:
        if "skipped" in result:
            groups_skipped_set.add(result["skipped"])
            continue
        y_pred_sum[result["kept_indices"]] += result["y_pred"]
        y_pred_count[result["kept_indices"]] += 1
        grp = result["group_name"]
        groups_evaluated_set.add(grp)
        if grp not in group_frontiers:
            group_frontiers[grp] = list()
        group_frontiers[grp].append(result["frontier"])

    ## average frontier metrics per group across repeats
    frontier_results = list()
    for grp in sorted(group_frontiers.keys()):
        metrics_avg = pd.DataFrame(group_frontiers[grp]).mean().to_dict()
        frontier_results.append({"group": grp, **metrics_avg})

    ## average predictions across repeats
    y_pred_out = np.full(shape = len(data_test), fill_value = np.nan)
    valid_pred = y_pred_count > 0
    y_pred_out[valid_pred] = y_pred_sum[valid_pred] / y_pred_count[valid_pred]

    ## metadata for group coverage tracking
    metadata = {
        "n_groups_evaluated": len(groups_evaluated_set),
        "n_groups_skipped": len(groups_skipped_set - groups_evaluated_set),
        "groups_evaluated": sorted(groups_evaluated_set),
        "groups_skipped": sorted(groups_skipped_set - groups_evaluated_set),
    }

    return pd.DataFrame(frontier_results), y_pred_out, metadata

## ----------------------------------------------------------------------------
## k-fold cross validation
## ----------------------------------------------------------------------------
def kfold_cross_valid(
    data: pd.DataFrame,
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    estimator_c: BaseEstimator,
    estimator_r: BaseEstimator,
    target: str = "target",
    n_splits: int = 10,
    n_repeats: int = 10,
    shuffle: bool = True,
    random_state: int = 42,
    n_jobs: int = -1,
    detail: bool = False,
    ) -> tuple[pd.DataFrame, np.ndarray]:

    """
    Desc: k-fold cross validation. Transforms are fit on the training fold
          only. Estimators are cloned per fold. Rows with nans are dropped
          per fold using training-fold-only statistics.
    Args:
        data: Training data with features, target, and group columns.
        feat_x: Graph invariant feature column names.
        feat_z: Process signature feature column names.
        estimator_c: Capacity estimator (cloned per fold).
        estimator_r: Residual estimator (cloned per fold).
        target: Target column name.
        n_splits: Number of folds.
        n_repeats: Number of repeated k-fold iterations.
        shuffle: Whether to shuffle before splitting.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs for fold execution (-1 for all cores).
        detail: If true, return one row per iteration. If false (default),
                return a single row with metrics averaged across iterations.
    Returns:
        Tuple of (frontier results dataframe, predicted values array).
        When detail is false, the frontier dataframe contains a single row
        with metrics averaged across all iterations. When detail is true,
        it contains one row per iteration with fold-averaged metrics.
        The prediction array is averaged across iterations in both cases.

    Raises:
        ValueError if feat_x or feat_z is empty.
        ValueError if n_splits < 2 or n_repeats < 1.
        ValueError if n_repeats > 1 and shuffle is False.
        ValueError propagated from _drop_nan_rows if fold-local inputs are
        inconsistent.
    """

    ## validate inputs
    if not feat_x:
        raise ValueError("feat_x must contain at least one column name.")
    if not feat_z:
        raise ValueError("feat_z must contain at least one column name.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    if n_repeats > 1 and not shuffle:
        raise ValueError("n_repeats > 1 requires shuffle = True to change the splits.")

    ## extract variables
    feat_x = list(feat_x)
    feat_z = list(feat_z)
    X = data[feat_x].apply(pd.to_numeric, errors = "coerce")
    Z = data[feat_z].apply(pd.to_numeric, errors = "coerce")
    y_star = _log_transformer(data[target]).astype(float)

    ## create splitter
    if n_repeats > 1:
        splitter = RepeatedKFold(
            n_splits = n_splits,
            n_repeats = n_repeats,
            random_state = random_state,
        )
    else:
        splitter = KFold(
            n_splits = n_splits,
            shuffle = shuffle,
            random_state = random_state,
        )

    fold_splits = list(splitter.split(X.values))

    ## iteration seed per fold for estimator random_state
    def _iter_seed(fold_id):
        if random_state is None:
            return None
        return random_state + fold_id // n_splits

    ## parallel fold execution (all folds dispatched at once)
    fold_results = Parallel(n_jobs = n_jobs)(
        delayed(_run_retrain_fold)(
            train_idx = train_idx,
            test_idx = test_idx,
            X = X,
            Z = Z,
            y_star = y_star,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = estimator_c,
            estimator_r = estimator_r,
            random_state = _iter_seed(fold_id),
        )
        for fold_id, (train_idx, test_idx) in enumerate(fold_splits)
    )

    ## accumulate predictions and group frontier dicts by iteration
    y_pred_sum = np.zeros(shape = len(data), dtype = float)
    y_pred_count = np.zeros(shape = len(data), dtype = int)
    iteration_frontiers = dict()

    for fold_id, result in enumerate(fold_results):
        if result is None:
            continue
        iteration = fold_id // n_splits
        y_pred_sum[result["kept_indices"]] += result["y_pred"]
        y_pred_count[result["kept_indices"]] += 1

        if iteration not in iteration_frontiers:
            iteration_frontiers[iteration] = list()
        iteration_frontiers[iteration].append(result["frontier"])

    ## build frontier results (one row per iteration)
    frontier_results = list()
    for iteration in sorted(iteration_frontiers.keys()):
        folds = iteration_frontiers[iteration]
        frontier_mean = pd.DataFrame(folds).mean().to_dict()
        frontier_results.append({
            "iteration": iteration + 1,
            "n_folds_used": len(folds),
            **frontier_mean,
        })

    ## average predictions over repeated out-of-fold appearances
    y_pred_out = np.full(shape = len(data), fill_value = np.nan)
    valid_pred = y_pred_count > 0
    y_pred_out[valid_pred] = y_pred_sum[valid_pred] / y_pred_count[valid_pred]

    ## build frontier dataframe
    frontier_df = pd.DataFrame(frontier_results)

    ## aggregate to a single row unless detailed per-iteration output requested
    if not detail and len(frontier_df) > 0:
        metric_cols = [c for c in frontier_df.columns if c not in ("iteration", "n_folds_used")]
        frontier_df = pd.DataFrame([frontier_df[metric_cols].mean().to_dict()])

    return frontier_df, y_pred_out
