## libraries
import numpy as np
import pandas as pd
from typing import Sequence
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import LeaveOneGroupOut, KFold

## modules
from src.scalers import _log_transform_target, _standardize_invariants
from src.metrics import frontier_metrics, central_metrics

## leave-one-group-out cross validation
def logo_cross_valid(
    estimator: BaseEstimator,
    data: pd.DataFrame,
    feat: Sequence[str],
    target: str = "target",
    group: str = "domain"
    ) -> pd.DataFrame:
    
    ## validate inputs and pre-process data
    if not feat:
        raise ValueError("feat argument must contain at least one column name.")

    X = data[feat].apply(pd.to_numeric, errors = "coerce")
    y = data[target]
    groups = data[group].values
    
    ## conduct training and evaluation across groups
    logo = LeaveOneGroupOut()
    central_results = list()
    frontier_results = list()
    for train_idx, test_idx in logo.split(X = X.values, y = y.values, groups = groups):
        
        ## split data
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx].values

        X_test = X.iloc[test_idx]
        y_true = y.iloc[test_idx].values

        ## all test samples from same group
        group_name = groups[test_idx][0]  
        
        ## standardize invariants on training fold only
        X_train_scaled, scaler = _standardize_invariants(X_train, feat)
        X_train_scaled = X_train_scaled[feat].values
        X_test_scaled = scaler.transform(X_test.astype(float))
        
        ## fit model on log-transformed targets
        model = clone(estimator = estimator)
        model.fit(X_train_scaled, y_train)

        ## compute maximum residual on training fold
        y_pred_train = model.predict(X_train_scaled)
        residuals = y_train - y_pred_train
        vr_target = 0.05
        q = 1.0 - vr_target  ## 99% quantile for frontier calibration
        c = float(np.quantile(residuals, q))

        ## predict with frontier calibration
        y_pred = model.predict(X_test_scaled) + c
        
        ## capture metrics in log-space
        central = central_metrics(y_true = y_true, y_pred = y_pred)
        frontier = frontier_metrics(y_true = y_true, y_pred = y_pred)

        central_results.append({
            "group": group_name,
            **central,
        })
        frontier_results.append({
            "group": group_name,
            **frontier,
        })
    
    return pd.DataFrame(frontier_results), pd.DataFrame(central_results)


## standard k-fold cross validation
def kfold_cross_valid(
    estimator: BaseEstimator,
    data: pd.DataFrame,
    feat: Sequence[str],
    target: str = "target",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = None,
    ) -> pd.DataFrame:

    ## validate inputs and pre-process data
    if not feat:
        raise ValueError("feat argument must contain at least one column name.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for k-fold cross validation.")

    X = data[feat].apply(pd.to_numeric, errors = "coerce")
    y = _log_transform_target(data[target])

    ## conduct training and evaluation across folds
    kfold = KFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
    central_results = list()
    frontier_results = list()
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X.values), start = 1):

        ## split data
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx].values

        X_test = X.iloc[test_idx]
        y_true = y.iloc[test_idx].values

        ## standardize invariants on training fold only
        X_train_scaled, scaler = _standardize_invariants(X_train, feat)
        X_train_scaled = X_train_scaled[feat].values
        X_test_scaled = scaler.transform(X_test.astype(float))

        ## fit model on log-transformed targets
        model = clone(estimator = estimator)
        model.fit(X_train_scaled, y_train)

        ## compute maximum residual on training fold
        y_pred_train = model.predict(X_train_scaled)
        residuals = y_train - y_pred_train
        vr_target = 0.01
        q = 1.0 - vr_target  ## 99% quantile for frontier calibration
        c = float(np.quantile(residuals, q))

        ## predict with frontier calibration
        y_pred = model.predict(X_test_scaled) + c

        ## capture metrics in log-space
        central = central_metrics(y_true = y_true, y_pred = y_pred)
        frontier = frontier_metrics(y_true = y_true, y_pred = y_pred)

        central_results.append({
            "fold": fold_idx,
            **central,
        })
        frontier_results.append({
            "fold": fold_idx,
            **frontier,
        })

    return pd.DataFrame(frontier_results), pd.DataFrame(central_results)