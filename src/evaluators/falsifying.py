## libraries
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Sequence
from pathlib import Path
from itertools import combinations
from joblib import Parallel, delayed

## path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.vectorizers.scalers import _log_transformer
from src.evaluators.metrics import consensus_metrics, FRONTIER_METRICS, CONSENSUS_METRICS
from src.evaluators.resampling import logo_cross_valid, logo_cross_valid_frozen


## ----------------------------------------------------------------------
## frontier falsifiability test
## ----------------------------------------------------------------------
def eval_falsified_frontier(
    data: pd.DataFrame,
    falsified: dict[str, pd.DataFrame],
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_jobs: int = -1,
    protocol: str = "frozen",
    ) -> pd.DataFrame:
    """
    Desc:
        Test whether original data produces better frontier envelope metrics
        than falsified data under the specified protocol.
    Args:
        data: clean evaluation dataframe.
        falsified: mapping of falsification method to dataframe.
        models: mapping of model name to estimator bundle.
        feat_x: graph invariant column names.
        feat_z: process signature column names.
        target: target column name.
        group: group column name for logo splitting.
        n_jobs: number of parallel jobs.
        protocol: "frozen" or "retrain".
    Returns:
        dataframe with frontier metrics per (model, method, condition, group).
    Raises:
        ValueError if protocol is not "frozen" or "retrain".
    """

    ## init feature lists as mutable for parallel jobs
    feat_x = list(feat_x)
    feat_z = list(feat_z)
    model_names = list(models.keys())

    ## original-data cv: once per model
    print(f"Running {len(model_names)} original + {len(model_names) * len(falsified)} false frontier jobs ({protocol})...")
    real_results = Parallel(n_jobs = n_jobs)(
        delayed(logo_cross_valid)(
            data = data,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = models[name].estimator_c,
            estimator_r = models[name].estimator_r,
            target = target,
            group = group,
            n_jobs = 1,
        )
        for name in model_names
    )
    real_cv = dict(zip(model_names, real_results))

    ## falsified-data cv: per (model, method)
    false_jobs = [
        (model_name, method_name, data_false)
        for method_name, data_false in falsified.items()
        for model_name in model_names
    ]

    if protocol == "retrain":
        false_results = Parallel(n_jobs = n_jobs)(
            delayed(logo_cross_valid)(
                data = data_false,
                feat_x = feat_x,
                feat_z = feat_z,
                estimator_c = models[model_name].estimator_c,
                estimator_r = models[model_name].estimator_r,
                target = target,
                group = group,
                n_jobs = 1,
            )
            for model_name, _, data_false in false_jobs
        )
    elif protocol == "frozen":
        false_results = Parallel(n_jobs = n_jobs)(
            delayed(logo_cross_valid_frozen)(
                data_train = data,
                data_test = data_false,
                feat_x = feat_x,
                feat_z = feat_z,
                estimator_c = models[model_name].estimator_c,
                estimator_r = models[model_name].estimator_r,
                target = target,
                group = group,
                n_jobs = 1,
            )
            for model_name, _, data_false in false_jobs
        )
        false_results = [(f, y) for f, y, _ in false_results]
    else:
        raise ValueError("protocol must be either 'frozen' or 'retrain'.")

    ## collect frontier rows
    rows = []
    for (model_name, method_name, _), (frontier_false, _) in zip(false_jobs, false_results):
        frontier_real, _ = real_cv[model_name]
        for condition, frontier in [("original", frontier_real), ("falsified", frontier_false)]:
            for _, frow in frontier.iterrows():
                row = {
                    "model": model_name,
                    "method": method_name,
                    "condition": condition,
                    "group": frow["group"],
                }
                for col in FRONTIER_METRICS:
                    row[col] = frow[col]
                rows.append(row)

    return pd.DataFrame(rows)


## ----------------------------------------------------------------------
## target-alignment falsifiability test
## ----------------------------------------------------------------------
def eval_falsified_alignment(
    data: pd.DataFrame,
    falsified: dict[str, pd.DataFrame],
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_jobs: int = -1,
    protocol: str = "frozen",
    ) -> pd.DataFrame:
    """
    Desc:
        Test whether original data produces better target-prediction alignment
        than falsified data under the specified protocol.
    Args:
        data: clean evaluation dataframe.
        falsified: mapping of falsification method to dataframe.
        models: mapping of model name to estimator bundle.
        feat_x: graph invariant column names.
        feat_z: process signature column names.
        target: target column name.
        group: group column name for logo splitting.
        n_jobs: number of parallel jobs.
        protocol: "frozen" or "retrain".
    Returns:
        dataframe with consensus metrics per (model, method, condition, group).
    Raises:
        ValueError if protocol is not "frozen" or "retrain".
    """

    ## init feature lists as mutable for parallel jobs
    feat_x = list(feat_x)
    feat_z = list(feat_z)
    model_names = list(models.keys())

    ## original-data cv: once per model
    print(f"Running {len(model_names)} original + {len(model_names) * len(falsified)} false alignment jobs ({protocol})...")
    real_results = Parallel(n_jobs = n_jobs)(
        delayed(logo_cross_valid)(
            data = data,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = models[name].estimator_c,
            estimator_r = models[name].estimator_r,
            target = target,
            group = group,
            n_jobs = 1,
        )
        for name in model_names
    )
    real_cv = dict(zip(model_names, real_results))

    ## falsified-data cv: per (model, method)
    false_jobs = [
        (model_name, method_name, data_false)
        for method_name, data_false in falsified.items()
        for model_name in model_names
    ]

    if protocol == "retrain":
        false_results = Parallel(n_jobs = n_jobs)(
            delayed(logo_cross_valid)(
                data = data_false,
                feat_x = feat_x,
                feat_z = feat_z,
                estimator_c = models[model_name].estimator_c,
                estimator_r = models[model_name].estimator_r,
                target = target,
                group = group,
                n_jobs = 1,
            )
            for model_name, _, data_false in false_jobs
        )
    elif protocol == "frozen":
        false_results = Parallel(n_jobs = n_jobs)(
            delayed(logo_cross_valid_frozen)(
                data_train = data,
                data_test = data_false,
                feat_x = feat_x,
                feat_z = feat_z,
                estimator_c = models[model_name].estimator_c,
                estimator_r = models[model_name].estimator_r,
                target = target,
                group = group,
                n_jobs = 1,
            )
            for model_name, _, data_false in false_jobs
        )
        false_results = [(f, y) for f, y, _ in false_results]
    else:
        raise ValueError("protocol must be either 'frozen' or 'retrain'.")

    ## collect alignment rows
    rows = []
    for (model_name, method_name, data_false), (_, y_pred_false) in zip(false_jobs, false_results):
        _, y_pred_real = real_cv[model_name]
        for condition, y_pred, data_eval in [
            ("original", y_pred_real, data),
            ("falsified", y_pred_false, data_false),
        ]:
            y_true = _log_transformer(data_eval[target]).astype(float).values
            groups = data_eval[group].values
            valid = np.isfinite(y_true) & np.isfinite(y_pred)

            for group_name in pd.unique(groups):
                mask = (groups == group_name) & valid
                if int(np.sum(mask)) < 2:
                    continue
                mvals = consensus_metrics(
                    y_true = y_true[mask],
                    y_pred = y_pred[mask],
                )
                row = {
                    "model": model_name,
                    "method": method_name,
                    "condition": condition,
                    "group": group_name,
                }
                for col in CONSENSUS_METRICS:
                    row[col] = mvals[col]
                rows.append(row)

    return pd.DataFrame(rows)


## ----------------------------------------------------------------------
## pairwise consensus falsifiability test
## ----------------------------------------------------------------------
def eval_falsified_consensus(
    data: pd.DataFrame,
    falsified: dict[str, pd.DataFrame],
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_jobs: int = -1,
    protocol: str = "frozen",
    ) -> pd.DataFrame:
    """
    Desc:
        Test whether original data produces higher inter-model frontier
        consensus than falsified data under the specified protocol.
    Args:
        data: clean evaluation dataframe.
        falsified: mapping of falsification method to dataframe.
        models: mapping of model name to estimator bundle.
        feat_x: graph invariant column names.
        feat_z: process signature column names.
        target: target column name.
        group: group column name for logo splitting.
        n_jobs: number of parallel jobs.
        protocol: "frozen" or "retrain".
    Returns:
        dataframe with pairwise consensus metrics per
        (method, condition, group, model_i, model_j).
    Raises:
        ValueError if protocol is not "frozen" or "retrain".
    """

    ## init feature lists as mutable for parallel jobs
    feat_x = list(feat_x)
    feat_z = list(feat_z)
    model_names = list(models.keys())

    ## original-data cv: once per model
    print(f"Running {len(model_names)} original + {len(model_names) * len(falsified)} false consensus jobs ({protocol})...")
    real_results = Parallel(n_jobs = n_jobs)(
        delayed(logo_cross_valid)(
            data = data,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = models[name].estimator_c,
            estimator_r = models[name].estimator_r,
            target = target,
            group = group,
            n_jobs = 1,
        )
        for name in model_names
    )
    pred_real = {
        name: np.asarray(r[1], dtype = float)
        for name, r in zip(model_names, real_results)
    }

    ## falsified-data cv: all (model, method) pairs
    false_jobs = [
        (model_name, method_name, data_false)
        for method_name, data_false in falsified.items()
        for model_name in model_names
    ]

    if protocol == "retrain":
        false_results = Parallel(n_jobs = n_jobs)(
            delayed(logo_cross_valid)(
                data = data_false,
                feat_x = feat_x,
                feat_z = feat_z,
                estimator_c = models[model_name].estimator_c,
                estimator_r = models[model_name].estimator_r,
                target = target,
                group = group,
                n_jobs = 1,
            )
            for model_name, _, data_false in false_jobs
        )
    elif protocol == "frozen":
        false_results = Parallel(n_jobs = n_jobs)(
            delayed(logo_cross_valid_frozen)(
                data_train = data,
                data_test = data_false,
                feat_x = feat_x,
                feat_z = feat_z,
                estimator_c = models[model_name].estimator_c,
                estimator_r = models[model_name].estimator_r,
                target = target,
                group = group,
                n_jobs = 1,
            )
            for model_name, _, data_false in false_jobs
        )
        false_results = [(f, y) for f, y, _ in false_results]
    else:
        raise ValueError("protocol must be either 'frozen' or 'retrain'.")

    ## index falsified predictions by (method, model)
    pred_false = {}
    for (model_name, method_name, _), (_, y_pred_false) in zip(false_jobs, false_results):
        pred_false[(method_name, model_name)] = np.asarray(y_pred_false, dtype = float)

    ## pairwise consensus per group
    rows = []
    for method_name, data_false in falsified.items():
        for model_i, model_j in combinations(model_names, 2):
            for condition, pred_map, data_eval in [
                ("original", pred_real, data),
                ("falsified", {n: pred_false[(method_name, n)] for n in model_names}, data_false),
            ]:
                y_i = pred_map[model_i]
                y_j = pred_map[model_j]
                groups = data_eval[group].values
                valid = np.isfinite(y_i) & np.isfinite(y_j)

                for group_name in pd.unique(groups):
                    mask = (groups == group_name) & valid
                    if int(np.sum(mask)) < 2:
                        continue
                    mvals = consensus_metrics(
                        y_true = y_i[mask],
                        y_pred = y_j[mask],
                    )
                    rows.append({
                        "method": method_name,
                        "condition": condition,
                        "group": group_name,
                        "model_i": model_i,
                        "model_j": model_j,
                        **mvals,
                    })

    return pd.DataFrame(rows)
