## libraries
import os
import sys
import json
import tempfile
import configparser
import numpy as np
import pandas as pd
from typing import Dict, Any, Sequence
from pathlib import Path

## path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.data.builders import data_builder
from src.vectorizers.scalers import _log_transformer
from src.evaluators.metrics import consensus_metrics
from src.evaluators.metrics import FRONTIER_METRICS, CONSENSUS_METRICS

## configs
config = configparser.ConfigParser()
config.read(os.path.join(root, 'conf', 'settings.ini'))

## constants
PATH_PROC = config['paths']['PATH_PROC'].strip('"')
PATH_FALS = config['paths']['PATH_FALS'].strip('"')


## ----------------------------------------------------------------------
## model-level worker for a single falsification method
## ----------------------------------------------------------------------
def _eval_falsify_model(
    model_name: str,
    model: Any,
    data_real: pd.DataFrame,
    data_false: pd.DataFrame,
    feat_x: list[str],
    feat_z: list[str],
    target: str,
    group: str,
    method_name: str,
    protocol: str = "frozen",
    ) -> list[dict]:
    """
    Desc:
        Compare the clean baseline to a falsification using either the
        frozen-manifold or retrain protocol.
    Args:
        model_name: display name of the model family.
        model: model bundle with estimator_c and estimator_r.
        data_real: original evaluation dataframe.
        data_false: falsified evaluation dataframe.
        feat_x: graph invariant feature names.
        feat_z: process signature feature names.
        target: target column name.
        group: group column name for logo splitting.
        method_name: name of the falsification method.
        protocol: evaluation protocol for the falsified condition.
                  "frozen" trains on clean data, tests on falsified.
                  "retrain" retrains entirely on falsified data.
    Returns:
        list of result dicts with frontier metrics per condition per group.
    """

    from src.evaluators.resampling import logo_cross_valid, logo_cross_valid_frozen

    print(f"  {model_name}: {method_name}...", end = " ", flush = True)

    rows = []

    ## real condition: standard logo-cv
    frontier_real, _ = logo_cross_valid(
        data = data_real,
        feat_x = feat_x,
        feat_z = feat_z,
        estimator_c = model.estimator_c,
        estimator_r = model.estimator_r,
        target = target,
        group = group,
        n_jobs = 1,
    )
    for _, frow in frontier_real.iterrows():
        row = {
            "model": model_name,
            "method": method_name,
            "condition": "real",
            "group": frow["group"],
        }
        for col in FRONTIER_METRICS:
            row[col] = frow[col]
        rows.append(row)

    ## falsified condition
    if protocol == "retrain":
        frontier_false, _ = logo_cross_valid(
            data = data_false,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = model.estimator_c,
            estimator_r = model.estimator_r,
            target = target,
            group = group,
            n_jobs = 1,
        )
    else:
        frontier_false, _, _ = logo_cross_valid_frozen(
            data_train = data_real,
            data_test = data_false,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = model.estimator_c,
            estimator_r = model.estimator_r,
            target = target,
            group = group,
            n_jobs = 1,
        )

    for _, frow in frontier_false.iterrows():
        row = {
            "model": model_name,
            "method": method_name,
            "condition": "falsified",
            "group": frow["group"],
        }
        for col in FRONTIER_METRICS:
            row[col] = frow[col]
        rows.append(row)

    print("done.")
    return rows


## ----------------------------------------------------------------------
## consensus assembly for a single prediction vector
## ----------------------------------------------------------------------
def _collect_consensus_rows(
    model_name: str,
    method_name: str,
    condition: str,
    data_eval: pd.DataFrame,
    y_pred: np.ndarray,
    target: str,
    group: str,
    ) -> list[dict[str, Any]]:
    """
    Desc:
        Convert cross-validated predictions into group-level consensus
        metrics under the same held-out partitions used for frontier
        evaluation.
    Args:
        model_name: display name of the model family.
        method_name: falsification method label.
        condition: real or falsified evaluation label.
        data_eval: dataframe aligned with y_pred.
        y_pred: out-of-fold or frozen-fold predictions in log space.
        target: target column name.
        group: group column name.
    Returns:
        list of dict rows containing consensus metrics per group.
    """

    y_true = _log_transformer(data_eval[target]).astype(float).values
    groups = data_eval[group].values
    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    rows = []
    for group_name in pd.unique(groups):
        mask = (groups == group_name) & valid
        if int(np.sum(mask)) < 2:
            continue

        metrics = consensus_metrics(
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
            row[col] = metrics[col]
        rows.append(row)

    return rows


## ----------------------------------------------------------------------
## model-level worker for falsification consensus
## ----------------------------------------------------------------------
def _eval_falsify_model_consensus(
    model_name: str,
    model: Any,
    data_real: pd.DataFrame,
    data_false: pd.DataFrame,
    feat_x: list[str],
    feat_z: list[str],
    target: str,
    group: str,
    method_name: str,
    protocol: str = "frozen",
    ) -> list[dict[str, Any]]:
    """
    Desc:
        Compare clean and falsified held-out predictions using
        consensus metrics that directly measure target-prediction
        alignment rather than frontier envelope quality.
    Args:
        model_name: display name of the model family.
        model: model bundle with estimator_c and estimator_r.
        data_real: original evaluation dataframe.
        data_false: falsified evaluation dataframe.
        feat_x: graph invariant feature names.
        feat_z: process signature feature names.
        target: target column name.
        group: group column name for logo splitting.
        method_name: name of the falsification method.
        protocol: evaluation protocol for the falsified condition.
                  "frozen" trains on clean data, tests on falsified.
                  "retrain" retrains entirely on falsified data.
    Returns:
        list of result dicts with consensus metrics per condition per group.
    """

    from src.evaluators.resampling import logo_cross_valid, logo_cross_valid_frozen

    print(f"  {model_name}: {method_name} consensus...", end = " ", flush = True)

    rows = []

    ## real condition
    _, y_pred_real = logo_cross_valid(
        data = data_real,
        feat_x = feat_x,
        feat_z = feat_z,
        estimator_c = model.estimator_c,
        estimator_r = model.estimator_r,
        target = target,
        group = group,
        n_jobs = 1,
    )
    rows.extend(_collect_consensus_rows(
        model_name = model_name,
        method_name = method_name,
        condition = "real",
        data_eval = data_real,
        y_pred = y_pred_real,
        target = target,
        group = group,
    ))

    ## falsified condition
    if protocol == "retrain":
        _, y_pred_false = logo_cross_valid(
            data = data_false,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = model.estimator_c,
            estimator_r = model.estimator_r,
            target = target,
            group = group,
            n_jobs = 1,
        )
    else:
        _, y_pred_false, _ = logo_cross_valid_frozen(
            data_train = data_real,
            data_test = data_false,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = model.estimator_c,
            estimator_r = model.estimator_r,
            target = target,
            group = group,
            n_jobs = 1,
        )

    rows.extend(_collect_consensus_rows(
        model_name = model_name,
        method_name = method_name,
        condition = "falsified",
        data_eval = data_false,
        y_pred = y_pred_false,
        target = target,
        group = group,
    ))

    print("done.")
    return rows


## ----------------------------------------------------------------------
## falsifiability test: real vs falsified data
## ----------------------------------------------------------------------
def eval_falsifiability(
    data: pd.DataFrame | None,
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_jobs: int = -1,
    random_state: int = 42,
    protocol: str = "frozen",
    path_proc: str | Path = "data/processed/",
    path_fals: str | Path = PATH_FALS,
    ) -> pd.DataFrame:
    """
    Desc:
        Run falsifiability tests by comparing clean logo-cv frontier
        performance against precomputed falsifications using either the
        frozen-manifold or retrain protocol. the frozen protocol trains
        on clean folds and evaluates on falsified held-out groups. the
        retrain protocol retrains entirely on falsified data.
    Args:
        data: evaluation dataframe with features, target, and group.
        models: mapping of model name to estimator bundle.
        feat_x: graph invariant column names.
        feat_z: process signature column names.
        target: target column name.
        group: group column name for logo splitting.
        n_jobs: number of parallel jobs across model-method tasks.
        random_state: retained for api compatibility; precomputed falsified
                      data are loaded from disk.
        protocol: evaluation protocol for the falsified condition.
                  "frozen" trains on clean data, tests on falsified.
                  "retrain" retrains entirely on falsified data.
        path_fals: directory containing precomputed falsified dataset
                   json files.
    Returns:
        dataframe with one row per (model, method, condition, group).
    """

    if data is None:
        data = data_builder(path_proc)

    from joblib import Parallel, delayed

    falsified = _load_falsified(
        path_fals = path_fals,
    )

    jobs = []
    for method_name, data_false in falsified.items():
        for model_name, model in models.items():
            jobs.append((model_name, model, data, data_false,
                         list(feat_x), list(feat_z), target, group, method_name))

    print(f"Running {len(jobs)} falsifiability jobs ({protocol})...")
    outputs = Parallel(n_jobs = n_jobs)(
        delayed(_eval_falsify_model)(
            model_name = mn,
            model = m,
            data_real = dr,
            data_false = df,
            feat_x = fx,
            feat_z = fz,
            target = t,
            group = g,
            method_name = meth,
            protocol = protocol,
        )
        for mn, m, dr, df, fx, fz, t, g, meth in jobs
    )

    rows = []
    for out in outputs:
        rows.extend(out)

    return pd.DataFrame(rows)


## ----------------------------------------------------------------------
## falsifiability consensus: real vs falsified data
## ----------------------------------------------------------------------
def eval_falsifiability_consensus(
    data: pd.DataFrame | None,
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_jobs: int = -1,
    random_state: int = 42,
    protocol: str = "frozen",
    path_proc: str | Path = "data/processed/",
    path_fals: str | Path = PATH_FALS,
    ) -> pd.DataFrame:
    """
    Desc:
        Run falsifiability tests using consensus metrics on held-out
        predictions. this complements frontier metrics by testing
        whether the learned mapping still preserves agreement with the
        target after falsification.
    Args:
        data: evaluation dataframe with features, target, and group.
        models: mapping of model name to estimator bundle.
        feat_x: graph invariant column names.
        feat_z: process signature column names.
        target: target column name.
        group: group column name for logo splitting.
        n_jobs: number of parallel jobs across model-method tasks.
        random_state: retained for api compatibility; precomputed falsified
                      data are loaded from disk.
        protocol: evaluation protocol for the falsified condition.
                  "frozen" trains on clean data, tests on falsified.
                  "retrain" retrains entirely on falsified data.
        path_fals: directory containing precomputed falsified dataset
                   json files.
    Returns:
        dataframe with one row per (model, method, condition, group).
    """

    if data is None:
        data = data_builder(path_proc)

    from joblib import Parallel, delayed

    falsified = _load_falsified(
        path_fals = path_fals,
    )

    jobs = []
    for method_name, data_false in falsified.items():
        for model_name, model in models.items():
            jobs.append((model_name, model, data, data_false,
                         list(feat_x), list(feat_z), target, group, method_name))

    print(f"Running {len(jobs)} falsifiability consensus jobs ({protocol})...")
    outputs = Parallel(n_jobs = n_jobs)(
        delayed(_eval_falsify_model_consensus)(
            model_name = mn,
            model = m,
            data_real = dr,
            data_false = df,
            feat_x = fx,
            feat_z = fz,
            target = t,
            group = g,
            method_name = meth,
            protocol = protocol,
        )
        for mn, m, dr, df, fx, fz, t, g, meth in jobs
    )

    rows = []
    for out in outputs:
        rows.extend(out)

    return pd.DataFrame(rows)
