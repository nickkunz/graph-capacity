## libraries
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Sequence, Mapping
from pathlib import Path
from itertools import combinations
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

## path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.vectorizers.scalers import _log_transformer
from src.evaluators.training import fit_predict_frontier
from src.evaluators.resampling import logo_cross_valid, logo_cross_valid_frozen
from src.evaluators.metrics import consensus_metrics, frontier_metrics
from src.evaluators.metrics import FRONTIER_METRICS, CONSENSUS_METRICS

## ----------------------------------------------------------------------------
## frontier falsifiability test
## ----------------------------------------------------------------------------
def eval_falsified_frontier(
    data_proc: pd.DataFrame,
    data_fals: dict[str, pd.DataFrame],
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_repeat: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    ) -> pd.DataFrame:

    """
    Desc:
        Test whether original data produces better frontier envelope metrics
        than falsified data under both frozen and retrain protocols.
    
    Args:
        original: clean evaluation dataframe.
        falsified: mapping of falsification method to dataframe.
        models: mapping of model name to estimator bundle.
        feat_x: graph invariant column names.
        feat_z: process signature column names.
        target: target column name.
        n_repeat: number of repeated cv fits to average across seeds.
        group: group column name for logo splitting.
        random_state: random seed forwarded to cv estimators when supported.
        n_jobs: number of parallel jobs.
    
    Returns:
        dataframe with frontier metrics per model, method, condition, group, track.
    """

    ## init feature lists as mutable for parallel jobs
    feat_x = list(feat_x)
    feat_z = list(feat_z)
    model_names = list(models.keys())
    if n_repeat < 1:
        raise ValueError("n_repeat must be >= 1")

    y_true_proc = _log_transformer(data_proc[target]).astype(float).values
    groups_proc = data_proc[group].values

    ## original-data cv: repeated across seeds, then average predictions
    real_results = Parallel(n_jobs = n_jobs)(
        delayed(logo_cross_valid)(
            data = data_proc,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = models[name].estimator_c,
            estimator_r = models[name].estimator_r,
            target = target,
            group = group,
            random_state = None if random_state is None else int(random_state) + repeat_idx,
            n_jobs = 1,
        )
        for name in model_names
        for repeat_idx in range(n_repeat)
    )
    real_cv = dict()
    for model_idx, model_name in enumerate(model_names):
        start = model_idx * n_repeat
        runs = real_results[start:start + n_repeat]
        pred_stack = np.vstack([y_pred for _, y_pred in runs])
        valid_pred = np.isfinite(pred_stack)
        y_pred_mean = np.full(shape = pred_stack.shape[1], fill_value = np.nan, dtype = float)
        valid_cols = np.any(valid_pred, axis = 0)
        if np.any(valid_cols):
            y_pred_mean[valid_cols] = np.nanmean(pred_stack[:, valid_cols], axis = 0)

        frontier_rows = []
        for group_name in pd.unique(groups_proc):
            mask = (
                (groups_proc == group_name)
                & np.isfinite(y_true_proc)
                & np.isfinite(y_pred_mean)
            )
            if int(np.sum(mask)) == 0:
                continue
            frontier_rows.append({
                "group": group_name,
                **frontier_metrics(y_true = y_true_proc[mask], y_pred = y_pred_mean[mask]),
            })

        real_cv[model_name] = (pd.DataFrame(frontier_rows), y_pred_mean)

    ## falsified-data evaluation jobs: explicit working-tree marker for source control
    false_jobs = [
        (model_name, method_name, data_test)
        for method_name, data_test in data_fals.items()
        for model_name in model_names
    ]

    frames = []
    for track in ("frozen", "retrain"):
        if track == "retrain":
            false_results = Parallel(n_jobs = n_jobs)(
                delayed(logo_cross_valid)(
                    data = data,
                    feat_x = feat_x,
                    feat_z = feat_z,
                    estimator_c = models[model_name].estimator_c,
                    estimator_r = models[model_name].estimator_r,
                    target = target,
                    group = group,
                    random_state = None if random_state is None else int(random_state) + repeat_idx,
                    n_jobs = 1,  ## avoid over-subscription of parallel jobs
                )
                for model_name, _, data in false_jobs
                for repeat_idx in range(n_repeat)
            )
        else:
            false_results = Parallel(n_jobs = n_jobs)(
                delayed(logo_cross_valid_frozen)(
                    data_train = data_proc,
                    data_test = data_test,
                    feat_x = feat_x,
                    feat_z = feat_z,
                    estimator_c = models[model_name].estimator_c,
                    estimator_r = models[model_name].estimator_r,
                    target = target,
                    group = group,
                    random_state = None if random_state is None else int(random_state) + repeat_idx,
                    n_jobs = 1,  ## avoid over-subscription of parallel jobs
                )
                for model_name, _, data_test in false_jobs
                for repeat_idx in range(n_repeat)
            )
            false_results = [(frontier, yhat) for frontier, yhat, _ in false_results]

        false_results_mean = []
        for job_idx, (_, _, data_eval) in enumerate(false_jobs):
            start = job_idx * n_repeat
            runs = false_results[start:start + n_repeat]
            pred_stack = np.vstack([y_pred for _, y_pred in runs])
            valid_pred = np.isfinite(pred_stack)
            y_pred_mean = np.full(shape = pred_stack.shape[1], fill_value = np.nan, dtype = float)
            valid_cols = np.any(valid_pred, axis = 0)
            if np.any(valid_cols):
                y_pred_mean[valid_cols] = np.nanmean(pred_stack[:, valid_cols], axis = 0)

            y_true_eval = _log_transformer(data_eval[target]).astype(float).values
            groups_eval = data_eval[group].values
            frontier_rows = []
            for group_name in pd.unique(groups_eval):
                mask = (
                    (groups_eval == group_name)
                    & np.isfinite(y_true_eval)
                    & np.isfinite(y_pred_mean)
                )
                if int(np.sum(mask)) == 0:
                    continue
                frontier_rows.append({
                    "group": group_name,
                    **frontier_metrics(y_true = y_true_eval[mask], y_pred = y_pred_mean[mask]),
                })

            false_results_mean.append((pd.DataFrame(frontier_rows), y_pred_mean))

        obs = list()
        for (model_name, method_name, _), (frontier_false, _) in zip(false_jobs, false_results_mean):
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
                    obs.append(row)

        frame = pd.DataFrame(obs)
        frame["track"] = track
        frames.append(frame)

    return pd.concat(frames, ignore_index = True)


## ----------------------------------------------------------------------
## target-alignment falsifiability test
## ----------------------------------------------------------------------
def eval_falsified_alignment(
    data_proc: pd.DataFrame,
    data_fals: dict[str, pd.DataFrame],
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_repeat: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    ) -> pd.DataFrame:
    """
    Desc:
        Test whether original data produces better target-prediction alignment
        than falsified data under both frozen and retrain protocols, using
        cross-validated predictions and global aggregation across all valid
        observations.
    Args:
        data_proc: clean evaluation dataframe.
        data_fals: mapping of falsification method to dataframe.
        models: mapping of model name to estimator bundle.
        feat_x: graph invariant column names.
        feat_z: process signature column names.
        target: target column name.
        n_repeat: number of repeated cv fits to average across seeds.
        group: group column name for logo splitting.
        random_state: random seed forwarded to cv estimators when supported.
        n_jobs: number of parallel jobs.
    Returns:
        dataframe with consensus metrics per
        (model, method, condition, group, track).
    """

    ## init feature lists as mutable for parallel jobs
    feat_x = list(feat_x)
    feat_z = list(feat_z)
    model_names = list(models.keys())
    if n_repeat < 1:
        raise ValueError("n_repeat must be >= 1")

    ## original-data cv: repeated across seeds, then average predictions
    real_results = Parallel(n_jobs = n_jobs)(
        delayed(logo_cross_valid)(
            data = data_proc,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = models[name].estimator_c,
            estimator_r = models[name].estimator_r,
            target = target,
            group = group,
            random_state = None if random_state is None else int(random_state) + repeat_idx,
            n_jobs = 1,
        )
        for name in model_names
        for repeat_idx in range(n_repeat)
    )
    real_cv = dict()
    for model_idx, model_name in enumerate(model_names):
        start = model_idx * n_repeat
        runs = real_results[start:start + n_repeat]
        pred_stack = np.vstack([y_pred for _, y_pred in runs])
        valid_pred = np.isfinite(pred_stack)
        y_pred_mean = np.full(shape = pred_stack.shape[1], fill_value = np.nan, dtype = float)
        valid_cols = np.any(valid_pred, axis = 0)
        if np.any(valid_cols):
            y_pred_mean[valid_cols] = np.nanmean(pred_stack[:, valid_cols], axis = 0)
        real_cv[model_name] = y_pred_mean

    ## falsified-data cv: per (model, method)
    false_jobs = [
        (model_name, method_name, data_false)
        for method_name, data_false in data_fals.items()
        for model_name in model_names
    ]

    frames = []
    for track in ("frozen", "retrain"):
        if track == "retrain":
            false_results = Parallel(n_jobs = n_jobs)(
                delayed(logo_cross_valid)(
                    data = data_false,
                    feat_x = feat_x,
                    feat_z = feat_z,
                    estimator_c = models[model_name].estimator_c,
                    estimator_r = models[model_name].estimator_r,
                    target = target,
                    group = group,
                    random_state = None if random_state is None else int(random_state) + repeat_idx,
                    n_jobs = 1,
                )
                for model_name, _, data_false in false_jobs
                for repeat_idx in range(n_repeat)
            )
        else:
            false_results = Parallel(n_jobs = n_jobs)(
                delayed(logo_cross_valid_frozen)(
                    data_train = data_proc,
                    data_test = data_false,
                    feat_x = feat_x,
                    feat_z = feat_z,
                    estimator_c = models[model_name].estimator_c,
                    estimator_r = models[model_name].estimator_r,
                    target = target,
                    group = group,
                    random_state = None if random_state is None else int(random_state) + repeat_idx,
                    n_jobs = 1,
                )
                for model_name, _, data_false in false_jobs
                for repeat_idx in range(n_repeat)
            )
            false_results = [(f, y) for f, y, _ in false_results]

        false_results_mean = []
        for job_idx in range(len(false_jobs)):
            start = job_idx * n_repeat
            runs = false_results[start:start + n_repeat]
            pred_stack = np.vstack([y_pred for _, y_pred in runs])
            valid_pred = np.isfinite(pred_stack)
            y_pred_mean = np.full(shape = pred_stack.shape[1], fill_value = np.nan, dtype = float)
            valid_cols = np.any(valid_pred, axis = 0)
            if np.any(valid_cols):
                y_pred_mean[valid_cols] = np.nanmean(pred_stack[:, valid_cols], axis = 0)
            false_results_mean.append(y_pred_mean)

        obs = []
        for (model_name, method_name, data_false), y_pred_false in zip(false_jobs, false_results_mean):
            y_pred_real = real_cv[model_name]
            for condition, y_pred, data_eval in [
                ("original", y_pred_real, data_proc),
                ("falsified", y_pred_false, data_false),
            ]:
                y_true = _log_transformer(data_eval[target]).astype(float).values
                valid = np.isfinite(y_true) & np.isfinite(y_pred)
                if int(np.sum(valid)) < 2:
                    continue

                mvals = consensus_metrics(
                    y_true = y_true[valid],
                    y_pred = y_pred[valid],
                )
                row = {
                    "model": model_name,
                    "method": method_name,
                    "condition": condition,
                    "group": "all",
                }
                for col in CONSENSUS_METRICS:
                    row[col] = mvals[col]
                obs.append(row)

        frame = pd.DataFrame(obs)
        frame["track"] = track
        frames.append(frame)

    return pd.concat(frames, ignore_index = True)


## ----------------------------------------------------------------------------
## pairwise consensus falsifiability test
## ----------------------------------------------------------------------------
def eval_falsified_consensus(
    data_proc: pd.DataFrame,
    data_fals: dict[str, pd.DataFrame],
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    n_repeat: int = 30,
    n_jobs: int = -1,
    random_state: int = 42,
    ) -> pd.DataFrame:
    """
    Desc:
        Test whether original data produces higher inter-model frontier
        consensus than falsified data under both frozen and retrain protocols,
        treating pairwise consensus as a fitted-frontier agreement analysis
        rather than a predictive resampling test.
    Args:
        data_proc: clean evaluation dataframe.
        data_fals: mapping of falsification method to dataframe.
        models: mapping of model name to estimator bundle.
        feat_x: graph invariant column names.
        feat_z: process signature column names.
        target: target column name.
        n_repeat: number of repeated fits to average.
        random_state: random seed forwarded to cv estimators when supported.
        n_jobs: number of parallel jobs.
    Returns:
        dataframe with pairwise consensus metrics per
        (method, condition, group, model_i, model_j, track).
    """

    ## init feature lists as mutable for parallel jobs
    feat_x = list(feat_x)
    feat_z = list(feat_z)
    model_names = list(models.keys())

    ## original-data full fit: once per model
    real_results = Parallel(n_jobs = n_jobs)(
        delayed(fit_predict_frontier)(
            data = data_proc,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = models[name].estimator_c,
            estimator_r = models[name].estimator_r,
            target = target,
            n_repeat = n_repeat,
            random_state = random_state,
        )
        for name in model_names
    )
    pred_real = {
        name: np.asarray(r["y_pred"], dtype = float)
        for name, r in zip(model_names, real_results)
    }
    fit_real = dict(zip(model_names, real_results))

    ## falsified-data cv: all (model, method) pairs
    false_jobs = [
        (model_name, method_name, data_false)
        for method_name, data_false in data_fals.items()
        for model_name in model_names
    ]

    frames = []
    for track in ("frozen", "retrain"):
        if track == "retrain":
            false_results = Parallel(n_jobs = n_jobs)(
                delayed(fit_predict_frontier)(
                    data = data_false,
                    feat_x = feat_x,
                    feat_z = feat_z,
                    estimator_c = models[model_name].estimator_c,
                    estimator_r = models[model_name].estimator_r,
                    target = target,
                    n_repeat = n_repeat,
                    random_state = random_state,
                )
                for model_name, _, data_false in false_jobs
            )
        else:
            false_results = [
                fit_predict_frontier(
                    data = data_false,
                    fit_result = fit_real[model_name],
                )
                for model_name, _, data_false in false_jobs
            ]

        ## index falsified predictions by (method, model)
        pred_false = {}
        for (model_name, method_name, _), fit_false in zip(false_jobs, false_results):
            pred_false[(method_name, model_name)] = np.asarray(fit_false["y_pred"], dtype = float)

        obs = []
        for method_name, data_false in data_fals.items():
            for model_i, model_j in combinations(model_names, 2):
                for condition, pred_map, data_eval in [
                    ("original", pred_real, data_proc),
                    ("falsified", {n: pred_false[(method_name, n)] for n in model_names}, data_false),
                ]:
                    y_i = pred_map[model_i]
                    y_j = pred_map[model_j]
                    valid = np.isfinite(y_i) & np.isfinite(y_j)
                    if int(np.sum(valid)) == 0:
                        continue
                    mvals = consensus_metrics(
                        y_true = y_i[valid],
                        y_pred = y_j[valid],
                    )
                    obs.append({
                        "method": method_name,
                        "condition": condition,
                        "group": "all",
                        "model_i": model_i,
                        "model_j": model_j,
                        **mvals,
                    })

        frame = pd.DataFrame(obs)
        frame["track"] = track
        frames.append(frame)

    return pd.concat(frames, ignore_index = True)


## ----------------------------------------------------------------------
## summarize falsification tests
## ----------------------------------------------------------------------
def stat_falsified_test(
    results: pd.DataFrame,
    feat_value: Sequence[str] | None = None,
    feat_group: Sequence[str] | None = None,
    feat_pairs: Sequence[str] | None = None,
    label_cond: str = "condition",
    label_orig: str = "original",
    label_fals: str = "falsified",
    group_order: Mapping[str, Sequence[str]] | None = None,
    decimals: int = 4,
    index: bool = True,
    ) -> pd.DataFrame:
    
    """
    Desc:
        Paired Wilcoxon signed-rank summary comparing original vs falsified
        conditions. Works for frontier, alignment, and pairwise consensus
        outputs by parameterising metric, grouping, and pairing columns.
    Args:
        results: output of any eval_falsified_* function.
        feat_value: metric columns to test (e.g. FRONTIER_METRICS or
            CONSENSUS_METRICS).
        feat_group: columns whose unique combinations define independent
            tests (e.g. ["track", "method"] or ["method"]). None -> one
            global test.
        feat_pairs: columns that align an original row with its falsified
            counterpart (e.g. ["model", "group"]). None -> inferred as
            every non-metric, non-condition, non-group column.
        label_cond: column that flags original vs falsified.
        label_orig: value in label_cond for original data.
        label_fals: value in label_cond for falsified data.
    Returns:
        Display-ready table with columns:
        [*feat_group, Metric?, Median <M> (Original), Median <M> (Falsified),
         Median Δ<M>, Wilcoxon W, Rank-biserial r, One-sided p,
         Holm-adjusted p, Sig.]
        where Metric is only included when len(feat_value) > 1.
    """
    
    feat_value = list(feat_value)
    feat_group = list(feat_group or [])
    group_display = [c.title() for c in feat_group]
    tail_cols = ["Wilcoxon W", "Rank-biserial r", "One-sided p", "Holm-adjusted p", "Sig."]

    ## infer pairing columns from non-metric/non-group fields when not provided
    reserved = set(feat_value) | {label_cond} | set(feat_group)
    pair_cols = list(feat_pairs) if feat_pairs is not None else [c for c in results.columns if c not in reserved]

    ## align original and falsified rows on group+pair keys
    merged = results.loc[results[label_cond] == label_orig].merge(
        results.loc[results[label_cond] == label_fals],
        on = feat_group + pair_cols,
        suffixes = ("_orig", "_fals"),
    )
    groups = merged.groupby(feat_group, sort = False) if feat_group else [((), merged)]

    ## compute paired stats per group x metric
    rows = []
    for group_key, grp in groups:
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        for metric in feat_value:
            x = grp[f"{metric}_orig"].to_numpy(dtype = float)
            y = grp[f"{metric}_fals"].to_numpy(dtype = float)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            n = len(x)
            med_o, med_f, med_d = (
                (float(np.median(x)), float(np.median(y)), float(np.median(x - y))) if n else (np.nan, np.nan, np.nan)
            )

            if n < 2 or np.allclose(x, y):
                w_stat, r_eff, p_val = np.nan, np.nan, np.nan
            else:
                w_stat, p_val = wilcoxon(x, y, alternative = "greater")
                r_eff = abs(1.0 - (2.0 * w_stat) / (n * (n + 1) / 2))

            rows.append((*group_key, metric, med_o, med_f, med_d, w_stat, r_eff, float(p_val)))

    summary = pd.DataFrame(rows, columns = feat_group + [
        "metric",
        "Median Original",
        "Median Falsified",
        "Median Δ",
        "Wilcoxon W",
        "Rank-biserial r",
        "One-sided p",
    ])

    ## holm-bonferroni correction with monotonic adjustment
    p_vals = summary["One-sided p"].to_numpy(copy = True)
    valid_p = np.isfinite(p_vals)
    holm = np.full(shape = len(p_vals), fill_value = np.nan, dtype = float)
    if np.any(valid_p):
        p_valid = p_vals[valid_p]
        m = len(p_valid)
        order = np.argsort(p_valid)

        ## holm adjusted p-values are cumulative maxima of scaled sorted p-values
        holm_sorted = np.maximum.accumulate(p_valid[order] * (m - np.arange(m)))
        holm_valid = np.empty(m, dtype = float)
        holm_valid[order] = np.minimum(holm_sorted, 1.0)
        holm[valid_p] = holm_valid
    summary["Holm-adjusted p"] = holm
    summary["Sig."] = summary["Holm-adjusted p"].map(
        lambda p: np.nan if not np.isfinite(p) else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )

    ## convert group/metric labels to display names
    summary = summary.rename(columns = {c: c.title() for c in feat_group})
    if len(feat_value) == 1:
        tag = feat_value[0].upper()
        summary = summary.rename(columns = {
            "Median Original": f"Median {tag} (Original)",
            "Median Falsified": f"Median {tag} (Falsified)",
            "Median Δ": f"Median Δ {tag}",
        }).drop(columns = ["metric"])
    else:
        summary = summary.rename(columns = {"metric": "Metric"})

    ## apply optional ordering, then format numeric output
    if group_order:
        for col, order_vals in group_order.items():
            dcol = col.title()
            if dcol in summary.columns:
                summary[dcol] = pd.Categorical(summary[dcol], categories = list(order_vals), ordered = True)
    if group_display:
        summary = summary.sort_values(group_display).reset_index(drop = True)

    if "Wilcoxon W" in summary.columns:
        summary["Wilcoxon W"] = summary["Wilcoxon W"].round(0).astype("Int64")
    round_cols = [c for c in summary.select_dtypes(include = [np.number]).columns if c != "Wilcoxon W"]
    if round_cols:
        summary[round_cols] = summary[round_cols].round(decimals)

    ## keep a stable display column order for single- and multi-metric outputs
    if "Metric" in summary.columns:
        value_cols_order = ["Metric", "Median Original", "Median Falsified", "Median Δ", *tail_cols]
    else:
        med_o = next((c for c in summary.columns if c.startswith("Median ") and c.endswith("(Original)")), "Median Original")
        med_f = next((c for c in summary.columns if c.startswith("Median ") and c.endswith("(Falsified)")), "Median Falsified")
        med_d = next((c for c in summary.columns if c.startswith("Median Δ")), "Median Δ")
        value_cols_order = [med_o, med_f, med_d, *tail_cols]

    summary = summary.reindex(columns = group_display + [c for c in value_cols_order if c in summary.columns])
    return summary.set_index(group_display) if (index and group_display) else summary
