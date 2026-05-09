## libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import parallel
from contextlib import contextmanager
from joblib.parallel import BatchCompletionCallBack
from typing import Sequence, Dict, Any, Iterator
from scipy.stats import rankdata, wilcoxon

## modules
from src.evaluators.metrics import (
    FRONTIER_METRICS,
    CONSENSUS_METRICS
)


SPECIFICATION_ORDER = [
    "additive",
    "interaction",
    "interaction_joint",
    "joint",
    "capacity_only",
    "dynamics_only",
]

## significance code helper
def _sig_code(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


## holm-bonferroni helper
def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype = float)
    adjusted = np.full(shape = len(p_values), fill_value = np.nan, dtype = float)
    valid = np.isfinite(p_values)
    if not np.any(valid):
        return adjusted

    p_valid = p_values[valid]
    n_tests = len(p_valid)
    order = np.argsort(p_valid)
    adjusted_sorted = np.maximum.accumulate(p_valid[order] * (n_tests - np.arange(n_tests)))
    adjusted_valid = np.empty(n_tests, dtype = float)
    adjusted_valid[order] = np.minimum(adjusted_sorted, 1.0)
    adjusted[valid] = adjusted_valid
    return adjusted

## joblib progress bar bridge
@contextmanager
def _tqdm_joblib(total: int, desc: str) -> Iterator[Any]:

    """
    Desc: bridge joblib completion callbacks into a simple tqdm progress bar.
    Args:
        total: total number of parallel jobs.
        desc: progress bar label.
    Returns:
        Iterator over the active tqdm progress bar context.
    """

    pbar = tqdm(total = total, desc = desc, unit = "model")

    class _TqdmBatchCompletionCallback(BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n = self.batch_size)
            return super().__call__(*args, **kwargs)

    batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        parallel.BatchCompletionCallBack = batch_callback
        pbar.close()

## --------------------------------------------------------------------------
## single-stage fold worker
## --------------------------------------------------------------------------
def _run_single_stage_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    F: pd.DataFrame,
    y_star: pd.Series,
    feats: list[str],
    estimator,
    random_state: int = 42,
    group_name: str | None = None,
    ) -> dict | None:

    """
    Desc: execute a single fold of single-stage logo-cv. trains a single
          estimator on the full feature set and evaluates frontier metrics.
    Args:
        train_idx: training indices.
        test_idx: test indices.
        F: feature dataframe (already numeric-coerced).
        y_star: log-transformed target series.
        feats: feature column names.
        estimator: estimator to clone and fit.
        random_state: random seed forwarded to estimator when supported.
        group_name: optional group label for the fold.
    Returns:
        dict with frontier metrics, predictions, and index mapping,
        or none if the fold is skipped due to insufficient data.
    """

    from sklearn.base import clone
    from src.vectorizers.scalers import _standardizer
    from src.evaluators.metrics import frontier_metrics

    ## split
    F_tr = F.iloc[train_idx]
    y_tr = y_star.iloc[train_idx].values.astype(float)
    F_te = F.iloc[test_idx]
    y_true = y_star.iloc[test_idx].values.astype(float)

    ## drop nans
    mask_tr = F_tr[feats].notna().all(axis = 1) & np.isfinite(y_tr)
    mask_te = F_te[feats].notna().all(axis = 1) & np.isfinite(y_true)
    F_tr = F_tr.loc[mask_tr]
    y_tr = y_tr[mask_tr.values]
    F_te = F_te.loc[mask_te]
    y_true = y_true[mask_te.values]

    if len(F_tr) < 2 or len(F_te) == 0:
        return None

    ## standardize (fit on train only)
    F_te = F_te[feats]
    F_tr_s, f_sc = _standardizer(F_tr, feats)
    F_tr_s = F_tr_s[feats].values.astype(float)
    F_te_s = f_sc.transform(F_te.astype(float))

    ## clone, fit, predict
    m = clone(estimator)
    if hasattr(m, "random_state"):
        m.set_params(random_state = random_state)
    m.fit(F_tr_s, y_tr)
    y_pred = m.predict(F_te_s).astype(float)

    ## frontier metrics
    kept_indices = test_idx[mask_te.values]
    frontier = frontier_metrics(y_true = y_true, y_pred = y_pred)

    return {
        "group_name": group_name,
        "frontier": frontier,
        "kept_indices": kept_indices,
        "y_pred": y_pred,
    }


## --------------------------------------------------------------------------
## single-stage logo cross validation
## --------------------------------------------------------------------------
def _single_stage_logo_cv(
    data: pd.DataFrame,
    feats: list[str],
    estimator,
    target: str,
    group: str,
    n_repeats: int = 30,
    random_state: int = 42,
    n_jobs: int = 1,
    ) -> tuple[pd.DataFrame, np.ndarray]:

    """
    Desc: run single-stage logo cross validation with fold-level parallelism.
          trains a single estimator on the full feature set f(feats) -> y*.
    Args:
        data: dataframe with features, target, and group columns.
        feats: feature column names.
        estimator: estimator to clone per fold.
        target: target column name.
        group: group column name for logo splitting.
        n_repeats: number of repeated cv seeds to average predictions.
        random_state: base random state for seed reproducibility.
        n_jobs: number of parallel jobs (-1 for all cores).
    Returns:
        tuple of (frontier results dataframe, predicted values array).
    """

    from sklearn.model_selection import LeaveOneGroupOut
    from joblib import Parallel, delayed
    from src.vectorizers.scalers import _log_transformer

    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    F = data[feats].apply(pd.to_numeric, errors = "coerce")
    y_star = _log_transformer(data[target]).astype(float)
    groups = data[group].values

    logo = LeaveOneGroupOut()
    fold_splits = list(logo.split(F.values, y_star.values, groups))

    ## parallel fold execution across all repeats
    fold_results = Parallel(n_jobs = n_jobs)(
        delayed(_run_single_stage_fold)(
            train_idx = train_idx,
            test_idx = test_idx,
            F = F,
            y_star = y_star,
            feats = feats,
            estimator = estimator,
            random_state = int(random_state) + repeat_idx,
            group_name = groups[test_idx][0],
        )
        for repeat_idx in range(n_repeats)
        for train_idx, test_idx in fold_splits
    )

    ## merge repeat-averaged predictions and metrics
    y_pred_sum = np.zeros(shape = len(data), dtype = float)
    y_pred_count = np.zeros(shape = len(data), dtype = int)
    group_frontiers = dict()
    for result in fold_results:
        if result is None:
            continue
        y_pred_sum[result["kept_indices"]] += result["y_pred"]
        y_pred_count[result["kept_indices"]] += 1
        grp = result["group_name"]
        if grp not in group_frontiers:
            group_frontiers[grp] = list()
        group_frontiers[grp].append(result["frontier"])

    frontier_rows = []
    for grp in sorted(group_frontiers.keys()):
        metrics_avg = pd.DataFrame(group_frontiers[grp]).mean().to_dict()
        frontier_rows.append({"group": grp, **metrics_avg})

    y_pred_out = np.full(shape = len(data), fill_value = np.nan)
    valid_pred = y_pred_count > 0
    y_pred_out[valid_pred] = y_pred_sum[valid_pred] / y_pred_count[valid_pred]

    return pd.DataFrame(frontier_rows), y_pred_out


## --------------------------------------------------------------------------
## separation model worker
## --------------------------------------------------------------------------
def _eval_separation_model(
    model_name: str,
    model,
    data: pd.DataFrame,
    data_aug: pd.DataFrame,
    feat_x: list[str],
    feat_z: list[str],
    feat_z_aug: list[str],
    interaction_cols: list[str],
    target: str,
    group: str,
    n_repeats: int,
    random_state: int,
    y_star_all: np.ndarray,
    groups_all: np.ndarray,
    names_all: np.ndarray,
    ) -> tuple[list[dict], list[dict]]:

    """
    Desc: evaluate all separation specifications for a single model.
          fold-level work stays sequential to avoid high overhead on the
          tiny 5-fold workload; parallelism is applied across models.
    Args:
        model_name: display name of the model family.
        model: model bundle with estimator_c and estimator_r.
        data: original evaluation dataframe.
        data_aug: augmented dataframe with interaction columns.
        feat_x: graph invariant feature names.
        feat_z: process signature feature names.
        feat_z_aug: augmented process feature names.
        interaction_cols: interaction column names.
        target: target column name.
        group: group column name.
        n_repeats: number of repeated cv seeds to average predictions.
        random_state: base random state for seed reproducibility.
        y_star_all: full target array for prediction reporting.
        groups_all: full group array for prediction reporting.
        names_all: full dataset-name array for prediction reporting.
    Returns:
        tuple of (frontier rows, prediction rows).
    """

    from src.evaluators.resampling import logo_cross_valid

    frontier_a, y_pred_a = logo_cross_valid(
        data = data,
        feat_x = feat_x,
        feat_z = feat_z,
        estimator_c = model.estimator_c,
        estimator_r = model.estimator_r,
        target = target,
        group = group,
        n_repeats = n_repeats,
        random_state = random_state,
        n_jobs = 1,
    )

    frontier_b, y_pred_b = logo_cross_valid(
        data = data_aug,
        feat_x = feat_x,
        feat_z = feat_z_aug,
        estimator_c = model.estimator_c,
        estimator_r = model.estimator_r,
        target = target,
        group = group,
        n_repeats = n_repeats,
        random_state = random_state,
        n_jobs = 1,
    )

    feat_joint = feat_x + feat_z
    frontier_c, y_pred_c = _single_stage_logo_cv(
        data = data,
        feats = feat_joint,
        estimator = model.estimator_c,
        target = target,
        group = group,
        n_repeats = n_repeats,
        random_state = random_state,
        n_jobs = 1,
    )

    feat_int_joint = feat_x + feat_z + interaction_cols
    frontier_d, y_pred_d = _single_stage_logo_cv(
        data = data_aug,
        feats = feat_int_joint,
        estimator = model.estimator_c,
        target = target,
        group = group,
        n_repeats = n_repeats,
        random_state = random_state,
        n_jobs = 1,
    )

    frontier_f, y_pred_f = _single_stage_logo_cv(
        data = data,
        feats = feat_x,
        estimator = model.estimator_c,
        target = target,
        group = group,
        n_repeats = n_repeats,
        random_state = random_state,
        n_jobs = 1,
    )

    frontier_g, y_pred_g = _single_stage_logo_cv(
        data = data,
        feats = feat_z,
        estimator = model.estimator_c,
        target = target,
        group = group,
        n_repeats = n_repeats,
        random_state = random_state,
        n_jobs = 1,
    )

    results = []
    predictions = []

    for spec, frontier in [
        ("additive", frontier_a), ("interaction", frontier_b),
        ("joint", frontier_c), ("interaction_joint", frontier_d),
        ("capacity_only", frontier_f), ("dynamics_only", frontier_g),
    ]:
        for _, frow in frontier.iterrows():
            row = {"model": model_name, "specification": spec, "group": frow["group"]}
            for col in FRONTIER_METRICS:
                row[col] = frow[col]
            results.append(row)

    for spec, y_pred_spec in [
        ("additive", y_pred_a), ("interaction", y_pred_b),
        ("joint", y_pred_c), ("interaction_joint", y_pred_d),
        ("capacity_only", y_pred_f), ("dynamics_only", y_pred_g),
    ]:
        for i in range(len(data)):
            if np.isfinite(y_pred_spec[i]) and np.isfinite(y_star_all[i]):
                predictions.append({
                    "model": model_name,
                    "specification": spec,
                    "dataset": names_all[i],
                    "group": groups_all[i],
                    "y_true": y_star_all[i],
                    "y_pred": y_pred_spec[i],
                    "abs_error": abs(y_star_all[i] - y_pred_spec[i]),
                })

    return results, predictions


## --------------------------------------------------------------------------
## capacity fold worker (attribution step 1)
## --------------------------------------------------------------------------
def _run_capacity_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: pd.DataFrame,
    Z: pd.DataFrame,
    y_star: pd.Series,
    feat_x: list[str],
    feat_z: list[str],
    estimator_c,
    random_state: int = 42,
    ) -> dict | None:

    """
    Desc: execute a single fold to compute out-of-fold C(X) predictions
          and slack values.
    Args:
        train_idx: training indices.
        test_idx: test indices.
        X: graph invariant features.
        Z: process signature features.
        y_star: log-transformed target series.
        feat_x: graph invariant column names.
        feat_z: process signature column names.
        estimator_c: capacity estimator (cloned internally).
        random_state: random seed forwarded to estimator when supported.
    Returns:
        dict with kept indices, c_hat predictions, and slack values,
        or none if the fold is skipped.
    """

    from sklearn.base import clone
    from src.evaluators.resampling import _drop_nan_rows
    from src.vectorizers.scalers import _standardizer

    X_tr, Z_tr, y_tr, _ = _drop_nan_rows(
        X = X.iloc[train_idx], Z = Z.iloc[train_idx],
        y = y_star.iloc[train_idx].values.astype(float),
        feat_x = feat_x, feat_z = feat_z,
    )
    X_te, Z_te, y_te, kept_te = _drop_nan_rows(
        X = X.iloc[test_idx], Z = Z.iloc[test_idx],
        y = y_star.iloc[test_idx].values.astype(float),
        feat_x = feat_x, feat_z = feat_z,
    )

    if len(X_tr) < 2 or len(X_te) == 0:
        return None

    X_te = X_te[feat_x]
    X_tr_s, x_sc = _standardizer(X_tr, feat_x)
    X_tr_s = X_tr_s[feat_x].values.astype(float)
    X_te_s = x_sc.transform(X_te.astype(float))

    mc = clone(estimator_c)
    if hasattr(mc, "random_state"):
        mc.set_params(random_state = random_state)
    mc.fit(X_tr_s, y_tr)
    c_hat = mc.predict(X_te_s).astype(float)

    kept_indices = test_idx[kept_te]

    return {
        "kept_indices": kept_indices,
        "c_hat": c_hat,
        "slack": y_te - c_hat,
    }


## --------------------------------------------------------------------------
## slack prediction fold worker (attribution step 2)
## --------------------------------------------------------------------------
def _run_slack_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feat_df: pd.DataFrame,
    feats: list[str],
    slack_oof: np.ndarray,
    c_hat_oof: np.ndarray,
    y_star: pd.Series,
    estimator_r,
    groups: np.ndarray,
    random_state: int = 42,
    ) -> dict | None:

    """
    Desc: execute a single fold predicting slack from features and
          computing frontier metrics for the reconstructed prediction.
    Args:
        train_idx: training indices.
        test_idx: test indices.
        feat_df: feature dataframe for this condition.
        feats: feature column names.
        slack_oof: out-of-fold slack array from step 1.
        c_hat_oof: out-of-fold C(X) predictions from step 1.
        y_star: log-transformed target series.
        estimator_r: residual estimator (cloned internally).
        groups: group labels array.
        random_state: random seed forwarded to estimator when supported.
    Returns:
        dict with group name, r_squared, frontier metrics, predictions,
        and index mapping, or none if the fold is skipped.
    """

    from sklearn.base import clone
    from src.vectorizers.scalers import _standardizer
    from src.evaluators.metrics import frontier_metrics

    group_name = groups[test_idx][0]

    F_tr = feat_df.iloc[train_idx]
    s_tr = slack_oof[train_idx]
    F_te = feat_df.iloc[test_idx]
    s_te = slack_oof[test_idx]
    y_te = y_star.iloc[test_idx].values.astype(float)

    ## mask: valid slack and valid features
    mask_tr = (
        F_tr[feats].notna().all(axis = 1)
        & np.isfinite(s_tr)
    )
    mask_te = (
        F_te[feats].notna().all(axis = 1)
        & np.isfinite(s_te)
    )
    F_tr = F_tr.loc[mask_tr]
    s_tr = s_tr[mask_tr.values]
    F_te = F_te.loc[mask_te]
    s_te = s_te[mask_te.values]
    y_te = y_te[mask_te.values]

    if len(F_tr) < 2 or len(F_te) == 0:
        return None

    F_te = F_te[feats]
    F_tr_s, f_sc = _standardizer(F_tr, feats)
    F_tr_s = F_tr_s[feats].values.astype(float)
    F_te_s = f_sc.transform(F_te.astype(float))

    ## fit residual estimator on slack
    mr = clone(estimator_r)
    if hasattr(mr, "random_state"):
        mr.set_params(random_state = random_state)
    mr.fit(F_tr_s, s_tr)
    s_pred = mr.predict(F_te_s).astype(float)

    ## identifiability: center predictions
    s_pred_tr = mr.predict(F_tr_s).astype(float)
    s_pred = (s_pred - np.mean(s_pred_tr)).astype(float)

    ## r-squared on slack
    ss_res = np.sum((s_te - s_pred) ** 2)
    ss_tot = np.sum((s_te - np.mean(s_te)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    ## reconstruct full prediction: y_pred = c_hat + s_pred
    c_te = c_hat_oof[test_idx][mask_te.values]
    y_pred = (c_te + s_pred).astype(float)

    kept_indices = test_idx[mask_te.values]
    frontier = frontier_metrics(y_true = y_te, y_pred = y_pred)

    return {
        "group_name": group_name,
        "r_squared": r2,
        "frontier": frontier,
        "kept_indices": kept_indices,
        "y_pred": y_pred,
    }


## --------------------------------------------------------------------------
## separation training
## --------------------------------------------------------------------------
def train_decomposed_separation(
    data: pd.DataFrame,
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_repeats: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    ) -> dict[str, Any]:

    """
    Desc: Run raw additive separation jobs. Post-processing is handled
        separately by compile_decomposed_separation.
    Args:
        data: training data with features, target, and group columns.
        models: mapping of model name to estimator with .estimator_c and
                .estimator_r attributes.
        feat_x: graph invariant feature column names.
        feat_z: process signature feature column names.
        target: target column name.
        group: group column name for logo splitting.
        n_repeats: number of repeated cv seeds to average predictions.
        random_state: base random state for seed reproducibility.
        n_jobs: number of parallel model workers (-1 for all cores).
    Returns:
        Dictionary with raw model outputs from the separation evaluation.
    """

    from joblib import Parallel, delayed
    from src.vectorizers.scalers import _log_transformer

    feat_x = list(feat_x)
    feat_z = list(feat_z)
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    ## create interaction features
    interaction_cols = []
    interaction_data = {}
    for x_col in feat_x:
        for z_col in feat_z:
            int_col = f"{x_col}_x_{z_col}"
            interaction_data[int_col] = (
                pd.to_numeric(data[x_col], errors = "coerce")
                * pd.to_numeric(data[z_col], errors = "coerce")
            )
            interaction_cols.append(int_col)

    data_aug = pd.concat([data, pd.DataFrame(interaction_data, index = data.index)], axis = 1)

    feat_z_aug = feat_z + interaction_cols

    ## target and group arrays for per-dataset tracking
    y_star_all = _log_transformer(data[target]).astype(float).values
    groups_all = data[group].values
    names_all = data["name"].values if "name" in data.columns else np.arange(len(data))
    model_items = list(models.items())

    if model_items:
        with _tqdm_joblib(total = len(model_items), desc = "Separation evaluation"):
            model_outputs = Parallel(n_jobs = n_jobs, verbose = 0)(
            delayed(_eval_separation_model)(
                    model_name = model_name,
                    model = model,
                    data = data,
                    data_aug = data_aug,
                    feat_x = feat_x,
                    feat_z = feat_z,
                    feat_z_aug = feat_z_aug,
                    interaction_cols = interaction_cols,
                    target = target,
                    group = group,
                    n_repeats = n_repeats,
                    random_state = random_state,
                    y_star_all = y_star_all,
                    groups_all = groups_all,
                    names_all = names_all,
                )
                for model_name, model in model_items
            )
    else:
        model_outputs = list()

    return {
        "model_outputs": model_outputs,
        "n_repeats": n_repeats,
        "random_state": random_state,
    }


## --------------------------------------------------------------------------
## separation compilation
## --------------------------------------------------------------------------
def compile_decomposed_separation(
    results: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Desc: Compile raw separation outputs into frontier and prediction tables.
        Consensus metrics (rho, rbo, dcr, ci) are computed per
        (model, specification, group) from the OOF predictions and merged
        onto the frontier rows alongside the frontier metrics.
    Args:
        results: Dictionary returned by train_decomposed_separation.
    Returns:
        Tuple of (frontier results dataframe, per-dataset predictions dataframe).
    """

    from src.evaluators.metrics import consensus_metrics

    model_outputs = results.get("model_outputs", list())
    frontier_rows = []
    prediction_rows = []
    for model_results, model_predictions in model_outputs:
        frontier_rows.extend(model_results)
        prediction_rows.extend(model_predictions)

    frontier_df = pd.DataFrame(frontier_rows)
    prediction_df = pd.DataFrame(prediction_rows)

    ## compute per-(model, specification, group) consensus metrics from OOF predictions
    if not prediction_df.empty and {"model", "specification", "group", "y_true", "y_pred"}.issubset(prediction_df.columns):
        consensus_rows = []
        for (model, spec, grp), grp_df in prediction_df.groupby(
            ["model", "specification", "group"], observed = True, sort = False
        ):
            y_true = grp_df["y_true"].to_numpy(dtype = float)
            y_pred = grp_df["y_pred"].to_numpy(dtype = float)
            valid = np.isfinite(y_true) & np.isfinite(y_pred)
            if valid.sum() >= 2:
                cm = consensus_metrics(y_true[valid], y_pred[valid])
            else:
                cm = {k: np.nan for k in ["r", "rho", "tau", "rbo", "dcr", "ci"]}
            consensus_rows.append({"model": model, "specification": spec, "group": grp, **cm})

        if consensus_rows:
            consensus_df = pd.DataFrame(consensus_rows)
            frontier_df = frontier_df.merge(
                consensus_df[["model", "specification", "group", "rho", "rbo", "dcr", "ci"]],
                on = ["model", "specification", "group"],
                how = "left",
            )

    return frontier_df, prediction_df


## --------------------------------------------------------------------------
## separation evaluation wrapper
## --------------------------------------------------------------------------
def eval_decomposed_separation(
    data: pd.DataFrame,
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_repeats: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Desc: Convenience wrapper that runs separation training and then compiles
        raw outputs into analysis-ready tables.
    Args:
        data: training data with features, target, and group columns.
        models: mapping of model name to estimator with .estimator_c and
            .estimator_r attributes.
        feat_x: graph invariant feature column names.
        feat_z: process signature feature column names.
        target: target column name.
        group: group column name for logo splitting.
        n_repeats: number of repeated cv seeds to average predictions.
        random_state: base random state for seed reproducibility.
        n_jobs: number of parallel model workers (-1 for all cores).
    Returns:
        Tuple of (frontier results dataframe, per-dataset predictions dataframe).
    """

    raw = train_decomposed_separation(
        data = data,
        models = models,
        feat_x = feat_x,
        feat_z = feat_z,
        target = target,
        group = group,
        n_repeats = n_repeats,
        random_state = random_state,
        n_jobs = n_jobs,
    )
    return compile_decomposed_separation(results = raw)


## backward-compatible alias
def eval_separation(
    data: pd.DataFrame,
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_repeats: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Desc: Convenience alias for eval_decomposed_separation.
    Args:
        data: training data with features, target, and group columns.
        models: mapping of model name to estimator bundles.
        feat_x: graph invariant feature column names.
        feat_z: process signature feature column names.
        target: target column name.
        group: group column name for logo splitting.
        n_repeats: number of repeated cv seeds to average predictions.
        random_state: base random state for seed reproducibility.
        n_jobs: number of parallel model workers (-1 for all cores).
    Returns:
        Tuple of (frontier results dataframe, per-dataset predictions dataframe).
    """

    return eval_decomposed_separation(
        data = data,
        models = models,
        feat_x = feat_x,
        feat_z = feat_z,
        target = target,
        group = group,
        n_repeats = n_repeats,
        random_state = random_state,
        n_jobs = n_jobs,
    )


## --------------------------------------------------------------------------
## attribution model worker
## --------------------------------------------------------------------------
def _eval_attribution_model(
    model_name: str,
    model,
    data: pd.DataFrame,
    feat_x: list[str],
    feat_z: list[str],
    target: str,
    group: str,
    n_repeats: int,
    random_state: int,
    ) -> tuple[list[dict], list[dict]]:

    """
    Desc: evaluate residual attribution for a single model. fold-level work stays
          sequential because there are only five folds; model-level
          parallelism gives better throughput with less overhead.
    Args:
        model_name: display name of the model family.
        model: model bundle with estimator_c and estimator_r.
        data: evaluation dataframe.
        feat_x: graph invariant feature names.
        feat_z: process signature feature names.
        target: target column name.
        group: group column name.
        n_repeats: number of repeated cv seeds to average predictions.
        random_state: base random state for seed reproducibility.
    Returns:
        tuple of (frontier rows, prediction rows).
    """

    from sklearn.model_selection import LeaveOneGroupOut
    from src.vectorizers.scalers import _log_transformer

    X = data[feat_x].apply(pd.to_numeric, errors = "coerce")
    Z = data[feat_z].apply(pd.to_numeric, errors = "coerce")
    y_star = _log_transformer(data[target]).astype(float)
    groups = data[group].values
    names = data["name"].values if "name" in data.columns else np.arange(len(data))

    logo = LeaveOneGroupOut()
    fold_splits = list(logo.split(X.values, y_star.values, groups))

    conditions = [
        ("X_to_slack", feat_x, X),
        ("Z_to_slack", feat_z, Z),
    ]
    prediction_sums = {
        feat_label: np.zeros(shape = len(data), dtype = float)
        for feat_label, _, _ in conditions
    }
    prediction_counts = {
        feat_label: np.zeros(shape = len(data), dtype = int)
        for feat_label, _, _ in conditions
    }
    frontier_groups = {feat_label: dict() for feat_label, _, _ in conditions}

    for repeat_idx in range(n_repeats):
        seed = int(random_state) + repeat_idx
        slack_oof = np.full(len(data), np.nan)
        c_hat_oof = np.full(len(data), np.nan)

        for train_idx, test_idx in fold_splits:
            result = _run_capacity_fold(
                train_idx = train_idx,
                test_idx = test_idx,
                X = X,
                Z = Z,
                y_star = y_star,
                feat_x = feat_x,
                feat_z = feat_z,
                estimator_c = model.estimator_c,
                random_state = seed,
            )
            if result is None:
                continue
            slack_oof[result["kept_indices"]] = result["slack"]
            c_hat_oof[result["kept_indices"]] = result["c_hat"]

        for feat_label, feats, feat_df in conditions:
            for train_idx, test_idx in fold_splits:
                result = _run_slack_fold(
                    train_idx = train_idx,
                    test_idx = test_idx,
                    feat_df = feat_df,
                    feats = feats,
                    slack_oof = slack_oof,
                    c_hat_oof = c_hat_oof,
                    y_star = y_star,
                    estimator_r = model.estimator_r,
                    groups = groups,
                    random_state = seed,
                )
                if result is None:
                    continue
                prediction_sums[feat_label][result["kept_indices"]] += result["y_pred"]
                prediction_counts[feat_label][result["kept_indices"]] += 1
                grp = result["group_name"]
                if grp not in frontier_groups[feat_label]:
                    frontier_groups[feat_label][grp] = list()
                frontier_groups[feat_label][grp].append({
                    "r_squared": result["r_squared"],
                    **result["frontier"],
                })

    results = []
    predictions = []
    for feat_label, _, _ in conditions:
        for grp in sorted(frontier_groups[feat_label].keys()):
            metrics_avg = pd.DataFrame(frontier_groups[feat_label][grp]).mean().to_dict()
            results.append({
                "model": model_name,
                "residual_features": feat_label,
                "group": grp,
                **metrics_avg,
            })

        y_pred_out = np.full(shape = len(data), fill_value = np.nan)
        valid_pred = prediction_counts[feat_label] > 0
        y_pred_out[valid_pred] = (
            prediction_sums[feat_label][valid_pred]
            / prediction_counts[feat_label][valid_pred]
        )
        for i in range(len(data)):
            if np.isfinite(y_pred_out[i]) and np.isfinite(y_star.iloc[i]):
                predictions.append({
                    "model": model_name,
                    "residual_features": feat_label,
                    "dataset": names[i],
                    "group": groups[i],
                    "y_true": float(y_star.iloc[i]),
                    "y_pred": y_pred_out[i],
                    "abs_error": abs(float(y_star.iloc[i]) - y_pred_out[i]),
                })

    return results, predictions


## --------------------------------------------------------------------------
## attribution training
## --------------------------------------------------------------------------
def train_decomposed_attribution(
    data: pd.DataFrame,
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_repeats: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    ) -> dict[str, Any]:

    """
    Desc: Run raw residual attribution jobs. Post-processing is handled separately
        by compile_decomposed_attribution.
    Args:
        data: training data with features, target, and group columns.
        models: mapping of model name to estimator with .estimator_c
                and .estimator_r attributes.
        feat_x: graph invariant feature column names.
        feat_z: process signature feature column names.
        target: target column name.
        group: group column name for logo splitting.
        n_repeats: number of repeated cv seeds to average predictions.
        random_state: base random state for seed reproducibility.
        n_jobs: number of parallel model workers (-1 for all cores).
    Returns:
        Dictionary with raw model outputs from the residual attribution evaluation.
    """

    from joblib import Parallel, delayed

    feat_x = list(feat_x)
    feat_z = list(feat_z)
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    model_items = list(models.items())

    if model_items:
        with _tqdm_joblib(total = len(model_items), desc = "Residual attribution evaluation"):
            model_outputs = Parallel(n_jobs = n_jobs, verbose = 0)(
            delayed(_eval_attribution_model)(
                    model_name = model_name,
                    model = model,
                    data = data,
                    feat_x = feat_x,
                    feat_z = feat_z,
                    target = target,
                    group = group,
                    n_repeats = n_repeats,
                    random_state = random_state,
                )
                for model_name, model in model_items
            )
    else:
        model_outputs = list()

    return {
        "model_outputs": model_outputs,
        "n_repeats": n_repeats,
        "random_state": random_state,
    }


## --------------------------------------------------------------------------
## attribution compilation
## --------------------------------------------------------------------------
def compile_decomposed_attribution(
    results: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Desc: Compile raw residual attribution outputs into frontier and prediction tables.
    Args:
        results: Dictionary returned by train_decomposed_attribution.
    Returns:
        Tuple of (frontier results dataframe, per-dataset predictions dataframe).
    """

    model_outputs = results.get("model_outputs", list())
    frontier_rows = []
    prediction_rows = []
    for model_results, model_predictions in model_outputs:
        frontier_rows.extend(model_results)
        prediction_rows.extend(model_predictions)

    return pd.DataFrame(frontier_rows), pd.DataFrame(prediction_rows)


## --------------------------------------------------------------------------
## attribution evaluation wrapper
## --------------------------------------------------------------------------
def eval_decomposed_attribution(
    data: pd.DataFrame,
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_repeats: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Desc: Convenience wrapper that runs residual attribution training and then
        compiles raw outputs into analysis-ready tables.
    Args:
        data: training data with features, target, and group columns.
        models: mapping of model name to estimator bundles.
        feat_x: graph invariant feature column names.
        feat_z: process signature feature column names.
        target: target column name.
        group: group column name for logo splitting.
        n_repeats: number of repeated cv seeds to average predictions.
        random_state: base random state for seed reproducibility.
        n_jobs: number of parallel model workers (-1 for all cores).
    Returns:
        Tuple of (frontier results dataframe, per-dataset predictions dataframe).
    """

    raw = train_decomposed_attribution(
        data = data,
        models = models,
        feat_x = feat_x,
        feat_z = feat_z,
        target = target,
        group = group,
        n_repeats = n_repeats,
        random_state = random_state,
        n_jobs = n_jobs,
    )
    return compile_decomposed_attribution(results = raw)


## backward-compatible alias
def eval_attribution(
    data: pd.DataFrame,
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_repeats: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Desc: Convenience alias for eval_decomposed_attribution.
    Args:
        data: training data with features, target, and group columns.
        models: mapping of model name to estimator bundles.
        feat_x: graph invariant feature column names.
        feat_z: process signature feature column names.
        target: target column name.
        group: group column name for logo splitting.
        n_repeats: number of repeated cv seeds to average predictions.
        random_state: base random state for seed reproducibility.
        n_jobs: number of parallel model workers (-1 for all cores).
    Returns:
        Tuple of (frontier results dataframe, per-dataset predictions dataframe).
    """

    return eval_decomposed_attribution(
        data = data,
        models = models,
        feat_x = feat_x,
        feat_z = feat_z,
        target = target,
        group = group,
        n_repeats = n_repeats,
        random_state = random_state,
        n_jobs = n_jobs,
    )


## --------------------------------------------------------------------------
## decomposition specification summary
## --------------------------------------------------------------------------
def stat_decomposed_summary(
    results: pd.DataFrame,
    metric: str | None = None,
    spec_order: Sequence[str] = SPECIFICATION_ORDER,
    metrics: Sequence[str] = ("ei", "vr", "mv", "ms"),
    decimals: int = 4,
    ) -> pd.DataFrame:

    """
    Desc: Summarize median frontier metrics by decomposition specification.
    Args:
        results: Frontier result table returned by compile_decomposed_separation.
        metric: Shorthand for metric group. "ei" uses frontier metrics (ei, vr,
            mv, ms); "ci" uses consensus metrics (ci, rho, rbo, dcr). Overrides
            the metrics argument when provided.
        spec_order: Specification order for table display.
        metrics: Frontier metric columns to summarize.
        decimals: Number of decimals to round.
    Returns:
        Display-ready dataframe indexed by specification, with EI or CI shown
            as IQR when present.
    Raises:
        ValueError: If required columns are missing.
    """

    metric_groups = {
        "ei": FRONTIER_METRICS,
        "ci": CONSENSUS_METRICS
    }
    if metric is not None:
        if metric not in metric_groups:
            raise ValueError(f"metric must be 'ei' or 'ci', got {metric!r}")
        metrics = metric_groups[metric]
    metrics = list(metrics)
    missing = sorted({"specification", *metrics} - set(results.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    iqr_metric = next((metric for metric in ["ei", "ci"] if metric in metrics), None)
    metrics_ordered = (
        [iqr_metric, *[metric for metric in metrics if metric != iqr_metric]]
        if iqr_metric is not None else metrics
    )
    observed_specs = results["specification"].dropna().drop_duplicates().tolist()
    present_specs = set(observed_specs)
    spec_index = [spec for spec in spec_order if spec in present_specs]
    spec_index.extend([spec for spec in observed_specs if spec not in spec_index])
    grouped = results.groupby(by = "specification", observed = True)[metrics_ordered]
    table = grouped.median().reindex(index = spec_index)
    q1 = grouped.quantile(q = 0.25).reindex(index = spec_index)
    q3 = grouped.quantile(q = 0.75).reindex(index = spec_index)

    def _format_iqr(row: pd.Series) -> str:
        if decimals is None:
            return f"{row['median']} [{row['q1']}, {row['q3']}]"

        return (
            f"{row['median']:.{decimals}f} "
            f"[{row['q1']:.{decimals}f}, {row['q3']:.{decimals}f}]"
        )

    if iqr_metric is not None:
        table = table.astype({iqr_metric: object})
        table[iqr_metric] = pd.DataFrame(
            data = {
                "median": table[iqr_metric],
                "q1": q1[iqr_metric],
                "q3": q3[iqr_metric],
            },
            index = table.index,
        ).apply(
            func = _format_iqr,
            axis = 1,
        )

    rename_map = {
        metric: f"{metric.upper()} [IQR]" if metric == iqr_metric else metric.upper()
        for metric in metrics_ordered
    }
    table = table.reindex(columns = metrics_ordered).rename(columns = rename_map)
    table = table.rename(index = lambda spec: str(spec).replace("_", " ").title())
    table.index.name = "Specification"

    if decimals is not None:
        numeric_cols = list(table.select_dtypes(include = [np.number]).columns)
        table[numeric_cols] = table[numeric_cols].round(decimals)

    return table


## --------------------------------------------------------------------------
## decomposition attribution test
## --------------------------------------------------------------------------
def stat_decomposed_attribution(
    predictions: pd.DataFrame,
    decimals: int = 4,
    index: bool = True,
    ) -> pd.DataFrame:

    """
    Desc: Test whether Z predicts slack better than X after C(X) extraction.
    Args:
        predictions: Prediction table returned by compile_decomposed_attribution.
        decimals: Number of decimals to round.
        index: If True, set label columns as the DataFrame index.
    Returns:
        Paired Wilcoxon test table.
    Raises:
        ValueError: If required columns are missing.
    """

    required = {"model", "residual_features", "group", "abs_error"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    domain_err = (
        predictions
        .groupby(by = ["model", "residual_features", "group"], observed = True)["abs_error"]
        .mean()
        .reset_index()
    )
    x_slack_err = (
        domain_err
        .query("residual_features == 'X_to_slack'")
        .set_index(keys = ["model", "group"])["abs_error"]
    )
    z_slack_err = (
        domain_err
        .query("residual_features == 'Z_to_slack'")
        .set_index(keys = ["model", "group"])["abs_error"]
    )

    delta = (x_slack_err - z_slack_err).dropna()
    n = len(delta)
    p_label = "One-sided p"
    tail_cols = ["Rank-biserial r", p_label, "Holm-adj. p", "Sig.", "Diff."]

    if n < 2 or int((delta != 0).sum()) < 2:
        p_value = np.nan
        r_effect = np.nan
    else:
        _, p_value = wilcoxon(x = delta.values, alternative = "greater")
        delta_nonzero = delta[delta != 0]
        ranks = rankdata(np.abs(delta_nonzero), method = "average")
        pos_rank_sum = float(np.sum(ranks[delta_nonzero > 0]))
        neg_rank_sum = float(np.sum(ranks[delta_nonzero < 0]))
        r_effect = (pos_rank_sum - neg_rank_sum) / float(np.sum(ranks))

    print(f"Paired One-Sided Test (Wilcoxon Signed-Rank): n = {n}")
    print("H₀: Δ MAE ≤ 0")
    print("H₁: Δ MAE > 0")
    print("Median Δ MAE: Median of paired differences, not the difference of marginal medians")
    print("Rank-biserial r: Paired effect size, positive values favor Z -> slack")
    print("One-sided p: Wilcoxon signed-rank p-value for H₁")
    print("Holm-adj. p: Holm-Bonferroni adjusted one-sided p-value")
    print("Diff.: Yes if Holm-adj. p < 0.05 and Median Δ MAE > 0")
    print("Significance codes reflect Holm-adj. p")
    print("*** p < 0.001, ** p < 0.01, * p < 0.05")

    holm = _holm_adjust([p_value])[0]
    summary = pd.DataFrame([{
        "Property": "Residual Attribution",
        "Comparison": "Topology vs Dynamics",
        "Median Δ MAE": delta.median(),
        "Rank-biserial r": r_effect,
        p_label: p_value,
        "Holm-adj. p": holm,
        "Sig.": "-" if not np.isfinite(holm) else _sig_code(float(holm)),
        "Diff.": (
            "Yes"
            if np.isfinite(holm) and np.isfinite(delta.median()) and holm < 0.05 and delta.median() > 0
            else "No"
            if np.isfinite(holm) and np.isfinite(delta.median())
            else "-"
        ),
    }])

    num_cols = [
        c for c in summary.columns
        if c.startswith("Median") or c in ["Rank-biserial r", p_label, "Holm-adj. p"]
    ]
    for col in num_cols:
        summary[col] = summary[col].apply(
            lambda v: f"{float(v):.{decimals}f}" if pd.notna(v) and np.isfinite(float(v)) else v
        )

    label_cols = ["Property", "Comparison"]
    summary = summary.reindex(columns = [*label_cols, "Median Δ MAE", *tail_cols])
    summary = summary.set_index(label_cols) if index else summary
    return summary.astype(object).where(pd.notna(summary), "-")


## --------------------------------------------------------------------------
## decomposition non-inferiority test
## --------------------------------------------------------------------------
def stat_decomposed_test(
    results: pd.DataFrame,
    delta: float = 0.05,
    metric: str = "ei",
    specs: Sequence[str] = (
        "interaction",
        "interaction_joint",
        "joint",
        "capacity_only",
        "dynamics_only",
    ),
    direction: str = "noninferiority",
    decimals: int = 4,
    index: bool = True,
    ) -> pd.DataFrame:

    """
    Desc:
        Test additive non-inferiority (richer specs) or inferiority (simpler
        specs) against a reference specification on a chosen metric.

    Args:
        results: Frontier result table returned by compile_decomposed_separation.
        delta: Non-inferiority / inferiority margin in metric units.
        metric: Column name of the performance metric to test ("ei" or "ci").
        specs: Specifications to compare against additive.
        direction: "noninferiority" tests that alternatives do not exceed
            additive by delta; "inferiority" tests that alternatives fall
            below additive by at least delta.
        decimals: Number of decimals to round.
        index: Whether to index the output by specification.
    Returns:
        Display-ready summary table.
    Raises:
        ValueError: If required columns are missing or direction is unknown.
    """

    if direction not in ("noninferiority", "inferiority"):
        raise ValueError(f"direction must be 'noninferiority' or 'inferiority', got {direction!r}")

    metric_label = metric.upper()
    required = {"model", "group", "specification", metric}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    additive_vals = (
        results
        .query("specification == 'additive'")
        .set_index(keys = ["model", "group"])[metric]
    )

    p_label = "One-sided p"
    decision_col = "NI." if direction == "noninferiority" else "Inf."
    tail_cols = ["Rank-biserial r", p_label, "Holm-adj. p", "Sig.", decision_col]
    median_col = f"Median Δ {metric_label}"
    rows = []
    for spec in specs:
        spec_vals = (
            results
            .query("specification == @spec")
            .set_index(keys = ["model", "group"])[metric]
        )
        gap = (spec_vals - additive_vals).dropna()
        n = len(gap)

        if direction == "noninferiority":
            margin_gap = gap - delta
        else:
            margin_gap = gap + delta

        if n < 2 or int((margin_gap != 0).sum()) < 2:
            p_w = np.nan
            r_effect = np.nan
        else:
            _, p_w = wilcoxon(x = margin_gap.values, alternative = "less")
            margin_nonzero = margin_gap[margin_gap != 0]
            ranks = rankdata(np.abs(margin_nonzero), method = "average")
            pos_rank_sum = float(np.sum(ranks[margin_nonzero > 0]))
            neg_rank_sum = float(np.sum(ranks[margin_nonzero < 0]))
            r_effect = (neg_rank_sum - pos_rank_sum) / float(np.sum(ranks))

        rows.append({
            "Specification": spec.replace("_", " ").title(),
            median_col: gap.median(),
            "Rank-biserial r": r_effect,
            p_label: p_w,
        })

    summary = pd.DataFrame(rows)
    holm = _holm_adjust(summary[p_label].to_numpy(dtype = float, copy = True))
    summary["Holm-adj. p"] = holm
    summary["Sig."] = summary["Holm-adj. p"].map(
        lambda p: "-" if not np.isfinite(p) else _sig_code(float(p))
    )
    median_gap = summary[median_col].to_numpy(dtype = float, copy = True)
    decision = np.full(shape = len(summary), fill_value = "-", dtype = object)
    valid_decision = np.isfinite(holm) & np.isfinite(median_gap)

    if direction == "noninferiority":
        decision[valid_decision] = np.where(
            (holm[valid_decision] < 0.05) & (median_gap[valid_decision] < delta),
            "Yes",
            "No",
        )
    else:
        decision[valid_decision] = np.where(
            (holm[valid_decision] < 0.05) & (median_gap[valid_decision] < -delta),
            "Yes",
            "No",
        )
    summary[decision_col] = decision

    n_by_spec = []
    for spec in specs:
        spec_vals = (
            results
            .query("specification == @spec")
            .set_index(keys = ["model", "group"])[metric]
        )
        n_by_spec.append(len((spec_vals - additive_vals).dropna()))
    n_display = int(n_by_spec[0]) if len(set(n_by_spec)) == 1 and len(n_by_spec) else "varies"

    if direction == "noninferiority":
        print(f"Paired Non-Inferiority Test (Wilcoxon Signed-Rank): n = {n_display}, δ = {delta}, metric = {metric_label}")
        print(f"H₀: Δ {metric_label} ≥ δ")
        print(f"H₁: Δ {metric_label} < δ")
        print(f"NI.: Yes if Holm-adj. p < 0.05 and Median Δ {metric_label} < δ")
    else:
        print(f"Paired Inferiority Test (Wilcoxon Signed-Rank): n = {n_display}, δ = {delta}, metric = {metric_label}")
        print(f"H₀: Δ {metric_label} ≥ -δ")
        print(f"H₁: Δ {metric_label} < -δ")
        print(f"Inf.: Yes if Holm-adj. p < 0.05 and Median Δ {metric_label} < -δ")
    print(f"Median Δ {metric_label}: Median of paired differences (spec − additive)")
    print("Rank-biserial r: Paired effect size, positive values favor the tested direction")
    print("One-sided p: Wilcoxon signed-rank p-value for H₁")
    print("Holm-adj. p: Holm-Bonferroni adjusted one-sided p-value")
    print("Significance codes reflect Holm-adj. p")
    print("*** p < 0.001, ** p < 0.01, * p < 0.05")

    summary = summary[["Specification", median_col, *tail_cols]]

    num_cols = [
        c for c in summary.columns
        if c.startswith("Median") or c in ["Rank-biserial r", p_label, "Holm-adj. p"]
    ]
    for col in num_cols:
        summary[col] = summary[col].apply(
            lambda v: f"{float(v):.{decimals}f}" if pd.notna(v) and np.isfinite(float(v)) else v
        )

    if index:
        summary = summary.set_index("Specification")
    return summary.astype(object).where(pd.notna(summary), "-")


## --------------------------------------------------------------------------
## additive specification summary
## --------------------------------------------------------------------------
def stat_decomposed_additive(
    results: pd.DataFrame,
    sufficiency: pd.DataFrame,
    delta: float = 0.05,
    decimals: int = 4,
    ) -> pd.DataFrame:

    """
    Desc: Summarize absolute additive EI and non-inferiority coverage.
    Args:
        results: Frontier result table returned by compile_decomposed_separation.
        sufficiency: Table returned by stat_decomposed_test.
        delta: Non-inferiority margin in EI points.
        decimals: Number of decimals to round.
    Returns:
        One-row display-ready additive summary table.
    Raises:
        ValueError: If required columns are missing.
    """

    required = {"model", "specification", "ei", "vr", "mv", "ms"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    additive_summary = (
        results
        .query("specification == 'additive'")
        .groupby(by = "model", observed = True)[["ei", "vr", "mv", "ms"]]
        .mean()
    )
    sufficiency_table = sufficiency.reset_index(drop = False)
    p_col = "Holm-adj. p" if "Holm-adj. p" in sufficiency_table.columns else "HOLM-ADJ. P"
    gap_col = "Median Δ EI" if "Median Δ EI" in sufficiency_table.columns else "MEAN Δ EI"
    n_sig = int((pd.to_numeric(sufficiency_table[p_col], errors = "coerce") < 0.05).sum())
    n_total = len(sufficiency_table)

    table = pd.DataFrame([{
        "MEAN EI": additive_summary["ei"].mean(),
        "SD EI": additive_summary["ei"].std(),
        "MAX MEDIAN Δ EI": pd.to_numeric(sufficiency_table[gap_col], errors = "coerce").max(),
        "MARGIN Δ": delta,
        "NON-INFERIOR": f"{n_sig}/{n_total}",
        "WORST ADJ. P": pd.to_numeric(sufficiency_table[p_col], errors = "coerce").max(),
    }])

    numeric_cols = list(table.select_dtypes(include = [np.number]).columns)
    table[numeric_cols] = table[numeric_cols].round(decimals)
    return table
