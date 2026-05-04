## libraries
import numpy as np
import pandas as pd
from typing import Any
from collections.abc import Mapping, Sequence

## modules
from src.evaluators.metrics import consensus_metrics
from src.evaluators.resampling import logo_cross_valid
from src.vectorizers.scalers import _log_transformer
from src.evaluators.metrics import CONSENSUS_METRICS

## ----------------------------------------------------------------------------
## prediction-quality evaluation
## ----------------------------------------------------------------------------
def eval_prediction_quality(
    data: pd.DataFrame,
    models: Mapping[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_repeats: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    ) -> pd.DataFrame:

    """
    Desc:
        Evaluates whether held-out LOGO predictions recover observed capacity
        structure within each held-out group.

    Args:
        data: Evaluation dataframe with features, target, and group columns.
        models: Mapping from model name to estimator bundle with `estimator_c`
            and `estimator_r` attributes.
        feat_x: Graph invariant feature column names.
        feat_z: Process signature feature column names.
        target: Target column name.
        group: Group column name for LOGO splitting and reporting.
        n_repeats: Number of repeated LOGO runs per model.
        random_state: Base random seed for reproducibility.
        n_jobs: Number of parallel jobs passed to LOGO cross-validation.

    Returns:
        DataFrame with one row per model and held-out group containing the
        consensus metrics between observed and predicted capacities.

    Raises:
        ValueError: If n_repeats is less than 1.
    """

    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")

    feat_x = list(feat_x)
    feat_z = list(feat_z)
    y_true_full = _log_transformer(data[target]).astype(float).to_numpy()
    groups = data[group].to_numpy()
    group_names = sorted(pd.Series(data = groups).dropna().unique())

    rows = list()
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        _, y_pred = logo_cross_valid(
            data = data,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = model.estimator_c,
            estimator_r = model.estimator_r,
            target = target,
            group = group,
            n_repeats = n_repeats,
            random_state = random_state,
            n_jobs = n_jobs,
        )

        for group_name in group_names:
            valid = (
                (groups == group_name)
                & np.isfinite(y_true_full)
                & np.isfinite(y_pred)
            )
            if int(np.sum(a = valid)) < 2:
                continue

            metrics = consensus_metrics(
                y_true = y_true_full[valid],
                y_pred = y_pred[valid],
            )
            rows.append({
                "model": model_name,
                "group": group_name,
                **metrics,
            })

    columns = ["model", "group", *CONSENSUS_METRICS]
    if not rows:
        return pd.DataFrame(columns = columns)

    result = pd.DataFrame(data = rows)
    return result[columns].sort_values(by = ["model", "group"]).reset_index(drop = True)


def summarize_prediction_quality(
    results: pd.DataFrame,
    group_col: str = "group",
    index_name: str = "Domain",
    group_label: str | None = None,
    n_repeats: int | None = 30,
    random_state: int | None = 42,
    decimals: int = 2,
    print_summary: bool = True,
    ) -> pd.DataFrame:

    """
    Desc:
        Builds a display table summarizing prediction-quality consensus metrics
        across fitted learners within each held-out group.

    Args:
        results: Prediction-quality result table from eval_prediction_quality.
        group_col: Column containing held-out group labels.
        index_name: Name assigned to the output table index.
        group_label: Human-readable group label used in printed notes.
        n_repeats: Number of repeated LOGO runs, for header display.
        random_state: Base random seed, for header display.
        decimals: Number of decimal places used in formatted output.
        print_summary: Whether to print the reporting convention header.

    Returns:
        Formatted summary table with CI median and IQR plus component medians.

    Raises:
        ValueError: If required metric columns are missing.
    """

    required_columns = {"model", group_col, "rho", "rbo", "dcr", "ci"}
    missing_columns = sorted(required_columns - set(results.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    metric_labels = {
        "rho": "ρ",
        "rbo": "RBO",
        "dcr": "DCR",
    }

    grouped = results.groupby(by = group_col, observed = True)
    ci_summary = grouped["ci"].agg(
        Median = "median",
        q1 = lambda values: values.quantile(q = 0.25),
        q3 = lambda values: values.quantile(q = 0.75),
    )
    ci_summary["CI [IQR]"] = ci_summary.apply(
        lambda row: (
            f"{row['Median']:.{decimals}f} "
            f"[{row['q1']:.{decimals}f}, {row['q3']:.{decimals}f}]"
        ),
        axis = 1,
    )

    component_summary = (
        grouped[["rho", "rbo", "dcr"]]
        .median()
        .rename(columns = metric_labels)
    )
    component_summary = component_summary.map(lambda value: f"{value:.{decimals}f}")

    result = (
        pd.concat(
            objs = [
                ci_summary[["CI [IQR]"]],
                component_summary,
            ],
            axis = 1,
        )
        .rename_axis(index = index_name)
        .sort_index()
    )

    if print_summary:
        n_models = results["model"].nunique()
        display_group = group_label if group_label is not None else index_name.lower()
        display_groups = display_group if display_group.endswith("s") else f"{display_group}s"
        if n_repeats is not None and random_state is not None:
            print(
                f"Cross-Validation: {n_models} models, {n_repeats} repeats "
                f"(seeds {random_state}-{random_state + n_repeats - 1})"
            )
        else:
            print(f"Cross-Validation: {n_models} models")
        print(
            "Across-model aggregation: median across learners within "
            f"held-out {display_groups}"
        )
        print(f"Resampling: LOGO {display_group} splits are fixed across repeats")
        print(
            f"Weighting: {display_groups} and models are equally weighted; "
            "results are not observation-weighted"
        )

    return result
