## libraries
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

## constants
DEFAULT_PARADIGM_ORDER = (
    "linear parametric",
    "non-linear ensemble",
    "neural networks",
)
DEFAULT_MODEL_TO_PARADIGM = {
    "linear_quantile": "linear parametric",
    "linear_convex": "linear parametric",
    "linear_laws": "linear parametric",
    "forest_quantile": "non-linear ensemble",
    "boosted_quantile": "non-linear ensemble",
    "xgboost_quantile": "non-linear ensemble",
    "neural_quantile": "neural networks",
    "neural_expectile": "neural networks",
    "neural_convex": "neural networks",
}
DEFAULT_CONSENSUS_PANELS = {
    "ρ": "rho",
    "RBO": "rbo",
    "DCR": "dcr",
}


def build_paradigm_consensus_matrices(
    results: pd.DataFrame,
    model_to_paradigm: Mapping[str, str] | None = None,
    paradigm_order: Sequence[str] | None = None,
    metric_panels: Mapping[str, str] | None = None,
    model_i: str = "model_i",
    model_j: str = "model_j",
    ) -> dict[str, pd.DataFrame]:

    """
    Desc:
        Build paradigm-by-paradigm median consensus matrices from pairwise
        learner comparison results.

    Args:
        results: Pairwise comparison table with model identifier columns and
            metric columns.
        model_to_paradigm: Mapping from model names to paradigm labels.
        paradigm_order: Row and column ordering for the output matrices.
        metric_panels: Mapping from display labels to metric column names.
        model_i: Column name for the first model in each pair.
        model_j: Column name for the second model in each pair.

    Returns:
        Dictionary mapping metric display labels to paradigm-by-paradigm
        median consensus matrices.

    Raises:
        ValueError: If required columns are missing, no metrics are provided,
            or a model lacks a paradigm label.
    """

    if model_to_paradigm is None:
        model_to_paradigm = DEFAULT_MODEL_TO_PARADIGM
    if paradigm_order is None:
        paradigm_order = DEFAULT_PARADIGM_ORDER
    if metric_panels is None:
        metric_panels = DEFAULT_CONSENSUS_PANELS

    paradigm_order = list(paradigm_order)
    metric_panels = dict(metric_panels)
    if not metric_panels:
        raise ValueError("metric_panels must contain at least one metric")

    required_columns = {model_i, model_j, *metric_panels.values()}
    missing_columns = sorted(required_columns - set(results.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    pairs = results.copy()
    pairs["_paradigm_i"] = pairs[model_i].map(model_to_paradigm)
    pairs["_paradigm_j"] = pairs[model_j].map(model_to_paradigm)

    missing_models = sorted(
        set(pairs.loc[pairs["_paradigm_i"].isna(), model_i])
        | set(pairs.loc[pairs["_paradigm_j"].isna(), model_j])
    )
    if missing_models:
        raise ValueError(f"Missing paradigm labels for models: {missing_models}")

    matrices = dict()
    for panel_label, metric in metric_panels.items():
        matrix = pd.DataFrame(
            data = np.nan,
            index = paradigm_order,
            columns = paradigm_order,
        )
        for row_label in paradigm_order:
            for column_label in paradigm_order:
                mask = (
                    (
                        (pairs["_paradigm_i"] == row_label)
                        & (pairs["_paradigm_j"] == column_label)
                    )
                    | (
                        (pairs["_paradigm_i"] == column_label)
                        & (pairs["_paradigm_j"] == row_label)
                    )
                )
                if mask.any():
                    matrix.loc[row_label, column_label] = pairs.loc[mask, metric].median()
        matrices[panel_label] = matrix

    return matrices


def plot_consensus(
    results: pd.DataFrame,
    model_to_paradigm: Mapping[str, str] | None = None,
    paradigm_order: Sequence[str] | None = None,
    metric_panels: Mapping[str, str] | None = None,
    model_i: str = "model_i",
    model_j: str = "model_j",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name: str = "Blues",
    decimals: int = 2,
    figsize: tuple[float, float] = (10.5, 3.8),
    title: str = "Learning consensus by estimator paradigm",
    colorbar_label: str = "Median pairwise consensus",
    show: bool = True,
    ) -> tuple[Figure, np.ndarray, dict[str, pd.DataFrame]]:

    """
    Desc:
        Plot one or more paradigm-by-paradigm consensus heatmaps with shared
        color scale and numeric cell annotations.

    Args:
        results: Pairwise comparison table with model identifier columns and
            metric columns.
        model_to_paradigm: Mapping from model names to paradigm labels.
        paradigm_order: Row and column ordering for the output matrices.
        metric_panels: Mapping from panel titles to metric column names.
        model_i: Column name for the first model in each pair.
        model_j: Column name for the second model in each pair.
        vmin: Lower bound for the shared color scale.
        vmax: Upper bound for the shared color scale.
        cmap_name: Matplotlib colormap name.
        decimals: Number of decimals shown in cell annotations.
        figsize: Figure size in inches.
        title: Figure title.
        colorbar_label: Label for the shared colorbar.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple containing the figure, axes array, and computed matrices.
    """

    matrices = build_paradigm_consensus_matrices(
        results = results,
        model_to_paradigm = model_to_paradigm,
        paradigm_order = paradigm_order,
        metric_panels = metric_panels,
        model_i = model_i,
        model_j = model_j,
    )

    if paradigm_order is None:
        paradigm_order = DEFAULT_PARADIGM_ORDER
    paradigm_order = list(paradigm_order)
    norm = Normalize(vmin = vmin, vmax = vmax)
    cmap = plt.get_cmap(cmap_name)

    def text_color(value: float) -> str:
        rgba = cmap(norm(value))
        luminance = rgba[0] * 0.299 + rgba[1] * 0.587 + rgba[2] * 0.114
        return "white" if luminance < 0.55 else "black"

    fig, axes = plt.subplots(
        nrows = 1,
        ncols = len(matrices),
        figsize = figsize,
        constrained_layout = True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    image = None

    for axis, (panel_label, matrix) in zip(axes_array, matrices.items()):
        image = axis.imshow(
            matrix.to_numpy(dtype = float),
            cmap = cmap,
            norm = norm,
            aspect = "equal",
        )
        for row_index, row_label in enumerate(paradigm_order):
            for column_index, column_label in enumerate(paradigm_order):
                value = matrix.loc[row_label, column_label]
                if pd.isna(value):
                    axis.text(
                        column_index, row_index, "NA",
                        ha = "center",
                        va = "center",
                        color = "black",
                        fontsize = 9,
                    )
                else:
                    axis.text(
                        column_index, row_index, f"{value:.{decimals}f}",
                        ha = "center",
                        va = "center",
                        color = text_color(float(value)),
                        fontsize = 9,
                    )
        axis.set_title(panel_label, fontsize = 11)
        axis.set_xticks(range(len(paradigm_order)))
        axis.set_yticks(range(len(paradigm_order)))
        axis.set_xticklabels(
            paradigm_order,
            rotation = 35,
            ha = "right",
            fontsize = 9,
        )
        axis.set_yticklabels(paradigm_order, fontsize = 9)
        axis.tick_params(axis = "both", which = "both", length = 0)
        for spine in axis.spines.values():
            spine.set_visible(False)

    if image is not None:
        colorbar = fig.colorbar(
            image,
            ax = list(axes_array),
            shrink = 0.82,
            pad = 0.02,
        )
        colorbar.set_label(colorbar_label, fontsize = 9)

    fig.suptitle(title, fontsize = 12)
    if show:
        plt.show()
    return fig, axes_array, matrices


def plot_decomposition_evidence(
    results: pd.DataFrame,
    noninferiority: pd.DataFrame,
    attribution: pd.DataFrame,
    delta: float,
    specification_order: Sequence[str] | None = None,
    figsize: tuple[float, float] = (13.5, 4.8),
    title: str = "Evidence for the additive decomposition",
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Plot a three-panel decomposition summary showing specification-level
        EI distributions, non-inferiority gaps relative to additive, and the
        residual-attribution contrast.

    Args:
        results: Decomposition results with at least specification and ei
            columns.
        noninferiority: Output table from stat_decomposed_test.
        attribution: Output table from stat_decomposed_attribution.
        delta: Non-inferiority margin used in the paired EI tests.
        specification_order: Display order for decomposition specifications.
        figsize: Figure size in inches.
        title: Figure title.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple containing the figure and axes array.

    Raises:
        ValueError: If required columns are missing.
    """

    if specification_order is None:
        specification_order = (
            "additive",
            "interaction",
            "interaction_joint",
            "joint",
            "capacity_only",
            "dynamics_only",
        )

    def format_specification(specification: str) -> str:
        return {
            "additive": "Additive",
            "interaction": "Interaction",
            "interaction_joint": "Interaction-Joint",
            "joint": "Joint",
            "capacity_only": "Capacity-Only",
            "dynamics_only": "Dynamics-Only",
        }.get(specification, str(specification).replace("_", " ").title())

    def coerce_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.replace(",", "", regex = False), errors = "coerce")

    required_results = {"specification", "ei"}
    missing_results = sorted(required_results - set(results.columns))
    if missing_results:
        raise ValueError(f"Missing required result columns: {missing_results}")

    ni_frame = noninferiority.reset_index().copy()
    attr_frame = attribution.reset_index().copy()

    required_ni = {"Specification", "Median Δ EI", "Holm-adj. p", "NI."}
    missing_ni = sorted(required_ni - set(ni_frame.columns))
    if missing_ni:
        raise ValueError(f"Missing required non-inferiority columns: {missing_ni}")

    required_attr = {"Property", "Comparison", "Median Δ MAE", "Holm-adj. p", "Diff."}
    missing_attr = sorted(required_attr - set(attr_frame.columns))
    if missing_attr:
        raise ValueError(f"Missing required attribution columns: {missing_attr}")

    summary = (
        results
        .groupby(by = "specification", observed = True)["ei"]
        .agg(
            median = "median",
            q1 = lambda values: values.quantile(0.25),
            q3 = lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    summary["display"] = summary["specification"].map(format_specification)

    ordered_specs = [spec for spec in specification_order if spec in set(summary["specification"])]
    ordered_specs.extend(
        spec for spec in summary["specification"].tolist() if spec not in ordered_specs
    )
    summary["specification"] = pd.Categorical(
        summary["specification"],
        categories = ordered_specs,
        ordered = True,
    )
    summary = summary.sort_values("specification", ascending = True).reset_index(drop = True)

    ni_frame["Median Δ EI"] = coerce_numeric(ni_frame["Median Δ EI"])
    ni_frame["Holm-adj. p"] = coerce_numeric(ni_frame["Holm-adj. p"])
    ni_frame["display"] = ni_frame["Specification"].astype(str)
    ni_frame["order"] = ni_frame["display"].map(
        {format_specification(spec): idx for idx, spec in enumerate(ordered_specs)}
    )
    ni_frame = ni_frame.sort_values("order").reset_index(drop = True)

    attr_frame["Median Δ MAE"] = coerce_numeric(attr_frame["Median Δ MAE"])
    attr_frame["Holm-adj. p"] = coerce_numeric(attr_frame["Holm-adj. p"])
    attr_row = attr_frame.iloc[0]

    fig = plt.figure(figsize = figsize, constrained_layout = True)
    grid = fig.add_gridspec(nrows = 1, ncols = 3, width_ratios = [1.5, 1.2, 0.9])
    axes = np.array([
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
    ], dtype = object)

    additive_color = "#0f4c5c"
    comparison_color = "#8fb8de"
    success_color = "#2d6a4f"
    warning_color = "#c16630"
    neutral_color = "#6c757d"

    y_summary = np.arange(len(summary))[::-1]
    for y_value, (_, row) in zip(y_summary, summary.iterrows()):
        is_additive = str(row["specification"]) == "additive"
        color = additive_color if is_additive else comparison_color
        axes[0].hlines(y = y_value, xmin = row["q1"], xmax = row["q3"], color = color, linewidth = 5, alpha = 0.8)
        axes[0].scatter(row["median"], y_value, s = 90 if is_additive else 70, color = color, edgecolor = "white", linewidth = 1.1, zorder = 3)
    axes[0].set_yticks(y_summary)
    axes[0].set_yticklabels(summary["display"], fontsize = 9)
    axes[0].set_xlabel("EI median with IQR", fontsize = 9)
    axes[0].set_title("Specification Performance", fontsize = 11)
    axes[0].grid(axis = "x", alpha = 0.18, linewidth = 0.8)
    for spine in ["top", "right", "left"]:
        axes[0].spines[spine].set_visible(False)
    axes[0].tick_params(axis = "y", length = 0)

    y_ni = np.arange(len(ni_frame))[::-1]
    gap_min = float(np.nanmin(np.r_[ni_frame["Median Δ EI"].to_numpy(dtype = float), 0.0]))
    gap_max = float(np.nanmax(np.r_[ni_frame["Median Δ EI"].to_numpy(dtype = float), delta]))
    gap_pad = max(0.02, 0.08 * (gap_max - gap_min if gap_max > gap_min else 1.0))
    axes[1].axvspan(gap_min - gap_pad, delta, color = "#d8f3dc", alpha = 0.55)
    axes[1].axvline(0.0, color = neutral_color, linewidth = 1.0)
    axes[1].axvline(delta, color = success_color, linewidth = 1.4, linestyle = "--")
    for y_value, (_, row) in zip(y_ni, ni_frame.iterrows()):
        decision = str(row["NI."])
        color = success_color if decision == "Yes" else warning_color if decision == "No" else neutral_color
        axes[1].hlines(y = y_value, xmin = 0.0, xmax = row["Median Δ EI"], color = color, linewidth = 2.5)
        axes[1].scatter(row["Median Δ EI"], y_value, s = 65, color = color, edgecolor = "white", linewidth = 1.0, zorder = 3)
        axes[1].text(
            x = gap_max + gap_pad * 0.15,
            y = y_value,
            s = decision,
            va = "center",
            ha = "left",
            fontsize = 8,
            color = color,
        )
    axes[1].set_xlim(gap_min - gap_pad, gap_max + gap_pad * 1.8)
    axes[1].set_yticks(y_ni)
    axes[1].set_yticklabels(ni_frame["display"], fontsize = 9)
    axes[1].set_xlabel("Median Δ EI", fontsize = 9)
    axes[1].set_title("Non-Inferiority vs Additive", fontsize = 11)
    axes[1].text(delta, len(ni_frame) - 0.2, f"δ = {delta:.2f}", color = success_color, fontsize = 8, ha = "left", va = "bottom")
    axes[1].grid(axis = "x", alpha = 0.18, linewidth = 0.8)
    for spine in ["top", "right", "left"]:
        axes[1].spines[spine].set_visible(False)
    axes[1].tick_params(axis = "y", length = 0)

    mae_value = float(attr_row["Median Δ MAE"])
    attr_color = success_color if str(attr_row["Diff."]) == "Yes" else warning_color
    attr_limit = max(abs(mae_value) * 1.35, 0.2)
    axes[2].axvline(0.0, color = neutral_color, linewidth = 1.0)
    axes[2].barh([0], [mae_value], color = attr_color, height = 0.42, alpha = 0.85)
    axes[2].scatter([mae_value], [0], s = 80, color = attr_color, edgecolor = "white", linewidth = 1.0, zorder = 3)
    axes[2].set_xlim(-attr_limit * 0.2, attr_limit)
    axes[2].set_ylim(-0.9, 0.9)
    axes[2].set_yticks([])
    axes[2].set_xlabel("Median Δ MAE", fontsize = 9)
    axes[2].set_title("Residual Attribution", fontsize = 11)
    axes[2].grid(axis = "x", alpha = 0.18, linewidth = 0.8)
    for spine in ["top", "right", "left"]:
        axes[2].spines[spine].set_visible(False)
    axes[2].text(
        x = 0.02,
        y = 0.96,
        s = (
            f"{attr_row['Comparison']}\n"
            f"Diff. = {attr_row['Diff.']}\n"
            f"Holm-adj. p = {attr_row['Holm-adj. p']}"
        ),
        transform = axes[2].transAxes,
        ha = "left",
        va = "top",
        fontsize = 8.5,
        bbox = {"boxstyle": "round,pad=0.35", "facecolor": "#f8f9fa", "edgecolor": "#d0d7de"},
    )
    axes[2].text(
        x = mae_value,
        y = 0.18,
        s = "favors dynamics" if mae_value >= 0 else "favors topology",
        ha = "center",
        va = "bottom",
        fontsize = 8,
        color = attr_color,
    )

    fig.suptitle(title, fontsize = 13)
    if show:
        plt.show()
    return fig, axes