## libraries
import colorsys
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.text import Text

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
    "CI": "ci",
    "ρ": "rho",
    "RBO": "rbo",
    "DCR": "dcr",
}
DEFAULT_PERTURBATION_ORDER = (
    "network",
    "invariants",
    "process",
    "signatures",
)
DEFAULT_PERTURBATION_PALETTE = {
    "network": "#2C6E91",
    "invariants": "#67AEDD",
    "process": "#7B5EA7",
    "signatures": "#A78CCB",
}
DEFAULT_METHOD_LINESTYLES = ("-", "--", ":")
DEFAULT_METHOD_SHADE_FACTORS = (0.65, 1.0, 1.45)


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
    cmap_name: str = "RdYlGn",
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

    for panel_index, (axis, (panel_label, matrix)) in enumerate(zip(axes_array, matrices.items())):
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
        if panel_index == 0:
            axis.set_yticklabels(paradigm_order, fontsize = 9)
        else:
            axis.set_yticklabels([])
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


def _filter_plot_track(
    results: pd.DataFrame,
    track: str | Sequence[str] | None,
    ) -> pd.DataFrame:

    if track is None or "track" not in results.columns:
        return results.copy()

    track_values = [track] if isinstance(track, str) else list(track)
    return results.loc[results["track"].isin(track_values)].copy()


def _resolve_plot_unit_column(
    results_transfer: pd.DataFrame,
    results_recovery: pd.DataFrame,
    unit_col: str | None,
    ) -> str:

    if unit_col is not None:
        if unit_col not in results_transfer.columns or unit_col not in results_recovery.columns:
            raise ValueError(f"Unit column '{unit_col}' must be present in both result tables")
        return unit_col

    for candidate in ("group", "domain"):
        if candidate in results_transfer.columns and candidate in results_recovery.columns:
            return candidate

    raise ValueError("Could not infer a shared unit column; expected 'group' or 'domain'")


def _sorted_plot_levels(values: pd.Series) -> list[object]:

    levels = pd.Series(values).dropna().unique().tolist()
    try:
        return sorted(levels, key = lambda value: float(value))
    except (TypeError, ValueError):
        return sorted(levels, key = lambda value: str(value))


def _normalized_plot_positions(levels: Sequence[object]) -> dict[object, float]:

    if len(levels) == 0:
        return dict()

    numeric = pd.to_numeric(pd.Series(levels), errors = "coerce").to_numpy(dtype = float)
    if np.all(np.isfinite(numeric)):
        span = float(np.max(numeric) - np.min(numeric))
        if span <= 0.0:
            positions = np.zeros(shape = len(levels), dtype = float)
        else:
            positions = (numeric - float(np.min(numeric))) / span
    else:
        positions = np.linspace(start = 0.0, stop = 1.0, num = len(levels))

    return {level: float(position) for level, position in zip(levels, positions)}


def _shade_plot_color(hex_color: str, factor: float) -> str:

    color = hex_color.lstrip("#")
    red = int(color[0:2], 16) / 255.0
    green = int(color[2:4], 16) / 255.0
    blue = int(color[4:6], 16) / 255.0
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    lightness_new = min(1.0, max(0.0, lightness * factor))
    red_new, green_new, blue_new = colorsys.hls_to_rgb(hue, lightness_new, saturation)
    return "#{:02x}{:02x}{:02x}".format(
        int(red_new * 255),
        int(green_new * 255),
        int(blue_new * 255),
    )


def _paired_plot_deltas(
    results: pd.DataFrame,
    value_col: str,
    delta_col: str,
    track: str | Sequence[str] | None,
    label_pert: str,
    label_base: str,
    unit_col: str,
    ) -> pd.DataFrame:

    data = _filter_plot_track(results = results, track = track)
    required_columns = {label_pert, "method", "intensity", "model", unit_col, value_col}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required perturbation columns: {missing_columns}")

    pair_columns = ["model", unit_col]
    if "track" in data.columns:
        pair_columns = ["track", *pair_columns]

    baseline = data.loc[data[label_pert] == label_base].copy()
    perturbed = data.loc[data[label_pert] != label_base].copy()
    if baseline.empty or perturbed.empty:
        return pd.DataFrame(columns = list(data.columns) + [delta_col])

    baseline_values = (
        baseline
        .groupby(by = pair_columns, observed = True)[value_col]
        .median()
        .reset_index()
        .rename(columns = {value_col: f"{value_col}_baseline"})
    )
    paired = perturbed.merge(
        right = baseline_values,
        on = pair_columns,
        how = "left",
    )
    paired[delta_col] = paired[value_col] - paired[f"{value_col}_baseline"]
    paired = paired.dropna(subset = [delta_col]).copy()
    return paired


def _aggregate_plot_sweep(
    paired: pd.DataFrame,
    delta_col: str,
    label_pert: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    group_columns = [label_pert, "method", "intensity"]
    aggregate = (
        paired
        .groupby(by = group_columns, observed = True)[delta_col]
        .median()
        .reset_index()
        .rename(columns = {delta_col: "delta"})
    )
    interval = (
        paired
        .groupby(by = group_columns, observed = True)[delta_col]
        .agg(
            q1 = lambda series: series.quantile(0.25),
            q3 = lambda series: series.quantile(0.75),
        )
        .reset_index()
    )
    return aggregate, interval


def _symmetric_plot_limit(
    aggregate: pd.DataFrame,
    interval: pd.DataFrame,
    margin: float,
    ) -> float:

    pieces = []
    if "delta" in aggregate.columns:
        pieces.append(pd.to_numeric(aggregate["delta"], errors = "coerce"))
    for column in ("q1", "q3"):
        if column in interval.columns:
            pieces.append(pd.to_numeric(interval[column], errors = "coerce"))

    finite_max = 0.0
    if pieces:
        values = pd.concat(objs = pieces, ignore_index = True).to_numpy(dtype = float)
        finite_values = np.abs(values[np.isfinite(values)])
        if finite_values.size:
            finite_max = float(np.max(finite_values))

    return max(finite_max * 1.12, float(margin) * 2.5, 0.05)


def plot_perturbation_superfigure(
    results_transfer: pd.DataFrame,
    results_recovery: pd.DataFrame,
    results_transfer_max: pd.DataFrame,
    results_recovery_max: pd.DataFrame,
    delta_ei: float,
    delta_ci: float,
    track: str | Sequence[str] | None = "frozen",
    perturbation_order: Sequence[str] | None = None,
    palette: Mapping[str, str] | None = None,
    method_linestyles: Sequence[str] | None = None,
    method_shade_factors: Sequence[float] | None = None,
    unit_col: str | None = None,
    sweep_ylim: tuple[float, float] = (-0.25, 0.25),
    figsize: tuple[float, float] = (14.2, 8.8),
    title: str = "Perturbation Stability Across Intensity and Joint Endpoint Response",
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Plot a perturbation superfigure that aligns EI and CI intensity
        sweeps with a max-intensity coordinate fingerprint by perturbation
        family.

    Args:
        results_transfer: Full perturbation transfer table with EI values.
        results_recovery: Full perturbation structural agreement table with CI
            values.
        results_transfer_max: Maximum-intensity transfer table including
            baseline rows.
        results_recovery_max: Maximum-intensity recovery table including
            baseline rows.
        delta_ei: EI equivalence margin.
        delta_ci: CI equivalence margin.
        track: Track or tracks to plot when a track column is present.
        perturbation_order: Column order for perturbation families.
        palette: Mapping from perturbation family to color.
        method_linestyles: Line styles used for method sweeps.
        method_shade_factors: Lightness factors used for method fingerprints.
        unit_col: Pairing column. If None, inferred from group/domain.
        sweep_ylim: Fixed y-axis limits for the EI and CI sweep rows.
        figsize: Figure size in inches.
        title: Figure title.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple containing the figure and 3 x n axes array.

    Raises:
        ValueError: If required columns are missing or no perturbation data are
            available.
    """

    unit_name = _resolve_plot_unit_column(
        results_transfer = results_transfer,
        results_recovery = results_recovery,
        unit_col = unit_col,
    )
    palette = dict(DEFAULT_PERTURBATION_PALETTE if palette is None else palette)
    method_linestyles = tuple(DEFAULT_METHOD_LINESTYLES if method_linestyles is None else method_linestyles)
    method_shade_factors = tuple(
        DEFAULT_METHOD_SHADE_FACTORS if method_shade_factors is None else method_shade_factors
    )

    transfer_paired = _paired_plot_deltas(
        results = results_transfer,
        value_col = "ei",
        delta_col = "delta_ei",
        track = track,
        label_pert = "perturbation",
        label_base = "baseline",
        unit_col = unit_name,
    )
    recovery_paired = _paired_plot_deltas(
        results = results_recovery,
        value_col = "ci",
        delta_col = "delta_ci",
        track = track,
        label_pert = "perturbation",
        label_base = "baseline",
        unit_col = unit_name,
    )
    transfer_aggregate, transfer_interval = _aggregate_plot_sweep(
        paired = transfer_paired,
        delta_col = "delta_ei",
        label_pert = "perturbation",
    )
    recovery_aggregate, recovery_interval = _aggregate_plot_sweep(
        paired = recovery_paired,
        delta_col = "delta_ci",
        label_pert = "perturbation",
    )

    transfer_max_paired = _paired_plot_deltas(
        results = results_transfer_max,
        value_col = "ei",
        delta_col = "delta_ei",
        track = track,
        label_pert = "perturbation",
        label_base = "baseline",
        unit_col = unit_name,
    )
    recovery_max_paired = _paired_plot_deltas(
        results = results_recovery_max,
        value_col = "ci",
        delta_col = "delta_ci",
        track = track,
        label_pert = "perturbation",
        label_base = "baseline",
        unit_col = unit_name,
    )

    fingerprint_columns = ["perturbation", "method", "model", unit_name]
    fingerprint = (
        transfer_max_paired[fingerprint_columns + ["delta_ei"]]
        .groupby(by = fingerprint_columns, observed = True)["delta_ei"]
        .median()
        .reset_index()
        .merge(
            right = (
                recovery_max_paired[fingerprint_columns + ["delta_ci"]]
                .groupby(by = fingerprint_columns, observed = True)["delta_ci"]
                .median()
                .reset_index()
            ),
            on = fingerprint_columns,
            how = "inner",
        )
    )

    available_perturbations = set(transfer_aggregate["perturbation"].unique())
    available_perturbations |= set(recovery_aggregate["perturbation"].unique())
    available_perturbations |= set(fingerprint["perturbation"].unique())
    if perturbation_order is None:
        ordered = [name for name in DEFAULT_PERTURBATION_ORDER if name in available_perturbations]
        ordered.extend(sorted(available_perturbations - set(ordered)))
        perturbation_order = ordered
    else:
        perturbation_order = [name for name in perturbation_order if name in available_perturbations]
    if not perturbation_order:
        raise ValueError("No perturbation rows are available for plotting")

    row_specs = [
        {
            "label": r"$\Delta$ EI",
            "margin": float(delta_ei),
            "aggregate": transfer_aggregate,
            "interval": transfer_interval,
        },
        {
            "label": r"$\Delta$ CI",
            "margin": float(delta_ci),
            "aggregate": recovery_aggregate,
            "interval": recovery_interval,
        },
    ]
    fig = plt.figure(figsize = figsize)
    outer_grid = fig.add_gridspec(
        nrows = 2,
        ncols = 1,
        height_ratios = [1.34, 1.62],
        hspace = 0.18,
    )
    sweep_grid = outer_grid[0].subgridspec(
        nrows = 2,
        ncols = len(perturbation_order),
        hspace = 0.10,
        wspace = 0.28,
    )
    fingerprint_grid = outer_grid[1].subgridspec(
        nrows = 1,
        ncols = len(perturbation_order),
        wspace = 0.28,
    )
    axes = np.empty(shape = (3, len(perturbation_order)), dtype = object)
    for column_index in range(len(perturbation_order)):
        if column_index == 0:
            axes[0, column_index] = fig.add_subplot(sweep_grid[0, column_index])
            axes[1, column_index] = fig.add_subplot(sweep_grid[1, column_index])
            axes[2, column_index] = fig.add_subplot(fingerprint_grid[0, column_index])
            continue
        axes[0, column_index] = fig.add_subplot(
            sweep_grid[0, column_index],
            sharey = axes[0, 0],
        )
        axes[1, column_index] = fig.add_subplot(
            sweep_grid[1, column_index],
            sharey = axes[1, 0],
        )
        axes[2, column_index] = fig.add_subplot(
            fingerprint_grid[0, column_index],
            sharex = axes[2, 0],
            sharey = axes[2, 0],
        )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top = 0.88, bottom = 0.08, left = 0.07, right = 0.985)
    neutral_color = "#C0C0C0"
    margin_color = "#E6F4EA"
    fingerprint_margin_color = "#CFEAD6"
    margin_edge_color = "#C9C9C9"
    margin_text_color = "#3D7F50"
    degradation_face = "#F2B8A8"
    degradation_text = "#8F2A2A"
    fingerprint_limit = 0.5
    fingerprint_ticks = np.linspace(-0.5, 0.5, 5)
    sweep_ticks = [float(sweep_ylim[0]), 0.0, float(sweep_ylim[1])]

    def _format_method_label(method: object) -> str:
        return str(method).replace("_", " ").title()

    def _compose_delta_ticks(
        base_ticks: Sequence[float],
        delta_ticks: Mapping[float, str],
        ) -> tuple[list[float], list[str]]:

        ordered_ticks = sorted(
            [float(tick) for tick in base_ticks] + [float(tick) for tick in delta_ticks.keys()]
        )
        merged_ticks: list[float] = []
        for tick in ordered_ticks:
            if merged_ticks and np.isclose(tick, merged_ticks[-1], atol = 1e-9, rtol = 0.0):
                continue
            merged_ticks.append(tick)

        labels = []
        for tick in merged_ticks:
            tick_label = None
            for delta_tick, delta_label in delta_ticks.items():
                if np.isclose(tick, float(delta_tick), atol = 1e-9, rtol = 0.0):
                    tick_label = delta_label
                    break
            labels.append(f"{tick:.2f}" if tick_label is None else tick_label)

        return merged_ticks, labels

    for row_index, row_spec in enumerate(row_specs):
        aggregate = row_spec["aggregate"]
        interval = row_spec["interval"]
        margin = row_spec["margin"]
        for column_index, perturbation in enumerate(perturbation_order):
            axis = axes[row_index, column_index]
            color = palette.get(perturbation, "#666666")
            aggregate_panel = aggregate.loc[aggregate["perturbation"] == perturbation].copy()
            interval_panel = interval.loc[interval["perturbation"] == perturbation].copy()

            if aggregate_panel.empty:
                axis.set_visible(False)
                continue

            axis.add_patch(
                Rectangle(
                    xy = (0.0, -float(margin)),
                    width = 1.0,
                    height = float(2.0 * margin),
                    facecolor = margin_color,
                    edgecolor = "none",
                    linewidth = 0.0,
                    zorder = 0,
                )
            )
            sweep_clip = Rectangle(
                xy = (0.0, float(sweep_ylim[0])),
                width = 1.0,
                height = float(sweep_ylim[1] - sweep_ylim[0]),
                transform = axis.transData,
            )
            axis.text(
                x = 0.03,
                y = float(margin) * 0.6,
                s = "Eq.",
                ha = "left",
                va = "center",
                fontsize = 6.8,
                color = margin_text_color,
                fontweight = "semibold",
                zorder = 5.5,
                bbox = {
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "#F3FBF5",
                    "edgecolor": "none",
                    "alpha": 0.92,
                },
            )
            axis.hlines(
                y = float(margin),
                xmin = 0.0,
                xmax = 1.0,
                colors = margin_edge_color,
                linestyles = "--",
                linewidths = 0.6,
                zorder = 2.0,
            )
            axis.hlines(
                y = -float(margin),
                xmin = 0.0,
                xmax = 1.0,
                colors = margin_edge_color,
                linestyles = "--",
                linewidths = 0.6,
                zorder = 2.0,
            )
            axis.hlines(
                y = 0.0,
                xmin = 0.0,
                xmax = 1.0,
                colors = neutral_color,
                linestyles = "--",
                linewidths = 0.85,
                zorder = 1,
            )
            axis.axvline(x = 1.0, color = "#000000", lw = 0.6, ls = "-", zorder = 4.5)

            methods = sorted(aggregate_panel["method"].unique())
            method_text_colors = {
                _format_method_label(method = method): _shade_plot_color(
                    hex_color = color,
                    factor = min(
                        method_shade_factors[method_index % len(method_shade_factors)],
                        1.18,
                    ),
                )
                for method_index, method in enumerate(methods)
            }
            for method_index, method in enumerate(methods):
                method_label = _format_method_label(method = method)
                line_style = method_linestyles[method_index % len(method_linestyles)]
                method_aggregate = aggregate_panel.loc[aggregate_panel["method"] == method].copy()
                method_interval = interval_panel.loc[interval_panel["method"] == method].copy()
                intensities = _sorted_plot_levels(values = method_aggregate["intensity"])
                intensity_lookup = _normalized_plot_positions(levels = intensities)
                method_aggregate["_x"] = method_aggregate["intensity"].map(intensity_lookup)
                method_interval["_x"] = method_interval["intensity"].map(intensity_lookup)
                method_aggregate = method_aggregate.sort_values(by = "_x")
                method_interval = method_interval.sort_values(by = "_x")

                if len(method_interval) > 1:
                    interval_artist = axis.fill_between(
                        method_interval["_x"].to_numpy(dtype = float),
                        method_interval["q1"].to_numpy(dtype = float),
                        method_interval["q3"].to_numpy(dtype = float),
                        color = color,
                        alpha = 0.20,
                        zorder = 2,
                        lw = 0,
                    )
                    interval_artist.set_clip_path(sweep_clip)
                line_artist = axis.plot(
                    method_aggregate["_x"].to_numpy(dtype = float),
                    method_aggregate["delta"].to_numpy(dtype = float),
                    color = color,
                    lw = 1.1,
                    ls = line_style,
                    marker = "o",
                    markersize = 4.4,
                    markerfacecolor = color,
                    markeredgecolor = color,
                    markeredgewidth = 0.0,
                    zorder = 4,
                    label = method_label,
                    solid_capstyle = "butt",
                )[0]
                line_artist.set_clip_path(sweep_clip)

            if row_index == 0:
                axis.set_title(
                    label = perturbation.capitalize(),
                    fontsize = 9.7,
                    color = color,
                    fontweight = "semibold",
                    pad = 6,
                )
                legend = axis.legend(
                    fontsize = 5.9,
                    loc = "lower left",
                    bbox_to_anchor = (0.03, 0.04),
                    borderaxespad = 0.0,
                    framealpha = 0.9,
                    edgecolor = "#D0D0D0",
                    handlelength = 1.9,
                    borderpad = 0.45,
                    labelspacing = 0.25,
                    handletextpad = 0.45,
                )
                for legend_text in legend.get_texts():
                    legend_text.set_color(method_text_colors.get(legend_text.get_text(), color))

            axis.set_xticks(ticks = np.linspace(start = 0.0, stop = 1.0, num = 5))
            if row_index == 0:
                axis.set_xticklabels(labels = [])
            else:
                axis.set_xticklabels(
                    labels = ["0", "0.25", "0.50", "0.75", "1\nMax"],
                    fontsize = 6.5,
                    color = "#555555",
                )
                axis.get_xticklabels()[-1].set_fontweight("bold")
            axis.set_xlim(left = 0.0, right = 1.03)
            axis.set_ylim(bottom = float(sweep_ylim[0]), top = float(sweep_ylim[1]))
            sweep_y_ticks, sweep_y_labels = _compose_delta_ticks(
                base_ticks = sweep_ticks,
                delta_ticks = {
                    -float(margin): r"$-\delta$",
                    float(margin): r"$+\delta$",
                },
            )
            axis.set_yticks(ticks = sweep_y_ticks)
            axis.set_yticklabels(labels = sweep_y_labels)
            for tick_label in axis.get_yticklabels():
                if tick_label.get_text() in {r"$-\delta$", r"$+\delta$"}:
                    tick_label.set_color("#8E8E8E")
            axis.tick_params(axis = "x", bottom = False)
            axis.tick_params(axis = "y", labelsize = 7)
            axis.spines[["top", "right", "bottom"]].set_visible(False)
            axis.spines["left"].set_linewidth(0.6)
            axis.spines["left"].set_color("#CCCCCC")
            axis.yaxis.grid(True, lw = 0.35, color = "#E8E8E8", zorder = 0)
            axis.set_axisbelow(True)

        axes[row_index, 0].set_ylabel(ylabel = row_spec["label"], fontsize = 9.2, labelpad = 8)

    for column_index, perturbation in enumerate(perturbation_order):
        axis = axes[2, column_index]
        color = palette.get(perturbation, "#666666")
        panel = fingerprint.loc[fingerprint["perturbation"] == perturbation].copy()
        if panel.empty:
            axis.set_visible(False)
            continue

        degradation_width = max(0.0, fingerprint_limit - float(delta_ei))
        degradation_height = max(0.0, fingerprint_limit - float(delta_ci))
        axis.add_patch(
            Rectangle(
                xy = (-fingerprint_limit, -fingerprint_limit),
                width = degradation_width,
                height = degradation_height,
                facecolor = degradation_face,
                edgecolor = "none",
                alpha = 0.42,
                zorder = 0,
            )
        )
        axis.text(
            x = -0.92 * fingerprint_limit,
            y = -0.95 * fingerprint_limit,
            s = "Degradation",
            ha = "left",
            va = "bottom",
            fontsize = 7.4,
            color = degradation_text,
            fontweight = "semibold",
        )
        axis.add_patch(
            Rectangle(
                xy = (-float(delta_ei), -float(delta_ci)),
                width = float(2.0 * delta_ei),
                height = float(2.0 * delta_ci),
                facecolor = fingerprint_margin_color,
                edgecolor = margin_edge_color,
                linewidth = 0.6,
                linestyle = "--",
                alpha = 0.82,
                zorder = 1.85,
            )
        )
        axis.text(
            x = float(delta_ei) * 0.95,
            y = float(delta_ci) * 0.85,
            s = "Eq.",
            ha = "right",
            va = "top",
            fontsize = 6.9,
            color = margin_text_color,
            fontweight = "semibold",
            zorder = 2.05,
        )
        axis.axhline(y = float(delta_ci), color = margin_edge_color, lw = 0.6, ls = "--", zorder = 2.0)
        axis.axhline(y = -float(delta_ci), color = margin_edge_color, lw = 0.6, ls = "--", zorder = 2.0)
        axis.axvline(x = float(delta_ei), color = margin_edge_color, lw = 0.6, ls = "--", zorder = 2.0)
        axis.axvline(x = -float(delta_ei), color = margin_edge_color, lw = 0.6, ls = "--", zorder = 2.0)
        axis.axhline(y = 0.0, color = "#7A7A7A", lw = 0.6, ls = "-", zorder = 2.1)
        axis.axvline(x = 0.0, color = "#7A7A7A", lw = 0.6, ls = "-", zorder = 2.1)

        methods = sorted(panel["method"].unique())
        for method_index, method in enumerate(methods):
            method_panel = panel.loc[panel["method"] == method].copy()
            if method_panel.empty:
                continue

            method_label = _format_method_label(method = method)
            method_depth = len(methods) - method_index
            shade = _shade_plot_color(
                hex_color = color,
                factor = method_shade_factors[method_index % len(method_shade_factors)],
            )
            x_values = method_panel["delta_ei"].to_numpy(dtype = float)
            y_values = method_panel["delta_ci"].to_numpy(dtype = float)
            axis.scatter(
                x = x_values,
                y = y_values,
                s = 18,
                color = shade,
                alpha = 0.26,
                edgecolors = "none",
                zorder = 3.0 + (0.1 * method_depth),
            )
            median_x = float(np.nanmedian(x_values))
            median_y = float(np.nanmedian(y_values))
            axis.plot(
                [0.0, median_x],
                [0.0, median_y],
                color = shade,
                lw = 1.1,
                alpha = 0.88,
                zorder = 4,
            )
            axis.scatter(
                x = [median_x],
                y = [median_y],
                s = 76,
                color = shade,
                edgecolors = "none",
                linewidths = 0.0,
                zorder = 5.0 + (0.1 * method_depth),
            )
            median_x_label = f"{median_x:+.2f}"
            median_y_label = f"{median_y:+.2f}"
            if median_x_label in {"+0.00", "-0.00"}:
                median_x_label = "0.00"
            if median_y_label in {"+0.00", "-0.00"}:
                median_y_label = "0.00"
            axis.text(
                x = 0.03,
                y = 0.950 - method_index * 0.061,
                s = f"{method_label}: $\\Delta$EI={median_x_label}, $\\Delta$CI={median_y_label}",
                transform = axis.transAxes,
                ha = "left",
                va = "top",
                fontsize = 6.9,
                color = shade,
                fontweight = "semibold",
            )

        axis.set_xlim(left = -fingerprint_limit, right = fingerprint_limit)
        axis.set_ylim(bottom = -fingerprint_limit, top = fingerprint_limit)
        fingerprint_x_ticks, fingerprint_x_labels = _compose_delta_ticks(
            base_ticks = fingerprint_ticks,
            delta_ticks = {
                -float(delta_ei): r"$-\delta$",
                float(delta_ei): r"$+\delta$",
            },
        )
        fingerprint_y_ticks, fingerprint_y_labels = _compose_delta_ticks(
            base_ticks = fingerprint_ticks,
            delta_ticks = {
                -float(delta_ci): r"$-\delta$",
                float(delta_ci): r"$+\delta$",
            },
        )
        axis.set_xticks(ticks = fingerprint_x_ticks)
        axis.set_xticklabels(labels = fingerprint_x_labels)
        axis.set_yticks(ticks = fingerprint_y_ticks)
        axis.set_yticklabels(labels = fingerprint_y_labels)
        for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
            if tick_label.get_text() in {r"$-\delta$", r"$+\delta$"}:
                tick_label.set_color("#8E8E8E")
        axis.tick_params(axis = "both", labelsize = 7)
        axis.grid(False)
        axis.set_aspect(aspect = "equal", adjustable = "box")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_visible(True)
        axis.spines["left"].set_color("#C9C9C9")
        axis.spines["bottom"].set_color("#C9C9C9")
        axis.spines["left"].set_linewidth(0.6)
        axis.spines["bottom"].set_linewidth(0.6)

    axes[2, 0].set_ylabel(ylabel = r"$\Delta$ CI at Maximum Perturbation Intensity", fontsize = 9.2, labelpad = 8)
    x_label_gap = axes[2, 0].yaxis.labelpad / 72.0 / float(fig.get_size_inches()[1])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fingerprint_xaxis_bbox = axes[2, 0].xaxis.get_tightbbox(renderer = renderer)
    sweep_xaxis_bbox = axes[1, 0].xaxis.get_tightbbox(renderer = renderer)
    if fingerprint_xaxis_bbox is None or sweep_xaxis_bbox is None:
        fingerprint_label_y = axes[2, 0].get_position().y0 - x_label_gap
        sweep_label_y = axes[1, 0].get_position().y0 - x_label_gap
    else:
        fingerprint_tick_bottom = fig.transFigure.inverted().transform(
            (0.0, fingerprint_xaxis_bbox.y0)
        )[1]
        sweep_tick_bottom = fig.transFigure.inverted().transform(
            (0.0, sweep_xaxis_bbox.y0)
        )[1]
        fingerprint_label_y = fingerprint_tick_bottom - x_label_gap
        sweep_label_y = sweep_tick_bottom - x_label_gap
    fig.text(
        x = 0.52,
        y = sweep_label_y,
        s = "Normalized Perturbation Intensity",
        ha = "center",
        va = "top",
        fontsize = 9.2,
    )
    fig.text(
        x = 0.52,
        y = fingerprint_label_y,
        s = r"$\Delta$ EI at Maximum Perturbation Intensity",
        ha = "center",
        va = "top",
        fontsize = 9.2,
    )
    for text_artist in fig.findobj(match = Text):
        text_artist.set_fontfamily(["Helvetica", "Arial", "sans-serif"])
        text_artist.set_fontsize(text_artist.get_fontsize() + 2.0)

    if show:
        plt.show()
    return fig, axes


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


def plot_decomposition_moneyshot(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    delta_ei: float,
    delta_ci: float,
    specification_order: Sequence[str] | None = None,
    figsize: tuple[float, float] = (14.8, 3.8),
    title: str = "Specification Fingerprint by Decomposition Alternative",
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Plot paired decomposition coordinate fingerprints showing how each
        alternative specification moves relative to the additive reference on
        efficiency and consensus.

    Args:
        results: Decomposition result table with model, group, specification,
            ei, and ci columns.
        predictions: Residual attribution prediction table retained for
            compatibility with previous notebook calls.
        delta_ei: Empirical EI margin used to contextualize paired gaps.
        delta_ci: Empirical CI margin used to contextualize paired gaps.
        specification_order: Display order for alternative specifications.
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
            "interaction",
            "interaction_joint",
            "joint",
            "capacity_only",
            "dynamics_only",
        )

    required_results = {"model", "group", "specification", "ei", "ci"}
    missing_results = sorted(required_results - set(results.columns))
    if missing_results:
        raise ValueError(f"Missing required result columns: {missing_results}")

    spec_labels = {
        "interaction": "Interaction",
        "interaction_joint": "Interaction Joint",
        "joint": "Joint",
        "capacity_only": "Capacity Only",
        "dynamics_only": "Dynamics Only",
    }
    spec_family = {
        "interaction": "Richer",
        "interaction_joint": "Richer",
        "joint": "Richer",
        "capacity_only": "Reduced",
        "dynamics_only": "Reduced",
    }
    spec_colors = {
        "interaction": "#9ecae1",
        "interaction_joint": "#4f83b6",
        "joint": "#1d4f7a",
        "capacity_only": "#f4a261",
        "dynamics_only": "#c96b27",
    }
    spec_markers = {
        "interaction": "o",
        "interaction_joint": "^",
        "joint": "D",
        "capacity_only": "s",
        "dynamics_only": "P",
    }

    additive = (
        results
        .query("specification == 'additive'")
        .set_index(keys = ["model", "group"])[["ei", "ci"]]
        .rename(columns = {"ei": "ei_additive", "ci": "ci_additive"})
    )

    paired_frames = []
    for specification in specification_order:
        alternative = (
            results
            .query("specification == @specification")
            .set_index(keys = ["model", "group"])[["ei", "ci"]]
            .rename(columns = {"ei": "ei_alt", "ci": "ci_alt"})
        )
        gaps = additive.join(other = alternative, how = "inner").dropna()
        if gaps.empty:
            continue
        paired_frames.append(
            gaps.assign(
                specification = specification,
                display = spec_labels.get(specification, specification.replace("_", " ").title()),
                family = spec_family.get(specification, "Other"),
                delta_ei = lambda frame: frame["ei_alt"] - frame["ei_additive"],
                delta_ci = lambda frame: frame["ci_alt"] - frame["ci_additive"],
            ).reset_index()[["model", "group", "specification", "display", "family", "delta_ei", "delta_ci"]]
        )

    paired = pd.concat(objs = paired_frames, ignore_index = True)
    paired["specification"] = pd.Categorical(
        values = paired["specification"],
        categories = list(specification_order),
        ordered = True,
    )

    from matplotlib.patches import Ellipse as MplEllipse

    def draw_cov_ellipse(axis: object, x: np.ndarray, y: np.ndarray, color: str, q: float = 0.80) -> None:

        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 3:
            return

        coordinates = np.column_stack((x[valid], y[valid]))
        covariance = np.cov(m = coordinates, rowvar = False)
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            return

        eigen_values, eigen_vectors = np.linalg.eigh(covariance)
        eigen_values = np.maximum(eigen_values, 0.0)
        order = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[order]
        eigen_vectors = eigen_vectors[:, order]

        if float(eigen_values[0]) <= 0.0:
            return

        scale = float(np.sqrt(-2.0 * np.log(max(1.0 - q, 1e-9))))
        width, height = 2.0 * scale * np.sqrt(eigen_values)
        angle = float(np.degrees(np.arctan2(eigen_vectors[1, 0], eigen_vectors[0, 0])))
        center = coordinates.mean(axis = 0)

        ellipse = MplEllipse(
            xy = (float(center[0]), float(center[1])),
            width = float(width),
            height = float(height),
            angle = angle,
            facecolor = color,
            edgecolor = color,
            linewidth = 1.1,
            alpha = 0.16,
            zorder = 1,
        )
        axis.add_patch(ellipse)

    fig, axes_raw = plt.subplots(
        nrows = 1,
        ncols = len(specification_order),
        figsize = figsize,
        sharex = True,
        sharey = True,
        constrained_layout = True,
    )
    axes = np.atleast_1d(axes_raw).astype(object)

    neutral_color = "#6c757d"
    margin_face = "#edf6ee"
    margin_edge = "#5b8c5a"
    additive_face = "#f6d8cc"
    additive_color = "#9a3412"

    axis_limits = (-0.5, 0.5)
    axis_ticks = [-0.5, -0.25, 0.0, 0.25, 0.5]

    for axis, specification in zip(axes, specification_order, strict = True):
        spec_points = paired.loc[paired["specification"] == specification]
        if spec_points.empty:
            axis.set_visible(False)
            continue

        color = spec_colors[specification]
        marker = spec_markers[specification]
        x = spec_points["delta_ei"].to_numpy(dtype = float)
        y = spec_points["delta_ci"].to_numpy(dtype = float)
        median_x = float(np.nanmedian(x))
        median_y = float(np.nanmedian(y))
        additive_share = float(np.mean((x < 0.0) & (y < 0.0)))
        margin_share = float(np.mean((np.abs(x) <= delta_ei) & (np.abs(y) <= delta_ci)))

        axis.add_patch(
            Rectangle(
                xy = (axis_limits[0], axis_limits[0]),
                width = abs(axis_limits[0]),
                height = abs(axis_limits[0]),
                facecolor = additive_face,
                edgecolor = "none",
                alpha = 0.72,
                zorder = 0,
            )
        )
        axis.add_patch(
            Rectangle(
                xy = (-delta_ei, -delta_ci),
                width = 2.0 * delta_ei,
                height = 2.0 * delta_ci,
                facecolor = margin_face,
                edgecolor = margin_edge,
                linewidth = 1.1,
                linestyle = "--",
                alpha = 0.72,
                zorder = 1,
            )
        )
        axis.axhline(y = 0.0, color = neutral_color, linewidth = 0.8, linestyle = "--", alpha = 0.42, zorder = 2)
        axis.axvline(x = 0.0, color = neutral_color, linewidth = 0.8, linestyle = "--", alpha = 0.42, zorder = 2)
        axis.scatter(
            x,
            y,
            s = 24,
            alpha = 0.30,
            color = color,
            linewidths = 0.0,
            zorder = 4,
        )
        draw_cov_ellipse(axis = axis, x = x, y = y, color = color, q = 0.80)
        axis.plot(
            [0.0, median_x],
            [0.0, median_y],
            color = color,
            linewidth = 1.3,
            alpha = 0.86,
            zorder = 5,
        )
        axis.scatter(
            [median_x],
            [median_y],
            s = 84,
            marker = marker,
            color = color,
            edgecolor = "white",
            linewidth = 1.2,
            zorder = 6,
        )
        axis.text(
            x = 0.04,
            y = 0.94,
            s = (
                f"P(both < 0) = {additive_share:.2f}\n"
                f"P(within δ) = {margin_share:.2f}"
            ),
            transform = axis.transAxes,
            ha = "left",
            va = "top",
            fontsize = 7.8,
            color = color,
            fontweight = "bold",
        )
        axis.text(
            x = 0.04,
            y = 0.04,
            s = "additive\nfavored",
            transform = axis.transAxes,
            ha = "left",
            va = "bottom",
            fontsize = 7.8,
            color = additive_color,
            fontweight = "bold",
        )
        axis.text(
            x = delta_ei + 0.015,
            y = delta_ci + 0.015,
            s = "±δ",
            ha = "left",
            va = "bottom",
            fontsize = 7.2,
            color = margin_edge,
        )
        axis.set_title(spec_labels[specification], fontsize = 10, fontweight = "bold", color = color)
        axis.set_xlim(*axis_limits)
        axis.set_ylim(*axis_limits)
        axis.set_xticks(ticks = axis_ticks)
        axis.set_yticks(ticks = axis_ticks)
        axis.set_aspect(aspect = "equal", adjustable = "box")
        axis.grid(alpha = 0.16, linewidth = 0.7)
        for spine in ["top", "right"]:
            axis.spines[spine].set_visible(False)

    axes[0].set_ylabel("Δ CI (alternative − additive)", fontsize = 9)
    fig.text(x = 0.5, y = 0.055, s = "Δ EI (alternative − additive)", ha = "center", fontsize = 9)
    fig.suptitle(title, x = 0.03, y = 1.02, ha = "left", fontsize = 13, fontweight = "bold")
    fig.text(
        x = 0.03,
        y = 0.935,
        s = (
            "Each point is a paired unit (model × domain): Δ = alternative − additive. "
            "Additive-favored loss appears in the lower-left quadrant; dashed boxes show ±δ practical-equivalence margins. "
            "Axes fixed to [-0.5, 0.5]."
        ),
        ha = "left",
        va = "top",
        fontsize = 8.5,
        color = "#666666",
        style = "italic",
    )

    if show:
        plt.show()
    return fig, axes


## ----------------------------------------------------------------------------
## residual handoff money shot
## ----------------------------------------------------------------------------
DEFAULT_DOMAIN_PALETTE = {
    "Earth & Physical Sciences":       "#C07D3A",
    "Life Sciences & Medicine":        "#3A7D55",
    "Technology & Information":        "#2C6E91",
    "Trade & Institutions":            "#B8860B",
    "Transportation & Infrastructure": "#8B3A3A",
}
DEFAULT_DOMAIN_LABELS = {
    "Earth & Physical Sciences":       "Earth\n& Physical",
    "Life Sciences & Medicine":        "Life\nSciences",
    "Technology & Information":        "Technology\n& Information",
    "Trade & Institutions":            "Trade &\nInstitutions",
    "Transportation & Infrastructure": "Transport &\nInfrastructure",
}
DEFAULT_MODEL_LABELS = {
    "linear_quantile":  "Linear Quantile",
    "linear_convex":    "Linear Convex",
    "linear_laws":      "Linear Laws",
    "forest_quantile":  "Forest Quantile",
    "boosted_quantile": "Boosted Quantile",
    "xgboost_quantile": "XGBoost Quantile",
    "neural_quantile":  "Neural Quantile",
    "neural_expectile": "Neural Expectile",
    "neural_convex":    "Neural Convex",
}


def _bootstrap_median_ci(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
    ) -> tuple[float, float, float]:

    """
    Desc: Percentile bootstrap CI for the median.
    Args:
        values: 1D array of finite values.
        n_bootstrap: Number of bootstrap resamples.
        rng: NumPy random generator.
        confidence: Confidence level for the interval.
    Returns:
        Tuple of (median, lower_bound, upper_bound).
    Raises:
        ValueError: If values is empty.
    """

    values = np.asarray(values, dtype = float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("values must contain at least one finite element")

    median = float(np.median(values))
    if values.size < 2:
        return median, median, median

    indices = rng.integers(low = 0, high = values.size, size = (n_bootstrap, values.size))
    medians = np.median(values[indices], axis = 1)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(medians, alpha))
    upper = float(np.quantile(medians, 1.0 - alpha))
    return median, lower, upper


def plot_residual_handoff_universality(
    predictions: pd.DataFrame,
    domain_palette: Mapping[str, str] | None = None,
    domain_labels: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    domain_order: Sequence[str] | None = None,
    model_order: Sequence[str] | None = None,
    n_bootstrap: int = 4000,
    random_state: int = 42,
    figsize: tuple[float, float] = (13.5, 11.2),
    title: str = "Process signatures explain what topology leaves behind",
    subtitle: str = (
        "After fitting the capacity stage $C(X)$, the residual slack $s = y^* - \\hat{C}(X)$ is\n"
        "predicted more accurately from process signatures $Z$ than from the same topology features $X$\n"
        "that produced it — universally, across every learning paradigm and every scientific domain."
    ),
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Plot a single figure summarizing the residual-handoff universality
        finding: after fitting C(X), the residual slack is better predicted
        from process features Z than from topology features X, robustly
        across models and domains.

    Args:
        predictions: Residual attribution prediction table with columns
            model, residual_features, dataset, group, abs_error.
        domain_palette: Mapping from domain label to color.
        domain_labels: Mapping from domain label to short display label.
        model_labels: Mapping from model identifier to display label.
        domain_order: Ordering of domains in summary panels.
        model_order: Ordering of models in the per-model forest.
        n_bootstrap: Bootstrap resample count for CI estimation.
        random_state: Seed for the bootstrap random generator.
        figsize: Figure size in inches.
        title: Figure title.
        subtitle: Figure subtitle shown beneath the title.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple containing the figure and a flat axes array.

    Raises:
        ValueError: If required columns are missing or only one residual
            feature source is present.
    """

    required = {"model", "residual_features", "dataset", "group", "abs_error"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sources = set(predictions["residual_features"].astype(str).unique())
    if not {"X_to_slack", "Z_to_slack"}.issubset(sources):
        raise ValueError(
            "predictions must include both X_to_slack and Z_to_slack residual_features"
        )

    domain_palette = dict(domain_palette or DEFAULT_DOMAIN_PALETTE)
    domain_labels = dict(domain_labels or DEFAULT_DOMAIN_LABELS)
    model_labels = dict(model_labels or DEFAULT_MODEL_LABELS)

    aggregated = (
        predictions
        .groupby(by = ["model", "dataset", "group", "residual_features"], observed = True)["abs_error"]
        .mean()
        .reset_index()
        .pivot_table(
            index = ["model", "dataset", "group"],
            columns = "residual_features",
            values = "abs_error",
            observed = True,
        )
        .rename_axis(columns = None)
        .reset_index()
        .dropna(subset = ["X_to_slack", "Z_to_slack"])
    )
    aggregated = aggregated.loc[
        (aggregated["X_to_slack"] > 0.0) & (aggregated["Z_to_slack"] > 0.0)
    ].copy()
    aggregated["log_ratio"] = np.log10(aggregated["X_to_slack"] / aggregated["Z_to_slack"])
    aggregated["favors_dynamics"] = aggregated["log_ratio"] > 0.0

    if domain_order is None:
        domain_order = list(
            dict.fromkeys(
                list(domain_palette.keys())
                + aggregated["group"].astype(str).unique().tolist()
            )
        )
        domain_order = [domain for domain in domain_order if domain in set(aggregated["group"])]
    if model_order is None:
        model_order = list(
            dict.fromkeys(
                list(model_labels.keys())
                + aggregated["model"].astype(str).unique().tolist()
            )
        )
        model_order = [model for model in model_order if model in set(aggregated["model"])]

    rng = np.random.default_rng(seed = random_state)

    domain_summary_rows = []
    for domain in domain_order:
        values = aggregated.loc[aggregated["group"] == domain, "log_ratio"].to_numpy(dtype = float)
        if values.size == 0:
            continue
        median, lower, upper = _bootstrap_median_ci(
            values = values,
            n_bootstrap = n_bootstrap,
            rng = rng,
        )
        domain_summary_rows.append({
            "domain": domain,
            "median": median,
            "lower": lower,
            "upper": upper,
            "share_dynamics": float((values > 0.0).mean()),
            "n": int(values.size),
        })
    domain_summary = pd.DataFrame(domain_summary_rows)

    model_summary_rows = []
    for model in model_order:
        values = aggregated.loc[aggregated["model"] == model, "log_ratio"].to_numpy(dtype = float)
        if values.size == 0:
            continue
        median, lower, upper = _bootstrap_median_ci(
            values = values,
            n_bootstrap = n_bootstrap,
            rng = rng,
        )
        model_summary_rows.append({
            "model": model,
            "median": median,
            "lower": lower,
            "upper": upper,
            "share_dynamics": float((values > 0.0).mean()),
            "n": int(values.size),
        })
    model_summary = pd.DataFrame(model_summary_rows)

    overall_median, overall_lower, overall_upper = _bootstrap_median_ci(
        values = aggregated["log_ratio"].to_numpy(dtype = float),
        n_bootstrap = n_bootstrap,
        rng = rng,
    )
    n_total = int(len(aggregated))
    n_dynamics = int(aggregated["favors_dynamics"].sum())
    share_dynamics = n_dynamics / n_total if n_total else float("nan")

    background_color = "#fbfaf6"
    grid_color = "#e3ddd0"
    diagonal_color = "#3a3a3a"
    dynamics_color = "#1f5e3a"
    topology_color = "#a14a2a"
    neutral_color = "#6c757d"
    headline_color = "#1f5e3a"

    fig = plt.figure(figsize = figsize, constrained_layout = False)
    fig.patch.set_facecolor(background_color)

    grid = fig.add_gridspec(
        nrows = 3,
        ncols = 2,
        height_ratios = [0.55, 2.6, 1.05],
        width_ratios = [2.7, 1.0],
        left = 0.07,
        right = 0.97,
        top = 0.80,
        bottom = 0.07,
        hspace = 0.45,
        wspace = 0.22,
    )
    kde_axis = fig.add_subplot(grid[0, 0])
    scatter_axis = fig.add_subplot(grid[1, 0])
    domain_axis = fig.add_subplot(grid[1, 1])
    model_axis = fig.add_subplot(grid[2, :])
    axes = np.array([kde_axis, scatter_axis, domain_axis, model_axis], dtype = object)

    for axis in axes:
        axis.set_facecolor(background_color)

    ## scatter ----------------------------------------------------------------
    x_values = aggregated["X_to_slack"].to_numpy(dtype = float)
    y_values = aggregated["Z_to_slack"].to_numpy(dtype = float)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0) & (y_values > 0.0)
    log_lower = float(np.log10(np.minimum(x_values[finite_mask], y_values[finite_mask]).min()))
    log_upper = float(np.log10(np.maximum(x_values[finite_mask], y_values[finite_mask]).max()))
    log_pad = max(0.15, 0.08 * (log_upper - log_lower))
    diagonal_lo = log_lower - log_pad
    diagonal_hi = log_upper + log_pad

    diagonal_x = np.linspace(start = diagonal_lo, stop = diagonal_hi, num = 200)
    scatter_axis.fill_between(
        x = 10.0 ** diagonal_x,
        y1 = 10.0 ** diagonal_x,
        y2 = 10.0 ** diagonal_hi,
        color = topology_color,
        alpha = 0.10,
        zorder = 0,
    )
    scatter_axis.fill_between(
        x = 10.0 ** diagonal_x,
        y1 = 10.0 ** diagonal_lo,
        y2 = 10.0 ** diagonal_x,
        color = dynamics_color,
        alpha = 0.12,
        zorder = 0,
    )
    scatter_axis.plot(
        10.0 ** diagonal_x,
        10.0 ** diagonal_x,
        color = diagonal_color,
        linewidth = 1.2,
        linestyle = "--",
        alpha = 0.7,
        zorder = 1,
    )

    for domain in domain_order:
        spec_mask = aggregated["group"] == domain
        if not spec_mask.any():
            continue
        scatter_axis.scatter(
            aggregated.loc[spec_mask, "X_to_slack"],
            aggregated.loc[spec_mask, "Z_to_slack"],
            s = 32,
            color = domain_palette.get(domain, "#555555"),
            edgecolor = "white",
            linewidth = 0.6,
            alpha = 0.88,
            label = domain,
            zorder = 3,
        )

    scatter_axis.set_xscale("log")
    scatter_axis.set_yscale("log")
    scatter_axis.set_xlim(10.0 ** diagonal_lo, 10.0 ** diagonal_hi)
    scatter_axis.set_ylim(10.0 ** diagonal_lo, 10.0 ** diagonal_hi)
    scatter_axis.set_aspect(aspect = "equal", adjustable = "box")
    scatter_axis.set_xlabel(
        "MAE predicting residual from topology  $X \\to s$",
        fontsize = 11,
    )
    scatter_axis.set_ylabel(
        "MAE predicting residual from process  $Z \\to s$",
        fontsize = 11,
    )
    scatter_axis.grid(which = "both", color = grid_color, alpha = 0.55, linewidth = 0.7)
    scatter_axis.tick_params(which = "both", labelsize = 9)
    for spine in ["top", "right"]:
        scatter_axis.spines[spine].set_visible(False)

    scatter_axis.text(
        x = 0.96,
        y = 0.55,
        s = "structure\nwins",
        transform = scatter_axis.transAxes,
        ha = "right",
        va = "center",
        fontsize = 11,
        fontweight = "bold",
        color = topology_color,
        alpha = 0.55,
    )
    scatter_axis.text(
        x = 0.05,
        y = 0.10,
        s = "process\nwins",
        transform = scatter_axis.transAxes,
        ha = "left",
        va = "center",
        fontsize = 11,
        fontweight = "bold",
        color = dynamics_color,
        alpha = 0.65,
    )

    headline = (
        f"{n_dynamics} of {n_total} matched (model × dataset) units favor process\n"
        f"({share_dynamics:.0%}; "
        f"median $\\log_{{10}}\\!\\left(\\mathrm{{MAE}}_X / \\mathrm{{MAE}}_Z\\right)$ = "
        f"{overall_median:+.2f}, 95% CI [{overall_lower:+.2f}, {overall_upper:+.2f}])"
    )
    scatter_axis.text(
        x = 0.03,
        y = 0.97,
        s = headline,
        transform = scatter_axis.transAxes,
        ha = "left",
        va = "top",
        fontsize = 10.5,
        color = headline_color,
        bbox = {
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": headline_color,
            "linewidth": 1.0,
            "alpha": 0.95,
        },
        zorder = 6,
    )

    legend = scatter_axis.legend(
        loc = "lower right",
        bbox_to_anchor = (1.0, 0.18),
        frameon = True,
        facecolor = "white",
        edgecolor = "#d0d7de",
        framealpha = 0.95,
        fontsize = 8.5,
        title = "Domain",
        title_fontsize = 9,
        markerscale = 1.3,
    )
    legend.get_frame().set_linewidth(0.7)

    ## kde strip --------------------------------------------------------------
    log_ratio_values = aggregated["log_ratio"].to_numpy(dtype = float)
    kde_x_max = float(np.max(np.abs(log_ratio_values))) * 1.05 + 0.1
    kde_x_min = -kde_x_max
    grid_x = np.linspace(start = kde_x_min, stop = kde_x_max, num = 512)
    bandwidth = max(
        0.08,
        1.06 * float(np.std(log_ratio_values)) * (log_ratio_values.size ** (-1.0 / 5.0)),
    )
    diff = grid_x[None, :] - log_ratio_values[:, None]
    kernel = np.exp(-0.5 * (diff / bandwidth) ** 2.0) / (bandwidth * np.sqrt(2.0 * np.pi))
    density = kernel.mean(axis = 0)

    kde_axis.fill_between(
        x = grid_x,
        y1 = 0.0,
        y2 = np.where(grid_x <= 0.0, density, 0.0),
        color = topology_color,
        alpha = 0.30,
        linewidth = 0,
    )
    kde_axis.fill_between(
        x = grid_x,
        y1 = 0.0,
        y2 = np.where(grid_x >= 0.0, density, 0.0),
        color = dynamics_color,
        alpha = 0.32,
        linewidth = 0,
    )
    kde_axis.plot(grid_x, density, color = "#2c2c2c", linewidth = 1.2)
    kde_axis.axvline(x = 0.0, color = neutral_color, linewidth = 1.0)
    kde_axis.axvline(
        x = overall_median,
        color = headline_color,
        linewidth = 1.6,
        linestyle = "--",
    )
    ymax = float(density.max()) * 1.18
    kde_axis.set_ylim(0.0, ymax)
    kde_axis.set_xlim(kde_x_min, kde_x_max)
    kde_axis.set_yticks([])
    kde_axis.set_xticks([])
    kde_axis.set_xlabel("")
    for spine in ["top", "right", "left"]:
        kde_axis.spines[spine].set_visible(False)
    kde_axis.spines["bottom"].set_color(neutral_color)
    kde_axis.text(
        x = kde_x_min + 0.04 * (kde_x_max - kde_x_min),
        y = ymax * 0.92,
        s = "structure",
        ha = "left",
        va = "top",
        fontsize = 9.5,
        color = topology_color,
    )
    kde_axis.text(
        x = kde_x_max - 0.04 * (kde_x_max - kde_x_min),
        y = ymax * 0.92,
        s = "process",
        ha = "right",
        va = "top",
        fontsize = 9.5,
        color = dynamics_color,
    )
    structure_share = 1.0 - share_dynamics
    kde_axis.text(
        x = (kde_x_min + 0.0) / 2.0,
        y = ymax * 0.50,
        s = f"{structure_share:.0%}\nof units",
        ha = "center",
        va = "center",
        fontsize = 11,
        fontweight = "bold",
        color = topology_color,
    )
    kde_axis.text(
        x = (kde_x_max + 0.0) / 2.0,
        y = ymax * 0.50,
        s = f"{share_dynamics:.0%}\nof units",
        ha = "center",
        va = "center",
        fontsize = 11,
        fontweight = "bold",
        color = dynamics_color,
    )
    kde_axis.text(
        x = overall_median,
        y = ymax * 1.12,
        s = f"median = {overall_median:+.2f}",
        ha = "center",
        va = "bottom",
        fontsize = 9,
        color = headline_color,
    )
    kde_axis.set_title(
        label = "Where each (model × dataset) unit lands on the residual-advantage axis  $\\log_{10}(\\mathrm{MAE}_X / \\mathrm{MAE}_Z)$",
        fontsize = 10.5,
        loc = "left",
        pad = 18,
    )

    ## per-domain forest ------------------------------------------------------
    n_domains = len(domain_summary)
    y_positions = np.arange(n_domains)[::-1]
    forest_extent = max(
        float(np.abs(domain_summary[["lower", "upper", "median"]].to_numpy()).max()),
        float(np.abs(model_summary[["median"]].to_numpy()).max()) + 0.15,
    )
    forest_x_max = min(forest_extent, 0.6) * 1.15 + 0.05
    forest_x_min = -forest_x_max

    domain_axis.axvspan(
        xmin = forest_x_min,
        xmax = 0.0,
        color = topology_color,
        alpha = 0.06,
        zorder = 0,
    )
    domain_axis.axvspan(
        xmin = 0.0,
        xmax = forest_x_max,
        color = dynamics_color,
        alpha = 0.07,
        zorder = 0,
    )
    domain_axis.axvline(x = 0.0, color = neutral_color, linewidth = 1.0, zorder = 1)

    for y_pos, (_, row) in zip(y_positions, domain_summary.iterrows()):
        color = domain_palette.get(str(row["domain"]), "#555555")
        clipped_lower = max(float(row["lower"]), forest_x_min)
        clipped_upper = min(float(row["upper"]), forest_x_max)
        domain_axis.hlines(
            y = y_pos,
            xmin = clipped_lower,
            xmax = clipped_upper,
            color = color,
            linewidth = 3.2,
            alpha = 0.9,
            zorder = 2,
        )
        if float(row["lower"]) < forest_x_min:
            domain_axis.scatter(forest_x_min + 0.015, y_pos, marker = "<", s = 28, color = color, zorder = 3)
        if float(row["upper"]) > forest_x_max:
            domain_axis.scatter(forest_x_max - 0.015, y_pos, marker = ">", s = 28, color = color, zorder = 3)
        marker_color = dynamics_color if row["median"] > 0.0 else topology_color
        domain_axis.scatter(
            row["median"],
            y_pos,
            s = 110,
            marker = "o",
            color = color,
            edgecolor = marker_color,
            linewidth = 1.6,
            zorder = 4,
        )
        domain_axis.text(
            x = forest_x_max * 0.97,
            y = y_pos,
            s = f"{row['share_dynamics']:.0%}",
            ha = "right",
            va = "center",
            fontsize = 8.5,
            color = "#3a3a3a",
        )

    domain_axis.set_yticks(y_positions)
    domain_axis.set_yticklabels(
        labels = [domain_labels.get(str(domain), str(domain)) for domain in domain_summary["domain"]],
        fontsize = 9,
    )
    domain_axis.set_xlim(forest_x_min, forest_x_max)
    domain_axis.set_ylim(-0.7, n_domains - 0.3)
    domain_axis.set_xlabel("median advantage (95% CI)", fontsize = 9.5)
    domain_axis.set_title("Every domain", fontsize = 11, loc = "left", pad = 6)
    domain_axis.tick_params(axis = "x", labelsize = 9)
    domain_axis.tick_params(axis = "y", length = 0)
    domain_axis.grid(axis = "x", color = grid_color, alpha = 0.55, linewidth = 0.7)
    for spine in ["top", "right", "left"]:
        domain_axis.spines[spine].set_visible(False)

    ## per-model forest -------------------------------------------------------
    n_models = len(model_summary)
    x_positions = np.arange(n_models)
    model_axis.axhspan(ymin = forest_x_min, ymax = 0.0, color = topology_color, alpha = 0.06, zorder = 0)
    model_axis.axhspan(ymin = 0.0, ymax = forest_x_max, color = dynamics_color, alpha = 0.07, zorder = 0)
    model_axis.axhline(y = 0.0, color = neutral_color, linewidth = 1.0, zorder = 1)

    paradigm_color = {
        "linear_quantile":  "#3b6e8f",
        "linear_convex":    "#3b6e8f",
        "linear_laws":      "#3b6e8f",
        "forest_quantile":  "#7a5a9a",
        "boosted_quantile": "#7a5a9a",
        "xgboost_quantile": "#7a5a9a",
        "neural_quantile":  "#a3623a",
        "neural_expectile": "#a3623a",
        "neural_convex":    "#a3623a",
    }

    for x_pos, (_, row) in zip(x_positions, model_summary.iterrows()):
        color = paradigm_color.get(str(row["model"]), "#555555")
        clipped_lower = max(float(row["lower"]), forest_x_min)
        clipped_upper = min(float(row["upper"]), forest_x_max)
        model_axis.vlines(
            x = x_pos,
            ymin = clipped_lower,
            ymax = clipped_upper,
            color = color,
            linewidth = 3.0,
            alpha = 0.9,
            zorder = 2,
        )
        if float(row["lower"]) < forest_x_min:
            model_axis.scatter(x_pos, forest_x_min + 0.02, marker = "v", s = 28, color = color, zorder = 3)
        if float(row["upper"]) > forest_x_max:
            model_axis.scatter(x_pos, forest_x_max - 0.02, marker = "^", s = 28, color = color, zorder = 3)
        marker_edge = dynamics_color if row["median"] > 0.0 else topology_color
        model_axis.scatter(
            x_pos,
            row["median"],
            s = 110,
            marker = "o",
            color = color,
            edgecolor = marker_edge,
            linewidth = 1.6,
            zorder = 4,
        )
        model_axis.text(
            x = x_pos,
            y = forest_x_max * 0.93,
            s = f"{row['share_dynamics']:.0%}",
            ha = "center",
            va = "top",
            fontsize = 8.5,
            color = "#3a3a3a",
        )

    model_axis.set_xticks(x_positions)
    model_axis.set_xticklabels(
        labels = [model_labels.get(str(name), str(name)) for name in model_summary["model"]],
        fontsize = 9,
        rotation = 22,
        ha = "right",
    )
    model_axis.set_ylim(forest_x_min, forest_x_max)
    model_axis.set_xlim(-0.6, n_models - 0.4)
    model_axis.set_ylabel("median advantage\n(95% CI)", fontsize = 9.5)
    model_axis.set_title("Every learning paradigm", fontsize = 11, loc = "left", pad = 6)
    model_axis.tick_params(axis = "y", labelsize = 9)
    model_axis.tick_params(axis = "x", length = 0)
    model_axis.grid(axis = "y", color = grid_color, alpha = 0.55, linewidth = 0.7)
    for spine in ["top", "right"]:
        model_axis.spines[spine].set_visible(False)

    ## title ------------------------------------------------------------------
    fig.text(
        x = 0.07,
        y = 0.965,
        s = title,
        ha = "left",
        va = "top",
        fontsize = 18,
        fontweight = "bold",
        color = "#1a1a1a",
    )
    fig.text(
        x = 0.07,
        y = 0.918,
        s = subtitle,
        ha = "left",
        va = "top",
        fontsize = 10.5,
        color = "#3a3a3a",
    )

    if show:
        plt.show()
    return fig, axes


## ----------------------------------------------------------------------------
## capacity law compass (cover-iconic radial figure)
## ----------------------------------------------------------------------------
def plot_capacity_law_compass(
    predictions: pd.DataFrame,
    domain_palette: Mapping[str, str] | None = None,
    domain_labels: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    domain_order: Sequence[str] | None = None,
    model_order: Sequence[str] | None = None,
    figsize: tuple[float, float] = (13.5, 13.5),
    title: str = "The capacity law",
    subtitle: str = (
        "Across every system, every paradigm, every domain — once topology fixes the\n"
        "capacity ceiling $C(X)$, the residual $s = y^* - \\hat{C}(X)$ is recovered by process, not by more topology."
    ),
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Render a radial compass of the residual-attribution law: each
        dataset is a wedge, each learning paradigm a bar within the
        wedge, bar length is the percent reduction in residual MAE
        achieved by process features Z over topology features X.
        Outward green bars = process wins; inward orange bars =
        topology wins.

    Args:
        predictions: Residual attribution prediction frame with
            columns model, residual_features, dataset, group, abs_error.
        domain_palette: Mapping from domain to color.
        domain_labels: Mapping from domain to short display label.
        model_labels: Mapping from model identifier to display label.
        domain_order: Ordering of domains around the compass.
        model_order: Ordering of models within each wedge.
        figsize: Figure size in inches.
        title: Figure title.
        subtitle: Figure subtitle.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple of (fig, axes) with axes a 1-element ndarray containing
        the polar axis.

    Raises:
        ValueError: If required columns are missing or only one
            residual feature source is present.
    """

    required = {"model", "residual_features", "dataset", "group", "abs_error"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sources = set(predictions["residual_features"].astype(str).unique())
    if not {"X_to_slack", "Z_to_slack"}.issubset(sources):
        raise ValueError(
            "predictions must include both X_to_slack and Z_to_slack residual_features"
        )

    domain_palette = dict(domain_palette or DEFAULT_DOMAIN_PALETTE)
    domain_labels = dict(domain_labels or DEFAULT_DOMAIN_LABELS)
    model_labels = dict(model_labels or DEFAULT_MODEL_LABELS)

    aggregated = (
        predictions
        .groupby(by = ["model", "dataset", "group", "residual_features"], observed = True)["abs_error"]
        .mean()
        .reset_index()
        .pivot_table(
            index = ["model", "dataset", "group"],
            columns = "residual_features",
            values = "abs_error",
            observed = True,
        )
        .rename_axis(columns = None)
        .reset_index()
        .dropna(subset = ["X_to_slack", "Z_to_slack"])
    )
    aggregated = aggregated.loc[
        (aggregated["X_to_slack"] > 0.0) & (aggregated["Z_to_slack"] > 0.0)
    ].copy()
    aggregated["pct_gain"] = 100.0 * (
        1.0 - aggregated["Z_to_slack"] / aggregated["X_to_slack"]
    )
    aggregated["log_ratio"] = np.log10(aggregated["X_to_slack"] / aggregated["Z_to_slack"])

    if domain_order is None:
        domain_order = [domain for domain in domain_palette.keys() if domain in set(aggregated["group"])]
        for domain in aggregated["group"].astype(str).unique():
            if domain not in domain_order:
                domain_order.append(domain)
    if model_order is None:
        model_order = [model for model in model_labels.keys() if model in set(aggregated["model"])]
        for model in aggregated["model"].astype(str).unique():
            if model not in model_order:
                model_order.append(model)

    paradigm_palette = {
        "linear_quantile":  "#3b6e8f",
        "linear_convex":    "#3b6e8f",
        "linear_laws":      "#3b6e8f",
        "forest_quantile":  "#7a5a9a",
        "boosted_quantile": "#7a5a9a",
        "xgboost_quantile": "#7a5a9a",
        "neural_quantile":  "#a3623a",
        "neural_expectile": "#a3623a",
        "neural_convex":    "#a3623a",
    }

    dataset_records: list[tuple[str, str]] = []
    for domain in domain_order:
        domain_subset = (
            aggregated.loc[aggregated["group"] == domain, ["dataset"]]
            .drop_duplicates()
            .sort_values("dataset")
        )
        for dataset in domain_subset["dataset"].tolist():
            dataset_records.append((str(dataset), str(domain)))
    n_datasets = len(dataset_records)
    n_models = len(model_order)

    if n_datasets == 0 or n_models == 0:
        raise ValueError("No (model, dataset) units available for plotting")

    angles_per_dataset = 2.0 * np.pi / n_datasets
    inter_wedge_pad = angles_per_dataset * 0.18
    bar_span = angles_per_dataset - inter_wedge_pad
    bar_width = bar_span / n_models

    background_color = "#fbfaf6"
    dynamics_color = "#1f7547"
    topology_color = "#b35a36"
    baseline_radius = 1.0
    radial_scale = 0.55
    max_clip = 60.0
    domain_band_inner = baseline_radius + radial_scale + 0.05
    domain_band_outer = domain_band_inner + 0.07
    dataset_label_radius = domain_band_outer + 0.10
    domain_label_radius = domain_band_outer + 0.30
    inner_disc_radius = baseline_radius - radial_scale - 0.02

    overall_pct = float(aggregated["pct_gain"].median())
    overall_log = float(aggregated["log_ratio"].median())
    n_total = int(len(aggregated))
    n_dynamics = int((aggregated["pct_gain"] > 0.0).sum())
    share_dynamics = n_dynamics / n_total if n_total else float("nan")

    fig = plt.figure(figsize = figsize, constrained_layout = False)
    fig.patch.set_facecolor(background_color)
    main_axis = fig.add_subplot(projection = "polar")
    main_axis.set_facecolor(background_color)
    main_axis.set_theta_zero_location(loc = "N")
    main_axis.set_theta_direction(direction = -1)

    ## reference rings
    theta_full = np.linspace(start = 0.0, stop = 2.0 * np.pi, num = 720)
    label_angle = 0.0
    for ref_pct in (15.0, 30.0, 45.0):
        ring_radius = baseline_radius + (ref_pct / max_clip) * radial_scale
        main_axis.plot(
            theta_full,
            np.full_like(theta_full, ring_radius),
            color = "#cdcabb",
            linewidth = 0.6,
            linestyle = (0, (2, 3)),
            zorder = 1,
        )
        main_axis.text(
            x = label_angle,
            y = ring_radius,
            s = f"+{int(ref_pct)}%",
            ha = "center",
            va = "center",
            fontsize = 7.5,
            color = "#9b9580",
            zorder = 1.5,
            bbox = {"boxstyle": "round,pad=0.10", "facecolor": background_color, "edgecolor": "none"},
        )
    for ref_pct in (-15.0, -30.0):
        ring_radius = baseline_radius + (ref_pct / max_clip) * radial_scale
        main_axis.plot(
            theta_full,
            np.full_like(theta_full, ring_radius),
            color = "#e3d8c8",
            linewidth = 0.5,
            linestyle = (0, (2, 4)),
            zorder = 1,
        )
    main_axis.plot(
        theta_full,
        np.full_like(theta_full, baseline_radius),
        color = "#3a3a3a",
        linewidth = 1.0,
        zorder = 2,
    )

    ## bars
    for dataset_index, (dataset, domain) in enumerate(dataset_records):
        wedge_start = dataset_index * angles_per_dataset + inter_wedge_pad / 2.0
        for model_index, model in enumerate(model_order):
            row = aggregated.loc[
                (aggregated["dataset"] == dataset) & (aggregated["model"] == model)
            ]
            if row.empty:
                continue
            gain = float(row["pct_gain"].iloc[0])
            gain_clipped = float(np.clip(gain, -max_clip, max_clip))
            radial_height = (gain_clipped / max_clip) * radial_scale
            bar_center = wedge_start + (model_index + 0.5) * bar_width
            face_color = dynamics_color if gain >= 0.0 else topology_color
            paradigm_edge = paradigm_palette.get(model, face_color)
            if radial_height >= 0.0:
                bottom = baseline_radius
                height = radial_height
            else:
                bottom = baseline_radius + radial_height
                height = -radial_height
            main_axis.bar(
                x = bar_center,
                height = height,
                width = bar_width * 0.86,
                bottom = bottom,
                color = face_color,
                edgecolor = paradigm_edge,
                linewidth = 0.7,
                alpha = 0.9,
                zorder = 3,
            )

    ## domain bands and labels
    domain_to_indices: dict[str, list[int]] = {}
    for index, (_, domain) in enumerate(dataset_records):
        domain_to_indices.setdefault(domain, []).append(index)

    for domain, indices in domain_to_indices.items():
        first = min(indices) * angles_per_dataset
        last = (max(indices) + 1) * angles_per_dataset
        theta = np.linspace(start = first + inter_wedge_pad / 4.0, stop = last - inter_wedge_pad / 4.0, num = 96)
        main_axis.fill_between(
            x = theta,
            y1 = domain_band_inner,
            y2 = domain_band_outer,
            color = domain_palette.get(domain, "#888888"),
            alpha = 0.92,
            zorder = 4,
        )
        midpoint = (first + last) / 2.0
        rotation_deg = np.degrees(midpoint) * -1.0
        if 90.0 < (np.degrees(midpoint) % 360.0) < 270.0:
            rotation_deg += 180.0
        main_axis.text(
            x = midpoint,
            y = domain_label_radius,
            s = domain_labels.get(domain, domain).replace("\n", " "),
            ha = "center",
            va = "center",
            fontsize = 11,
            fontweight = "bold",
            color = domain_palette.get(domain, "#444444"),
            rotation = rotation_deg,
            rotation_mode = "anchor",
            zorder = 5,
        )

    ## dataset labels
    for dataset_index, (dataset, _) in enumerate(dataset_records):
        midpoint = (dataset_index + 0.5) * angles_per_dataset
        deg = np.degrees(midpoint) % 360.0
        if 90.0 < deg < 270.0:
            rotation_deg = -deg + 180.0
            ha = "right"
        else:
            rotation_deg = -deg
            ha = "left"
        main_axis.text(
            x = midpoint,
            y = dataset_label_radius,
            s = dataset,
            ha = ha,
            va = "center",
            fontsize = 8.5,
            color = "#2a2a2a",
            rotation = rotation_deg,
            rotation_mode = "anchor",
            zorder = 5,
        )

    ## inner disc with headline
    inner_theta = np.linspace(start = 0.0, stop = 2.0 * np.pi, num = 360)
    main_axis.fill_between(
        x = inner_theta,
        y1 = 0.0,
        y2 = inner_disc_radius,
        color = "white",
        alpha = 0.92,
        zorder = 6,
    )
    main_axis.plot(
        inner_theta,
        np.full_like(inner_theta, inner_disc_radius),
        color = "#3a3a3a",
        linewidth = 0.8,
        zorder = 6.5,
    )
    headline_lines = [
        f"$\\bf{{{n_dynamics}/{n_total}}}$",
        "tests favor process",
        "",
        f"({share_dynamics:.0%} of model × system pairs)",
        "",
        "median residual",
        "error reduction",
        f"$\\bf{{{overall_pct:+.1f}\\%}}$",
    ]
    main_axis.text(
        x = 0.0,
        y = 0.0,
        s = "\n".join(headline_lines),
        ha = "center",
        va = "center",
        fontsize = 11.5,
        color = "#1a1a1a",
        zorder = 7,
    )

    ## axis cosmetics
    main_axis.set_ylim(bottom = 0.0, top = domain_label_radius + 0.18)
    main_axis.set_yticks([])
    main_axis.set_xticks([])
    main_axis.spines["polar"].set_visible(False)
    main_axis.grid(visible = False)

    ## legend strip (bars: green out / orange in)
    legend_axis = fig.add_axes(rect = (0.06, 0.045, 0.36, 0.04))
    legend_axis.set_facecolor(background_color)
    legend_axis.set_xlim(-1.0, 1.0)
    legend_axis.set_ylim(0.0, 1.0)
    legend_axis.axis("off")
    legend_axis.add_patch(Rectangle(xy = (-0.95, 0.30), width = 0.85, height = 0.40, facecolor = topology_color, edgecolor = "none"))
    legend_axis.add_patch(Rectangle(xy = (0.10, 0.30), width = 0.85, height = 0.40, facecolor = dynamics_color, edgecolor = "none"))
    legend_axis.text(x = -0.525, y = 0.92, s = "topology wins", ha = "center", va = "bottom", fontsize = 9, color = topology_color, fontweight = "bold")
    legend_axis.text(x = 0.525, y = 0.92, s = "process wins", ha = "center", va = "bottom", fontsize = 9, color = dynamics_color, fontweight = "bold")
    legend_axis.text(x = -0.525, y = 0.10, s = "bar points inward", ha = "center", va = "top", fontsize = 8, color = "#5a5a5a", style = "italic")
    legend_axis.text(x = 0.525, y = 0.10, s = "bar points outward", ha = "center", va = "top", fontsize = 8, color = "#5a5a5a", style = "italic")

    ## paradigm legend (right side)
    paradigm_legend_axis = fig.add_axes(rect = (0.58, 0.045, 0.36, 0.04))
    paradigm_legend_axis.set_facecolor(background_color)
    paradigm_legend_axis.set_xlim(0.0, 1.0)
    paradigm_legend_axis.set_ylim(0.0, 1.0)
    paradigm_legend_axis.axis("off")
    paradigm_groups = [
        ("Linear", "#3b6e8f", "Linear Quantile · Linear Convex · Linear Laws"),
        ("Tree-based", "#7a5a9a", "Forest · Boosted · XGBoost"),
        ("Neural", "#a3623a", "Neural Quantile · Expectile · Convex"),
    ]
    paradigm_legend_axis.text(x = 0.0, y = 0.92, s = "9 learning paradigms per wedge:", ha = "left", va = "bottom", fontsize = 9, color = "#3a3a3a", fontweight = "bold")
    for index, (name, color, members) in enumerate(paradigm_groups):
        y_pos = 0.55 - index * 0.30
        paradigm_legend_axis.add_patch(Rectangle(xy = (0.0, y_pos), width = 0.04, height = 0.18, facecolor = color, edgecolor = "none"))
        paradigm_legend_axis.text(x = 0.06, y = y_pos + 0.09, s = f"{name}: {members}", ha = "left", va = "center", fontsize = 7.8, color = "#3a3a3a")

    ## title
    fig.text(
        x = 0.50,
        y = 0.975,
        s = title,
        ha = "center",
        va = "top",
        fontsize = 22,
        fontweight = "bold",
        color = "#1a1a1a",
    )
    fig.text(
        x = 0.50,
        y = 0.940,
        s = subtitle,
        ha = "center",
        va = "top",
        fontsize = 11,
        color = "#3a3a3a",
    )

    fig.subplots_adjust(left = 0.04, right = 0.96, top = 0.92, bottom = 0.10)

    if show:
        plt.show()
    return fig, np.array([main_axis], dtype = object)


def _plot_transfer_invariance_legacy(
    results_data_domain: pd.DataFrame,
    results_data_disc: pd.DataFrame,
    results_data_5fold: pd.DataFrame,
    results_data_10fold: pd.DataFrame,
    disc_domain_map: Mapping[str, str],
    *,
    domain_palette: Mapping[str, str] | None = None,
    domain_labels: Mapping[str, str] | None = None,
    equivalence_margin: float | None = None,
    feasibility_threshold: float = 0.05,
    figsize: tuple[float, float] = (9.6, 13.2),
    title: str = "The capacity law transfers across every scientific domain",
    subtitle: str = (
        "After exiling entire domains and disciplines from training, predictive efficiency "
        "(EI) is statistically indistinguishable from random-split baselines — "
        "held-out generalization is a no-op."
    ),
    background_color: str = "#fbfaf6",
    random_state: int = 42,
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Money-shot visualization of held-out-group transfer invariance. Shows
        that EI under leave-one-domain-out and leave-one-discipline-out
        cross-validation is statistically indistinguishable from random k-fold
        baselines, both distributionally (top strip) and per held-out group
        (bottom forest of paired Delta EI from pooled random baseline).

    Args:
        results_data_domain: DataFrame from logo_cross_valid on 'domain', with
            columns including 'model', 'group', 'ei'.
        results_data_disc: DataFrame from logo_cross_valid on 'discipline'.
        results_data_5fold: DataFrame from kfold_cross_valid (5-fold), with
            one row per model and a 'model' + 'ei' column.
        results_data_10fold: DataFrame from kfold_cross_valid (10-fold).
        disc_domain_map: Mapping from discipline -> domain.
        domain_palette: Optional override for domain -> color.
        domain_labels: Optional override for domain -> short label.
        equivalence_margin: Half-width of the equivalence band on Delta EI. If
            None, derives the margin from the IQR of pooled random-split
            baseline EI values using spec_marginal_delta.
        feasibility_threshold: Estimators with transfer EI at or below this
            value are treated as feasibility failures (predictions violating
            the capacity bound) and excluded from per-row Delta EI summaries.
            Per-row counts of feasible / total estimators are annotated when
            any failure is present.
        figsize: Figure size in inches.
        title: Headline title.
        subtitle: Subtitle below the title.
        background_color: Figure background color.
        random_state: RNG seed for reproducibility.
        show: If True, calls plt.show().

    Returns:
        Tuple of (figure, axes_array) where axes_array is [strip_axis,
        forest_axis].

    Raises:
        ValueError: If required columns are missing or no valid rows remain.
    """

    palette = dict(domain_palette) if domain_palette is not None else dict(DEFAULT_DOMAIN_PALETTE)
    labels = dict(domain_labels) if domain_labels is not None else dict(DEFAULT_DOMAIN_LABELS)

    ## validate required columns
    for name, frame, cols in (
        ("results_data_domain", results_data_domain, ("model", "group", "ei")),
        ("results_data_disc",   results_data_disc,   ("model", "group", "ei")),
        ("results_data_5fold",  results_data_5fold,  ("model", "ei")),
        ("results_data_10fold", results_data_10fold, ("model", "ei")),
    ):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    ## pooled random baseline per model (mean of 5-fold and 10-fold)
    base = pd.concat(
        [results_data_5fold[["model", "ei"]], results_data_10fold[["model", "ei"]]],
        ignore_index = True,
    )
    base = base.dropna(subset = ["ei"])
    baseline_per_model = base.groupby("model")["ei"].mean()
    if baseline_per_model.empty:
        raise ValueError("no valid baseline EI rows in 5-fold/10-fold results")

    ## paired Delta EI = transfer EI - pooled random baseline EI for that model
    def _attach_delta(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.dropna(subset = ["ei"]).copy()
        out["baseline_ei"] = out["model"].map(baseline_per_model)
        out = out.dropna(subset = ["baseline_ei"])
        out["delta_ei"] = out["ei"].astype(float) - out["baseline_ei"].astype(float)
        out["feasible"] = out["ei"].astype(float) > float(feasibility_threshold)
        return out

    domain_delta = _attach_delta(results_data_domain)
    disc_delta = _attach_delta(results_data_disc)
    domain_delta["domain"] = domain_delta["group"]
    disc_delta["domain"] = disc_delta["group"].map(disc_domain_map).fillna("Unknown")

    ## ordering: domain order from palette
    domain_order = [d for d in palette if d in domain_delta["domain"].unique()]
    extras = [d for d in domain_delta["domain"].unique() if d not in domain_order]
    domain_order = domain_order + sorted(extras)

    ## domain rows: ordered by palette
    domain_rows = list()
    for dom in domain_order:
        sub = domain_delta[domain_delta["domain"] == dom]
        if len(sub) == 0:
            continue
        feasible_mask = sub["feasible"].to_numpy()
        domain_rows.append({
            "label":       dom,
            "kind":        "domain",
            "domain":      dom,
            "values":      sub.loc[feasible_mask, "delta_ei"].astype(float).to_numpy(),
            "n_total":     int(len(sub)),
            "n_feasible": int(feasible_mask.sum()),
        })

    ## discipline rows: ordered by domain then median Delta EI (feasible only) ascending
    disc_rows: list[dict] = list()
    disc_per_group = disc_delta.groupby("group")
    feasible_disc = disc_delta[disc_delta["feasible"]]
    disc_summary = (
        feasible_disc.groupby("group")["delta_ei"]
        .median()
        .rename("median_delta")
        .reset_index()
    )
    ## include groups with zero feasible models so they still render
    all_groups = pd.DataFrame({"group": sorted(disc_delta["group"].unique())})
    disc_summary = all_groups.merge(disc_summary, on = "group", how = "left")
    disc_summary["median_delta"] = disc_summary["median_delta"].fillna(0.0)
    disc_summary["domain"] = disc_summary["group"].map(disc_domain_map).fillna("Unknown")
    disc_summary["_drank"] = disc_summary["domain"].map(
        {d: i for i, d in enumerate(domain_order)}
    ).fillna(len(domain_order))
    disc_summary = disc_summary.sort_values(
        by = ["_drank", "median_delta"], ascending = [True, False],
    ).reset_index(drop = True)
    for _, row in disc_summary.iterrows():
        sub = disc_per_group.get_group(row["group"])
        feasible_mask = sub["feasible"].to_numpy()
        disc_rows.append({
            "label":       row["group"],
            "kind":        "discipline",
            "domain":      row["domain"],
            "values":      sub.loc[feasible_mask, "delta_ei"].astype(float).to_numpy(),
            "n_total":     int(len(sub)),
            "n_feasible": int(feasible_mask.sum()),
        })

    ## headline statistics (over feasible estimators only)
    rng = np.random.default_rng(seed = random_state)
    all_rows = domain_rows + disc_rows
    nonempty_rows = [r for r in all_rows if r["values"].size > 0]
    all_delta = np.concatenate(
        [r["values"] for r in nonempty_rows]
    ).astype(float) if nonempty_rows else np.array([], dtype = float)
    overall_med = float(np.median(all_delta)) if all_delta.size else float("nan")
    n_groups = len(all_rows)
    n_within = sum(
        1 for r in nonempty_rows
        if abs(float(np.median(r["values"]))) <= equivalence_margin
    )
    ## per-row feasibility totals for the headline
    n_total_estimators = sum(r["n_total"] for r in all_rows)
    n_feasible_estimators = sum(r["n_feasible"] for r in all_rows)

    ## figure scaffolding
    fig = plt.figure(figsize = figsize, facecolor = background_color)
    gs = fig.add_gridspec(
        nrows = 2, ncols = 1,
        height_ratios = [0.22, 0.78],
        hspace = 0.20,
        left = 0.21, right = 0.905,
        top = 0.805, bottom = 0.060,
    )
    strip_axis = fig.add_subplot(gs[0, 0])
    forest_axis = fig.add_subplot(gs[1, 0])
    for ax in (strip_axis, forest_axis):
        ax.set_facecolor(background_color)

    ## ---- top panel: regime distribution overlay ----
    ## LOGO regimes are filtered to feasible estimators only (consistent with
    ## the per-group panel below); random-split regimes show all values.
    def _feasible_ei(frame: pd.DataFrame) -> np.ndarray:
        ei = frame["ei"].dropna().to_numpy()
        return ei[ei > float(feasibility_threshold)]

    regimes = [
        ("Random (10-Fold)",   results_data_10fold["ei"].dropna().to_numpy(), "#9aa0a6", "o"),
        ("Random (5-Fold)",    results_data_5fold["ei"].dropna().to_numpy(),  "#6b7075", "o"),
        ("Domain (LOGO)",      _feasible_ei(results_data_domain),             "#2C6E91", "D"),
        ("Discipline (LOGO)",  _feasible_ei(results_data_disc),               "#3A7D55", "D"),
    ]
    strip_jitter = 0.18
    for i, (name, vals, color, marker) in enumerate(regimes):
        if vals.size == 0:
            continue
        y_center = len(regimes) - 1 - i
        ## individual points
        jitter = rng.uniform(low = -strip_jitter, high = strip_jitter, size = vals.size)
        strip_axis.scatter(
            vals, np.full_like(vals, y_center, dtype = float) + jitter,
            s = 12, color = color, alpha = 0.32, edgecolors = "none", zorder = 2,
        )
        ## IQR bar + median tick
        q1, med, q3 = np.quantile(vals, [0.25, 0.50, 0.75])
        strip_axis.plot(
            [q1, q3], [y_center, y_center],
            color = color, lw = 5.5, alpha = 0.45, solid_capstyle = "round", zorder = 3,
        )
        strip_axis.scatter(
            [med], [y_center], s = 75, color = color, marker = marker,
            edgecolors = "white", linewidths = 0.9, zorder = 5,
        )

    ## strip axis cosmetics
    all_vals = np.concatenate([v for _, v, _, _ in regimes if v.size > 0])
    pad = 0.04
    x_lo_strip = float(np.min(all_vals)) - pad
    x_hi_strip = float(np.max(all_vals)) + pad
    strip_axis.set_xlim(x_lo_strip, x_hi_strip)
    strip_axis.set_ylim(-0.7, len(regimes) - 0.3)
    strip_axis.set_yticks(range(len(regimes)))
    strip_axis.set_yticklabels(
        [name for name, _, _, _ in regimes][::-1], fontsize = 9,
    )
    for ytick, (_, _, color, _) in zip(strip_axis.get_yticklabels(), regimes):
        ytick.set_color(color)
    strip_axis.set_xlabel("Efficiency Index (EI)", fontsize = 9.5, labelpad = 4)
    strip_axis.tick_params(axis = "x", labelsize = 8.5)
    strip_axis.tick_params(axis = "y", left = False, pad = 4)
    strip_axis.spines[["top", "right", "left"]].set_visible(False)
    strip_axis.spines["bottom"].set_linewidth(0.7)
    strip_axis.xaxis.grid(True, lw = 0.4, color = "#EBEBEB", zorder = 0)
    strip_axis.set_axisbelow(True)
    strip_axis.text(
        x = 0.0, y = 1.04,
        s = "A   EI distribution under transfer vs. random-split baselines",
        transform = strip_axis.transAxes,
        ha = "left", va = "bottom",
        fontsize = 10.5, fontweight = "semibold", color = "#1a1a1a",
    )

    ## ---- bottom panel: held-out group forest of Delta EI ----
    n_domain = len(domain_rows)
    n_disc = len(disc_rows)
    spacer = 1.2
    ## y positions: domains at top, then spacer, then disciplines below
    y_positions: list[float] = list()
    for i in range(n_domain):
        y_positions.append((n_disc + spacer) + (n_domain - 1 - i))
    domain_y = list(y_positions)
    for i in range(n_disc):
        y_positions.append(n_disc - 1 - i)
    disc_y = y_positions[n_domain:]

    rows = domain_rows + disc_rows

    ## equivalence band
    forest_axis.axvspan(
        -equivalence_margin, equivalence_margin,
        color = "#e8efe9", alpha = 0.7, zorder = 0,
    )
    forest_axis.axvline(0.0, color = "#7a7a7a", lw = 1.0, zorder = 1)
    forest_axis.axvline(
        -equivalence_margin, color = "#a8b3a9", lw = 0.7, ls = (0, (3, 3)), zorder = 1,
    )
    forest_axis.axvline(
        +equivalence_margin, color = "#a8b3a9", lw = 0.7, ls = (0, (3, 3)), zorder = 1,
    )

    ## domain / discipline section divider
    divider_y = n_disc + spacer / 2.0 - 0.5
    forest_axis.axhline(divider_y, color = "#d0d0d0", lw = 0.6, zorder = 1)

    ## render rows
    for row, y in zip(rows, y_positions):
        color = palette.get(row["domain"], "#555555")
        vals = row["values"]
        ## feasibility annotation: only show when at least one estimator failed
        if row["n_feasible"] < row["n_total"]:
            forest_axis.text(
                x = 1.005,
                y = y,
                s = f"{row['n_feasible']}/{row['n_total']} feasible",
                transform = forest_axis.get_yaxis_transform(),
                ha = "left", va = "center",
                fontsize = 6.8, color = "#9a8a6a", fontstyle = "italic",
                clip_on = False,
            )
        if vals.size == 0:
            continue
        q1, med, q3 = np.quantile(vals, [0.25, 0.50, 0.75])
        ## faint individual points
        jitter = rng.uniform(low = -0.18, high = 0.18, size = vals.size)
        forest_axis.scatter(
            vals, np.full_like(vals, y, dtype = float) + jitter,
            s = 10, color = color, alpha = 0.25, edgecolors = "none", zorder = 2,
        )
        ## IQR line
        forest_axis.plot(
            [q1, q3], [y, y],
            color = color, lw = 2.0, alpha = 0.55, solid_capstyle = "round", zorder = 3,
        )
        ## median marker (size emphasizes domain rows)
        marker_size = 95 if row["kind"] == "domain" else 48
        forest_axis.scatter(
            [med], [y], s = marker_size, color = color,
            edgecolors = "white", linewidths = 0.9, zorder = 5,
        )

    ## y-tick labels
    all_y = list(y_positions)
    all_labels: list[str] = list()
    for r in rows:
        if r["kind"] == "domain":
            all_labels.append(labels.get(r["label"], r["label"]).replace("\n", " "))
        else:
            all_labels.append(r["label"])
    ## sort by ascending y for matplotlib tick semantics (so labels match positions)
    order = np.argsort(all_y)
    forest_axis.set_yticks([all_y[i] for i in order])
    forest_axis.set_yticklabels([all_labels[i] for i in order], fontsize = 7.6)
    for tick_lbl, idx in zip(forest_axis.get_yticklabels(), order):
        r = rows[idx]
        color = palette.get(r["domain"], "#555555")
        tick_lbl.set_color(color)
        if r["kind"] == "domain":
            tick_lbl.set_fontsize(9.0)
            tick_lbl.set_fontweight("bold")

    ## x-axis
    all_delta_finite = all_delta[np.isfinite(all_delta)]
    x_lim = max(
        equivalence_margin * 2.4,
        float(np.quantile(np.abs(all_delta_finite), 0.99)) + 0.02,
    )
    forest_axis.set_xlim(-x_lim, x_lim)
    forest_axis.set_ylim(-0.8, n_disc + spacer + n_domain - 0.2)
    forest_axis.spines[["top", "right", "left"]].set_visible(False)
    forest_axis.spines["bottom"].set_linewidth(0.7)
    forest_axis.xaxis.grid(True, lw = 0.4, color = "#EBEBEB", zorder = 0)
    forest_axis.set_axisbelow(True)
    forest_axis.tick_params(axis = "y", left = False, pad = 4)
    forest_axis.tick_params(axis = "x", labelsize = 8.5)
    forest_axis.set_xlabel(
        "$\\Delta$EI  =  held-out transfer EI  $-$  pooled random-split baseline EI   (paired per model)",
        fontsize = 9.5, labelpad = 6,
    )

    ## panel B label
    forest_axis.text(
        x = 0.0, y = 1.012,
        s = (
            f"B   Per-group transfer penalty $\\Delta$EI "
            f"(shaded $\\pm${equivalence_margin:.2f} equivalence band)"
        ),
        transform = forest_axis.transAxes,
        ha = "left", va = "bottom",
        fontsize = 10.5, fontweight = "semibold", color = "#1a1a1a",
    )

    ## ---- header: title + subtitle + headline statistic plate ----
    fig.text(
        x = 0.50, y = 0.978, s = title,
        ha = "center", va = "top",
        fontsize = 18, fontweight = "bold", color = "#1a1a1a",
    )
    fig.text(
        x = 0.50, y = 0.948, s = subtitle,
        ha = "center", va = "top",
        fontsize = 10, color = "#3a3a3a",
    )

    ## headline statistic plate (centered just below subtitle)
    feasibility_share = (
        n_feasible_estimators / n_total_estimators
        if n_total_estimators > 0 else 0.0
    )
    feasibility_label = f"{feasibility_share:.0%}".replace("%", "\\%")
    stat_text = (
        f"median $\\Delta$EI across feasible transfers: "
        f"$\\bf{{{overall_med:+.3f}}}$   "
        f"(n = {all_delta.size} estimator×group transfers)"
        f"      ·      "
        f"$\\bf{{{n_within}\\,/\\,{n_groups}}}$ held-out-group medians within "
        f"$\\pm${equivalence_margin:.2f} EI of baseline"
        f"\nfeasibility: "
        f"$\\bf{{{n_feasible_estimators}\\,/\\,{n_total_estimators}}}$ "
        f"({feasibility_share:.0%}) estimator×group transfers satisfy the "
        f"capacity bound (EI $>$ {feasibility_threshold:.2f})"
    )
    fig.text(
        x = 0.50, y = 0.895, s = stat_text,
        ha = "center", va = "top",
        fontsize = 9.5, color = "#1a1a1a", linespacing = 1.55,
        bbox = {
            "boxstyle":  "round,pad=0.55",
            "facecolor": "#f1efe6",
            "edgecolor": "#d6d2c3",
            "linewidth": 0.6,
        },
    )

    if show:
        plt.show()
    return fig, np.array([strip_axis, forest_axis], dtype = object)


def plot_transfer_invariance(
    results_data_domain: pd.DataFrame,
    results_data_disc: pd.DataFrame,
    results_data_5fold: pd.DataFrame,
    results_data_10fold: pd.DataFrame,
    disc_domain_map: Mapping[str, str],
    *,
    domain_palette: Mapping[str, str] | None = None,
    domain_labels: Mapping[str, str] | None = None,
    equivalence_margin: float | None = None,
    feasibility_threshold: float = 0.05,
    figsize: tuple[float, float] = (9.6, 10.2),
    title: str = "The capacity law transfers across every scientific domain",
    subtitle: str = (
        "After exiling entire disciplines from training, predictive efficiency "
        "(EI) is statistically indistinguishable from random-split baselines — "
        "held-out generalization is a no-op."
    ),
    background_color: str = "#fbfaf6",
    random_state: int = 42,
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Money-shot discipline transfer visualization. Shows per-discipline
        paired Delta EI against each model's pooled random-split baseline on a
        fixed [-1, 1] scale. Domain aggregate rows and the regime strip panel
        are intentionally omitted so the figure focuses only on held-out
        discipline transfer penalties.

    Args:
        results_data_domain: DataFrame from logo_cross_valid on 'domain'. This
            argument is retained for API compatibility but is not drawn.
        results_data_disc: DataFrame from logo_cross_valid on 'discipline',
            with columns including 'model', 'group', and 'ei'.
        results_data_5fold: DataFrame from kfold_cross_valid (5-fold), with
            one row per model and a 'model' + 'ei' column.
        results_data_10fold: DataFrame from kfold_cross_valid (10-fold).
        disc_domain_map: Mapping from discipline -> domain.
        domain_palette: Optional override for domain -> color.
        domain_labels: Optional override for domain -> short label.
        equivalence_margin: Half-width of the equivalence band on Delta EI. If
            None, derives the margin from the IQR of pooled random-split
            baseline EI values using spec_marginal_delta.
        feasibility_threshold: Estimators with transfer EI at or below this
            value are treated as feasibility failures and excluded from per-row
            Delta EI summaries.
        figsize: Figure size in inches.
        title: Headline title.
        subtitle: Subtitle below the title.
        background_color: Figure background color.
        random_state: RNG seed for reproducibility.
        show: If True, calls plt.show().

    Returns:
        Tuple of (figure, axes_array) where axes_array contains the discipline
        forest axis.

    Raises:
        ValueError: If required columns are missing or no valid rows remain.
    """

    del results_data_domain

    palette = dict(domain_palette) if domain_palette is not None else dict(DEFAULT_DOMAIN_PALETTE)
    labels = dict(domain_labels) if domain_labels is not None else dict(DEFAULT_DOMAIN_LABELS)

    ## validate required columns
    for frame_name, frame, required_columns in (
        ("results_data_disc",   results_data_disc,   ("model", "group", "ei")),
        ("results_data_5fold",  results_data_5fold,  ("model", "ei")),
        ("results_data_10fold", results_data_10fold, ("model", "ei")),
    ):
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"{frame_name} missing columns: {missing_columns}")

    ## pooled random baseline per model
    baseline = pd.concat(
        [results_data_5fold[["model", "ei"]], results_data_10fold[["model", "ei"]]],
        ignore_index = True,
    )
    baseline = baseline.dropna(subset = ["ei"])
    baseline_per_model = baseline.groupby("model")["ei"].mean()
    if baseline_per_model.empty:
        raise ValueError("no valid baseline EI rows in 5-fold/10-fold results")

    empirical_equivalence_margin = equivalence_margin is None
    if equivalence_margin is None:
        from src.evaluators.metrics import spec_marginal_delta

        equivalence_margin = spec_marginal_delta(
            results = baseline.assign(reference = "baseline"),
            feat_value = ["ei"],
            label_ref = "reference",
            value_ref = "baseline",
            method = "iqr",
            scale = 1.0,
            decimals = 2,
        )
    else:
        equivalence_margin = float(equivalence_margin)
    margin_label = "empirical $\\delta$" if empirical_equivalence_margin else "$\\delta$"

    ## within-model random-split residuals on Delta EI scale (centered at 0)
    baseline_residuals = (
        baseline["ei"].astype(float)
        - baseline["model"].map(baseline_per_model).astype(float)
    ).dropna().to_numpy()
    if baseline_residuals.size >= 2:
        baseline_q1, baseline_q3 = np.quantile(baseline_residuals, [0.25, 0.75])
    else:
        baseline_q1, baseline_q3 = 0.0, 0.0

    ## paired Delta EI = discipline LOGO EI - pooled random baseline EI
    disc_delta = results_data_disc.dropna(subset = ["ei"]).copy()
    disc_delta["baseline_ei"] = disc_delta["model"].map(baseline_per_model)
    disc_delta = disc_delta.dropna(subset = ["baseline_ei"])
    disc_delta["delta_ei"] = disc_delta["ei"].astype(float) - disc_delta["baseline_ei"].astype(float)
    disc_delta["feasible"] = disc_delta["ei"].astype(float) > float(feasibility_threshold)
    disc_delta["domain"] = disc_delta["group"].map(disc_domain_map).fillna("Unknown")
    if disc_delta.empty:
        raise ValueError("no valid discipline transfer rows remain after pairing baselines")

    ## order disciplines by domain and feasible median Delta EI
    domain_order = [domain for domain in palette if domain in disc_delta["domain"].unique()]
    extra_domains = [domain for domain in disc_delta["domain"].unique() if domain not in domain_order]
    domain_order = domain_order + sorted(extra_domains)
    domain_rank = {domain: index for index, domain in enumerate(domain_order)}

    feasible_disc = disc_delta[disc_delta["feasible"]]
    disc_summary = (
        feasible_disc.groupby("group")["delta_ei"]
        .median()
        .rename("median_delta")
        .reset_index()
    )
    all_groups = pd.DataFrame({"group": sorted(disc_delta["group"].unique())})
    disc_summary = all_groups.merge(disc_summary, on = "group", how = "left")
    disc_summary["median_delta"] = disc_summary["median_delta"].fillna(0.0)
    disc_summary["domain"] = disc_summary["group"].map(disc_domain_map).fillna("Unknown")
    disc_summary["_domain_rank"] = disc_summary["domain"].map(domain_rank).fillna(len(domain_order))
    disc_summary = disc_summary.sort_values(
        by = ["_domain_rank", "median_delta"],
        ascending = [True, False],
    ).reset_index(drop = True)

    grouped_disc = dict(tuple(disc_delta.groupby("group")))
    disc_rows: list[dict] = list()
    for _, summary_row in disc_summary.iterrows():
        group_name = summary_row["group"]
        group_frame = grouped_disc[group_name]
        feasible_frame = group_frame[group_frame["feasible"]]
        disc_rows.append({
            "label":      group_name,
            "domain":     summary_row["domain"],
            "values":     feasible_frame["delta_ei"].astype(float).to_numpy(),
            "n_total":    int(len(group_frame)),
            "n_feasible": int(len(feasible_frame)),
        })

    nonempty_rows = [row for row in disc_rows if row["values"].size > 0]
    if not nonempty_rows:
        raise ValueError("no feasible discipline transfer rows remain")

    all_delta = np.concatenate([row["values"] for row in nonempty_rows]).astype(float)
    rng = np.random.default_rng(seed = random_state)
    overall_med = float(np.median(all_delta))
    n_total_estimators = sum(row["n_total"] for row in disc_rows)
    n_feasible_estimators = sum(row["n_feasible"] for row in disc_rows)
    feasibility_share = (
        n_feasible_estimators / n_total_estimators
        if n_total_estimators > 0 else 0.0
    )
    feasibility_label = f"{feasibility_share:.0%}".replace("%", "\\%")

    ## figure scaffolding
    fig = plt.figure(figsize = figsize, facecolor = background_color)
    fig.subplots_adjust(left = 0.21, right = 0.855, top = 0.805, bottom = 0.075)
    forest_axis = fig.add_subplot(111)
    forest_axis.set_facecolor(background_color)

    ## random-split baseline IQR band (within-model residuals on Delta EI scale)
    if baseline_q3 > baseline_q1:
        forest_axis.axvspan(
            baseline_q1, baseline_q3,
            color = "#d9e3ec", alpha = 0.85, zorder = 0,
        )
        forest_axis.axvline(
            baseline_q1, color = "#9fb1c2", lw = 0.6, ls = (0, (2, 3)), zorder = 1,
        )
        forest_axis.axvline(
            baseline_q3, color = "#9fb1c2", lw = 0.6, ls = (0, (2, 3)), zorder = 1,
        )

    ## equivalence band
    forest_axis.axvspan(
        -equivalence_margin, equivalence_margin,
        color = "#e8efe9", alpha = 0.55, zorder = 0,
    )
    forest_axis.axvline(0.0, color = "#7a7a7a", lw = 1.0, zorder = 1)
    forest_axis.axvline(
        -equivalence_margin, color = "#a8b3a9", lw = 0.7, ls = (0, (3, 3)), zorder = 1,
    )
    forest_axis.axvline(
        +equivalence_margin, color = "#a8b3a9", lw = 0.7, ls = (0, (3, 3)), zorder = 1,
    )

    n_disc = len(disc_rows)
    y_positions = [n_disc - 1 - row_index for row_index in range(n_disc)]

    ## domain separators
    previous_domain = None
    for row_index, row in enumerate(disc_rows):
        if previous_domain is not None and row["domain"] != previous_domain:
            forest_axis.axhline(
                n_disc - row_index - 0.5,
                color = "#d0d0d0", lw = 0.6, zorder = 1,
            )
        previous_domain = row["domain"]

    ## render discipline rows
    for row, y_position in zip(disc_rows, y_positions):
        color = palette.get(row["domain"], "#555555")
        values = row["values"]
        if values.size == 0:
            continue
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        jitter = rng.uniform(low = -0.18, high = 0.18, size = values.size)
        forest_axis.scatter(
            values, np.full_like(values, y_position, dtype = float) + jitter,
            s = 10, color = color, alpha = 0.25, edgecolors = "none", zorder = 2,
        )
        forest_axis.plot(
            [q1, q3], [y_position, y_position],
            color = color, lw = 2.0, alpha = 0.55, solid_capstyle = "round", zorder = 3,
        )
        forest_axis.scatter(
            [median], [y_position], s = 48, color = color,
            edgecolors = "white", linewidths = 0.9, zorder = 5,
        )

    ## y-axis discipline labels
    tick_order = np.argsort(y_positions)
    forest_axis.set_yticks([y_positions[index] for index in tick_order])
    forest_axis.set_yticklabels([disc_rows[index]["label"] for index in tick_order], fontsize = 7.6)
    for tick_label, row_index in zip(forest_axis.get_yticklabels(), tick_order):
        domain = disc_rows[row_index]["domain"]
        tick_label.set_color(palette.get(domain, "#555555"))

    ## right-margin domain labels
    for domain in domain_order:
        matching_positions = [
            y_positions[row_index]
            for row_index, row in enumerate(disc_rows)
            if row["domain"] == domain
        ]
        if not matching_positions:
            continue
        y_mid = (min(matching_positions) + max(matching_positions)) / 2.0
        forest_axis.text(
            x = 1.025,
            y = y_mid,
            s = labels.get(domain, domain),
            transform = forest_axis.get_yaxis_transform(),
            ha = "left", va = "center",
            fontsize = 7.0, color = palette.get(domain, "#555555"),
            fontweight = "semibold", linespacing = 1.35,
            clip_on = False,
        )

    ## fixed x-axis scale and cosmetics
    forest_axis.set_xlim(-1.0, 1.0)
    forest_axis.set_xticks(np.linspace(-1.0, 1.0, 9))
    forest_axis.set_xticks(np.arange(-1.0, 1.0001, 0.05), minor = True)
    forest_axis.set_ylim(-0.8, n_disc - 0.2)
    forest_axis.spines[["top", "right", "left"]].set_visible(False)
    forest_axis.spines["bottom"].set_linewidth(0.7)
    forest_axis.xaxis.grid(True, lw = 0.4, color = "#EBEBEB", zorder = 0)
    forest_axis.set_axisbelow(True)
    forest_axis.tick_params(axis = "y", left = False, pad = 4)
    forest_axis.tick_params(axis = "x", labelsize = 8.5)
    forest_axis.set_xlabel(
        "$\\Delta$EI  =  discipline LOGO EI  $-$  pooled random-split baseline EI   "
        "(fixed scale: -1 to +1)",
        fontsize = 9.5, labelpad = 6,
    )
    forest_axis.text(
        x = 0.0, y = 1.012,
        s = (
            f"Per-discipline transfer penalty $\\Delta$EI "
            f"(dot = median; bar = IQR; "
            f"blue = random-split IQR [{baseline_q1:+.2f}, {baseline_q3:+.2f}]; "
            f"green {margin_label} = $\\pm${equivalence_margin:.2f})"
        ),
        transform = forest_axis.transAxes,
        ha = "left", va = "bottom",
        fontsize = 10.5, fontweight = "semibold", color = "#1a1a1a",
    )

    ## header
    fig.text(
        x = 0.50, y = 0.978, s = title,
        ha = "center", va = "top",
        fontsize = 18, fontweight = "bold", color = "#1a1a1a",
    )
    fig.text(
        x = 0.50, y = 0.948, s = subtitle,
        ha = "center", va = "top",
        fontsize = 10, color = "#3a3a3a",
    )

    stat_text = (
        f"median $\\Delta$EI across feasible model×discipline transfers: "
        f"$\\bf{{{overall_med:+.3f}}}$   "
        f"(n = {all_delta.size})"
        f"      ·      "
        f"$\\bf{{{feasibility_label}}}$ raw-EI feasible "
        f"({n_feasible_estimators}/{n_total_estimators}; EI > {feasibility_threshold:.2f})"
    )
    fig.text(
        x = 0.50, y = 0.895, s = stat_text,
        ha = "center", va = "top",
        fontsize = 9.5, color = "#1a1a1a", linespacing = 1.55,
        bbox = {
            "boxstyle":  "round,pad=0.55",
            "facecolor": "#f1efe6",
            "edgecolor": "#d6d2c3",
            "linewidth": 0.6,
        },
    )

    if show:
        plt.show()
    return fig, np.array([forest_axis], dtype = object)


