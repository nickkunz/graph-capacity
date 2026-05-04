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