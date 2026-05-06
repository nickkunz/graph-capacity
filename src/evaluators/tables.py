## libraries
import pandas as pd
from typing import Sequence


## ----------------------------------------------------------------------
## main table builder
## ----------------------------------------------------------------------
def main_table(
    results_left: pd.DataFrame,
    results_right: pd.DataFrame,
    index_feat: Sequence[str],
    value_feat_left: Sequence[str],
    value_feat_right: Sequence[str],
    header_left: str,
    header_right: str,
    short_feat_left: Sequence[str] | None = None,
    short_feat_right: Sequence[str] | None = None,
    row_order: Sequence[tuple] | None = None,
) -> pd.DataFrame:
    """
    Desc:
        Merge two stat result DataFrames side-by-side and return a
        display-ready table with a two-level MultiIndex column header.
    Args:
        results_left: DataFrame containing index_feat and value_feat_left
            as regular columns (call .reset_index() beforehand if needed).
        results_right: DataFrame containing index_feat and value_feat_right
            as regular columns.
        index_feat: column names to merge on and set as the display index.
        value_feat_left: columns to select from results_left.
        value_feat_right: columns to select from results_right.
        header_left: top-level MultiIndex label for the left columns.
        header_right: top-level MultiIndex label for the right columns.
        short_feat_left: optional display names for the second MultiIndex
            level of the left columns. Defaults to value_feat_left.
        short_feat_right: optional display names for the second MultiIndex
            level of the right columns. Defaults to value_feat_right.
        row_order: optional list of index-value tuples that controls the
            display row order. Each tuple must have one value per index_col
            (e.g. [("network", "rewire"), ...]). Rows not listed are
            appended at the end in their original order.
    Returns:
        DataFrame with a two-level MultiIndex column header, indexed by
        index_feat.
    """

    index_feat = list(index_feat)
    value_feat_left = list(value_feat_left)
    value_feat_right = list(value_feat_right)
    short_feat_left = list(short_feat_left) if short_feat_left is not None else value_feat_left
    short_feat_right = list(short_feat_right) if short_feat_right is not None else value_feat_right

    ## select and merge
    left = results_left[index_feat + value_feat_left].copy()
    right = results_right[index_feat + value_feat_right].copy()
    merged = left.merge(right, on = index_feat)

    ## set display index
    merged = merged.set_index(index_feat)

    ## apply row ordering
    if row_order is not None:
        ordered = [k if len(index_feat) > 1 else k[0] for k in row_order]
        present = [k for k in ordered if k in merged.index]
        rest = [k for k in merged.index if k not in set(present)]
        merged = merged.loc[present + rest]

    ## assign two-level column header
    merged.columns = pd.MultiIndex.from_tuples(
        [(header_left, col) for col in short_feat_left]
        + [(header_right, col) for col in short_feat_right]
    )

    return merged
