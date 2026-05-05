## libraries
import pandas as pd
from pandas.io.formats.style import Styler

## modules
from src.evaluators.metrics import FRONTIER_METRICS

## ----------------------------------------------------------------------------
## domain transfer result helper
## ----------------------------------------------------------------------------
def compile_domain_transfer(results: dict) -> pd.DataFrame:

    """
    Desc:
        Converts a dictionary of domain-transfer frontiers into a single
        formatted DataFrame, moving the "model" column to the front.

    Args:
        results: Dict mapping model names to their frontier dataframes.

    Returns:
        A concatenated DataFrame with the 'model' column moved to index 0.
    """

    frame = pd.concat(results.values(), ignore_index = True)
    feat = ["model"] + [c for c in frame.columns if c != "model"]
    return frame[feat]


## ----------------------------------------------------------------------------
## domain transfer results
## ----------------------------------------------------------------------------
def results_domain_transfer(
    *results: dict[str, pd.DataFrame],
    keys: tuple[str, str] = None,
    indicies: tuple[str, str] = None,
    n_repeats: int = 30,
    random_state: int = 42,
    decimals: int = 3
    ) -> pd.DataFrame | Styler:

    """
    Desc:
        Compiles one or more dictionaries of domain-transfer results into a
        single DataFrame. With a single dict, concatenates frontiers and moves
        "model" to the front. With multiple dicts and keys, groups each dict
        by model, computes means, and builds a multi-index summary table.

    Args:
        *results: One or more dicts mapping names to DataFrames.
        keys: Optional list of group labels for multi-index rows.
        indicies: Names for the multi-index levels. Defaults to
            ("procedure", "method").
        n_repeats: Optional number of repeats used in CV, for printing only.
        random_state: Optional base random seed used in CV, for printing only.
        decimals: If set, returns a styled table with left-justified index and
            the given decimal precision. If None, returns a plain DataFrame.

    Returns:
        A DataFrame or Styler of compiled domain-transfer results.
    """

    ## single dict: concatenate and reorder columns
    if len(results) == 1 and keys is None:
        frame = pd.concat(results[0].values(), ignore_index = True)
        feat = ["model"] + [c for c in frame.columns if c != "model"]
        result = frame[feat]

    ## multiple dicts with keys: grouped summary table
    else:
        if keys is None:
            keys = [f"Group {i}" for i in range(len(results))]

        ## print summary header
        first_data = next(iter(results[0].values()))
        n_models = first_data["model"].nunique() if "model" in first_data.columns else None
        if n_repeats is None and "repeat" in first_data.columns:
            n_repeats = int(first_data["repeat"].nunique())
        if n_repeats is not None and random_state is not None:
            print(
                f"Cross-Validation: {n_models} models, {n_repeats} repeats "
                f"(seeds {random_state}-{random_state + n_repeats - 1})"
            )
        else:
            print(f"Cross-Validation: {n_models} models")
        print("Across-model aggregation: median of model-level means")
        print(
            "Resampling: LOGO splits are fixed across repeats; "
            "random k-fold splits are reshuffled across repeats"
        )
        print(
            "Weighting: groups, folds, repeats, and models are equally weighted; "
            "results are not observation-weighted"
        )
        blocks = []
        for key, table in zip(keys, results):
            for label, data in table.items():
                if hasattr(data, "data"):
                    data = data.data
                numeric = data.select_dtypes(include = "number").drop(
                    columns = ["iteration", "repeat", "fold", "n_folds_used"],
                    errors = "ignore",
                )
                if "model" in data.columns:
                    model_means = numeric.groupby(data["model"]).mean()
                    summary = model_means.median()
                    ei_q1 = model_means["ei"].quantile(0.25)
                    ei_q3 = model_means["ei"].quantile(0.75)
                else:
                    summary = numeric.median()
                    ei_q1 = numeric["ei"].quantile(0.25)
                    ei_q3 = numeric["ei"].quantile(0.75)

                ## format ei with iqr
                summary = summary.astype(object)
                if decimals is not None:
                    summary["ei"] = f"{summary['ei']:.{decimals}f} [{ei_q1:.{decimals}f}-{ei_q3:.{decimals}f}]"
                else:
                    summary["ei"] = f"{summary['ei']} [{ei_q1}-{ei_q3}]"
                summary.name = (key, label)
                blocks.append(summary)
        result = pd.DataFrame(blocks)
        result.index = pd.MultiIndex.from_tuples(result.index, names = indicies)

        ## reorder columns with ei first
        cols = result.columns.tolist()
        cols.remove("ei")
        result = result[["ei"] + cols]

        ## capitalize core metrics
        rename_map = {
            c: "EI [IQR]" if c == "ei" else c.upper()
            for c in FRONTIER_METRICS
            if c in result.columns
        }
        result = result.rename(columns = rename_map)

    ## optionally return styled output
    if decimals is not None:
        return result.style.set_table_styles([
            {
                "selector": "th.row_heading, th.index_name",
                "props": [("text-align", "left"), ("vertical-align", "top")],
            }
        ]).format(precision = decimals)
    return result