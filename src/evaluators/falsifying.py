## libraries
import os
import json
import pandas as pd
from pathlib import Path
from typing import Sequence, Dict, Any

## modules
from src.evaluators.metrics import FRONTIER_METRICS


def _build_falsified_from_json(
    data: pd.DataFrame,
    feat_x: list[str],
    feat_z: list[str],
    target: str,
    falsified_dir: str,
) -> dict[str, pd.DataFrame]:

    """Build falsified DataFrames from precomputed JSON payloads.

    This avoids re-generating random falsifications on each run by
    reusing the outputs of `src/data/falsifiers.py` stored in
    `data/falsified/`.

    Returns a dict mapping method name -> falsified DataFrame.
    """

    methods = ["target_remap", "random_generate", "vector_generate"]
    falsified = {method: data.copy(deep=True) for method in methods}

    # Ensure columns can accept float values (JSON outputs are float-heavy).
    for method_df in falsified.values():
        for col in list(feat_x) + list(feat_z) + [target]:
            if col in method_df.columns and pd.api.types.is_integer_dtype(method_df[col].dtype):
                method_df[col] = method_df[col].astype(float)

    for idx, row in data.iterrows():
        name = row.get("name")
        if not isinstance(name, str):
            continue

        json_path = os.path.join(falsified_dir, f"{name}.json")
        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, "r") as fp:
                payload = json.load(fp)
        except Exception:
            continue

        for method in methods:
            payload_method = payload.get(method) or {}
            invariants = payload_method.get("invariants", {})
            signatures = payload_method.get("signatures", {})
            events = payload_method.get("events", [])

            # compute maximum target across events (like data_builder does)
            if events and isinstance(events, list):
                targets = [e.get(target) for e in events if isinstance(e, dict)]
                targets = [t for t in targets if isinstance(t, (int, float))]
                if targets:
                    max_target = max(targets)
                    # preserve column dtype to avoid cast errors
                    try:
                        col_dtype = falsified[method][target].dtype
                        if pd.api.types.is_integer_dtype(col_dtype):
                            max_target = int(round(max_target))
                        else:
                            max_target = float(max_target)
                    except Exception:
                        pass
                    falsified[method].at[idx, target] = max_target

            for col, val in invariants.items():
                if col not in falsified[method].columns:
                    continue
                try:
                    falsified[method].at[idx, col] = val
                except (TypeError, ValueError, pd.errors.LossySetitemError):
                    # If dtype mismatches (e.g., int column, float value),
                    # promote column to float and retry.
                    falsified[method][col] = falsified[method][col].astype(float)
                    falsified[method].at[idx, col] = float(val)

            for col, val in signatures.items():
                if col not in falsified[method].columns:
                    continue
                try:
                    falsified[method].at[idx, col] = val
                except (TypeError, ValueError, pd.errors.LossySetitemError):
                    falsified[method][col] = falsified[method][col].astype(float)
                    falsified[method].at[idx, col] = float(val)

    return falsified


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
    ) -> list[dict]:

    """
    Desc: evaluate a single model on real vs falsified data under
          standard logo-cv. each condition is self-consistent: the
          model is trained on its own data and tested on held-out
          data from the same condition. if the model captures genuine
          structure, performance on real data should dominate.
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
    Returns:
        list of result dicts with frontier metrics per condition per group.
    """

    from src.evaluators.resampling import logo_cross_valid

    print(f"  {model_name}: {method_name}...", end = " ", flush = True)

    rows = []
    for condition, df in [("real", data_real), ("falsified", data_false)]:
        frontier, _ = logo_cross_valid(
            data = df,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = model.estimator_c,
            estimator_r = model.estimator_r,
            target = target,
            group = group,
            n_jobs = 1,
        )
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

    print("done.")
    return rows


## ----------------------------------------------------------------------
## falsifiability test: real vs falsified data
## ----------------------------------------------------------------------
def eval_falsifiability(
    data: pd.DataFrame,
    models: Dict[str, Any],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str = "target",
    group: str = "domain",
    n_jobs: int = -1,
    falsified_dir: Optional[str] = "data/falsified",
    ) -> pd.DataFrame:

    """
    Desc: run falsifiability tests by comparing frontier metrics on real
          data against three falsification methods, each of which breaks
          a specific structural relationship. each condition is evaluated
          under standard logo-cv (train on own data, test on own
          held-out data). if the model captures genuine structure,
          performance on real data should dominate all falsified
          conditions.

          the three methods are:
            1. target_remap: permute y* across rows, keeping features
               fixed. breaks the y* <-> (x\', z\') association.
            2. random_generate: replace all features with independent
               uniform draws, keeping y* fixed. tests whether arbitrary
               features can substitute for real measurements.
            3. vector_generate: replace features with moment-matched
               normal draws, keeping y* fixed. tests whether
               statistically plausible feature vectors suffice.
    Args:
        data: training data with features, target, and group columns.
        models: mapping of model name to estimator with .estimator_c
                and .estimator_r attributes.
        feat_x: graph invariant feature column names.
        feat_z: process signature feature column names.
        target: target column name.
        group: group column name for logo splitting.
        n_jobs: number of parallel model workers (-1 for all cores).
        falsified_dir: optional directory containing precomputed falsified
            JSON payloads (as produced by `src/data/falsifiers.py`). If
            provided and valid, data is loaded from disk instead of
            being randomly generated.
    Returns:
        dataframe with one row per (model, method, condition, group)
        containing frontier metrics for each.
    """

    from joblib import Parallel, delayed

    feat_x = list(feat_x)
    feat_z = list(feat_z)

    ## build falsified datasets at the dataframe level
    if falsified_dir:
        # Allow relative paths (relative to the repo root) in addition to absolute.
        if not os.path.isabs(falsified_dir):
            root = Path(__file__).resolve().parents[2]
            falsified_dir = os.path.join(str(root), falsified_dir)

    if falsified_dir and os.path.isdir(falsified_dir):
        print(f"Using precomputed falsified data from: {falsified_dir}")
        falsified = _build_falsified_from_json(
            data = data,
            feat_x = feat_x,
            feat_z = feat_z,
            target = target,
            falsified_dir = falsified_dir,
        )
    else:
        raise FileNotFoundError(
            f"Falsified data directory not found: {falsified_dir}"
        )

    ## build parallel jobs: one per (model, method)
    jobs = []
    for method_name, data_false in falsified.items():
        for model_name, model in models.items():
            jobs.append((model_name, model, data, data_false,
                         feat_x, feat_z, target, group, method_name))

    print(f"Running {len(jobs)} falsifiability jobs...")
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
        )
        for mn, m, dr, df, fx, fz, t, g, meth in jobs
    )

    ## flatten results
    all_rows = []
    for result_rows in outputs:
        all_rows.extend(result_rows)

    return pd.DataFrame(all_rows)
