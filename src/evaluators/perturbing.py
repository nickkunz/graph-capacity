## libraries
import sys
import random
import warnings
import igraph as ig
import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path
from typing import Sequence, Optional, Tuple, Dict, Literal, Any
from sklearn.utils import resample
from sklearn.base import BaseEstimator
from joblib import parallel, Parallel, delayed
from joblib.parallel import BatchCompletionCallBack
from contextlib import contextmanager
from scipy.stats import rankdata, wilcoxon
from tqdm import tqdm

## path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.append(str(root))
    
## modules
from src.evaluators.metrics import FRONTIER_METRICS
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.evaluators.resampling import (
    logo_cross_valid,
    logo_cross_valid_frozen
)
from src.data.helpers import (
    _force_finite, 
    _force_finite_dict,
    _clip_unit_interval
)

## joblib progress bar bridge
@contextmanager
def _tqdm_joblib(total: int, desc: str):
    pbar = tqdm(total = total, desc = desc, unit = "job")
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

## entropy of the degree distribution
def _degree_entropy(degree: np.ndarray, n_nodes: int) -> float:
    if n_nodes <= 0 or degree.size == 0:
        return 0.0
    counts = np.bincount(degree)
    probs = counts[counts > 0].astype(float) / float(n_nodes)
    eps = 1e-16
    return -float(np.sum(probs * np.log(probs + eps)))

## entropy of the degree distribution weighted by degree
def _joint_degree_entropy(degree: np.ndarray, denom: float) -> float:
    if denom <= 0.0 or degree.size == 0:
        return 0.0

    counts = np.bincount(degree)
    nonzero = np.flatnonzero(counts)

    ## stub-weighted degree probabilities pk ~ k * p(k) / <k>
    pk = (nonzero.astype(float) * counts[nonzero].astype(float)) / float(denom)
    pk = pk[pk > 0]

    eps = 1e-16
    marginal = -float(np.sum(pk * np.log(pk + eps)))
    return 2.0 * marginal

## skewness and kurtosis with guard for low-variance cases
def _skew_kurtosis(degree: np.ndarray, discrete: bool = True) -> Tuple[float, float]:
    x = degree.astype(int) if discrete else degree
    skewness = 0.0
    kurtosis = 0.0

    ## guard: need variance > 0 (unique values for int, std for float)
    has_spread = np.unique(x).size > 1 if discrete else float(np.std(x)) > 1e-9
    if x.size >= 3 and has_spread:
        skewness = _force_finite(stats.skew(x, bias = False), default = 0.0)
    if x.size >= 4 and has_spread:
        kurtosis = _force_finite(stats.kurtosis(x, fisher = True, bias = False), default = 0.0)

    return float(skewness), float(kurtosis)

## core invariants that can be estimated from the degree sequence alone
def _degree_invariants(degree: np.ndarray, n_nodes: int, n_edges: float, keys: Sequence[str]) -> Dict[str, float]:
    if n_nodes <= 0 or n_edges <= 0.0 or degree.size == 0: return {k: 0.0 for k in keys}
    
    S1, S2, degree_max = float(np.sum(degree)), float(np.sum(degree ** 2)), float(np.max(degree))
    mean_excess = (S2 / max(S1, 1.0)) - 1.0
    diam = max(2.0, float(np.log(max(n_nodes, 2))) / float(np.log(mean_excess))) if mean_excess > 1.0 else float(n_nodes)
    
    is_discrete = np.allclose(degree, np.round(degree))
    degree_int = np.round(degree).astype(int)
    inv_d = 1.0 / np.maximum(degree, 1.0)

    feat = {
        "n_nodes": float(n_nodes),
        "n_edges": n_edges,
        "diameter": diam,
        "radius": 0.5 * diam,
        "degeneracy": min(degree_max, float(np.sqrt(2.0 * n_edges))),
        "maximum_degree": degree_max,
        "degree_variance": float(np.var(degree)),
        "degree_entropy": _degree_entropy(degree_int, n_nodes),
        "joint_degree_entropy": _joint_degree_entropy(degree_int, float(np.sum(degree_int)) if not is_discrete else S1) if n_edges > 0 else 0.0,
        "normalized_laplacian_second_moment": _force_finite(1.0 + (2.0 * n_edges * (n_nodes / max(S1, 1.0)) ** 2) / max(n_nodes, 1.0), 0.0),
        "normalized_laplacian_third_moment": _force_finite(1.0 + (((S2 - S1) ** 3) / max(S1 ** 3 * n_nodes, 1.0)), 0.0),
        "random_walk_triangle_weight": _force_finite(6.0 * (((S2 - S1) ** 2) / max(6.0 * S1, 1.0)) * float(inv_d.mean()) / max(S1, 1.0), 0.0),
        "random_walk_fourth_moment": _force_finite((1.0 / max(n_nodes, 1.0)) * float(np.sum(degree * S2 / max(S1 ** 2, 1.0) + max(mean_excess, 0.0) * inv_d)), 0.0),
        "adjacency_fourth_moment_per_node": _force_finite((S1 + (S2 - S1) + (S2 ** 2) / max(2.0 * S1, 1.0)) / max(n_nodes, 1.0), 0.0)
    }
    feat["k_core_size"] = float(np.sum(degree >= feat["degeneracy"]))
    feat["degree_skewness"], feat["degree_kurtosis"] = _skew_kurtosis(degree, discrete=is_discrete)
    
    for k in keys: feat.setdefault(k, 0.0)
    return feat

## degree-preserving rewiring
def _rewire_estimate(
    invariants: Dict[str, float],
    degrees: np.ndarray,
    n_nodes: int,
    n_edges: int,
    intensity: float,
    ) -> Dict[str, float]:
    """
    Desc:
        Interpolates between observed invariants and configuration-model
        expectations under degree-preserving rewiring.

    Args:
        invariants: Observed invariant dict.
        degrees: 1-d array of vertex degrees.
        n_nodes: Number of vertices.
        n_edges: Number of edges.
        intensity: Perturbation strength in [0, 1].

    Returns:
        Dict of estimated invariants.
    """
    alpha = _clip_unit_interval(1.0 - (1.0 - float(intensity)) ** 2)
    keys = list(invariants.keys())
    cm_invariants = _degree_invariants(degrees, n_nodes, float(n_edges), keys)

    out: Dict[str, float] = {}
    for k in keys:
        b0 = float(invariants.get(k, 0.0))
        c0 = float(cm_invariants.get(k, b0))
        out[k] = (1.0 - alpha) * b0 + alpha * c0

    return out

## node sampling estimate (analytical)
def _node_sample_estimate(
    invariants: Dict[str, float],
    degrees: np.ndarray,
    n_nodes: int,
    n_edges: int,
    intensity: float,
) -> Dict[str, float]:
    """
    Desc:
        Estimates invariants after uniform random node removal.
        Fraction `intensity` of nodes are removed; surviving edges
        are those whose *both* endpoints remain.

    Args:
        invariants: Observed invariant dict.
        degrees: 1-d array of vertex degrees.
        n_nodes: Number of vertices.
        n_edges: Number of edges.
        intensity: Fraction of nodes to remove in [0, 1].

    Returns:
        Dict of estimated invariants.
    """
    p = 1.0 - _clip_unit_interval(float(intensity))  # survival probability
    keys = list(invariants.keys())

    if p <= 0.0:
        return {k: 0.0 for k in keys}
    if p >= 1.0:
        return dict(invariants)

    degree = np.asarray(degrees).reshape(-1).astype(float)

    ## surviving node count and expected degree after node sampling
    n_nodes_p = max(int(round(n_nodes * p)), 1)
    degree_p = degree * p  # each neighbour survives with probability p
    n_edge_p = float(n_edges) * (p ** 2)  # both endpoints must survive

    out = _degree_invariants(degree_p, n_nodes_p, n_edge_p, keys)

    ## overrides that need the original base values
    out["n_articulation_points"] = float(
        invariants.get("n_articulation_points", 0.0)
    ) * p

    out["n_bridges"] = float(
        invariants.get("n_bridges", 0.0)
    ) * p

    out["global_clustering"] = float(
        invariants.get("global_clustering", 0.0)
    )  # clustering coefficient is scale-free under uniform sampling

    out["degree_assortativity"] = float(
        invariants.get("degree_assortativity", 0.0)
    )

    for k in keys:
        out.setdefault(k, float(invariants.get(k, 0.0)) * p)

    return out


## bernoulli edge densification (analytical)
def _densify_estimate(
    invariants: Dict[str, float],
    degrees: np.ndarray,
    n_nodes: int,
    n_edges: int,
    intensity: float,
    ) -> Dict[str, float]:
    """
    Desc:
        Estimates invariants after adding random edges uniformly among
        non-edges. The `intensity` fraction of existing edges is the
        *expected number of new edges* to add, drawn from the complement
        graph via independent Bernoulli trials.

    Args:
        invariants: Observed invariant dict.
        degrees: 1-d array of vertex degrees.
        n_nodes: Number of vertices.
        n_edges: Number of edges.
        intensity: Perturbation strength in [0, 1] as a fraction of |E|.

    Returns:
        Dict of estimated invariants.
    """
    x = _clip_unit_interval(float(intensity))
    keys = list(invariants.keys())

    if x <= 0.0:
        return dict(invariants)

    degree = np.asarray(degrees).reshape(-1).astype(float)

    ## number of edges to add and complement size
    n_add = float(n_edges) * x
    max_edges = float(n_nodes * (n_nodes - 1)) / 2.0
    n_complement = max(max_edges - float(n_edges), 1.0)

    ## per non-edge addition probability
    q = min(n_add / n_complement, 1.0)

    ## densified degree sequence: each node gains non-neighbour edges
    degree_q = degree + (float(n_nodes - 1) - degree) * q
    n_edge_q = float(n_edges) + n_add

    out = _degree_invariants(degree_q, n_nodes, n_edge_q, keys)

    ## overrides for structural invariants
    out["n_articulation_points"] = float(
        invariants.get("n_articulation_points", 0.0)
    ) * max(1.0 - q, 0.0)

    out["n_bridges"] = float(
        invariants.get("n_bridges", 0.0)
    ) * max(1.0 - q, 0.0)

    ## clustering increases with densification: added edges create
    ## new triangles; approximate via ER triangle probability
    base_clustering = float(invariants.get("global_clustering", 0.0))
    out["global_clustering"] = min(base_clustering + (1.0 - base_clustering) * q, 1.0)

    out["degree_assortativity"] = float(
        invariants.get("degree_assortativity", 0.0)
    ) * max(1.0 - q, 0.0)

    for k in keys:
        out.setdefault(k, float(invariants.get(k, 0.0)))

    return out

## ----------------------------------------------------------------------------
## analytical perturbation
## ----------------------------------------------------------------------------
def analytical_perturb(
    invariants: Dict[str, float],
    degrees: np.ndarray,
    n_nodes: int,
    n_edges: int,
    method: Literal["degree_preserving_rewire", "uniform_node_sampling", "bernoulli_edge_densification"] = "degree_preserving_rewire",
    intensity: float = 0.1,
    ) -> Dict[str, float]:
    
    """
    Desc:
        Estimates graph invariants after a topological perturbation without
        constructing the perturbed graph. Returns the same key set as
        invariants.

    Args:
        invariants: Dict returned by GraphInvariants(graph).all().
        degrees: 1-d array of vertex degrees.
        n_nodes: Number of vertices.
        n_edges: Number of edges.
        method: Analytical perturbation model.
        intensity: Perturbation strength in [0, 1].

    Returns:
        Dict of estimated invariants (finite floats).

    Raises:
        ValueError: If perturbation model is unsupported.
    """

    degree = np.asarray(degrees).reshape(-1).astype(float)
    x = _clip_unit_interval(float(intensity))

    if method == "degree_preserving_rewire":
        out = _rewire_estimate(invariants, degree, n_nodes, n_edges, x)
        return _force_finite_dict(out)

    if method == "uniform_node_sampling":
        out = _node_sample_estimate(invariants, degree, n_nodes, n_edges, x)
        return _force_finite_dict(out)

    if method == "bernoulli_edge_densification":
        out = _densify_estimate(invariants, degree, n_nodes, n_edges, x)
        return _force_finite_dict(out)

    raise ValueError(f"unsupported perturbation model: {method}")

## ----------------------------------------------------------------------------
## network perturbation
## ----------------------------------------------------------------------------
def network_perturb(
    graph: ig.Graph,
    method: str = "rewire",
    intensity: float = 0.1,
    n_swaps: Optional[int] = None,
    random_state: int = 42,
    ) -> dict:
    
    """
    Desc:
        Modifies graph topology and recomputes graph invariants.

    Args:
        graph: Original igraph.Graph object.
        method: Perturbation method ('rewire', 'node_sample', 'densify').
        intensity: Fraction of edges/nodes to modify.
        n_swaps: Number of rewiring swaps (for rewire). Defaults to |n_edges| * intensity.

    Returns:
        Dict of recomputed graph invariants.

    Raises:
        ValueError: If method is unknown.
    """

    ## rng init
    rng = np.random.default_rng(random_state)

    ## init graph
    G = graph.copy()
    n_edges = G.ecount()
    n_nodes = G.vcount()

    ## degree-preserving rewiring
    if method == "rewire":
        if n_swaps is None:
            n_swaps = max(1, int(n_edges * intensity)) if intensity > 0 else 0
        ig.set_random_number_generator(random.Random(random_state))
        G.rewire(n = n_swaps, mode = "simple")

    ## node sampling (remove nodes)
    elif method == "node_sample":
        n_remove = int(n_nodes * intensity)
        if n_remove > 0 and n_remove < n_nodes:
            nodes_to_remove = rng.choice(n_nodes, n_remove, replace = False).tolist()
            G.delete_vertices(nodes_to_remove)

    ## densification (add edges)
    elif method == "densify":
        n_add = int(n_edges * intensity)
        if n_add > 0:
            existing = set(tuple(sorted(e.tuple)) for e in G.es)
            complement = [
                (u, v) for u in range(n_nodes)
                for v in range(u + 1, n_nodes)
                if (u, v) not in existing
            ]
            if len(complement) > 0:
                n_add = min(n_add, len(complement))
                chosen = rng.choice(len(complement), size = n_add, replace = False)
                for idx in chosen:
                    G.add_edge(*complement[idx])

    else:
        raise ValueError(f"unknown network perturbation method: {method}")

    ## recompute invariants
    return GraphInvariants(G).all()

## ----------------------------------------------------------------------------
## invariant perturbation
## ----------------------------------------------------------------------------
def invariant_perturb(
    X: pd.DataFrame,
    method: str = "noise",
    noise: float = 0.05,
    subset: float = 0.8,
    random_state: int = 42,
    ) -> pd.DataFrame:
    
    """
    Desc:
        Directly modifies invariant encoding.

    Args:
        X: DataFrame of graph invariants.
        method: Perturbation method ('noise', 'jitter', 'subset').
        noise: Standard deviation of noise (relative to feature std).
        subset: Fraction of features to keep (for subset).

    Returns:
        Perturbed feature matrix.

    Raises:
        ValueError: If method is unknown.
    """

    rng = np.random.default_rng(random_state)

    ## copy to avoid modifying original
    X_new = X.copy()

    ## additive gaussian noise scaled by feature standard deviation
    if method == "noise":
        for col in X_new.columns:
            std = X_new[col].std()
            if not (std > 0):
                std = float(np.mean(np.abs(X_new[col])))
            if std > 0:
                X_new[col] += rng.normal(0, std * noise, size = len(X_new))

    ## multiplicative jitter to simulate measurement error
    elif method == "jitter":
        jitter = rng.normal(0, noise, size = X_new.shape)
        X_new *= (1 + jitter)

        ## clip only inherently non-negative invariants
        for col in X_new.columns:
            if (X[col] >= 0).all():
                X_new[col] = np.clip(X_new[col], a_min = 0, a_max = None)

    ## random feature ablation (permute values to destroy structure)
    elif method == "subset":
        n_features = X_new.shape[1]
        n_keep = int(n_features * subset)
        drop_indices = rng.choice(
            X_new.columns,
            size = n_features - n_keep,
            replace = False
        )
        for col in drop_indices:
            X_new[col] = rng.permutation(X_new[col].values)

    else:
        raise ValueError(f"unknown invariant perturbation method: {method}")

    return X_new

## ----------------------------------------------------------------------------
## process perturbation
## ----------------------------------------------------------------------------
def process_perturb(
    counts: np.ndarray,
    method: str = "scaling",
    param: float = 1.0,
    random_state: int = 42,
    ) -> dict:
    
    """
    Desc:
        Modifies process signatures by perturbing the underlying count process.

    Args:
        counts: Array of event counts (aggregated time series).
        method: Perturbation method ('scaling', 'smoothing', 'burst_smoothing', 'bootstrapping').
        param: Parameter for the method (scale factor, window size, etc).

    Returns:
        Dict of recomputed process signatures.

    Raises:
        ValueError: If method is unknown.
    """

    rng = np.random.default_rng(random_state)

    ## copy counts to avoid modifying original
    S_new = counts.copy().astype(float)

    ## power-law scaling to reshape the intensity distribution
    if method == "scaling":
        S_new = np.power(np.maximum(S_new, 0) + 1, float(param)) - 1

    ## rolling average smoothing to reduce noise (param = window size)
    elif method == "smoothing":
        window = int(max(1, param))
        if window > 1:
            S_new = pd.Series(S_new).rolling(
                window = window,
                min_periods = 1
            ).mean().values

    ## log-damping to reduce burstiness and heavy tails
    elif method == "burst_smoothing":
        S_new = np.log1p(S_new)

    ## destroy temporal structure (param = fraction of series to resample)
    elif method == "bootstrapping":
        n = len(S_new)
        k = max(1, int(round(n * float(param))))
        S_new = rng.choice(S_new, size = k, replace = True)
        if k < n:
            S_new = np.concatenate([S_new, rng.choice(S_new, size = n - k, replace = True)])

    else:
        raise ValueError(f"unknown process perturbation method: {method}")

    ## recompute signatures
    data_temp = pd.DataFrame({"counts": S_new, "idx": range(len(S_new))})
    signatures = ProcessSignatures(data_temp, sort_by = ["idx"], target = "counts")
    return signatures.all()

## ----------------------------------------------------------------------------
## signature perturbation
## ----------------------------------------------------------------------------
def signature_perturb(
    X: pd.DataFrame,
    Z: pd.DataFrame,
    y: pd.Series,
    method: str = "bootstrap",
    fraction: float = 1.0,
    random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Desc:
        Modifies raw observations of the (X, Z, y) tuples.

    Args:
        X: DataFrame of graph invariants.
        Z: DataFrame of process signatures.
        y: Series of target values.
        method: Perturbation method ('bootstrap', 'subsample', 'additive_noise').
        fraction: Fraction of samples (for subsample), or noise level (for additive_noise).

    Returns:
        Tuple of (X_new, Z_new, y_new).
    """

    n_samples = len(y)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(random_state)

    if method == "bootstrap":
        ## bootstrap keeps original sample size
        new_indices = resample(indices, n_samples = n_samples, replace = True, random_state = random_state)

    elif method == "subsample":
        ## subsample uses fraction
        new_n = int(max(1, np.floor(n_samples * fraction)))
        new_indices = resample(indices, n_samples = new_n, replace = False, random_state = random_state)

    elif method == "additive_noise":
        ## additive gaussian noise scaled by y standard deviation
        y_new = y.copy().astype(float)
        std = float(y_new.std())
        if std > 0:
            noise = rng.normal(0, std * fraction, size = n_samples)
            y_new = y_new + noise
            y_new = np.maximum(y_new, 0.0)  # counts are non-negative
        return X.copy(), Z.copy(), pd.Series(y_new, name = y.name)

    else:
        new_indices = indices

    return X.iloc[new_indices], Z.iloc[new_indices], y.iloc[new_indices]

## ----------------------------------------------------------------------------
## temporal perturbation
## ----------------------------------------------------------------------------
def temporal_perturb(
    event_times: Sequence[float],
    scale: str = "1D",
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    ) -> Tuple[float, dict]:
    
    """
    Desc:
        Modifies aggregation target temporal resolution.
        Recomputes y(Delta_t) and S(Delta_t).

    Args:
        event_times: List/Array of raw event timestamps **or day offsets
                     (integers / floats)**.
        scale: Pandas offset alias (e.g., '1D', '1H', '15min').
        start_time: Start of observation window.
        end_time: End of observation window.

    Returns:
        Tuple of (max_rate_y, signatures_dict).

    Notes:
        If `event_times` are numeric the function assumes they represent
        days since some arbitrary origin.  In that case `scale` is expected
        to be a days‑based alias (e.g. '2D', '7D') and `y` is reported per
        day rather than per second.
    """

    if len(event_times) == 0:
        return 0.0, {}

    arr = np.asarray(event_times)

    # ---------- numeric branch ---------- #
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):

        # heuristic: if values look like Unix epoch timestamps (> 1e8),
        # convert to proper datetimes and fall through to the timestamp branch
        if float(np.median(arr)) > 1e8:
            try:
                ts = pd.to_datetime(arr, unit='s')
            except Exception:
                ts = pd.to_datetime(arr, unit='ms')
        else:
            # treat values as day indices
            days = arr.astype(int)
            lo, hi = int(days.min()), int(days.max())
            full_idx = np.arange(lo, hi + 1)
            counts = pd.Series(1, index=days)
            daily_counts = counts.groupby(level=0).sum().reindex(full_idx, fill_value=0)
            daily_counts = daily_counts.sort_index()

            # parse scale duration in days ('7D' -> 7, '14D' -> 14)
            if scale.endswith("D"):
                try:
                    bin_width = int(scale[:-1]) if scale[:-1] else 1
                except Exception:
                    bin_width = 1
            else:
                bin_width = 1
            bin_width = max(bin_width, 1)

            # re-aggregate daily counts into multi-day windows
            if bin_width > 1:
                day_vals = daily_counts.index.values
                bin_labels = (day_vals - lo) // bin_width
                counts_binned = daily_counts.groupby(bin_labels).sum()
            else:
                counts_binned = daily_counts

            y_count = float(counts_binned.max())
            y_val = y_count / float(bin_width)

            data_temp = pd.DataFrame({"counts": counts_binned.values, "idx": range(len(counts_binned))})
            sigs = ProcessSignatures(data_temp, sort_by=["idx"], target="counts")
            return y_val, sigs.all()
    else:
        # ---------- parse to datetime ---------- #
        ts = pd.to_datetime(event_times)

    # ---------- timestamp branch ---------- #
    if start_time is None:
        start_time = ts.min()
    if end_time is None:
        end_time = ts.max()

    full_range = pd.date_range(start=start_time, end=end_time, freq=scale)

    df = pd.DataFrame({"t": ts}).set_index("t")
    counts_binned = (
        df.assign(count=1)
        .resample(scale)
        .sum()
        .reindex(full_range, fill_value=0)
    )["count"]

    y_count = float(counts_binned.max())
    if len(counts_binned) >= 2:
        duration = float((counts_binned.index[1] - counts_binned.index[0]).total_seconds())
    else:
        duration = float(pd.Timedelta(scale).total_seconds())

    duration = max(duration, 1.0)
    y_val = y_count / duration

    data_temp = pd.DataFrame({"counts": counts_binned.values, "idx": range(len(counts_binned))})
    sigs = ProcessSignatures(data_temp, sort_by=["idx"], target="counts")
    return y_val, sigs.all()


## ----------------------------------------------------------------------------
## perturbation evaluation pipeline
## ----------------------------------------------------------------------------

## feature mapping: perturbation type to feature columns
FEAT_MAP = {
    "network":    "x",
    "invariants": "x",
    "process":    "z",
    "signatures":  "z",
    "temporal":   "z",
}

## json key to perturbation type
KEY_TO_TYPE = {
    "network_perturbed":    "network",
    "invariants_perturbed": "invariants",
    "process_perturbed":    "process",
    "signatures_perturbed": "signatures",
    "temporal_perturbed":   "temporal",
}

## worker for a single perturbation setting
def _run_perturbation(
    model_name: str,
    model: BaseEstimator,
    pert_type: str,
    method: str,
    intensity: str,
    pert_df: pd.DataFrame,
    data: pd.DataFrame,
    feat_cols: Sequence[str],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    group: str,
    target: str,
    random_state: int,
    ) -> dict | None:

    """
    Desc: worker for a single (model, perturbation, method, intensity)
          combination. runs frozen + retrain logo-cv.
    Args:
        model_name: name of the estimator.
        model: estimator with .estimator_c and .estimator_r attributes.
        pert_type: perturbation family name.
        method: perturbation method within the family.
        intensity: perturbation intensity level.
        pert_df: perturbed feature dataframe indexed by dataset.
        data: clean baseline dataframe.
        feat_cols: feature columns affected by this perturbation type.
        feat_x: graph invariant feature column names.
        feat_z: process signatures feature column names.
        group: group column name.
        target: target column name.
        random_state: base random state for repeat reproducibility.
    Returns:
        dict with "key", "frozen", and "retrain", or None if skipped.
    """

    lookup = pert_df.set_index("dataset")
    data_mod = data.copy()
    for col in feat_cols:
        if col in lookup.columns:
            data_mod[col] = data_mod["name"].map(lookup[col])

    ## temporal perturbation: also replace target
    if pert_type == "temporal" and target in lookup.columns:
        data_mod[target] = data_mod["name"].map(lookup[target])

    ## drop rows without perturbation data
    required = list(feat_cols)
    if pert_type == "temporal":
        required = required + [target]
    data_mod = data_mod.dropna(subset = required).reset_index(drop = True)
    if len(data_mod) < 2:
        return None

    key = (model_name, pert_type, method, intensity)

    ## frozen manifold: train on clean, evaluate on perturbed
    frontier_fr, _, _ = logo_cross_valid_frozen(
        data_train = data,
        data_test = data_mod,
        feat_x = feat_x,
        feat_z = feat_z,
        estimator_c = model.estimator_c,
        estimator_r = model.estimator_r,
        target = target,
        group = group,
        random_state = random_state,
        n_jobs = 1,
    )
    frontier_fr["model"] = model_name
    frontier_fr["perturbation"] = pert_type
    frontier_fr["method"] = method
    frontier_fr["intensity"] = intensity

    ## retrain manifold: train on perturbed, evaluate on perturbed
    frontier_rt, _ = logo_cross_valid(
        data = data_mod,
        feat_x = feat_x,
        feat_z = feat_z,
        estimator_c = model.estimator_c,
        estimator_r = model.estimator_r,
        target = target,
        group = group,
        random_state = random_state,
        n_jobs = 1,
    )
    frontier_rt["model"] = model_name
    frontier_rt["perturbation"] = pert_type
    frontier_rt["method"] = method
    frontier_rt["intensity"] = intensity

    return {"key": key, "frozen": frontier_fr, "retrain": frontier_rt}

## collect per-group frontier metrics for each perturbation setting
def _aggregate_frontier(results_dict: dict, track: str) -> pd.DataFrame:

    """
    Desc: collect per-group frontier metrics for each perturbation setting.
          emits one row per (model, perturbation, method, intensity, group)
          so downstream paired statistics can pair on (model, group) rather
          than averaging domains out before pairing.
    Args:
        results_dict: mapping of (model, pert_type, method, intensity)
                      to frontier dataframe.
        track: label for the evaluation track (default "frozen").
    Returns:
        dataframe with one row per perturbation setting and group.
    """

    rows = []
    for (model_name, pert_type, method, intensity), frontier in results_dict.items():
        for _, frow in frontier.iterrows():
            row = {
                "track": track,
                "model": model_name,
                "perturbation": pert_type,
                "method": method,
                "intensity": intensity,
                "group": frow["group"],
            }
            for col in FRONTIER_METRICS:
                row[col] = frow[col]
            rows.append(row)
    return pd.DataFrame(rows)

## ----------------------------------------------------------------------------
## main evaluation pipeline
## ----------------------------------------------------------------------------
def eval_perturbed(
    data: pd.DataFrame,
    models: Dict[str, Any],
    data_pert: dict,
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    group: str = "domain",
    target: str = "target",
    random_state: int = 42,
    n_jobs: int = -1
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Desc: run the full perturbation evaluation pipeline. computes baseline
          logo-cv for each model, then evaluates every perturbation setting
          under frozen + retrain tracks. returns aggregated metrics with
          directed deltas and recovery ratios.
    Args:
        data: clean baseline dataframe with features, target, and group columns.
        models: mapping of model name to estimator with .estimator_c and
                .estimator_r attributes.
        data_pert: nested perturbation dict from load_perturbed_data()
                   with schema {json_key: {method: {intensity: DataFrame}}}.
        feat_x: graph invariant feature column names.
        feat_z: process signatures feature column names.
        group: group column name.
        target: target column name.
        random_state: base random state for seed reproducibility (default 42).
        n_jobs: number of parallel workers (-1 for all cores).
    Returns:
        tuple of (results_data, recovery_data).
        results_data: full aggregated metrics for both tracks including baselines,
            with directed delta columns (Δ *) for non-baseline rows (NaN for baseline).
        recovery_data: recovery ratios (ρ *) measuring fraction of frozen-track
            degradation eliminated by retraining.
    """

    ## resolve feature column mapping
    feat_lookup = {"x": list(feat_x), "z": list(feat_z)}

    ## baselines (one per model)
    results_frozen = dict()
    results_retrain = dict()
    for model_name, model in models.items():
        frontier_base, _ = logo_cross_valid(
            data = data,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = model.estimator_c,
            estimator_r = model.estimator_r,
            target = target,
            group = group,
            random_state = random_state,
            n_jobs = 1,
        )
        frontier_base["model"] = model_name
        results_frozen[(model_name, "baseline", None, None)] = frontier_base
        results_retrain[(model_name, "baseline", None, None)] = frontier_base

    ## build job list
    jobs = []
    for json_key, methods in data_pert.items():
        pert_type = KEY_TO_TYPE.get(json_key)
        if pert_type not in FEAT_MAP:
            continue
        feat_cols = feat_lookup[FEAT_MAP[pert_type]]
        for method, intensities in methods.items():
            for intensity, pert_df in intensities.items():
                for model_name, model in models.items():
                    jobs.append((
                        model_name, model, pert_type, method, intensity,
                        pert_df, data, feat_cols, feat_x, feat_z, group, target,
                        random_state,
                    ))

    ## parallel execution
    if jobs:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action = "ignore",
                message = ".*A worker stopped while some jobs were given to the executor.*",
                category = UserWarning,
            )
            with _tqdm_joblib(total = len(jobs), desc = "Perturbation training"):
                outputs = Parallel(n_jobs = n_jobs, verbose = 0)(
                    delayed(_run_perturbation)(*args) for args in jobs
                )
    else:
        outputs = list()

    ## collect results
    n_ok = 0
    for result in outputs:
        if result is None:
            continue
        key = result["key"]
        results_frozen[key] = result["frozen"]
        results_retrain[key] = result["retrain"]
        n_ok += 1

    ## aggregate frontier metrics across groups for both tracks
    agg_frozen = _aggregate_frontier(results_dict = results_frozen, track = "frozen")
    agg_retrain = _aggregate_frontier(results_dict = results_retrain, track = "retrain")
    results_data = pd.concat([agg_frozen, agg_retrain], ignore_index = True)

    ## baseline lookup keyed on (model, group) to match per-group aggregation
    baseline_lookup = (
        agg_frozen.query("perturbation == 'baseline'")
        .drop_duplicates(subset = ["model", "group"])
        .set_index(["model", "group"])[FRONTIER_METRICS]
    )

    ## directed deltas (positive = degradation; NaN for baseline rows)
    pair_index = pd.MultiIndex.from_arrays([
        results_data["model"].to_numpy(),
        results_data["group"].to_numpy(),
    ])
    for col in FRONTIER_METRICS:
        base = pd.Series(
            baseline_lookup[col].reindex(pair_index).to_numpy(),
            index = results_data.index,
        )
        delta = (base - results_data[col]) if col == "ei" else (results_data[col] - base)
        results_data[f"Δ {col.upper()}"] = delta.where(results_data["perturbation"] != "baseline")

    ## recovery ratios: fraction of frozen degradation eliminated by retraining
    delta_cols = [f"Δ {c.upper()}" for c in FRONTIER_METRICS]
    pert_rows = results_data.query("perturbation != 'baseline'")
    frozen_deltas = (
        pert_rows.query("track == 'frozen'")
        .set_index(["model", "perturbation", "method", "intensity", "group"])[delta_cols]
    )
    retrain_deltas = (
        pert_rows.query("track == 'retrain'")
        .set_index(["model", "perturbation", "method", "intensity", "group"])[delta_cols]
    )
    common_idx = frozen_deltas.index.intersection(retrain_deltas.index)
    fr = frozen_deltas.loc[common_idx]
    rt = retrain_deltas.loc[common_idx]
    recovery = (fr - rt) / fr.replace(0, np.nan)
    recovery.columns = [c.replace("Δ", "ρ") for c in recovery.columns]
    recovery_data = recovery.reset_index()
    recovery_data["perturbation"] = recovery_data["perturbation"].astype(str)

    return results_data, recovery_data

## ----------------------------------------------------------------------------
## statistical testing with summary tables
## ----------------------------------------------------------------------------
def stat_perturbed_test(
    results: pd.DataFrame,
    feat_value: Sequence[str],
    feat_pairs: Sequence[str] | None = None,
    feat_group: Sequence[str] = ["track", "method"],
    pert_type: str | None = None,
    track: str | Sequence[str] | None = None,
    label_pert: str = "perturbation",
    label_base: str = "baseline",
    decimals: int = 4,
    index: bool = True,
    ) -> pd.DataFrame:

    """
    Desc:
        Paired Wilcoxon signed-rank summary comparing original baseline vs
        perturbed metrics under each perturbation grouping.

    Args:
        results: Full aggregated output from eval_perturb, including
            baseline rows.
        feat_value: Metric columns to test (for example ["ei"]).
        feat_pairs: Columns that align a baseline row with each perturbed
            row. Defaults to ["model", "group"] to match the falsification
            pipeline's (model × domain) pairing convention.
        feat_group: Columns whose unique combinations define independent
            tests (default ["track", "method"]).
        pert_type: Perturbation type to restrict to (e.g. "network",
            "invariants", "process", "signatures").
            None -> use all perturbation types.
        label_pert: Column that flags baseline vs perturbed rows.
        label_base: Value in label_pert for baseline rows.
        track: Evaluation track to restrict to (e.g. "frozen", "retrain").
            None -> use all tracks.
        decimals: Number of decimal places for display (default 4).
        index: If True, set group columns as DataFrame index.

    Returns:
        Display-ready table with columns:
        [*feat_group, Metric?, Median <M> (Original), Median <M> (Perturbed),
        Median Δ <M>, Positive Δ , Wilcoxon W+, Rank-biserial r, One-sided p,
        Holm-adj. p, Sig.].
    """

    ## filter to a single perturbation type when specified
    if pert_type is not None:
        results = results.loc[
            results[label_pert].isin([label_base, pert_type])
        ].copy()

    ## filter to specified track(s)
    if track is not None and "track" in results.columns:
        track_vals = [track] if isinstance(track, str) else list(track)
        results = results.loc[results["track"].isin(track_vals)].copy()

    feat_value = list(feat_value)
    feat_group = list(feat_group or [])
    group_display = [c.replace("_", " ").title() for c in feat_group]
    p_label = "One-sided p"
    tail_cols = ["Wilcoxon W+", "Rank-biserial r", p_label, "Holm-adj. p", "Sig"]
    pair_cols = list(feat_pairs) if feat_pairs is not None else ["model", "group"]

    data = results.copy()

    baseline = data.loc[data[label_pert] == label_base].copy()
    perturbed = data.loc[data[label_pert] != label_base].copy()
    merge_keys = ["track", *pair_cols] if "track" in data.columns else list(pair_cols)

    merged = perturbed.loc[:, feat_group + pair_cols + feat_value].merge(
        baseline.loc[:, merge_keys + feat_value],
        on = merge_keys,
        suffixes = ("_pert", "_orig"),
    )

    ## count n per group for header
    if feat_group:
        n_pairs_by_group = merged.groupby(feat_group, sort = False).size()
        unique_n = np.array(pd.unique(n_pairs_by_group), dtype = float)
    else:
        unique_n = np.array([merged.shape[0]], dtype = float)
    if len(unique_n) == 0:
        n_display = 0
    elif len(unique_n) == 1:
        n_display = int(unique_n[0])
    else:
        n_display = f"{int(np.min(unique_n))}-{int(np.max(unique_n))}"

    metric_label = feat_value[0].upper() if len(feat_value) == 1 else ", ".join(v.upper() for v in feat_value)
    print(f"=== Perturbation: Original vs Perturbed Median {metric_label} (n = {n_display}) ===")
    print(f"H₁: Perturbation lowers median {metric_label}")
    print("*** p < 0.001, ** p < 0.01, * p < 0.05")

    groups = merged.groupby(feat_group, sort = False) if feat_group else [((), merged)]

    ## compute paired stats per group x metric
    rows = list()
    for group_key, grp in groups:
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        for metric in feat_value:
            x = grp[f"{metric}_orig"].to_numpy(dtype = float)
            y = grp[f"{metric}_pert"].to_numpy(dtype = float)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            n = len(x)
            d = x - y
            n_pos = int(np.sum(d > 0))
            if n:
                med_o = float(np.median(x))
                med_p = float(np.median(y))
                med_d = med_o - med_p
            else:
                med_o, med_p, med_d = np.nan, np.nan, np.nan

            n_eff = int(np.sum(d != 0))
            if n < 2 or n_eff < 2:
                w_stat, r_eff, p_val = np.nan, np.nan, np.nan
            else:
                ## one-sided paired test: significance indicates frontier collapse
                w_stat, p_val = wilcoxon(x, y, alternative = "greater")

                ## rank-biserial r (kerby 2014)
                d_nz = d[d != 0]
                ranks = rankdata(np.abs(d_nz), method = "average")
                pos_rank_sum = float(np.sum(ranks[d_nz > 0]))
                neg_rank_sum = float(np.sum(ranks[d_nz < 0]))
                r_eff = (pos_rank_sum - neg_rank_sum) / float(np.sum(ranks))

            positive_delta = float(n_pos) / float(n) if n > 0 else np.nan
            rows.append((*group_key, metric, med_o, med_p, med_d, positive_delta, w_stat, r_eff, float(p_val)))

    summary = pd.DataFrame(rows, columns = feat_group + [
        "metric",
        "Median Original",
        "Median Perturbed",
        "Median Δ ",
        "Positive Δ ",
        "Wilcoxon W+",
        "Rank-biserial r",
        p_label,
    ])

    ## holm-bonferroni correction with monotonic adjustment
    summary[p_label] = pd.to_numeric(summary[p_label], errors = "coerce")
    p_value = summary[p_label].to_numpy(dtype = float, copy = True)
    p_valid = np.isfinite(p_value)
    holm = np.full(shape = len(p_value), fill_value = np.nan, dtype = float)
    if np.any(p_valid):
        p_valid = p_value[p_valid]
        m = len(p_valid)
        order = np.argsort(p_valid)
        holm_sorted = np.maximum.accumulate(p_valid[order] * (m - np.arange(m)))
        holm_valid = np.empty(m, dtype = float)
        holm_valid[order] = np.minimum(holm_sorted, 1.0)
        holm[p_valid] = holm_valid
    summary["Holm-adj. p"] = holm
    summary["Sig"] = summary["Holm-adj. p"].map(
        lambda p: np.nan if not np.isfinite(p) else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )

    ## convert group/metric labels to display names
    summary = summary.rename(columns = {c: c.replace("_", " ").title() for c in feat_group})
    if len(feat_value) == 1:
        tag = feat_value[0].upper()
        summary = summary.rename(columns = {
            "Median Original": f"Median {tag} (Original)",
            "Median Perturbed": f"Median {tag} (Perturbed)",
            "Median Δ ": f"Median Δ  {tag}",
        }).drop(columns = ["metric"])
    else:
        summary = summary.rename(columns = {"metric": "Metric"})

    ## apply native display ordering, then format numeric output
    if "Track" in summary.columns:
        summary["Track"] = pd.Categorical(
            summary["Track"],
            categories = ["frozen", "retrain"],
            ordered = True,
        )
    if "Perturbation" in summary.columns:
        pert_order = list(pd.unique(perturbed[label_pert]))
        summary["Perturbation"] = pd.Categorical(
            summary["Perturbation"],
            categories = pert_order,
            ordered = True,
        )
    if "Method" in summary.columns:
        method_order = list(pd.unique(perturbed["method"]))
        summary["Method"] = pd.Categorical(
            summary["Method"],
            categories = method_order,
            ordered = True,
        )
    if group_display:
        summary = summary.sort_values(group_display).reset_index(drop = True)

    if "Wilcoxon W+" in summary.columns:
        summary["Wilcoxon W+"] = summary["Wilcoxon W+"].round(0).astype("Int64")
    round_cols = [c for c in summary.select_dtypes(include = [np.number]).columns if c != "Wilcoxon W+"]
    if round_cols:
        summary[round_cols] = summary[round_cols].round(decimals)

    ## keep a stable display column order
    if "Metric" in summary.columns:
        value_cols_order = ["Metric", "Median Original", "Median Perturbed", "Median Δ ", "Positive Δ ", *tail_cols]
    else:
        med_o = next((c for c in summary.columns if c.startswith("Median ") and c.endswith("(Original)")), "Median Original")
        med_p = next((c for c in summary.columns if c.startswith("Median ") and c.endswith("(Perturbed)")), "Median Perturbed")
        med_d = next((c for c in summary.columns if c.startswith("Median Δ ")), "Median Δ ")
        value_cols_order = [med_o, med_p, med_d, "Positive Δ ", *tail_cols]

    summary = summary.reindex(columns = group_display + [c for c in value_cols_order if c in summary.columns])
    summary = summary.set_index(group_display) if (index and group_display) else summary
    summary = summary.astype(object).where(pd.notna(summary), '-')
    return summary

## specify delta from baseline variability across learners
def spec_marginal_delta(
    results: pd.DataFrame,
    feat_value: Sequence[str],
    track: str | Sequence[str] | None = None,
    label_pert: str = "perturbation",
    label_base: str = "baseline",
    method: Literal["mad", "iqr", "max"] = "iqr",
    scale: float = 0.5,
    decimals: int = 2,
    ) -> float:

    """
    Desc:
        Compute a data-driven equivalence margin (delta) from the natural
        variability of baseline metric values across learners. Delta is
        anchored entirely to original data performance. It is pre-specifed
        and independent of any perturbation effect.

    Args:
        results: Full aggregated output from eval_perturb, including
            baseline rows.
        feat_value: Metric columns (e.g. ["ei"]).
        feat_pairs: Unused (kept for API compatibility).
        track: Evaluation track to restrict to.
        label_pert: Perturbation label column.
        label_base: Baseline label.
        method: Dispersion estimator ("mad" for median absolute deviation,
            "iqr" for interquartile range, "max" for range).
        scale: Multiplier applied to the dispersion estimate
            (default 0.5).
        decimals: Number of decimal places to round the result
            (default 2).

    Returns:
        Scalar equivalence margin delta.
    """

    data = results.copy()

    if track is not None and "track" in data.columns:
        track_vals = [track] if isinstance(track, str) else list(track)
        data = data.loc[data["track"].isin(track_vals)]

    baseline = data.loc[data[label_pert] == label_base]

    vals = np.concatenate([
        baseline[m].to_numpy(dtype=float) for m in feat_value
    ])
    vals = vals[np.isfinite(vals)]

    if len(vals) < 2:
        return 0.05  # fallback

    if method == "mad":
        dispersion = float(np.median(np.abs(vals - np.median(vals))))
    elif method == "iqr":
        dispersion = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    elif method == "max":
        dispersion = float(np.max(vals) - np.min(vals))
    else:
        raise ValueError(f"unknown method: {method}")

    return round(max(float(scale * dispersion), 1e-6), decimals)


## ----------------------------------------------------------------------------
## tost equivalence test for perturbation stability
## ----------------------------------------------------------------------------
def stat_perturbed_tost(
    results: pd.DataFrame,
    feat_value: Sequence[str],
    feat_pairs: Sequence[str] | None = None,
    feat_group: Sequence[str] = ["track"],
    pert_type: str | None = None,
    track: str | Sequence[str] | None = None,
    delta: float = 0.05,
    label_pert: str = "perturbation",
    label_base: str = "baseline",
    decimals: int = 4,
    index: bool = True,
    ) -> pd.DataFrame:

    """
    Desc:
        Two one-sided tests (TOST) for equivalence of baseline vs perturbed
        metrics. Rejection means the perturbation effect is negligibly small
        (within ±delta). Uses paired Wilcoxon signed-rank tests for each
        direction.

    Args:
        results: Full aggregated output from eval_perturb, including
            baseline rows.
        feat_value: Metric columns to test (e.g. ["ei"]).
        feat_pairs: Columns aligning baseline to perturbed rows.
            Defaults to ["model", "group"] to match the falsification
            pipeline's (model × domain) pairing convention.
        feat_group: Columns whose unique combinations define independent
            tests (default ["track"]).
        pert_type: Perturbation family to restrict to (e.g. "network").
        track: Evaluation track to restrict to (e.g. "frozen", "retrain").
            None -> use all tracks.
        delta: Equivalence margin on the metric scale. The null hypothesis
            is |Δ| >= delta; rejection means |Δ| < delta.
        label_pert: Perturbation label column.
        label_base: Baseline label.
        decimals: Display rounding.
        index: Whether to set group columns as index.

    Returns:
        DataFrame with columns: [*group, Median Δ, TOST p,
            Holm-adj. p, Decision].
    """

    ## filter to a single perturbation type when specified
    if pert_type is not None:
        results = results.loc[
            results[label_pert].isin([label_base, pert_type])
        ].copy()

    ## filter to specified track(s)
    if track is not None and "track" in results.columns:
        track_vals = [track] if isinstance(track, str) else list(track)
        results = results.loc[results["track"].isin(track_vals)].copy()

    feat_value = list(feat_value)
    feat_group = list(feat_group or [])
    group_display = [c.replace("_", " ").title() for c in feat_group]
    pair_cols = list(feat_pairs) if feat_pairs is not None else ["model", "group"]

    ## normalize group and pairing columns for the merge
    metric_label = (
        feat_value[0].upper()
        if len(feat_value) == 1
        else ", ".join(v.upper() for v in feat_value)
    )
    data = results.copy()
    baseline = data.loc[data[label_pert] == label_base].copy()
    perturbed = data.loc[data[label_pert] != label_base].copy()
    merge_keys = (
        ["track", *pair_cols]
        if "track" in data.columns
        else list(pair_cols)
    )

    ## pair baseline and perturbed rows by track+model (or custom pair cols)
    merged = baseline.merge(
        right = perturbed,
        on = merge_keys,
        suffixes = ("_orig", "_pert"),
        how = "inner",
    )

    ## restore group columns lost to merge suffixing
    for col in feat_group:
        if col not in merged.columns and f"{col}_pert" in merged.columns:
            merged[col] = merged[f"{col}_pert"]
    groups = (
        merged.groupby(feat_group, sort = False)
        if feat_group
        else [((), merged)]
    )

    ## compute sample size range for header display
    if feat_group:
        n_pairs_by_group = merged.groupby(feat_group, sort = False).size()
        unique_n = np.array(pd.unique(n_pairs_by_group), dtype = float)
    else:
        unique_n = np.array([merged.shape[0]], dtype = float)
    if len(unique_n) == 0:
        n_display = 0
    elif len(unique_n) == 1:
        n_display = int(unique_n[0])
    else:
        n_display = f"{int(np.min(unique_n))}-{int(np.max(unique_n))}"

    rows = list()
    for group_key, grp in groups:
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        for metric in feat_value:
            x = grp[f"{metric}_orig"].to_numpy(dtype = float)
            y = grp[f"{metric}_pert"].to_numpy(dtype = float)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            n = len(x)
            d = x - y
            med_d = float(np.median(d)) if n else np.nan

            if n < 2:
                p_upper = p_lower = p_tost = np.nan
            else:
                ## upper test: are values less than delta?
                d_upper = d - delta
                _, p_upper = wilcoxon(d_upper, alternative = "less")

                ## lower test: are values greater than -delta?
                d_lower = d + delta
                _, p_lower = wilcoxon(d_lower, alternative = "greater")

                ## tost p = worst-case one-sided p-value
                p_tost = max(p_upper, p_lower)

            ## paired rank-biserial effect size from wilcoxon
            if n >= 2:
                w_plus, _ = wilcoxon(d, alternative = "greater")
                t_sum = n * (n + 1) / 2
                r_rb = (2 * w_plus / t_sum) - 1
            else:
                r_rb = np.nan

            row = dict(zip(feat_group, group_key))
            tag = metric.upper()
            row[f"Median Δ {tag}"] = round(med_d, decimals)
            row["Rank-biserial r"] = round(r_rb, decimals) if np.isfinite(r_rb) else np.nan
            row["TOST p"] = round(p_tost, decimals) if np.isfinite(p_tost) else np.nan
            rows.append(row)

    summary = pd.DataFrame(rows)

    ## holm-bonferroni step-down adjustment
    p_value = summary["TOST p"].to_numpy(dtype = float, copy = True)
    p_valid = np.isfinite(p_value)
    holm = np.full(len(p_value), np.nan, dtype = float)
    if np.any(p_valid):
        p_valid = p_value[p_valid]
        m = len(p_valid)
        order = np.argsort(p_valid)
        holm_sorted = np.maximum.accumulate(p_valid[order] * (m - np.arange(m)))
        holm_valid = np.empty(m, dtype = float)
        holm_valid[order] = np.minimum(holm_sorted, 1.0)
        holm[np.isfinite(p_value)] = holm_valid
    summary["Holm-adj. p"] = holm
    summary["Sig."] = summary["Holm-adj. p"].map(
        lambda p: "-" if not np.isfinite(p) else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    summary["Eq."] = summary["Sig."].map(
        lambda s: "-" if s == "-" else "Yes" if s != "" else "No"
    )

    ## fixed decimal formatting for display
    num_cols = [c for c in summary.columns if c.startswith("Median") or c in ["Rank-biserial r", "TOST p", "Holm-adj. p"]]
    for col in num_cols:
        summary[col] = summary[col].apply(
            lambda v: f"{float(v):.{decimals}f}" if pd.notna(v) and np.isfinite(float(v)) else v
        )

    ## final display cleanup
    summary = summary.rename(columns = {c: c.replace("_", " ").title() for c in feat_group})
    summary = summary.astype(object).where(pd.notna(summary), "-")
    if index and group_display:
        summary = summary.set_index(group_display)

    print(f"TOST Equivalence (Wilcoxon Signed-Rank): n = {n_display}, δ = {delta}")
    print(f"H₀: |Δ {metric_label}| ≥ δ")
    print(f"H₁: |Δ {metric_label}| < δ")
    print(f"Median Δ {metric_label}: Median of paired differences, not the difference of marginal medians")
    print(f"Rank-biserial r: Paired effect size, equivalence determined by TOST")
    print(f"TOST p: max(Upper p, Lower p)")
    print(f"Holm-adj. p: Holm-Bonferroni adjusted TOST p-value")
    print("Significance codes reflect Holm-adj. p")
    print("*** p < 0.001, ** p < 0.01, * p < 0.05")

    return summary

## ----------------------------------------------------------------------------
## maximum-intensity selector for perturbation results
## ----------------------------------------------------------------------------
def find_perturbed_max(
    results: pd.DataFrame,
    intensity_col: str = "intensity",
    label_pert: str = "perturbation",
    label_base: str = "baseline",
    feat_group: Sequence[str] | None = None,
    pert_order: Sequence[str] | None = None,
    ) -> pd.DataFrame:

    """
    Desc:
        Keep baseline rows and the strongest shared perturbation
        setting for each perturbation family and method. This is a
        single-pass filter: it computes the maximum intensity per
        group directly from the data, then retains only those rows.
        group directly from the data, then retains only those rows. Perturbation
        families are filtered according to `pert_order`.
    
    Args:
        results: Full eval_perturbed output including baseline rows,
            or a per-observation delta frame (perturbed_all). If
            baseline rows are present they are preserved unchanged.
        intensity_col: Intensity column name.
        label_pert: Column that distinguishes baseline from perturbed rows.
        label_base: Value in label_pert that marks baseline rows.
        feat_group: Columns defining the shared intensity ladder
            (default ["perturbation", "method"]).
        pert_order: Allowed perturbation families. Rows whose
            perturbation label is not in this list are dropped. None
            keeps all non-baseline families.

    Returns:
        DataFrame containing baseline rows (if any) plus the strongest-
        intensity rows for each group.
    """
    
    data = results.copy()
    feat_group = list(feat_group or ["perturbation", "method"])
    
    baseline = data.loc[data[label_pert] == label_base]
    perturbed = data.loc[data[label_pert] != label_base].copy()

    if pert_order is not None:
        perturbed = perturbed.loc[
            perturbed[label_pert].isin(pert_order)
        ]

    perturbed[intensity_col] = pd.to_numeric(
        perturbed[intensity_col],
        errors = "coerce",
    )

    max_int = (
        perturbed
        .groupby(feat_group, as_index = False)[intensity_col]
        .max()
    )
    strongest = perturbed.merge(
        max_int,
        on = feat_group + [intensity_col],
        how = "inner",
    )

    return pd.concat(
        [baseline, strongest],
        ignore_index = True,
    )

# ## perturbation delta summary with metric medians only
# def stat_perturbed_delta(
#     results: pd.DataFrame,
#     metrics: Sequence[str] | None = None,
#     feat_group: Sequence[str] = ["track", "perturbation"],
#     track_order: Sequence[str] = ("baseline", "frozen", "retrain"),
#     perturb_order: Sequence[str] | None = None,
#     decimals: int = 4,
#     ) -> pd.DataFrame:

#     """
#     Desc:
#         Compute a grouped median summary of perturbation results for display.

#     Args:
#         results: Output of eval_perturb (results_data or perturbed_all).
#         metrics: Metric columns to aggregate. Defaults to Δ *
#             columns detected from the dataframe.
#         feat_group: Grouping columns for the summary
#             (default ["track", "perturbation"]).
#         track_order: Ordered categories for track column.
#         perturb_order: Ordered categories for perturbation column.
#             None -> inferred from data.
#         decimals: Number of decimal places to round.

#     Returns:
#         DataFrame: [*feat_group, *metrics] with median metric values per group.
#     """

#     feat_group = list(feat_group or ["track", "perturbation"])
#     if metrics is None:
#         rho_cols = [c for c in results.columns if c.startswith("ρ ")]
#         d_cols = [c for c in results.columns if c.startswith("Δ ")]
#         metrics = rho_cols if rho_cols else d_cols
#     metrics = list(metrics)

#     source = results.copy()
#     if "track" in source.columns and "track" in feat_group:
#         source["track"] = pd.Categorical(
#             source["track"],
#             categories = list(track_order),
#             ordered = True,
#         )
#     if "perturbation" in source.columns and "perturbation" in feat_group:
#         if perturb_order is None:
#             perturb_order = list(pd.unique(results["perturbation"]))
#         source["perturbation"] = pd.Categorical(
#             source["perturbation"],
#             categories = list(perturb_order),
#             ordered = True,
#         )

#     available = [m for m in metrics if m in source.columns]
#     summary = source.groupby(by = feat_group, observed = True)[available].median()
#     if decimals is not None:
#         summary = summary.round(decimals)

#     return summary
