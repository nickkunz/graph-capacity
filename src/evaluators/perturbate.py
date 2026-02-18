## libraries
import igraph as ig
import numpy as np
import pandas as pd
from typing import Sequence, Optional, Union, Tuple, List
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample

## modules
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.vectorizers.scalers import _log_transformer, _standardizer
from src.evaluators.metrics import frontier_metrics


## --------------------
## network perturbation
## --------------------
def network_perturb(
    graph: ig.Graph,
    method: str = "rewire",
    intensity: float = 0.1,
    n_swaps: Optional[int] = None,
    ) -> dict:

    """
    Desc:
        Modifies graph topology G and recomputes graph invariants.
    
    Methods:
        - 'rewire': Degree-preserving rewiring.
        - 'sparsify': Randomly remove edges (sparsification).
        - 'densify': Randomly add edges (densification).
    
    Args:
        graph: Original igraph.Graph object.
        method: Perturbation method name.
        intensity: Fraction of edges to modify (for sparsify/densify).
        n_swaps: Number of rewiring swaps (for rewire). If None, defaults to |E| * intensity.
    
    Returns:
        dict: Recomputed graph invariants.
    
    Raises:
        ValueError: If method is unknown or if intensity is out of range.
    """
    
    ## init graph
    G = graph.copy()
    n_edges = G.ecount()
    n_nodes = G.vcount()

    ## degree-preserving rewiring
    if method == "rewire":
        if n_swaps is None:
            n_swaps = int(n_edges * intensity)
        # Simplified call for compatibility
        G.rewire(n=n_swaps, mode="simple")
    
    ## sparsification (remove edges)
    elif method == "sparsify":
        n_remove = int(n_edges * intensity)
        if n_remove > 0:
            edges_to_remove = np.random.choice(n_edges, n_remove, replace=False)
            G.delete_edges(edges_to_remove)
            
    ## densification (add edges)
    elif method == "densify":
        ## this is inefficient for dense graphs but fine for sparse
        n_add = int(n_edges * intensity)
        if n_add > 0:
            import itertools
            existing = set(tuple(sorted(e.tuple)) for e in G.es)
            added = 0
            while added < n_add:
                u, v = np.random.randint(0, n_nodes, 2)
                if u != v:
                    edge = tuple(sorted((u, v)))
                    if edge not in existing:
                        G.add_edge(u, v)
                        existing.add(edge)
                        added += 1

    else:
        raise ValueError(f"Unknown structural perturbation method: {method}")

    ## recompute invariants
    return GraphInvariants(G).all()


## ----------------------
## invariant perturbation
## ----------------------
def invariant_perturb(
    X: pd.DataFrame,
    method: str = "noise",
    noise_level: float = 0.05,
    subset_fraction: float = 0.8,
) -> pd.DataFrame:

    """
    Modifies invariant encoding x directly.
    
    Methods:
        - 'noise': Add Gaussian noise to features.
        - 'subset': Randomly zero out a subset of features (ablation).
        - 'jitter': Multiplicative jitter.
        
    Args:
        X: Dataframe of graph invariants.
        method: Perturbation method.
        noise_level: Standard deviation of noise (relative to feature std).
        subset_fraction: Fraction of features to keep (for subset).
        
    Returns:
        pd.DataFrame: Perturbed feature matrix.
    """
    
    X_new = X.copy()
    
    if method == "noise":
        ## additive gaussian noise scaled by feature std
        for col in X_new.columns:
            std = X_new[col].std()
            if std > 0:
                noise = np.random.normal(0, std * noise_level, size=len(X_new))
                X_new[col] += noise

    elif method == "jitter":
        ## multiplicative jitter (1 + noise)
        noise = np.random.normal(0, noise_level, size=X_new.shape)
        X_new *= (1 + noise)
        ## clip only inherently non-negative invariants
        for col in X_new.columns:
            if (X[col] >= 0).all():
                X_new[col] = np.clip(X_new[col], a_min = 0, a_max = None)
        
    elif method == "subset":
        ## random feature ablation (permute values to destroy structure)
        n_features = X_new.shape[1]
        n_keep = int(n_features * subset_fraction)
        drop_indices = np.random.choice(
            X_new.columns, 
            size = n_features - n_keep, 
            replace = False
        )
        for col in drop_indices:
            X_new[col] = np.random.permutation(X_new[col].values)

    else:
        raise ValueError(f"Unknown representation perturbation method: {method}")
        
    return X_new


## --------------------
## process perturbation
## --------------------
def process_perturb(
    counts: np.ndarray,
    method: str = "scaling",
    param: float = 1.0,
) -> dict:

    """
    Modifies process signatures S by perturbing the underlying count process.
    
    Methods:
        - 'scaling': Intensity scaling (multiply counts by param).
        - 'smoothing': Rolling average smoothing.
        - 'resample': Bootstrap resample the counts sequence (destroys temporal order).
        - 'burst_smoothing': Apply log transformation to dampen bursts.
        
    Args:
        counts: Array of event counts (aggregated time series).
        method: Perturbation method.
        param: Parameter for the method (scale factor, window size, etc).
        
    Returns:
        dict: Recomputed process signatures.
    """
    
    S_new = counts.copy().astype(float)
    
    if method == "scaling":
        ## intensity scaling with optional mean-normalization for comparability
        S_new = S_new * float(param)
        mean_val = float(np.mean(S_new))
        if mean_val > 0:
            S_new = S_new / mean_val

    elif method == "smoothing":
        window = int(max(1, param))
        if window > 1:
            S_new = pd.Series(S_new).rolling(window=window, min_periods=1).mean().values
            
    elif method == "burst_smoothing":
        ## log-dampening: log(1 + x)
        S_new = np.log1p(S_new)
        
    elif method == "resample":
        ## destroy temporal structure (test for memory)
        S_new = np.random.choice(S_new, size=len(S_new), replace=True)
        
    else:
        raise ValueError(f"Unknown process perturbation method: {method}")

    ## recompute signatures
    ## we need a dummy dataframe structure for ProcessSignatures
    df_temp = pd.DataFrame({"counts": S_new, "idx": range(len(S_new))})
    sigs = ProcessSignatures(df_temp, sort_by=["idx"], target="counts")
    
    ## calculate all available signatures
    return sigs.all()


## ----------------------
## signature perturbation
## ----------------------
def signature_perturb(
    X: pd.DataFrame,
    Z: pd.DataFrame,
    y: pd.Series,
    method: str = "bootstrap",
    fraction: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

    """
    Modifies raw realizations (observations) of the (x, S, y) tuples.
    
    Methods:
        - 'bootstrap': Bootstrap resampling with replacement.
        - 'subsample': Random subsampling without replacement.
        - 'outlier_removal': Remove points with high residuals (needs pre-fit).
    
    Returns:
        Tuple of (X_new, Z_new, y_new)
    """
    
    n_samples = len(y)
    indices = np.arange(n_samples)
    
    if method == "bootstrap":
        ## bootstrap keeps original sample size
        new_indices = resample(indices, n_samples = n_samples, replace = True)

    elif method == "subsample":
        ## subsample uses fraction
        new_n = int(max(1, np.floor(n_samples * fraction)))
        new_indices = resample(indices, n_samples = new_n, replace = False)

    else:
        new_indices = indices

    return X.iloc[new_indices], Z.iloc[new_indices], y.iloc[new_indices]


## --------------------
## temporal perturbation
## --------------------
def temporal_perturb(
    event_times: Sequence[float],
    scale: str = "1D",
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
) -> Tuple[float, dict]:

    """
    Modifies aggregation target temporal resolution.
    Recomputes y(Delta_t) and S(Delta_t).
    
    Args:
        event_times: List/Array of raw event timestamps.
        scale: Pandas offset alias (e.g., '1D', '1H', '15min').
        start_time: Start of observation window.
        end_time: End of observation window.
        
    Returns:
        Tuple: (max_rate_y, signatures_S)
    """
    
    if len(event_times) == 0:
            return 0.0, {}
            
    ## convert to series
    ts = pd.to_datetime(event_times)
    if start_time is None: start_time = ts.min()
    if end_time is None: end_time = ts.max()
    
    ## create full range index to ensure zero counts are preserved
    full_range = pd.date_range(start=start_time, end=end_time, freq=scale)
    
    ## aggregate counts correctly using resample on timestamp index
    df = pd.DataFrame({"t": ts})
    df = df.set_index("t")
    counts_binned = (
        df
        .assign(count = 1)
        .resample(scale)
        .sum()
        .reindex(full_range, fill_value = 0)
    )["count"]
    
    ## compute y (max rate) normalized by duration
    y_count = float(counts_binned.max())
    
    if len(counts_binned) >= 2:
        duration = float((counts_binned.index[1] - counts_binned.index[0]).total_seconds())
    else:
        duration = float(pd.Timedelta(scale).total_seconds())

    duration = max(duration, 1.0)
    y_val = y_count / duration
    
    ## recompute signatures
    df_temp = pd.DataFrame({"counts": counts_binned.values, "idx": range(len(counts_binned))})
    sigs = ProcessSignatures(df_temp, sort_by=["idx"], target="counts")
    
    ## calculate all available signatures
    return y_val, sigs.all()