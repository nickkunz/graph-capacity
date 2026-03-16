## libraries
import dcor
import warnings
import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr, kendalltau, pearsonr, ConstantInputWarning

## constants
FRONTIER_METRICS = ["vr", "mv", "ms", "ea", "ei"]
CONSENSUS_METRICS = ["r", "rho", "tau", "dcr", "rbo"]
ORDERING_METRICS = [
    "monotonic_index", 
    "violation_magnitude", 
    "spearman_rho", 
    "kendall_tau", 
    "rank_r2"
]

## violation rate
def _violation_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ VR fraction of points that violate the frontier. """

    ## compute violation: true above frontier
    v = np.maximum(0.0, y_true - y_pred)
    
    ## violation rate is fraction with positive violation
    return float(np.mean(v > 0.0))

## mean violation magnitude
def _mean_violation(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ MV mean size of violations conditional on violating. """

    ## compute violation: true above frontier
    v = np.maximum(0.0, y_true - y_pred)
    
    ## restrict to positive violations only
    mask = v > 0.0
    if not np.any(mask):
        return 0.0
    return float(v[mask].mean())

## mean slack
def _mean_slack(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    """ MS average slack (how far below the frontier the data lie). """

    ## compute slack: frontier above true
    s = np.maximum(0.0, y_pred - y_true)
    return float(s.mean())

## excess area
def _excess_area(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    
    """ EA total slack normalized by total observed magnitude. 
    High excess area means the frontier is much higher than the data. """
    
    ## compute slack: frontier above true
    s = np.maximum(0.0, y_pred - y_true)
    denom = np.sum(y_true) + eps
    return float(np.sum(s) / denom)

## efficiency index
def _efficiency_index(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:

    """ EI efficiency index that combines violation rate.
    Mean violation, and excess area. Higher is better. """

    ## reuse helper metrics to ensure a single definition of each quantity
    vr = _violation_rate(y_true = y_true, y_pred = y_pred)
    mv = _mean_violation(y_true = y_true, y_pred = y_pred)
    ea = _excess_area(y_true = y_true, y_pred = y_pred, eps = eps)

    ## transform to [0, 1] range with smooth saturation
    minus_vr = np.clip(1.0 - vr, eps, 1.0)
    ea_score = 1.0 / (1.0 + ea)
    mv_score = 1.0 / (1.0 + mv + eps)

    ## geometric mean via log-space (numerically stable)
    log_ei = (np.log(minus_vr) + np.log(ea_score) + np.log(mv_score)) / 3.0
    return float(np.exp(log_ei))

## joint frontier metrics
def frontier_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> dict:

    """ Compute all frontier metrics and return as a dictionary. """
    
    metrics = {
        key: func(y_true, y_pred, eps=eps) if key in {"ea", "ei"} else func(y_true, y_pred)
        for key, func in [
            ("vr", _violation_rate),
            ("mv", _mean_violation),
            ("ms", _mean_slack),
            ("ea", _excess_area),
        ]
    }
    metrics["ei"] = _efficiency_index(
        y_true = y_true,
        y_pred = y_pred,
        eps = eps
    )
    return metrics

## pearson correlation
def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Linear correlation of predicted capacities with true capacities. """

    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)
    if y_true.size < 2 or y_pred.size < 2:
        return 0.0
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = ConstantInputWarning)
        r, _ = pearsonr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0

## spearman rank correlation
def _spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Global monotone agreement of predicted capacities. """

    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)
    if y_true.size < 2 or y_pred.size < 2:
        return 0.0
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = ConstantInputWarning)
        rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else 0.0


## kendall rank correlation
def _kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Pairwise ordering stability (probability of concordant pairs). """

    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)
    if y_true.size < 2 or y_pred.size < 2:
        return 0.0
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = ConstantInputWarning)
        tau, _ = kendalltau(y_true, y_pred)
    return float(tau) if not np.isnan(tau) else 0.0


## distance correlation
def _distance_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Non-linear dependence measured by distance correlation. """

    try:
        return float(dcor.distance_correlation(y_true, y_pred))
    except Exception:
        return 0.0

## rank-biased overlap
def _rank_biased_overlap(y_true: np.ndarray, y_pred: np.ndarray, p: float) -> float:

    """ Frontier-focused agreement; emphasizes the top of the ranking. """
    
    s = np.argsort(y_true)[::-1]
    t = np.argsort(y_pred)[::-1]
    n = len(s)
    
    score = 0.0
    weight = 1.0
    overlap = 0
    seen_s = set()
    seen_t = set()
    
    for d in range(1, n + 1):
        idx_s = s[d-1]
        idx_t = t[d-1]
        
        if idx_s == idx_t:
            overlap += 1
        else:
            if idx_s in seen_t:
                overlap += 1
            if idx_t in seen_s:
                overlap += 1
        
        seen_s.add(idx_s)
        seen_t.add(idx_t)
        
        score += weight * (overlap / d)
        weight *= p
        
        if weight < 1e-6:
            break
            
    return float(score * (1.0 - p))

## joint consensus metrics
def consensus_metrics(y_true: np.ndarray, y_pred: np.ndarray, p: float = 0.9) -> dict:

    """ Compute all consensus metrics and return as a dictionary. """

    return {
        "r": _pearson_r(y_true, y_pred),
        "rho": _spearman_rho(y_true, y_pred),
        "tau": _kendall_tau(y_true, y_pred),
        "dcr": _distance_corr(y_true, y_pred),
        "rbo": _rank_biased_overlap(y_true, y_pred, p = p)
    }

## compute structural index via pca
def compute_kappa(K_vect: np.ndarray, y_pred: np.ndarray | None = None) -> np.ndarray:

    """ Compute the structural index kappa via PCA on the standardized graph invariants.
    If y_pred is provided, the sign of kappa is adjusted to match the correlation. """

    pca = PCA(n_components = 1)
    kappa = pca.fit_transform(K_vect).flatten()

    ## safely fix sign indeterminacy
    if y_pred is not None:
        y_pred = np.asarray(y_pred)
        if np.std(kappa) > 0 and np.std(y_pred) > 0:
            corr = np.corrcoef(kappa, y_pred)[0,1]
            if not np.isnan(corr) and corr < 0:
                kappa = -kappa

    return kappa

## monotonic index for joint frontier
def monotonic_index(kappa: np.ndarray, y_pred: np.ndarray) -> float:

    """ Monotonicity index that evaluates the agreement of the predicted frontier 
    with the structural ordering. Higher is better. """

    kappa = np.asarray(kappa)
    y_pred = np.asarray(y_pred)

    n = len(kappa)
    if n < 2:
        return np.nan

    total = 0
    agree = 0
    for i, j in combinations(range(n), 2):
        if kappa[i] == kappa[j]:
            continue
        total += 1
        if (kappa[i] - kappa[j]) * (y_pred[i] - y_pred[j]) >= 0:
            agree += 1

    return float(agree / total) if total > 0 else np.nan

## structural violation magnitude
def violation_magnitude(kappa: np.ndarray, y_pred: np.ndarray) -> float:

    """ Structural violation magnitude that evaluates the degree of violation 
    of the predicted frontier with the structural ordering. Lower is better. """
    
    kappa = np.asarray(kappa)
    y_pred = np.asarray(y_pred)

    n = len(kappa)
    if n < 2:
        return np.nan

    total = 0
    violation = 0.0
    for i, j in combinations(range(n), 2):
        dk = kappa[i] - kappa[j]
        dy = y_pred[i] - y_pred[j]
        total += abs(dk * dy)
        if dk * dy < 0:
            violation += abs(dk * dy)

    return float(violation / total) if total > 0 else np.nan

## structural association 
def structural_association(kappa: np.ndarray, y_pred: np.ndarray) -> dict:

    """ Structural association metrics between structural index and frontier.
    Returns Spearman (monotonicity), Kendall (ordering), and Rank R^2 (strength). """

    kappa = np.asarray(kappa)
    y_pred = np.asarray(y_pred)

    if len(kappa) < 2:
        return {
            "spearman_rho": np.nan,
            "kendall_tau": np.nan,
            "rank_r2": np.nan
        }

    rho, _ = spearmanr(kappa, y_pred)
    tau, _ = kendalltau(kappa, y_pred)
    
    ## rank r2 is effectively rho^2
    r2 = rho**2 if not np.isnan(rho) else np.nan

    return {
        "spearman_rho": float(rho) if not np.isnan(rho) else np.nan,
        "kendall_tau": float(tau) if not np.isnan(tau) else np.nan,
        "rank_r2": float(r2) if not np.isnan(r2) else np.nan
    }

## joint structural ordering metrics
def structural_ordering(kappa: np.ndarray, y_pred: np.ndarray) -> dict:

    """ Compute all structural ordering metrics and return as a dictionary. """

    ## handle list or array input
    kappa = np.asarray(kappa)
    y_pred = np.asarray(y_pred)

    ## compute base metrics
    results = {
        "monotonic_index": monotonic_index(kappa, y_pred),
        "violation_magnitude": violation_magnitude(kappa, y_pred)
    }

    ## add association metrics (spearman, kendall, rank_r2)
    results.update(structural_association(kappa, y_pred))

    return results
