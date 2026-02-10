## libraries
import numpy as np
from scipy.stats import spearmanr, kendalltau, wasserstein_distance

## violation rate
def _violation_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    """ Fraction of points that violate the frontier. """

    ## compute violation: true above frontier
    v = np.maximum(0.0, y_true - y_pred)
    
    ## violation rate is fraction with positive violation
    return float(np.mean(v > 0.0))


## mean violation magnitude
def _mean_violation(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Mean size of violations conditional on violating. """

    ## compute violation: true above frontier
    v = np.maximum(0.0, y_true - y_pred)
    
    ## restrict to positive violations only
    mask = v > 0.0
    if not np.any(mask):
        return 0.0
    return float(v[mask].mean())


## mean slack
def _mean_slack(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    """ Average slack (how far below the frontier the data lie). """

    ## compute slack: frontier above true
    s = np.maximum(0.0, y_pred - y_true)
    return float(s.mean())


## excess area
def _excess_area(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    
    """ Total slack normalized by total observed magnitude. 
    High excess area means the frontier is much higher than the data. """
    
    ## compute slack: frontier above true
    s = np.maximum(0.0, y_pred - y_true)
    denom = np.sum(y_true) + eps
    return float(np.sum(s) / denom)


## efficiency index
def _efficiency_index(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:

    """ Efficiency index that combines violation rate.
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


## combined frontier metrics
def frontier_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> dict:

    """ Compute all frontier metrics and return as a dictionary. """
    
    metrics = {
        "vr": _violation_rate(y_true, y_pred),
        "mv": _mean_violation(y_true, y_pred),
        "ms": _mean_slack(y_true, y_pred),
        "ea": _excess_area(y_true, y_pred, eps = eps),
    }
    metrics["ei"] = _efficiency_index(
        y_true = y_true,
        y_pred = y_pred,
        eps = eps,
    )
    return metrics


## spearman rank correlation
def _spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Global monotone agreement of predicted capacities. """

    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else 0.0


## kendall rank correlation
def _kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Pairwise ordering stability (probability of concordant pairs). """

    tau, _ = kendalltau(y_true, y_pred)
    return float(tau) if not np.isnan(tau) else 0.0


## rank-biased overlap
def _rank_biased_overlap(y_true: np.ndarray, y_pred: np.ndarray, p: float = 0.9) -> float:

    """ Frontier-focused agreement; emphasizes the top of the ranking. """
    
    ## sort indices descending
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


## top-k rank overlap
def _top_k_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:

    """ Discrete stability of which graphs define the frontier. """

    ## indices of top k elements
    top_a = set(np.argsort(y_true)[-k:])
    top_b = set(np.argsort(y_pred)[-k:])
    
    if not top_a or not top_b:
        return 0.0

    ## jaccard similarity
    intersect = len(top_a.intersection(top_b))
    union = len(top_a.union(top_b))
    
    return float(intersect / union)


## wasserstein distance
def _wasserstein_dist(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Distributional proximity of the predicted frontier envelopes. """

    return float(wasserstein_distance(y_true, y_pred))


## combined convergence metrics
def convergence_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10, p: float = 0.9) -> dict:

    """ Compute all convergence metrics and return as a dictionary. """

    return {
        "rho": _spearman_rho(y_true, y_pred),
        "tau": _kendall_tau(y_true, y_pred),
        "rbo": _rank_biased_overlap(y_true, y_pred, p = p),
        "jaccard": _top_k_overlap(y_true, y_pred, k = k),
        "emd": _wasserstein_dist(y_true, y_pred),
    }
