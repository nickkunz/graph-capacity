## libraries
import dcor
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.decomposition import PCA

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


## pearson correlation
def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Linear correlation of predicted capacities with true capacities. """

    r, _ = pearsonr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0

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


## distance correlation
def _distance_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Non-linear dependence measured by distance correlation. """

    try:
        return float(dcor.distance_correlation(y_true, y_pred))
    except Exception:
        return 0.0


## rank-biased overlap
def _rank_biased_overlap(y_true: np.ndarray, y_pred: np.ndarray, p: float = 0.5) -> float:

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


## combined convergence metrics
def convergence_metrics(y_true: np.ndarray, y_pred: np.ndarray, p: float = 0.9) -> dict:

    """ Compute all convergence metrics and return as a dictionary. """

    return {
        "r": _pearson_r(y_true, y_pred),
        "rho": _spearman_rho(y_true, y_pred),
        "tau": _kendall_tau(y_true, y_pred),
        "dcr": _distance_corr(y_true, y_pred),
        "rbo": _rank_biased_overlap(y_true, y_pred, p = p)
    }


## compute structural index via pca
def compute_kappa(X_scaled, y_pred = None):

    """ Compute the structural index kappa via PCA on the standardized graph invariants.
    If y_pred is provided, the sign of kappa is adjusted to match the correlation. """

    pca = PCA(n_components = 1)
    kappa = pca.fit_transform(X_scaled).flatten()

    ## safely fix sign indeterminacy
    if y_pred is not None:
        y_pred = np.asarray(y_pred)
        if np.std(kappa) > 0 and np.std(y_pred) > 0:
            corr = np.corrcoef(kappa, y_pred)[0,1]
            if not np.isnan(corr) and corr < 0:
                kappa = -kappa

    return kappa


## monotonic index for joint frontier
def monotonic_index(kappa, y_pred):

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

    return agree / total if total > 0 else np.nan

## rank consistency across attainable frontier
def rank_consistency(y_pred):

    """ Rank consistency metrics that evaluate the stability of the ordering 
    across different points on the attainable frontier. Higher is better. """

    ## force array and check dimensions
    ## expects shape (n_bootstraps, n_samples)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim < 2 or y_pred.shape[0] < 2:
        return {
            "kendall_tau_mean" : np.nan,
            "spearman_rho_mean" : np.nan,
            "rbo_mean" : np.nan
        }

    taus = list()
    rhos = list()
    rbos = list()

    ## define reference frontier (e.g., median or first estimate)
    ref = y_pred[0]

    ## compare all other realization to the reference
    for y in y_pred[1:]:
        tau, _ = kendalltau(ref, y)
        rho, _ = spearmanr(ref, y)
        rbo = _rank_biased_overlap(ref, y)

        taus.append(tau)
        rhos.append(rho)
        rbos.append(rbo)

    return {
        "kendall_tau_mean" : np.mean(taus),
        "spearman_rho_mean" : np.mean(rhos),
        "rbo_mean" : np.mean(rbos)
    }

## isotonic feasibility consistency (joint frontier)
def isotonic_feasibility(kappa, y_pred, y_true):

    """ Isotonic feasibility consistency that evaluates the agreement of the 
    predicted frontier with the structural ordering. Higher is better. """

    kappa = np.asarray(kappa)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    ## feasible region
    mask = y_true <= y_pred

    k_f = kappa[mask]
    f_f = y_pred[mask]

    n = len(k_f)

    if n < 2:
        return {
            "isotonic_feasibility_score" : np.nan,
            "feasible_pairs" : 0
        }

    total = 0
    agree = 0

    for i, j in combinations(range(n), 2):

        ## skip structural ties
        if k_f[i] == k_f[j]:
            continue

        total += 1

        ## check ordering agreement
        if (k_f[i] - k_f[j]) * (f_f[i] - f_f[j]) >= 0:
            agree += 1

    score = agree / total if total > 0 else np.nan

    return {
        "isotonic_feasibility_score" : score,
        "feasible_pairs" : total
    }