## libraries
import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    max_error,
    mean_absolute_error,
    median_absolute_error
)   

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


## standard regression metrics
def central_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    
    """ Compute standard pointwise regression metrics of predicted values."""

    return {
        "mse": mean_squared_error(y_true = y_true, y_pred = y_pred),
        "mae": mean_absolute_error(y_true = y_true, y_pred = y_pred),
        "medae": median_absolute_error(y_true = y_true, y_pred = y_pred),
        "mxe": max_error(y_true = y_true, y_pred = y_pred),
    }
