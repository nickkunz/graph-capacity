## modules
import numpy as np 
from src.evaluators.metrics import (
    monotonic_index,
    rank_consistency,
    isotonic_feasibility
)

## structural ordering evaluation
def structural_ordering(kappa, y_true, y_pred):
    results = {}

    ## handle list or array input
    y_pred = np.asarray(y_pred)

    ## determine representative frontier
    ## if y_pred is 2d (realizations x nodes), take the median for scalar metrics
    if y_pred.ndim == 2:
        y_hat = np.median(a = y_pred, axis = 0)
    else:
        y_hat = y_pred

    ## 1. monotonic structural ordering index
    results["monotonic_index"] = monotonic_index(kappa, y_hat)

    ## 2. rank stability across metrics
    if y_pred.ndim == 2 and y_pred.shape[0] > 1:
        rank_stats = rank_consistency(y_pred)
        results.update(rank_stats)

    ## 3. isotonic feasibility preservation
    iso = isotonic_feasibility(kappa, y_hat, y_true)
    results.update(iso)

    return results