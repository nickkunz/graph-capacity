## libraries
import pandas as pd
from typing import Dict, Sequence, Literal
from sklearn.base import BaseEstimator

## modules
from src.vectorizers.scalers import _standardizer
from src.evaluators.metrics import structural_ordering, compute_kappa
from src.evaluators.training import fit_predict_frontier

## ------------------------------------
## structural ordering evaluation
## ------------------------------------
def eval_order(
    data: pd.DataFrame,
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    estimator_c: BaseEstimator,
    estimator_r: BaseEstimator,
    vect_k: Literal["feat_x", "feat_z"] = "feat_x",
    target: str = "target",
    ) -> Dict[str, float]:
    
    """
    Desc:
        Evaluates structural ordering for a single model on the full dataset.
    
    Args:
        data: Input DataFrame containing features and target.
        feat_x: List of column names for graph invariants.
        feat_z: List of column names for process signatures.
        estimator_c: Estimator for Capacity Potential (C).
        estimator_r: Estimator for Slack/Efficiency (R).
        vect_k: Which feature vector to use for computing structural index kappa ("feat_x" or "feat_z").
        target: Name of the target column.
        
    Returns:
        Dictionary containing structural ordering metrics for the model.

    Raises:
        ValueError: If feat_x or feat_z is empty, or if vect_k is not "feat_x" or "feat_z".
    """

    ## validate inputs
    if not feat_x:
        raise ValueError("feat_x must contain at least one column name.")
    if not feat_z:
        raise ValueError("feat_z must contain at least one column name.")
    if vect_k not in ["feat_x", "feat_z"]:
        raise ValueError("vect_k must be either 'feat_x' or 'feat_z'.")

    ## init variables
    X = data[feat_x].apply(pd.to_numeric, errors = "coerce")
    Z = data[feat_z].apply(pd.to_numeric, errors = "coerce")
    
    ## standardize features
    X_scaled, _ = _standardizer(X, feat_x)
    Z_scaled, _ = _standardizer(Z, feat_z)
    
    ## extract values for training
    X_train = X_scaled[feat_x].values.astype(float)
    Z_train = Z_scaled[feat_z].values.astype(float)

    ## select feature set based on vect_k argument
    if vect_k == "feat_x":
        K_train = X_train
    else:
        K_train = Z_train

    ## compute structural index kappa
    kappa = compute_kappa(K_vect = K_train)

    ## prediction surface
    y_pred, _, _ = fit_predict_frontier(
        data = data,
        feat_x = feat_x,
        feat_z = feat_z,
        estimator_c = estimator_c,
        estimator_r = estimator_r,
        target = target
    )

    ## evaluate structural order
    return structural_ordering(
        kappa = kappa,
        y_pred = y_pred
    )
