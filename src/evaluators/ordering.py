## libraries
import numpy as np
import pandas as pd
from typing import Dict, Any, Sequence, Literal
from sklearn.base import clone

## modules
from src.vectorizers.scalers import _standardizer, _log_transformer
from src.evaluators.metrics import structural_ordering, compute_kappa

## ------------------------------------
## structural ordering evaluation
## ------------------------------------
def evaluate_ordering(
    data: pd.DataFrame,
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    models: Dict[str, Any],
    vect_k: Literal["feat_x", "feat_z"] = "feat_x",
    target: str = "target",
    ) -> pd.DataFrame:
    
    """
    Desc:
        Evaluates structural ordering for a suite of models on the full dataset.
    
    Args:
        data: Input DataFrame containing features and target.
        feat_x: List of column names for graph invariants.
        feat_z: List of column names for process signatures.
        models: Dictionary mapping model names to model wrappers (must have estimator_c and estimator_r).
        vect_k: Which feature vector to use for computing structural index kappa ("feat_x" or "feat_z").
        target: Name of the target column.
        
    Returns:
        DataFrame containing structural ordering metrics for each model.

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
    y_star = _log_transformer(data[target]).astype(float)
    
    ## standardize features
    X_scaled, _ = _standardizer(X, feat_x)
    Z_scaled, _ = _standardizer(Z, feat_z)
    
    ## extract values for training
    X_train = X_scaled[feat_x].values.astype(float)
    Z_train = Z_scaled[feat_z].values.astype(float)
    y_train = y_star.values.astype(float)

    ## compute structural index kappa
    ## select feature set based on vect_k argument
    if vect_k == "feat_x":
        K_train = X_train
    else:
        K_train = Z_train
    
    kappa = compute_kappa(K_vect = K_train)

    ## evaluation loop
    results_list = list()
    for name, model_wrapper in models.items():
        
        ## clone estimators to ensure fresh training
        model_c = clone(model_wrapper.estimator_c)
        model_r = clone(model_wrapper.estimator_r)
        
        ## fit C (graph invariants -> log capacity)
        model_c.fit(X_train, y_train)
        c_hat = model_c.predict(X_train).astype(float)
        
        ## fit R (process signatures -> slack)
        ## slack = true capacity - predicted structure capacity
        slack = (y_train - c_hat).astype(float)
        model_r.fit(Z_train, slack)
        r_hat = model_r.predict(Z_train).astype(float)
        
        ## final prediction: y = C(x) + R(z)
        y_pred = c_hat + r_hat

        ## evaluate structural ordering
        ordering_metrics = structural_ordering(
            kappa = kappa,
            y_pred = y_pred
        )
        
        ## store results
        ordering_metrics["model"] = name
        results_list.append(ordering_metrics)

    ## convert to dataframe
    results_data = pd.DataFrame(results_list)
    
    ## reorder columns to having model first
    cols = ["model"] + [c for c in results_data.columns if c != "model"]
    return results_data[cols]