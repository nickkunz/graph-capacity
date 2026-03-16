## libraries
import numpy as np
import pandas as pd
from sklearn.base import clone

## modules
from src.vectorizers.scalers import _log_transformer, _standardizer

## ------------------------
## fit and predict frontier
## ------------------------
def fit_predict_frontier(
    data: pd.DataFrame,
    feat_x: list[str],
    feat_z: list[str],
    estimator_c,
    estimator_r,
    target: str = "target",
    n_repeat: int = 1,
    random_state: int = 42
	) -> tuple[np.ndarray, object, object]:

    """
    Desc:
        Fits the frontier model and returns predictions. The frontier model decomposes 
        the log-transformed target variable into a sum of two components:
        - C: the contribution from graph invariants (structural features)
        - R: the contribution from process signatures (temporal features)
        
        The model is trained in two stages:
        1. Fit C using the graph invariants to predict the log-transformed target.
        2. Fit R using the process signatures to predict the residuals from the first stage.
        
        An identifiability constraint is applied to R to ensure it has zero mean, 
        allowing for a meaningful decomposition of the target variable. The final 
        prediction is the sum of the predicted C and R components, representing 
        the estimated log target values based on the frontier model.
    
    Args:
        data: A pandas DataFrame containing the features and target variable.
        feat_x: A list of column names in `data` to be used as graph invariants (X).
        feat_z: A list of column names in `data` to be used as process signatures (Z).
        estimator_c: A scikit-learn compatible estimator for modeling C.
        estimator_r: A scikit-learn compatible estimator for modeling R.
        target: The name of the target variable column in `data` (default is "target").
        random_state: Optional random state forwarded to the cloned estimators (default is None).
    
    Returns:
        y_pred: A numpy array of predicted log target values based on the frontier model.
        model_c: The fitted estimator for modeling C.
        model_r: The fitted estimator for modeling R.

    Raises:
        ValueError: If the specified target column is not in the DataFrame.
        ValueError: If any of the specified feature columns are not in the DataFrame.
    """

    ## init variables
    X = data[feat_x].apply(pd.to_numeric, errors = "coerce")
    Z = data[feat_z].apply(pd.to_numeric, errors = "coerce")
    y_star = _log_transformer(data[target]).astype(float)

    ## standardize graph invariants
    X_scaled, _ = _standardizer(X, feat_x)
    X_scaled = X_scaled[feat_x].values.astype(float)
    
	## standardize process signatures
    Z_scaled, _ = _standardizer(Z, feat_z)
    Z_scaled = Z_scaled[feat_z].values.astype(float)

    ## repeat training several times and average predictions
    if n_repeat < 1:
        raise ValueError("n_repeat must be >= 1")

    y_pred_sum = np.zeros_like(y_star, dtype = float)
    last_model_c = None
    last_model_r = None

    for i in range(n_repeat):
        ## ensure fresh parameters for each repeat
        model_c = clone(estimator_c)
        model_r = clone(estimator_r)

        ## set random_state on estimators if supported
        seed = None if random_state is None else int(random_state) + i
        if seed is not None:
            for m in (model_c, model_r):
                if hasattr(m, "random_state"):
                    m.set_params(random_state = seed)

        ## train C on graph invariants
        model_c.fit(X_scaled, y_star)
        c_hat = model_c.predict(X_scaled).astype(float)

        ## train R on process signatures and residualized target
        slack = (y_star - c_hat).astype(float)
        model_r.fit(Z_scaled, slack)
        r_hat = model_r.predict(Z_scaled).astype(float)

        ## identifiability constraint: zero-mean log-factor
        r_hat = (r_hat - np.mean(r_hat)).astype(float)

        ## accumulate prediction
        y_pred_sum += (c_hat + r_hat).astype(float)
        last_model_c = model_c
        last_model_r = model_r

    y_pred = (y_pred_sum / float(n_repeat)).astype(float)
    return y_pred, last_model_c, last_model_r
