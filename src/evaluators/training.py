## libraries
import numpy as np
import pandas as pd
from typing import Any, Mapping, Sequence
from sklearn.base import clone

## modules
from src.vectorizers.scalers import (
    _log_transformer,
    _standardizer
)

## ----------------------------------------------------------------------------
## fit and predict frontier
## ----------------------------------------------------------------------------
def fit_predict_frontier(
    data: pd.DataFrame,
    feat_x: Sequence[str] | None = None,
    feat_z: Sequence[str] | None = None,
    estimator_c: Any | None = None,
    estimator_r: Any | None = None,
    target: str = "target",
    n_repeat: int = 30,
    random_state: int = 42,
    fit_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:

    """
    Desc:
        Fits repeated full-data frontier models and returns the averaged
        predictions together with repeat-level predictions, fitted models,
        and preprocessing state.
        When `fit_result` is provided, reuses that fitted bundle to score a
        new dataframe without retraining.

        The frontier model decomposes the log-transformed target variable into
        a sum of two components:
        - C: the contribution from graph invariants (structural features)
        - R: the contribution from process signatures (temporal features)
        
        The model is trained in two stages:
        1. Fit C using the graph invariants to predict the log-transformed 
            target.
        2. Fit R using the process signatures to predict the residuals from the 
            first stage.
        
        An identifiability constraint is applied to R to ensure it has zero mean,
        allowing for a meaningful decomposition of the target variable. The final 
        prediction is the sum of the predicted C and R components, representing 
        the estimated log target values based on the frontier model.

    Args:
        data: A pandas DataFrame containing the features and target variable.
        feat_x: A list of graph invariant columns used as X.
        feat_z: A list of process signature columns used as Z.
        estimator_c: A scikit-learn compatible estimator for modeling C.
        estimator_r: A scikit-learn compatible estimator for modeling R.
        target: The name of the target variable column in `data`.
        n_repeat: Number of repeated fits to average.
        random_state: Optional random state forwarded to the cloned estimators.
        fit_result: Optional fitted bundle used for frozen transfer.

    Returns:
        Dictionary containing averaged predictions and repeat-level
        predictions. The reusable fitted state needed for frozen transfer is
        returned under `fit_result`.

    Raises:
        ValueError: If n_repeat is less than 1.
        ValueError: If fewer than two valid rows remain after coercion.
        ValueError: If transfer mode is requested without a non-empty bundle.
    """

    if fit_result is not None:
        bundle = dict(fit_result.get("fit_result", fit_result))
        feat_x = list(bundle["feat_x"])
        feat_z = list(bundle["feat_z"])
        models_c = list(bundle["models_c"])
        models_r = list(bundle["models_r"])
        r_train_means = list(bundle["r_train_means"])
        if not models_c or not models_r:
            raise ValueError("fit_result must contain at least one fitted model")

        X = data[feat_x].apply(pd.to_numeric, errors = "coerce")
        Z = data[feat_z].apply(pd.to_numeric, errors = "coerce")
        valid = (
            X[feat_x].notna().all(axis = 1)
            & Z[feat_z].notna().all(axis = 1)
        ).to_numpy()
        y_pred_repeats = np.full(
            shape = (len(models_c), len(data)),
            fill_value = np.nan,
            dtype = float,
        )
        if int(np.sum(valid)) > 0:
            X_scaled = bundle["x_scaler"].transform(
                X.loc[valid, feat_x].astype(float)
            )
            Z_scaled = bundle["z_scaler"].transform(
                Z.loc[valid, feat_z].astype(float)
            )
            for idx, (model_c, model_r, r_mean) in enumerate(
                zip(models_c, models_r, r_train_means)
            ):
                c_hat = model_c.predict(X_scaled).astype(float)
                r_hat = model_r.predict(Z_scaled).astype(float)
                y_pred_repeats[idx, valid] = (
                    c_hat + (r_hat - float(r_mean))
                ).astype(float)

        y_pred = np.full(shape = len(data), fill_value = np.nan, dtype = float)
        valid_cols = np.any(np.isfinite(y_pred_repeats), axis = 0)
        if np.any(valid_cols):
            y_pred[valid_cols] = np.nanmean(y_pred_repeats[:, valid_cols], axis = 0)
        return {
            "y_pred": y_pred,
            "y_pred_repeats": y_pred_repeats,
            "fit_result": bundle,
        }

    if n_repeat < 1:
        raise ValueError("n_repeat must be >= 1")
    if feat_x is None or feat_z is None:
        raise ValueError("feat_x and feat_z are required when fit_result is not provided")
    if estimator_c is None or estimator_r is None:
        raise ValueError("estimator_c and estimator_r are required when fit_result is not provided")

    feat_x = list(feat_x)
    feat_z = list(feat_z)
    X = data[feat_x].apply(pd.to_numeric, errors = "coerce")
    Z = data[feat_z].apply(pd.to_numeric, errors = "coerce")
    y_star = _log_transformer(data[target]).astype(float)

    valid = (
        X[feat_x].notna().all(axis = 1)
        & Z[feat_z].notna().all(axis = 1)
        & y_star.notna()
    ).to_numpy()
    if int(np.sum(valid)) < 2:
        raise ValueError("frontier fit requires at least two valid rows")

    X_valid = X.loc[valid, feat_x]
    Z_valid = Z.loc[valid, feat_z]
    y_valid = y_star.loc[valid].to_numpy(dtype = float)

    X_scaled_df, x_scaler = _standardizer(X_valid, feat_x)
    X_scaled = X_scaled_df[feat_x].values.astype(float)

    ## standardize process signatures
    Z_scaled_df, z_scaler = _standardizer(Z_valid, feat_z)
    Z_scaled = Z_scaled_df[feat_z].values.astype(float)

    y_pred_repeats = np.full(
        shape = (n_repeat, len(data)),
        fill_value = np.nan,
        dtype = float,
    )
    models_c = []
    models_r = []
    r_train_means = []

    for i in range(n_repeat):
        model_c = clone(estimator_c)
        model_r = clone(estimator_r)

        seed = None if random_state is None else int(random_state) + i
        if seed is not None:
            for m in (model_c, model_r):
                if hasattr(m, "random_state"):
                    m.set_params(random_state = seed)

        model_c.fit(X_scaled, y_valid)
        c_hat = model_c.predict(X_scaled).astype(float)

        slack = (y_valid - c_hat).astype(float)
        model_r.fit(Z_scaled, slack)
        r_hat = model_r.predict(Z_scaled).astype(float)

        r_mean = float(np.mean(r_hat))
        y_pred_repeats[i, valid] = (c_hat + (r_hat - r_mean)).astype(float)

        models_c.append(model_c)
        models_r.append(model_r)
        r_train_means.append(r_mean)

    y_pred = np.full(shape = len(data), fill_value = np.nan, dtype = float)
    valid_cols = np.any(np.isfinite(y_pred_repeats), axis = 0)
    if np.any(valid_cols):
        y_pred[valid_cols] = np.nanmean(y_pred_repeats[:, valid_cols], axis = 0)

    return {
        "y_pred": y_pred,
        "y_pred_repeats": y_pred_repeats,
        "fit_result": {
            "models_c": models_c,
            "models_r": models_r,
            "r_train_means": r_train_means,
            "x_scaler": x_scaler,
            "z_scaler": z_scaler,
            "feat_x": feat_x,
            "feat_z": feat_z,
        },
    }
