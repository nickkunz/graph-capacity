## libraries
from typing import Dict, Any

## modules
from src.estimators.linear_quantile import LinearQuantile
from src.estimators.linear_laws import LinearLAWS
from src.estimators.linear_convex import LinearConvex
from src.estimators.forest_quantile import ForestQuantile
from src.estimators.gradient_quantile import BoostingQuantile
from src.estimators.xgboost_quantile import XGBoostQuantile
from src.estimators.neural_networks import (
    NeuralQuantile,
    NeuralExpectile,
    NeuralConvex
)

## load all estimators
def load_estimators(input_dims: int | None = None, random_state: int = 42) -> Dict[str, Any]:
    return {
        ## linear parametric
        "linear_quantile": LinearQuantile(),
        "linear_convex": LinearConvex(),
        "linear_laws": LinearLAWS(),

        ## non-parametric ensembles
        "forest_quantile": ForestQuantile(random_state = random_state),
        "boosted_quantile": BoostingQuantile(random_state = random_state),
        "xgboost_quantile": XGBoostQuantile(random_state = random_state),

        ## neural networks
        "neural_quantile": NeuralQuantile(input_dims = input_dims, random_state = random_state),
        "neural_expectile": NeuralExpectile(input_dims = input_dims, random_state = random_state),
        "neural_convex": NeuralConvex(input_dims = input_dims, random_state = random_state),
    }
