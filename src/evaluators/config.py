## libraries
import pandas as pd
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

## constants
FEAT_X = [
    "n_nodes",
    "n_edges",
    "n_articulation_points",
    "n_bridges",
    "diameter",
    "radius",
    "degeneracy",
    "k_core_size",
    "maximum_degree",
    "degree_variance",
    "global_clustering",
    "degree_assortativity",
    "degree_entropy",
    "joint_degree_entropy",
    "degree_skewness",
    "normalized_laplacian_second_moment",
    "normalized_laplacian_third_moment",
    "random_walk_triangle_weight",
    "random_walk_fourth_moment",
    "adjacency_fourth_moment_per_node",
]
FEAT_Z = [
    "lag1_autocorr",
    "coef_variation",
    "fano_factor",
    "norm_succ_diff",
    "hurst_exponent",
    "trend_coeff",
    "count_skewness",
]
TARGET = "target"

## loaders
def load_data(filepath: str = "../data/main.csv") -> pd.DataFrame:
    return pd.read_csv(filepath)

def load_models(input_dims: int = None) -> Dict[str, Any]:
    return {
        ## linear parametric
        "linear_quantile": LinearQuantile(),
        "linear_convex": LinearConvex(),
        "linear_laws": LinearLAWS(),

        ## non-parametric ensembles
        "forest_quantile": ForestQuantile(),
        "boosting_quantile": BoostingQuantile(),
        "xgboost_quantile": XGBoostQuantile(),

        ## neural networks
        "neural_quantile": NeuralQuantile(input_dims = input_dims),
        "neural_expectile": NeuralExpectile(input_dims = input_dims),
        "neural_convex": NeuralConvex(input_dims = input_dims)
    }
