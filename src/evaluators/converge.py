"""Cross-validated convergence analysis with conformal calibration.

This script mirrors the convergence analysis that lives in
``notebooks/experiments3.ipynb`` but wraps it in a pure Python entry-point so
it can be reused in automation or from the CLI. It loads the feature matrix,
runs Leave-One-Group-Out training with conformal residual shifts, and prints
pairwise correlations plus dispersion statistics across every model family.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, QuantileRegressor, Ridge
from sklearn.model_selection import LeaveOneGroupOut

# Ensure the repository root is on sys.path when executed as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.append(str(REPO_ROOT))

from src.vectorizers.scalers import _log_transform_target, _standardize_invariants


DEFAULT_DROP_COLUMNS = ["name", "domain", "discipline"]
DEFAULT_GROUP_COLUMN = "domain"
DEFAULT_MODELS = ["quant", "ridge", "lasso", "rf", "gbm"]
QUANTILE_FAMILY = {"quant", "gbm"}
SQUARED_FAMILY = {"ridge", "lasso"}


def _build_estimators(quantile: float) -> Mapping[str, BaseEstimator]:
	"""Return the superset of estimators keyed by short names."""

	estimators: Dict[str, BaseEstimator] = {
		"quant": QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs"),
		"ridge": Ridge(alpha=0.1, positive=True),
		"lasso": Lasso(alpha=0.1, positive=True),
		"rf": RandomForestRegressor(
            criterion = "absolute_error",  ## L1 = hinge-like
            bootstrap = True
        ),
		"gbm": GradientBoostingRegressor(
            loss = 'quantile',
            alpha = 0.99  ## 99th percentile
        ),
	}

	return estimators


def prediction_correlation(y_pred_1: np.ndarray, y_pred_2: np.ndarray) -> float:
	"""Spearman correlation between two prediction vectors with safe fallbacks."""

	flat_1 = np.asarray(y_pred_1).ravel()
	flat_2 = np.asarray(y_pred_2).ravel()

	is_const_1 = np.allclose(flat_1, flat_1[0])
	is_const_2 = np.allclose(flat_2, flat_2[0])

	if is_const_1 and is_const_2:
		return 1.0 if np.isclose(flat_1[0], flat_2[0]) else 0.0
	if is_const_1 or is_const_2:
		return 0.0

	rho, _ = spearmanr(flat_1, flat_2)
	return float(0.0 if np.isnan(rho) else rho)


def prediction_dispersion(predictions_dict: Mapping[str, np.ndarray]) -> Dict[str, float]:
	"""Coefficient of variation across model predictions."""

	pred_matrix = np.column_stack(list(predictions_dict.values()))
	cv = np.std(pred_matrix, axis=1) / np.mean(pred_matrix, axis=1)

	return {
		"mean_cv": float(np.mean(cv)),
		"max_cv": float(np.max(cv)),
		"cv_per_instance": cv,
	}


def _pairwise_correlations(predictions: Mapping[str, np.ndarray]) -> List[Tuple[str, str, float]]:
	"""Compute pairwise correlations for every model combination."""

	names = list(predictions.keys())
	results: List[Tuple[str, str, float]] = []

	for i in range(len(names)):
		for j in range(i + 1, len(names)):
			name_i, name_j = names[i], names[j]
			rho = prediction_correlation(predictions[name_i], predictions[name_j])
			results.append((name_i, name_j, rho))

	return results


def _summarize_family(
	predictions: Mapping[str, np.ndarray],
	members: Iterable[str],
) -> Tuple[float, List[Tuple[str, str, float]]]:
	"""Return mean correlation plus detailed tuples for a subset of models."""

	present = [name for name in members if name in predictions]
	corr_tuples = _pairwise_correlations({k: predictions[k] for k in present})
	if not corr_tuples:
		return float("nan"), []

	rho_values = [rho for _, _, rho in corr_tuples]
	return float(np.mean(rho_values)), corr_tuples


def _format_pairwise(results: Sequence[Tuple[str, str, float]]) -> str:
	lines = [f"  {a} vs {b}: ρ = {rho:.3f}" for a, b, rho in results]
	return "\n".join(lines)


def compute_convergence_summary(predictions: Mapping[str, np.ndarray]) -> str:
	"""Return a multi-line human-readable convergence report."""

	pairwise = _pairwise_correlations(predictions)
	rho_values = np.array([rho for _, _, rho in pairwise]) if pairwise else np.array([])

	dispersion = prediction_dispersion(predictions)

	lines: List[str] = []
	lines.append("=== TRUE MODEL CONVERGENCE (FULL DATA) ===")
	if pairwise:
		lines.append(_format_pairwise(pairwise))
	else:
		lines.append("  (not enough models to compute pairwise correlations)")

	if rho_values.size:
		lines.extend(
			[
				"",
				"=== SUMMARY ===",
				f"Mean pairwise correlation: {rho_values.mean():.3f}",
				f"Min pairwise correlation: {rho_values.min():.3f}",
				f"Max pairwise correlation: {rho_values.max():.3f}",
				f"Std pairwise correlation: {rho_values.std():.3f}",
			]
		)
	else:
		lines.append("\n=== SUMMARY ===\nInsufficient models for statistics")

	lines.append(f"Prediction dispersion (CV): {dispersion['mean_cv']:.3f}")

	# Family summaries
	quant_mean, quant_pairs = _summarize_family(predictions, QUANTILE_FAMILY)
	sq_mean, sq_pairs = _summarize_family(predictions, SQUARED_FAMILY)

	if quant_pairs:
		lines.extend(
			[
				"",
				"=== WITHIN-FAMILY CONVERGENCE (Quantile Models) ===",
				_format_pairwise(quant_pairs),
				f"  mean: {quant_mean:.3f}",
			]
		)

	if sq_pairs:
		lines.extend(
			[
				"",
				"=== WITHIN-FAMILY CONVERGENCE (Squared-Loss Models) ===",
				_format_pairwise(sq_pairs),
				f"  mean: {sq_mean:.3f}",
			]
		)

	# Cross family
	quant_preds = {k: predictions[k] for k in predictions if k in QUANTILE_FAMILY}
	sq_preds = {k: predictions[k] for k in predictions if k in SQUARED_FAMILY}
	cross_results: List[Tuple[str, str, float]] = []
	for q_name, q_pred in quant_preds.items():
		for s_name, s_pred in sq_preds.items():
			rho = prediction_correlation(q_pred, s_pred)
			cross_results.append((q_name, s_name, rho))

	if cross_results:
		lines.extend(
			[
				"",
				"=== CROSS-FAMILY CONVERGENCE (Quantile vs Squared-Loss) ===",
				_format_pairwise(cross_results),
				f"  mean: {np.mean([rho for _, _, rho in cross_results]):.3f}",
			]
		)

	return "\n".join(lines)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Fit all models on full data and report convergence metrics",
	)
	parser.add_argument(
		"--data-path",
		type=Path,
		default=Path("outputs/data/data.csv"),
		help="CSV file containing the aggregated invariants and targets.",
	)
	parser.add_argument(
		"--target-column",
		default="target",
		help="Column name containing the response variable.",
	)
	parser.add_argument(
		"--drop-columns",
		nargs="*",
		default=DEFAULT_DROP_COLUMNS,
		help="Columns to exclude from the feature set.",
	)
	parser.add_argument(
		"--group-column",
		default=DEFAULT_GROUP_COLUMN,
		help="Column used to define Leave-One-Group-Out folds.",
	)
	parser.add_argument(
		"--models",
		default=",".join(DEFAULT_MODELS),
		help="Comma-separated list of estimator keys to run.",
	)
	parser.add_argument(
		"--quantile",
		type=float,
		default=0.99,
		help="Quantile used by frontier-oriented models.",
	)
	parser.add_argument(
		"--vr-target",
		type=float,
		default=0.01,
		help="Target violation rate for conformal calibration.",
	)
	parser.add_argument(
		"--log-level",
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
		help="Logging verbosity.",
	)
	return parser.parse_args()


def select_models(all_estimators: Mapping[str, BaseEstimator], selection: str) -> Dict[str, BaseEstimator]:
	requested = [name.strip() for name in selection.split(",") if name.strip()]
	if not requested:
		raise ValueError("At least one model must be specified via --models")

	selected: Dict[str, BaseEstimator] = {}
	for name in requested:
		if name not in all_estimators:
			logging.warning("Estimator '%s' is unavailable and will be skipped", name)
			continue
		selected[name] = all_estimators[name]

	if not selected:
		raise ValueError("None of the requested models could be constructed")
	return selected


def load_features(
	data_path: Path,
	drop_columns: Sequence[str],
	target_column: str,
	group_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
	df = pd.read_csv(data_path)
	missing_target = target_column not in df.columns
	if missing_target:
		raise ValueError(f"Target column '{target_column}' not present in {data_path}")
	if group_column not in df.columns:
		raise ValueError(f"Group column '{group_column}' not present in {data_path}")

	drop_set = set(drop_columns) | {target_column}
	feature_cols = [c for c in df.columns if c not in drop_set]
	if not feature_cols:
		raise ValueError("No feature columns remain after dropping exclusions")

	X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
	y_series = _log_transform_target(df[target_column])
	group_series = df[group_column]
	return X_df, y_series, group_series, feature_cols


def fit_predictions(
	estimators: Mapping[str, BaseEstimator],
	X_df: pd.DataFrame,
	y_series: pd.Series,
	group_series: pd.Series,
	feature_cols: Sequence[str],
	vr_target: float,
) -> Dict[str, np.ndarray]:
	"""Generate conformally calibrated LOGO predictions for every estimator."""

	logo = LeaveOneGroupOut()
	groups = group_series.values
	if len(groups) != len(X_df):
		raise ValueError("Group column must align with feature matrix")

	predictions: Dict[str, np.ndarray] = {
		name: np.full(len(X_df), np.nan, dtype=float) for name in estimators
	}
	q = 1.0 - vr_target

	for fold_idx, (train_idx, test_idx) in enumerate(
		logo.split(X=X_df.values, y=y_series.values, groups=groups),
		start=1,
	):
		logging.debug("LOGO fold %d: train=%d test=%d", fold_idx, len(train_idx), len(test_idx))

		X_train = X_df.iloc[train_idx]
		y_train = y_series.iloc[train_idx].values
		X_test = X_df.iloc[test_idx]
		X_train_scaled, scaler = _standardize_invariants(X_train, feature_cols)
		X_train_scaled = X_train_scaled[feature_cols].values
		X_test_scaled = scaler.transform(X_test[feature_cols].astype(float))

		for name, estimator in estimators.items():
			model = clone(estimator)
			model.fit(X_train_scaled, y_train)

			y_pred_train = model.predict(X_train_scaled)
			residuals = y_train - y_pred_train
			c = float(np.quantile(residuals, q, method="linear"))
			pred_fold = model.predict(X_test_scaled) + c
			predictions[name][test_idx] = pred_fold

	for name, values in predictions.items():
		if np.isnan(values).any():
			raise RuntimeError(f"Missing predictions for estimator '{name}' after LOGO folds")

	return predictions


def main() -> None:
	args = parse_args()
	logging.basicConfig(level=getattr(logging, args.log_level.upper()))

	all_estimators = _build_estimators(args.quantile)
	estimators = select_models(all_estimators, args.models)

	X_df, y_series, group_series, feature_cols = load_features(
		data_path=args.data_path,
		drop_columns=args.drop_columns,
		target_column=args.target_column,
		group_column=args.group_column,
	)

	predictions = fit_predictions(
		estimators=estimators,
		X_df=X_df,
		y_series=y_series,
		group_series=group_series,
		feature_cols=feature_cols,
		vr_target=args.vr_target,
	)

	report = compute_convergence_summary(predictions)
	print(report)


if __name__ == "__main__":
	main()
