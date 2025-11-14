## libraries
import numpy as np
import pandas as pd
from typing import Sequence, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler

## zero-centered and unit-variance normalization
def standardize_invariants(data: pd.DataFrame, feat_invar: Sequence[str]) -> Tuple[pd.DataFrame, StandardScaler]:
	if not feat_invar:
		raise ValueError("feat_invar argument must contain at least one column name.")
	data_copy = data.copy(deep = True)
	scaler = StandardScaler().fit(data_copy[feat_invar].astype(float))
	data_copy[feat_invar] = scaler.transform(data_copy[feat_invar].astype(float))
	return data_copy, scaler

## normalized log-scaled target values
def normalize_omegalog(target: pd.Series) -> Tuple[pd.DataFrame, MinMaxScaler]:
	series = pd.to_numeric(target, errors = 'coerce')
	if series.empty or series.isna().all():
		raise ValueError("target series must contain at least one numeric value")
	log_series = np.log1p(series.clip(lower=0))
	scaler = MinMaxScaler()
	data = scaler.fit_transform(log_series.to_frame()).ravel()
	return pd.Series(data = data, index = target.index, dtype = float), scaler
