## libraries
import numpy as np
import pandas as pd
from typing import Sequence, Tuple
from sklearn.preprocessing import StandardScaler

## zero-mean standardization
def _standardizer(data: pd.DataFrame, feat: Sequence[str]) -> Tuple[pd.DataFrame, StandardScaler]:
	if not feat:
		raise ValueError("feat argument must contain at least one column name.")
	data_copy = data.copy(deep = True)
	scaler = StandardScaler().fit(data_copy[feat].astype(float))
	data_copy[feat] = scaler.transform(data_copy[feat].astype(float))
	return data_copy, scaler

## log transformation
def _log_transformer(target: pd.Series) -> pd.Series:
	series = pd.to_numeric(target, errors = 'coerce')
	if series.empty or series.isna().all():
		raise ValueError("target series must contain at least one numeric value")
	log_series = np.log1p(series.clip(lower = 0))
	return pd.Series(data = log_series, index = target.index, dtype = float)
