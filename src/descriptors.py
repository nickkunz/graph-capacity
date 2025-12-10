## logging
import logging
logger = logging.getLogger(__name__)

## libraries
import numpy as np
import pandas as pd

## modules
from src.utils import _ensure_finite

## process features class
class ProcessDescriptors:
    """
    Desc:
        Computes five universal process descriptors for an ordered sequence
        of aggregated counts.

    Args:
        data: Pandas Dataframe containing the observations.
        sort_by: List or tuple of column names to sort the data by.
        target: Column name of the target counts to analyze.

    Returns:
        dict with keys:
            rate: Mean count rate.
            irregularity: Measure of sequential correlation.
            heterogeneity: Coefficient of variation of counts.
            simultaneity: Fano factor of counts.
            burstiness: Mean absolute difference between consecutive counts.
    Raises:
        TypeError: If data is not a pandas dataframe.
        ValueError: If sort_by is empty or contains invalid columns.
                    If target is an invalid column.
                    If there are fewer than two observations.
    """

    def __init__(self, data, sort_by, target):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a Pandas Dataframe.")

        if not isinstance(sort_by, (list, tuple)):
            raise TypeError("sort_by must be a list or tuple")

        if len(sort_by) == 0:
            raise ValueError("sort_by must contain at least one column")

        for col in sort_by:
            if col not in data.columns:
                raise ValueError(f"sort_by column '{col}' not found in dataframe")

        if target not in data.columns:
            raise ValueError(f"target column '{target}' not found in dataframe")

        data_sorted = data.sort_values(sort_by)
        counts = data_sorted[target].astype(float).to_numpy()

        if len(counts) < 2:
            raise ValueError("must have at least two observations")

        self.counts = counts

    def _rate(self):
        m = float(np.mean(self.counts))
        return _ensure_finite(m, 0.0)

    def _irregularity(self):
        x = self.counts
        x_center = x - x.mean()
        num = np.sum(x_center[:-1] * x_center[1:])
        den = np.sum(x_center**2)
        val = num / den if den > 1e-12 else 0.0
        return _ensure_finite(val, 0.0)

    def _heterogeneity(self, rate):
        if rate <= 0:
            return 0.0
        sd = float(np.std(self.counts, ddof=1))
        cv = sd / (rate + 1e-12)
        return _ensure_finite(cv, 0.0)

    def _simultaneity(self, rate):
        if rate <= 0:
            return 0.0
        var = float(np.var(self.counts, ddof=1))
        fano = var / (rate + 1e-12)
        return _ensure_finite(fano, 0.0)

    def _burstiness(self, rate):
        diffs = np.abs(np.diff(self.counts))
        if len(diffs) == 0:
            return 0.0
        b = float(np.mean(diffs)) / (rate + 1e-12)
        return _ensure_finite(b, 0.0)

    ## compute all features and ensure they are finite
    def all(self):
        rate = self._rate()

        features = {
            "mean_count": rate,
            "lag1_autocorr": self._irregularity(),
            "coef_variation": self._heterogeneity(rate),
            "fano_factor": self._simultaneity(rate),
            "norm_succ_diff": self._burstiness(rate)
        }

        for k, v in features.items():
            if not np.isfinite(v):
                raise ValueError(f"process feature '{k}' is non-finite ({v})")

        return features
