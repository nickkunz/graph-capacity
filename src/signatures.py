## logging
import logging
logger = logging.getLogger(__name__)

## libraries
import numpy as np
import pandas as pd

## modules
from src.utils import _ensure_finite

## process signatures class
class ProcessSignatures:
    """
    Desc:
        Computes seven universal process signatures for an ordered sequence
        of aggregated counts.

    Args:
        data: Pandas Dataframe containing the observations.
        sort_by: List or tuple of column names to sort the data by.
        target: Column name of the target counts to analyze.

    Returns:
        dict with keys:
            lag1_autocorr: Measure of sequential correlation.
            coef_variation: Coefficient of variation of counts.
            fano_factor: Fano factor of counts.
            norm_succ_diff: Mean absolute difference between consecutive counts.
            rec_time_shape: Weibull shape parameter of recurrence times.
            hurst_exponent: Long-range dependence exponent.
            entropy_rate: First-order Markov entropy rate of event occurrence.

    Raises:
        TypeError: If data is not a pandas dataframe.
        ValueError: If sort_by is empty or contains invalid columns.
                    If target is an invalid column.
                    If there are fewer than two observations.
    """

    def __init__(self, data, sort_by, target):
        ## validate input dataframe
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a Pandas Dataframe.")

        ## validate sort_by
        if not isinstance(sort_by, (list, tuple)):
            raise TypeError("sort_by must be a list or tuple")
        if len(sort_by) == 0:
            raise ValueError("sort_by must contain at least one column")

        ## ensure sort_by columns exist
        for col in sort_by:
            if col not in data.columns:
                raise ValueError(f"sort_by column '{col}' not found in dataframe")

        ## ensure target exists
        if target not in data.columns:
            raise ValueError(f"target column '{target}' not found in dataframe")

        ## sort and extract count sequence
        data_sorted = data.sort_values(sort_by)
        counts = data_sorted[target].astype(float).to_numpy()

        ## require minimum temporal length
        if len(counts) < 2:
            raise ValueError("must have at least two observations")

        self.counts = counts

    ## compute centered lag-1 autocorrelation
    def _lag1_autocorr(self):
        x = self.counts
        x_center = x - x.mean()
        num = np.sum(x_center[:-1] * x_center[1:])
        den = np.sum(x_center**2)
        val = num / den if den > 1e-12 else 0.0
        return _ensure_finite(val, 0.0)

    ## compute coefficient of variation (scale-free dispersion)
    def _coef_variation(self):
        m = float(np.mean(self.counts))
        if m <= 0:
            return 0.0
        sd = float(np.std(self.counts, ddof = 1))
        cv = sd / (m + 1e-12)
        return _ensure_finite(cv, 0.0)

    ## compute overdispersion relative to mean (fano factor)
    def _fano_factor(self):
        m = float(np.mean(self.counts))
        if m <= 0:
            return 0.0
        var = float(np.var(self.counts, ddof = 1))
        fano = var / (m + 1e-12)
        return _ensure_finite(fano, 0.0)

    ## compute normalized mean absolute successive difference (local roughness)
    def _norm_succ_diff(self):
        m = float(np.mean(self.counts))
        if m <= 0:
            return 0.0
        diffs = np.abs(np.diff(self.counts))
        if len(diffs) == 0:
            return 0.0
        b = float(np.mean(diffs)) / (m + 1e-12)
        return _ensure_finite(b, 0.0)

    ## recurrence times helper
    def _recurrence_times(self):
        idx = np.where(self.counts > 0)[0]  ## positions of non-zero events and interarrivals
        if len(idx) < 2:
            return np.array([1.0], dtype = float)
        return np.diff(idx).astype(float)

    ## hurst exponent via rescaled range
    def _hurst_exponent(self):
        x = self.counts.astype(float)
        n = len(x)
        if n < 8:
            return 0.5

        x_center = x - x.mean()
        y = np.cumsum(x_center)
        r = np.max(y) - np.min(y)
        s = np.std(x_center, ddof = 1)

        ## degenerate fallback
        if s <= 1e-12 or r <= 1e-12:
            return 0.5

        rs = r / s
        h = np.log(rs) / np.log(float(n))
        h = max(0.0, min(1.0, h))
        return _ensure_finite(h, 0.5)

    ## compute correlation between counts and time index (dimensionless trend)
    def _trend_coeff(self):
        x = self.counts
        t = np.arange(len(x), dtype = float)

        ## center both variables
        xc = x - x.mean()
        tc = t - t.mean()

        num = np.sum(xc * tc)
        den = np.sqrt(np.sum(xc**2) * np.sum(tc**2))

        beta = num / den if den > 1e-12 else 0.0
        return _ensure_finite(float(beta), 0.0)

    ## compute skewness of counts distribution
    def _count_skewness(self):
        x = self.counts
        if len(x) < 3:
            return 0.0
        
        m = float(np.mean(x))
        sd = float(np.std(x, ddof = 1))
        
        if sd <= 1e-12:
            return 0.0
        
        ## standardized third moment
        x_std = (x - m) / sd
        skew = float(np.mean(x_std**3))
        
        return _ensure_finite(skew, 0.0)

    ## compute all signatures and ensure they are finite
    def all(self):
        signatures = {
            "lag1_autocorr": self._lag1_autocorr(),
            "coef_variation": self._coef_variation(),
            "fano_factor": self._fano_factor(),
            "norm_succ_diff": self._norm_succ_diff(),
            "hurst_exponent": self._hurst_exponent(),
            "trend_coeff": self._trend_coeff(),
            "count_skewness": self._count_skewness()
        }

        ## validate numerical stability of all signatures
        for k, v in signatures.items():
            if not np.isfinite(v):
                raise ValueError(f"process feature '{k}' is non-finite ({v})")

        return signatures
