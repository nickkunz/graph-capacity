## logging
import logging
logger = logging.getLogger(__name__)

## libraries
import numpy as np

## utilities
from src.utils import _ensure_finite

####
## process descriptors for marked temporal point processes
####

class ProcessDescriptors:
    """
    Desc:
        Computes five universal process descriptors for a marked temporal
        point process defined on an igraph.Graph. These descriptors are 
        domain-agnostic and correspond to:

            rate          : event arrival frequency
            irregularity  : temporal disorder / non-Poissonity 
            intensity     : mean event magnitude
            heterogeneity : normalized dispersion of magnitudes
            simultaneity  : temporal synchrony via a Fano factor

        All outputs are guaranteed finite.

    Args:
        graph (igraph.Graph): unused except for interface symmetry with invariant extractor
        times (array-like): event timestamps (sorted or unsorted)
        marks (array-like): event magnitudes
        nodes (array-like): node indices for each event (not used in temporal descriptors)
    """

    def __init__(self, graph, times, marks, nodes):
        from igraph import Graph
        if not isinstance(graph, Graph):
            raise TypeError("Input graph must be an igraph.Graph object")

        self.graph = graph
        self.times = np.asarray(times, dtype=float)
        self.marks = np.asarray(marks, dtype=float)
        self.nodes = np.asarray(nodes, dtype=int)

        if not (len(self.times) == len(self.marks) == len(self.nodes)):
            raise ValueError("times, marks, and nodes must have equal length")

    
    ## rate: event arrival frequency λ = n / (T_end - T_start)
    def _rate(self):
        t = np.sort(self.times)
        n = len(t)
        if n < 2:
            return 0.0

        duration = t[-1] - t[0]
        val = n / duration if duration > 0 else 0.0
        return _ensure_finite(val, 0.0)

    
    ## irregularity: lag-1 autocorrelation of interarrival times τ
    def _irregularity(self):
        t = np.sort(self.times)
        tau = np.diff(t)

        if len(tau) < 2:
            return 0.0

        tau_centered = tau - tau.mean()
        num = np.sum(tau_centered[:-1] * tau_centered[1:])
        den = np.sum(tau_centered**2)

        val = num / den if den > 0 else 0.0
        return _ensure_finite(val, 0.0)

    
    ## intensity: mean event magnitude μ
    def _intensity(self):
        if len(self.marks) == 0:
            return 0.0
        val = float(np.mean(self.marks))
        return _ensure_finite(val, 0.0)

    
    ## heterogeneity: normalized spread of magnitudes κ = σ / μ
    def _heterogeneity(self, intensity):
        if intensity <= 0 or len(self.marks) < 2:
            return 0.0
        sd = float(np.std(self.marks, ddof=1))
        val = sd / (intensity + 1e-12)
        return _ensure_finite(val, 0.0)

    
    ## simultaneity: temporal synchrony using fano factor
    ## F = Var(N_w) / Mean(N_w), counts in sliding windows
    def _simultaneity(self):
        t = np.sort(self.times)
        n = len(t)

        if n < 3:
            return 0.0

        ## choose window length as median interarrival time
        tau = np.diff(t)
        if len(tau) == 0:
            return 0.0

        w = float(np.median(tau))
        if w <= 0:
            return 0.0

        ## construct window starts
        t_start = t[0]
        t_end = t[-1]
        n_windows = int((t_end - t_start) / w)
        if n_windows < 2:
            return 0.0

        window_starts = t_start + w * np.arange(n_windows)
        counts = np.zeros(n_windows, dtype=float)

        ## efficient event counting in windows
        idx = 0
        for i, ws in enumerate(window_starts):
            we = ws + w
            ## count events in [ws, we)
            c = 0
            while idx < n and t[idx] < we:
                if t[idx] >= ws:
                    c += 1
                idx += 1
            counts[i] = c

        m = np.mean(counts)
        v = np.var(counts, ddof=1) if len(counts) > 1 else 0.0

        if m <= 0:
            return 0.0

        fano = v / m
        return _ensure_finite(fano, 0.0)

    
    ## compute all descriptors
    def all(self) -> dict:
        features = {}

        rate = self._rate()
        features['rate'] = rate

        irregularity = self._irregularity()
        features['irregularity'] = irregularity

        intensity = self._intensity()
        features['intensity'] = intensity

        heterogeneity = self._heterogeneity(intensity)
        features['heterogeneity'] = heterogeneity

        simultaneity = self._simultaneity()
        features['simultaneity'] = simultaneity

        ## final check
        for k, v in features.items():
            if not np.isfinite(v):
                raise ValueError(f"Process descriptor '{k}' is non-finite ({v}).")

        return features
