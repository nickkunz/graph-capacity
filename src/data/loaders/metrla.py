
## libraries
import os
import sys
import numpy as np
import pandas as pd
import igraph as ig
from typing import Optional, Dict, Any
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import _load_network_pygt, _build_network_pygt, _create_igraph_object
from src.invariants import GraphInvariants
from src.descriptors import ProcessDescriptors

## process metr-la dataset into daily event aggregates
def _process_events_metrla(data: DynamicGraphTemporalSignal, sample_rate_minutes: int = 5, thres_percentile: int = 1, thres_min: float = 1e-6) -> pd.DataFrame:
    
    ## calculate samples per day
    samples_per_day = (24 * 60) // sample_rate_minutes

    ## collect speeds [nodes × time]
    speed = np.stack([snap.y.numpy().flatten() for snap in data], axis=1)

    ## derive binary congestion via per-node threshold
    threshold = np.maximum(np.percentile(speed, thres_percentile, axis=1), thres_min)[:, None]
    congested = speed <= threshold

    ## detect congestion stop events (transition from not congested to congested)
    stop_events = np.zeros_like(congested, dtype=np.int32)
    stop_events[:, 1:] = (~congested[:, 1:] & congested[:, :-1]).astype(np.int32)
    events_per_sample = stop_events.sum(axis=0)

    ## trim to full days and reshape for daily aggregation
    num_days = events_per_sample.size // samples_per_day
    if num_days == 0:
        return pd.DataFrame({"day": [], "target": []})
    
    totals = num_days * samples_per_day
    daily_events = events_per_sample[:totals].reshape(-1, samples_per_day).sum(axis=1)

    ## create output dataframe
    return pd.DataFrame({
        "date": pd.to_datetime(
            pd.date_range(
                start = '2012-03-01', 
                periods = num_days, 
                freq = 'D'
            )
        ).date,
        "target": daily_events.astype(np.int64)
    })

## metr-la traffic network
class MetrLaProcessor:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.dataset: Optional[DynamicGraphTemporalSignal] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.features: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        loader = METRLADatasetLoader(raw_data_dir = self.raw_data_dir)
        self.dataset = _load_network_pygt(loader = loader)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.dataset is None:
            self.load_data()
        nodes, edges = _build_network_pygt(dataset = self.dataset)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.dataset is None:
            self.load_data()
        self.events = _process_events_metrla(data = self.dataset)
        return self

    def process_descriptors(self):
        """Computes process descriptors over daily congestion events."""
        if self.events is None:
            self.process_events()
        self.features = ProcessDescriptors(
            data = self.events.copy(),
            sort_by = ["date"],
            target = "target"
        ).all()
        return self

    def run(self):
        """ Executes the pipeline and returns the final result. """
        self.process_network()
        self.process_descriptors()
        self.process_events()
        return {
            "invariants": self.invariants,
            "features": self.features,
            "events": self.events.to_dict(orient = "records")
        }
