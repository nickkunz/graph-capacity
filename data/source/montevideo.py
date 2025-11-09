## libraries
import os
import sys
import numpy as np
import pandas as pd
import igraph as ig
from typing import Optional, Dict, Any
from torch_geometric_temporal.dataset import MontevideoBusDatasetLoader
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import _load_network_pygt, _build_network_pygt, _create_igraph_object
from src.invariants import GraphInvariants

## process montevideo dataset into daily event aggregates
def _process_events_montevideo(data: DynamicGraphTemporalSignal, hours: int = 24, percentile: int = 99) -> pd.DataFrame:

    ## collect standardized inflow [nodes × hours]
    signal = np.stack([snap.y.numpy().flatten() for snap in data], axis=1)

    ## global thresholding for high activity
    threshold = np.percentile(a=signal, q=percentile)
    activity = signal > threshold

    ## count total high-activity events per hour
    events_per_hour = activity.sum(axis=0)

    ## trim to full days and reshape for daily aggregation
    num_days = events_per_hour.size // hours
    if num_days == 0:
        return pd.DataFrame({"day": [], "target": []})

    totals = num_days * hours
    daily_events = events_per_hour[:totals].reshape(-1, hours).sum(axis=1)

    ## create output dataframe
    return pd.DataFrame({
        "day": pd.to_datetime(pd.date_range(start='2020-01-01', periods=num_days, freq='D')).date,
        "target": daily_events.astype(np.int64)
    })

## montevideo bus network
class MontevideoProcessor:
    def __init__(self):
        self.dataset: Optional[DynamicGraphTemporalSignal] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        loader = MontevideoBusDatasetLoader()
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
        self.events = _process_events_montevideo(data = self.dataset)
        return self

    def run(self):
        """ Executes the pipeline and returns the final result. """
        self.process_network()
        self.process_events()
        return {
            "invariants": self.invariants,
            "events": self.events.to_dict(orient = "records")
        }