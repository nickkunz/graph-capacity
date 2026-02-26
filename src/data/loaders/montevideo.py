## libraries
import os
import sys
import numpy as np
import pandas as pd
import igraph as ig
from pathlib import Path
from typing import Optional, Dict, Any
from torch_geometric_temporal.dataset import MontevideoBusDatasetLoader
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

## path
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.data.helpers import (
    _load_network_pygt,
    _build_network_pygt,
    _create_igraph_object
)

## process montevideo dataset into daily event aggregates
def _process_events_montevideo(data, hours: int = 24, perc: int = 99) -> pd.DataFrame:

    ## collect standardized inflow [nodes × hours]
    ys = []
    for snap in data:
        ys.append(snap.y.detach().cpu().numpy().ravel())
    signal = np.column_stack(ys)

    ## global thresholding for high activity
    threshold = np.percentile(a = signal, q = perc)
    activity = signal > threshold

    ## count total high-activity events per hour
    events = activity.sum(axis = 0)

    ## trim to full days
    totals = (events.size // hours) * hours
    events = events[:totals].reshape(-1, hours)

    ## daily aggregation
    daily = events.sum(axis = 1).astype(np.int32)

    ## construct dataframe
    return pd.DataFrame({
        "day": np.arange(daily.size, dtype = np.int64),
        "target": daily
    })

## montevideo bus network
class MontevideoProcessor:
    def __init__(self):
        self.dataset: Optional[DynamicGraphTemporalSignal] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.signatures: Optional[Dict[str, Any]] = None
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

    def process_signatures(self):
        """Computes process signatures over daily high-activity events."""
        if self.events is None:
            self.process_events()
        self.signatures = ProcessSignatures(
            data = self.events.copy(),
            sort_by = ["day"],
            target = "target"
        ).all()
        return self

    def run(self):
        """ Executes the pipeline and returns the final result. """
        self.process_network()
        self.process_signatures()
        self.process_events()
        return {
            "invariants": self.invariants,
            "signatures": self.signatures,
            "events": self.events.to_dict(orient = "records")
        }