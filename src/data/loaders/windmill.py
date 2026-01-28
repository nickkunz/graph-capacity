## libraries
import os
import sys
import numpy as np
import pandas as pd
import igraph as ig
from typing import Optional, Dict, Any
from torch_geometric_temporal.dataset import WindmillOutputLargeDatasetLoader
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.utilities import _load_network_pygt, _build_network_pygt, _create_igraph_object
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures

## process windmill dataset into daily event aggregates
def _process_events_wind(data: DynamicGraphTemporalSignal, hours: int = 24, thres: float = 1e-6) -> pd.DataFrame:

    ## build node × time matrix
    y = np.column_stack([snap.y.numpy() for snap in data])

    ## binarize production
    on = (y > thres).astype(np.int8)

    ## transitions per hour across all nodes
    events = np.abs(np.diff(on, axis = 1)).sum(axis = 0)
    events = np.concatenate([np.zeros(1, dtype = np.int32), events]).astype(np.int32)

    ## trim to full days and reshape
    totals = (events.size // hours) * hours
    daily = events[:totals].reshape(-1, hours).sum(axis = 1).astype(np.int64)
    
    return pd.DataFrame({
        "day": np.arange(daily.size, dtype = np.int64),
        "target": daily
        }
    )

## windmill power output network
class WindmillProcessor:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.dataset: Optional[DynamicGraphTemporalSignal] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.signatures: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        loader = WindmillOutputLargeDatasetLoader(raw_data_dir = self.raw_data_dir)
        self.dataset = _load_network_pygt(loader=loader)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.dataset is None:
            self.load_data()
        nodes, edges = _build_network_pygt(dataset = self.dataset)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_signatures(self):
        """Computes process signatures over daily turbine transitions."""
        if self.events is None:
            self.process_events()
            self.signatures = ProcessSignatures(
            data = self.events.copy(),
            sort_by = ["day"],
            target = "target"
        ).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.dataset is None:
            self.load_data()
        self.events = _process_events_wind(data = self.dataset)
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

