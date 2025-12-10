## libraries
import os
import sys
import pandas as pd
from torch_geometric.datasets import JODIEDataset
from typing import Optional, Dict, Any

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import _create_igraph_object, _aggregate_by_day, _load_network_pyg, _build_network_pyg
from src.invariants import GraphInvariants
from src.descriptors import ProcessDescriptors

## load jodie wikipedia events from raw data
def load_events_jodie(data: JODIEDataset) -> pd.DataFrame:
    data = data[0]
    return pd.DataFrame({
        'src': data.src.numpy(),
        'dst': data.dst.numpy(),
        'timestamp': data.t.numpy()
    })

## process jodie wikipedia events
def process_events_jodie(data: pd.DataFrame) -> pd.DataFrame:
    data['day'] = (data['timestamp'] // 86400)
    return data.groupby('day').agg(
        target=('day', 'size')
    ).reset_index()

## jodie network
class JodieProcessor:
    def __init__(self, root_path: str, name: str):
        self.root_path = root_path
        self.name = name
        self.data_raw: Optional[JODIEDataset] = None
        self.graph = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.descriptors: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_raw = _load_network_pyg(
            dataset = "JODIEDataset",
            root = self.root_path,
            name = self.name
        )
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_raw is None:
            self.load_data()
        nodes, edges = _build_network_pyg(data = self.data_raw)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_raw is None:
            self.load_data()
        events = load_events_jodie(data = self.data_raw)
        self.events = process_events_jodie(data = events)
        return self

    def process_descriptors(self):
        """Computes process descriptors over daily interaction counts."""
        if self.events is None:
            self.process_events()
        self.descriptors = ProcessDescriptors(
            data = self.events.copy(),
            sort_by = ["day"],
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
            "descriptors": self.descriptors,
            "events": self.events.to_dict(orient = "records")
        }
