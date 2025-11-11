## libraries
import os
import sys
import pandas as pd
import igraph as ig
from typing import Optional, Dict, Any
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import _load_network_pygt, _build_network_pygt, _create_igraph_object, _load_events_zip
from src.invariants import GraphInvariants

## process chickenpox events data
def _process_events_chickenpox(data: pd.DataFrame) -> pd.DataFrame:

    ## create independent copy and standardize column names
    data = data.copy()
    data.rename(columns = str.lower, inplace = True)

    ## ensure date column is datetime
    if "date" not in data.columns:
        raise RuntimeError("Expected 'date' column not found in data.")
    
    ## parse dates and drop invalid entries
    data['date'] = pd.to_datetime(
        arg = data['date'], 
        format = '%d/%m/%Y', 
        errors = 'coerce'
    )
    data.dropna(subset = ['date'], inplace = True)

    ## sum all counts per date
    feat_num = data.select_dtypes(include = 'number').columns.tolist()
    feat_sum = data[feat_num].sum(axis = 1)

    ## construct final dataframe
    return pd.DataFrame({
        "day": data['date'].dt.date,
        "target": feat_sum.astype("int64", errors = "ignore")
    })

## chickenpox network
class ChickenpoxProcessor:
    def __init__(self, url: str, name: str):
        self.url = url
        self.name = name
        self.data_network: Optional[Any] = None
        self.data_events: Optional[pd.DataFrame] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        loader = ChickenpoxDatasetLoader()
        self.dataset_network = _load_network_pygt(loader = loader)
        self.data_events = _load_events_zip(url = self.url, name=self.name)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.graph is None:
            self.load_data()
        nodes, edges = _build_network_pygt(dataset = self.dataset_network)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_events is None:
            self.load_data()
        self.events = _process_events_chickenpox(data = self.data_events)
        return self

    def run(self):
        """ Executes the pipeline and returns the final result. """
        self.process_network()
        self.process_events()
        return {
            "invariants": self.invariants,
            "events": self.events.to_dict(orient = "records")
        }
