## libraries
import os
import sys
import numpy as np
import pandas as pd
import igraph as ig
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import _build_network_pygt, _create_igraph_object, _load_network_pygt, _load_events_zip
from src.invariants import GraphInvariants

## process wikimaths json to get daily event counts
def process_events_wiki(data: dict) -> pd.DataFrame:
    if "time_periods" not in data:
        raise ValueError("Input data is missing 'time_periods' key.")

    ## define the documented observation window
    start = "2019-03-16"
    periods = 731

    ## calculate the sum of events for each day
    events = [
        {'target': int(np.sum(data.get(str(i), {}).get("y", 0)))}
        for i in range(data["time_periods"])
    ]

    ## filter for the documented 731-day window and add correct dates
    data_events = pd.DataFrame(events)
    data_events = data_events.tail(periods).copy()
    data_events['date'] = pd.to_datetime(pd.date_range(start = start, periods = periods)).date
    return data_events[['date', 'target']]

## wikimaths network
class WikiProcessor:
    def __init__(self, url: str, name: str = "wikivital_mathematics.json"):
        self.url = url
        self.name = name
        self.data_network = None
        self.data_events = None
        self.graph = None
        self.invariants = None
        self.events = None

    def load_data(self):
        """ Loads the raw data from source. """
        loader = WikiMathsDatasetLoader()
        self.data_network = _load_network_pygt(loader = loader)
        self.data_events = _load_events_zip(
            url = self.url,
            name = self.name
        )
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_network is None:
            self.load_data()
        nodes, edges = _build_network_pygt(dataset = self.data_network)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_events is None:
            self.load_data()
        self.events = process_events_wiki(data = self.data_events)
        return self

    def run(self):
        """ Executes the pipeline and returns the final result. """
        self.process_network()
        self.process_events()
        return {
            "invariants": self.invariants,
            "events": self.events.to_dict(orient = "records")
        }
