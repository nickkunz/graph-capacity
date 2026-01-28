## libraries
import os
import sys
import pandas as pd
import osmnx as ox
import networkx as nx
import igraph as ig
from typing import Optional, Dict, Any

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures

## extract street network graph
def _load_network_idling(query, network_type = 'drive', simplify = False):
    if not query:
        raise ValueError("Query must be provided to extract the graph.")
    if not isinstance(query, str) or not query:
        raise ValueError("Query must be a non-empty string.")
    if isinstance(query, str):
        query = query.strip()
    valid_types = ['drive', 'walk', 'bike', 'all']
    if network_type not in valid_types:
        raise ValueError(f"Invalid network type. Choose from {valid_types}.")
    return ox.graph_from_place(
        query = query,
        network_type = network_type,
        simplify = simplify
    )

## process street network graph
def _process_network_idling(G: nx.MultiDiGraph) -> ig.Graph:

    ## reproject into specified coordinate system
    incoming_graph_data = ox.project_graph(G = G, to_crs = 'EPSG:4326')  ## WGS84

    ## remove redundant attributes from nodes after reprojection
    for _, data in incoming_graph_data.nodes(data = True):
        for attr in ['x', 'y', 'lat', 'lon', 'street_count']:
            data.pop(attr, None)

    # convert to a simple, undirected igraph object without parallel edges or self-loops
    g = nx.Graph(incoming_graph_data = incoming_graph_data)
    return ig.Graph.from_networkx(g = g)

## load events data
def _load_events_idling(path: str) -> pd.DataFrame:
    """
    Loads and concatenates all CSV files from a directory.
    """
    try:
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not files:
            raise FileNotFoundError(f"No CSV files found in the directory: {path}")
        return pd.concat([pd.read_csv(os.path.join(path, f)) for f in files], ignore_index=True)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

## process events data
def _process_events_idling(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.assign(date = pd.to_datetime(data['datetime'], unit = 's').dt.date)
        .groupby('date')
        .size()
        .reset_index(name = 'target')
    )

## idling vehicle network
class IdlingProcessor:
    def __init__(self, path_events: str, query: str = "Halifax, Canada"):
        self.path_events = path_events
        self.query = query
        self.data_network: Optional[nx.MultiDiGraph] = None
        self.data_events: Optional[pd.DataFrame] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.signatures: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_network = _load_network_idling(query = self.query)
        self.data_events = _load_events_idling(path = self.path_events)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_network is None:
            self.load_data()
        self.graph = _process_network_idling(G = self.data_network)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_events is None:
            self.load_data()
        self.events = _process_events_idling(data = self.data_events)
        return self

    def process_signatures(self):
        """Computes process signatures over daily idling events."""
        if self.events is None:
            self.process_events()
        self.signatures = ProcessSignatures(
            data = self.events.copy(),
            sort_by = ["date"],
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
