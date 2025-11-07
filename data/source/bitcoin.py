## libraries
import os
import sys
import torch
import pandas as pd
import igraph
from torch_geometric.datasets import BitcoinOTC
from typing import Optional, Dict, Any

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import _create_igraph_object, _aggregate_by_day
from src.invariants import GraphInvariants

## build tripartite user–rating–user network
def build_network_bitcoin(data: BitcoinOTC, index: int = 10) -> tuple[list[str], list[tuple[str, str]]]:

    ## single graph object
    data_idx = data[index] ## snapshot at given index

    ## extract users and ratings
    users = torch.unique(data_idx.edge_index).cpu().numpy()
    user_nodes = [f"user_{u}" for u in users]

    ## all possible rating nodes (–10 to +10)
    rating_nodes = [f"rating_{r}" for r in range(-10, 11)]

    ## extract directed edges
    src_users = data_idx.edge_index[0].cpu().numpy()
    dst_users = data_idx.edge_index[1].cpu().numpy()
    ratings = data_idx.edge_attr.cpu().numpy().astype(int).flatten()

    ## build tripartite edges: user → rating → user
    edges = []
    for s, d, r in zip(src_users, dst_users, ratings):
        r_node = f"rating_{r}"
        edges.append((f"user_{s}", r_node))
        edges.append((r_node, f"user_{d}"))

    ## combine nodes
    nodes = user_nodes + rating_nodes
    return nodes, edges

## load event counts from bitcoin-otc dataset
def load_events_bitcoin(data: BitcoinOTC, index: int = 10) -> pd.DataFrame:
    
    ## ensure data is downloaded
    _ = data[index]  ## snapshot at given index
    data_raw = data.raw_paths[0]  ## ./BitcoinOTC/raw/soc-sign-bitcoinotc.csv

    ## read raw columns: src, dst, rating, timestamp
    data = pd.read_csv(
        filepath_or_buffer = data_raw,
        header = None,
        names = ["src", "dst", "rating", "timestamp"],
        dtype = {"src": int, "dst": int, "rating": int, "timestamp": float}
    )

    ## build datetime
    data["datetime"] = pd.to_datetime(
        arg = data["timestamp"],
        unit = "s",  ## seconds since epoch
        utc = True
    )
    return data[["datetime"]]

## bitcoin user–rating–user network
class BitcoinProcessor:
    def __init__(self, root_path: str, index: int = 10):  ## strictly the 11th snapshot
        self.root_path: str = root_path
        self.index: int = index
        self.data_raw: Optional[BitcoinOTC] = None
        self.graph: Optional[igraph.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_raw = BitcoinOTC(root = self.root_path + "bitcoin")
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_raw is None:
            self.load_data()
        nodes, edges = build_network_bitcoin(data = self.data_raw, index = self.index)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_raw is None:
            self.load_data()
        events = load_events_bitcoin(data = self.data_raw, index = self.index)
        self.events = _aggregate_by_day(events, datetime = 'datetime')
        return self

    def run(self):
        """ Executes the pipeline and returns the final result. """
        self.process_network()
        self.process_events()
        return {
            "invariants": self.invariants,
            "events": self.events.to_dict(orient = "records")
        }
