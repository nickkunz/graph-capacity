## libraries
import os
import sys
import certifi
import itertools
import pandas as pd
import igraph as ig
import gwosc.datasets as gw
from typing import Optional, Dict, Any

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.invariants import GraphInvariants
from src.utils import _create_igraph_object, _aggregate_by_day
from src.descriptors import ProcessDescriptors

## load gravitational wave open science center network data
def _load_network_gwosc():
    return {
        "H1": (46.4553, -119.4077),
        "L1": (30.5630, -90.7740),
        "V1": (43.6314,  10.5045),
        "K1": (36.4127, 137.3093),
        "G1": (52.2462,  9.8083),
    }

## construct gravitational wave open science center network data
def _build_network_gwosc(data: dict) -> tuple[list[str], list[tuple]]:
    nodes = list(data.keys())
    edges = list(itertools.combinations(iterable = nodes, r = 2))
    return nodes, edges

## load gravitational wave open science center event data
def _load_events_gwosc(url: str):
    os.environ['SSL_CERT_FILE'] = certifi.where()
    return pd.read_csv(
        filepath_or_buffer = url,
        usecols = ["GPS", "commonName", "catalog.shortName"],
        na_filter = False
    )

## process gravitational wave open science center event data
def _process_events_gwosc(events: pd.DataFrame, network: set) -> pd.DataFrame:
    return (
        events[events["catalog.shortName"].astype(str).str.contains("GWTC", na = False)]
        .assign(
            datetime = lambda data: pd.to_datetime("1980-01-06", utc = True) + pd.to_timedelta(pd.to_numeric(data["GPS"], errors = "coerce"), unit = "s"),
            network = lambda data: data['commonName'].apply(lambda x: list(gw.event_detectors(x)))
        )
        .drop(columns=["GPS"])
        .dropna(subset = ["datetime"])
        .sort_values("datetime")
        .loc[lambda data: data["network"].apply(lambda dets: len(set(dets) & network) > 0)]
        .reset_index(drop = True)
    )

## gravitational wave open science center network
class GwoscProcessor:
    def __init__(self, url: str):
        self.url = url
        self.data_network: Optional[Dict[str, tuple]] = None
        self.data_events: Optional[pd.DataFrame] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.descriptors: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_network = _load_network_gwosc()
        self.data_events = _load_events_gwosc(url = self.url)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_network is None:
            self.load_data()
        nodes, edges = _build_network_gwosc(data = self.data_network)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_events is None or self.data_network is None:
            self.load_data()
        network_nodes = set(self.data_network.keys())
        events = _process_events_gwosc(events = self.data_events, network = network_nodes)
        self.events = _aggregate_by_day(
            data = events,
            datetime = 'datetime',
            label = "date"
        )
        return self

    def process_descriptors(self):
        """Computes process descriptors over daily detections."""
        if self.events is None:
            self.process_events()
        self.descriptors = ProcessDescriptors(
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
            "descriptors": self.descriptors,
            "events": self.events.to_dict(orient = "records")
        }
