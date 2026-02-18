## libraries
import io
import os
import sys
import zipfile
import certifi
import numpy as np
import pandas as pd
import requests
import igraph as ig
from typing import Optional, Dict, Any
from datetime import datetime, timezone

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.data.utilities import _create_igraph_object, _aggregate_by_day

## filter auger stations by array and validate coordinates
def _process_network_auger(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data
        .dropna(subset = ["id", "northing", "easting", "sd1500", "sd750"])
        .assign(id = lambda data: data["id"].astype(int).astype(str))
        [["id", "northing", "easting", "sd1500", "sd750"]]
        .reset_index(drop = True)
    )

## build station graph with mutual 6-nn at nominal spacing
def _build_network_auger(station_map: pd.DataFrame, array: str = "both"):
    
    ## build both arrays and combine
    if array == "both":
        edges_1500, nodes_1500 = _build_network_auger(station_map, array = "sd1500")
        edges_750, nodes_750 = _build_network_auger(station_map, array = "sd750")

        ## combine unique nodes and all edges
        all_nodes = list(set(nodes_1500 + nodes_750))
        all_edges = edges_1500 + edges_750
        return all_edges, all_nodes
    
    ## filter
    sel = station_map.copy()
    if array in ("sd1500", "sd750"):
        sel = sel[sel[array] == 1].reset_index(drop = True)
    if len(sel) < 2:
        return [], []

    ## ids and coordinates
    ids = sel["id"].astype(str).tolist()
    xy = sel[["easting", "northing"]].to_numpy(dtype = float)
    n = len(ids)

    ## spacing and tolerance
    spacing = 1500.0 if array == "sd1500" else 750.0
    tol = spacing * 1.1

    ## pairwise distances
    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]
    D = np.hypot(dx, dy)
    np.fill_diagonal(D, np.inf)

    ## for each node: files within tol, then take up to 6 nearest
    nbrs = []
    for i in range(n):
        cand = np.where(D[i] <= tol)[0]
        if cand.size > 6:
            cand = cand[np.argsort(D[i, cand])[:6]]
        nbrs.append(set(cand))

    ## mutual condition: i in N6(j) and j in N6(i)
    edges = []
    for i in range(n):
        for j in nbrs[i]:
            if i < j and i in nbrs[j]:
                edges.append((ids[i], ids[j]))

    return edges, ids


## load auger cosmic ray events from summary zip
def _load_events_auger(url: str, timeout: int = 120) -> pd.DataFrame:
    
    ## download and extract zip
    response = requests.get(url = url, timeout = timeout)
    response.raise_for_status()
    
    z = zipfile.ZipFile(io.BytesIO(response.content))
    
    ## find csv files
    names = z.namelist()
    files = [n for n in names if n.lower().endswith(".csv")]
    
    ## read all csv files
    frames = []
    for name in files:
        try:
            with z.open(name) as fh:
                df = pd.read_csv(fh)
                frames.append(df)
        except Exception:
            continue
    
    if not frames:
        raise RuntimeError("no readable csv found in auger summary.zip")
    
    return pd.concat(frames, ignore_index = True)

## process auger cosmic ray events, convert time and extract key fields
def _process_events_auger(data: pd.DataFrame) -> pd.DataFrame:
    
    ## convert gps time to datetime (seconds since 1980-01-06 UTC)
    gps_epoch = datetime(1980, 1, 6, tzinfo = timezone.utc)
    
    return (
        data
        .assign(
            event_id = lambda df: df["id"].astype(str),
            datetime = lambda df: gps_epoch + pd.to_timedelta(df["gpstime"], unit = "s"),
            energy = lambda df: pd.to_numeric(df["sd_energy"], errors = "coerce"),
            zenith_deg = lambda df: pd.to_numeric(df["sd_theta"], errors = "coerce")
        )
        [["event_id", "datetime", "energy", "zenith_deg"]]
        .dropna(subset = ["datetime"])
        .sort_values("datetime")
        .reset_index(drop = True)
    )


## auger cosmic ray network
class AugerProcessor:
    def __init__(self, url_network: str, url_events: str):
        self.url_network = url_network
        self.url_events = url_events
        self.data_network: Optional[pd.DataFrame] = None
        self.data_events: Optional[pd.DataFrame] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.signatures: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        os.environ['SSL_CERT_FILE'] = certifi.where()
        self.data_network = pd.read_csv(filepath_or_buffer=self.url_network)
        self.data_events = _load_events_auger(url=self.url_events)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_network is None:
            self.load_data()
        station_map = _process_network_auger(data=self.data_network)
        edges, nodes = _build_network_auger(station_map=station_map)
        self.graph = _create_igraph_object(nodes=nodes, edges=edges)
        self.invariants = GraphInvariants(graph=self.graph).all()
        return self

    def process_signatures(self):
        """Computes process signatures over daily high-energy events."""
        if self.events is None:
            self.process_events()
        self.signatures = ProcessSignatures(
            data = self.events.copy(),
            sort_by = ["date"],
            target = "target"
        ).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_events is None:
            self.load_data()

        events_processed = _process_events_auger(data=self.data_events)
        self.events = _aggregate_by_day(
            data = events_processed,
            datetime = 'datetime',
            label = 'date'
        )
        return self

    def run(self):
        """ Executes the pipeline and returns the final result. """
        self.process_network()
        self.process_signatures()
        self.process_events()
        return {
            "invariants": self.invariants,
            "signatures": self.signatures,
            "graph": self.graph,
            "events": self.events.to_dict(orient="records")
        }
