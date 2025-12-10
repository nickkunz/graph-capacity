## libraries
import os
import sys
import pandas as pd
from typing import Any, Dict, Optional

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import _create_igraph_object, _request_with_retry
from src.invariants import GraphInvariants
from src.descriptors import ProcessDescriptors

## load faers drug–reaction reporting network data
def _load_network_faers(id: str, url: str) -> pd.DataFrame:
    q = f'patient.drug.medicinalproduct.exact:"{id.upper()}"'
    limit, skip = 1000, 0
    
    obs = []
    while True:
        params = {
            "search": q,
            "limit": limit,
            "skip": skip,
        }
        try:
            response = _request_with_retry(
                url = url,
                params = params,
                timeout = 60,
            )
        except RuntimeError as e:
            if "404" in str(e):
                break
            raise e
        results = (response.json() or {}).get("results", [])
        if not results:
            break

        for rec in results:
            reactions = []
            for rx in (rec.get("patient", {}) or {}).get("reaction", []) or []:
                term = (rx.get("reactionmeddrapt") or "").strip().upper()
                if term:
                    reactions.append(term)
            drugs = []
            for d in (rec.get("patient", {}) or {}).get("drug", []) or []:
                lbl = (d.get("medicinalproduct") or "").strip().upper()
                if lbl:
                    drugs.append(lbl)
            for lbl in drugs:
                for term in reactions:
                    obs.append({"drug": lbl, "reaction": term})
        skip += limit

    if not obs:
        raise RuntimeError(f"no drug–reaction data found for {id}")
    return pd.DataFrame(obs)

## process faers drug–reaction reporting network data
def _process_network_faers(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data[["drug", "reaction"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop = True)
    )

## build faers drug–reaction reporting network data
def _build_network_faers(data: pd.DataFrame) -> tuple[list[str], list[tuple]]:
    data_network = data[["drug", "reaction"]].dropna().drop_duplicates()
    nodes = pd.unique(data_network[['drug', 'reaction']].values.ravel('K')).tolist()
    edges = [tuple(x) for x in data_network.to_numpy()]
    return nodes, edges

## load faers adverse event reports
def _load_events_faers(id: str, url: str) -> pd.DataFrame:
    q = f'patient.drug.medicinalproduct.exact:"{id.upper()}"'
    limit, skip = 1000, 0
    obs = []

    while True:
        params = {
            "search": q,
            "limit": limit,
            "skip": skip,
        }
        try:
            response = _request_with_retry(
                url = url, 
                params = params, 
                timeout = 60,
            )
        except RuntimeError as e:
            if "404" in str(e):
                break
            raise e
        results = (response.json() or {}).get("results", [])
        if not results:
            break
        for rec in results:
            date_str = rec.get("receiptdate") or rec.get("receivedate")
            if not date_str:
                continue
            dt = pd.to_datetime(date_str, format = "%Y%m%d", errors = "coerce")
            if pd.isna(dt):
                continue
            for rx in (rec.get("patient", {}) or {}).get("reaction", []) or []:
                term = (rx.get("reactionmeddrapt") or "").strip().upper()
                if not term:
                    continue
                for d in (rec.get("patient", {}) or {}).get("drug", []) or []:
                    lbl = (d.get("medicinalproduct") or "").strip().upper()
                    if lbl:
                        obs.append({"drug": lbl, "reaction": term, "date": dt.normalize()})
        skip += limit

    if not obs:
        raise RuntimeError(f"no event data found for {id}")
    return pd.DataFrame(obs)

## process faers adverse event reports
def _process_events_faers(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns = ["date", "target"])
    data["date"] = pd.to_datetime(arg = data["date"]).dt.date
    return data.groupby("date").size().reset_index(name = "target")

## faers adverse event network
class FaersProcessor:
    def __init__(self, id: str, url: str):
        self.id = id
        self.url = url
        self.data_network: Optional[pd.DataFrame] = None
        self.data_events: Optional[pd.DataFrame] = None
        self.graph: Optional[Any] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.features: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_network = _load_network_faers(id = self.id, url = self.url)
        self.data_events = _load_events_faers(id = self.id, url = self.url)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_network is None:
            self.load_data()
        data_network = _process_network_faers(data = self.data_network)
        nodes, edges = _build_network_faers(data = data_network)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_descriptors(self):
        """Computes process descriptors over daily adverse events."""
        if self.events is None:
            self.process_events()
            self.features = ProcessDescriptors(
            data = self.events.copy(),
            sort_by = ["date"],
            target = "target"
        ).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_events is None:
            self.load_data()
        self.events = _process_events_faers(data = self.data_events)
        return self

    def run(self):
        """ Executes the pipeline and returns the final result. """
        self.process_network()
        self.process_descriptors()
        self.process_events()
        return {
            "invariants": self.invariants,
            "features": self.features,
            "events": self.events.to_dict(orient = "records")
        }
