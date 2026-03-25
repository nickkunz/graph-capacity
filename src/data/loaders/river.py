## libraries
import os
import sys
import requests
import numpy as np
import pandas as pd
import igraph as ig
from pathlib import Path
from io import StringIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

## path
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.data.helpers import (
    _create_igraph_object, 
    _aggregate_by_day,
    _request_with_retry
)

## create a requests session with a connection pool
def _create_session(max_workers: int, max_retries: int) -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections = max_workers,
        pool_maxsize = max_workers * 2,
        max_retries = max_retries,
    )
    session.mount("https://", adapter)
    return session

## find stations from USGS NWIS inventory
def _load_network_nwis(
    session: requests.Session,
    url: str,
    params: dict[str, str],
    timeout: int = 60
    ) -> pd.DataFrame:

    ## load data
    response = _request_with_retry(url = url, params = dict(params), timeout = timeout, use_cache = True)

    ## parse response
    if response.text.strip().lower().startswith("<!doctype html") or response.text.strip().lower().startswith("<html"):
        raise ValueError(
            "NWIS inventory endpoint returned HTML (likely deprecated/redirected endpoint or invalid params)."
        )

    lines = [l for l in response.text.split("\n") if not l.startswith("#") and l.strip()]
    if len(lines) < 3:
        raise ValueError("No station data returned.")

    ## build dataframe    
    data = pd.read_csv(StringIO("\n".join(lines)), sep = "\t")
    data = data.rename(columns = {c: c.strip() for c in data.columns})
    if "site_no" not in data.columns:
        raise ValueError(f"Expected 'site_no' column in NWIS response, got columns: {list(data.columns)}")

    data = data[data["site_no"].astype(str).str.strip().str.isnumeric()]

    ## output unique site ids
    return data["site_no"].unique().tolist()

## find station metadata from USGS NWIS inventory
def _load_network_nwis_metadata(
    session: requests.Session, 
    url: str, 
    ids: str,
    timeout: int = 10, match: str = "colorado river",
    exclude: tuple[str, ...] = ("little colorado", "bill williams", "verde", "gila", "havasu", "paria")
    ) -> Optional[tuple[str, dict[str, float]]]:

    ## send request
    params = {"format": "rdb", "sites": ids, "siteOutput": "expanded"}
    response = _request_with_retry(url = url, params = params, timeout = timeout, use_cache = True)

    ## parse response
    lines = [l for l in response.text.split("\n") if not l.startswith("#") and l.strip()]
    if len(lines) < 3:
        raise ValueError("No station data returned.")
    
    ## build geocoords
    hdr, val = lines[0].split("\t"), lines[2].split("\t")
    lat = float(val[hdr.index("dec_lat_va")])
    lon = float(val[hdr.index("dec_long_va")])
    raw = val[hdr.index("station_nm")].strip()

    ## normalize search criteria
    name = raw.lower()
    match = match.lower()
    exclude = tuple(e.lower() for e in exclude)

    ## filter to requested river segment
    if match in name and all(b not in name for b in exclude):
        return ids, {"lat": lat, "lon": lon, "name": name}

## execute metadata requests in parallel
def _execute_network_nwis(
    session: requests.Session,
    url: str,
    ids: list[str],
    timeout: int = 60,
    max_workers: int = 20
    ) -> dict[str, dict]:

    data: dict[str, dict] = {}
    if not ids:
        return data
    with ThreadPoolExecutor(max_workers = max_workers) as ex:
        futures = [ex.submit(_load_network_nwis_metadata, session, url, i, timeout) for i in ids]
        with tqdm(total = len(futures), desc = "Progress", unit = "site") as pbar:
            for i in as_completed(futures):
                try:
                    response = i.result()
                    if response:
                        site, meta = response
                        data[site] = meta
                except Exception:
                    pass
                pbar.update(1)
    return data

## reorder network to avoid edge crossings
def _build_network_nwis(coords: dict):

    ## extract points
    ids = list(coords.keys())
    points = np.array([[coords[s]["lat"], coords[s]["lon"]] for s in ids])
    n = len(ids)
    if n < 2:
        return [], ids

    ## haversine distance matrix
    def haversine(p1, p2):
        R = 6371.0
        lat1, lon1 = np.radians(p1)
        lat2, lon2 = np.radians(p2)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(points[i], points[j])
            distances[i, j] = distances[j, i] = d

    ## start at westernmost point
    start = int(np.argmin(points[:, 1]))

    ## greedy nearest-neighbor
    used = np.zeros(n, dtype = bool)
    path = [start]
    used[start] = True
    cur = start
    for _ in range(n - 1):
        j = np.ma.array(data = distances[cur], mask = used).argmin().item()
        path.append(j)
        used[j] = True
        cur = j

    ## remove crossings
    def segments_cross(a, b, c, d):
        def orient(p, q, r):
            return np.sign((q[1] - p[1]) * (r[0] - p[0]) - (q[0] - p[0]) * (r[1] - p[1]))
        return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)

    improved = True
    while improved:
        improved = False
        for i in range(n - 3):
            for k in range(i + 2, n - 1):
                a, b = points[path[i]], points[path[i + 1]]
                c, d = points[path[k]], points[path[k + 1]]
                if segments_cross(a, b, c, d):
                    path[i + 1:k + 1] = reversed(path[i + 1:k + 1])
                    improved = True

    ## create edges map from site ids
    nodes = [ids[i] for i in path]
    nodes = list(reversed(nodes))
    edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
    return edges, nodes

## fetch high-flow events for a single station
def _load_events_nwis(
    session: requests.Session, 
    url: str, 
    ids: str, 
    meta_data: dict,
    start_date: str,
    end_date: str,
    param_code: str = "00060",
    threshold: float = 0.99,
    timeout: int = 60
    ) -> tuple[str, list[dict]]:

    ## send request
    params = {
        "format": "json", 
        "sites": ids, 
        "startDT": start_date, 
        "endDT": end_date, 
        "parameterCd": param_code
    }
    response = _request_with_retry(url = url, params = params, timeout = timeout, use_cache = True)
    
    ## parse response
    data = response.json()
    if "value" not in data or "timeSeries" not in data["value"]:
        return None
    
    time = data["value"]["timeSeries"]
    if not time or not time[0].get("values"):
        return None
    
    ## build dataframe
    vals = time[0]["values"][0]["value"]
    data = pd.DataFrame(vals)
    data["dateTime"] = pd.to_datetime(data["dateTime"], utc = True)
    data["value"] = pd.to_numeric(data["value"], errors = "coerce")
    data = data.dropna(subset = ["value"])
    
    if data.empty:
        return None
    
    ## identify high-flow events (peaks over threshold)
    threshold = data["value"].quantile(threshold)

    ## boolean mask of values above threshold
    above = data["value"] > threshold

    ## find local maxima among consecutive above-threshold runs
    groups = (above != above.shift()).cumsum()
    peaks = (
        data[above]
        .groupby(groups[above])
        .apply(lambda g: g.loc[g["value"].idxmax()])
        .reset_index(drop = True)
    )

    ## format events
    events = [
        {
            "site_no": ids,
            "datetime": row["dateTime"],
            "discharge_cfs": row["value"],
            "lat": meta_data["lat"],
            "lon": meta_data["lon"]
        }
        for _, row in peaks.iterrows()
    ]
    
    return ids, events if events else None

## execute event requests in parallel
def _execute_events_nwis(
    session: requests.Session,
    url: str, 
    meta_data: dict[str, dict],
    start_date: str,
    end_date: str,
    param_code: str = "00060",
    threshold: float = 0.99,
    timeout: int = 60,
    max_workers: int = 20) -> pd.DataFrame:
    
    all_events: list[dict] = []
    if not meta_data:
        return pd.DataFrame()
    
    with ThreadPoolExecutor(max_workers = max_workers) as ex:
        futures = [
            ex.submit(_load_events_nwis, session, url, site_id, meta_data[site_id],
                      start_date, end_date, param_code, threshold, timeout)
            for site_id, _ in meta_data.items()
        ]
        
        with tqdm(total = len(futures), desc = "Progress", unit = "site") as pbar:
            for i in as_completed(futures):
                try:
                    response = i.result()
                    if response:
                        _, events = response
                        if events is not None:
                            for event in events:
                                all_events.append(event)
                except Exception:
                    pass
                pbar.update(1)
    
    return pd.DataFrame(all_events)
    

## usgs nwis network
class NwisProcessor:
    def __init__(self, url_site: str, url_iv: str, params: dict, start_date: str, end_date: str, max_workers: int = 20, max_retries: int = 3):
        self.url_site = url_site
        self.url_iv = url_iv
        self.params = params
        self.start_date = start_date
        self.end_date = end_date
        self.max_workers = max_workers
        self.session = _create_session(max_workers = max_workers, max_retries = max_retries)
        self.station_ids: Optional[list[str]] = None
        self.station_metadata: Optional[dict] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.signatures: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.station_ids = _load_network_nwis(
            session=self.session,
            url=self.url_site,
            params=self.params
        )
        self.station_metadata = _execute_network_nwis(
            session=self.session,
            url=self.url_site,
            ids=self.station_ids,
            max_workers=self.max_workers
        )
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.station_metadata is None:
            self.load_data()
        
        edges, nodes = _build_network_nwis(coords=self.station_metadata)
        self.graph = _create_igraph_object(nodes=nodes, edges=edges)
        self.invariants = GraphInvariants(graph=self.graph).all()
        return self

    def process_signatures(self):
        """Computes process signatures on daily high-flow events."""
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
        if self.station_metadata is None:
            self.load_data()

        data_events = _execute_events_nwis(
            session=self.session,
            url=self.url_iv,
            meta_data=self.station_metadata,
            start_date=self.start_date,
            end_date=self.end_date,
            max_workers=self.max_workers
        )
        self.events = _aggregate_by_day(
            data = data_events, 
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
            "events": self.events.to_dict(orient = "records")
        }
