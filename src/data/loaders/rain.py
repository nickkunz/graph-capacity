## libraries
import os
import sys
import certifi
import warnings
import igraph as ig
import pandas as pd
from typing import Optional, Dict, Any
from meteostat import Stations, Hourly
from scipy.spatial import Delaunay

## path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

## modules
from src.data.helpers import _create_igraph_object, _aggregate_by_day
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures

## load meteostat weather station data
def _load_network_rain(start: pd.Timestamp, end: pd.Timestamp, country: str = None) -> pd.DataFrame:

    ## validate date range
    a = pd.Timestamp(start) if not isinstance(start, pd.Timestamp) else start
    b = pd.Timestamp(end) if not isinstance(end, pd.Timestamp) else end

    ## load station data
    st = Stations()
    st = st.inventory(freq = 'hourly', required = (a, b))
    data = st.fetch()

    ## filter by country if provided
    ## filter by country if provided
    if country:
        if not isinstance(country, str) or len(country) != 2:
            raise ValueError("The 'country' argument must be a two-character string (e.g., 'US', 'SG', etc.).")
        data = data[data['country'] == country.upper()]

    if data.empty:
        raise RuntimeError("No stations found with hourly inventory in the specified window.")
    return data.reset_index().rename(columns = {'id': 'station'})

## process meteostat weather station data
def _process_network_rain(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data
        .assign(
            latitude = lambda d: pd.to_numeric(d['latitude'], errors = 'coerce'),
            longitude = lambda d: pd.to_numeric(d['longitude'], errors = 'coerce')
        )
        .dropna(subset = ['latitude', 'longitude'])
        .sort_values('station')
        .drop_duplicates(subset = ['latitude', 'longitude'], keep = 'first')
        .reset_index(drop = True)
    )

## build meteostat weather station network
def _build_network_rain(data: pd.DataFrame) -> tuple[list[str], list[tuple]]:

    ## extract coordinates
    nodes = data['station'].astype(str).tolist()

    ## fallback for too few points
    n = len(nodes)
    if n < 3:
        return nodes, [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]

    points = data[['longitude', 'latitude']].to_numpy()
    triangles = Delaunay(points)

    ## create edges from triangulation simplices
    edges = set()
    for simplex in triangles.simplices:
        edges.add(tuple(sorted((nodes[simplex[0]], nodes[simplex[1]]))))
        edges.add(tuple(sorted((nodes[simplex[1]], nodes[simplex[2]]))))
        edges.add(tuple(sorted((nodes[simplex[2]], nodes[simplex[0]]))))
            
    return nodes, list(edges)

## load meteostat weather events for given stations and date range
def _load_events_rain(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:

    ## validate date range
    a = pd.Timestamp(start) if not isinstance(start, pd.Timestamp) else start
    b = pd.Timestamp(end) if not isinstance(end, pd.Timestamp) else end

    ## load hourly event data for specified stations and date range
    ids = data['station'].tolist()

    ## suppress runtime warning from meteostat fetch
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")
        obs = Hourly(ids, a, b).fetch()
    if obs.empty:
        raise RuntimeError("No hourly data returned.")

    ## pre-process timestamps
    obs = obs.reset_index().rename(columns = {'time': 'timestamp'})
    obs['timestamp'] = obs['timestamp'].dt.tz_localize('UTC')

    return obs

## process meteostat events for precipitation > 1mm
def _process_events_rain(data: pd.DataFrame) -> pd.DataFrame:
    return data[data['prcp'] > 1.0].copy()

class RainProcessor:
    def __init__(self, country: str, start_date: str, end_date: str):
        self.country = country
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.data_network_raw: Optional[pd.DataFrame] = None
        self.data_events_raw: Optional[pd.DataFrame] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.signatures: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        os.environ['SSL_CERT_FILE'] = certifi.where()
        self.data_network_raw = _load_network_rain(
            start=self.start_date,
            end=self.end_date,
            country=self.country
        )
        self.data_events_raw = _load_events_rain(
            data=self.data_network_raw,
            start=self.start_date,
            end=self.end_date
        )
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_network_raw is None:
            self.load_data()
        
        data_processed = _process_network_rain(data=self.data_network_raw)
        nodes, edges = _build_network_rain(data=data_processed)
        
        self.graph = _create_igraph_object(nodes=nodes, edges=edges)
        self.invariants = GraphInvariants(graph=self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_events_raw is None:
            self.load_data()
            
        data_events = _process_events_rain(data=self.data_events_raw)
        self.events = _aggregate_by_day(
            data = data_events, 
            datetime = 'timestamp',
            label = 'date'
        )
        return self

    def process_signatures(self):
        """Computes process signatures over daily precipitation events."""
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