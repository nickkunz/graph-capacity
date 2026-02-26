## libraries
import os
import sys
import logging
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any
from io import StringIO
import igraph as ig

## logging
logger = logging.getLogger(__name__)

## path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

## modules
from src.data.helpers import _create_igraph_object, _aggregate_by_day
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures

## xml data loader
def _load_network_seismic(url: str, params: dict, namespace: dict, row_path: str, col_map: dict, timeout: int = 60) -> pd.DataFrame:

    ## fetch and parse xml data
    response = requests.get(url = url, params = params, timeout = timeout)
    response.raise_for_status()
    root = ET.fromstring(text = response.content)
    
    ## find all parent elements to become rows
    records = []
    for item in root.findall(row_path, namespace):
        record = {}
        for col_name, path in col_map.items():
            value = None
            try:
                if "@" in path:
                    ## extract an attribute
                    tag_path, attr_name = path.split('@')
                    element = item.find(tag_path, namespace)
                    if element is not None:
                        value = element.get(attr_name)
                else:
                    ## extract element text
                    element = item.find(path, namespace)
                    if element is not None:
                        value = element.text
            except Exception:
                pass  ## none if parsing fails
            record[col_name] = value
        records.append(record)
        
    return pd.DataFrame(records)

## process iris station data
def _process_network_seismic(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data
        .assign(
            lat = lambda d: pd.to_numeric(d['lat'], errors = 'coerce'),
            lon = lambda d: pd.to_numeric(d['lon'], errors = 'coerce')
        )
        .dropna(subset = ['lat', 'lon'])
        .sort_values('code')  ## deterministic ordering
        .drop_duplicates(subset = ['code'], keep = 'first')
        .drop_duplicates(subset = ['lat', 'lon'], keep = 'first')
        .reset_index(drop = True)
    )

## build iris station network
def _build_network_seismic(data: pd.DataFrame) -> tuple[list[tuple], list[str]]:
    
    ## extract coordinates
    nodes = data['code'].astype(str).tolist()
    n = len(nodes)
    if n < 2:
        return [], []

    ## create edges with all possible pairs
    edges = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]
    if nodes and edges:
        return nodes, edges

## load iris seismic events from usgs feed
def _load_events_seismic(params: dict, url: str = "https://earthquake.usgs.gov/fdsnws/event/1/query") -> pd.DataFrame:

    if 'format' not in params:
        params['format'] = 'csv'

    if params['format'] != 'csv':
        raise ValueError("This function is designed to parse CSV format. Please use format='csv'.")

    ## check if date range spans more than a month
    if 'starttime' in params and 'endtime' in params:
        start_date = pd.to_datetime(params['starttime'])
        end_date = pd.to_datetime(params['endtime'])
        
        ## if date range is longer than 31 days, split into monthly chunks
        if (end_date - start_date).days > 31:
            all_data = []
            
            ## generate month-by-month date ranges
            current_date = start_date
            while current_date < end_date:
                ## calculate end of current month chunk
                next_date = current_date + pd.DateOffset(months=1)
                chunk_end = min(next_date, end_date)
                
                ## create params for this chunk
                chunk_params = params.copy()
                chunk_params['starttime'] = current_date.strftime('%Y-%m-%d')
                chunk_params['endtime'] = chunk_end.strftime('%Y-%m-%d')
                
                ## fetch data for this chunk
                try:
                    response = requests.get(url = url, params=chunk_params)
                    response.raise_for_status()
                    chunk_data = pd.read_csv(StringIO(response.text))
                    if not chunk_data.empty:
                        all_data.append(chunk_data)
                except requests.exceptions.RequestException as e:
                    logger.error(f"An error occurred for {chunk_params['starttime']} to {chunk_params['endtime']}: {e}")
                
                ## move to next month
                current_date = next_date
            
            ## concatenate all chunks
            if all_data:
                return pd.concat(all_data, ignore_index=True)
            else:
                return pd.DataFrame()
    
    ## if date range is 31 days or less, make a single request
    try:
        response = requests.get(url = url, params=params)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred during the request: {e}")
        return pd.DataFrame()

## process iris seismic events, convert time and magnitude
def _process_events_seismic(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data
        .assign(
            datetime = lambda d: pd.to_datetime(d['time'], errors = 'coerce', utc = True),
            magnitude = lambda d: pd.to_numeric(d['mag'], errors = 'coerce')
        )
        .loc[lambda d: d['status'] == 'reviewed']  ## only include reviewed events
        .dropna(subset = ['datetime'])
        .sort_values('datetime')
        .reset_index(drop = True)
    )


## parameters for seismic data
url = "https://service.iris.edu/fdsnws/station/1/query"
params = {"level": "station", "format": "xml", "network": "IU"}
namespace = {"ns": "http://www.fdsn.org/xml/station/1"}

## path to repeating row elements
row_path = ".//ns:Station"

## mapping of column names to data within each station
col_map = {
    "code": ".@code",  ## code attribute of the station tag
    "lat": ".//ns:Latitude",  ## text of the latitude tag
    "lon": ".//ns:Longitude"  ## text of the longitude tag
}

## seismic event network
class SeismicProcessor:
    def __init__(self, url_network: str, params_network: dict, namespace: dict, row_path: str, col_map: dict, url_events: str, params_events: dict):
        self.url_network = url_network
        self.params_network = params_network
        self.namespace = namespace
        self.row_path = row_path
        self.col_map = col_map
        self.url_events = url_events
        self.params_events = params_events
        self.data_network_raw: Optional[pd.DataFrame] = None
        self.data_events_raw: Optional[pd.DataFrame] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.signatures: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_network_raw = _load_network_seismic(
            url=self.url_network,
            params=self.params_network,
            namespace=self.namespace,
            row_path=self.row_path,
            col_map=self.col_map
        )
        self.data_events_raw = _load_events_seismic(params=self.params_events, url=self.url_events)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_network_raw is None:
            self.load_data()
        
        data_processed = _process_network_seismic(data=self.data_network_raw)
        nodes, edges = _build_network_seismic(data=data_processed)
        
        self.graph = _create_igraph_object(nodes=nodes, edges=edges)
        self.invariants = GraphInvariants(graph=self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_events_raw is None:
            self.load_data()
            
        data_events = _process_events_seismic(data=self.data_events_raw)
        self.events = _aggregate_by_day(
            data = data_events,
            datetime = 'datetime',
            label = 'date'
        )
        return self

    def process_signatures(self):
        """Computes process signatures over daily seismic events."""
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
            "graph": self.graph,
            "events": self.events.to_dict(orient="records")
        }

