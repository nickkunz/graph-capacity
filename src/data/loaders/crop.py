## libraries
import os
import sys
import os
import certifi
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import _aggregate_by_day, _create_igraph_object
from src.invariants import GraphInvariants
from src.features import ProcessFeatures

## helper to load network data
def _load_network_data(url: str, cols: list[str], error_msg: str, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    os.environ['SSL_CERT_FILE'] = certifi.where()
    data = pd.read_csv(filepath_or_buffer = url, usecols = cols, dtype = dtype)
    if data.empty:
        raise RuntimeError(error_msg)
    return data

## load croppol network data
def _load_network_croppol(url_sampling: str, url_field: str) -> pd.DataFrame:
    try:
        data_left = _load_network_data(
            url = url_sampling,
            cols = ["study_id", "site_id", "pollinator", "abundance"],
            error_msg = "No croppol sampling data found."
        )
        data_right = _load_network_data(
            url = url_field,
            cols = ["site_id", "crop"],
            error_msg = "No croppol field data found."
        )
        data = pd.merge(
            left = data_left, 
            right = data_right, 
            on = "site_id"
        )
        if data.empty:
            raise RuntimeError("Merged CropPol data is empty.")
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading and merging CropPol data: {e}")

## process croppol network data
def _process_network_croppol(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.rename(columns = str.lower)
        .dropna(subset = ["site_id", "pollinator", "abundance", "crop"])
        .assign(abundance = lambda d: pd.to_numeric(d["abundance"], errors = "coerce"))
        .dropna(subset = ["abundance"])
        .reset_index(drop = True)
    )

## build croppol network data
def build_network_croppol(data: pd.DataFrame) -> tuple[list[str], list[tuple]]:
    
    ## extract unique nodes (crops and pollinators)
    crops = data["crop"].dropna().unique().tolist()
    polls = data["pollinator"].dropna().unique().tolist()
    nodes = list(set(crops + polls))

    ## extract unique edges (crop-pollinator pairs)
    edges = data[["crop", "pollinator"]].dropna().drop_duplicates()
    edges = [tuple(x) for x in edges.to_numpy()]
    return nodes, edges

## load croppol events
def _load_events_croppol(url_sampling: str, url_field: str) -> pd.DataFrame:
    try:
        data_left = _load_network_data(
            url = url_sampling,
            cols = ["site_id", "pollinator", "abundance", "total_sampled_time"],
            error_msg = "No croppol sampling data found."
        )
        data_right = _load_network_data(
            url = url_field,
            cols = ['site_id', 'crop', 'sampling_year', 'sampling_start_month', 'sampling_end_month', 'use_visits_or_abundance'],
            error_msg = "No croppol field data found.",
            dtype = {
                'sampling_start_month': 'Int64', 
                'sampling_end_month': 'Int64'
            }
        )
        data = pd.merge(
            left = data_left, 
            right = data_right, 
            on = "site_id"
        )
        if data.empty:
            raise RuntimeError("Merged CropPol data is empty.")
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading and merging CropPol data: {e}")

## process croppol events
def _process_events_croppol(data: pd.DataFrame) -> pd.DataFrame:
    """Clean CropPol event data using a chained pandas pipeline and return raw daily rows.

    This version stops before day-level aggregation so that aggregation can be performed
    externally. Each returned row represents the first day of the sampled month with its
    discrete abundance count (as `target`). Duplicate days may exist and should be
    aggregated downstream if desired.

    Logic preserved from prior implementation:
    - keep only complete temporal observations
    - restrict to daily windows (<= 24 hours)
    - require same start/end month and a clean numeric year
    - keep discrete counts (non-rate), non-negative integer abundance
    - construct day as first day of (year, month)
    """

    return (
        data.copy()
        ## compute helper columns and coerce types once
        .assign(
            sampling_hours=lambda d: pd.to_numeric(d['total_sampled_time'], errors='coerce') / 60,
            sampling_year=lambda d: pd.to_numeric(d['sampling_year'], errors='coerce'),
            sampling_start_month=lambda d: pd.to_numeric(d['sampling_start_month'], errors='coerce'),
            sampling_end_month=lambda d: pd.to_numeric(d['sampling_end_month'], errors='coerce'),
            abundance=lambda d: pd.to_numeric(d['abundance'], errors='coerce'),
            use_visits_or_abundance=lambda d: d['use_visits_or_abundance'].astype(str)
        )
        ## omit incomplete temporal observations
        .dropna(subset=['sampling_start_month', 'sampling_end_month', 'total_sampled_time'])
        ## filter for daily events only (<= 24 hours)
        .query('sampling_hours <= 24')
        ## restrict to same-month intervals and clean year data (non-numeric years become NaN)
        .query('sampling_start_month == sampling_end_month')
        .dropna(subset=['sampling_year'])
        ## keep discrete, non-negative integer abundance and explicit abundance usage
        .pipe(lambda d: d[
            d['abundance'].notna()
            & np.isclose(d['abundance'], np.rint(d['abundance']), atol=1e-9)
            & (d['abundance'] >= 0)
            & d['use_visits_or_abundance'].str.contains('abundance', case=False, na=False)
        ])
        ## construct day as first of month
        .assign(
            day=lambda d: pd.to_datetime(
                dict(
                    year=d['sampling_year'].astype(int),
                    month=d['sampling_start_month'].astype(int),
                    day=1
                )
            ).dt.date
        )
        ## keep only required columns
        .loc[:, ['day', 'abundance']]
        .rename(columns={'abundance': 'target'})
        .astype({'target': 'int'})
    )

## crop pollinator network
class CropProcessor:
    def __init__(self, url_sampling: str, url_field: str):
        self.url_sampling = url_sampling
        self.url_field = url_field
        self.data_network: Optional[pd.DataFrame] = None
        self.data_events: Optional[pd.DataFrame] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None
        self.features: Optional[Dict[str, Any]] = None

    def load_data(self):
        """Loads the raw data from source."""
        if self.data_network is None:
            self.data_network = _load_network_croppol(self.url_sampling, self.url_field)
            self.data_events = _load_events_croppol(self.url_sampling, self.url_field)
        return self

    def process_network(self):
        """Cleans raw data, builds an undirected crop–pollinator interaction graph, and computes general graph invariants."""
        if self.data_network is None:
            self.load_data()
        data_network = _process_network_croppol(data = self.data_network)
        nodes, edges = build_network_croppol(data = data_network)
        graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph).all()
        return self

    def process_features(self):
        """Computes process features over the daily event counts."""
        if self.events is None:
            self.process_events()

        self.features = ProcessFeatures(
            data = self.events.copy(),
            sort_by = ["date"],
            target = "target"
        ).all()
        return self

    def process_events(self):
        """Processes the event data."""
        if self.data_network is None:
            self.load_data()
        self.events = _process_events_croppol(data = self.data_events)
        self.events = _aggregate_by_day(
            data = self.events,
            datetime = 'day',
            label = 'date'
        )
        return self

    def run(self):
        """Executes the pipeline and returns the final result."""
        self.process_network()
        self.process_features()
        self.process_events()
        return {
            "invariants": self.invariants,
            "features": self.features,
            "events": self.events.to_dict(orient = "records")
        }
