## libraries
import os
import sys
import igraph
import pandas as pd
from typing import Optional, Dict, Any

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import _aggregate_by_day, _create_igraph_object, _request_with_retry
from src.invariants import GraphInvariants
from src.descriptors import ProcessDescriptors

## load world bank project data
def load_network_worldbank(url: str, start_year: int, end_year: int) -> pd.DataFrame:

    ## base query params
    base = {
        "format": "json",
        "frmYear": int(start_year),
        "toYear": int(end_year),
        "fl": "lendinginstr, totalamt, countryshortname, boardapprovaldate",
    }

    ## paginate until all records fetched   
    data = list()
    offset, rows = 0, 2000
    while True:
        params = {**base, "rows": rows, "os": offset}
        response  = _request_with_retry(url = url, params = params)
        payload = response.json()
        projects = payload.get("projects", {})
        if not projects:
            break
            
        ## accumulate results
        data.extend(projects.values())
        total = int(payload.get("total", 0))
        offset += rows
        if offset >= total:
            break
    if not data:
        raise RuntimeError("no projects returned for the specified window")
    return pd.DataFrame(data)

## load world bank country metadata
def load_metadata_worldbank(url: str, timeout: int = 60) -> pd.DataFrame:
    response = _request_with_retry(url = url, params = {"format": "json"}, timeout = timeout)
    json = response.json()
    if not isinstance(json, list) or len(json) < 2:
        raise RuntimeError("unexpected response structure from world bank country api")
    records = json[1]
    return pd.json_normalize(records)

## process world bank project data
def process_network_worldbank(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.copy()
        .explode("countryshortname")
        .assign(
            totalamt = lambda df: pd.to_numeric(
                arg = df["totalamt"].astype(str).str.replace(",", ""), 
                errors = "coerce"
            ),
            boardapprovaldate = lambda df: pd.to_datetime(
                arg = df["boardapprovaldate"], 
                errors = "coerce"
            ).dt.strftime("%Y-%m-%d"),
        )
        .dropna(
            subset = [
                "countryshortname", 
                "lendinginstr", 
                "boardapprovaldate", 
                "totalamt"
            ]
        )
        .query(
            "countryshortname != '' and "
            "lendinginstr != '' and "
            "boardapprovaldate != '' and "
            "totalamt > 0"
        )
    )

## all comments lower case
def build_network_worldbank(data: pd.DataFrame, data_meta: pd.DataFrame) -> tuple[list[str], list[tuple[str, str]]]:

    ## validate presence of required columns
    if not {'lendinginstr', 'countryshortname'}.issubset(data.columns):
        raise ValueError("data must contain 'lendinginstr' and 'countryshortname'")
    if not {'name', 'lendingType.id', 'incomeLevel.id'}.issubset(data_meta.columns):
        raise ValueError("data_meta must contain 'name', 'lendingType.id', and 'incomeLevel.id'")

    ## extract unique nodes
    instruments = data['lendinginstr'].dropna().unique()
    countries = data['countryshortname'].dropna().unique()

    instrument_nodes = [f"instrument::{x}" for x in instruments]
    country_nodes = [f"country::{x}" for x in countries]
    nodes = instrument_nodes + country_nodes

    ## prepare simplified eligibility metadata
    meta = (
        data_meta[['name', 'lendingType.id', 'incomeLevel.id']]
        .rename(columns={'name': 'countryshortname'})
        .drop_duplicates(subset='countryshortname')
    )
    meta['ida_eligible'] = meta['lendingType.id'].isin(['IDX', 'IDB'])
    meta['ibrd_eligible'] = meta['lendingType.id'].isin(['IBR', 'IBD'])
    meta['blend_eligible'] = meta['lendingType.id'].isin(['IDB', 'IBR'])
    meta['income_group'] = meta['incomeLevel.id'].fillna('NA')

    ## helper to determine eligibility for a given instrument
    def eligible(instr: str, row) -> bool:
        instr_l = instr.lower()
        if "ida" in instr_l and not row['ida_eligible']:
            return False
        if "ibrd" in instr_l and not row['ibrd_eligible']:
            return False
        if "blend" in instr_l and not row['blend_eligible']:
            return False
        
        ## income group constraints (optional heuristics)
        if "high" in instr_l and row['income_group'] != 'HIC':
            return False
        if "low" in instr_l and row['income_group'] not in ['LIC', 'LMC']:
            return False
        return True

    ## restrict metadata to countries appearing in data
    meta = meta.loc[meta['countryshortname'].isin(countries)]

    ## generate edges based on rules instead of realized co-occurrence
    edges = []
    for _, c_row in meta.iterrows():
        for instr in instruments:
            if eligible(instr, c_row):
                edges.append((f"instrument::{instr}", f"country::{c_row['countryshortname']}"))

    return nodes, edges

## world bank projects network
class WorldBankProcessor:
    def __init__(self, url_projects, url_meta, start_year, end_year):
        self.url_projects: str = url_projects
        self.url_meta: str = url_meta
        self.start_year: str = start_year
        self.end_year: str = end_year
        self.data_raw: Optional[pd.DataFrame] = None
        self.data_meta: Optional[pd.DataFrame] = None
        self.data_processed: Optional[pd.DataFrame] = None
        self.graph: Optional[igraph.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.features: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_raw = load_network_worldbank(
            url = self.url_projects,
            start_year = self.start_year,
            end_year = self.end_year
        )
        self.data_meta = load_metadata_worldbank(
            url = self.url_meta
        )
        return self

    def process_data(self):
        """ Processes the raw data. """
        if self.data_raw is None or self.data_meta is None:
            self.load_data()
        self.data_processed = process_network_worldbank(
            data = self.data_raw
        )
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_processed is None:
            self.process_data()
        nodes, edges = build_network_worldbank(
            data = self.data_processed, 
            data_meta = self.data_meta
        )
        self.graph = _create_igraph_object(
            nodes = nodes,
            edges = edges
        )
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_processed is None:
            self.process_data()
        self.events = _aggregate_by_day(
            data = self.data_processed,
            datetime = 'boardapprovaldate',
            label = 'date'
        )
        return self

    def process_descriptors(self):
        """Computes process descriptors over daily project approvals."""
        if self.events is None:
            self.process_events()
        self.features = ProcessDescriptors(
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
            "features": self.features,
            "events": self.events.to_dict(orient = "records")
        }
