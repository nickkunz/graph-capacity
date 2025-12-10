## libraries
import os
import sys
import time
import requests
import pandas as pd
import igraph
from typing import Optional, Dict, Any, Iterator

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import _create_igraph_object, _aggregate_by_day, _request_with_retry
from src.invariants import GraphInvariants
from src.descriptors import ProcessDescriptors

## load network contracts from endpoint
def _post_network_federal(
    url: str, 
    start_date: str, 
    end_date: str, 
    page: int = 1, 
    page_size: int = 100,
    keyword: str | None = None) -> dict:
    json = {
        "filters": {
            "time_period": [{"start_date": start_date, "end_date": end_date}],
            "award_type_codes": ["A", "B", "C", "D"]  ## federal procurement contracts
        },
        "fields": [
            "Award ID",
            "Recipient Name",
            "Start Date",
            "End Date",
            "Award Amount",
            "Awarding Agency",
            "Awarding Sub Agency",
            "recipient_id"
        ],
        "page": page,  ## pagination
        "limit": page_size,  ## limit results
        "sort": "Award Amount",  ## sort by award amount
        "order": "desc"  ## descending order
    }
    if keyword:
        json["filters"]["keywords"] = [keyword]
    try:
        response = _request_with_retry(url = url, method = 'POST', json = json)
        return response.json()
    except Exception as e:
        raise RuntimeError(f"Failed POST request to {url}: {str(e)}")

## iterate over network contracts with pagination
def _iter_network_federal(
        url: str, 
        start_date: str, 
        end_date: str,
        page_size: int = 100,
        max_pages: int = 1_000, 
        max_records: int = 10_000, 
        keyword: str | None = None, 
        sleep: float = 0.5) -> Iterator[dict]:
    fetched_records = 0
    current_page = 1
    while current_page <= max_pages:
        json_data = _post_network_federal(
            url = url,
            start_date = start_date,
            end_date = end_date,
            page = current_page,
            page_size = page_size,
            keyword = keyword
        )
        results = json_data.get("results", [])
        if not results:
            break
        for rec in results:
            yield rec
            fetched_records += 1
            if fetched_records >= max_records:
                return
        meta_data = json_data.get("page_metadata") or {}
        has_next = meta_data.get("hasNext", False)
        if not has_next:
            break
        current_page += 1
        if sleep > 0:
            time.sleep(sleep)

## load network contracts
def load_network_federal(
    url: str, 
    start_date: str, 
    end_date: str, 
    max_pages: int = 100, 
    max_records: int = 10_000, 
    keyword: str | None = None) -> pd.DataFrame:
    data = list(_iter_network_federal(
        url = url,
        start_date = start_date, 
        end_date = end_date, 
        max_pages = max_pages,
        max_records = max_records, 
        keyword = keyword)
    )
    if not data:
        return pd.DataFrame()
    else:
        return pd.DataFrame(data)
    
## process network contracts
def process_network_federal(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.copy()
        .query("`Award Amount` > 0")
        .assign(
            **{
                "Start Date": lambda df: pd.to_datetime(df["Start Date"], errors = "coerce"),
                "End Date": lambda df: pd.to_datetime(df["End Date"], errors = "coerce"),
                "Recipient Name": lambda df: df["Recipient Name"].str.strip().str.upper(),
                "Awarding Agency": lambda df: df["Awarding Agency"].str.strip().str.upper()
            }
        )
    )

## build network from contracts
def build_network_federal(data: pd.DataFrame) -> tuple[list[str], list[tuple[str, str]]]:

    ## make copy to avoid erroring on original data
    data_copy = data.copy()

    ## create recipient node id (use recipient_id if available, else recipient name)
    data_copy["recipient_node_id"] = data_copy["recipient_id"].fillna(data_copy["Recipient Name"])

    ## unique agencies and recipients
    agencies = data_copy["Awarding Agency"].unique()
    recipients = data_copy["recipient_node_id"].unique()

    ## create nodes
    agency_nodes = [f"agency::{name}" for name in agencies]
    recipient_nodes = [f"recipient::{name}" for name in recipients]
    nodes = agency_nodes + recipient_nodes

    ## create edges
    edge_pairs = data_copy[["Awarding Agency", "recipient_node_id"]].drop_duplicates()
    edges = [
        (f"agency::{row['Awarding Agency']}", f"recipient::{row['recipient_node_id']}")
        for _, row in edge_pairs.iterrows()
    ]
    
    return nodes, edges

## process federal contract event counts
def process_events_federal(data: pd.DataFrame) -> pd.DataFrame:
    if "Start Date" not in data.columns:
        raise ValueError("Input DataFrame must contain 'Start Date' column.")
    return (
        data[["Start Date"]]
        .copy()
        .rename(columns={"Start Date": "datetime"})
    )

## federal agency–recipient network
class FederalProcessor:
    def __init__(self, url: str, start_date: str, end_date: str, keyword: str = "waterfowl"):
        self.url: str = url
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.keyword: str = keyword
        self.data_raw: Optional[pd.DataFrame] = None
        self.data_processed: Optional[pd.DataFrame] = None
        self.graph: Optional[igraph.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.descriptors: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_raw = load_network_federal(
            url = self.url,
            start_date = self.start_date,
            end_date = self.end_date,
            keyword = self.keyword
        )
        return self

    def process_data(self):
        """ Processes the raw data. """
        if self.data_raw is None:
            self.load_data()
        self.data_processed = process_network_federal(data = self.data_raw)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_processed is None:
            self.process_data()
        nodes, edges = build_network_federal(data = self.data_processed)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_processed is None:
            self.process_data()
        events = process_events_federal(data = self.data_processed)
        self.events = _aggregate_by_day(
            data = events, 
            datetime = "datetime",
            label = "date"
        )
        return self

    def process_descriptors(self):
        """Computes process descriptors over daily contract awards."""
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
