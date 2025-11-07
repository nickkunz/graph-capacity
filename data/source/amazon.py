## libraries
import os
import sys
import gzip
import requests
import pandas as pd
from io import BytesIO

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import _create_igraph_object, _aggregate_by_day
from src.invariants import GraphInvariants

## load amazon network
def load_network_amazon(url: str, root_path: str, name: str, timeout: int = 30) -> pd.DataFrame:
    
    ## configure path
    file_name = os.path.basename(url)
    path_dir = os.path.join(root_path, name)
    path_local = os.path.join(path_dir, file_name)
    os.makedirs(name = path_dir, exist_ok = True)

    ## load from disk or download
    if os.path.exists(path = path_local):
        with gzip.open(path_local, 'rt', encoding = 'latin-1') as f:
            content = f.read()
    else:
        try:
            response = requests.get(url = url, stream = True, timeout = timeout)
            response.raise_for_status()
            with open(path_local, 'wb') as f:
                f.write(response.content)
            with gzip.GzipFile(fileobj = BytesIO(response.content)) as gz:
                content = gz.read().decode(encoding = 'latin-1')
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse data from {url}: {str(e)}")

    ## parse reviews
    try:
        reviews = list()
        current_review = dict()
        for line in content.split('\n'):
            if line.strip():
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    if 'product/productId' in key:
                        current_review['product_id'] = value
                    elif 'review/userId' in key:
                        current_review['user_id'] = value
                    elif 'review/score' in key:
                        current_review['rating'] = float(value)
                    elif 'review/time' in key:
                        current_review['timestamp'] = int(value)
                    elif 'review/helpfulness' in key:
                        current_review['helpfulness'] = value
            else:
                if current_review and all(k in current_review for k in ['product_id', 'user_id', 'rating', 'timestamp']):
                    reviews.append(current_review)
                current_review = {}
        return pd.DataFrame(reviews)
    except Exception as e:
        raise RuntimeError(f"Failed to parse reviews from data: {str(e)}")

## build amazon network
def build_network_amazon(data: pd.DataFrame) -> tuple[list[str], list[tuple[str, str]]]:
    
    ## validate input
    if not {'user_id', 'product_id'}.issubset(data.columns):
        raise ValueError("DataFrame must contain 'user_id' and 'product_id' columns.")

    ## unique users and prods
    users = data['user_id'].unique()
    prods = data['product_id'].unique()

    ## create prefixed node lists for clarity
    user_nodes = [f"user::{u}" for u in users]
    prod_nodes = [f"product::{p}" for p in prods]
    nodes = user_nodes + prod_nodes

    ## create edges from unique user-product pairs
    edge_pairs = data[['user_id', 'product_id']].drop_duplicates()
    edges = [
        (f"user::{row['user_id']}", f"product::{row['product_id']}")
        for _, row in edge_pairs.iterrows()
    ]
    
    return nodes, edges

## process amazon review event counts
def process_events_amazon(data: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in data.columns:
        raise ValueError("Input DataFrame must contain 'timestamp' column.")

    return (
        data[["timestamp"]]
        .copy()
        .assign(datetime = lambda df: pd.to_datetime(df["timestamp"], unit = "s", utc = True))
        [["datetime"]]
    )

## amazon user-product network
class AmazonProcessor:
    def __init__(self, root_path: str, url: str, name: str):
        self.root_path = root_path
        self.url = url
        self.name = name
        self.data_raw = None
        self.graph = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_raw = load_network_amazon(url = self.url, root_path = self.root_path, name = self.name)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_raw is None:
            self.load_data()
        nodes, edges = build_network_amazon(data = self.data_raw)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_raw is None:
            self.load_data()
        events = process_events_amazon(data = self.data_raw)
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
