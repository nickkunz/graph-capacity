## libraries
import os
import sys
import gzip
import pandas as pd
from io import BytesIO, StringIO

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.data.utilities import (
    _create_igraph_object, 
    _aggregate_by_day,
    _request_with_retry
)

## load amazon network
def _decode_amazon_content(content_bytes: bytes) -> str:
    try:
        if content_bytes.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj = BytesIO(content_bytes)) as gz:
                return gz.read().decode(encoding = 'latin-1')
        return content_bytes.decode(encoding = 'latin-1')
    except Exception as e:
        raise RuntimeError(f"Failed to decode Amazon review content: {str(e)}")

def _parse_amazon_text_reviews(content: str) -> pd.DataFrame:
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
    if current_review and all(k in current_review for k in ['product_id', 'user_id', 'rating', 'timestamp']):
        reviews.append(current_review)
    return pd.DataFrame(reviews)

def _parse_amazon_csv_reviews(content: str) -> pd.DataFrame:
    def _read_csv(**kwargs):
        return pd.read_csv(StringIO(content), **kwargs)
    try:
        df = _read_csv()
    except Exception:
        df = _read_csv(header = None, names = ['user_id', 'product_id', 'rating', 'timestamp'])
    required_cols = {'user_id', 'product_id', 'rating', 'timestamp'}
    if not required_cols.issubset(df.columns):
        df = _read_csv(header = None, names = ['user_id', 'product_id', 'rating', 'timestamp'])
    else:
        df = df[['user_id', 'product_id', 'rating', 'timestamp']]

    df = df.dropna(subset = ['user_id', 'product_id'])
    df['rating'] = pd.to_numeric(df['rating'], errors = 'coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors = 'coerce')
    df = df.dropna(subset = ['rating', 'timestamp'])
    df['rating'] = df['rating'].astype(float)
    df['timestamp'] = df['timestamp'].astype(int)
    return df

def _parse_amazon_reviews(content: str) -> pd.DataFrame:
    if 'product/productId' in content and 'review/userId' in content:
        return _parse_amazon_text_reviews(content = content)
    return _parse_amazon_csv_reviews(content = content)

def _load_network_amazon(url: str, root_path: str, name: str, timeout: int = 30) -> pd.DataFrame:
    file_name = os.path.basename(url)
    path_dir = os.path.join(root_path, name)
    path_local = os.path.join(path_dir, file_name)
    os.makedirs(name = path_dir, exist_ok = True)

    ## load from disk or download
    try:
        if os.path.exists(path = path_local):
            with open(path_local, 'rb') as f:
                content_bytes = f.read()
        else:
            response = _request_with_retry(url = url, params = {}, timeout = timeout)
            content_bytes = response.content
            with open(path_local, 'wb') as f:
                f.write(content_bytes)
        content = _decode_amazon_content(content_bytes = content_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to load or parse data from {url}: {str(e)}")

    ## parse reviews
    try:
        return _parse_amazon_reviews(content = content)
    except Exception as e:
        raise RuntimeError(f"Failed to parse reviews from data: {str(e)}")

## build amazon network
def _build_network_amazon(data: pd.DataFrame) -> tuple[list[str], list[tuple[str, str]]]:
    
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
def _process_events_amazon(data: pd.DataFrame) -> pd.DataFrame:
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
        self.invariants = None
        self.signatures = None
        self.events = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_raw = _load_network_amazon(url = self.url, root_path = self.root_path, name = self.name)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_raw is None:
            self.load_data()
        nodes, edges = _build_network_amazon(data = self.data_raw)
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        self.invariants = GraphInvariants(graph = self.graph).all()
        return self

    def process_signatures(self):
        """Computes process signatures over the daily event counts."""
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
        if self.data_raw is None:
            self.load_data()
        data_events = _process_events_amazon(data = self.data_raw)
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
            "graph": self.graph,
            "events": self.events.to_dict(orient = "records")
        }
