## libraries
import io
import os
import sys
import time
import gzip
import ssl
import tarfile
import requests
import torch
import pandas as pd
import urllib.request
from io import BytesIO
from torch_geometric.datasets import BitcoinOTC

## project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

## modules
from src.utils import _create_igraph_object, _dict_to_json
from src.invariants import GraphInvariants, BipartiteInvariants

## build tripartite user–rating–user network
def build_network_bitcoin(data: BitcoinOTC) -> tuple[list[str], list[tuple[str, str]]]:

    ## single graph object
    data_idx = data[10] ## 11th snapshot

    ## extract users and ratings
    users = torch.unique(data_idx.edge_index).cpu().numpy()
    user_nodes = [f"user_{u}" for u in users]

    ## all possible rating nodes (–10 to +10)
    rating_nodes = [f"rating_{r}" for r in range(-10, 11)]

    ## extract directed edges
    src_users = data_idx.edge_index[0].cpu().numpy()
    dst_users = data_idx.edge_index[1].cpu().numpy()
    ratings = data_idx.edge_attr.cpu().numpy().astype(int).flatten()

    ## build tripartite edges: user → rating → user
    edges = []
    for s, d, r in zip(src_users, dst_users, ratings):
        r_node = f"rating_{r}"
        edges.append((f"user_{s}", r_node))
        edges.append((r_node, f"user_{d}"))

    ## combine nodes
    nodes = user_nodes + rating_nodes
    return nodes, edges


## load network contracts from endpoint
def _post_network_contracts(
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
        response = requests.post(url = url, json = json, timeout = 120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise RuntimeError(f"Failed POST request to {url}: {str(e)}")

## iterate over network contracts with pagination
def _iter_network_contracts(
        url: str, 
        start_date: str, 
        end_date: str,
        page_size: int = 100,
        max_pages: int = 1_000, 
        max_records: int = 10_000, 
        keyword: str | None = None, 
        sleep: float = 0.5) -> dict:
    fetched_records = 0
    current_page = 1
    while current_page <= max_pages:
        json_data = _post_network_contracts(
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
def load_network_contracts(
    url: str, 
    start_date: str, 
    end_date: str, 
    max_pages: int = 100, 
    max_records: int = 10_000, 
    keyword: str | None = None) -> pd.DataFrame:
    data = list(_iter_network_contracts(
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
def process_network_contracts(data: pd.DataFrame) -> pd.DataFrame:
    
    ## make copy to avoid erroring on original data
    data_copy = data.copy()

    ## material award amounts
    data_copy = data_copy[data_copy["Award Amount"] > 0]

    ## parse dates
    data_copy['Start Date'] = pd.to_datetime(
        arg = data_copy['Start Date'], 
        errors = 'coerce'
    )
    data_copy['End Date'] = pd.to_datetime(
        arg = data_copy['End Date'], 
        errors = 'coerce'
    )

    ## clean names
    data_copy['Recipient Name'] = data_copy['Recipient Name'].str.strip().str.upper()
    data_copy['Awarding Agency'] = data_copy['Awarding Agency'].str.strip().str.upper()

    return data_copy

## build network from contracts
def build_network_contracts(data: pd.DataFrame) -> tuple[list[str], list[tuple[str, str]]]:

    ## make copy to avoid erroring on original data
    data_copy = data.copy()

    ## create recipient node id (use recipient_id if available, else recipient name)
    data_copy['recipient_node_id'] = data_copy['recipient_id'].fillna(data_copy['Recipient Name'])

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


## load mooc_actions from the snap tarball robustly (streaming, stdlib parsing)
def load_network_mooc(url: str) -> pd.DataFrame:
    
    ## create ssl context that can handle self-signed certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    ## download archive into memory with ssl context
    with urllib.request.urlopen(url = url, context = ssl_context) as r:
        tar_bytes = r.read()

    ## open outer tar.gz
    tf = tarfile.open(fileobj = io.BytesIO(tar_bytes), mode = "r:gz")

    ## locate inner member (ignore macos '._' files)
    member = None
    for m in tf.getmembers():
        if not m.isfile():
            continue
        base = os.path.basename(m.name)
        if base.startswith("._"):
            continue
        name_lower = base.lower()
        if name_lower == "mooc_actions.tsv" or name_lower == "mooc_actions.tsv.gz":
            member = m
            break
    if member is None:
        raise FileNotFoundError("mooc_actions.tsv(.gz) not found in archive")

    ## extract raw bytes and handle inner gzip if present
    file = tf.extractfile(member = member)
    if file is None:
        raise FileNotFoundError(f"could not extract file: {member.name}")

    ## read directly into dataframe
    data = pd.read_csv(
        filepath_or_buffer = file,
        sep = '\t',
        header = 0,  ## first row header
        encoding = 'latin1'  ## robust encoding
    )
    if data.empty:
        raise RuntimeError("No valid rows parsed from data.")
    
    ## build dataframe
    data = data.rename(
        columns = {
            "USERID": "src", 
            "TARGETID": "dst", 
            "TIMESTAMP": "timestamp"
        }
    )
    
    ## retain only relevant columns
    data = data.astype({'src': int, 'dst': int, 'timestamp': int})
    return data[["src", "dst", "timestamp"]]


## compute network size from data
def compute_network_mooc(data: pd.DataFrame) -> dict:
    if not {"src", "dst"}.issubset(data.columns):
        raise ValueError("Data must have columns 'src' and 'dst'.")
    m = data["src"].nunique()
    n = data["dst"].nunique()
    return m, n


## load amazon network
def load_network_amazon(url: str, timeout: int = 30) -> pd.DataFrame:    
    try:
        response = requests.get(url = url, stream = True, timeout = timeout)
        response.raise_for_status()
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


## execute
if __name__ == '__main__':

    ## configuration paths
    root_data = "data/original/"
    path_out = "data/processed/networks/"

    ## us fed contract procurement
    data_contracts = load_network_contracts(
        url = "https://api.usaspending.gov/api/v2/search/spending_by_award/",
        start_date = "2014-01-01", 
        end_date = "2024-12-31",
        max_records = 99999,
        keyword = "waterfowl"
    )
    data_contracts = process_network_contracts(data = data_contracts)
    nodes_contracts, edges_contracts = build_network_contracts(data = data_contracts)
    graph_contracts = _create_igraph_object(nodes = nodes_contracts, edges = edges_contracts)
    invar_contracts = GraphInvariants(graph = graph_contracts).all()
    _dict_to_json(invar = invar_contracts, path = path_out + "contracts.json")

    ## mooc student actions network
    data_mooc = load_network_mooc(
        url = "https://snap.stanford.edu/data/act-mooc.tar.gz"
    )
    m_mooc, n_mooc = compute_network_mooc(data = data_mooc)
    invar_mooc = BipartiteInvariants(m = m_mooc, n = n_mooc).all()
    _dict_to_json(invar = invar_mooc, path = path_out + "mooc.json")

    ## amazon product review network
    data_amazon = load_network_amazon(
        url = "https://snap.stanford.edu/data/finefoods.txt.gz"
    )
    nodes_amazon, edges_amazon = build_network_amazon(data = data_amazon)
    graph_amazon = _create_igraph_object(nodes_amazon, edges_amazon)
    invar_amazon = GraphInvariants(graph_amazon).all()
    _dict_to_json(invar = invar_amazon, path = path_out + "amazon.json")

    ## bitcoin user–rating–user network
    data_bitcoin = BitcoinOTC(root = root_data + "BitcoinOTC")
    nodes_bitcoin, edges_bitcoin = build_network_bitcoin(data = data_bitcoin)
    graph_bitcoin = _create_igraph_object(nodes = nodes_bitcoin, edges = edges_bitcoin)
    invar_bitcoin = GraphInvariants(graph = graph_bitcoin).all()
    _dict_to_json(invar = invar_bitcoin, path = path_out + "trade.json")



