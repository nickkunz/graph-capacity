## libraries
import os
import io
import ssl
import time
import json
import torch
import zipfile
import requests
import importlib
import numpy as np
import pandas as pd
import igraph as ig
import urllib.request
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

## daily aggregation
def _aggregate_by_day(data: pd.DataFrame, datetime: str) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(data[datetime]):
        data[datetime] = pd.to_datetime(data[datetime], errors = 'coerce')    
    data['day'] = data[datetime].dt.date
    return data.groupby('day').agg(
        target = (datetime, 'size')
    ).reset_index()

## igraph object from nodes and edges
def _create_igraph_object(nodes: list[str], edges: list[tuple]) -> ig.Graph:
    g = ig.Graph(directed = False)
    g.add_vertices(nodes)
    g.add_edges(edges)
    return g

## finite value guarantee
def _ensure_finite(value, default = 0.0) -> float:
    ## convert inf, -inf, nan to default value
    if not np.isfinite(value):
        return default
    return value

## save dictionary to json
def _save_to_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(p = path), exist_ok = True)
    with open(path, 'w') as fp:
        json.dump(obj = data, fp = fp, indent = 2, default = str)

## make get request with retries
def _request_with_retry(
    url: str, 
    method: str = 'GET',
    params: dict = None,
    json: dict = None,
    retries: int = 3, 
    timeout: int = 60, 
    sleep: float = 0.5
    ) -> requests.Response:
    for attempt in range(retries):
        try:
            if method.upper() == 'GET':
                response = requests.get(
                    url = url,
                    params = params,
                    timeout = timeout
                )
            elif method.upper() == 'POST':
                response = requests.post(
                    url = url,
                    json = json,
                    timeout = timeout
                )
            else:
                raise ValueError(f"Unsupported request method: {method}")
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(sleep * (2 ** attempt))
            else:
                raise RuntimeError(f"Request failed after {retries} retries: {e}")

## load generic snap temporal dataset
def _load_network_snap(url: str) -> pd.DataFrame:
    try:
        ## create ssl context that does not verify certificates
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        ## open url with ssl context
        with urllib.request.urlopen(url = url, context = context) as response:
            data = pd.read_csv(
                response,
                sep = r'\s+',
                header = None,
                names = ["src", "dst", "timestamp"],
                comment = '#',
                compression = 'gzip',
                engine = 'python'  # engine for regex separator
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {url} with pandas: {e}")
    if data.empty:
        raise RuntimeError(f"No data loaded from {url}")
    return data

## build generic snap temporal dataset (fully connected bipartite)
def _compute_network_snap(data: pd.DataFrame) -> tuple[int, int]:
    m = pd.concat([data["src"], data["dst"]]).nunique()
    data["day"] = pd.to_datetime(data["timestamp"], unit = "s", utc = True).dt.floor("1D")
    days = pd.date_range(start = data["day"].min(), end = data["day"].max(), freq = "1D")
    n = len(days)
    return m, n

## load network from pytorch geometric temporal dataset
def _load_network_pygt(loader):
    try:
        dataset = loader.get_dataset()
        if dataset is None:
            raise RuntimeError("PyTorch Geometric Temporal dataset loader returned None.")
        return dataset
        
    ## handle specific errors
    except AttributeError as e:
        raise RuntimeError(f"Invalid PyTorch Geometric Temporal loader class: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load PyTorch Geometric Temporal dataset: {e}")

## build network from pytorch geometric temporal dataset
def _build_network_pygt(dataset: DynamicGraphTemporalSignal) -> tuple[list[str], list[tuple]]:
    try:
        first_snap = next(iter(dataset))        
        if not hasattr(first_snap, 'num_nodes') or not hasattr(first_snap, 'edge_index'):
            raise RuntimeError("PyTorch Geometric Temporal dataset snapshot missing required attributes (num_nodes or edge_index).")
        
        num_nodes = first_snap.num_nodes
        edge_index = first_snap.edge_index.numpy()
        
        ## create node list (node ids as strings)
        nodes = [str(i) for i in range(num_nodes)]
        
        ## create edge list from edge index tensor
        edges = [(str(edge_index[0, i]), str(edge_index[1, i])) 
                 for i in range(edge_index.shape[1])]        
        if not nodes:
            raise RuntimeError("No nodes extracted from PyTorch Geometric Temporal dataset.")
        return nodes, edges

    ## handle specific errors        
    except StopIteration:
        raise RuntimeError("PyTorch Geometric Temporal dataset is empty - no snapshots available.")
    except AttributeError as e:
        raise RuntimeError(f"Invalid PyTorch Geometric Temporal dataset structure: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to build network from PyTorch Geometric Temporal dataset: {e}")

## load data from zipfile from endpoint
def _load_events_zip(url: str, name: str, timeout: int = 30) -> pd.DataFrame:
    try:
        response = requests.get(url = url, timeout = timeout)
        response.raise_for_status()
        file = io.BytesIO(initial_bytes = response.content)
        
        ## split name by '/' to handle nested paths
        parts = name.split(sep = '/')
        current = zipfile.ZipFile(file = file)

        ## recursively open nested zip files
        for p in parts[:-1]:
            with current.open(p) as nested:
                file_nested = io.BytesIO(initial_bytes = nested.read())
                current = zipfile.ZipFile(file = file_nested)

        ## open final file
        final = parts[-1]
        with current.open(final) as file_final:
            if final.endswith('.csv'):
                data = pd.read_csv(filepath_or_buffer = file_final)
            elif final.endswith('.json'):
                data = json.load(fp = file_final)
            else:
                raise ValueError(f"Unsupported file type: {final}. Only .csv and .json are supported.")

        ## check for empty data
        if isinstance(data, pd.DataFrame) and data.empty:
            raise RuntimeError("Downloaded data is empty.")
        if isinstance(data, dict) and not data:
            raise RuntimeError("Downloaded data is empty.")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to download data from {url}: {str(e)}")

## load pytorch geometric dataset by name
def _load_network_pyg(dataset: str, root: str, **kwargs) -> object:
    try:
        dataset_module = importlib.import_module('torch_geometric.datasets')
        dataset_class = getattr(dataset_module, dataset)
        return dataset_class(root = root, **kwargs)
    except AttributeError:
        raise ImportError(f"Dataset '{dataset}' not found in torch_geometric.datasets.")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset}': {e}")

## build network from a generic pytorch geometric dataset
def _build_network_pyg(data: object) -> tuple[list[str], list[tuple]]:
    
    ## handle both single graph object and dataset of graph objects
    graph_objects = list(data) if hasattr(data, '__iter__') and not \
        hasattr(data, 'edge_index') else [data]
    if not graph_objects:
        return [], []

    ## concatenate all edge indices from graph objects
    all_edges = [g.edge_index for g in graph_objects if hasattr(g, 'edge_index') and \
                 g.edge_index is not None]
    if not all_edges:
        return [], []
    edges = torch.cat(all_edges, dim = 1)
    if edges.numel() == 0:
        return [], []

    ## create unique undirected edges
    edges_undirect, _ = torch.sort(edges, dim = 0)
    edges_unique = torch.unique(edges_undirect, dim = 1)

    ## determine all nodes present in the graph
    nodes_all = torch.unique(edges_unique).cpu().numpy()
    nodes = [str(n) for n in nodes_all]

    ## convert edge tensor to list of tuples
    edge_list = [tuple(map(str, edge)) for edge in edges_unique.t().cpu().numpy()]

    return nodes, edge_list
