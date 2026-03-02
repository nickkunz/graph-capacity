## libraries
import os
import io
import sys
import time
import json
import torch
import logging
import hashlib
import zipfile
import requests
import importlib
import numpy as np
import pandas as pd
import igraph as ig
import configparser
from typing import Any, Dict
from pathlib import Path
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

## path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## config
config = configparser.ConfigParser()
config.read(os.path.join(root, 'conf', 'settings.ini'))

## logging
logger = logging.getLogger(__name__)

## daily aggregation
def _aggregate_by_day(data: pd.DataFrame, datetime: str, label: str = 'day') -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(data[datetime]):
        data[datetime] = pd.to_datetime(data[datetime], errors = 'coerce')    
    data[label] = data[datetime].dt.date
    return data.groupby(label).agg(
        target = (datetime, 'size')
    ).reset_index()

## igraph object from nodes and edges
def _create_igraph_object(nodes: list[str], edges: list[tuple]) -> ig.Graph:
    g = ig.Graph(directed = False)
    g.add_vertices(nodes)
    g.add_edges(edges)
    return g

## finite value guarantee
def _force_finite(value: Any, default: float = 0.0) -> float:
    ## convert inf, -inf, nan to default value
    if not np.isfinite(value):
        return default
    return value

## finite feature guarantee for dictionary of features
def _force_finite_dict(features: Dict[str, float], default: float = 0.0) -> Dict[str, float]:
    return {k: _force_finite(v, default = default) for k, v in features.items()}

## clip scalar unit interval zero to one [0, 1]
def _clip_unit_interval(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    value = _force_finite(value, default = 0.0)  ## force finite
    return max(0.0, min(1.0, value))

## save dictionary to json
def _save_to_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(p = path), exist_ok = True)
    with open(path, 'w') as fp:
        json.dump(obj = data, fp = fp, indent = 2, default = str)

## find path for cache directory
def _cache_dir(namespace: str = "http") -> str:
    path = Path(__file__).resolve().parents[2] / "cache" / namespace
    path.mkdir(parents = True, exist_ok = True)
    return str(path)

## generate cache key from request parameters
def _cache_key(url: str, method: str = "GET", params: dict = None, payload: dict = None) -> str:
    payload = {
        "url": str(url),
        "method": str(method).upper(),
        "params": params or dict(),
        "payload": payload or dict(),
    }
    canonical = json.dumps(payload, sort_keys = True, default = str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

## create a cached response object for a given content and url
def _cache_response(content: bytes, url: str, status_code: int = 200) -> requests.Response:
    response = requests.Response()
    response._content = content
    response.status_code = status_code
    response.url = url
    response.encoding = "utf-8"
    response.headers["X-Cache"] = "HIT"
    return response

## make get request with retries
def _request_with_retry(
    url: str, 
    method: str = 'GET',
    params: dict = None,
    json: dict = None,
    retries: int = 3, 
    timeout: int = 60, 
    sleep: float = 0.5,
    use_cache: bool = True,
    cache_namespace: str = "http",
    force_refresh: bool = False,
    ) -> requests.Response:

    ## caching layer
    key = _cache_key(url = url, method = method, params = params, payload = json)
    cache_path = os.path.join(_cache_dir(cache_namespace), f"{key}.bin")
    if use_cache and force_refresh:
        logger.info(f"Cache refresh enabled ({cache_namespace}): bypassing cache read for {url}")
    if use_cache and (not force_refresh) and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fp:
                logger.info(f"Cache hit ({cache_namespace}): {url}")
                return _cache_response(content = fp.read(), url = url)
        except Exception as e:
            logger.warning(f"Cache read failed ({cache_namespace}) for {url}: {e}")
    if use_cache:
        logger.info(f"Cache miss ({cache_namespace}): {url}")
    else:
        logger.info(f"Cache disabled: {url}")

    ## retry logic
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

            ## save to cache if enabled
            if use_cache:
                try:
                    with open(cache_path, "wb") as fp:
                        fp.write(response.content)
                    logger.info(f"Cache write ({cache_namespace}): {url}")
                except Exception as e:
                    logger.warning(f"Cache write failed ({cache_namespace}) for {url}: {e}")
            return response

        ## catch specific request exceptions and retry
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(sleep * (2 ** attempt))
            else:
                raise RuntimeError(f"Request failed after {retries} retries: {e}")

## load generic snap temporal dataset
def _load_network_snap(url: str) -> pd.DataFrame:
    try:
        response = _request_with_retry(url = url, timeout = 60, use_cache = True)
        data = pd.read_csv(
            filepath_or_buffer = io.BytesIO(response.content),
            sep = r'\s+',
            header = None,
            names = ["src", "dst", "timestamp"],
            comment = '#',
            compression = 'gzip',
            engine = 'python'
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {url} with pandas: {e}")
    if data.empty:
        raise RuntimeError(f"No data loaded from {url}")
    return data

## build generic snap temporal dataset (fully connected bipartite)
def _compute_network_snap(data: pd.DataFrame, unix_time: bool = True) -> tuple[int, int]:
    m = pd.concat([data["src"], data["dst"]]).nunique()
    if unix_time:
        data["day"] = pd.to_datetime(data["timestamp"], unit = "s", utc = True).dt.floor("1D")
        days = pd.date_range(start = data["day"].min(), end = data["day"].max(), freq = "1D")
        n = len(days)
    else:
        ## relative timestamps: enumerate days
        data["day"] = data["timestamp"] // 86400
        n = int(data["day"].max()) + 1
    return m, n

## load network from pytorch geometric temporal dataset
def _load_network_pygt(loader: object) -> DynamicGraphTemporalSignal:
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
        response = _request_with_retry(url = url, timeout = timeout, use_cache = True)
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

## extract event counts from events dataframe
def _extract_counts(events: pd.DataFrame | None):
    if events is None or (isinstance(events, pd.DataFrame) and events.empty):
        return None
    if isinstance(events, pd.DataFrame):
        for col in ('target', 'count'):
            if col in events.columns:
                return events[col].to_numpy()
    return None

## extract timestamps from processor object
def _extract_timestamps(proc: Any, name: str | None = None) -> pd.Series | None:
    try:
        from src.data.loaders.bitcoin import load_events_bitcoin
        if hasattr(proc, 'data_raw') and proc.data_raw is not None:
            index = getattr(proc, 'index', 10)
            res = load_events_bitcoin(proc.data_raw, index = index)
            if isinstance(res, dict) and 'datetime' in res:
                return res['datetime']
            if isinstance(res, pd.DataFrame) and 'datetime' in res.columns:
                return res['datetime']
    except Exception:
        pass  ## fail silently and fall back to generic checks

    ## check for common timestamp/datetime columns in data attributes
    for attr in ('data', 'data_raw', 'data_events', 'data_events_raw'):
        data = getattr(proc, attr, None)
        if data is not None and isinstance(data, pd.DataFrame):
            for col in ('timestamp', 'datetime'):
                if col in data.columns:
                    return data[col]

    ## check events attribute as final fallback
    events = getattr(proc, 'events', None)
    if events is not None and isinstance(events, pd.DataFrame):
        for col in ('timestamp', 'datetime'):
            if col in events.columns:
                return events[col]
    return None

## convert unix timestamps to datetime
def _to_datetime(values: pd.Series | None) -> pd.Series | None:
    if values is None or len(values) == 0:
        return None
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit = 's')
    return pd.to_datetime(values)


# -----------------------------------------------------------------------------
# perturbation table helpers
# -----------------------------------------------------------------------------

## constants
PERT_TYPE = [
    ('network_perturbed', 'network', 'invariants', 'net'),
    ('invariants_perturbed', 'invariants', 'invariants', 'inv'),
    ('process_perturbed', 'process', 'signatures', 'proc'),
    ('signatures_perturbed', 'signature', 'signatures', 'sig'),
    ('temporal_aggregated', 'temporal', 'events', 'tmp'),
]
PERT_JSON_BY_TYPE = {
    type: json_key for json_key, type, _, _ in PERT_TYPE
}
PATH_PERT = config['paths']['PATH_PERT'].strip('"')

## prefix feature keys in a dictionary with a given prefix
def _prefix_features(features: dict, prefix: str) -> dict:
    """ Add a prefix to each key in the features dictionary. """

    ## ensure features is a dictionary
    return {f"{prefix}__{k}": v for k, v in features.items()}

## collect perturbated data from json files
def _index_perturbs(pert_path: str) -> dict:
    """ Iterate over all perturbation json files and extract features into an index. """

    ## iterate over all json files in the perturbation directory
    index = dict()
    path = Path(pert_path)
    for json_path in sorted(path.glob("*.json")):
        data_name = json_path.stem
        with open(json_path, "r") as f:
            data = json.load(f)

        ## iterate over defined perturbation types and extract features into index
        for json_key, type, feat_key, prefix in PERT_TYPE:
            for rec in data.get(json_key, []):
                
                ## support for both intensity or param field
                intensity = rec.get("intensity", rec.get("param"))
                method = rec.get("method")

                ## temporal aggregation: method is always "aggregation", intensity is the scale
                if type == "temporal":
                    method = "aggregation"
                    intensity = rec.get("scale")

                key = (type, method, intensity)
                if key not in index:
                    index[key] = dict()

                ## create observation with dataset name and prefixed features
                obs = {"dataset": data_name}

                ## temporal aggregation: compute signatures from events list
                if type == "temporal" and isinstance(rec.get("events"), list):
                    events_df = pd.DataFrame(rec["events"])
                    if "target" in events_df.columns and len(events_df) >= 2:
                        from src.vectorizers.signatures import ProcessSignatures
                        events_df["idx"] = range(len(events_df))
                        sigs = ProcessSignatures(events_df, sort_by = ["idx"], target = "target")
                        obs.update(_prefix_features(sigs.all(), prefix))
                else:
                    feat_val = rec.get(feat_key, dict()) if feat_key is not None else dict()
                    if isinstance(feat_val, dict):
                        obs.update(_prefix_features(feat_val, prefix))
                index[key][data_name] = obs
    return index

## create perturbation data from indexed perturbation data
def load_perturbs(pert_path: str | None = None, schema: str = "tuple") -> dict:
    """ Create perturbation tables indexed by tuple or payload schema. """

    ## fallback to default path if not provided
    if pert_path is None:
        pert_path = os.path.join(root, PATH_PERT)

    ## build tables from indexed perturbation data
    data_dict = dict()
    index = _index_perturbs(pert_path)
    for key in sorted(index.keys()):
        type, method, intensity = key
        data = pd.DataFrame(list(index[key].values()))
        data = data.sort_values("dataset").reset_index(drop = True)
        data_dict[key] = data
        logger.info(
            f"Table ({type}, {method}, {intensity}): "
            f"{len(data)} datasets"
        )

    ## default tuple schema: {(type, method, intensity): DataFrame}
    if schema == "tuple":
        logger.info(f"Created {len(data_dict)} perturbation tables from {pert_path}")
        return data_dict

    ## payload schema: {json_key: {method: {intensity: DataFrame}}}
    if schema == "payload":
        payload_dict = dict()
        for (type, method, intensity), data in data_dict.items():
            json_key = PERT_JSON_BY_TYPE.get(type, type)
            if json_key not in payload_dict:
                payload_dict[json_key] = dict()
            if method not in payload_dict[json_key]:
                payload_dict[json_key][method] = dict()
            payload_dict[json_key][method][intensity] = data
        logger.info(f"Created {len(payload_dict)} perturbation groups from {pert_path}")
        return payload_dict

    raise ValueError("schema must be either 'tuple' or 'payload'")
