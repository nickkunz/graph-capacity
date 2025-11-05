## libraries
import numpy as np
import pandas as pd
import igraph as ig

## daily aggregation
def _aggregate_by_day(data: pd.DataFrame, datetime: str) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(data[datetime]):
        data[datetime] = pd.to_datetime(data[datetime], errors = 'coerce')    
    data['date'] = data[datetime].dt.date
    return data.groupby('date').agg(
        target = (datetime, 'size')
    ).reset_index()

## haversine distance matrix
def _compute_haversine(a, b) -> np.ndarray:
    radius = 6371.0
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) ** 2
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

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
