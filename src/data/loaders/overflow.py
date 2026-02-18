## libraries
import os
import sys
import logging
import itertools
import pandas as pd
from typing import Optional, Dict, Any

## logging
logger = logging.getLogger(__name__)

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.vectorizers.invariants import BipartiteInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.data.utilities import (
    _aggregate_by_day, 
    _load_network_snap,
    _compute_network_snap,
    _create_igraph_object
)

## stackoverflow user-user network
class OverflowProcessor:
    def __init__(self, url: str):
        self.url = url
        self.data: Optional[pd.DataFrame] = None
        self.graph: Optional[Any] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.signatures: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data = _load_network_snap(url = self.url)
        return self

    def process_network(self):
        """ Builds the complete bipartite K_{m,n} graph and computes analytic invariants. """
        if self.data is None:
            self.load_data()

        ## compute bipartite dimensions and invariants
        m, n = _compute_network_snap(data = self.data, unix_time = True)
        self.invariants = BipartiteInvariants(m = m, n = n).all()

        ## build complete bipartite graph K_{m,n}
        users = pd.concat([self.data['src'], self.data['dst']]).unique()
        days = pd.date_range(start = self.data['day'].min(), end = self.data['day'].max(), freq = '1D')
        user_nodes = [f"user::{str(u)}" for u in users]
        day_nodes = [f"day::{str(d.date())}" for d in days]
        nodes = user_nodes + day_nodes
        edges = list(itertools.product(user_nodes, day_nodes))
        self.graph = _create_igraph_object(nodes = nodes, edges = edges)
        return self

    def process_signatures(self):
        """Computes process signatures over daily event counts."""
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
        if self.data is None:
            self.load_data()
        self.events = _aggregate_by_day(
            data = self.data, 
            datetime = 'day',
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
