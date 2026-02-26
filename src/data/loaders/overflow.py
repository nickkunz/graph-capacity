## libraries
import os
import sys
import logging
import pandas as pd
from typing import Optional, Dict, Any

## path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

## modules
from src.vectorizers.invariants import BipartiteInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.data.helpers import (
    _aggregate_by_day, 
    _load_network_snap,
    _compute_network_snap
)

## logging
logger = logging.getLogger(__name__)

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
