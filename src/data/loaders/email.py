## libraries
import sys
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any

## path
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.vectorizers.invariants import BipartiteInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.data.helpers import (
    _load_network_snap,
    _compute_network_snap
)

## process email events
def process_events_email(data: pd.DataFrame) -> pd.DataFrame:
    data['day'] = (data['timestamp'] // 86400)
    return data.groupby('day').agg(
        target=('day', 'size')
    ).reset_index()

## email user-user network
class EmailProcessor:
    def __init__(self, url: str):
        self.url = url
        self.data: Optional[pd.DataFrame] = None
        self.graph: Optional[Any] = None
        self.dimensions: Optional[tuple[int, int]] = None
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
        self.dimensions = (int(m), int(n))
        self.invariants = BipartiteInvariants(m = m, n = n).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data is None:
            self.load_data()
        self.events = process_events_email(data = self.data)
        return self

    def process_signatures(self):
        """Computes process signatures over daily email events."""
        if self.events is None:
            self.process_events()
        self.signatures = ProcessSignatures(
            data = self.events.copy(),
            sort_by = ["day"],
            target = "target"
        ).all()
        return self

    def run(self):
        """ Executes the pipeline and returns the final result. """
        self.process_network()
        self.process_signatures()
        self.process_events()
        
        return {
            "invariants": self.invariants,
            "signatures": self.signatures,
            "events": self.events.to_dict(orient = "records")
        }
