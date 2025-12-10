## libraries
import os
import sys
import pandas as pd
from typing import Optional, Dict, Any

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import _aggregate_by_day, _load_network_snap, _compute_network_snap
from src.invariants import BipartiteInvariants
from src.descriptors import ProcessDescriptors

## college message user-user network
class CollegeProcessor:
    def __init__(self, url: str):
        self.url = url
        self.data: Optional[pd.DataFrame] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.descriptors: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data = _load_network_snap(url = self.url)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data is None:
            self.load_data()
        m, n = _compute_network_snap(data = self.data, unix_time = True)
        self.invariants = BipartiteInvariants(m = m, n = n).all()
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

    def process_descriptors(self):
        """Computes process descriptors over daily event counts."""
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
