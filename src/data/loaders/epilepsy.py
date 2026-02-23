## libraries
import os
import sys
import logging
import numpy as np
import pandas as pd
import igraph as ig
from typing import Optional, Dict, Any

## logging
logger = logging.getLogger(__name__)

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.helpers import _request_with_retry, _create_igraph_object
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures

## load eeg electrode network
def _load_network_epilepsy() -> pd.DataFrame:
    """
    Desc: 
        Loads the fixed 19-channel EEG electrode network (10-20 international system).
        Represents spatial connectivity of scalp electrodes.
    
    Args:
        None
    
    Returns:
        pd.DataFrame: A DataFrame representing the EEG electrode network.
    
    Raises:
        None
    """
    data = {
        "electrode": [
            "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
            "T3", "C3", "CZ", "C4", "T4",
            "T5", "P3", "PZ", "P4", "T6",
            "O1", "O2"
        ]
    }
    return pd.DataFrame(data)

## build eeg electrode network
def _build_network_epilepsy(data: pd.DataFrame) -> tuple[list[str], list[tuple]]:
    nodes = data["electrode"].tolist()
    
    ## define anatomically meaningful connections in 10-20 system
    edges = [
        ("FP1", "F7"), ("FP1", "F3"), ("FP2", "F4"), ("FP2", "F8"),
        ("F7", "T3"), ("F3", "C3"), ("F3", "FZ"), ("FZ", "CZ"), ("F4", "C4"), ("F4", "FZ"), ("F8", "T4"),
        ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"), ("C4", "T4"),
        ("T3", "T5"), ("C3", "P3"), ("CZ", "PZ"), ("C4", "P4"), ("T4", "T6"),
        ("T5", "P3"), ("P3", "PZ"), ("PZ", "P4"), ("P4", "T6"),
        ("T5", "O1"), ("P3", "O1"), ("PZ", "O1"), ("PZ", "O2"), ("P4", "O2"), ("T6", "O2"),
        ("O1", "O2")
    ]
    return nodes, edges

## load seizure events for a specific patient
def _load_events_epilepsy(url: str, ids: str) -> pd.DataFrame:
    
    ## load data
    url_summary = f"{url}/{ids}/{ids}-summary.txt"
    response = _request_with_retry(url = url_summary)
    
    ## parse summary file for seizure annotations
    seizures = list()
    lines = response.text.split('\n')
    current_file = None
    file_start_time = None
    for i in lines:
        i = i.strip()
        
        ## extract recording file name
        if i.startswith('File Name:'):
            current_file = i.split(':', 1)[1].strip()
            
        ## extract file start date/time (format: "14:43:04")
        if i.startswith('File Start Time:'):
            file_start_time = i.split(':', 1)[1].strip()
            
        ## extract seizure onset (seconds from file start)
        if 'Seizure' in i and 'Start Time' in i:
            try:
                onset_str = i.split()[-2]  ## get seconds value
                onset_seconds = int(onset_str)
                
                if current_file and file_start_time:
                    seizures.append({
                        'file': current_file,
                        'file_start_time': file_start_time,
                        'onset_seconds': onset_seconds
                    })
            except (ValueError, IndexError):
                continue
    
    if not seizures:
        raise RuntimeError(f"No seizures found in {ids} summary")
    
    ## convert to proper datetime format
    data = pd.DataFrame(seizures)
    
    ## parse file start times and compute absolute seizure timestamps
    ## CHB-MIT uses sequential recording dates starting from a base date
    base_dates = {
        'chb01': '2010-02-19', 'chb02': '2010-03-17', 'chb03': '2010-03-11',
        'chb04': '2010-03-15', 'chb05': '2010-03-08', 'chb06': '2010-03-20',
        'chb07': '2010-04-01', 'chb08': '2010-04-05', 'chb09': '2010-12-16',
        'chb10': '2011-02-25', 'chb11': '2010-05-01', 'chb12': '2010-05-10',
        'chb13': '2010-06-01', 'chb14': '2010-06-10', 'chb15': '2010-07-01',
        'chb16': '2010-07-15', 'chb17': '2010-08-01', 'chb18': '2010-08-15',
        'chb19': '2010-09-01', 'chb20': '2010-09-15', 'chb21': '2010-10-01',
        'chb22': '2010-10-15', 'chb23': '2010-11-01', 'chb24': '2010-11-15'
    }
    base_date = base_dates.get(ids, '2010-01-01')
    
    ## extract file number to determine day offset
    data['file_num'] = data['file'].str.extract(r'_(\d+)\.edf')[0].astype(int)
    data['day_offset'] = (data['file_num'] - 1) // 24  ## approximate 1 hour per file
    
    ## combine base date + day offset + file start time + seizure onset
    data['date'] = pd.to_datetime(base_date) + pd.to_timedelta(data['day_offset'], unit = 'D')
    data['file_start_dt'] = data['date'] + pd.to_timedelta(data['file_start_time'])
    data['datetime'] = data['file_start_dt'] + pd.to_timedelta(data['onset_seconds'], unit = 's')
    return data[['datetime']].sort_values('datetime').reset_index(drop = True)

def _process_events_epilepsy(events: pd.DataFrame) -> pd.DataFrame:
    events['date'] = events['datetime'].dt.date
    return events.groupby('date').size().reset_index(name = 'target')

## epilepsy seizure network
class EpilepsyProcessor:
    def __init__(self, url: str, ids: list[str]):
        self.url = url
        self.ids = ids
        self.data_network: Optional[pd.DataFrame] = None
        self.data_events: Optional[pd.DataFrame] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.signatures: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_network = _load_network_epilepsy()
        
        all_events = []
        for i in self.ids:
            try:
                events = _load_events_epilepsy(url = self.url, ids = i)
                if not events.empty:
                    events['patient'] = i
                    all_events.append(events)
            except Exception as e:
                logger.warning(f"Skipping {i}: {e}")
                continue
        
        if not all_events:
            raise RuntimeError("No seizure events found across all patients")
        
        self.data_events = pd.concat(all_events, ignore_index=True)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_network is None:
            self.load_data()
        nodes, edges = _build_network_epilepsy(data=self.data_network)
        self.graph = _create_igraph_object(nodes=nodes, edges=edges)
        self.invariants = GraphInvariants(graph=self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_events is None:
            self.load_data()
        self.events = _process_events_epilepsy(events = self.data_events)
        return self

    def process_signatures(self):
        """Computes process signatures on daily seizure counts."""
        if self.events is None:
            self.process_events()
        self.signatures = ProcessSignatures(
            data = self.events.copy(),
            sort_by = ["date"],
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
            "graph": self.graph,
            "events": self.events.to_dict(orient="records")
        }
