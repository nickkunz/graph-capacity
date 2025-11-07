## libraries
import io
import os
import sys
import ssl
import tarfile
import urllib.request
import pandas as pd
from typing import Optional, Dict, Any

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import _aggregate_by_day
from src.invariants import BipartiteInvariants

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

## process event counts
def process_events_mooc(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(
        datetime = lambda df: pd.to_datetime(arg = df["timestamp"], unit = "s", utc = True),
        day = lambda df: df["timestamp"] // (24 * 60 * 60),
    )


## mooc user-action network
class MoocProcessor:
    def __init__(self, url: str):
        self.url: str = url
        self.data_raw: Optional[pd.DataFrame] = None
        self.data_processed: Optional[pd.DataFrame] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_raw = load_network_mooc(url = self.url)
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_raw is None:
            self.load_data()
        m, n = compute_network_mooc(data = self.data_raw)
        self.invariants = BipartiteInvariants(m = m, n = n).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_raw is None:
            self.load_data()
        data_events = process_events_mooc(self.data_raw.copy())
        self.events = _aggregate_by_day(data = data_events, datetime = 'datetime')
        return self

    def run(self):
        """ Executes the pipeline and returns the final result. """
        self.process_network()
        self.process_events()
        return {
            "invariants": self.invariants,
            "events": self.events.to_dict(orient="records")
        }
