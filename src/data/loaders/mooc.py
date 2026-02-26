## libraries
import io
import os
import sys
import ssl
import tarfile
import urllib.request
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

## path
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.vectorizers.invariants import BipartiteInvariants
from src.vectorizers.signatures import ProcessSignatures

## load mooc_actions from the snap tarball robustly (streaming, stdlib parsing)
def _load_network_mooc(url: str) -> pd.DataFrame:
    
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
def _compute_network_mooc(data: pd.DataFrame) -> dict:
    if not {"src", "dst"}.issubset(data.columns):
        raise ValueError("Data must have columns 'src' and 'dst'.")
    m = data["src"].nunique()
    n = data["dst"].nunique()
    return m, n

## process event counts
def _process_events_mooc(data: pd.DataFrame) -> pd.DataFrame:
    data["day"] = data["timestamp"] // (24 * 60 * 60)
    return data.groupby("day").size().reset_index(name="target")

## mooc user-action network
class MoocProcessor:
    def __init__(self, url: str):
        self.url: str = url
        self.data: Optional[pd.DataFrame] = None
        self.graph: Optional[Any] = None
        self.dimensions: Optional[tuple[int, int]] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.signatures: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data = _load_network_mooc(url = self.url)
        return self

    def process_network(self):
        """ Builds the complete bipartite K_{m,n} graph and computes analytic invariants. """
        if self.data is None:
            self.load_data()

        ## compute bipartite dimensions and invariants
        m, n = _compute_network_mooc(data = self.data)
        self.dimensions = (int(m), int(n))
        self.invariants = BipartiteInvariants(m = m, n = n).all()
        return self

    def process_signatures(self):
        """Computes process signatures over daily event counts."""
        if self.events is None:
            self.process_events()
        self.signatures = ProcessSignatures(
            data = self.events.copy(),
            sort_by = ["day"],
            target = "target"
        ).all()
        return self
 
    def process_events(self):
        """ Processes the event data. """
        if self.data is None:
            self.load_data()
        self.events = _process_events_mooc(self.data.copy())
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
