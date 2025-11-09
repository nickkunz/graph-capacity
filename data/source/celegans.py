## libraries
import os
import sys
import numpy as np
import pandas as pd
import igraph as ig
from typing import Optional, Dict, Any

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import _create_igraph_object, _aggregate_by_day
from src.invariants import GraphInvariants

## load caenorhabditis elegans network data
def _load_network_celegans() -> pd.DataFrame:

    """
    Loads the fixed substrate network of C. elegans developmental stages.

    This represents the atemporal, directed acyclic graph (DAG) of all known
    biological transitions, merging normal, dauer, and recovery pathways.

    Provenance:
        - Cassada RC & Russell RL (1975) Dev Biol 46:326–342
        - Golden JW & Riddle DL (1984) Dev Biol 102:368–378

    Returns:
        pd.DataFrame[source, target, weight]: The static substrate network.
    """
    
    ## The directed graph below merges:
    ## - Normal L1→L4→Adult pathway (Cassada & Russell, Fig. 1)
    ## - Dauer alternative at L2/L3 decision point (Golden & Riddle, Fig. 1)
    ## - Recovery branches (Golden & Riddle, Fig. 3)
    ## This represents the atemporal set of all biologically possible transitions.
    data = [
        # --- normal development (20 °C, total ≈ 50 h) ---
        ("egg", "L1_hatching"),
        ("L1_hatching", "L1_feeding"),
        ("L1_feeding", "L1_L2_molt"),
        ("L1_L2_molt", "L2_feeding"),
        ("L2_feeding", "L2_L3_molt"),
        ("L2_L3_molt", "L3_normal"),
        ("L3_normal", "L3_feeding"),
        ("L3_feeding", "L3_L4_molt"),
        ("L3_L4_molt", "L4_feeding"),
        ("L4_feeding", "L4_adult_molt"),
        ("L4_adult_molt", "adult_reproductive"),

        # --- dauer formation branch (alternative L3) ---
        ("L2_L3_molt", "predauer_L2d"),
        ("predauer_L2d", "dauer_L3_entry"),
        ("dauer_L3_entry", "dauer_cuticle_formation"),
        ("dauer_cuticle_formation", "dauer_L3_complete"),

        # --- dauer maintenance sequence (extended quiescence) ---
        ("dauer_L3_complete", "dauer_day3"),
        ("dauer_day3", "dauer_day5"),
        ("dauer_day5", "dauer_day7"),
        ("dauer_day7", "dauer_day10"),
        ("dauer_day10", "dauer_day14"),
        ("dauer_day14", "dauer_day20"),
        ("dauer_day20", "dauer_day30"),

        # --- recovery after dauer (7–10 d; Golden & Riddle 1984, Fig. 3) ---
        ("dauer_day7", "dauer_recovery_initiation"),
        ("dauer_recovery_initiation", "dauer_exit_molt"),
        ("dauer_exit_molt", "L4_recovery"),
        ("L4_recovery", "L4_feeding_recovery"),
        ("L4_feeding_recovery", "adult_molt_recovery"),
        ("adult_molt_recovery", "adult_reproductive_recovery"),

        # --- late recovery (≥ 14 d dauer) ---
        ("dauer_day14", "late_recovery_init"),
        ("late_recovery_init", "late_dauer_exit"),
        ("late_dauer_exit", "late_L4"),
        ("late_L4", "late_L4_feeding"),
        ("late_L4_feeding", "late_adult_molt"),
        ("late_adult_molt", "late_adult_reproductive"),

        # --- ultra-late recovery (≈ 30 d dauer) ---
        ("dauer_day30", "ultra_late_recovery"),
        ("ultra_late_recovery", "ultra_late_exit"),
        ("ultra_late_exit", "ultra_late_L4"),
        ("ultra_late_L4", "ultra_late_adult"),
        ("ultra_late_adult", "ultra_late_reproductive"),
    ]
    return pd.DataFrame(
        data = data, 
        columns = ["source", "target"]
    )

## construct caenorhabditis elegans network data
def _build_network_celegans(data: pd.DataFrame) -> tuple[list[str], list[tuple[str, str]]]:

    ## extract unique nodes from both source and target columns
    nodes = pd.unique(data[['source', 'target']].values.ravel('K')).tolist()

    ## extract edges as a list of tuples
    edges = [tuple(x) for x in data.to_numpy()]

    return nodes, edges

def _load_events_celegans() -> pd.DataFrame:
    """
    Loads the temporal sequence of C. elegans developmental events.

    Timings approximate observed midpoints at 20–25 °C and represent a canonical
    timeline using population mean transition times.

    Provenance:
        - Cassada & Russell 1975, Table 1: total 50 h egg→adult
        - Golden & Riddle 1984: Dauer induction and recovery timings

    Returns:
        pd.DataFrame[event_id, source, target, pathway, time_hours, time_days, day_bin]:
        The time-stamped event sequence.
    """
    ## Timings below approximate observed midpoints at 20–25 °C:
    ## - Cassada & Russell 1975, Table 1: total 50 h egg→adult
    ## - Dauer induction: 26–30 h post-hatch (Golden & Riddle 1984)
    ## - Dauer longevity: days–weeks; 30 d used as canonical long-term reference
    ## - Recovery: 12–24 h after re-feeding (Golden & Riddle 1984)
    ## - Canonical developmental timeline using population mean transition times.
    # -  Each transition occurs once at its mean time (not individual observations).
    data = [
        # --- normal pathway ---
        {"source": "egg", "target": "L1_hatching", "time_hours": 0, "pathway": "normal"},
        {"source": "L1_hatching", "target": "L1_feeding", "time_hours": 2, "pathway": "normal"},
        {"source": "L1_feeding", "target": "L1_L2_molt", "time_hours": 12, "pathway": "normal"},
        {"source": "L1_L2_molt", "target": "L2_feeding", "time_hours": 14, "pathway": "normal"},
        {"source": "L2_feeding", "target": "L2_L3_molt", "time_hours": 24, "pathway": "normal"},
        {"source": "L2_L3_molt", "target": "L3_normal", "time_hours": 26, "pathway": "normal"},
        {"source": "L3_normal", "target": "L3_feeding", "time_hours": 28, "pathway": "normal"},
        {"source": "L3_feeding", "target": "L3_L4_molt", "time_hours": 36, "pathway": "normal"},
        {"source": "L3_L4_molt", "target": "L4_feeding", "time_hours": 38, "pathway": "normal"},
        {"source": "L4_feeding", "target": "L4_adult_molt", "time_hours": 48, "pathway": "normal"},
        {"source": "L4_adult_molt", "target": "adult_reproductive", "time_hours": 50, "pathway": "normal"},

        # --- dauer formation (branch from L2/L3 molt) ---
        {"source": "L2_L3_molt", "target": "predauer_L2d", "time_hours": 26, "pathway": "dauer"},
        {"source": "predauer_L2d", "target": "dauer_L3_entry", "time_hours": 30, "pathway": "dauer"},
        {"source": "dauer_L3_entry", "target": "dauer_cuticle_formation", "time_hours": 36, "pathway": "dauer"},
        {"source": "dauer_cuticle_formation", "target": "dauer_L3_complete", "time_hours": 48, "pathway": "dauer"},

        # --- dauer maintenance ---
        {"source": "dauer_L3_complete", "target": "dauer_day3", "time_hours": 72, "pathway": "dauer"},
        {"source": "dauer_day3", "target": "dauer_day5", "time_hours": 120, "pathway": "dauer"},
        {"source": "dauer_day5", "target": "dauer_day7", "time_hours": 168, "pathway": "dauer"},
        {"source": "dauer_day7", "target": "dauer_day10", "time_hours": 240, "pathway": "dauer"},
        {"source": "dauer_day10", "target": "dauer_day14", "time_hours": 336, "pathway": "dauer"},
        {"source": "dauer_day14", "target": "dauer_day20", "time_hours": 480, "pathway": "dauer"},
        {"source": "dauer_day20", "target": "dauer_day30", "time_hours": 720, "pathway": "dauer"},

        # --- recovery (after ≈ 7 d dauer) ---
        {"source": "dauer_day7", "target": "dauer_recovery_initiation", "time_hours": 170, "pathway": "recovery"},
        {"source": "dauer_recovery_initiation", "target": "dauer_exit_molt", "time_hours": 180, "pathway": "recovery"},
        {"source": "dauer_exit_molt", "target": "L4_recovery", "time_hours": 192, "pathway": "recovery"},
        {"source": "L4_recovery", "target": "L4_feeding_recovery", "time_hours": 204, "pathway": "recovery"},
        {"source": "L4_feeding_recovery", "target": "adult_molt_recovery", "time_hours": 228, "pathway": "recovery"},
        {"source": "adult_molt_recovery", "target": "adult_reproductive_recovery", "time_hours": 240, "pathway": "recovery"},

        # --- late & ultra-late recoveries (14–33 d) ---
        {"source": "dauer_day14", "target": "late_recovery_init", "time_hours": 338, "pathway": "late_recovery"},
        {"source": "late_recovery_init", "target": "late_dauer_exit", "time_hours": 348, "pathway": "late_recovery"},
        {"source": "late_dauer_exit", "target": "late_L4", "time_hours": 360, "pathway": "late_recovery"},
        {"source": "late_L4", "target": "late_L4_feeding", "time_hours": 372, "pathway": "late_recovery"},
        {"source": "late_L4_feeding", "target": "late_adult_molt", "time_hours": 396, "pathway": "late_recovery"},
        {"source": "late_adult_molt", "target": "late_adult_reproductive", "time_hours": 408, "pathway": "late_recovery"},
        {"source": "dauer_day30", "target": "ultra_late_recovery", "time_hours": 722, "pathway": "ultra_late"},
        {"source": "ultra_late_recovery", "target": "ultra_late_exit", "time_hours": 732, "pathway": "ultra_late"},
        {"source": "ultra_late_exit", "target": "ultra_late_L4", "time_hours": 744, "pathway": "ultra_late"},
        {"source": "ultra_late_L4", "target": "ultra_late_adult", "time_hours": 780, "pathway": "ultra_late"},
        {"source": "ultra_late_adult", "target": "ultra_late_reproductive", "time_hours": 792, "pathway": "ultra_late"},
    ]

    events = pd.DataFrame(data = data)
    events["time_days"] = events["time_hours"] / 24
    events["day"] = np.floor(events["time_days"]).astype(int)
    return events

## c. elegans network
class CelegansProcessor:
    def __init__(self):
        self.data_network: Optional[pd.DataFrame] = None
        self.data_events: Optional[pd.DataFrame] = None
        self.graph: Optional[ig.Graph] = None
        self.invariants: Optional[Dict[str, Any]] = None
        self.events: Optional[pd.DataFrame] = None

    def load_data(self):
        """ Loads the raw data from source. """
        self.data_network = _load_network_celegans()
        self.data_events = _load_events_celegans()
        return self

    def process_network(self):
        """ Builds the network and computes invariants. """
        if self.data_network is None:
            self.load_data()
        
        nodes, edges = _build_network_celegans(data=self.data_network)
        self.graph = _create_igraph_object(nodes=nodes, edges=edges)
        self.invariants = GraphInvariants(graph=self.graph).all()
        return self

    def process_events(self):
        """ Processes the event data. """
        if self.data_events is None:
            self.load_data()
        
        self.events = _aggregate_by_day(data=self.data_events, datetime='day')
        return self

    def run(self):
        """ Executes the pipeline and returns the final result. """
        self.process_network()
        self.process_events()
        return {
            "invariants": self.invariants,
            "events": self.events.to_dict(orient="records")
        }