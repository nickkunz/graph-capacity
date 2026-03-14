## libraries
import os
import sys
import json
import logging
import configparser
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path

## path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.data.helpers import _save_to_json

## logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    stream = sys.stdout
)

## configs
config = configparser.ConfigParser()
config.read(os.path.join(root, 'conf', 'settings.ini'))

## constants
PATH_PROC = config['paths']['PATH_PROC'].strip('"')
PATH_FALS = config['paths']['PATH_FALS'].strip('"')


## ----------------------------------------------------------------------
## random target remapping: f(x\', z\') -> y*
## ----------------------------------------------------------------------
def _falsify_target(
    invariants: dict[str, Any],
    signatures: dict[str, Any],
    events: list[dict],
    random_state: int = 42,
    ) -> dict[str, Any]:

    """
    Desc: randomly permute the target values across events while keeping
          invariants and signatures fixed. breaks the mapping
          f(X, Z) -> y* so a genuine model should show degraded
          performance on the falsified target.
    Args:
        invariants: graph invariant features for this dataset.
        signatures: process signature features for this dataset.
        events: list of event dicts with a 'target' key.
        random_state: random seed for reproducibility.
    Returns:
        dict with original invariants, original signatures, and
        events whose target values have been randomly permuted.
    """

    rng = np.random.RandomState(seed = random_state)
    targets = [e['target'] for e in events]
    permuted = rng.permutation(targets).tolist()
    falsified_events = []
    for event, new_target in zip(events, permuted):
        row = dict(event)
        row['target'] = new_target
        falsified_events.append(row)

    return {
        'invariants': dict(invariants),
        'signatures': dict(signatures),
        'events': falsified_events,
    }


## ----------------------------------------------------------------------
## random network and observation generation: rand(G, S)
## ----------------------------------------------------------------------
def _falsify_generate(
    invariants: dict[str, Any],
    signatures: dict[str, Any],
    events: list[dict],
    random_state: int = 42,
    ) -> dict[str, Any]:

    """
    Desc: replace invariants and signatures with independent draws from
          uniform distributions matching each field's original range.
          keeps event targets (y*) in place. breaks all authentic
          feature structure so posterior predictive power should vanish.
    Args:
        invariants: graph invariant features for this dataset.
        signatures: process signature features for this dataset.
        events: list of event dicts with a 'target' key.
        random_state: random seed for reproducibility.
    Returns:
        dict with uniformly randomised invariants and signatures,
        original events unchanged.
    """

    rng = np.random.RandomState(seed = random_state)

    ## randomise invariants
    gen_inv = {}
    for key, val in invariants.items():
        fval = float(val)
        gen_inv[key] = float(rng.uniform(low = 0.0, high = max(fval * 2, 1.0)))

    ## randomise signatures
    gen_sig = {}
    for key, val in signatures.items():
        fval = float(val)
        gen_sig[key] = float(rng.uniform(low = 0.0, high = max(fval * 2, 1.0)))

    return {
        'invariants': gen_inv,
        'signatures': gen_sig,
        'events': list(events),
    }


## ----------------------------------------------------------------------
## random vector generation: rand(x\', z\')
## ----------------------------------------------------------------------
def _falsify_vectors(
    invariants: dict[str, Any],
    signatures: dict[str, Any],
    events: list[dict],
    random_state: int = 42,
    ) -> dict[str, Any]:

    """
    Desc: replace each invariant and signature value with an
          independent draw from a normal distribution centred at the
          original value. preserves marginal location but breaks the
          joint structure between features and the target.
    Args:
        invariants: graph invariant features for this dataset.
        signatures: process signature features for this dataset.
        events: list of event dicts (returned unchanged).
        random_state: random seed for reproducibility.
    Returns:
        dict with normally randomised invariants and signatures,
        original events unchanged.
    """

    rng = np.random.RandomState(seed = random_state)

    ## randomise invariants with normal noise around original value
    gen_inv = {}
    for key, val in invariants.items():
        fval = float(val)
        sigma = max(abs(fval) * 0.1, 1e-12)
        gen_inv[key] = float(rng.normal(loc = fval, scale = sigma))

    ## randomise signatures with normal noise around original value
    gen_sig = {}
    for key, val in signatures.items():
        fval = float(val)
        sigma = max(abs(fval) * 0.1, 1e-12)
        gen_sig[key] = float(rng.normal(loc = fval, scale = sigma))

    return {
        'invariants': gen_inv,
        'signatures': gen_sig,
        'events': list(events),
    }


## ----------------------------------------------------------------------
## per-dataset falsification
## ----------------------------------------------------------------------
def _execute_falsifications(
    invariants: dict[str, Any],
    signatures: dict[str, Any],
    events: list[dict],
    name: str,
    random_state: int = 42,
    ) -> dict[str, Any]:

    """
    Desc: run all three falsification methods on a single dataset and
          return a dict keyed by method name containing the falsified
          invariants, signatures, and events.
    Args:
        invariants: graph invariant features for this dataset.
        signatures: process signature features for this dataset.
        events: list of event dicts with a 'target' key.
        name: dataset name for logging.
        random_state: random seed for reproducibility.
    Returns:
        dict mapping each falsification method name to its output.
    """

    results = {}

    methods = {
        'target_remap': _falsify_target,
        'random_generate': _falsify_generate,
        'vector_generate': _falsify_vectors,
    }

    for method_name, method_fn in methods.items():
        try:
            results[method_name] = method_fn(
                invariants = invariants,
                signatures = signatures,
                events = events,
                random_state = random_state,
            )
            logging.info(f"  {name}: {method_name} done.")
        except Exception as exc:
            logging.warning(
                f"  {name}: {method_name} failed: {exc}"
            )

    return results


## ----------------------------------------------------------------------
## falsification pipeline
## ----------------------------------------------------------------------
def json_falsifier(
    random_state: int = 42,
    ) -> None:

    """
    Desc: run the falsification pipeline across all processed datasets.
          for each dataset in data/processed/, apply three falsification
          methods and save results to data/falsified/{name}.json. skips
          datasets that already have a corresponding falsified file.
    Args:
        random_state: random seed forwarded to all falsification methods.
    Returns:
        None.
    """

    ## ensure falsified directory exists
    os.makedirs(name = PATH_FALS, exist_ok = True)

    ## discover processed json files
    json_files = sorted(
        f for f in os.listdir(PATH_PROC) if f.endswith('.json')
    )
    logging.info(f"Found {len(json_files)} processed datasets.")

    for file_name in json_files:
        namedata = os.path.splitext(file_name)[0]
        fals_path = os.path.join(PATH_FALS, f"{namedata}.json")

        ## skip if already falsified
        if os.path.exists(fals_path):
            logging.info(
                f"{namedata} falsifications already exist at "
                f"{fals_path}. Skipping."
            )
            continue

        ## load processed json
        proc_path = os.path.join(PATH_PROC, file_name)
        logging.info(f"Falsifying {namedata}...")
        with open(proc_path, 'r') as fp:
            payload = json.load(fp)

        invariants = payload.get('invariants', {})
        signatures = payload.get('signatures', {})
        events = payload.get('events', [])

        ## run falsification methods
        data = _execute_falsifications(
            invariants = invariants,
            signatures = signatures,
            events = events,
            name = namedata,
            random_state = random_state,
        )

        ## save falsified data
        _save_to_json(data = data, path = fals_path)
        logging.info(f"{namedata} falsifications saved to {fals_path}")


## primary execution
if __name__ == '__main__':
    json_falsifier()
