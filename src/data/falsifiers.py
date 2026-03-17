## libraries
import os
import sys
import json
import logging
import configparser
import numpy as np
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

## cross-data target remapping
def _permute_target_mapping(path_proc: str | Path, random_state: int = 42) -> dict[str, dict[str, Any]]:

    """
    Desc: 
        Permute the dataset-level target y* across processed data while keeping 
        each dataset's invariants, signatures, and event history fixed.
        This breaks the f(x', z') -> y* mapping.

    Args:
        path_proc: project directory containing processed dataset JSON files.
        random_state: random seed for reproducibility.

    Returns:
        dict keyed by dataset name with falsified payloads containing the
        original invariants/signatures/events and a permuted row target.
    
    Raises:
        FileNotFoundError: if the specified processed data directory does not
            exist or contains no JSON files.
        ValueError: if fewer than two processed JSON files are found in the
            directory.
    """

    ## load processed json data from specified directory
    json_path = Path(path_proc)
    if not json_path.exists() or not json_path.is_dir():
        raise FileNotFoundError(
            f"Processed data directory not found: {json_path!r}. "
            f"Ensure PATH_PROC points to an existing folder containing '*.json' files."
        )
    json_files = sorted(f for f in os.listdir(json_path) if f.endswith('.json'))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in processed data directory: {json_path!r}. "
            f"Ensure PATH_PROC points to a folder containing '*.json' files."
        )

    ## read each json file into a list of dictionaries
    json_data: list[dict[str, Any]] = list()
    for file in json_files:
        name = os.path.splitext(file)[0]
        with open(json_path / file, 'r') as fp:
            payload = json.load(fp)
        json_data.append({
            'name': name,
            'invariants': payload.get('invariants', dict()),
            'signatures': payload.get('signatures', dict()),
            'events': payload.get('events', []),
        })

    ## check for at least two datasets to permute
    n = len(json_data)
    if n < 2:
        raise ValueError(
            f"Target remapping requires at least two processed json_data, but found {n}. "
            f"Ensure {path_proc!r} contains at least two '*.json' files."
        )

    ## create a deranged permutation of dataset indices to shuffle targets
    rand = np.random.RandomState(seed = random_state)
    perm = np.arange(n)
    while True:
        rand.shuffle(perm)
        if not np.any(perm == np.arange(n)):
            break

    ## construct falsified payloads with permuted targets but original features/events
    json_perm = dict()
    for i, data in enumerate(json_data):
        json_perm[data['name']] = {
            'invariants': dict(data['invariants']),
            'signatures': dict(data['signatures']),
            'events': list(json_data[perm[i]]['events']),
        }
    return json_perm

## feature generation: independent uniform draws per feature
def _generate_random_features(path_proc: str | Path, random_state: int = 42) -> dict[str, dict[str, Any]]:

    """
    Desc:
        Generate new random features for each dataset by sampling each
        invariant/signature independently from a uniform distribution.

        For each feature, the range is computed across all processed datasets
        (min/max). Then each dataset's feature is drawn independently from
        Uniform(min, max) while keeping the event history fixed.

    Args:
        path_proc: project directory containing processed dataset JSON files.
        random_state: random seed for reproducibility.

    Returns:
        dict keyed by dataset name with falsified payloads containing
        generated invariants/signatures and original events.

    Raises:
        FileNotFoundError: if the specified processed data directory does not
            exist or contains no JSON files.
    """

    ## load processed json data from specified directory
    json_path = Path(path_proc)
    if not json_path.exists() or not json_path.is_dir():
        raise FileNotFoundError(
            f"Processed data directory not found: {json_path!r}. "
            f"Ensure PATH_PROC points to an existing folder containing '*.json' files."
        )
    json_files = sorted(f for f in os.listdir(json_path) if f.endswith('.json'))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in processed data directory: {json_path!r}. "
            f"Ensure PATH_PROC points to a folder containing '*.json' files."
        )

    ## read each json file into a list of dictionaries
    json_data: list[dict[str, Any]] = []
    for file in json_files:
        name = os.path.splitext(file)[0]
        with open(json_path / file, 'r') as fp:
            payload = json.load(fp)
        json_data.append({
            'name': name,
            'invariants': payload.get('invariants', dict()),
            'signatures': payload.get('signatures', dict()),
            'events': payload.get('events', []),
        })

    ## compute per-feature ranges across all datasets
    feature_ranges: dict[str, tuple[float, float]] = dict()
    for payload in json_data:
        for key, val in payload['invariants'].items():
            try:
                x = float(val)
            except Exception:
                continue
            if not np.isfinite(x):
                continue
            lo, hi = feature_ranges.get(key, (np.inf, -np.inf))
            feature_ranges[key] = (min(lo, x), max(hi, x))
        for key, val in payload['signatures'].items():
            try:
                x = float(val)
            except Exception:
                continue
            if not np.isfinite(x):
                continue
            lo, hi = feature_ranges.get(key, (np.inf, -np.inf))
            feature_ranges[key] = (min(lo, x), max(hi, x))

    ## generate new features for each dataset using the computed ranges
    rand = np.random.RandomState(seed = random_state)
    json_rand = dict()
    for payload in json_data:
        rand_inv = dict()
        rand_sig = dict()
        for key, (lo, hi) in feature_ranges.items():
            if hi <= lo:
                hi = lo + 1.0
            value = float(rand.uniform(low = lo, high = hi))
            if key in payload['invariants']:
                rand_inv[key] = value
            if key in payload['signatures']:
                rand_sig[key] = value

        json_rand[payload['name']] = {
            'invariants': rand_inv,
            'signatures': rand_sig,
            'events': list(payload['events']),
        }

    return json_rand

## cross-data vector feature generation
def _generate_vector_features(path_proc: str | Path, random_state: int = 42) -> dict[str, dict[str, Any]]:

    """ 
    Desc:
        Generate falsified features by resampling across datasets (bootstrap)
        and adding gaussian jitter. This keeps marginal distributions plausible
        while breaking the joint (x,z) -> y mapping.

    Args:
        path_proc: project directory containing processed dataset JSON files.
        random_state: random seed for reproducibility.

    Returns:
        dict keyed by dataset name with falsified payloads containing
        generated invariants/signatures and original events.

    Raises:
        FileNotFoundError: if the specified processed data directory does not
            exist or contains no JSON files.
    """

    ## load processed json data from specified directory
    json_path = Path(path_proc)
    if not json_path.exists() or not json_path.is_dir():
        raise FileNotFoundError(
            f"Processed data directory not found: {json_path!r}. "
            f"Ensure PATH_PROC points to an existing folder containing '*.json' files."
        )
    json_files = sorted(f for f in os.listdir(json_path) if f.endswith('.json'))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in processed data directory: {json_path!r}. "
            f"Ensure PATH_PROC points to a folder containing '*.json' files."
        )

    ## read each json file into a list of dictionaries
    json_data: list[dict[str, Any]] = []
    for file in json_files:
        name = os.path.splitext(file)[0]
        with open(json_path / file, 'r') as fp:
            payload = json.load(fp)
        json_data.append({
            'name': name,
            'invariants': payload.get('invariants', dict()),
            'signatures': payload.get('signatures', dict()),
            'events': payload.get('events', []),
        })

    ## build per-feature pools across all datasets for bootstrap resampling
    feature_pools: dict[str, np.ndarray] = dict()
    for payload in json_data:
        for key, val in {**payload['invariants'], **payload['signatures']}.items():
            try:
                x = float(val)
            except Exception:
                continue
            if not np.isfinite(x):
                continue
            feature_pools.setdefault(key, []).append(x)

    for key, pool in feature_pools.items():
        feature_pools[key] = np.array(pool, dtype=float)

    rand = np.random.RandomState(seed = random_state)

    json_rand = dict()
    for payload in json_data:
        rand_inv = dict()
        rand_sig = dict()

        for key, pool in feature_pools.items():
            if pool.size == 0:
                continue
            base = float(rand.choice(pool, size = 1))
            sigma = max(float(np.nanstd(pool, ddof = 0)), 1e-12)
            value = float(base + rand.normal(loc = 0.0, scale = sigma, size = 1))
            if key in payload['invariants']:
                rand_inv[key] = value
            if key in payload['signatures']:
                rand_sig[key] = value

        json_rand[payload['name']] = {
            'invariants': rand_inv,
            'signatures': rand_sig,
            'events': list(payload['events']),
        }

    return json_rand




## ----------------------------------------------------------------------
## falsification pipeline
## ----------------------------------------------------------------------
def json_falsifier(
    random_state: int = 42,
    force: bool = False,
    ) -> None:

    """
    Desc: run the falsification pipeline across all processed json_data.
          for each dataset in data/processed/, apply three falsification
          methods and save results to data/falsified/{name}.json. skips
          json_data that already have a corresponding falsified file.
    Args:
        random_state: random seed forwarded to all falsification methods.
        force: if true, overwrite existing falsified json files.
    Returns:
        None.
    """

    ## ensure falsified directory exists
    os.makedirs(name = PATH_FALS, exist_ok = True)

    target_permuted = _permute_target_mapping(
        path_proc = PATH_PROC,
        random_state = random_state,
    )

    random_generated = _generate_random_features(
        path_proc = PATH_PROC,
        random_state = random_state,
    )

    vector_generated = _generate_vector_features(
        path_proc = PATH_PROC,
        random_state = random_state,
    )

    logging.info(f"Found {len(target_permuted)} processed json_data.")

    for namedata, payload in target_permuted.items():
        fals_path = os.path.join(PATH_FALS, f"{namedata}.json")

        ## skip if already falsified unless force overwrite is requested
        if os.path.exists(fals_path) and not force:
            logging.info(
                f"{namedata} falsifications already exist at "
                f"{fals_path}. Skipping."
            )
            continue

        logging.info(f"Falsifying {namedata}...")

        data = {
            'target_remap': payload,
            'random_generate': random_generated.get(namedata, dict()),
            'vector_generate': vector_generated.get(namedata, dict()),
        }

        _save_to_json(data = data, path = fals_path)
        logging.info(f"{namedata} falsifications saved to {fals_path}")


## primary execution
if __name__ == '__main__':
    json_falsifier(force = True)
