## libraries
import os
import json
import pandas as pd
import numpy as np
import logging
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DIR_DEFAULT = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR_DEFAULT = os.path.join(PROJECT_ROOT, 'outputs')

## logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    stream = sys.stdout
)

## metadata
def get_dataset_meta():
    return {
        'amazon': {'discipline': 'Social', 'domain': 'E-commerce'},
        'auger': {'discipline': 'Physics', 'domain': 'Astrophysics'},
        'bitcoin': {'discipline': 'Social', 'domain': 'Finance'},
        'celegans': {'discipline': 'Biology', 'domain': 'Neuroscience'},
        'chickenpox': {'discipline': 'Biology', 'domain': 'Epidemiology'},
        'college': {'discipline': 'Social', 'domain': 'Communication'},
        'crop': {'discipline': 'Biology', 'domain': 'Ecology'},
        'email': {'discipline': 'Social', 'domain': 'Communication'},
        'epilepsy': {'discipline': 'Biology', 'domain': 'Medicine'},
        'faers': {'discipline': 'Biology', 'domain': 'Medicine'},
        'federal': {'discipline': 'Social', 'domain': 'Economics'},
        'gwosc': {'discipline': 'Physics', 'domain': 'Astrophysics'},
        'idling': {'discipline': 'Environment', 'domain': 'Urban'},
        'jodie': {'discipline': 'Social', 'domain': 'Information Science'},
        'metrla': {'discipline': 'Transport', 'domain': 'Urban'},
        'montevideo': {'discipline': 'Transport', 'domain': 'Urban'},
        'mooc': {'discipline': 'Social', 'domain': 'Education'},
        'overflow': {'discipline': 'Social', 'domain': 'Information Science'},
        'pemsbay': {'discipline': 'Transport', 'domain': 'Urban'},
        'rain': {'discipline': 'Environment', 'domain': 'Meteorology'},
        'river': {'discipline': 'Environment', 'domain': 'Hydrology'},
        'seismic': {'discipline': 'Environment', 'domain': 'Seismology'},
        'wiki': {'discipline': 'Social', 'domain': 'Information Science'},
        'windmill': {'discipline': 'Engineering', 'domain': 'Energy'},
        'world': {'discipline': 'Social', 'domain': 'Economics'},
    }

## main builder
def create_master_dataset(processed_dir, output_path):

    meta = get_dataset_meta()
    rows = []
    invariant_keys = []

    json_files = sorted(f for f in os.listdir(processed_dir) if f.endswith(".json"))
    logging.info(f"found {len(json_files)} datasets.")

    for fname in json_files:

        dataset = os.path.splitext(fname)[0]
        fpath = os.path.join(processed_dir, fname)

        logging.info(f"loading {fname}")

        with open(fpath, "r") as f:
            obj = json.load(f)

        invariants = obj.get("invariants", {})
        events = obj.get("events", [])

        if not isinstance(events, list) or len(events) == 0:
            logging.warning(f"no events in {fname} — skipping.")
            continue

        df_e = pd.DataFrame(events)
        df_e["target"] = pd.to_numeric(df_e["target"], errors="coerce")
        df_e.dropna(subset=["target"], inplace=True)

        if df_e.empty:
            logging.warning(f"all events invalid in {fname} — skipping.")
            continue

        r_max = df_e["target"].max()

        row = {
            "name": dataset,
            "discipline": meta.get(dataset, {}).get("discipline", "Unknown"),
            "domain": meta.get(dataset, {}).get("domain", "Unknown"),
            "r_max": r_max
        }

        ## add invariants
        for k, v in invariants.items():
            row[k] = v
            if k not in invariant_keys:
                invariant_keys.append(k)

        rows.append(row)

    ## build dataframe
    df = pd.DataFrame(rows)
    logging.info(f"initial shape: {df.shape}")

    df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)
    logging.info(f"after deduplication: {df.shape}")

    ## standardize invariants
    inv_cols = [c for c in invariant_keys if c in df.columns]
    if len(inv_cols) > 0:
        X = df[inv_cols].astype(float)
        mean = X.mean()
        std = X.std(ddof=0).replace(0, np.nan)
        df[inv_cols] = ((X - mean) / std).fillna(0.0)
        logging.info("standardized invariants.")

    ## compute omega_log
    u = np.log1p(df["r_max"])
    u_min, u_max = u.min(), u.max()

    if u_max > u_min:
        df["omega_log"] = ((u - u_min) / (u_max - u_min)).clip(0, 1)
    else:
        df["omega_log"] = np.nan
        logging.warning("could not compute omega_log.")

    ## reorder columns
    ordered = ["name","discipline","domain"] + inv_cols + ["r_max","omega_log"]
    df = df[ordered]

    ## save valid dataset
    df.to_csv(output_path, index=False)
    logging.info(f"saved master dataset to {output_path}")

    # ------------------------------------------------------------
    # create structurally invalid randomized version
    # ------------------------------------------------------------
    ## this preserves invariants and targets but destroys all mapping
    logging.info("creating structurally invalid randomized dataset.")

    rng = np.random.default_rng(12345)

    df_invalid = df.copy()

    ## independent permutations for invariants and targets
    perm_invariants = rng.permutation(len(df))
    perm_targets = rng.permutation(len(df))

    ## break mapping: invariants come from perm_invariants, targets from perm_targets
    df_invalid[inv_cols] = df[inv_cols].iloc[perm_invariants].to_numpy()
    df_invalid["omega_log"] = df["omega_log"].iloc[perm_targets].to_numpy()
    df_invalid["r_max"] = df["r_max"].iloc[perm_targets].to_numpy()
    df_invalid["name"] = df["name"].iloc[perm_targets].to_numpy()

    invalid_path = output_path.replace(".csv", "_struct_invalid.csv")
    df_invalid.to_csv(invalid_path, index=False)

    logging.info(f"saved structurally invalid dataset to {invalid_path}")
    logging.info(f"valid shape: {df.shape}, invalid shape: {df_invalid.shape}")


## entry
if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR_DEFAULT, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR_DEFAULT, "master_dataset_rand.csv")
    create_master_dataset(PROCESSED_DIR_DEFAULT, output_path)
