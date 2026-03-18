## libraries
import os
import sys
import json
import tempfile
import logging
import configparser
import pandas as pd
from pathlib import Path

## path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    stream = sys.stdout
)
logger = logging.getLogger(__name__)

## configs
config = configparser.ConfigParser()
config.read(os.path.join(root, 'conf', 'settings.ini'))

## constants
PATH_PROC = config['paths']['PATH_PROC'].strip('"')
PATH_PERT = config['paths']['PATH_PERT'].strip('"')
PATH_FALS = config['paths']['PATH_FALS'].strip('"')
PATH_MAIN = os.path.join(root, 'data', 'main.csv')

PERT_SPEC = [
    {"key": "network_perturbed",    "type": "network",    "feat": "invariants"},
    {"key": "invariants_perturbed", "type": "invariants", "feat": "invariants"},
    {"key": "process_perturbed",    "type": "process",    "feat": "signatures"},
    {"key": "signatures_perturbed", "type": "signature",  "feat": "signatures"},
    {"key": "temporal_aggregated",  "type": "temporal",   "feat": "events"},
]

NAME_AMAZON = config['names']['NAME_AMAZON']
NAME_BITCOIN = config['names']['NAME_BITCOIN']
NAME_FEDERAL = config['names']['NAME_FEDERAL']
NAME_MOOC = config['names']['NAME_MOOC']
NAME_WORLD = config['names']['NAME_WORLD']
NAME_WIKI = config['names']['NAME_WIKI']
NAME_JODIE = config['names']['NAME_JODIE']
NAME_OVERFLOW = config['names']['NAME_OVERFLOW']
NAME_EMAIL = config['names']['NAME_EMAIL']
NAME_COLLEGE = config['names']['NAME_COLLEGE']
NAME_CELEGANS = config['names']['NAME_CELEGANS']
NAME_IDLING = config['names']['NAME_IDLING']
NAME_WINDMILL = config['names']['NAME_WINDMILL']
NAME_METRLA = config['names']['NAME_METRLA']
NAME_PEMSBAY = config['names']['NAME_PEMSBAY']
NAME_MONTEVIDEO = config['names']['NAME_MONTEVIDEO']
NAME_CROP = config['names']['NAME_CROP']
NAME_FAERS = config['names']['NAME_FAERS']
NAME_EPILEPSY = config['names']['NAME_EPILEPSY']
NAME_CHICKENPOX = config['names']['NAME_CHICKENPOX']
NAME_GWOSC = config['names']['NAME_GWOSC']
NAME_RIVER = config['names']['NAME_RIVER']
NAME_AUGER = config['names']['NAME_AUGER']
NAME_SEISMIC = config['names']['NAME_SEISMIC']
NAME_RAIN = config['names']['NAME_RAIN']

metadata = (
    (NAME_GWOSC, 'Earth & Physical Sciences', 'Cosmology'),
    (NAME_RIVER, 'Earth & Physical Sciences', 'Hydrology'),
    (NAME_AUGER, 'Earth & Physical Sciences', 'Physics'),
    (NAME_SEISMIC, 'Earth & Physical Sciences', 'Geology'),
    (NAME_RAIN, 'Earth & Physical Sciences', 'Meteorology'),
    (NAME_CELEGANS, 'Life Sciences & Medicine', 'Biology'),
    (NAME_CROP, 'Life Sciences & Medicine', 'Ecology'),
    (NAME_CHICKENPOX, 'Life Sciences & Medicine', 'Epidemiology'),
    (NAME_FAERS, 'Life Sciences & Medicine', 'Pharmacology'),
    (NAME_EPILEPSY, 'Life Sciences & Medicine', 'Neurology'),
    (NAME_FEDERAL, 'Trade & Institutions', 'Government'),
    (NAME_MOOC, 'Trade & Institutions', 'Education'),
    (NAME_AMAZON, 'Trade & Institutions', 'Commerce'),
    (NAME_BITCOIN, 'Trade & Institutions', 'Cryptocurrency'),
    (NAME_WORLD, 'Trade & Institutions', 'International Development'),
    (NAME_WIKI, 'Technology & Information', 'Knowledge Graphs'),
    (NAME_JODIE, 'Technology & Information', 'Interaction Networks'),
    (NAME_OVERFLOW, 'Technology & Information', 'Online Forums'),
    (NAME_EMAIL, 'Technology & Information', 'Email Exchanges'),
    (NAME_COLLEGE, 'Technology & Information', 'Social Networks'),
    (NAME_IDLING, 'Transportation & Infrastructure', 'Mobility'),
    (NAME_WINDMILL, 'Transportation & Infrastructure', 'Energy'),
    (NAME_METRLA, 'Transportation & Infrastructure', 'Traffic'),
    (NAME_PEMSBAY, 'Transportation & Infrastructure', 'Traffic II'),
    (NAME_MONTEVIDEO, 'Transportation & Infrastructure', 'Ridership'),
)

## observation rate maximum
def _rate_max(data, target = 'target'):

    """ Return the observation(s) with the maximum value in the target column. """

    idx_max = data[target].idxmax()
    data_max = data.loc[[idx_max]].copy()
    data_max.drop(
        columns = ['day', 'date'],
        inplace = True,
        errors = 'ignore'
    )
    return data_max

## metadata parsing
def _parse_metadata(namedata, metadata = metadata):

    """ Return domain and discipline for a dataset name from metadata. """

    ## parse metadata for the dataset name
    for name, domain, discipline in metadata:
        if name == namedata:
            return domain, discipline

    ## no name match found
    return 'Unknown', 'Unknown'

## metadata insertion
def _insert_metadata(data, namedata, metadata = metadata):

    """ Insert metadata fields and invariant/signature fields into the data. """

    ## parse metadata for the dataset name, warning if no match is found
    data = data.copy()
    domain, discipline = _parse_metadata(namedata = namedata, metadata = metadata)
    if domain == 'Unknown' and discipline == 'Unknown':
        logging.warning(f"No metadata match found for dataset '{namedata}'.")

    ## insert metadata
    data['name'] = namedata
    data['domain'] = domain
    data['discipline'] = discipline
    return data

## feature insertion
def _insert_features(data, invariants, invariant_order, signatures, signature_order):

    """ Insert invariant and signature fields into the data and track column order. """

    ## insert invariants and update column order
    data = data.copy()
    for key, value in invariants.items():
        data[key] = value
        if key not in invariant_order:
            invariant_order.append(key)

    ## insert signatures and update column order
    for key, value in signatures.items():
        data[key] = value
        if key not in signature_order:
            signature_order.append(key)

    return data

## json data discovery
def _find_json_payload(path_proc):

    """ Return sorted JSON filenames from the processed directory. """

    ## quality check to ensure processed path exists
    return sorted(f for f in os.listdir(path_proc) if f.endswith('.json'))

## json data normalization
def _load_json_payload(file_path):

    """ Load and unpack a processed dataset JSON payload. """

    ## quality check to ensure file exists
    with open(file_path, 'r') as f:
        data = json.load(f)

    return (
        data.get('invariants', {}),
        data.get('signatures', {}),
        data.get('events', list())
    )

# event normalization
def _normalize_events(events, target = 'target'):

    """ Ensuring target is numeric and date/day are parsed if present. """

    ## quality check to ensure target field and at least one event exists
    data_obs = pd.DataFrame(events)
    if data_obs.empty or target not in data_obs.columns:
        return None

    ## detect date and day fields
    data_obs = data_obs.reset_index(drop = True).copy()
    date_has = 'date' in data_obs.columns
    day_has = 'day' in data_obs.columns

    ## parse date and day fields if they exist, coercing errors
    if date_has:
        data_obs['date'] = pd.to_datetime(
            arg = data_obs['date'],
            errors = 'coerce'
        )
    else:
        data_obs['date'] = pd.Series(
            data = pd.NaT,
            index = data_obs.index,
            dtype = 'datetime64[ns]'
        )
    if day_has:
        data_obs['day'] = pd.to_numeric(
            arg = data_obs['day'],
            errors = 'coerce'
        ).astype('Int64')
    else:
        data_obs['day'] = pd.Series(
            data = pd.NA,
            index = data_obs.index,
            dtype = 'Int64'
        )

    ## ensure target field is numeric
    data_obs[target] = pd.to_numeric(data_obs[target], errors = 'coerce')

    ## drop events with missing target or date/day values
    subset = [target] + (['day'] if day_has else list())
    data_obs = data_obs.dropna(subset = subset)

    ## return None if no valid events remain
    return None if data_obs.empty else data_obs

## column reordering
def _reorder_features(data_main, invariant_order, signature_order):

    """ Return data with a consistent output column order. """

    ## build ordered feature columns from discovered invariant/signature keys
    feat_inv = [i for i in invariant_order if i in data_main.columns]
    feat_sig = [i for i in signature_order if i in data_main.columns]
    feat_ord = ['name', 'domain', 'discipline'] + feat_inv + feat_sig + ['target']

    ## add any missing ordered columns to maintain stable schema
    for i in feat_ord:
        if i not in data_main.columns:
            data_main[i] = None

    return data_main[feat_ord]

## main processing function per dataset
def _process_per_data(file_path, namedata, invariant_order, signature_order, target = 'target'):

    """ Load and process single JSON payload and output data with the maximum target value. """

    ## load json data
    invariants, signatures, events = _load_json_payload(
        file_path = file_path
    )

    ## normalize events
    data_obs = _normalize_events(
        events = events,
        target = target
    )

    ## return None if no valid events remain
    if data_obs is None:
        return None

    ## insert metadata fields
    data_obs = _insert_metadata(
        data = data_obs,
        namedata = namedata,
        metadata = metadata
    )

    ## insert invariant and signature fields
    data_obs = _insert_features(
        data = data_obs,
        invariants = invariants,
        invariant_order = invariant_order,
        signatures = signatures,
        signature_order = signature_order
    )

    data_max = _rate_max(
        data = data_obs,
        target = target
    )
    return data_max

## main processing function for all datasets
def _process_all_data(data_list, invariant_order, signature_order):

    """ Concatenate all dataset maxima and ensure consistent column order. """

    ## concatenate all dataset maxima and ensure consistent column order
    data_main = pd.concat(data_list, ignore_index = True) if data_list else pd.DataFrame()
    return _reorder_features(
        data_main = data_main,
        invariant_order = invariant_order,
        signature_order = signature_order
    )

## main processing function
def data_builder(path: str | Path) -> pd.DataFrame | None:

    """
    Desc:
        Create main table by loading JSON files and extracting the observation
        with the maximum target value for each dataset. Concatenate results into
        a single table with consistent column order.

    Args:
        path: Path to the directory containing JSON files.

    Returns:
        Main table dataframe, or None if no valid data was processed.

    Raises:
        FileNotFoundError: JSON directory does not exist or no files.
        ValueError: No valid data from JSON files resulting in an empty main table.
    """

    ## find each json file to process
    json_files = _find_json_payload(path_proc = path)

    ## process each json file and append to list
    data_list = list()
    invariant_order = list()
    signature_order = list()
    for file_name in json_files:
        namedata = os.path.splitext(file_name)[0]
        file_path = os.path.join(path, file_name)
        logging.info(f"Started processing {file_name}...")
        data_processed = _process_per_data(
            file_path = file_path,
            namedata = namedata,
            invariant_order = invariant_order,
            signature_order = signature_order
        )
        if data_processed is None:
            continue
        data_list.append(data_processed)
        logging.info(f"Finished processing {file_name}.")

    if not data_list:
        logging.error("No data was processed. Exiting.")
        return None

    ## concatenate all dataset maxima and ensure consistent column order
    data_main = _process_all_data(data_list, invariant_order, signature_order)

    ## clean floating imprecision from processing and replace with exact zero
    numeric = data_main.select_dtypes(include = 'number')
    data_main.loc[:, numeric.columns] = numeric.mask(numeric.abs() < 1e-12, 0.0)    
    return data_main

## main dataframe saver
def data_saver(data: pd.DataFrame, path_data: str | Path, force: bool = False) -> None:
    
    """ Save the main dataframe to disk and check for existing file depending on the force flag. """

    ## check if main table already exists
    if os.path.exists(path_data) and not force:
        logging.info(f"Main table already exists at {path_data}. Skipping.")
        return

    ## save main data to disk
    data.to_csv(path_data, index = False)
    logging.info(f"Main table saved at {path_data} with shape: {data.shape}")


## list json files in a directory with validation
def _list_json_files(path):

    ## validate path, directory status, and presence of json files
    path = Path(root) / path
    if not path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    ## list and sort json files and ensure at least one is found
    json_files = sorted([f for f in path.iterdir() if f.suffix == '.json'])
    if not json_files:
        raise ValueError(f"No JSON files found in directory: {path}")

    return json_files

## shared json key lister with consistency check across files
def _list_json_keys(path: str | Path) -> list[str]:

    ## list json files in directory with validation
    json_files = _list_json_files(path = path)

    ## track common keys across all json files 
    common_keys: set[str] | None = None
    inconsistent = False
    for json_file in json_files:
        with open(json_file, 'r') as fp:
            payload = json.load(fp)
        keys = set(payload.keys())
        if common_keys is None:
            common_keys = keys
        else:
            if common_keys != keys:
                inconsistent = True
            common_keys &= keys

    ## return sorted list of common keys or empty list if none found
    if common_keys is None:
        return list()
    if inconsistent:
        raise ValueError(
            "Inconsistent JSON payload keys across files. "
            f"Common keys: {sorted(common_keys)}"
        )
    return sorted(common_keys)

## json data normalization
def _load_json_payload(file_path):

    """ Load and unpack a processed dataset JSON payload. """

    ## quality check to ensure file exists
    with open(file_path, 'r') as f:
        data = json.load(f)

    return (
        data.get('invariants', {}),
        data.get('signatures', {}),
        data.get('events', list())
    )

## event normalization
def _normalize_events(events, target = 'target'):

    """ Ensuring target is numeric and date/day are parsed if present. """

    ## quality check to ensure target field and at least one event exists
    data_obs = pd.DataFrame(events)
    if data_obs.empty or target not in data_obs.columns:
        return None

    ## detect date and day fields
    data_obs = data_obs.reset_index(drop = True).copy()
    date_has = 'date' in data_obs.columns
    day_has = 'day' in data_obs.columns

    ## parse date and day fields if they exist, coercing errors
    if date_has:
        data_obs['date'] = pd.to_datetime(
            arg = data_obs['date'],
            errors = 'coerce'
        )
    else:
        data_obs['date'] = pd.Series(
            data = pd.NaT,
            index = data_obs.index,
            dtype = 'datetime64[ns]'
        )
    if day_has:
        data_obs['day'] = pd.to_numeric(
            arg = data_obs['day'],
            errors = 'coerce'
        ).astype('Int64')
    else:
        data_obs['day'] = pd.Series(
            data = pd.NA,
            index = data_obs.index,
            dtype = 'Int64'
        )

    ## ensure target field is numeric
    data_obs[target] = pd.to_numeric(data_obs[target], errors = 'coerce')

    ## drop events with missing target or date/day values
    subset = [target] + (['day'] if day_has else list())
    data_obs = data_obs.dropna(subset = subset)

    ## return None if no valid events remain
    return None if data_obs.empty else data_obs

## feature insertion
def _insert_features(data, invariants, invariant_order, signatures, signature_order):

    """ Insert invariant and signature fields into the data and track column order. """

    ## insert invariants and update column order
    data = data.copy()
    for key, value in invariants.items():
        data[key] = value
        if key not in invariant_order:
            invariant_order.append(key)

    ## insert signatures and update column order
    for key, value in signatures.items():
        data[key] = value
        if key not in signature_order:
            signature_order.append(key)

    return data

## collect perturbed data from json files
def _index_perturbs(path_pert: str) -> dict:

    """ Iterate over all perturbation json files and extract features into an index. """

    ## iterate over all json files in the perturbation directory
    index = dict()
    path = Path(path_pert)
    for json_path in sorted(path.glob("*.json")):
        data_name = json_path.stem
        with open(json_path, "r") as f:
            data = json.load(f)

        ## iterate over defined perturbation types and extract features into index
        for spec in PERT_SPEC:
            json_key, pert_type, feat_key = spec["key"], spec["type"], spec["feat"]

            # Support both old list-of-records format and new nested format
            records = data.get(json_key, [])
            if isinstance(records, dict):
                nested = []
                for method, intensities in records.items():
                    if isinstance(intensities, dict):
                        for intensity, rec in intensities.items():
                            if isinstance(rec, dict):
                                rec = dict(rec)
                            else:
                                rec = {}
                            rec["method"] = method
                            rec["intensity"] = intensity
                            nested.append(rec)
                    elif isinstance(intensities, list):
                        for rec in intensities:
                            if isinstance(rec, dict):
                                rec.setdefault("method", method)
                                nested.append(rec)
                records = nested

            for rec in records:
                method = rec.get("method")

                ## temporal: explicit method field in new records; fall back to aggregation for old records
                if pert_type == "temporal":
                    if "method" in rec:
                        method = rec["method"]
                        intensity = rec.get("intensity", rec.get("scale"))
                    else:
                        ## backward compat for records without method field
                        method = "aggregation"
                        intensity = rec.get("scale")
                else:
                    intensity = rec.get("intensity", rec.get("param"))

                idx_key = (pert_type, method, intensity)
                if idx_key not in index:
                    index[idx_key] = dict()

                ## create observation with dataset name and features
                obs = {"dataset": data_name}

                ## temporal aggregation: compute signatures from events list
                if pert_type == "temporal" and isinstance(rec.get("events"), list):
                    events_df = pd.DataFrame(rec["events"])
                    if "target" in events_df.columns and len(events_df) >= 2:
                        from src.vectorizers.signatures import ProcessSignatures
                        events_df["idx"] = range(len(events_df))
                        sigs = ProcessSignatures(
                            data = events_df, 
                            sort_by = ["idx"], 
                            target = "target"
                        )
                        obs.update(sigs.all())
                        obs["target"] = int(events_df["target"].max())
                else:
                    feat_val = rec.get(feat_key, dict()) if feat_key is not None else dict()
                    if isinstance(feat_val, dict):
                        obs.update(feat_val)
                index[idx_key][data_name] = obs
    return index

## ----------------------------------------------------------------------------
## ------ data loading for processed, perturbations, and falsifications -------
## ----------------------------------------------------------------------------

## load processed data
def load_processed_data(path_main: str | Path = PATH_MAIN) -> pd.DataFrame:

    """ Load the main dataframe from disk. """

    return pd.read_csv(filepath_or_buffer = path_main)

## create perturbation data from indexed perturbation data
def load_perturbed_data(path_pert: str | Path = PATH_PERT) -> dict:

    """
    Desc:
        Load precomputed perturbation payloads from disk and convert them
        into per-dataset DataFrames organized by payload key, method, and
        intensity.

    Args:
        path_pert: Path to perturbation JSON directory. If None, defaults
                   to the configured perturbation directory.

    Returns:
        Dict mapping payload JSON keys to method->intensity->DataFrame.

    Raises:
        FileNotFoundError: perturbation directory does not exist.
        ValueError: no perturbation JSON files were found.
    """

    ## deterministic sort for mixed key types
    def _sort_key(key: tuple) -> tuple[str, str, str]:
        return tuple(str(v) for v in key)

    ## build tables from indexed perturbation data
    data_dict = dict()
    index = _index_perturbs(path_pert)
    for key in sorted(index.keys(), key = _sort_key):
        pert_type, method, intensity = key
        data = pd.DataFrame(list(index[key].values()))
        data = data.sort_values("dataset").reset_index(drop = True)
        data.insert(1, "method", method)
        data.insert(2, "intensity", intensity)
        data_dict[key] = data
        logger.info(
            f"Table ({pert_type}, {method}, {intensity}): "
            f"{len(data)} datasets"
        )

    ## lookup: pert_type -> json_key
    _type_to_key = {spec["type"]: spec["key"] for spec in PERT_SPEC}

    ## payload schema: {json_key: {method: {intensity: DataFrame}}}
    dict_data = dict()
    for (pert_type, method, intensity), data in data_dict.items():
        json_key = _type_to_key.get(pert_type, pert_type)
        if json_key not in dict_data:
            dict_data[json_key] = dict()
        if method not in dict_data[json_key]:
            dict_data[json_key][method] = dict()
        dict_data[json_key][method][intensity] = data
    logger.info(f"Created {len(dict_data)} perturbation groups from {path_pert}")
    return dict_data

## load falsified data
def load_falsified_data(path_fals: str | Path = PATH_FALS) -> dict:

    """
    Desc:
        Load precomputed falsified datasets from disk using data_builder
        to construct per-dataset evaluation dataframes per falsification method.

    Args:
        path_fals: directory containing falsified dataset json files.

    Returns:
        dict mapping dataset name to a nested dict of falsification method
        names to dataframes.

    Raises:
        FileNotFoundError: falsified data directory does not exist or
                           has no json files.
    """

    ## list json files in falsification directory with validation
    json_files = _list_json_files(path_fals)
    methods = _list_json_keys(path_fals)

    ## build falsified dataframes by processing each json file with data_builder()
    dict_data = dict()
    for file_path in json_files:
        name = file_path.stem
        with open(file_path, 'r') as fp:
            payload = json.load(fp)

        dict_data_meth = dict()
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in methods:
                if i not in payload or not isinstance(payload[i], dict):
                    continue
                payload = {
                    'invariants': payload[i].get('invariants', dict()),
                    'signatures': payload[i].get('signatures', dict()),
                    'events': payload[i].get('events', list()),
                }
                out_path = os.path.join(temp_dir, file_path.name)
                with open(out_path, 'w') as fp:
                    json.dump(payload, fp)
                data_false = data_builder(path = temp_dir)
                if data_false is not None and not data_false.empty:
                    dict_data_meth[i] = data_false
                os.remove(out_path)

        if dict_data_meth:
            dict_data[name] = dict_data_meth

    return dict_data
