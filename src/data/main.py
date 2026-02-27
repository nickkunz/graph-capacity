## libraries
import os
import sys
import json
import logging
import configparser
import pandas as pd
from pathlib import Path

## path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.data.processors import json_processor

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

METADATA = (
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
def _parse_metadata(namedata, metadata = METADATA):
    """ Return domain and discipline for a dataset name from METADATA. """

    ## parse metadata for the dataset name
    for name, domain, discipline in metadata:
        if name == namedata:
            return domain, discipline
    
    ## no name match found
    return 'Unknown', 'Unknown'

## metadata insertion
def _insert_metadata(data, namedata, metadata = METADATA):
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

    ## insert metadata fields
    data_obs = _insert_metadata(
        data = data_obs,
        namedata = namedata,
        metadata = METADATA
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
def data_builder(path_proc, path_data):

    """
    Desc:
        Create main table by loading processed JSON files, extracting the observation 
        with the maximum target value for each dataset, and concatenating results into
        a single table with consistent column order.
    
    Args:
        path_proc (str): Path to the directory containing processed JSON files.
        path_data (str): Path to save the resulting main table CSV file.
    
    Returns:
        None: The function saves the main table to disk and does not return any value.
    
    Raises:
        FileNotFoundError: If the processed JSON directory does not exist or contains no JSON files.
        ValueError: If no valid data is processed from the JSON files resulting in an empty.
    """
    
    data_list = list()
    invariant_order = list()
    signature_order = list()
    
    ## find each json file to process
    json_files = _find_json_payload(path_proc)
    logging.info(f"Found {len(json_files)} JSON files to process.")

    ## process each json file and append to list
    for file_name in json_files:
        namedata = os.path.splitext(file_name)[0]
        file_path = os.path.join(path_proc, file_name)
        logging.info(f"Processing {file_name}...")
        data_processed = _process_per_data(
            file_path = file_path,
            namedata = namedata,
            invariant_order = invariant_order,
            signature_order = signature_order
        )
        if not data_processed:
            continue
        data_list.append(data_processed)

    if not data_list:
        logging.error("No data was processed. Exiting.")
        return

    ## concatenate all dataset maxima and ensure consistent column order
    data_main = _process_all_data(data_list, invariant_order, signature_order)

    ## clean floating imprecision from processing and replace with exact zero
    numeric = data_main.select_dtypes(include = 'number')
    data_main.loc[:, numeric.columns] = numeric.mask(numeric.abs() < 1e-12, 0.0)

    ## save main data to disk
    data_main.to_csv(path_data, index = False)
    logging.info(f"Main table shape: {data_main.shape}")

## primary execution
if __name__ == '__main__':

    ## run json data processor to create payloads for main table creation
    try:
        logging.info("Running data processor...")
        json_processor()
        logging.info("Data processor completed.")
    except Exception as e:
        logging.warning(f"Failed to run processor: {e}.")
        logging.info("Continuing to create main table from existing processed data...")

    ## run main data builder that reads the json files and writes them to disk 
    try:
        os.makedirs(name = os.path.join(root, 'data'), exist_ok = True)
        path_data = os.path.join(root, 'data', 'main.csv')
        logging.info("Creating main table from processed JSON files...")
        data_builder(path_proc = os.path.join(root, PATH_PROC), path_data = path_data)
        logging.info("Main table creation completed.")
    except Exception as e:
        logging.error(f"Failed to create main table: {e}")
