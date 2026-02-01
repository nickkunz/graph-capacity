## libraries
import os
import json
import pandas as pd
import logging
import sys
import configparser

## paths and config
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DIR_DEFAULT = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR_DEFAULT = DATA_DIR
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'conf', 'settings.ini')
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

## dataset name constants
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

## dataset metadata mapping (name, discipline, domain)
DATASET_METADATA = [
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
]

## logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def get_dataset_meta():
    """
    Returns a dictionary mapping dataset names to their discipline and domain.
    """
    return {
        name: {'discipline': discipline, 'domain': domain}
        for name, domain, discipline in DATASET_METADATA
    }

def normalize_event_fields(events):
    """Return a normalized DataFrame plus metadata flags for day/date availability."""
    df_events = pd.DataFrame(events)
    if df_events.empty:
        return None, False, False

    df_events = df_events.reset_index(drop=True)

    has_date = 'date' in df_events.columns
    if has_date:
        df_events['date'] = pd.to_datetime(df_events['date'], errors='coerce')
    else:
        df_events['date'] = pd.Series(pd.NaT, index=df_events.index, dtype='datetime64[ns]')

    has_day = 'day' in df_events.columns
    if has_day:
        df_events['day'] = pd.to_numeric(df_events['day'], errors='coerce').astype('Int64')
    else:
        df_events['day'] = pd.Series(pd.NA, index=df_events.index, dtype='Int64')

    df_events['target'] = pd.to_numeric(df_events['target'], errors='coerce')

    required_columns = ['target']
    if has_day:
        required_columns.append('day')

    df_events.dropna(subset=required_columns, inplace=True)

    if df_events.empty:
        return None, has_day, has_date

    return df_events, has_day, has_date

def add_metadata_columns(df_events, dataset_name, meta_map, invariants, invariant_order, signatures, descriptor_order):
    df_events = df_events.copy()
    df_events['name'] = dataset_name
    df_events['discipline'] = meta_map.get(dataset_name, {}).get('discipline', 'Unknown')
    df_events['domain'] = meta_map.get(dataset_name, {}).get('domain', 'Unknown')

    for key, value in invariants.items():
        df_events[key] = value
        if key not in invariant_order:
            invariant_order.append(key)

    for key, value in signatures.items():
        df_events[key] = value
        if key not in descriptor_order:
            descriptor_order.append(key)

    return df_events

def select_max_event(df_events, has_day, has_date):
    max_idx = df_events['target'].idxmax()
    df_max = df_events.loc[[max_idx]].copy()
    log_day = df_max['day'].iloc[0] if has_day else None
    log_date = df_max['date'].iloc[0] if has_date else None
    df_max.drop(columns=['day', 'date'], inplace=True, errors='ignore')
    return df_max, log_day, log_date

def process_dataset(file_path, dataset_name, meta_map, invariant_order, descriptor_order):
    with open(file_path, 'r') as f:
        data = json.load(f)

    invariants = data.get('invariants', {})
    signatures = data.get('signatures', {})
    events = data.get('events', [])

    if not events:
        logging.warning(f"No events found in {os.path.basename(file_path)}. Skipping.")
        return None

    df_events, has_day, has_date = normalize_event_fields(events)

    if df_events is None:
        logging.warning(f"All events in {os.path.basename(file_path)} are invalid after parsing. Skipping.")
        return None

    df_events = add_metadata_columns(
        df_events,
        dataset_name,
        meta_map,
        invariants,
        invariant_order,
        signatures,
        descriptor_order
    )
    df_max, log_day, log_date = select_max_event(df_events, has_day, has_date)

    logging.info(
        "Selected max event for %s with target %s (day=%s, date=%s)",
        dataset_name,
        df_max['target'].iloc[0],
        log_day,
        log_date
    )

    return df_events, df_max

def build_all_dataframe(all_data, invariant_order, descriptor_order):
    master_all_df = pd.concat(all_data, ignore_index=True)
    invariant_cols = [col for col in invariant_order if col in master_all_df.columns]
    descriptor_cols = [col for col in descriptor_order if col in master_all_df.columns]
    all_column_order = ['name', 'domain', 'discipline'] + invariant_cols + descriptor_cols + ['day', 'date', 'target']

    for col in all_column_order:
        if col not in master_all_df.columns:
            master_all_df[col] = None

    return master_all_df[all_column_order]

def build_max_dataframe(max_data, invariant_order, descriptor_order):
    master_max_df = pd.concat(max_data, ignore_index=True) if max_data else pd.DataFrame()
    invariant_cols = [col for col in invariant_order if col in master_max_df.columns]
    descriptor_cols = [col for col in descriptor_order if col in master_max_df.columns]
    max_column_order = ['name', 'domain', 'discipline'] + invariant_cols + descriptor_cols + ['target']

    for col in max_column_order:
        if col not in master_max_df.columns:
            master_max_df[col] = None

    return master_max_df[max_column_order]

def create_master_dataframe(processed_dir, output_all_path, output_max_path):
    """
    Loads all intermediate .json objects from the processed directory into two CSV files:

    1. data_all.csv: one row per observed event across all datasets, retaining both day
         and date information for each measurement. Invariants and process signatures are
         appended as dataset-level columns alongside metadata.
    2. data.csv: one row per dataset capturing only the event with the largest target
         value. These rows omit the day and date columns while preserving invariants and
         signatures.
    """
    meta_map = get_dataset_meta()
    all_data = []
    max_data = []
    invariant_order = []
    descriptor_order = []
    
    json_files = [f for f in os.listdir(processed_dir) if f.endswith('.json')]
    logging.info(f"Found {len(json_files)} JSON files to process.")

    for file_name in json_files:
        dataset_name = os.path.splitext(file_name)[0]
        file_path = os.path.join(processed_dir, file_name)

        logging.info(f"Processing {file_name}...")

        processed = process_dataset(file_path, dataset_name, meta_map, invariant_order, descriptor_order)

        if not processed:
            continue

        df_events, df_max = processed
        all_data.append(df_events)
        max_data.append(df_max)

    if not all_data:
        logging.error("No data was processed. Exiting.")
        return

    master_max_df = build_max_dataframe(max_data, invariant_order, descriptor_order)

    master_max_df.to_csv(output_max_path, index=False)
    logging.info(f"Max observations DataFrame saved to {output_max_path}")
    logging.info(f"data.csv shape: {master_max_df.shape}")

## main execution
if __name__ == '__main__':
    try:
        from src.data.processor import run_processor
        logging.info("Running data processor...")
        run_processor()
    except Exception as e:
        logging.warning(f"Failed to run processor: {e}. Proceeding with existing data.")

    os.makedirs(
        name = DATA_DIR, 
        exist_ok = True
    )
    
    output_max_path = os.path.join(DATA_DIR, 'data.csv')
    
    create_master_dataframe(PROCESSED_DIR_DEFAULT, None, output_max_path)
