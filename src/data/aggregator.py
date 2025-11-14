## libraries
import os
import json
import pandas as pd
import logging
import sys
import configparser

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DIR_DEFAULT = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR_DEFAULT = os.path.join(PROJECT_ROOT, 'outputs', 'data')

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'conf', 'settings.ini')
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

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

# (name, domain, discipline)
DATASET_METADATA = [
    (NAME_AMAZON, 'Social', 'E-commerce'),
    (NAME_AUGER, 'Physics', 'Astrophysics'),
    (NAME_BITCOIN, 'Social', 'Finance'),
    (NAME_CELEGANS, 'Biology', 'Neuroscience'),
    (NAME_CHICKENPOX, 'Biology', 'Epidemiology'),
    (NAME_COLLEGE, 'Social', 'Communication'),
    (NAME_CROP, 'Biology', 'Ecology'),
    (NAME_EMAIL, 'Social', 'Communication'),
    (NAME_EPILEPSY, 'Biology', 'Medicine'),
    (NAME_FAERS, 'Biology', 'Medicine'),
    (NAME_FEDERAL, 'Social', 'Economics'),
    (NAME_GWOSC, 'Physics', 'Astrophysics'),
    (NAME_IDLING, 'Environment', 'Urban'),
    (NAME_JODIE, 'Social', 'Information Science'),
    (NAME_METRLA, 'Transport', 'Urban'),
    (NAME_MONTEVIDEO, 'Transport', 'Urban'),
    (NAME_MOOC, 'Social', 'Education'),
    (NAME_OVERFLOW, 'Social', 'Information Science'),
    (NAME_PEMSBAY, 'Transport', 'Urban'),
    (NAME_RAIN, 'Environment', 'Meteorology'),
    (NAME_RIVER, 'Environment', 'Hydrology'),
    (NAME_SEISMIC, 'Environment', 'Seismology'),
    (NAME_WIKI, 'Social', 'Information Science'),
    (NAME_WINDMILL, 'Engineering', 'Energy'),
    (NAME_WORLD, 'Social', 'Economics'),
]

# Configure logging
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

def create_master_dataframe(processed_dir, output_path):
    """
    Loads all intermediate .json objects from the processed directory into a single
    DataFrame and saves it as a CSV file.

    The DataFrame will have the following columns:
    - name: The name of the dataset.
    - domain: The broad domain of the dataset.
    - discipline: The specific discipline of the dataset.
    - All invariants from the JSON file.
    - day: The enumeration of the collection interval.
    - date: The calendar date of the event (if available).
    - target: The target count value of the events.
    """
    meta_map = get_dataset_meta()
    all_data = []
    invariant_order = []
    
    json_files = [f for f in os.listdir(processed_dir) if f.endswith('.json')]
    logging.info(f"Found {len(json_files)} JSON files to process.")

    for file_name in json_files:
        dataset_name = os.path.splitext(file_name)[0]
        file_path = os.path.join(processed_dir, file_name)
        
        logging.info(f"Processing {file_name}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        invariants = data.get('invariants', {})
        events = data.get('events', [])
        
        if not events:
            logging.warning(f"No events found in {file_name}. Skipping.")
            continue

        # Create a DataFrame for the events
        df_events = pd.DataFrame(events)

        if df_events.empty:
            logging.warning(f"Events in {file_name} are empty. Skipping.")
            continue

        df_events = df_events.reset_index(drop=True)

        if 'date' in df_events.columns:
            df_events['date'] = pd.to_datetime(df_events['date'], errors='coerce')
        else:
            df_events['date'] = pd.NaT

        if 'day' in df_events.columns:
            df_events['day'] = pd.to_numeric(df_events['day'], errors='coerce')
        else:
            df_events['day'] = df_events.index

        # Ensure targets are numeric so idxmax behaves as expected
        df_events['target'] = pd.to_numeric(df_events['target'], errors='coerce')

        # Drop rows where 'day' or 'target' could not be parsed
        df_events.dropna(subset=['day', 'target'], inplace=True)

        if df_events.empty:
            logging.warning(f"All events in {file_name} are invalid after parsing. Skipping.")
            continue

        # Keep only the event with the largest target for this invariant vector
        max_idx = df_events['target'].idxmax()
        df_events = df_events.loc[[max_idx]].copy()

        # Add other columns
        df_events['name'] = dataset_name
        df_events['discipline'] = meta_map.get(dataset_name, {}).get('discipline', 'Unknown')
        df_events['domain'] = meta_map.get(dataset_name, {}).get('domain', 'Unknown')

        # Add invariants as columns
        for key, value in invariants.items():
            df_events[key] = value
            if key not in invariant_order:
                invariant_order.append(key)

        log_day = df_events['day'].iloc[0]
        log_date = df_events['date'].iloc[0]

        logging.info(
            "Selected max event for %s with target %s (day=%s, date=%s)",
            dataset_name,
            df_events['target'].iloc[0],
            log_day,
            log_date
        )

        all_data.append(df_events)

    if not all_data:
        logging.error("No data was processed. Exiting.")
        return

    # Concatenate all DataFrames
    master_df = pd.concat(all_data, ignore_index=True)

    
    # Reorder columns
    invariant_cols = [col for col in invariant_order if col in master_df.columns]
    column_order = ['name', 'domain', 'discipline'] + invariant_cols + ['day', 'date', 'target']
    
    # Ensure all columns exist, fill missing with NaN
    for col in column_order:
        if col not in master_df.columns:
            master_df[col] = None
            
    master_df = master_df[column_order]

    # Save to CSV
    master_df.to_csv(output_path, index=False)
    logging.info(f"Master DataFrame saved to {output_path}")
    logging.info(f"DataFrame shape: {master_df.shape}")
    logging.info("Columns: " + ", ".join(master_df.columns))


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR_DEFAULT, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR_DEFAULT, 'master_dataset.csv')
    create_master_dataframe(PROCESSED_DIR_DEFAULT, output_path)
