## libraries
import os
import json
import pandas as pd
import logging
import sys

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

def create_master_dataframe(processed_dir, output_path):
    """
    Loads all intermediate .json objects from the processed directory into a single
    DataFrame and saves it as a CSV file.

    The DataFrame will have the following columns:
    - name: The name of the dataset.
    - discipline: The academic discipline of the dataset.
    - domain: The specific domain of the dataset.
    - All invariants from the JSON file.
    - day: The timestamp of the event.
    - target: The target value of the event.
    """
    meta_map = get_dataset_meta()
    all_data = []
    
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
        
        # Convert 'day' to datetime, coercing errors
        df_events['day'] = pd.to_datetime(df_events['day'], errors='coerce')
        
        # Drop rows where 'day' could not be parsed
        df_events.dropna(subset=['day'], inplace=True)
        
        # Add other columns
        df_events['name'] = dataset_name
        df_events['discipline'] = meta_map.get(dataset_name, {}).get('discipline', 'Unknown')
        df_events['domain'] = meta_map.get(dataset_name, {}).get('domain', 'Unknown')
        
        # Add invariants as columns
        for key, value in invariants.items():
            df_events[key] = value
            
        all_data.append(df_events)

    if not all_data:
        logging.error("No data was processed. Exiting.")
        return

    # Concatenate all DataFrames
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns
    invariant_cols = list(invariants.keys())
    column_order = ['name', 'discipline', 'domain'] + invariant_cols + ['day', 'target']
    
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
    # Define paths
    PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'processed'))
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'master_dataset.csv')
    
    create_master_dataframe(PROCESSED_DIR, OUTPUT_PATH)
