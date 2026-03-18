## libraries
import os
import sys
import logging
import argparse
import configparser
from pathlib import Path

## path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.data.processors import json_processor
from src.data.perturbers import json_perturber
from src.data.falsifiers import json_falsifier
from src.data.builders import data_builder, data_saver

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

## command-line argument parsing
def _parse_args():
    parser = argparse.ArgumentParser(
        description = "Create the main table from processed datasets, optionally forcing a re-run."
    )
    parser.add_argument(
        "--force",
        action = "store_true",
        help = "Re-run all processing steps even if the output already exists.",
    )
    return parser.parse_args()

## primary execution
if __name__ == '__main__':

    ## parse command-line arguments
    args = _parse_args()

    ## run json data processor to create payloads for main table creation
    try:
        logging.info("Running data processor...")
        json_processor(force = args.force)
        logging.info("Data processor completed.")
    except Exception as e:
        logging.warning(f"Failed to run processor: {e}.")
        logging.info("Continuing to create main table from existing processed data...")

    ## run perturbation pipeline to create perturbed json payloads
    try:
        logging.info("Running data perturber...")
        json_perturber(force = args.force)
        logging.info("Data perturber completed.")
    except Exception as e:
        logging.warning(f"Failed to run perturber: {e}.")
        logging.info("Continuing to create main table from existing data...")

    ## run falsification pipeline to create falsified json payloads
    try:
        logging.info("Running data falsifier...")
        json_falsifier(force = args.force)
        logging.info("Data falsifier completed.")
    except Exception as e:
        logging.warning(f"Failed to run falsifier: {e}.")
        logging.info("Continuing to create main table from existing data...")

    ## run main data builder that reads the json files and writes them to disk
    try:
        os.makedirs(name = os.path.join(root, 'data'), exist_ok = True)
        path_data = os.path.join(root, 'data', 'main.csv')
        logging.info("Creating main table from processed JSON files...")
        data_main = data_builder(path = os.path.join(root, PATH_PROC))
        if data_main is not None:
            data_saver(data = data_main, path_data = path_data, force = args.force)
        logging.info("Main table creation completed.")
    except Exception as e:
        logging.error(f"Failed to create main table: {e}")
