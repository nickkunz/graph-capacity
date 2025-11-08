## libraries
import os
import sys
import logging
import configparser

## modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import _save_to_json
from data.source.bitcoin import BitcoinProcessor
from data.source.amazon import AmazonProcessor
from data.source.federal import FederalProcessor
from data.source.mooc import MoocProcessor
from data.source.worldbank import WorldBankProcessor

## config settings
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config', 'settings.ini'))

## config constants
PATH_ROOT = config['paths']['PATH_ROOT'].strip('"')
PATH_OUT = config['paths']['PATH_OUT'].strip('"')

NAME_AMAZON = config['names']['NAME_AMAZON']
NAME_BITCOIN = config['names']['NAME_BITCOIN']
NAME_FEDERAL = config['names']['NAME_FEDERAL']
NAME_MOOC = config['names']['NAME_MOOC']
NAME_WORLD = config['names']['NAME_WORLD']

URL_AMAZON = config['urls']['URL_AMAZON']
URL_FEDERAL = config['urls']['URL_FEDERAL']
URL_MOOC = config['urls']['URL_MOOC']
URL_WORLD_NETWORK = config['urls']['URL_WORLD_NETWORK']
URL_WORLD_METADATA = config['urls']['URL_WORLD_METADATA']

## logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    stream = sys.stdout
)

## execute
if __name__ == '__main__':
    
    ## --- federal --- ##
    federal_path = PATH_OUT + NAME_FEDERAL + '.json'
    if not os.path.exists(federal_path):
        logging.info("Processing Federal data...")
        data_federal = FederalProcessor(
            url = URL_FEDERAL,
            start_date = "2014-01-01",
            end_date = "2024-12-31"
        ).run()
        _save_to_json(data = data_federal, path = federal_path)
        logging.info(f"Federal data saved to {federal_path}")
    else:
        logging.info(f"Federal data already exists at {federal_path}. Skipping data source.")

    ## --- mooc --- ##
    mooc_path = PATH_OUT + NAME_MOOC + '.json'
    if not os.path.exists(mooc_path):
        logging.info("Processing MOOC data...")
        data_mooc = MoocProcessor(
            url = URL_MOOC
        ).run()
        _save_to_json(data = data_mooc, path = mooc_path)
        logging.info(f"MOOC data saved to {mooc_path}")
    else:
        logging.info(f"MOOC data already exists at {mooc_path}. Skipping data source.")

    ## --- bitcoin --- ##
    bitcoin_path = PATH_OUT + NAME_BITCOIN + '.json'
    if not os.path.exists(bitcoin_path):
        logging.info("Processing Bitcoin data...")
        data_bitcoin = BitcoinProcessor(
            root_path = PATH_ROOT
        ).run()
        _save_to_json(data = data_bitcoin, path = bitcoin_path)
        logging.info(f"Bitcoin data saved to {bitcoin_path}")
    else:
        logging.info(f"Bitcoin data already exists at {bitcoin_path}. Skipping data source.")

    ## --- amazon --- ##
    amazon_path = PATH_OUT + NAME_AMAZON + '.json'
    if not os.path.exists(amazon_path):
        logging.info("Processing Amazon data...")
        data_amazon = AmazonProcessor(
            root_path = PATH_ROOT, 
            url = URL_AMAZON, 
            name = NAME_AMAZON
        ).run()
        _save_to_json(data = data_amazon, path = amazon_path)
        logging.info(f"Amazon data saved to {amazon_path}")
    else:
        logging.info(f"Amazon data already exists at {amazon_path}. Skipping data source.")

    ## --- world bank --- ##
    world_path = PATH_OUT + NAME_WORLD + '.json'
    if not os.path.exists(world_path):
        logging.info("Processing World Bank data...")
        data_world = WorldBankProcessor(
            url_projects = URL_WORLD_NETWORK,
            url_meta = URL_WORLD_METADATA,
            start_year = "2014",
            end_year = "2024"
        ).run()
        _save_to_json(data = data_world, path = world_path)
        logging.info(f"World Bank data saved to {world_path}")
    else:
        logging.info(f"World Bank data already exists at {world_path}. Skipping data source.")
