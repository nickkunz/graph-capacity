## libraries
import os
import sys
import logging

## modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import _save_to_json
from data.scripts.trade.bitcoin import BitcoinProcessor
from data.scripts.trade.amazon import AmazonProcessor

## configuration paths
PATH_ROOT = "data/original/"
PATH_OUT = "data/processed/"

## logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    stream = sys.stdout
)

## execute
if __name__ == '__main__':
    
    ## --- bitcoin --- ##
    bitcoin_path = PATH_OUT + "bitcoin.json"
    if not os.path.exists(bitcoin_path):
        logging.info("Processing Bitcoin data...")
        data_bitcoin = BitcoinProcessor(root_path = PATH_ROOT).run()    
        _save_to_json(data = data_bitcoin, path = bitcoin_path)
        logging.info(f"Bitcoin data saved to {bitcoin_path}")
    else:
        logging.info(f"Bitcoin data already exists at {bitcoin_path}. Skipping.")

    ## --- amazon --- ##
    amazon_path = PATH_OUT + "amazon.json"
    if not os.path.exists(amazon_path):
        logging.info("Processing Amazon data...")
        data_amazon = AmazonProcessor(root_path = PATH_ROOT).run()  
        _save_to_json(data = data_amazon, path = amazon_path)
        logging.info(f"Amazon data saved to {amazon_path}")
    else:
        logging.info(f"Amazon data already exists at {amazon_path}. Skipping.")