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
from data.source.world import WorldBankProcessor
from data.source.wiki import WikiProcessor
from data.source.jodie import JodieProcessor
from data.source.overflow import OverflowProcessor
from data.source.email import EmailProcessor
from data.source.college import CollegeProcessor

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
NAME_WIKI = config['names']['NAME_WIKI']
NAME_JODIE = config['names']['NAME_JODIE']
NAME_OVERFLOW = config['names']['NAME_OVERFLOW']
NAME_EMAIL = config['names']['NAME_EMAIL']
NAME_COLLEGE = config['names']['NAME_COLLEGE']

URL_AMAZON = config['urls']['URL_AMAZON']
URL_FEDERAL = config['urls']['URL_FEDERAL']
URL_MOOC = config['urls']['URL_MOOC']
URL_WORLD_NETWORK = config['urls']['URL_WORLD_NETWORK']
URL_WORLD_METADATA = config['urls']['URL_WORLD_METADATA']
URL_WIKI_EVENTS = config['urls']['URL_WIKI_EVENTS']
URL_OVERFLOW = config['urls']['URL_OVERFLOW']
URL_EMAIL = config['urls']['URL_EMAIL']
URL_COLLEGE = config['urls']['URL_COLLEGE']

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

    ## --- wiki --- ##
    wiki_path = PATH_OUT + NAME_WIKI + '.json'
    if not os.path.exists(wiki_path):
        logging.info("Processing Wiki data...")
        data_wiki = WikiProcessor(
            url_events = URL_WIKI_EVENTS
        ).run()
        _save_to_json(data = data_wiki, path = wiki_path)
        logging.info(f"Wiki data saved to {wiki_path}")
    else:
        logging.info(f"Wiki data already exists at {wiki_path}. Skipping data source.")

    ## --- jodie --- ##
    jodie_path = PATH_OUT + NAME_JODIE + '.json'
    if not os.path.exists(jodie_path):
        logging.info("Processing JODIE data...")
        data_jodie = JodieProcessor(
            root_path=PATH_ROOT,
            name="wikipedia"  # JODIE dataset requires a specific name like 'wikipedia' or 'reddit'
        ).run()
        _save_to_json(data=data_jodie, path=jodie_path)
        logging.info(f"JODIE data saved to {jodie_path}")
    else:
        logging.info(f"JODIE data already exists at {jodie_path}. Skipping data source.")

    ## --- stackoverflow --- ##
    overflow_path = PATH_OUT + NAME_OVERFLOW + '.json'
    if not os.path.exists(overflow_path):
        logging.info("Processing Stack Overflow data...")
        data_overflow = OverflowProcessor(
            url = URL_OVERFLOW
        ).run()
        _save_to_json(data = data_overflow, path = overflow_path)
        logging.info(f"Stack Overflow data saved to {overflow_path}")
    else:
        logging.info(f"Stack Overflow data already exists at {overflow_path}. Skipping data source.")

    ## --- email --- ##
    email_path = PATH_OUT + NAME_EMAIL + '.json'
    if not os.path.exists(email_path):
        logging.info("Processing Email data...")
        data_email = EmailProcessor(
            url = URL_EMAIL
        ).run()
        _save_to_json(data = data_email, path = email_path)
        logging.info(f"Email data saved to {email_path}")
    else:
        logging.info(f"Email data already exists at {email_path}. Skipping data source.")

    ## --- college --- ##
    college_path = PATH_OUT + NAME_COLLEGE + '.json'
    if not os.path.exists(college_path):
        logging.info("Processing College Message data...")
        data_college = CollegeProcessor(
            url = URL_COLLEGE
        ).run()
        _save_to_json(data = data_college, path = college_path)
        logging.info(f"College Message data saved to {college_path}")
    else:
        logging.info(f"College Message data already exists at {college_path}. Skipping data source.")
