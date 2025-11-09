## libraries
import os
import sys
import logging
import configparser

## modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import _save_to_json
from data.source.federal import FederalProcessor
from data.source.bitcoin import BitcoinProcessor
from data.source.amazon import AmazonProcessor
from data.source.mooc import MoocProcessor
from data.source.world import WorldBankProcessor
from data.source.wiki import WikiProcessor
from data.source.jodie import JodieProcessor
from data.source.overflow import OverflowProcessor
from data.source.email import EmailProcessor
from data.source.college import CollegeProcessor
from data.source.idling import IdlingProcessor
from data.source.windmill import WindmillProcessor
from data.source.metrla import MetrLaProcessor
from data.source.pemsbay import PemsBayProcessor
from data.source.montevideo import MontevideoProcessor
from data.source.celegans import CelegansProcessor

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
NAME_IDLING = config['names']['NAME_IDLING']
NAME_WINDMILL = config['names']['NAME_WINDMILL']
NAME_METRLA = config['names']['NAME_METRLA']
NAME_PEMSBAY = config['names']['NAME_PEMSBAY']
NAME_MONTEVIDEO = config['names']['NAME_MONTEVIDEO']
NAME_CELEGANS = config['names']['NAME_CELEGANS']

URL_AMAZON = config['urls']['URL_AMAZON'].strip('"')
URL_FEDERAL = config['urls']['URL_FEDERAL'].strip('"')
URL_MOOC = config['urls']['URL_MOOC'].strip('"')
URL_WORLD_NETWORK = config['urls']['URL_WORLD_NETWORK'].strip('"')
URL_WORLD_METADATA = config['urls']['URL_WORLD_METADATA'].strip('"')
URL_WIKI = config['urls']['URL_WIKI'].strip('"')
URL_OVERFLOW = config['urls']['URL_OVERFLOW'].strip('"')
URL_EMAIL = config['urls']['URL_EMAIL'].strip('"')
URL_COLLEGE = config['urls']['URL_COLLEGE'].strip('"')

FILE_IDLING_EVENTS = config['files']['FILE_IDLING_EVENTS'].strip('"')

## logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    stream = sys.stdout
)

## execute
if __name__ == '__main__':
    
    ## --- federal contracts --- ##
    federal_path = os.path.join(PATH_OUT, f"{NAME_FEDERAL}.json")
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

    ## --- mooc students --- ##
    mooc_path = os.path.join(PATH_OUT, f"{NAME_MOOC}.json")
    if not os.path.exists(mooc_path):
        logging.info("Processing MOOC data...")
        data_mooc = MoocProcessor(url = URL_MOOC).run()
        _save_to_json(data = data_mooc, path = mooc_path)
        logging.info(f"MOOC data saved to {mooc_path}")
    else:
        logging.info(f"MOOC data already exists at {mooc_path}. Skipping data source.")

    ## --- bitcoin trust --- ##
    bitcoin_path = os.path.join(PATH_OUT, f"{NAME_BITCOIN}.json")
    if not os.path.exists(bitcoin_path):
        logging.info("Processing Bitcoin data...")
        data_bitcoin = BitcoinProcessor(root_path = PATH_ROOT, name = NAME_BITCOIN).run()
        _save_to_json(data = data_bitcoin, path = bitcoin_path)
        logging.info(f"Bitcoin data saved to {bitcoin_path}")
    else:
        logging.info(f"Bitcoin data already exists at {bitcoin_path}. Skipping data source.")

    ## --- amazon reviews --- ##
    amazon_path = os.path.join(PATH_OUT, f"{NAME_AMAZON}.json")
    if not os.path.exists(amazon_path):
        logging.info("Processing Amazon data...")
        data_amazon = AmazonProcessor(root_path = PATH_ROOT, url = URL_AMAZON, name = NAME_AMAZON).run()
        _save_to_json(data = data_amazon, path = amazon_path)
        logging.info(f"Amazon data saved to {amazon_path}")
    else:
        logging.info(f"Amazon data already exists at {amazon_path}. Skipping data source.")

    ## --- world bank --- ##
    world_path = os.path.join(PATH_OUT, f"{NAME_WORLD}.json")
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

    ## --- math wiki --- ##
    wiki_path = os.path.join(PATH_OUT, f"{NAME_WIKI}.json")
    if not os.path.exists(wiki_path):
        logging.info("Processing Wiki data...")
        data_wiki = WikiProcessor(url_events = URL_WIKI).run()
        _save_to_json(data = data_wiki, path = wiki_path)
        logging.info(f"Wiki data saved to {wiki_path}")
    else:
        logging.info(f"Wiki data already exists at {wiki_path}. Skipping data source.")

    ## --- jodie wiki --- ##
    jodie_path = os.path.join(PATH_OUT, f"{NAME_JODIE}.json")
    if not os.path.exists(jodie_path):
        logging.info("Processing JODIE data...")
        data_jodie = JodieProcessor(root_path = PATH_ROOT, name = "wikipedia").run() ## name required
        _save_to_json(data=data_jodie, path=jodie_path)
        logging.info(f"JODIE data saved to {jodie_path}")
    else:
        logging.info(f"JODIE data already exists at {jodie_path}. Skipping data source.")

    ## --- MathOverflow --- ##
    overflow_path = os.path.join(PATH_OUT, f"{NAME_OVERFLOW}.json")
    if not os.path.exists(overflow_path):
        logging.info("Processing MathOverflow data...")
        data_overflow = OverflowProcessor(url = URL_OVERFLOW).run()
        _save_to_json(data = data_overflow, path = overflow_path)
        logging.info(f"MathOverflow data saved to {overflow_path}")
    else:
        logging.info(f"MathOverflow data already exists at {overflow_path}. Skipping data source.")

    ## --- eu-core email --- ##
    email_path = os.path.join(PATH_OUT, f"{NAME_EMAIL}.json")
    if not os.path.exists(email_path):
        logging.info("Processing EU-Core Email data...")
        data_email = EmailProcessor(url = URL_EMAIL).run()
        _save_to_json(data = data_email, path = email_path)
        logging.info(f"EU-Core Email data saved to {email_path}")
    else:
        logging.info(f"EU-Core Email data already exists at {email_path}. Skipping data source.")

    ## --- college --- ##
    college_path = os.path.join(PATH_OUT, f"{NAME_COLLEGE}.json")
    if not os.path.exists(college_path):
        logging.info("Processing UC Irvine College Message data...")
        data_college = CollegeProcessor(url = URL_COLLEGE).run()
        _save_to_json(data = data_college, path = college_path)
        logging.info(f"UC Irvine College Message data saved to {college_path}")
    else:
        logging.info(f"UC Irvine College Message data already exists at {college_path}. Skipping data source.")

    ## --- idling --- ##
    idling_path = os.path.join(PATH_OUT, f"{NAME_IDLING}.json")
    if not os.path.exists(idling_path):
        logging.info("Processing Halifax idling data...")
        data_idling = IdlingProcessor(events_path = FILE_IDLING_EVENTS).run()
        _save_to_json(data = data_idling, path = idling_path)
        logging.info(f"Halifax idling data saved to {idling_path}")
    else:
        logging.info(f"Halifax idling data already exists at {idling_path}. Skipping data source.")

    ## --- windmill --- ##
    windmill_path = os.path.join(PATH_OUT, f"{NAME_WINDMILL}.json")
    if not os.path.exists(windmill_path):
        logging.info("Processing Windmill data...")
        data_windmill = WindmillProcessor(raw_data_dir = os.path.join(PATH_ROOT, NAME_WINDMILL)).run()
        _save_to_json(data = data_windmill, path = windmill_path)
        logging.info(f"Windmill data saved to {windmill_path}")
    else:
        logging.info(f"Windmill data already exists at {windmill_path}. Skipping data source.")

    ## --- metr-la --- ##
    metrla_path = os.path.join(PATH_OUT, f"{NAME_METRLA}.json")
    if not os.path.exists(metrla_path):
        logging.info("Processing METR-LA data...")
        data_metrla = MetrLaProcessor(raw_data_dir = os.path.join(PATH_ROOT, NAME_METRLA)).run()
        _save_to_json(data = data_metrla, path = metrla_path)
        logging.info(f"METR-LA data saved to {metrla_path}")
    else:
        logging.info(f"METR-LA data already exists at {metrla_path}. Skipping data source.")

    ## --- pems-bay --- ##
    pemsbay_path = os.path.join(PATH_OUT, f"{NAME_PEMSBAY}.json")
    if not os.path.exists(pemsbay_path):
        logging.info("Processing PEMS-BAY data...")
        data_pemsbay = PemsBayProcessor(raw_data_dir = os.path.join(PATH_ROOT, NAME_PEMSBAY)).run()
        _save_to_json(data = data_pemsbay, path = pemsbay_path)
        logging.info(f"PEMS-BAY data saved to {pemsbay_path}")
    else:
        logging.info(f"PEMS-BAY data already exists at {pemsbay_path}. Skipping data source.")

    ## --- montevideo --- ##
    montevideo_path = os.path.join(PATH_OUT, f"{NAME_MONTEVIDEO}.json")
    if not os.path.exists(montevideo_path):
        logging.info("Processing Montevideo data...")
        data_montevideo = MontevideoProcessor().run()
        _save_to_json(data = data_montevideo, path = montevideo_path)
        logging.info(f"Montevideo data saved to {montevideo_path}")
    else:
        logging.info(f"Montevideo data already exists at {montevideo_path}. Skipping data source.")

    ## --- c. elegans --- ##
    celegans_path = os.path.join(PATH_OUT, f"{NAME_CELEGANS}.json")
    if not os.path.exists(celegans_path):
        logging.info("Processing C. Elegans data...")
        data_celegans = CelegansProcessor().run()
        _save_to_json(data = data_celegans, path = celegans_path)
        logging.info(f"C. Elegans data saved to {celegans_path}")
    else:
        logging.info(f"C. Elegans data already exists at {celegans_path}. Skipping data source.")

