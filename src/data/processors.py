## libraries
import os
import sys
import logging
import configparser

## project root
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root not in sys.path:
    sys.path.append(root)

## modules
from src.data.helpers import _save_to_json
from src.data.loaders.federal import FederalProcessor
from src.data.loaders.bitcoin import BitcoinProcessor
from src.data.loaders.amazon import AmazonProcessor
from src.data.loaders.mooc import MoocProcessor
from src.data.loaders.world import WorldBankProcessor
from src.data.loaders.wiki import WikiProcessor
from src.data.loaders.jodie import JodieProcessor
from src.data.loaders.overflow import OverflowProcessor
from src.data.loaders.email import EmailProcessor
from src.data.loaders.celegans import CelegansProcessor
from src.data.loaders.college import CollegeProcessor
from src.data.loaders.idling import IdlingProcessor
from src.data.loaders.windmill import WindmillProcessor
from src.data.loaders.metrla import MetrLaProcessor
from src.data.loaders.pemsbay import PemsBayProcessor
from src.data.loaders.montevideo import MontevideoProcessor
from src.data.loaders.crop import CropProcessor
from src.data.loaders.faers import FaersProcessor
from src.data.loaders.epilepsy import EpilepsyProcessor
from src.data.loaders.gwosc import GwoscProcessor
from src.data.loaders.river import NwisProcessor
from src.data.loaders.auger import AugerProcessor
from src.data.loaders.seismic import SeismicProcessor
from src.data.loaders.rain import RainProcessor
from src.data.loaders.chickenpox import ChickenpoxProcessor

## configs
config = configparser.ConfigParser()
config.read(os.path.join(root, 'conf', 'settings.ini'))

## constants
PATH_ROOT = config['paths']['PATH_ROOT'].strip('"')
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

URL_AMAZON = config['urls']['URL_AMAZON'].strip('"')
URL_FEDERAL = config['urls']['URL_FEDERAL'].strip('"')
URL_MOOC = config['urls']['URL_MOOC'].strip('"')
URL_WORLD_NETWORK = config['urls']['URL_WORLD_NETWORK'].strip('"')
URL_WORLD_METADATA = config['urls']['URL_WORLD_METADATA'].strip('"')
URL_WIKI = config['urls']['URL_WIKI'].strip('"')
URL_OVERFLOW = config['urls']['URL_OVERFLOW'].strip('"')
URL_EMAIL = config['urls']['URL_EMAIL'].strip('"')
URL_COLLEGE = config['urls']['URL_COLLEGE'].strip('"')
URL_CROP_SAMPLING = config['urls']['URL_CROP_SAMPLING'].strip('"')
URL_CROP_FIELD = config['urls']['URL_CROP_FIELD'].strip('"')
URL_FAERS = config['urls']['URL_FAERS'].strip('"')
URL_EPILEPSY = config['urls']['URL_EPILEPSY'].strip('"')
URL_CHICKENPOX_EVENTS = config['urls']['URL_CHICKENPOX_EVENTS'].strip('"')
URL_GWOSC = config['urls']['URL_GWOSC'].strip('"')
URL_RIVER_SITE = config['urls']['URL_RIVER_SITE'].strip('"')
URL_RIVER_IV = config['urls']['URL_RIVER_IV'].strip('"')
URL_AUGER_NETWORK = config['urls']['URL_AUGER_NETWORK'].strip('"')
URL_AUGER_EVENTS = config['urls']['URL_AUGER_EVENTS'].strip('"')
URL_SEISMIC_NETWORK = config['urls']['URL_SEISMIC_NETWORK'].strip('"')
URL_SEISMIC_EVENTS = config['urls']['URL_SEISMIC_EVENTS'].strip('"')

## logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    stream = sys.stdout
)

## main
def processor():

    ## ensure processed directory exists
    os.makedirs(PATH_PROC, exist_ok = True)
    
    ## --- federal contracts --- ##
    federal_path = os.path.join(PATH_PROC, f"{NAME_FEDERAL}.json")
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
    mooc_path = os.path.join(PATH_PROC, f"{NAME_MOOC}.json")
    if not os.path.exists(mooc_path):
        logging.info("Processing MOOC data...")
        data_mooc = MoocProcessor(url = URL_MOOC).run()
        _save_to_json(data = data_mooc, path = mooc_path)
        logging.info(f"MOOC data saved to {mooc_path}")
    else:
        logging.info(f"MOOC data already exists at {mooc_path}. Skipping data source.")

    ## --- bitcoin trust --- ##
    bitcoin_path = os.path.join(PATH_PROC, f"{NAME_BITCOIN}.json")
    if not os.path.exists(bitcoin_path):
        logging.info("Processing Bitcoin data...")
        data_bitcoin = BitcoinProcessor(root_path = PATH_ROOT, name = NAME_BITCOIN).run()
        _save_to_json(data = data_bitcoin, path = bitcoin_path)
        logging.info(f"Bitcoin data saved to {bitcoin_path}")
    else:
        logging.info(f"Bitcoin data already exists at {bitcoin_path}. Skipping data source.")

    ## --- amazon reviews --- ##
    amazon_path = os.path.join(PATH_PROC, f"{NAME_AMAZON}.json")
    if not os.path.exists(amazon_path):
        logging.info("Processing Amazon data...")
        data_amazon = AmazonProcessor(root_path = PATH_ROOT, url = URL_AMAZON, name = NAME_AMAZON).run()
        _save_to_json(data = data_amazon, path = amazon_path)
        logging.info(f"Amazon data saved to {amazon_path}")
    else:
        logging.info(f"Amazon data already exists at {amazon_path}. Skipping data source.")

    ## --- world bank --- ##
    world_path = os.path.join(PATH_PROC, f"{NAME_WORLD}.json")
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
    wiki_path = os.path.join(PATH_PROC, f"{NAME_WIKI}.json")
    if not os.path.exists(wiki_path):
        logging.info("Processing Wiki data...")
        data_wiki = WikiProcessor(url = URL_WIKI, name = "wikivital_mathematics.json").run()
        _save_to_json(data = data_wiki, path = wiki_path)
        logging.info(f"Wiki data saved to {wiki_path}")
    else:
        logging.info(f"Wiki data already exists at {wiki_path}. Skipping data source.")

    ## --- jodie wiki --- ##
    jodie_path = os.path.join(PATH_PROC, f"{NAME_JODIE}.json")
    if not os.path.exists(jodie_path):
        logging.info("Processing JODIE data...")
        data_jodie = JodieProcessor(root_path = PATH_ROOT, name = "wikipedia").run() ## name required
        _save_to_json(data=data_jodie, path=jodie_path)
        logging.info(f"JODIE data saved to {jodie_path}")
    else:
        logging.info(f"JODIE data already exists at {jodie_path}. Skipping data source.")

    ## --- mathoverflow --- ##
    overflow_path = os.path.join(PATH_PROC, f"{NAME_OVERFLOW}.json")
    if not os.path.exists(overflow_path):
        logging.info("Processing MathOverflow data...")
        data_overflow = OverflowProcessor(url = URL_OVERFLOW).run()
        _save_to_json(data = data_overflow, path = overflow_path)
        logging.info(f"MathOverflow data saved to {overflow_path}")
    else:
        logging.info(f"MathOverflow data already exists at {overflow_path}. Skipping data source.")

    ## --- eu-core email --- ##
    email_path = os.path.join(PATH_PROC, f"{NAME_EMAIL}.json")
    if not os.path.exists(email_path):
        logging.info("Processing EU-Core Email data...")
        data_email = EmailProcessor(url = URL_EMAIL).run()
        _save_to_json(data = data_email, path = email_path)
        logging.info(f"EU-Core Email data saved to {email_path}")
    else:
        logging.info(f"EU-Core Email data already exists at {email_path}. Skipping data source.")

    ## --- college --- ##
    college_path = os.path.join(PATH_PROC, f"{NAME_COLLEGE}.json")
    if not os.path.exists(college_path):
        logging.info("Processing UC Irvine College Message data...")
        data_college = CollegeProcessor(url = URL_COLLEGE).run()
        _save_to_json(data = data_college, path = college_path)
        logging.info(f"UC Irvine College Message data saved to {college_path}")
    else:
        logging.info(f"UC Irvine College Message data already exists at {college_path}. Skipping data source.")

    ## --- idling --- ##
    idling_path = os.path.join(PATH_PROC, f"{NAME_IDLING}.json")
    if not os.path.exists(idling_path):
        logging.info("Processing Halifax idling data...")
        data_idling = IdlingProcessor(path_events = PATH_ROOT + "idling/").run()
        _save_to_json(data = data_idling, path = idling_path)
        logging.info(f"Halifax idling data saved to {idling_path}")
    else:
        logging.info(f"Halifax idling data already exists at {idling_path}. Skipping data source.")

    ## --- windmill --- ##
    windmill_path = os.path.join(PATH_PROC, f"{NAME_WINDMILL}.json")
    if not os.path.exists(windmill_path):
        logging.info("Processing Windmill data...")
        data_windmill = WindmillProcessor(raw_data_dir = os.path.join(PATH_ROOT, NAME_WINDMILL)).run()
        _save_to_json(data = data_windmill, path = windmill_path)
        logging.info(f"Windmill data saved to {windmill_path}")
    else:
        logging.info(f"Windmill data already exists at {windmill_path}. Skipping data source.")

    ## --- metr-la --- ##
    metrla_path = os.path.join(PATH_PROC, f"{NAME_METRLA}.json")
    if not os.path.exists(metrla_path):
        logging.info("Processing METR-LA data...")
        data_metrla = MetrLaProcessor(raw_data_dir = os.path.join(PATH_ROOT, NAME_METRLA)).run()
        _save_to_json(data = data_metrla, path = metrla_path)
        logging.info(f"METR-LA data saved to {metrla_path}")
    else:
        logging.info(f"METR-LA data already exists at {metrla_path}. Skipping data source.")

    ## --- pems-bay --- ##
    pemsbay_path = os.path.join(PATH_PROC, f"{NAME_PEMSBAY}.json")
    if not os.path.exists(pemsbay_path):
        logging.info("Processing PEMS-BAY data...")
        data_pemsbay = PemsBayProcessor(raw_data_dir = os.path.join(PATH_ROOT, NAME_PEMSBAY)).run()
        _save_to_json(data = data_pemsbay, path = pemsbay_path)
        logging.info(f"PEMS-BAY data saved to {pemsbay_path}")
    else:
        logging.info(f"PEMS-BAY data already exists at {pemsbay_path}. Skipping data source.")

    ## --- montevideo --- ##
    path_montevideo = os.path.join(PATH_PROC, f"{NAME_MONTEVIDEO}.json")
    if not os.path.exists(path_montevideo):
        logging.info("Processing Montevideo data...")
        data_montevideo = MontevideoProcessor().run()
        _save_to_json(data = data_montevideo, path = path_montevideo)
        logging.info(f"Montevideo data saved to {path_montevideo}")
    else:
        logging.info(f"Montevideo data already exists at {path_montevideo}. Skipping data source.")

    ## --- crop pollinator --- ##
    path_crop = os.path.join(PATH_PROC, f"{NAME_CROP}.json")
    if not os.path.exists(path_crop):
        logging.info("Processing CropPol data...")
        data_crop = CropProcessor(url_sampling = URL_CROP_SAMPLING, url_field = URL_CROP_FIELD).run()
        _save_to_json(data = data_crop, path = path_crop)
        logging.info(f"CropPol data saved to {path_crop}")
    else:
        logging.info(f"CropPol data already exists at {path_crop}. Skipping data source.")

    ## --- faers --- ##
    path_faers = os.path.join(PATH_PROC, f"{NAME_FAERS}.json")
    if not os.path.exists(path_faers):
        logging.info("Processing FAERS data...")
        data_faers = FaersProcessor(id = "IMATINIB", url = URL_FAERS).run()
        _save_to_json(data = data_faers, path = path_faers)
        logging.info(f"FAERS data saved to {path_faers}")
    else:
        logging.info(f"FAERS data already exists at {path_faers}. Skipping data source.")

    ## --- c. elegans --- ##
    path_celegans = os.path.join(PATH_PROC, f"{NAME_CELEGANS}.json")
    if not os.path.exists(path_celegans):
        logging.info("Processing C. Elegans data...")
        data_celegans = CelegansProcessor().run()
        _save_to_json(data = data_celegans, path = path_celegans)
        logging.info(f"C. Elegans data saved to {path_celegans}")
    else:
        logging.info(f"C. Elegans data already exists at {path_celegans}. Skipping data source.")

    ## --- epilepsy --- ##
    path_epilepsy = os.path.join(PATH_PROC, f"{NAME_EPILEPSY}.json")
    if not os.path.exists(path_epilepsy):
        logging.info("Processing Epilepsy data...")
        ids = [f'chb{i:02d}' for i in range(1, 25)]
        data_epilepsy = EpilepsyProcessor(url = URL_EPILEPSY, ids = ids).run()
        _save_to_json(data = data_epilepsy, path = path_epilepsy)
        logging.info(f"Epilepsy data saved to {path_epilepsy}")
    else:
        logging.info(f"Epilepsy data already exists at {path_epilepsy}. Skipping data source.")

    ## --- chickenpox --- ##
    path_chickenpox = os.path.join(PATH_PROC, f"{NAME_CHICKENPOX}.json")
    if not os.path.exists(path_chickenpox):
        logging.info("Processing Chickenpox data...")
        data_chickenpox = ChickenpoxProcessor(
            url = URL_CHICKENPOX_EVENTS, 
            name = "hungary_chickenpox.csv"
        ).run()
        _save_to_json(data = data_chickenpox, path = path_chickenpox)
        logging.info(f"Chickenpox data saved to {path_chickenpox}")
    else:
        logging.info(f"Chickenpox data already exists at {path_chickenpox}. Skipping data source.")

    ## --- gwosc --- ##
    path_gwosc = os.path.join(PATH_PROC, f"{NAME_GWOSC}.json")
    if not os.path.exists(path_gwosc):
        logging.info("Processing GWOSC data...")
        data_gwosc = GwoscProcessor(url = URL_GWOSC).run()
        _save_to_json(data = data_gwosc, path = path_gwosc)
        logging.info(f"GWOSC data saved to {path_gwosc}")
    else:
        logging.info(f"GWOSC data already exists at {path_gwosc}. Skipping data source.")

    ## --- nwis --- ##
    path_nwis = os.path.join(PATH_PROC, f"{NAME_RIVER}.json")
    if not os.path.exists(path_nwis):
        logging.info("Processing NWIS river data...")
        params = {
            "format": "rdb",
            "group_key": "huc_cd",
            "huc_cd": "1501",
            "siteType": "ST",
            "agency_cd": "USGS",
            "siteStatus": "active",
        }
        data_nwis = NwisProcessor(
            url_inventory = URL_RIVER_INVENTORY,
            url_site = URL_RIVER_SITE,
            url_iv = URL_RIVER_IV,
            params = params,
            start_date = "2014-01-01",
            end_date = "2024-12-31"
        ).run()
        _save_to_json(data=data_nwis, path=path_nwis)
        logging.info(f"NWIS river data saved to {path_nwis}")
    else:
        logging.info(f"NWIS river data already exists at {path_nwis}. Skipping data source.")

    ## --- auger --- ##
    path_auger = os.path.join(PATH_PROC, f"{NAME_AUGER}.json")
    if not os.path.exists(path_auger):
        logging.info("Processing Auger data...")
        data_auger = AugerProcessor(
            url_network = URL_AUGER_NETWORK,
            url_events = URL_AUGER_EVENTS
        ).run()
        _save_to_json(data = data_auger, path=path_auger)
        logging.info(f"Auger data saved to {path_auger}")
    else:
        logging.info(f"Auger data already exists at {path_auger}. Skipping data source.")

    ## --- seismic --- ##
    path_seismic = os.path.join(PATH_PROC, f"{NAME_SEISMIC}.json")
    if not os.path.exists(path_seismic):
        logging.info("Processing Seismic data...")
        
        ## define parameters for the seismic data
        params_network = {"level": "station", "format": "xml", "network": "IU"}
        namespace = {"ns": "http://www.fdsn.org/xml/station/1"}
        row_path = ".//ns:Station"
        col_map = {
            "code": ".@code",
            "lat": ".//ns:Latitude",
            "lon": ".//ns:Longitude"
        }
        params_events = {
            "starttime": "2023-01-01",
            "endtime": "2023-12-31"
        }

        data_seismic = SeismicProcessor(
            url_network=URL_SEISMIC_NETWORK,
            params_network=params_network,
            namespace=namespace,
            row_path=row_path,
            col_map=col_map,
            url_events=URL_SEISMIC_EVENTS,
            params_events=params_events
        ).run()
        _save_to_json(data=data_seismic, path=path_seismic)
        logging.info(f"Seismic data saved to {path_seismic}")
    else:
        logging.info(f"Seismic data already exists at {path_seismic}. Skipping data source.")

    ## --- rain --- ##
    path_rain = os.path.join(PATH_PROC, f"{NAME_RAIN}.json")
    if not os.path.exists(path_rain):
        logging.info("Processing Rain data...")
        data_rain = RainProcessor(
            country = "LA",  ## iso country code for laos
            start_date = "2024-01-01",
            end_date = "2024-03-31"
        ).run()
        _save_to_json(data=data_rain, path=path_rain)
        logging.info(f"Rain data saved to {path_rain}")
    else:
        logging.info(f"Rain data already exists at {path_rain}. Skipping data source.")

## execution
if __name__ == '__main__':
    processor()
