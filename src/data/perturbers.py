## libraries
import os
import sys
import logging
import configparser
import numpy as np
import pandas as pd

## directory
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root not in sys.path:
    sys.path.append(root)

## modules
from src.data.helpers import _save_to_json, _extract_counts
from src.evaluators.perturbing import network_perturb, analytical_perturb, process_perturb, invariant_perturb
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures
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
PATH_PERT = config['paths']['PATH_PERT'].strip('"')

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
URL_RIVER_INVENTORY = config['urls']['URL_RIVER_INVENTORY'].strip('"')
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

## ----------------------------
## network perturbation (G space)
## ----------------------------
NETWORK_METHODS = {
    'rewire':      (0.01, 0.05, 0.15, 0.30, 0.50),
    'sparsify':    (0.01, 0.05, 0.15, 0.30, 0.50),
    'node_sample': (0.01, 0.05, 0.15, 0.30, 0.50),
}

## --------------------------------
## invariant perturbation (x encoding)
## --------------------------------
INVARIANT_METHODS = {
    'noise':  (0.01, 0.05, 0.10, 0.15, 0.25),   ## additive gaussian
    'jitter': (0.01, 0.05, 0.10, 0.15, 0.25),   ## multiplicative
    'subset': (0.90, 0.80, 0.70, 0.60, 0.50),   ## retain fraction
}

## ------------------------------
## process perturbation (S space)
## ------------------------------
PROCESS_METHODS = {
    'scaling': (0.75, 0.90, 1.10, 1.25, 1.50),        ## symmetric deformation
    'smoothing': (2, 3, 5, 7, 10),                    ## bounded smoothing window
    'bootstrapping': (0.10, 0.20, 0.30, 0.40, 0.50),  ## partial temporal destruction
}

## --------------------------------------
## signature-level perturbation (z encoding)
## --------------------------------------
SIGNATURE_METHODS = {
    'noise':  (0.005, 0.01, 0.025, 0.05, 0.10),
    'jitter': (0.005, 0.01, 0.025, 0.05, 0.10),
    'subset': (0.95, 0.90, 0.85, 0.80, 0.70),
}

## ----------------------
## temporal resolution
## ----------------------
TEMPORAL_SCALES = ('2D', '3D', '7D', '14D', '30D')



## helper functions
def _is_fully_connected_bipartite(graph) -> bool:
    """Check if a graph is a fully connected bipartite graph."""
    if graph.vcount() == 0 or graph.ecount() == 0:
        return False
    is_bip, types = graph.is_bipartite(return_types=True)
    if not is_bip or types is None:
        return False
    n1 = sum(types)
    n2 = len(types) - n1
    return graph.ecount() == n1 * n2

def _run_all_perturbations(proc, name):
    """Run network, process, and temporal perturbations for a given processor."""
    results = {}

    ## --- network perturbation --- ##
    graph = getattr(proc, 'graph', None)
    if graph is not None:
        
        ## ensure simple undirected graph (remove multi-edges and self-loops)
        graph.simplify()

        ## check for fully connected bipartite structure to determine if analytical perturbation can be used
        network_results = []
        analytical = _is_fully_connected_bipartite(graph)
        if analytical:
            degrees = np.array(graph.degree(), dtype=float)
            n_nodes = graph.vcount()
            n_edges = graph.ecount()
            logging.info(f"  Using analytical perturbation for {name} ({n_nodes:,} nodes, {n_edges:,} edges)")
        invariants = GraphInvariants(graph).all(analytical = analytical)
        for method, intensities in NETWORK_METHODS.items():
            for intensity in intensities:
                if analytical:
                    try:
                        features = analytical_perturb(
                            invariants = invariants,
                            degrees = degrees,
                            n_nodes = n_nodes,
                            n_edges = n_edges,
                            method = {"rewire": "degree_preserving_rewire", "sparsify": "bernoulli_edge_thinning", "node_sample": "uniform_node_sampling"}[method],
                            intensity = float(intensity),
                        )
                    except Exception as exc:
                        logging.warning(f"Analytical {method} @ {intensity:.2f} failed for {name}: {exc}")
                        continue
                else:
                    try:
                        features = network_perturb(graph, method = method, intensity = float(intensity))
                    except Exception as exc:
                        logging.warning(f"Network {method} @ {intensity:.2f} failed for {name}: {exc}")
                        continue
                network_results.append({
                    'method': method,
                    'intensity': float(intensity),
                    'invariants': features
                })
        results['network_perturbed'] = network_results
        logging.info(f"  Network perturbation: {len(network_results)} records")
    else:
        logging.warning(f"  No graph object for {name}, skipping network perturbation.")

    ## --- invariant perturbation --- ##
    if graph is not None:
        base_inv = GraphInvariants(graph).all(analytical = analytical)
        base_df = pd.DataFrame([base_inv])
        invariant_pert_results = []
        for method, params in INVARIANT_METHODS.items():
            for param in params:
                try:
                    perturbed_df = invariant_perturb(
                        base_df.copy(),
                        method = method,
                        noise = float(param) if method != 'subset' else 0.05,
                        subset = float(param) if method == 'subset' else 0.8,
                    )
                    row = perturbed_df.iloc[0].to_dict()
                except Exception as exc:
                    logging.warning(f"Invariant {method} @ {param:.3f} failed for {name}: {exc}")
                    continue
                invariant_pert_results.append({
                    'method': method,
                    'intensity': float(param),
                    'invariants': row
                })
        results['invariants_perturbed'] = invariant_pert_results
        logging.info(f"  Invariant perturbation: {len(invariant_pert_results)} records")

    ## --- process perturbation --- ##
    events = getattr(proc, 'events', None)
    counts = _extract_counts(events)

    if counts is not None and len(counts) > 0:
        process_results = []
        for method, params in PROCESS_METHODS.items():
            for param in params:
                try:
                    sigs = process_perturb(counts, method = method, param = float(param))
                except Exception as exc:
                    logging.warning(f"Process {method} @ {param} failed for {name}: {exc}")
                    continue
                process_results.append({
                    'method': method,
                    'intensity': float(param),
                    'signatures': sigs
                })
        results['process_perturbed'] = process_results
        logging.info(f"  Process perturbation: {len(process_results)} records")
    else:
        logging.warning(f"  No count series for {name}, skipping process perturbation.")

    ## --- signature perturbation (z -> z') --- ##
    if counts is not None and len(counts) > 0:
        data_temp = pd.DataFrame({"counts": counts, "idx": range(len(counts))})
        base_sigs = ProcessSignatures(data_temp, sort_by = ["idx"], target = "counts").all()
        base_sig_df = pd.DataFrame([base_sigs])
        sig_pert_results = []
        for method, params in SIGNATURE_METHODS.items():
            for param in params:
                try:
                    perturbed_df = invariant_perturb(
                        base_sig_df.copy(),
                        method = method,
                        noise = float(param) if method != 'subset' else 0.05,
                        subset = float(param) if method == 'subset' else 0.8,
                    )
                    row = perturbed_df.iloc[0].to_dict()
                except Exception as exc:
                    logging.warning(f"Signature {method} @ {param:.3f} failed for {name}: {exc}")
                    continue
                sig_pert_results.append({
                    'method': method,
                    'intensity': float(param),
                    'signatures': row
                })
        results['signatures_perturbed'] = sig_pert_results
        logging.info(f"  Signature perturbation: {len(sig_pert_results)} records")

    ## --- temporal aggregation --- ##
    if events is not None and isinstance(events, pd.DataFrame) and not events.empty:
        date_col = next((c for c in ('date', 'datetime', 'timestamp') if c in events.columns), None)
        target_col = next((c for c in ('target', 'count') if c in events.columns), None)

        if date_col is not None and target_col is not None:
            temporal_results = []
            df_temp = events[[date_col, target_col]].copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            df_temp = df_temp.set_index(date_col).sort_index()

            for scale in TEMPORAL_SCALES:
                resampled = df_temp[target_col].resample(scale).sum()
                records = [
                    {'date': str(dt.date()), 'target': int(val)}
                    for dt, val in resampled.items()
                ]
                temporal_results.append({
                    'scale': scale,
                    'events': records
                })

            results['temporal_aggregated'] = temporal_results
            logging.info(f"  Temporal aggregation: {len(temporal_results)} scales")
        else:
            logging.warning(f"  No date/target columns for {name}, skipping temporal aggregation.")
    else:
        logging.warning(f"  No events for {name}, skipping temporal aggregation.")

    return results


## execute
def perturber():

    ## ensure perturbation directory exists
    os.makedirs(name = PATH_PERT, exist_ok = True)

    ## --- federal contracts --- ##
    federal_path = os.path.join(PATH_PERT, f"{NAME_FEDERAL}.json")
    if not os.path.exists(federal_path):
        logging.info("Perturbing Federal data...")
        proc = FederalProcessor(
            url = URL_FEDERAL,
            start_date = "2014-01-01",
            end_date = "2024-12-31"
        )
        proc.run()
        results = _run_all_perturbations(proc, NAME_FEDERAL)
        _save_to_json(data = results, path = federal_path)
        logging.info(f"Federal perturbations saved to {federal_path}")
    else:
        logging.info(f"Federal perturbations already exist at {federal_path}. Skipping.")

    ## --- mooc students --- ##
    mooc_path = os.path.join(PATH_PERT, f"{NAME_MOOC}.json")
    if not os.path.exists(mooc_path):
        logging.info("Perturbing MOOC data...")
        proc = MoocProcessor(url = URL_MOOC)
        proc.run()
        results = _run_all_perturbations(proc, NAME_MOOC)
        _save_to_json(data = results, path = mooc_path)
        logging.info(f"MOOC perturbations saved to {mooc_path}")
    else:
        logging.info(f"MOOC perturbations already exist at {mooc_path}. Skipping.")

    ## --- bitcoin trust --- ##
    bitcoin_path = os.path.join(PATH_PERT, f"{NAME_BITCOIN}.json")
    if not os.path.exists(bitcoin_path):
        logging.info("Perturbing Bitcoin data...")
        proc = BitcoinProcessor(root_path = PATH_ROOT, name = NAME_BITCOIN)
        proc.run()
        results = _run_all_perturbations(proc, NAME_BITCOIN)
        _save_to_json(data = results, path = bitcoin_path)
        logging.info(f"Bitcoin perturbations saved to {bitcoin_path}")
    else:
        logging.info(f"Bitcoin perturbations already exist at {bitcoin_path}. Skipping.")

    ## --- amazon reviews --- ##
    # amazon_path = os.path.join(PATH_PERT, f"{NAME_AMAZON}.json")
    # if not os.path.exists(amazon_path):
    #     logging.info("Perturbing Amazon data...")
    #     proc = AmazonProcessor(root_path = PATH_ROOT, url = URL_AMAZON, name = NAME_AMAZON)
    #     proc.run()
    #     results = _run_all_perturbations(proc, NAME_AMAZON)
    #     _save_to_json(data = results, path = amazon_path)
    #     logging.info(f"Amazon perturbations saved to {amazon_path}")
    # else:
    #     logging.info(f"Amazon perturbations already exist at {amazon_path}. Skipping.")

    ## --- world bank --- ##
    world_path = os.path.join(PATH_PERT, f"{NAME_WORLD}.json")
    if not os.path.exists(world_path):
        logging.info("Perturbing World Bank data...")
        proc = WorldBankProcessor(
            url_projects = URL_WORLD_NETWORK,
            url_meta = URL_WORLD_METADATA,
            start_year = "2014",
            end_year = "2024"
        )
        proc.run()
        results = _run_all_perturbations(proc, NAME_WORLD)
        _save_to_json(data = results, path = world_path)
        logging.info(f"World Bank perturbations saved to {world_path}")
    else:
        logging.info(f"World Bank perturbations already exist at {world_path}. Skipping.")

    ## --- math wiki --- ##
    wiki_path = os.path.join(PATH_PERT, f"{NAME_WIKI}.json")
    if not os.path.exists(wiki_path):
        logging.info("Perturbing Wiki data...")
        proc = WikiProcessor(url = URL_WIKI, name = "wikivital_mathematics.json")
        proc.run()
        results = _run_all_perturbations(proc, NAME_WIKI)
        _save_to_json(data = results, path = wiki_path)
        logging.info(f"Wiki perturbations saved to {wiki_path}")
    else:
        logging.info(f"Wiki perturbations already exist at {wiki_path}. Skipping.")

    ## --- jodie wiki --- ##
    jodie_path = os.path.join(PATH_PERT, f"{NAME_JODIE}.json")
    if not os.path.exists(jodie_path):
        logging.info("Perturbing JODIE data...")
        proc = JodieProcessor(root_path = PATH_ROOT, name = "wikipedia")
        proc.run()
        results = _run_all_perturbations(proc, NAME_JODIE)
        _save_to_json(data = results, path = jodie_path)
        logging.info(f"JODIE perturbations saved to {jodie_path}")
    else:
        logging.info(f"JODIE perturbations already exist at {jodie_path}. Skipping.")

    ## --- mathoverflow --- ##
    overflow_path = os.path.join(PATH_PERT, f"{NAME_OVERFLOW}.json")
    if not os.path.exists(overflow_path):
        logging.info("Perturbing MathOverflow data...")
        proc = OverflowProcessor(url = URL_OVERFLOW)
        proc.run()
        results = _run_all_perturbations(proc, NAME_OVERFLOW)
        _save_to_json(data = results, path = overflow_path)
        logging.info(f"MathOverflow perturbations saved to {overflow_path}")
    else:
        logging.info(f"MathOverflow perturbations already exist at {overflow_path}. Skipping.")

    ## --- eu-core email --- ##
    email_path = os.path.join(PATH_PERT, f"{NAME_EMAIL}.json")
    if not os.path.exists(email_path):
        logging.info("Perturbing EU-Core Email data...")
        proc = EmailProcessor(url = URL_EMAIL)
        proc.run()
        results = _run_all_perturbations(proc, NAME_EMAIL)
        _save_to_json(data = results, path = email_path)
        logging.info(f"EU-Core Email perturbations saved to {email_path}")
    else:
        logging.info(f"EU-Core Email perturbations already exist at {email_path}. Skipping.")

    ## --- college --- ##
    college_path = os.path.join(PATH_PERT, f"{NAME_COLLEGE}.json")
    if not os.path.exists(college_path):
        logging.info("Perturbing UC Irvine College Message data...")
        proc = CollegeProcessor(url = URL_COLLEGE)
        proc.run()
        results = _run_all_perturbations(proc, NAME_COLLEGE)
        _save_to_json(data = results, path = college_path)
        logging.info(f"UC Irvine College Message perturbations saved to {college_path}")
    else:
        logging.info(f"UC Irvine College Message perturbations already exist at {college_path}. Skipping.")

    ## --- idling --- ##
    idling_path = os.path.join(PATH_PERT, f"{NAME_IDLING}.json")
    if not os.path.exists(idling_path):
        logging.info("Perturbing Halifax idling data...")
        proc = IdlingProcessor(path_events = PATH_ROOT + "idling/")
        proc.run()
        results = _run_all_perturbations(proc, NAME_IDLING)
        _save_to_json(data = results, path = idling_path)
        logging.info(f"Halifax idling perturbations saved to {idling_path}")
    else:
        logging.info(f"Halifax idling perturbations already exist at {idling_path}. Skipping.")

    ## --- windmill --- ##
    windmill_path = os.path.join(PATH_PERT, f"{NAME_WINDMILL}.json")
    if not os.path.exists(windmill_path):
        logging.info("Perturbing Windmill data...")
        proc = WindmillProcessor(raw_data_dir = os.path.join(PATH_ROOT, NAME_WINDMILL))
        proc.run()
        results = _run_all_perturbations(proc, NAME_WINDMILL)
        _save_to_json(data = results, path = windmill_path)
        logging.info(f"Windmill perturbations saved to {windmill_path}")
    else:
        logging.info(f"Windmill perturbations already exist at {windmill_path}. Skipping.")

    ## --- metr-la --- ##
    metrla_path = os.path.join(PATH_PERT, f"{NAME_METRLA}.json")
    if not os.path.exists(metrla_path):
        logging.info("Perturbing METR-LA data...")
        proc = MetrLaProcessor(raw_data_dir = os.path.join(PATH_ROOT, NAME_METRLA))
        proc.run()
        results = _run_all_perturbations(proc, NAME_METRLA)
        _save_to_json(data = results, path = metrla_path)
        logging.info(f"METR-LA perturbations saved to {metrla_path}")
    else:
        logging.info(f"METR-LA perturbations already exist at {metrla_path}. Skipping.")

    ## --- pems-bay --- ##
    pemsbay_path = os.path.join(PATH_PERT, f"{NAME_PEMSBAY}.json")
    if not os.path.exists(pemsbay_path):
        logging.info("Perturbing PEMS-BAY data...")
        proc = PemsBayProcessor(raw_data_dir = os.path.join(PATH_ROOT, NAME_PEMSBAY))
        proc.run()
        results = _run_all_perturbations(proc, NAME_PEMSBAY)
        _save_to_json(data = results, path = pemsbay_path)
        logging.info(f"PEMS-BAY perturbations saved to {pemsbay_path}")
    else:
        logging.info(f"PEMS-BAY perturbations already exist at {pemsbay_path}. Skipping.")

    ## --- montevideo --- ##
    montevideo_path = os.path.join(PATH_PERT, f"{NAME_MONTEVIDEO}.json")
    if not os.path.exists(montevideo_path):
        logging.info("Perturbing Montevideo data...")
        proc = MontevideoProcessor()
        proc.run()
        results = _run_all_perturbations(proc, NAME_MONTEVIDEO)
        _save_to_json(data = results, path = montevideo_path)
        logging.info(f"Montevideo perturbations saved to {montevideo_path}")
    else:
        logging.info(f"Montevideo perturbations already exist at {montevideo_path}. Skipping.")

    ## --- crop pollinator --- ##
    crop_path = os.path.join(PATH_PERT, f"{NAME_CROP}.json")
    if not os.path.exists(crop_path):
        logging.info("Perturbing CropPol data...")
        proc = CropProcessor(url_sampling = URL_CROP_SAMPLING, url_field = URL_CROP_FIELD)
        proc.run()
        results = _run_all_perturbations(proc, NAME_CROP)
        _save_to_json(data = results, path = crop_path)
        logging.info(f"CropPol perturbations saved to {crop_path}")
    else:
        logging.info(f"CropPol perturbations already exist at {crop_path}. Skipping.")

    ## --- faers --- ##
    faers_path = os.path.join(PATH_PERT, f"{NAME_FAERS}.json")
    if not os.path.exists(faers_path):
        logging.info("Perturbing FAERS data...")
        proc = FaersProcessor(id = "IMATINIB", url = URL_FAERS)
        proc.run()
        results = _run_all_perturbations(proc, NAME_FAERS)
        _save_to_json(data = results, path = faers_path)
        logging.info(f"FAERS perturbations saved to {faers_path}")
    else:
        logging.info(f"FAERS perturbations already exist at {faers_path}. Skipping.")

    ## --- c. elegans --- ##
    celegans_path = os.path.join(PATH_PERT, f"{NAME_CELEGANS}.json")
    if not os.path.exists(celegans_path):
        logging.info("Perturbing C. Elegans data...")
        proc = CelegansProcessor()
        proc.run()
        results = _run_all_perturbations(proc, NAME_CELEGANS)
        _save_to_json(data = results, path = celegans_path)
        logging.info(f"C. Elegans perturbations saved to {celegans_path}")
    else:
        logging.info(f"C. Elegans perturbations already exist at {celegans_path}. Skipping.")

    ## --- epilepsy --- ##
    epilepsy_path = os.path.join(PATH_PERT, f"{NAME_EPILEPSY}.json")
    if not os.path.exists(epilepsy_path):
        logging.info("Perturbing Epilepsy data...")
        ids = [f'chb{i:02d}' for i in range(1, 25)]
        proc = EpilepsyProcessor(url = URL_EPILEPSY, ids = ids)
        proc.run()
        results = _run_all_perturbations(proc, NAME_EPILEPSY)
        _save_to_json(data = results, path = epilepsy_path)
        logging.info(f"Epilepsy perturbations saved to {epilepsy_path}")
    else:
        logging.info(f"Epilepsy perturbations already exist at {epilepsy_path}. Skipping.")

    ## --- chickenpox --- ##
    chickenpox_path = os.path.join(PATH_PERT, f"{NAME_CHICKENPOX}.json")
    if not os.path.exists(chickenpox_path):
        logging.info("Perturbing Chickenpox data...")
        proc = ChickenpoxProcessor(
            url = URL_CHICKENPOX_EVENTS,
            name = "hungary_chickenpox.csv"
        )
        proc.run()
        results = _run_all_perturbations(proc, NAME_CHICKENPOX)
        _save_to_json(data = results, path = chickenpox_path)
        logging.info(f"Chickenpox perturbations saved to {chickenpox_path}")
    else:
        logging.info(f"Chickenpox perturbations already exist at {chickenpox_path}. Skipping.")

    ## --- gwosc --- ##
    gwosc_path = os.path.join(PATH_PERT, f"{NAME_GWOSC}.json")
    if not os.path.exists(gwosc_path):
        logging.info("Perturbing GWOSC data...")
        proc = GwoscProcessor(url = URL_GWOSC)
        proc.run()
        results = _run_all_perturbations(proc, NAME_GWOSC)
        _save_to_json(data = results, path = gwosc_path)
        logging.info(f"GWOSC perturbations saved to {gwosc_path}")
    else:
        logging.info(f"GWOSC perturbations already exist at {gwosc_path}. Skipping.")

    ## --- nwis --- ##
    river_path = os.path.join(PATH_PERT, f"{NAME_RIVER}.json")
    if not os.path.exists(river_path):
        logging.info("Perturbing NWIS river data...")
        params = {
            "format": "rdb",
            "group_key": "huc_cd",
            "huc_cd": "1501",
            "siteType": "ST",
            "agency_cd": "USGS",
            "siteStatus": "active",
        }
        proc = NwisProcessor(
            url_inventory = URL_RIVER_INVENTORY,
            url_site = URL_RIVER_SITE,
            url_iv = URL_RIVER_IV,
            params = params,
            start_date = "2014-01-01",
            end_date = "2024-12-31"
        )
        proc.run()
        results = _run_all_perturbations(proc, NAME_RIVER)
        _save_to_json(data = results, path = river_path)
        logging.info(f"NWIS river perturbations saved to {river_path}")
    else:
        logging.info(f"NWIS river perturbations already exist at {river_path}. Skipping.")

    ## --- auger --- ##
    auger_path = os.path.join(PATH_PERT, f"{NAME_AUGER}.json")
    if not os.path.exists(auger_path):
        logging.info("Perturbing Auger data...")
        proc = AugerProcessor(
            url_network = URL_AUGER_NETWORK,
            url_events = URL_AUGER_EVENTS
        )
        proc.run()
        results = _run_all_perturbations(proc, NAME_AUGER)
        _save_to_json(data = results, path = auger_path)
        logging.info(f"Auger perturbations saved to {auger_path}")
    else:
        logging.info(f"Auger perturbations already exist at {auger_path}. Skipping.")

    ## --- seismic --- ##
    seismic_path = os.path.join(PATH_PERT, f"{NAME_SEISMIC}.json")
    if not os.path.exists(seismic_path):
        logging.info("Perturbing Seismic data...")

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

        proc = SeismicProcessor(
            url_network = URL_SEISMIC_NETWORK,
            params_network = params_network,
            namespace = namespace,
            row_path = row_path,
            col_map = col_map,
            url_events = URL_SEISMIC_EVENTS,
            params_events = params_events
        )
        proc.run()
        results = _run_all_perturbations(proc, NAME_SEISMIC)
        _save_to_json(data = results, path = seismic_path)
        logging.info(f"Seismic perturbations saved to {seismic_path}")
    else:
        logging.info(f"Seismic perturbations already exist at {seismic_path}. Skipping.")

    ## --- rain --- ##
    rain_path = os.path.join(PATH_PERT, f"{NAME_RAIN}.json")
    if not os.path.exists(rain_path):
        logging.info("Perturbing Rain data...")
        proc = RainProcessor(
            country = "LA",  ## iso country code for laos
            start_date = "2024-01-01",
            end_date = "2024-03-31"
        )
        proc.run()
        results = _run_all_perturbations(proc, NAME_RAIN)
        _save_to_json(data = results, path = rain_path)
        logging.info(f"Rain perturbations saved to {rain_path}")
    else:
        logging.info(f"Rain perturbations already exist at {rain_path}. Skipping.")

if __name__ == '__main__':
    perturber()
