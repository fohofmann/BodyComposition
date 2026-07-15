#!/usr/bin/env python

# import libraries
import argparse
import logging
from pathlib import Path
import os
from time import time
from BodyComposition.utils.config import update_config
import requests
from tqdm import tqdm
import zipfile
import shutil

# set up logging
path_log = Path(f'./logs/pre_downloadmodels_{int(time())}.log')
path_log.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=path_log)

# defining models per collection
definition_pipelines = {
    'BodyComposition': ['CTDeepRot-2D', 'VertebralBodiesCT-ResEncL', 'BodyCompositionCT-ResEncL'],
    'BodyCompositionFast': ['CTDeepRot-2D', 'VertebralBodiesCT-ResEncM', 'BodyCompositionCT-ResEncM'],
    'SarcopeniaTotalSegmentator': ['CTDeepRot-2D', 'TotalSegmentator-spine', 'TotalSegmentator-muscles', 'TotalSegmentator-body', 'TotalSegmentator-vertebrae_body', 'TotalSegmentator-tissue_types'],
    'SarcopeniaTotalSegmentatorFast': ['CTDeepRot-2D', 'TotalSegmentator-spine', 'TotalSegmentator-tissue_types'],
    'SarcopeniaStanfordFast': ['CTDeepRot-2D', 'Stanford-Spine', 'Stanford-Tissue'],
    'BodyAndOrganAnalysis': ['CTDeepRot-2D', 'BodyAndOrganAnalysis', 'TotalSegmentator-spine'],
}

# defining sources
definition_sources = {
    'CTDeepRot-2D': {
        'source': 'ctdeeprot',
    },
    'VertebralBodiesCT-ResEncL': {
        'source': 'huggingface',
        'hf_id': 'fhofmann/VertebralBodiesCT-ResEncL',
        'local_id': 'int-vertebrae'
    },
    'VertebralBodiesCT-ResEncM': {
        'source': 'huggingface',
        'hf_id': 'fhofmann/VertebralBodiesCT-ResEncM',
        'local_id': 'int-vertebrae'
    },
    'BodyCompositionCT-ResEncL': {
         'source': 'huggingface',
         'hf_id': 'fhofmann/BodyCompositionCT-ResEncL',
         'local_id': 'int-bodycomposition'
    },
    'BodyCompositionCT-ResEncM': {
         'source': 'huggingface',
         'hf_id': 'fhofmann/BodyCompositionCT-ResEncM'
    },
    'TotalSegmentator-spine': {
        'source': 'totalsegmentator',
        'ts_id': 292
    },
    'TotalSegmentator-muscles': {
        'source': 'totalsegmentator',
        'ts_id': 294
    },
    'TotalSegmentator-body': {
        'source': 'totalsegmentator',
        'ts_id': 299
    },
    'TotalSegmentator-vertebrae_body': {
        'source': 'totalsegmentator',
        'ts_id': 302
    },
    'TotalSegmentator-tissue_types': {
        'source': 'totalsegmentator',
        'ts_id': 481
    },
    'Stanford-Spine': {
        'source': 'huggingface',
        'hf_id': 'louisblankemeier/stanford_spine',
        'local_id': 'stanford-spine',
    },
    'Stanford-Tissue': {
        'source': 'huggingface',
        'hf_id': 'stanfordmimi/multilevel_muscle_adipose_tissue',
        'local_id': 'stanford-tissue',
    },
    'BodyAndOrganAnalysis': {
        'source': 'github',
        'url': 'https://github.com/UMEssen/Body-and-Organ-Analysis/releases/download/BCA-BodyRegionsWeights-v0.1.0/Task542_BCA_inference.zip',
        'local_id': 'boa'
    }
}

# Helper function to download files with retries and progress bar
def download_file(url, dest_path, retries=5, delay=5):
    for attempt in range(retries):
        try:
            headers = {}
            if dest_path.exists():
                headers['Range'] = f'bytes={dest_path.stat().st_size}-'
            response = requests.get(url, stream=True, headers=headers, timeout=10)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0)) + dest_path.stat().st_size if 'Range' in headers else int(response.headers.get('content-length', 0))
            mode = 'ab' if 'Range' in headers else 'wb'
            with open(dest_path, mode) as file, tqdm(
                desc=dest_path.name,
                total=total_size,
                initial=dest_path.stat().st_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            # Check if the download is complete
            if dest_path.stat().st_size == total_size:
                return
            else:
                logging.warning(f"Download incomplete: {dest_path.stat().st_size} of {total_size} bytes downloaded.")
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt + 1 < retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise
    logging.error(f"Download not completed after {retries} retries. Try to restart.")

# Helper function to unzip files
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Helper function to clean up directory structure
def reduce_directory_dimensions(directory: Path):
    while True:
        items = list(directory.iterdir())
        if len(items) != 1 or not items[0].is_dir():
            break
        subdir = items[0]
        for item in subdir.iterdir():
            shutil.move(str(item), str(directory))
        subdir.rmdir()



def main():
    """
    Download models from specified sources, place them in the directories specified in the config file.
    Usage: bin/pre_download_models.py --pipeline BodyCompositionFast; better using entrypoint
    """

    # argument parser
    parser = argparse.ArgumentParser(description='Download models from specified sources, place them in the directories specified in the config file')
    parser.add_argument('--pipeline', '-p', type=str,
                        help='Collection of models needed for specific pipeline to be downloaded',
                        choices=['BodyComposition', 'BodyCompositionFast',
                                 'SarcopeniaTotalSegmentator', 'SarcopeniaTotalSegmentatorFast',
                                 'SarcopeniaStanfordFast',
                                 'BodyAndOrganAnalysis'])
    parser.add_argument('--model', '-m', type=str,
                        help='Single model to be downloaded',
                        choices=['CTDeepRot-2D', 'VertebralBodiesCT-ResEncM', 'VertebralBodiesCT-ResEncL', 'BodyCompositionCT-ResEncM', 'BodyCompositionCT-ResEncL',
                                 'TotalSegmentator-total', 'TotalSegmentator-spine', 'TotalSegmentator-body', 'TotalSegmentator-vertebrae_body', 'TotalSegmentator-tissue_types',
                                 'Stanford-Spine', 'Stanford-Tissue',
                                 'BodyAndOrganAnalysis'])
    args = parser.parse_args()

    # set up logging
    logging.info('LOAD MODELS')
    
    # load config
    config_dict = update_config({}, Path('./config/config.yaml'))
    config_weights = config_dict['paths']['weights']
    logging.info('config loaded')

    # create cache directory
    cache_dir = Path(config_dict['paths']['cache'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # load model names
    if args.pipeline:
        model_titles = definition_pipelines[args.pipeline]
    elif args.model:
        model_titles = [args.model]
    else:
        logging.error('No pipeline or model specified')
        return

    # set up TotalSegmentator if needed
    if any(definition_sources[model_title]['source'] == 'totalsegmentator' for model_title in model_titles):
    
        # define environment variables
        path_tseg_config = str(config_dict['paths']['totalsegmentator_config'])
        path_tseg_weights = str(config_weights['totalsegmentator'])
        if path_tseg_config not in ('None', ''):
            os.environ["TOTALSEG_HOME_DIR"] = path_tseg_config
        if path_tseg_weights not in ('None', ''):
            os.environ["TOTALSEG_WEIGHTS_PATH"] = path_tseg_weights

        # adapted from https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/bin/totalseg_download_weights.py
        from totalsegmentator.libs import download_pretrained_weights
        from totalsegmentator.config import setup_totalseg, set_config_key

        setup_totalseg()
        set_config_key("statistics_disclaimer_shown", True)

        logging.info(f'TotalSegmentator setup: config={path_tseg_config}, weights={path_tseg_weights}')

    # set up HuggingFace if needed
    if any(definition_sources[model_title]['source'] == 'huggingface' for model_title in model_titles):
        os.environ["HF_HOME"] = str(cache_dir)
        from huggingface_hub import snapshot_download 
        logging.info(f"HuggingFace setup: cache={str(cache_dir)}")

    # loop through models
    for model_title in model_titles:
        model = definition_sources[model_title]

        if model['source'] == 'ctdeeprot':
            from BodyComposition.orientation.ctdeeprot import ensure_checkpoint

            orientation_model = config_dict['orientation']['model']
            checkpoint = ensure_checkpoint(
                orientation_model['checkpoint_path'],
                auto_download=True,
                download_url=orientation_model['download_url'],
                expected_sha256=orientation_model['sha256'],
            )
            logging.info(f"Downloaded and verified `CTDeepRot-2D` at `{checkpoint}`.")
        elif model['source'] == 'huggingface':
            download_dir = config_weights[model['local_id']]
            logging.info(f"Downloading `{model_title}` from `huggingface.co/{model['hf_id']}` to `{download_dir}`...")
            snapshot_download(repo_id=model['hf_id'], local_dir=download_dir, ignore_patterns=["**/model_best*", "**/logs*" "**/*.zip", "*.zip", "**/*.md", "*.md",".txt", ".png"])
            logging.info(f"  finished.")
        elif model['source'] == 'totalsegmentator':
            logging.info(f"Downloading `TotalSegmentator/{model_title}`, task_id {model['ts_id']} to `{config_weights['totalsegmentator']}`...")
            download_pretrained_weights(model['ts_id'])
            logging.info(f"  finished.")
        elif model['source'] == 'github':
            download_dir = config_weights[model['local_id']]
            logging.info(f"Downloading `{model_title}` from `{model['url']}` to `{download_dir}`...")
            zip_path = cache_dir / 'download_tmp.zip'
            try:
                download_file(model['url'], zip_path)
                if os.path.exists(download_dir):
                    logging.info(f"  directory `{download_dir}` already exists, skipping. If something went wrong, delete the directory and try again.")
                    continue
                unzip_file(zip_path, download_dir)
                reduce_directory_dimensions(Path(download_dir))
                os.remove(zip_path)
                logging.info(f"  finished.")
            except Exception as e:
                logging.error(f"Failed to download or extract `{model_title}`: {e}")
            logging.info(f"  finished.")

        else:
            logging.error(f"Unknown source {model['source']}")
            return

    logging.info('DOWNLOADED ALL MODELS')

if __name__ == "__main__":
    main() # parser is in main to be available when using pyproject.toml entrypoint
