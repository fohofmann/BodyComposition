#!/usr/bin/env python

# import libraries
import argparse
import logging
from pathlib import Path
import os
from time import time
from BodyComposition.utils.config import update_config

# defining models per collection
definition_pipelines = {
    'BodyComposition': ['VertebralBodiesCT-ResEncL', 'TotalSegmentator-body', 'TotalSegmentator-tissue_types'],
    'BodyCompositionFast': ['VertebralBodiesCT-ResEncM', 'TotalSegmentator-tissue_types' ],
    'SarcopeniaTotalSegmentator': ['TotalSegmentator-spine', 'TotalSegmentator-muscles', 'TotalSegmentator-body', 'TotalSegmentator-vertebrae_body', 'TotalSegmentator-tissue_types'],
    'SarcopeniaTotalSegmentatorFast': ['TotalSegmentator-spine', 'TotalSegmentator-tissue_types'],
    'SarcopeniaStanford': ['Stanford-Spine', 'TotalSegmentator-body', 'TotalSegmentator-vertebrae_body', 'TotalSegmentator-tissue_types'],
}

# defining sources
definition_sources = {
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
    # 'BodyCompositionCT-ResEncL': {
    #     'source': 'huggingface',
    #     'hf_id': 'fhofmann/BodyCompositionCT-ResEncL'
    # }, # soon at your service
    # 'BodyCompositionCT-ResEncM': {
    #     'source': 'huggingface',
    #     'hf_id': 'fhofmann/BodyCompositionCT-ResEncM'
    # }, # soon at your service
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
    }
}


def main():
    """
    Download models from specified sources, place them in the directories specified in the config file.
    Usage: bin/pre_download_models.py --pipeline BodyCompositionFast; better using entrypoint
    """

    # argument parser
    parser = argparse.ArgumentParser(description='Download models from specified sources, place them in the directories specified in the config file')
    parser.add_argument('--pipeline', '-p', type=str,
                        help='Collection of models needed for specific pipeline to be downloaded',
                        choices=['BodyComposition', 'BodyCompositionFast', 'SarcopeniaTotalSegmentator', 'SarcopeniaTotalSegmentatorFast', 'SarcopeniaStanford'])
    parser.add_argument('--model', '-m', type=str,
                        help='Single model to be downloaded',
                        choices=['VertebralBodiesCT-ResEncM', 'VertebralBodiesCT-ResEncL', 'BodyCompositionCT-ResEncM', 'BodyCompositionCT-ResEncL',
                                 'TotalSegmentator-total','TotalSegmentator-body', 'TotalSegmentator-vertebrae_body', 'TotalSegmentator-tissue_types',
                                 'Stanford-Spine'])
    args = parser.parse_args()

    # set up logging
    path_log = Path(f'./logs/pre_downloadmodels_{int(time())}.log')
    path_log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=path_log)
    logging.info('LOAD MODELS')
    
    # load config
    config_dict = update_config({}, Path('./config/config.yaml'))
    config_weights = config_dict['paths']['weights']
    logging.info('config loaded')

    # load model names
    if args.pipeline:
        model_titles = definition_pipelines[args.pipeline]
    elif args.model:
        model_titles = [model]
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
        os.environ["HF_HOME"] = str(config_dict['paths']['huggingface_cache'])
        from huggingface_hub import snapshot_download 
        logging.info(f"HuggingFace setup: cache={str(config_dict['paths']['huggingface_cache'])}")

    # loop through models
    for model_title in model_titles:
        model = definition_sources[model_title]

        if model['source'] == 'huggingface':
            download_dir = config_weights[model['local_id']]
            logging.info(f"Downloading `{model_title}` from `huggingface.co/{model['hf_id']}` to `{download_dir}`...")
            snapshot_download(repo_id=model['hf_id'], local_dir=download_dir, ignore_patterns=["**/model_best*", "**/logs*" "**/*.zip", "*.zip", "**/*.md", "*.md",".txt", ".png"])
            logging.info(f"  finished.")
        elif model['source'] == 'totalsegmentator':
            logging.info(f"Downloading `TotalSegmentator/{model_title}`, task_id {model['ts_id']} to `{config_weights['totalsegmentator']}`...")
            download_pretrained_weights(model['ts_id'])
            logging.info(f"  finished.")
        else:
            logging.error(f"Unknown source {model['source']}")
            return
        
    logging.info('DOWNLOADED ALL MODELS')

if __name__ == "__main__":
    main() # parser is in main to be available when using pyproject.toml entrypoint
    