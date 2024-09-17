#!/usr/bin/env python
import argparse
import ast
import os
from pathlib import Path
from time import time
import logging
import re
import multiprocessing

# specific imports
from BodyComposition.utils.config import update_config
from BodyComposition.utils.logging import init_logging
from BodyComposition.utils.datalist import DatalistBuilder
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

# helper functions
def looks_like_dict(s):
    try:
        ast.literal_eval(s)
        return True
    except (SyntaxError, ValueError):
        return False
def looks_like_path(s):
    return os.path.isfile(s)

# main
def main():
    """
    Use this to perform segmentation only using, and save files in the directories corresponding to the pipeline's configuration structure.
    """

    # generate timestamp
    timer_pipeline = time()
    timestamp = int(timer_pipeline)

    # parse arguments
    parser = argparse.ArgumentParser(description='Run batch through pipeline.')
    parser.add_argument('--input', '-i', type=str, default='./data/images',
                        help='Path to input, either directory (e.g., `data/images`) or datalist file (*.json) or single NiFTI file.')
    parser.add_argument('--filter', '-f', type=str, default=r'.*\.nii\.gz$',
                        help='Filter to subset input files.')   
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to configuration file (*.yaml), or dictionary. Can be used to update the default configuration.')
    parser.add_argument('--method', '-m', type=str, default='BodyCompositionFast',
                        help='Name of pipeline method to be run.')
    parser.add_argument('--model', '-n', type=str, required=True,
                        help='Name of model to be used.')
    parser.add_argument('--part_id', type=int, default=0)
    parser.add_argument('--num_parts', type=int, default=1)
    args = parser.parse_args()

    # transform config
    if args.config is None:
        config = None
    elif looks_like_dict(args.config):
        config = ast.literal_eval(args.config)
    elif looks_like_path(args.config):
        config = Path(args.config)
    else:
        config = None

    # simplify input_filter: remove all non-alphanumeric characters
    input_filter_simple = re.sub(r'\W+', '', args.filter)
    
    # load config: general < pipeline specific < input
    config_dict = update_config({}, Path('./config/config.yaml'))
    config_dict = update_config(config_dict, Path('./config/labels.yaml'))
    config_dict = update_config(config_dict, Path('./config') / f'{args.method}.yaml')
    if config is not None:
        config_dict = update_config(config_dict, config)

    # load logging, default: level_file=logging.INFO, level_console= logging.WARNING
    path_logging = Path(str(config_dict['paths']['logs']).format(method=args.method,filter=input_filter_simple,timestamp=timestamp))
    init_logging(file=path_logging,
                 level_file=config_dict['logging_level']['file'],
                 level_console=config_dict['logging_level']['console'])
    logging.info(f"loaded config, start logging now...")

    # load workspace path: argument < config < none, allow string formatting
    workspace = config_dict['paths']['workspace']
    if str(workspace) not in ('None', ''):
        workspace = Path(str(workspace).format(method=args.method,filter=input_filter_simple,timestamp=timestamp))
    else:
        workspace = None

    # create datalist
    # iterable tuple (id, file, workspace=output_dir) of inputs according to criteria
    datalist = DatalistBuilder(input_path = args.input,
                               input_filter = args.filter,
                               workspace = workspace,
                               part_id=args.part_id,
                               num_parts=args.num_parts)    

    # check if datalist is empty
    if len(datalist) == 0:
        logging.warning("No files found. Exiting.")
        return

    # print
    print(f"Model: {args.model}")

    # create input paths
    input_files = [ [str(file[1])] for file in datalist]

    # settings and paths depending on model
    if args.model == 'VertebralBodiesCT-ResEncL':
        output_files = [str(file[2] / f"{file[0]}_int-vertebrae.nii.gz") for file in datalist]
        model_path = config_dict['paths']['weights']['int-vertebrae'] + '/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres'
        model_folds = [0, 1, 2, 3, 4]
        model_mirror = True
    elif args.model == 'VertebralBodiesCT-ResEncM':
        output_files = [str(file[2] / f"{file[0]}_int-vertebrae.nii.gz") for file in datalist]
        model_path = config_dict['paths']['weights']['int-vertebrae'] + '/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres'
        model_folds = 'all'
        model_mirror = True
    elif args.model == 'BodyCompositionCT-ResEncL':
        output_files = [str(file[2] / f"{file[0]}_int-bodycomposition.nii.gz") for file in datalist]
        model_path = config_dict['paths']['weights']['int-bodycomposition'] + '/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres'
        model_folds = [0, 1, 2, 3, 4]
        model_mirror = True
    elif args.model == 'BodyCompositionCT-ResEncM':
        output_files = [str(file[2] / f"{file[0]}_int-bodycomposition.nii.gz") for file in datalist]
        model_path = config_dict['paths']['weights']['int-bodycomposition'] + '/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres'
        model_folds = 'all'
        model_mirror = True
    else:
        raise ValueError(f"Model {args.model} not recognized.")
    
    # set device
    if torch.cuda.is_available():
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        logging.info(f'device set to cuda: using {torch.cuda.get_device_name()}.')
    else:
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
        logging.info(f'CUDA not available, using cpu w/ {torch.get_num_threads()} threads')
    
    # initialize the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=model_mirror,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=True,
        allow_tqdm=True,
    )

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds=model_folds,
        checkpoint_name='checkpoint_final.pth',
    )

    print(f"Input files: {input_files}")

    # run the prediction
    predictor.predict_from_files(input_files,
                                 output_files,
                                 save_probabilities=False, overwrite=config_dict['run']['reset'],
                                 num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    
    # print
    logging.info("FINISHED PIPELINE.")

if __name__ == "__main__":
    main() # parser is in main to be available when using pyproject.toml entrypoint
