# general imports
from nibabel import Nifti1Image
from pathlib import Path
from typing import Union
from time import time
import logging
import re

# specific imports
from BodyComposition.utils.config import update_config
from BodyComposition.utils.logging import init_logging
from BodyComposition.pipeline import PipelineBuilder, run_file, run_batch
from BodyComposition.utils.datalist import DatalistBuilder


def bodycomposition(input: Union[str, Path, Nifti1Image],
                    input_filter: str = r'.*\.nii\.gz$',
                    workspace: Union[str, Path] = None,
                    method: str = 'bodycomposition',
                    config: Union[dict, Path] = None,
                    ):
    """python API for body composition analysis.
    Args:
        input: Path to  directory / datalist file / NIfTI file, or directly a NIfTI file.
        input_filter: Regular expression to filter input files.
        workspace: path to output directory. if none, output will be stored in the same directory as the input.
        method: pipeline method to be used.
        config: path to configuration file or dictionary with configuration. Overwrites default configuration.
    """

    # generate timestamp
    timer_pipeline = time()
    timestamp = int(timer_pipeline)

    # simplify input_filter: remove all non-alphanumeric characters
    input_filter_simple = re.sub(r'\W+', '', input_filter)
    
    # load config: general < pipeline specific < input
    config_dict = update_config({}, Path('./config/general.yaml'))
    config_dict = update_config(config_dict, Path('./config/labels.yaml'))
    config_dict = update_config(config_dict, Path('./config') / f'{method}.yaml')
    if config is not None:
        config_dict = update_config(config_dict, config)

    # load logging, default: level_file=logging.INFO, level_console= logging.WARNING
    path_logging = Path(str(config_dict['paths']['logs']).format(method=method,filter=input_filter_simple,timestamp=timestamp))
    init_logging(file=path_logging,
                 level_file=config_dict['logging_level']['file'],
                 level_console=config_dict['logging_level']['console'])
    logging.info(f"loaded config, start logging now...")

    # load workspace path: argument < config < none, allow string formatting
    if workspace is None:
        workspace = config_dict['paths']['workspace']
    if str(workspace) not in ('None', ''):
        workspace = Path(str(workspace).format(method=method,filter=input_filter_simple,timestamp=timestamp))
    else:
        workspace = None

    # build pipeline
    pipeline = PipelineBuilder(method = method,
                               config = config_dict,
                               timestamp = timestamp)
    
    # get input and output files, inputs = requirements, outputs = relevant if overwrite is set
    io_inputs, io_outputs = pipeline.get_io()
    
    # if input is nifti, single execution and return
    if isinstance(input, Nifti1Image):
        if len(io_inputs) != 1 or io_inputs[0] != "tmp/index": 
            raise ValueError(f'Pipeline requires more than just a single nifti ({io_inputs}). Use an input directory instead.')
        else:
            output = run_file(pipeline, input, workspace)

    # elif: input is directory, datalist, path to file
    else:
        # iterable tuple (id, file, workspace=output_dir) of inputs according to criteria
        datalist = DatalistBuilder(input_path = input,
                                   input_filter = input_filter,
                                   workspace = workspace,
                                   io_inputs = io_inputs,
                                   io_outputs = io_outputs,)

        # remove all outputs? WARNING: this deletes all files!
        if config_dict['run']['reset']:
            datalist.reset_outputs()

        # skip completed cases?
        if config_dict['run']['skip']:
            datalist.skip_completed()
        
        logging.info(f"final datalist: {len(datalist)} cases\n")

        # run pipeline
        output = run_batch(pipeline, datalist)      
    
    logging.info(f" completed {len(datalist)} case(s) in {time() - timer_pipeline:.1f}s.\n\n")
    return output 
