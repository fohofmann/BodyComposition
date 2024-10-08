import logging
from time import time
from typing import Dict, List, Tuple
from pathlib import Path
from BodyComposition.utils.nifti import NiftiDataContainer
from BodyComposition.pipeline_registry import pipeline_registry
import traceback
import signal
from tqdm import tqdm
import sys
import tempfile
import torch
import multiprocessing
import os

# timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError()
signal.signal(signal.SIGALRM, timeout_handler)

# underlying processor
def _run_case(pipeline, memory):
    try:
        signal.alarm(pipeline.config['run']['timeout'])
        output = pipeline(memory)
        signal.alarm(0)
        return output
    except TimeoutError:
        logging.warning(f"TIMEOUT CASE {memory['id']}\n")
    except Exception as e:
        logging.error(f"ERROR CASE {memory['id']}:\n {e}\n {traceback.format_exc()}\n")
    return None

# run pipeline on single file
def run_file(pipeline, input_file, workspace=None):
    logging.info(f"STARTING PIPELINE:\n")
    with tempfile.TemporaryDirectory(prefix="tmp_") as path_tmp:
        logging.info(f"created temporary directory: {path_tmp}")
        memory = {'id': 'tmp',
                  'workspace': workspace or Path(path_tmp),
                  'tmp/index': NiftiDataContainer(Path(path_tmp)/'input/tmp.nii.gz'),}
        memory['tmp/index'].data_nib = input_file
        output = _run_case(pipeline, memory)
    logging.info("FINISHED PIPELINE.")
    return output

# run pipeline on batch of files
def run_batch(pipeline, input_datalist):
    logging.info(f"STARTING PIPELINE:\n")
    output = None
    for caseid, input_file, workspace in tqdm(input_datalist, total=len(input_datalist),
                                               desc="Processing", unit="case", position=0, leave=True, file=sys.stdout, ncols=80):
        memory = {'id': caseid,
                  'workspace': workspace,
                  'tmp/index': NiftiDataContainer(input_file),}
        output = _run_case(pipeline, memory)
    logging.info("FINISHED PIPELINE.")
    return output



class PipelineAction():
    """Base class for pipeline actions."""

    def __init__(self, pipeline, task=None):
        logging.info(f' initialize {self.__class__.__name__}{f"/{task}" if task else ""}')
        self.config = pipeline.config
        self.io_inputs = []
        self.io_outputs = []
        pass

    def __call__(self, memory, task=None) -> Dict:
        logging.info(f'{self.__class__.__name__}{f"/{task}" if task else ""}')

    def __repr__(self):
        return self.__class__.__name__



class PipelineBuilder():
    """Pipeline class"""

    def __init__(self, method: str, config: Dict, timestamp: int):
        """Initialize pipeline."""
        logging.info(f'BUILDING PIPELINE {method.upper()}:')
        
        # save main parameters as attributes
        self.method = method
        self.config = config
        self.timestamp = timestamp

        # set device
        if torch.cuda.is_available():
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            self.device = torch.device('cuda')
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
            logging.info(f'device set to cuda: using {torch.cuda.get_device_name()}.')
        else:
            torch.set_num_threads(multiprocessing.cpu_count())
            self.device = torch.device('cpu')
            logging.info(f'CUDA not available, using cpu w/ {torch.get_num_threads()} threads')

        # check if method is valid
        if method not in pipeline_registry:
            raise ValueError(f'Undefined pipeline method: {method}')

        # function factory: load actions, use list of functions
        logging.info(f'initalizing pipeline actions:')
        self.actions = pipeline_registry[method](pipeline=self)

        # check if all actions are valid
        for action in self.actions:
            if not isinstance(action, PipelineAction):
                raise TypeError(f'Invalid PipelineAction: {action}')

        # log
        logging.info(f'building completed.\n')
        

    def get_io(self) -> Tuple[List, List]:
        """
        get input and output files for built pipeline
        returns: list of input files and list of output files, excluding all files generated by pipeline and tmp
        """
        io_inputs = []
        io_outputs = []
        io_outputs_set = set()
        for action in self.actions:
            io_inputs.extend(input for input in action.io_inputs if input not in io_outputs_set and not input.startswith('tmp/'))
            new_outputs = [output for output in action.io_outputs if not output.startswith('tmp/')]
            io_outputs.extend(new_outputs)
            io_outputs_set.update(new_outputs)
        return io_inputs, io_outputs
    
    def get_licenses(self) -> List[str]:
        """
        get licenses for built pipeline
        returns: list of licenses required by pipeline
        """
        licenses = []
        for action in self.actions:
            if hasattr(action, 'licenses'):
                licenses.extend(action.licenses)
        return list(set(licenses))


    def __call__(self, memory):
        logging.info(f"PROCESSING CASE {memory['id']}:")
        logging.info(f"workspace: {memory['workspace']}")
        timer = time()
        for action in self.actions:
            action(memory)
        logging.info(f"FINISHED CASE {memory['id']} ({time() - timer:.1f}s)\n")
        return memory.get('tmp/return', None)


if __name__ == "__main__":
    raise RuntimeError("Not made to be called directly.")
