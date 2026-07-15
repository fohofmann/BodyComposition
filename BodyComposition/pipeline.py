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
import multiprocessing
import os


class ActionContractError(RuntimeError):
    """Raised when a pipeline action does not satisfy its declared I/O contract."""


class PipelineExecutionError(RuntimeError):
    """Raised after one or more cases fail during pipeline execution."""

# timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError()
signal.signal(signal.SIGALRM, timeout_handler)

# underlying processor
def _run_case(pipeline, memory):
    try:
        signal.alarm(pipeline.config['run']['timeout'])
        output = pipeline(memory)
        return True, output
    except TimeoutError as error:
        logging.warning(f"TIMEOUT CASE {memory['id']}\n")
        return False, error
    except Exception as error:
        logging.error(f"ERROR CASE {memory['id']}:\n {error}\n {traceback.format_exc()}\n")
        return False, error
    finally:
        signal.alarm(0)


def _raise_case_failures(case_ids):
    if case_ids:
        raise PipelineExecutionError(
            f"Pipeline failed for {len(case_ids)} case(s): {', '.join(case_ids)}. "
            "See the pipeline log for the original exception(s)."
        )

# run pipeline on single file
def run_file(pipeline, input_file, workspace=None):
    logging.info(f"STARTING PIPELINE:\n")
    with tempfile.TemporaryDirectory(prefix="tmp_") as path_tmp:
        logging.info(f"created temporary directory: {path_tmp}")
        memory = {'id': 'tmp',
                  'workspace': workspace or Path(path_tmp),
                  'tmp/index': NiftiDataContainer(Path(path_tmp)/'input/tmp.nii.gz'),}
        memory['tmp/index'].img = input_file
        success, output = _run_case(pipeline, memory)
        _raise_case_failures([] if success else [memory['id']])
    logging.info("FINISHED PIPELINE.")
    return output

# run pipeline on batch of files
def run_batch(pipeline, input_datalist):
    logging.info(f"STARTING PIPELINE:\n")
    output = None
    failed_case_ids = []
    for caseid, input_file, workspace in tqdm(input_datalist, total=len(input_datalist),
                                               desc="Processing", unit="case", position=0, leave=True, file=sys.stdout, ncols=80):
        memory = {'id': caseid,
                  'workspace': workspace,
                  'tmp/index': NiftiDataContainer(input_file),}
        success, case_output = _run_case(pipeline, memory)
        if success:
            output = case_output
        else:
            failed_case_ids.append(caseid)
    logging.info("FINISHED PIPELINE.")
    _raise_case_failures(failed_case_ids)
    return output



class PipelineAction():
    """Base class for pipeline actions."""

    def __init__(self, pipeline, task=None):
        logging.info(f' initialize {self.__class__.__name__}{f"/{task}" if task else ""}')
        self.config = pipeline.config
        self.io_inputs = []
        self.io_outputs = []
        # File-backed completion markers are distinct from in-memory outputs.
        # Actions that persist files opt in explicitly.
        self.io_persisted_outputs = []
        self.io_reset_outputs = []
        pass

    def __call__(self, memory, task=None) -> Dict:
        logging.info(f'{self.__class__.__name__}{f"/{task}" if task else ""}')
        self.validate_inputs(memory)

    @staticmethod
    def _is_available(memory: Dict, key: str) -> bool:
        if key in memory:
            return True
        if key.startswith(('tmp/', 'res/')):
            return False
        workspace = memory.get('workspace')
        case_id = memory.get('id')
        if workspace is None or case_id is None:
            return False
        return (Path(workspace) / key.format(caseid=case_id)).exists()

    def validate_inputs(self, memory: Dict) -> None:
        missing = [key for key in self.io_inputs if not self._is_available(memory, key)]
        if missing:
            raise ActionContractError(
                f'{self.__class__.__name__} missing declared input(s): {", ".join(missing)}.'
            )

    def validate_outputs(self, memory: Dict) -> None:
        missing = [key for key in self.io_outputs if not self._is_available(memory, key)]
        if missing:
            raise ActionContractError(
                f'{self.__class__.__name__} did not create declared output(s): '
                f'{", ".join(missing)}.'
            )

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

        from BodyComposition.utils.config import validate_config
        validate_config(self.config)

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                'PyTorch is required to build model-backed pipelines. '
                'Pure measurement modules and CLI help can be used without it.'
            ) from exc

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
        if self.config["orientation"]["enabled"]:
            from BodyComposition.actions.orientation import AssessOrientation

            self.actions.insert(0, AssessOrientation(self))

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
        produced_outputs = set()
        for action in self.actions:
            io_inputs.extend(
                input_name
                for input_name in action.io_inputs
                if input_name not in produced_outputs and not input_name.startswith('tmp/')
            )
            produced_outputs.update(action.io_outputs)
            io_outputs.extend(
                output_name
                for output_name in action.io_persisted_outputs
                if not output_name.startswith('tmp/')
            )
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

    def get_reset_outputs(self) -> List[str]:
        """Return file outputs that a requested reset must remove."""
        return list(dict.fromkeys(
            output_name
            for action in self.actions
            for output_name in action.io_reset_outputs
        ))


    def __call__(self, memory):
        logging.info(f"PROCESSING CASE {memory['id']}:")
        logging.info(f"workspace: {memory['workspace']}")
        timer = time()
        for action in self.actions:
            action(memory)
            action.validate_outputs(memory)
        logging.info(f"FINISHED CASE {memory['id']} ({time() - timer:.1f}s)\n")
        return memory.get('tmp/return', None)


if __name__ == "__main__":
    raise RuntimeError("Not made to be called directly.")
