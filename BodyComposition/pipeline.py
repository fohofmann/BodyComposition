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
        _try_render_failure_page(pipeline, memory, error)
        return False, error
    except Exception as error:
        logging.error(f"ERROR CASE {memory['id']}:\n {error}\n {traceback.format_exc()}\n")
        _try_render_failure_page(pipeline, memory, error)
        return False, error
    finally:
        signal.alarm(0)


def _try_render_failure_page(pipeline, memory, error):
    """Best-effort explicit status page; never mask the original failure."""

    try:
        from BodyComposition.reporting.contracts import ReportingSettings
        from BodyComposition.reporting.service import render_failed_case_report

        settings = ReportingSettings.from_mapping(pipeline.config)
        if not settings.enabled:
            return
        stage = str(memory.get("tmp/current_action", "pipeline"))
        prepared_outcome = memory.get("tmp/prepared_image")
        prepared_image = getattr(prepared_outcome, "prepared_image", None)
        vertebral_result = memory.get("tmp/vertebral_result")
        result = render_failed_case_report(
            case_id=str(memory["id"]),
            analysis_id=str(memory.get("analysis_id", "analysis_unavailable")),
            output_directory=Path(memory["workspace"]) / "reports" / str(memory["id"]),
            settings=settings,
            failure_stage=stage,
            failure_code=f"{stage}_{error.__class__.__name__}",
            prepared_image=prepared_image,
            vertebral_result=vertebral_result,
        )
        memory["tmp/report_result"] = result
    except Exception as report_error:
        logging.error(
            "Could not generate explicit report failure page for case %s: %s",
            memory.get("id", "unknown"),
            report_error.__class__.__name__,
        )


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
    report_results = []
    report_workspaces = []
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
        if "tmp/report_result" in memory:
            report_results.append(memory["tmp/report_result"])
            report_workspaces.append(Path(workspace))
    reporting = pipeline.config.get("reporting", {})
    if reporting.get("enabled") and reporting.get("combined_pdf") and report_results:
        if len(report_results) != len(input_datalist):
            failed_case_ids.append("report_export_incomplete")
        elif len(set(report_workspaces)) != 1:
            failed_case_ids.append("report_export_requires_common_workspace")
        else:
            try:
                from BodyComposition.reporting.contracts import ReportingSettings
                from BodyComposition.reporting.service import collate_reports

                export_id = f"run-{pipeline.timestamp}"
                memory_output = collate_reports(
                    report_results,
                    export_id=export_id,
                    output_directory=(
                        report_workspaces[0] / "aggregate" / "reports" / export_id
                    ),
                    settings=ReportingSettings.from_mapping(pipeline.config),
                )
                logging.info("combined report: %s", memory_output["pdf_path"])
            except Exception as error:
                logging.error("Combined report generation failed: %s", error)
                failed_case_ids.append("combined_report")
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
        if self.config["reporting"]["enabled"] and not any(
            action.__class__.__name__ == "RenderCaseReport" for action in self.actions
        ):
            raise ValueError(
                f"Reporting is not supported by pipeline {method!r}; use a canonical "
                "BodyComposition pipeline or the explicit post-hoc reporting service."
            )

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
            memory["tmp/current_action"] = action.__class__.__name__
            action(memory)
            action.validate_outputs(memory)
        logging.info(f"FINISHED CASE {memory['id']} ({time() - timer:.1f}s)\n")
        return memory.get('tmp/return', None)


if __name__ == "__main__":
    raise RuntimeError("Not made to be called directly.")
