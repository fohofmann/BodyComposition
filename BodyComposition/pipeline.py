"""Internal action executor used by the stable service facade.

The mutable action-memory dictionary remains an implementation detail.  Public
callers receive immutable :mod:`BodyComposition.results` objects instead.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, time

from BodyComposition.config import PipelineConfig
from BodyComposition.execution import release_accelerator_cache


def _allocated_cpu_count() -> int:
    """Respect scheduler/cgroup affinity before falling back to host CPU count."""

    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            count = len(affinity(0))
            if count > 0:
                return count
        except OSError:
            pass
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            count = int(slurm_cpus)
            if count > 0:
                return count
        except ValueError:
            pass
    return max(1, multiprocessing.cpu_count())


def _resolve_device_name(torch_module, requested: str) -> str:
    """Resolve only backend type; CUDA visibility/selection remains external."""

    if requested == "cpu":
        return "cpu"
    if requested not in {"auto", "cuda"}:
        raise ValueError("runtime.device must be auto, cpu, or cuda.")
    cuda_available = bool(torch_module.cuda.is_available())
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("runtime.device=cuda was requested but CUDA is unavailable.")
        return "cuda"
    if cuda_available:
        return "cuda"
    return "cpu"


class ActionContractError(RuntimeError):
    """Raised when a pipeline action does not satisfy its declared I/O contract."""


class PipelineAction:
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

    def __call__(self, memory, task=None) -> None:
        logging.info(f'{self.__class__.__name__}{f"/{task}" if task else ""}')
        self.validate_inputs(memory)

    @staticmethod
    def _is_available(memory: dict, key: str) -> bool:
        if key in memory:
            return True
        if key.startswith(('tmp/', 'res/')):
            return False
        workspace = memory.get('workspace')
        case_id = memory.get('id')
        if workspace is None or case_id is None:
            return False
        return (Path(workspace) / key.format(caseid=case_id)).exists()

    def validate_inputs(self, memory: dict) -> None:
        missing = [key for key in self.io_inputs if not self._is_available(memory, key)]
        if missing:
            raise ActionContractError(
                f'{self.__class__.__name__} missing declared input(s): {", ".join(missing)}.'
            )

    def validate_outputs(self, memory: dict) -> None:
        missing = [key for key in self.io_outputs if not self._is_available(memory, key)]
        if missing:
            raise ActionContractError(
                f'{self.__class__.__name__} did not create declared output(s): '
                f'{", ".join(missing)}.'
            )

    def __repr__(self):
        return self.__class__.__name__



class InternalPipeline:
    """Validated model executor reused across every case in a batch."""

    def __init__(self, config: PipelineConfig, *, timestamp: int):
        self.public_config = PipelineConfig.model_validate(config)
        self.config = self.public_config.to_runtime_dict()
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

        requested_device = self.public_config.device
        device_name = _resolve_device_name(torch, requested_device)
        configured_threads = int(self.public_config.model_dump()["runtime"]["cpu_threads"])
        self.cpu_threads = configured_threads or _allocated_cpu_count()
        torch.set_num_threads(self.cpu_threads)
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        self.device = torch.device(device_name)
        if device_name == "cuda":
            logging.info(
                "using externally visible CUDA device %s with %s allocated CPU threads",
                torch.cuda.get_device_name(),
                self.cpu_threads,
            )
        else:
            logging.info(
                "using %s with %s allocated CPU threads",
                device_name,
                self.cpu_threads,
            )
        try:
            import SimpleITK as sitk

            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(self.cpu_threads)
        except (ImportError, AttributeError):
            pass
        if self.public_config.model_dump()["runtime"]["deterministic"]:
            torch.use_deterministic_algorithms(True, warn_only=True)

        from BodyComposition.actions.orientation import AssessOrientation
        from BodyComposition.pipelines.bodycomposition import canonical_actions

        self.actions = [AssessOrientation(self), *canonical_actions(self)]
        self.low_memory_mode = False
        self.execution_strategy = "stage_aware_model_cache"
        if self.config["reporting"]["enabled"] and not any(
            action.__class__.__name__ == "RenderCaseReport" for action in self.actions
        ):
            raise ValueError("Reporting was enabled but no report action was composed.")

        for action in self.actions:
            if not isinstance(action, PipelineAction):
                raise TypeError(f'Invalid PipelineAction: {action}')

        logging.info('canonical pipeline initialized with %s actions', len(self.actions))

    def get_reset_outputs(self) -> list[str]:
        """Return file outputs that a requested reset must remove."""
        return list(dict.fromkeys(
            output_name
            for action in self.actions
            for output_name in action.io_reset_outputs
        ))

    @staticmethod
    def _release_action_model(action) -> None:
        release = getattr(action, "release_model", None)
        if callable(release):
            try:
                release()
            except Exception:
                logging.exception("Could not release model state for %s.", action)

    def enable_low_memory_mode(self) -> None:
        """Keep at most the currently executing segmentation bundle resident."""

        if self.low_memory_mode:
            return
        self.low_memory_mode = True
        logging.warning(
            "low-memory execution enabled: segmentation bundles will be unloaded between stages"
        )
        for action in self.actions:
            configure = getattr(action, "set_low_memory_mode", None)
            if callable(configure):
                configure(True)
            self._release_action_model(action)
        release_accelerator_cache(self.device)

    def release_models(self) -> None:
        for action in self.actions:
            self._release_action_model(action)
        release_accelerator_cache(self.device)

    def prepare_exclusive_model(self, requesting_action: PipelineAction) -> None:
        """Release other model bundles before a memory-intensive support stage."""

        for action in self.actions:
            if action is not requesting_action:
                self._release_action_model(action)
        release_accelerator_cache(self.device)


    def __call__(self, memory: dict):
        logging.info(f"PROCESSING CASE {memory['id']}:")
        timer = time()
        started_monotonic = monotonic()
        timeout_seconds = int(self.public_config.model_dump()["runtime"]["timeout_seconds"])
        stage_events = memory.setdefault("tmp/stage_events", [])
        lease = memory.get("tmp/case_lease")
        for action in self.actions:
            if lease is not None and lease.state.stop_requested():
                raise KeyboardInterrupt
            elapsed = monotonic() - started_monotonic
            if elapsed >= timeout_seconds:
                raise TimeoutError(
                    f"Case time budget exceeded before {action.__class__.__name__}."
                )
            memory["tmp/current_action"] = action.__class__.__name__
            if lease is not None:
                lease.set_stage(action.__class__.__name__)
            stage_started = datetime.now(UTC)
            stage_timer = monotonic()
            try:
                action(memory)
                action.validate_outputs(memory)
            except BaseException as error:
                stage_events.append(
                    {
                        "stage": action.__class__.__name__,
                        "started_at": stage_started.isoformat(),
                        "ended_at": datetime.now(UTC).isoformat(),
                        "duration_seconds": monotonic() - stage_timer,
                        "result_code": type(error).__name__,
                    }
                )
                raise
            finally:
                if self.low_memory_mode:
                    self._release_action_model(action)
                    release_accelerator_cache(self.device)
            stage_events.append(
                {
                    "stage": action.__class__.__name__,
                    "started_at": stage_started.isoformat(),
                    "ended_at": datetime.now(UTC).isoformat(),
                    "duration_seconds": monotonic() - stage_timer,
                    "result_code": "succeeded",
                }
            )
            if monotonic() - started_monotonic >= timeout_seconds:
                raise TimeoutError(
                    f"Case time budget exceeded after {action.__class__.__name__}."
                )
        logging.info(f"FINISHED CASE {memory['id']} ({time() - timer:.1f}s)\n")
        return memory.get('tmp/return')


if __name__ == "__main__":
    raise RuntimeError("Not made to be called directly.")
