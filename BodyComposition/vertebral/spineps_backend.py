"""Cohesive upstream-first SPINEPS/VERIDAH backend surface."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from BodyComposition.orientation.core import OrientationOutcome
from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.vertebral.contracts import (
    ExecutionStatus,
    QCFlag,
    QCSeverity,
    VertebralResult,
)
from BodyComposition.vertebral.spineps_assets import (
    AssetVerificationError,
    ModelSyncReport,
    sync_models as sync_pinned_models,
    verify_model_bundle,
)
from BodyComposition.vertebral.spineps_manifest import (
    SPINEPS_BACKEND_ID,
    SPINEPS_VERSION,
    TPTBOX_COMPAT_VERSION,
)
from BodyComposition.vertebral.spineps_runtime import SpinepsRuntime


MODEL_SYNC_COMMAND = "bodycomposition_download_models --model SPINEPS-VERIDAH"


@dataclass(frozen=True)
class ReadinessCheck:
    """One stable readiness assertion and its remediation."""

    component: str
    ready: bool
    detail: str
    remediation: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "ready": self.ready,
            "detail": self.detail,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class ReadinessReport:
    """Structured, non-downloading SPINEPS preflight result."""

    backend_id: str
    model_root: Path
    device: str
    checks: tuple[ReadinessCheck, ...]

    @property
    def ready(self) -> bool:
        return all(item.ready for item in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "ready": self.ready,
            "model_root": str(self.model_root),
            "device": self.device,
            "checks": [item.as_dict() for item in self.checks],
        }


def _callable_check(module_name: str, attribute: str) -> ReadinessCheck:
    component = f"{module_name}.{attribute}"
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, attribute)
    except (ImportError, AttributeError) as error:
        return ReadinessCheck(
            component,
            False,
            f"Pinned callable is unavailable: {error}",
            "Run `uv sync --frozen` in the BodyComposition environment.",
        )
    return ReadinessCheck(
        component,
        callable(value),
        "Pinned callable is available." if callable(value) else "Resolved object is not callable.",
        None if callable(value) else "Recreate the environment with `uv sync --frozen`.",
    )


def _writable_location(path: Path, component: str) -> ReadinessCheck:
    candidate = Path(path)
    probe = candidate
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    ready = probe.is_dir() and os.access(probe, os.W_OK | os.X_OK)
    return ReadinessCheck(
        component,
        ready,
        f"Writable ancestor: {probe}." if ready else f"No writable ancestor for {candidate}.",
        None if ready else f"Choose a writable {component.replace('_', ' ')}.",
    )


def orientation_qc_flags(prepared: OrientationOutcome) -> tuple[QCFlag, ...]:
    flags = [
        QCFlag(
            code=f"orientation_{item.code}",
            stage="orientation",
            severity=QCSeverity.WARNING,
            reason=item.reason,
            observed={**dict(item.observed), "source_severity": item.severity},
            thresholds=dict(item.threshold),
            suggested_review_action=item.suggested_action,
        )
        for item in prepared.result.review_flags
    ]
    if prepared.result.orientation_changed and not any(
        flag.code == "orientation_changed" for flag in flags
    ):
        flags.append(
            QCFlag(
                code="orientation_changed",
                stage="orientation",
                severity=QCSeverity.WARNING,
                reason="preparation stage changed the CT orientation before vertebral inference.",
                observed={
                    "original_orientation_code": prepared.result.original_orientation_code,
                    "prepared_orientation_code": prepared.result.prepared_orientation_code,
                },
                suggested_review_action="Review the orientation scout and vertebral overlay.",
            )
        )
    return tuple(flags)


def _prepared_geometry(prepared: OrientationOutcome) -> ImageGeometry:
    image = prepared.prepared_image
    if image.GetDimension() != 3:
        raise TypeError("SPINEPS requires a three-dimensional prepared image.")
    return ImageGeometry(
        size_xyz=tuple(int(value) for value in image.GetSize()),
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        origin_lps_xyz=tuple(float(value) for value in image.GetOrigin()),
        direction_lps=tuple(float(value) for value in image.GetDirection()),
    )


def _failed_run_result(
    prepared: OrientationOutcome,
    *,
    code: str,
    reason: str,
    observed: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> VertebralResult:
    return VertebralResult(
        backend_id=SPINEPS_BACKEND_ID,
        execution_status=ExecutionStatus.FAILED,
        geometry=_prepared_geometry(prepared),
        whole_vertebra_labels=None,
        vertebral_body_labels=None,
        provenance=dict(provenance or {}),
        qc_flags=(
            QCFlag(
                code=code,
                stage="vertebral",
                severity=QCSeverity.ERROR,
                reason=reason,
                observed=dict(observed or {}),
            ),
        ),
        error_summary=reason,
    )


class SpinepsVeridahAdapter:
    """Pinned SPINEPS/VERIDAH adapter with check, sync_models, and run."""

    def __init__(
        self,
        model_root: Path,
        *,
        device: str = "auto",
        runtime_factory: Callable[..., SpinepsRuntime] = SpinepsRuntime.from_model_root,
    ) -> None:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("SPINEPS device must be auto, cpu, or cuda.")
        self.model_root = Path(model_root)
        self.requested_device = device
        self._runtime_factory = runtime_factory
        self._runtime: SpinepsRuntime | None = None

    def _resolved_device(self) -> tuple[str, ReadinessCheck]:
        try:
            import torch
        except ImportError as error:
            return "cpu", ReadinessCheck(
                "device",
                False,
                f"PyTorch is unavailable: {error}",
                "Run `uv sync --frozen`.",
            )
        cuda_available = bool(torch.cuda.is_available())
        device = (
            "cuda"
            if self.requested_device == "cuda"
            or (self.requested_device == "auto" and cuda_available)
            else "cpu"
        )
        ready = self.requested_device != "cuda" or cuda_available
        return device, ReadinessCheck(
            "device",
            ready,
            f"Resolved {self.requested_device!r} to {device}; CUDA available={cuda_available}.",
            None if ready else "Use device=cpu or run in the CUDA-enabled Docker image.",
        )

    def check(
        self,
        *,
        output_root: Path | None = None,
        full_models: bool = False,
    ) -> ReadinessReport:
        """Check readiness without downloading or modifying model assets."""

        checks: list[ReadinessCheck] = []
        for distribution, expected in (
            ("SPINEPS", SPINEPS_VERSION),
            ("TPTBox", TPTBOX_COMPAT_VERSION),
        ):
            try:
                observed = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                observed = None
            ready = observed == expected
            checks.append(
                ReadinessCheck(
                    f"package:{distribution}",
                    ready,
                    f"Expected {expected}; observed {observed or 'not installed'}.",
                    None if ready else "Run `uv sync --frozen`.",
                )
            )
        checks.extend(
            _callable_check(module, function)
            for module, function in (
                ("spineps.get_models", "get_actual_model"),
                ("spineps.seg_run", "output_paths_from_input"),
                ("spineps.seg_run", "process_img_nii"),
                (
                    "TPTBox.segmentation.VibeSeg.inference_nnunet",
                    "run_inference_on_file",
                ),
            )
        )
        try:
            verify_model_bundle(self.model_root, full=full_models)
        except (AssetVerificationError, OSError) as error:
            checks.append(
                ReadinessCheck(
                    "model_bundle",
                    False,
                    str(error),
                    MODEL_SYNC_COMMAND,
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    "model_bundle",
                    True,
                    "All pinned SPINEPS/VERIDAH and VibeSeg Dataset100 assets are verified.",
                )
            )
        checks.append(_writable_location(self.model_root, "model_cache"))
        if output_root is not None:
            checks.append(_writable_location(Path(output_root), "output_root"))
        device, device_check = self._resolved_device()
        checks.append(device_check)
        return ReadinessReport(
            backend_id=SPINEPS_BACKEND_ID,
            model_root=self.model_root,
            device=device,
            checks=tuple(checks),
        )

    def sync_models(self, **kwargs: Any) -> ModelSyncReport:
        """Synchronize the exact pinned upstream assets."""

        return sync_pinned_models(self.model_root, **kwargs)

    def run(
        self,
        prepared: OrientationOutcome,
        *,
        attempt_directory: Path,
        case_id: str,
    ) -> VertebralResult:
        """Run on preparation stage's prepared image; never reorient or fall back."""

        if not isinstance(prepared, OrientationOutcome):
            raise TypeError("SPINEPS run requires the preparation stage OrientationOutcome object.")
        result: VertebralResult
        if self._runtime is None:
            report = self.check(output_root=Path(attempt_directory), full_models=False)
            if not report.ready:
                failed_checks = [item for item in report.checks if not item.ready]
                result = _failed_run_result(
                    prepared,
                    code="backend_not_ready",
                    reason="The pinned SPINEPS/VERIDAH backend is not ready; run the documented preflight and model-sync command.",
                    observed={
                        "components": [item.component for item in failed_checks],
                        "remediations": sorted(
                            {
                                item.remediation
                                for item in failed_checks
                                if item.remediation is not None
                            }
                        ),
                    },
                )
            else:
                try:
                    self._runtime = self._runtime_factory(
                        self.model_root,
                        use_cpu=report.device == "cpu",
                    )
                except Exception as error:
                    logging.exception(
                        "SPINEPS/VERIDAH runtime initialization failed for case %s.",
                        case_id,
                    )
                    result = _failed_run_result(
                        prepared,
                        code="runtime_initialization_failed",
                        reason=(
                            "SPINEPS/VERIDAH runtime initialization failed; "
                            "see the case log for the original exception."
                        ),
                        observed={"exception_type": type(error).__name__},
                    )
                else:
                    result = self._run_runtime(prepared, attempt_directory, case_id)
        else:
            result = self._run_runtime(prepared, attempt_directory, case_id)
        orientation_flags = orientation_qc_flags(prepared)
        provenance: Mapping[str, Any] = {
            **dict(result.provenance),
            "prepared_orientation_state": prepared.result.state.value,
            "orientation_changed": prepared.result.orientation_changed,
            "orientation_manual_review_required": prepared.result.manual_review_required,
        }
        return replace(
            result,
            provenance=provenance,
            qc_flags=tuple(result.qc_flags) + orientation_flags,
        )

    def _run_runtime(
        self,
        prepared: OrientationOutcome,
        attempt_directory: Path,
        case_id: str,
    ) -> VertebralResult:
        assert self._runtime is not None
        try:
            return self._runtime.run(
                prepared,
                attempt_directory=Path(attempt_directory),
                case_id=case_id,
            )
        except Exception as error:
            logging.exception(
                "SPINEPS/VERIDAH inference failed for case %s.",
                case_id,
            )
            return _failed_run_result(
                prepared,
                code="upstream_inference_exception",
                reason=(
                    "SPINEPS/VERIDAH inference raised an exception; "
                    "see the case log for the original exception."
                ),
                observed={"exception_type": type(error).__name__},
            )
