"""Pipeline actions and selection for common-contract vertebral backends."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import SimpleITK as sitk

from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.geometry import (
    GeometryError,
    ImageGeometry,
    assert_same_physical_domain,
)
from BodyComposition.utils.nifti import NiftiDataContainer
from BodyComposition.vertebral.contracts import ExecutionStatus, VertebralResult
from BodyComposition.vertebral.review import write_spine_review
from BodyComposition.vertebral.spineps_adapter import adapt_spineps_sitk_outputs
from BodyComposition.vertebral.spineps_backend import (
    SpinepsVeridahAdapter,
    orientation_qc_flags,
)
from BodyComposition.vertebral.spineps_manifest import SPINEPS_BACKEND_ID

SPINEPS_BODY_MASK = "masks/vertebral_bodies.nii.gz"
SPINEPS_SEMANTIC_MASK = "masks/spine_semantic.nii.gz"
SPINEPS_WHOLE_MASK = "masks/vertebra_labels.nii.gz"
SPINEPS_RESULT_JSON = "qc/vertebral_result.json"
SPINEPS_REVIEW_PNG = "qc/spine_review.png"


def _atomic_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    return destination


def _suffix(path: Path) -> str:
    return ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix


def _write_json_atomic(payload: Any, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def _redact_absolute_paths(value: Any, model_cache_root: Path) -> Any:
    if isinstance(value, str):
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            return value
        try:
            relative = candidate.relative_to(model_cache_root)
        except ValueError:
            return "<external-path>"
        return f"<mounted-model-cache>/{relative.as_posix()}"
    if isinstance(value, dict):
        return {
            key: _redact_absolute_paths(item, model_cache_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_absolute_paths(item, model_cache_root) for item in value]
    return value


def _copy_native_output(source: Path, destination: Path, model_cache_root: Path) -> Path:
    if source.suffix.lower() != ".json":
        return _atomic_copy(source, destination)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return _write_json_atomic(
        _redact_absolute_paths(payload, model_cache_root),
        destination,
    )


def _geometry_from_sitk(image: sitk.Image) -> ImageGeometry:
    """Read three-dimensional geometry in SimpleITK's explicit ``xyz`` order."""

    if image.GetDimension() != 3:
        raise GeometryError(
            "The orientation-prepared image must be three-dimensional, "
            f"got {image.GetDimension()}D."
        )
    return ImageGeometry(
        size_xyz=tuple(int(value) for value in image.GetSize()),
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        origin_lps_xyz=tuple(float(value) for value in image.GetOrigin()),
        direction_lps=tuple(float(value) for value in image.GetDirection()),
    )


class SegmSpinepsVeridah(PipelineAction):
    """Run the default pinned backend on the immutable orientation-prepared image."""

    def __init__(self, pipeline):
        super().__init__(pipeline)
        if not self.config["orientation"]["enabled"]:
            raise ValueError(
                "spineps_veridah_ct_v1 requires orientation.enabled=true so it receives "
                "the orientation-prepared image contract."
            )
        settings = self.config["vertebrae"]["spineps"]
        self.adapter = SpinepsVeridahAdapter(
            self.config["paths"]["weights"]["spineps"],
            device=settings["device"],
        )
        self.model_cache_root = Path(
            self.config["paths"]["weights"]["totalsegmentator"]
        ).expanduser()
        self.save_native_outputs = settings["save_native_outputs"]
        self.review_enabled = settings["review_enabled"]
        self.io_inputs = ["tmp/prepared_image"]
        self.io_outputs = [SPINEPS_BODY_MASK, "tmp/vertebral_result", SPINEPS_RESULT_JSON]
        self.io_persisted_outputs = [SPINEPS_BODY_MASK, SPINEPS_RESULT_JSON]
        if self.save_native_outputs:
            self.io_outputs.extend((SPINEPS_SEMANTIC_MASK, SPINEPS_WHOLE_MASK))
            self.io_persisted_outputs.extend((SPINEPS_SEMANTIC_MASK, SPINEPS_WHOLE_MASK))
        if self.review_enabled:
            self.io_outputs.append(SPINEPS_REVIEW_PNG)
            self.io_persisted_outputs.append(SPINEPS_REVIEW_PNG)
        self.io_reset_outputs = [
            SPINEPS_BODY_MASK,
            SPINEPS_SEMANTIC_MASK,
            SPINEPS_WHOLE_MASK,
            SPINEPS_RESULT_JSON,
            SPINEPS_REVIEW_PNG,
        ]

    def set_low_memory_mode(self, enabled: bool = True) -> None:
        self.adapter.set_low_memory_mode(enabled)

    def release_model(self) -> None:
        self.adapter.release_models()

    def _path(self, memory: dict, template: str) -> Path:
        return Path(memory["workspace"]) / template.format(caseid=memory["id"])

    def _container_from_array(
        self,
        path: Path,
        array_zyx: np.ndarray,
        reference: sitk.Image,
    ) -> NiftiDataContainer:
        image = sitk.GetImageFromArray(array_zyx)
        image.CopyInformation(reference)
        container = NiftiDataContainer(path, dtype=np.uint8)
        container.img = image
        container.validate()
        return container

    def _native_outputs(self, memory: dict, result: VertebralResult) -> dict[str, Path]:
        if not self.save_native_outputs:
            return {}
        output: dict[str, Path] = {}
        root = Path(memory["workspace"]) / "native" / "spineps"
        for name, source in sorted(result.native_outputs.items()):
            source_path = Path(source)
            if not source_path.is_file():
                continue
            destination = root / f"{name}{_suffix(source_path)}"
            output[name] = _copy_native_output(
                source_path,
                destination,
                self.model_cache_root,
            )
        return output

    def _promote(self, memory: dict, result: VertebralResult) -> VertebralResult:
        prepared = memory["tmp/prepared_image"]
        reference = prepared.prepared_image
        reference_geometry = result.geometry
        if result.vertebral_body_labels is None or reference_geometry is None:
            raise RuntimeError(result.error_summary or "SPINEPS did not return vertebral-body labels.")

        body_path = self._path(memory, SPINEPS_BODY_MASK)
        body = self._container_from_array(body_path, result.vertebral_body_labels, reference)
        assert_same_physical_domain(
            reference_geometry,
            body.geometry,
            reference_name="SPINEPS result",
            candidate_name="canonical vertebral-body mask",
        )
        body.save_to_file()
        memory[SPINEPS_BODY_MASK] = body

        promoted_native = self._native_outputs(memory, result)
        if self.save_native_outputs:
            semantic_source = result.native_outputs.get("out_spine")
            if semantic_source is None or not Path(semantic_source).is_file():
                raise RuntimeError("SPINEPS did not retain the native semantic output.")
            semantic_path = self._path(memory, SPINEPS_SEMANTIC_MASK)
            _atomic_copy(Path(semantic_source), semantic_path)
            semantic = NiftiDataContainer(semantic_path, dtype=np.uint8)
            semantic.load_from_file()
            semantic.validate()
            assert_same_physical_domain(
                reference_geometry,
                semantic.geometry,
                reference_name="SPINEPS result",
                candidate_name="native semantic mask",
            )
            memory[SPINEPS_SEMANTIC_MASK] = semantic

            if result.whole_vertebra_labels is None:
                raise RuntimeError("SPINEPS did not return native whole-vertebra labels.")
            whole_path = self._path(memory, SPINEPS_WHOLE_MASK)
            whole = self._container_from_array(whole_path, result.whole_vertebra_labels, reference)
            whole.save_to_file()
            memory[SPINEPS_WHOLE_MASK] = whole

        result = replace(result, native_outputs=promoted_native)
        if self.review_enabled:
            review_path = write_spine_review(
                prepared,
                result,
                self._path(memory, SPINEPS_REVIEW_PNG),
                case_id=str(memory["id"]),
            )
            memory[SPINEPS_REVIEW_PNG] = review_path
        result_path = _write_json_atomic(
            result.summary(),
            self._path(memory, SPINEPS_RESULT_JSON),
        )
        memory[SPINEPS_RESULT_JSON] = result_path
        memory["tmp/vertebral_result"] = result
        return result

    def _reuse_identical(self, memory: dict) -> VertebralResult | None:
        if not self.config["run"]["skip"] or not self.save_native_outputs:
            return None
        paths = {
            "body": self._path(memory, SPINEPS_BODY_MASK),
            "semantic": self._path(memory, SPINEPS_SEMANTIC_MASK),
            "whole": self._path(memory, SPINEPS_WHOLE_MASK),
            "summary": self._path(memory, SPINEPS_RESULT_JSON),
        }
        required_paths = list(paths.values())
        if self.review_enabled:
            required_paths.append(self._path(memory, SPINEPS_REVIEW_PNG))
        if not all(path.is_file() for path in required_paths):
            return None
        try:
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        previous = summary.get("provenance", {}).get("orientation", {})
        current = memory["tmp/prepared_image"].result
        if previous.get("prepared_pixel_sha256") != current.prepared_pixel_sha256:
            return None
        try:
            prepared_geometry = _geometry_from_sitk(
                memory["tmp/prepared_image"].prepared_image
            )
            semantic = sitk.ReadImage(str(paths["semantic"]))
            whole = sitk.ReadImage(str(paths["whole"]))
            body = NiftiDataContainer(paths["body"], dtype=np.uint8)
            body.load_from_file()
            body.validate()
            assert_same_physical_domain(
                prepared_geometry,
                body.geometry,
                reference_name="orientation-prepared CT",
                candidate_name="reused vertebral-body mask",
            )
            result = adapt_spineps_sitk_outputs(
                semantic,
                whole,
                prepared_geometry,
                upstream_error_code="ALL_DONE",
                native_outputs={
                    "out_spine": paths["semantic"],
                    "out_vert": paths["whole"],
                },
                provenance={
                    **dict(summary.get("provenance", {})),
                    "reused_identical": True,
                },
            )
        except (GeometryError, OSError, RuntimeError, ValueError):
            return None
        if (
            result.execution_status != ExecutionStatus.SUCCEEDED
            or result.vertebral_body_labels is None
            or not np.array_equal(body.data, result.vertebral_body_labels)
        ):
            return None
        result = replace(
            result,
            execution_status=ExecutionStatus.SKIPPED_IDENTICAL,
            qc_flags=tuple(result.qc_flags) + orientation_qc_flags(memory["tmp/prepared_image"]),
        )
        memory[SPINEPS_BODY_MASK] = body
        semantic_container = NiftiDataContainer(paths["semantic"], dtype=np.uint8)
        semantic_container.load_from_file()
        memory[SPINEPS_SEMANTIC_MASK] = semantic_container
        whole_container = NiftiDataContainer(paths["whole"], dtype=np.uint8)
        whole_container.load_from_file()
        memory[SPINEPS_WHOLE_MASK] = whole_container
        memory[SPINEPS_RESULT_JSON] = paths["summary"]
        if self.review_enabled and self._path(memory, SPINEPS_REVIEW_PNG).is_file():
            memory[SPINEPS_REVIEW_PNG] = self._path(memory, SPINEPS_REVIEW_PNG)
        memory["tmp/vertebral_result"] = result
        return result

    def __call__(self, memory):
        super().__call__(memory)
        reused = self._reuse_identical(memory)
        if reused is not None:
            return
        attempt_parent = Path(memory["workspace"]) / ".stage"
        attempt_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="spineps-", dir=attempt_parent) as temporary:
            result = self.adapter.run(
                memory["tmp/prepared_image"],
                attempt_directory=Path(temporary),
                case_id=str(memory["id"]),
            )
            memory["tmp/vertebral_result"] = result
            if result.execution_status != ExecutionStatus.SUCCEEDED:
                _write_json_atomic(result.summary(), self._path(memory, SPINEPS_RESULT_JSON))
                raise RuntimeError(result.error_summary or "SPINEPS/VERIDAH inference failed.")
            self._promote(memory, result)


def vertebral_backend_actions(pipeline) -> tuple[list[PipelineAction], str]:
    """Return explicit actions and body-mask key; never select a fallback."""

    backend = pipeline.config["vertebrae"]["backend"]
    if backend == SPINEPS_BACKEND_ID:
        return [SegmSpinepsVeridah(pipeline)], SPINEPS_BODY_MASK
    if backend in {"vertebral_bodies_resenc_l", "vertebral_bodies_resenc_m"}:
        from BodyComposition.actions.segm_int import SegmIntVertebrae

        preset = "ResEncL" if backend.endswith("_l") else "ResEncM"
        return [SegmIntVertebrae(pipeline, image="tmp/index", model=preset)], SegmIntVertebrae.output_label_name
    raise ValueError(
        f"Unknown vertebral backend {backend!r}. No alternative backend was attempted."
    )
