"""Pinned SPINEPS execution with an explicit, precomputed VibeSeg crop mask."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import SimpleITK as sitk
from psutil import virtual_memory

from BodyComposition.orientation.core import OrientationOutcome
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain
from BodyComposition.vertebral.contracts import VertebralResult
from BodyComposition.vertebral.spineps_adapter import (
    adapt_spineps_outputs,
    adapt_spineps_sitk_outputs,
)
from BodyComposition.vertebral.spineps_assets import (
    ASSET_MANIFEST_NAME,
    VIBESEG_INSTALL_DIR,
    sha256_file,
    verify_installed_vibeseg,
)
from BodyComposition.vertebral.spineps_manifest import (
    VIBESEG_CROP_ASSETS,
    VIBESEG_CROP_RELEASE,
)
from BodyComposition.vertebral.spineps_session import SpinepsModelSession

_CITATION_REMINDER_CONFIGURED = False
_TPTBOX_INFERENCE_LOCK = threading.RLock()
_TPTBOX_CPU_TELEMETRY_ADAPTER = {
    "id": "tptbox-0.7.5-cpu-memory-telemetry",
    "upstream_repository": "https://github.com/Hendrik-code/TPTBox",
    "upstream_revision": "acaaf16f74fb0fe8fc555b23cf4e0230efc49753",
    "upstream_function": (
        "TPTBox.segmentation.nnUnet_utils.predictor."
        "nnUNetPredictor.predict_sliding_window_return_logits"
    ),
    "reason": "Pinned TPTBox 0.7.5 calls CUDA memory telemetry for a CPU device.",
    "effect": "Disable GPU waiting and use available host memory on CPU only.",
}


def _configure_upstream_citation_reminder() -> None:
    """Keep machine stdout clean while retaining explicit project attribution."""

    global _CITATION_REMINDER_CONFIGURED
    if _CITATION_REMINDER_CONFIGURED:
        return
    os.environ["SPINEPS_TURN_OF_CITATION_REMINDER"] = "TRUE"
    from spineps.utils import citation_reminder

    # SPINEPS 2.0.0 registers an exit-time stdout banner at import. Machine
    # stdout remains reserved for CLI results; attribution is retained in
    # notices, model provenance, documentation, and the stderr log.
    atexit.unregister(citation_reminder.print_citation_reminder)
    logging.info(
        "SPINEPS/VERIDAH inference requires citation; see THIRD_PARTY_NOTICES.md."
    )
    _CITATION_REMINDER_CONFIGURED = True


def _make_bids_file(input_path: Path) -> Any:
    from TPTBox import BIDS_FILE

    return BIDS_FILE(str(input_path), dataset=str(input_path.parent))


def _derive_output_paths(img_ref: Any, derivative_name: str) -> Mapping[str, Path]:
    from spineps.seg_run import output_paths_from_input

    return output_paths_from_input(
        img_ref,
        derivative_name,
        None,
        input_format=img_ref.format,
        non_strict_mode=True,
    )


@contextmanager
def _tptbox_vibeseg_runtime_guard(device: str) -> Iterator[dict[str, str] | None]:
    """Make the pinned upstream VibeSeg CPU telemetry device-safe."""

    with _TPTBOX_INFERENCE_LOCK:
        if device.casefold().partition(":")[0] != "cpu":
            yield None
            return

        import TPTBox.segmentation.nnUnet_utils.predictor as predictor_module

        original_util = predictor_module.get_gpu_util
        original_memory = predictor_module.get_gpu_memory_MB

        def cpu_gpu_util(_device: Any) -> float:
            return 0.0

        def cpu_free_gpu_memory_mb(_device: Any) -> float:
            return float(virtual_memory().available / 1024**2)

        predictor_module.get_gpu_util = cpu_gpu_util
        predictor_module.get_gpu_memory_MB = cpu_free_gpu_memory_mb
        try:
            yield dict(_TPTBOX_CPU_TELEMETRY_ADAPTER)
        finally:
            predictor_module.get_gpu_util = original_util
            predictor_module.get_gpu_memory_MB = original_memory


def _run_explicit_vibeseg(
    trained_model: Path,
    input_nii: Any,
    output_path: Path,
    device: str,
    *,
    cache_model: bool = True,
) -> dict[str, str] | None:
    import TPTBox.segmentation.nnUnet_utils.inference_api as inference_api
    from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file

    # InternalPipeline initializes PyTorch's process-global inter-op pool before
    # any model work. The pinned TPTBox loader tracks the same one-time state in
    # this module flag; keeping it synchronized avoids a redundant initialization
    # attempt after CTDeepRot has already used PyTorch.
    inference_api._interop = True

    with _tptbox_vibeseg_runtime_guard(device) as runtime_adapter:
        run_inference_on_file(
            trained_model,
            [input_nii],
            out_file=output_path,
            override=False,
            keep_size=False,
            padd=5,
            max_folds=None,
            ddevice=device,
            memory_base=5500,
            memory_factor=25,
            auto_download=False,
            cache_model=cache_model,
        )
    return runtime_adapter


def _run_spineps(img_ref: Any, models: Any, derivative_name: str) -> Any:
    from spineps.seg_run import process_img_nii

    return process_img_nii(
        img_ref=img_ref,
        model_semantic=models.semantic,
        model_instance=models.instance,
        model_labeling=models.labeling,
        derivative_name=derivative_name,
        save_modelres_mask=False,
        save_softmax_logits=False,
        save_debug_data=False,
        save_raw=True,
        auto_crop_to_spine=False,
        proc_lab_force_no_tl_anomaly=False,
        ignore_bids_filter=True,
        # The CT model's BIDS descriptor says "iso", but SPINEPS accepts
        # anisotropic CT by resampling to its 0.8-mm grid and returning outputs
        # in input space. This adapter validates the input and output physical
        # domains around that operation.
        ignore_compatibility_issues=True,
        return_output_instead_of_save=False,
        verbose=False,
    )


def _geometry_from_sitk(image: sitk.Image) -> ImageGeometry:
    return ImageGeometry(
        size_xyz=tuple(int(value) for value in image.GetSize()),
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        origin_lps_xyz=tuple(float(value) for value in image.GetOrigin()),
        direction_lps=tuple(float(value) for value in image.GetDirection()),
    )


def _as_sitk_image(image: Any) -> sitk.Image:
    if isinstance(image, sitk.Image):
        return image
    converter = getattr(image, "to_simpleITK", None)
    if not callable(converter):
        raise TypeError("SPINEPS returned an output that cannot be converted to SimpleITK.")
    converted = converter()
    if not isinstance(converted, sitk.Image):
        raise TypeError("SPINEPS output conversion did not return a SimpleITK image.")
    return converted


def _error_code_name(error_code: Any) -> str:
    name = getattr(error_code, "name", None)
    return str(name if name is not None else error_code)


def _bids_subject_label(case_id: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]", "", case_id)
    if not label:
        raise ValueError("case_id must contain at least one ASCII letter or digit.")
    return label


def _stage_prepared_ct(image: sitk.Image, attempt_directory: Path, case_id: str) -> Path:
    input_directory = attempt_directory / "input"
    input_directory.mkdir(parents=True, exist_ok=True)
    staged = input_directory / f"sub-{_bids_subject_label(case_id)}_ct.nii.gz"
    if staged.exists():
        raise RuntimeError(f"Attempt-local SPINEPS input already exists: {staged}.")
    temporary = staged.with_name(f".{staged.name.removesuffix('.nii.gz')}.partial.nii.gz")
    sitk.WriteImage(image, str(temporary))
    os.replace(temporary, staged)
    return staged


def _vibeseg_inventory_provenance(
    model_root: Path,
    trained_model: Path,
) -> dict[str, Any]:
    """Summarize the verified manifest without rehashing multi-gigabyte weights."""

    bundle_root = Path(model_root) / VIBESEG_INSTALL_DIR
    manifest_path = bundle_root / ASSET_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("files")
    if not isinstance(records, list):
        raise RuntimeError("The verified VibeSeg manifest has no file inventory.")
    relative_model = Path(trained_model).relative_to(bundle_root).as_posix()
    prefix = f"{relative_model}/"
    model_records = [
        record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and (record["path"] == relative_model or record["path"].startswith(prefix))
    ]
    if not model_records:
        raise RuntimeError("The verified VibeSeg inventory does not contain its trained model.")
    canonical_inventory = json.dumps(
        model_records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    folds = sorted(
        path.name.removeprefix("fold_")
        for path in Path(trained_model).glob("fold_*")
        if path.is_dir()
    )
    if not folds:
        raise RuntimeError("The verified VibeSeg trained model has no fold directories.")
    return {
        "release": VIBESEG_CROP_RELEASE,
        "trained_model_relative_path": relative_model,
        "trained_model_inventory": {
            "file_count": len(model_records),
            "total_bytes": sum(int(record["bytes"]) for record in model_records),
            "sha256": hashlib.sha256(canonical_inventory).hexdigest(),
            "asset_manifest_sha256": sha256_file(manifest_path),
        },
        "fold_selection": folds,
    }


class SpinepsRuntime:
    """Run the pinned upstream API without an implicit model lookup or download."""

    def __init__(
        self,
        model_session: SpinepsModelSession,
        vibeseg_trained_model: Path,
        *,
        derivative_name: str = "derivatives_spineps",
        make_bids_file: Callable[[Path], Any] = _make_bids_file,
        derive_output_paths: Callable[[Any, str], Mapping[str, Path]] = _derive_output_paths,
        run_vibeseg: Callable[..., Mapping[str, str] | None] = _run_explicit_vibeseg,
        run_spineps: Callable[[Any, Any, str], Any] = _run_spineps,
        vibeseg_bundle_provenance: Mapping[str, Any] | None = None,
        cache_vibeseg_model: bool = True,
    ) -> None:
        self.model_session = model_session
        self.vibeseg_trained_model = Path(vibeseg_trained_model)
        self.derivative_name = derivative_name
        self._make_bids_file = make_bids_file
        self._derive_output_paths = derive_output_paths
        self._run_vibeseg = run_vibeseg
        self._run_spineps = run_spineps
        self._vibeseg_bundle_provenance = dict(vibeseg_bundle_provenance or {})
        self.cache_vibeseg_model = bool(cache_vibeseg_model)
        self._lock = threading.Lock()

    @classmethod
    def from_model_root(
        cls,
        model_root: Path,
        *,
        use_cpu: bool = False,
        derivative_name: str = "derivatives_spineps",
        cache_vibeseg_model: bool = True,
    ) -> SpinepsRuntime:
        model_root = Path(model_root)
        trained_model = verify_installed_vibeseg(
            model_root,
            VIBESEG_CROP_ASSETS,
            full=False,
        )
        return cls(
            SpinepsModelSession(model_root, use_cpu=use_cpu),
            trained_model,
            derivative_name=derivative_name,
            cache_vibeseg_model=cache_vibeseg_model,
            vibeseg_bundle_provenance=_vibeseg_inventory_provenance(
                model_root,
                trained_model,
            ),
        )

    def set_low_memory_mode(self, enabled: bool = True) -> None:
        self.cache_vibeseg_model = not bool(enabled)
        if enabled:
            self.release_models()

    def release_models(self) -> None:
        self.model_session.unload()

    def _existing_native_outputs(self, output_paths: Mapping[str, Path]) -> dict[str, Path]:
        return {
            name: path
            for name, raw_path in output_paths.items()
            if (path := Path(raw_path)).is_file()
        }

    def _ensure_vibeseg_crop(
        self,
        img_ref: Any,
        output_path: Path,
        reference: ImageGeometry,
    ) -> Mapping[str, str] | None:
        runtime_adapter: Mapping[str, str] | None = None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.is_file():
            device = "cpu" if self.model_session.use_cpu else "cuda"
            arguments = (
                self.vibeseg_trained_model,
                img_ref.open_nii(),
                output_path,
                device,
            )
            if self._run_vibeseg is _run_explicit_vibeseg:
                runtime_adapter = self._run_vibeseg(
                    *arguments,
                    cache_model=self.cache_vibeseg_model,
                )
            else:
                # Custom runners implement the stable four-argument adapter.
                self._run_vibeseg(*arguments)
        if not output_path.is_file():
            raise RuntimeError("VibeSeg did not create the required SPINEPS crop segmentation.")
        crop_geometry = _geometry_from_sitk(sitk.ReadImage(str(output_path)))
        assert_same_physical_domain(
            reference,
            crop_geometry,
            reference_name="prepared CT",
            candidate_name="VibeSeg crop output",
        )
        return runtime_adapter

    def run(
        self,
        prepared: OrientationOutcome,
        *,
        attempt_directory: Path,
        case_id: str,
    ) -> VertebralResult:
        if not isinstance(prepared, OrientationOutcome):
            raise TypeError("SPINEPS run requires the orientation-stage outcome object.")
        _configure_upstream_citation_reminder()
        reference_geometry = _geometry_from_sitk(prepared.prepared_image)
        staged_input = _stage_prepared_ct(
            prepared.prepared_image,
            Path(attempt_directory),
            case_id,
        )
        staged_geometry = _geometry_from_sitk(sitk.ReadImage(str(staged_input)))
        assert_same_physical_domain(
            reference_geometry,
            staged_geometry,
            reference_name="prepared CT",
            candidate_name="staged SPINEPS input",
        )

        with _TPTBOX_INFERENCE_LOCK, self._lock:
            img_ref = self._make_bids_file(staged_input)
            output_paths = {
                name: Path(path)
                for name, path in self._derive_output_paths(img_ref, self.derivative_name).items()
            }
            if "out_vibeseg" not in output_paths:
                raise RuntimeError("SPINEPS did not define its required VibeSeg output path.")
            runtime_adapter = self._ensure_vibeseg_crop(
                img_ref,
                output_paths["out_vibeseg"],
                reference_geometry,
            )
            vibeseg_output = output_paths["out_vibeseg"]
            models = self.model_session.load()
            response = self._run_spineps(img_ref, models, self.derivative_name)

        provenance = dict(self.model_session.provenance)
        crop_model = dict(provenance.get("required_crop_model", {}))
        crop_model.update(self._vibeseg_bundle_provenance)
        crop_model["output"] = {
            "bytes": vibeseg_output.stat().st_size,
            "sha256": sha256_file(vibeseg_output),
        }
        if runtime_adapter is not None:
            crop_model["runtime_adapter"] = dict(runtime_adapter)
        provenance["required_crop_model"] = crop_model
        provenance.update(
            {
                "vibeseg_model_id": "dataset100",
                "vibeseg_precomputed": True,
                "input_compatibility_policy": (
                    "bodycomposition_validated_ct_with_upstream_resampling"
                ),
                "staged_input_name": staged_input.name,
                "orientation": prepared.result.to_dict(),
            }
        )
        native_outputs = self._existing_native_outputs(output_paths)
        if not isinstance(response, tuple):
            raise TypeError("SPINEPS returned an unexpected response object.")
        if len(response) == 4:
            semantic, vertebra, _centroids, error_code = response
            return adapt_spineps_sitk_outputs(
                _as_sitk_image(semantic),
                _as_sitk_image(vertebra),
                reference_geometry,
                upstream_error_code=_error_code_name(error_code),
                native_outputs=native_outputs,
                provenance=provenance,
            )
        if len(response) != 2:
            raise TypeError(f"SPINEPS returned an unexpected tuple of length {len(response)}.")

        _paths, error_code = response
        error_name = _error_code_name(error_code)
        if error_name in {"OK", "ALL_DONE"}:
            semantic_path = output_paths.get("out_spine")
            vertebra_path = output_paths.get("out_vert")
            if semantic_path is None or vertebra_path is None:
                raise RuntimeError("SPINEPS succeeded without defining its saved outputs.")
            if not semantic_path.is_file() or not vertebra_path.is_file():
                raise RuntimeError("SPINEPS succeeded without creating its saved label outputs.")
            return adapt_spineps_sitk_outputs(
                sitk.ReadImage(str(semantic_path)),
                sitk.ReadImage(str(vertebra_path)),
                reference_geometry,
                upstream_error_code=error_name,
                native_outputs=native_outputs,
                provenance=provenance,
            )
        return adapt_spineps_outputs(
            None,
            None,
            reference_geometry,
            upstream_error_code=error_name,
            native_outputs=native_outputs,
            provenance=provenance,
        )
