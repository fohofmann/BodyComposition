"""Deterministic analysis identity and privacy-safe runtime provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from BodyComposition.config import PipelineConfig
from BodyComposition.model_manager import ModelStatus, model_bundle_digest
from BodyComposition.utils.geometry import ImageGeometry


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_array_sha256(array: np.ndarray) -> str:
    """Hash an image array with its dtype and explicit axis shape."""

    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.data)
    return digest.hexdigest()


def image_pixel_sha256(image: sitk.Image) -> str:
    return pixel_array_sha256(sitk.GetArrayViewFromImage(image))


def image_summary(
    path: str | Path,
    *,
    series_uid: str | None = None,
) -> tuple[sitk.Image, dict[str, Any]]:
    """Read one NIfTI CT or one unambiguous DICOM CT series."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input CT not found: {source}.")
    suffix = ".nii.gz" if source.name.lower().endswith(".nii.gz") else source.suffix.lower()
    if not (source.is_file() and suffix in {".nii", ".nii.gz"}):
        from BodyComposition.dicom import dicom_image_summary

        image, summary, _ = dicom_image_summary(source, series_uid=series_uid)
        return image, summary

    if series_uid is not None:
        raise ValueError("A DICOM Series Instance UID cannot be used with a NIfTI input.")
    image = sitk.ReadImage(str(source))
    geometry = ImageGeometry.from_sitk(image)
    geometry.validate()
    pixels = sitk.GetArrayViewFromImage(image)
    if not np.all(np.isfinite(pixels)):
        raise ValueError("Input CT contains non-finite pixels.")
    geometry_summary = {
        "size_xyz": list(geometry.size_xyz),
        "spacing_xyz": list(geometry.spacing_xyz),
        "origin_lps_xyz": list(geometry.origin_lps_xyz),
        "direction_lps": list(geometry.direction_lps),
    }
    summary = {
        "source_reference": "content-addressed-local-input",
        "source_content_sha256": file_sha256(source),
        "source_byte_size": source.stat().st_size,
        "input_pixel_sha256": image_pixel_sha256(image),
        "input_format": "nifti",
        "geometry": geometry_summary,
    }
    from BodyComposition.dicom import enrich_nifti_summary_from_conversion_metadata

    return image, enrich_nifti_summary_from_conversion_metadata(source, summary)


def _run_git(root: Path, *arguments: str) -> bytes | None:
    try:
        process = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return process.stdout


def _package_version() -> str:
    try:
        return importlib.metadata.version("BodyComposition")
    except importlib.metadata.PackageNotFoundError:
        from BodyComposition import __version__

        return __version__


def source_state() -> dict[str, Any]:
    """Return source identity without exposing local repository paths."""

    root = Path(__file__).resolve().parents[1]
    commit_raw = _run_git(root, "rev-parse", "HEAD")
    status_raw = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    commit = (
        commit_raw.decode("ascii").strip()
        if commit_raw
        else os.environ.get("BODYCOMPOSITION_GIT_SHA")
    )
    if commit in {"", "unknown"}:
        commit = None
    dirty = bool(status_raw and status_raw.strip())
    tree_digest = os.environ.get("BODYCOMPOSITION_SOURCE_SHA256")
    if tree_digest in {"", "unknown"}:
        tree_digest = None
    if status_raw is not None:
        digest = hashlib.sha256(status_raw)
        diff = _run_git(root, "diff", "--binary", "HEAD", "--", "BodyComposition", "pyproject.toml", "uv.lock")
        if diff:
            digest.update(diff)
        for line in status_raw.decode("utf-8", errors="replace").splitlines():
            if not line.startswith("?? "):
                continue
            relative = line[3:]
            candidate = root / relative
            if candidate.is_file() and (
                relative.startswith("BodyComposition/")
                or relative in {"pyproject.toml", "uv.lock"}
            ):
                digest.update(relative.encode("utf-8"))
                digest.update(candidate.read_bytes())
        tree_digest = digest.hexdigest()
    return {
        "package_version": _package_version(),
        "git_commit": commit,
        "source_dirty": dirty,
        "source_tree_sha256": tree_digest,
        "container_image_digest": os.environ.get("BODYCOMPOSITION_IMAGE_DIGEST"),
        "environment_path_variables": [
            name
            for name in (
                "BODYCOMPOSITION_MODEL_ROOT",
                "BODYCOMPOSITION_OUTPUT_ROOT",
            )
            if name in os.environ
        ],
    }


def package_lock_digest() -> str | None:
    root = Path(__file__).resolve().parents[1]
    lock = root / "uv.lock"
    if lock.is_file():
        return file_sha256(lock)
    embedded = os.environ.get("BODYCOMPOSITION_UV_LOCK_SHA256", "").lower()
    if len(embedded) == 64 and all(character in "0123456789abcdef" for character in embedded):
        return embedded
    return None


def runtime_provenance(*, device: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "host_class": f"{platform.system().lower()}-{platform.machine().lower()}",
        "python_version": platform.python_version(),
        "requested_device": device,
    }
    try:
        import torch

        if device in {"auto", "cuda"} and torch.cuda.is_available():
            device_type = "cuda"
            accelerator_name = torch.cuda.get_device_name()
        else:
            device_type = "cpu"
            accelerator_name = None
        value.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device_type": device_type,
                "gpu_name": accelerator_name,
            }
        )
    except ImportError:
        value.update({"torch_version": None, "cuda_version": None, "device_type": "unavailable", "gpu_name": None})
    return value


def manifest_configuration(config: PipelineConfig) -> dict[str, Any]:
    """Return resolved configuration without serializing private cache paths."""

    value = config.normalized()
    value["models"]["root"] = "<mounted-model-cache>"
    return value


def analysis_identity(
    *,
    input_summary: Mapping[str, Any],
    config: PipelineConfig,
    source: Mapping[str, Any],
    models: Sequence[ModelStatus],
) -> tuple[str, dict[str, Any]]:
    model_digest = model_bundle_digest(models)
    payload = {
        "input_pixel_sha256": input_summary["input_pixel_sha256"],
        "input_geometry_sha256": canonical_digest(input_summary["geometry"]),
        "scientific_configuration_sha256": config.scientific_digest(),
        "code": {
            "package_version": source.get("package_version"),
            "git_commit": source.get("git_commit"),
            "source_dirty": source.get("source_dirty"),
            "source_tree_sha256": source.get("source_tree_sha256"),
        },
        "models_sha256": model_digest,
    }
    return canonical_digest(payload), {
        "identity_payload": payload,
        "configuration": manifest_configuration(config),
        "configuration_sha256": config.digest(),
        "queue_configuration_sha256": config.queue_digest(),
        "scientific_configuration_sha256": config.scientific_digest(),
        "package_lock_sha256": package_lock_digest(),
        "models": [model.provenance() for model in models],
        "models_sha256": model_digest,
        "code": dict(source),
        "runtime": runtime_provenance(device=config.device),
    }
