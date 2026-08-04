"""Pinned, cache-only synchronization for BOA Task 542 model assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from BodyComposition.tissue_backends.boa import BOA_BACKEND_ID

UPSTREAM_PROJECT = "BOA: Body and Organ Analysis"
UPSTREAM_REPOSITORY = "https://github.com/UMEssen/Body-and-Organ-Analysis"
UPSTREAM_VERSION = "1.0.2"
UPSTREAM_COMMIT = "6e761702a738ba681e6f7a3a7c944b00b186c3a8"
UPSTREAM_WEIGHT_RELEASE = "v1.0.0-weights"
UPSTREAM_WEIGHT_RELEASE_COMMIT = "e80a6d79466f64edb35373ff7647b40d3d17824d"
CODE_LICENSE = "Apache-2.0"
WEIGHT_LICENSE = "Apache-2.0"
CITATION_DOI = "https://doi.org/10.1097/RLI.0000000000001040"
REDISTRIBUTION_MODE = "user_model_sync"
DATASET_DIRECTORY = "Dataset542_BCA_inference"
TRAINER_DIRECTORY = "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
ARCHIVE_NAME = f"{DATASET_DIRECTORY}.zip"
DOWNLOAD_URL = (
    f"{UPSTREAM_REPOSITORY}/releases/download/{UPSTREAM_WEIGHT_RELEASE}/{ARCHIVE_NAME}"
)
ARCHIVE_BYTES = 1_150_464_739
ARCHIVE_SHA256 = "096d6cdb45524273026d30e3bd290e8fd10990dec02f0aa0669d214b71bf6d4f"
ASSET_MANIFEST_NAME = ".bodycomposition-asset.json"
COPY_CHUNK_BYTES = 1024 * 1024
MAX_UNCOMPRESSED_ARCHIVE_BYTES = 4 * 1024**3

EXPECTED_FILES: Mapping[str, Mapping[str, int | str]] = {
    "dataset.json": {
        "byte_size": 517,
        "sha256": "54d5af1502d43f5b8d04c8e452b9624e24e16cbfd3f9c10082016ae703363af3",
    },
    "plans.json": {
        "byte_size": 11_153,
        "sha256": "1924845f54187b98497dd6140735c80733de0b18a2cdd9e88d2eb390b817481f",
    },
    "fold_0/checkpoint_final.pth": {
        "byte_size": 247_185_150,
        "sha256": "be2a31402cc9228e830ac84161ff22654bfa29e7706bd31374a8ba104baea5a0",
    },
    "fold_1/checkpoint_final.pth": {
        "byte_size": 247_185_598,
        "sha256": "ea9c0b590bb247d91263c2bed5dfbda4d079d3099fcf320d183c4f3ff96b7d9e",
    },
    "fold_2/checkpoint_final.pth": {
        "byte_size": 247_183_678,
        "sha256": "d616c76c75715de4a8916f6e29fa9e32412b0ed77309b21d1694e0b47e33f0e1",
    },
    "fold_3/checkpoint_final.pth": {
        "byte_size": 247_186_430,
        "sha256": "7dc5a77304a515e837d14e5f46d41da5a648849504e9aa86ce9ec4416bf670b7",
    },
    "fold_4/checkpoint_final.pth": {
        "byte_size": 247_185_086,
        "sha256": "d7473f09947efe7c3c854bfb1401f06602e1930587cdc1d502785fc01469db74",
    },
}


class BoaAssetError(RuntimeError):
    """Raised when the pinned BOA asset cannot be verified or installed."""


class UnsafeBoaArchiveError(BoaAssetError):
    """Raised when an archive member is unsafe to extract."""


@dataclass(frozen=True)
class BoaAssetReport:
    model_directory: Path
    ready: bool
    checked_files: Mapping[str, str]
    errors: tuple[str, ...]
    install_manifest_verified: bool


def model_asset_record() -> dict:
    """Return path-free BOA model and distribution provenance."""

    return {
        "asset_id": BOA_BACKEND_ID,
        "name": "BOA BCA body regions (Task 542)",
        "upstream_project": UPSTREAM_PROJECT,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_version": UPSTREAM_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_release": UPSTREAM_WEIGHT_RELEASE,
        "upstream_release_commit": UPSTREAM_WEIGHT_RELEASE_COMMIT,
        "citation_doi": CITATION_DOI,
        "task_id": 542,
        "download_url": DOWNLOAD_URL,
        "archive_name": ARCHIVE_NAME,
        "archive_byte_size": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "expected_files": [
            {"relative_path": relative, **dict(expected)}
            for relative, expected in EXPECTED_FILES.items()
        ],
        "code_license": CODE_LICENSE,
        "weight_license": WEIGHT_LICENSE,
        "weight_license_status": (
            "official_archive_dataset_metadata_and_repository_declare_Apache-2.0"
        ),
        "redistribution_mode": REDISTRIBUTION_MODE,
        "compatibility": {
            "BodyComposition": ">=1.0.0rc1",
            "nnUNetv2": "==2.5.2",
            "storage": "nnUNet_results_dataset_directory",
        },
        "validation_set_version": "boa-task542-adapter-validation-v1",
    }


def boa_model_directory(weights_root: str | Path) -> Path:
    return Path(weights_root).expanduser().resolve() / DATASET_DIRECTORY / TRAINER_DIRECTORY


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_manifest_status(model_directory: Path) -> tuple[bool, str | None]:
    path = model_directory / ASSET_MANIFEST_NAME
    if not path.exists() and not path.is_symlink():
        return False, "missing_install_manifest"
    if not path.is_file():
        return False, "invalid_install_manifest"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "invalid_install_manifest"
    if payload != {"schema_version": 1, "asset": model_asset_record()}:
        return False, "install_manifest_mismatch"
    return True, None


def _write_install_manifest(model_directory: Path) -> None:
    descriptor = {"schema_version": 1, "asset": model_asset_record()}
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{ASSET_MANIFEST_NAME}.",
        suffix=".partial",
        dir=model_directory,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(descriptor, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, model_directory / ASSET_MANIFEST_NAME)
    finally:
        temporary.unlink(missing_ok=True)


def check_boa_model(weights_root: str | Path) -> BoaAssetReport:
    model_directory = boa_model_directory(weights_root)
    checked: dict[str, str] = {}
    errors: list[str] = []
    for relative, expected in EXPECTED_FILES.items():
        path = model_directory / relative
        if not path.is_file():
            errors.append(f"missing:{relative}")
            continue
        if path.stat().st_size != expected["byte_size"]:
            errors.append(f"size_mismatch:{relative}")
            continue
        observed = _sha256(path)
        checked[relative] = observed
        if observed != expected["sha256"]:
            errors.append(f"sha256_mismatch:{relative}")
    manifest_verified, manifest_error = _install_manifest_status(model_directory)
    if manifest_error is not None:
        errors.append(manifest_error)
    return BoaAssetReport(
        model_directory=model_directory,
        ready=not errors,
        checked_files=checked,
        errors=tuple(errors),
        install_manifest_verified=manifest_verified,
    )


def require_boa_model(weights_root: str | Path) -> BoaAssetReport:
    report = check_boa_model(weights_root)
    if report.ready:
        return report
    details = ", ".join(report.errors)
    raise FileNotFoundError(
        "Pinned BOA Task 542 model is not ready in the mounted model directory "
        f"({details}). Run `bodycomposition models sync --tissue-backend boa` "
        "before inference."
    )


def _copy_download(source: BinaryIO, destination: Path) -> None:
    written = 0
    with destination.open("wb") as output:
        while chunk := source.read(COPY_CHUNK_BYTES):
            output.write(chunk)
            written += len(chunk)
            if written > ARCHIVE_BYTES:
                raise BoaAssetError("Downloaded BOA archive exceeds its pinned size.")
        output.flush()
        os.fsync(output.fileno())


def _verify_archive(path: Path) -> None:
    if not path.is_file() or path.stat().st_size != ARCHIVE_BYTES:
        observed_size = path.stat().st_size if path.is_file() else None
        raise BoaAssetError(
            f"BOA archive size mismatch: expected {ARCHIVE_BYTES}, observed {observed_size}."
        )
    observed_sha256 = _sha256(path)
    if observed_sha256 != ARCHIVE_SHA256:
        raise BoaAssetError(
            "BOA archive SHA-256 mismatch: expected "
            f"{ARCHIVE_SHA256}, observed {observed_sha256}."
        )


def _member_target(root: Path, member: zipfile.ZipInfo) -> Path:
    if "\\" in member.filename:
        raise UnsafeBoaArchiveError(
            f"Backslash-separated archive path is not allowed: {member.filename!r}."
        )
    relative = PurePosixPath(member.filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise UnsafeBoaArchiveError(f"Unsafe path in BOA archive: {member.filename!r}.")
    file_type = (member.external_attr >> 16) & 0o170000
    if stat.S_ISLNK(file_type):
        raise UnsafeBoaArchiveError(
            f"Symbolic links are not allowed in BOA archives: {member.filename!r}."
        )
    if member.flag_bits & 0x1:
        raise UnsafeBoaArchiveError(
            f"Encrypted BOA archive member is not supported: {member.filename!r}."
        )
    return root.joinpath(*relative.parts)


def _archive_junk(member: zipfile.ZipInfo) -> bool:
    parts = PurePosixPath(member.filename).parts
    return (
        "__MACOSX" in parts
        or any(part == ".DS_Store" or part.startswith("._") for part in parts)
    )


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        expanded_bytes = sum(member.file_size for member in members)
        if expanded_bytes > MAX_UNCOMPRESSED_ARCHIVE_BYTES:
            raise UnsafeBoaArchiveError(
                f"BOA archive expands to {expanded_bytes} bytes, above the safety limit."
            )
        for member in members:
            target = _member_target(destination, member)
            if _archive_junk(member):
                continue
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise UnsafeBoaArchiveError(
                    f"Duplicate BOA archive member: {member.filename!r}."
                )
            with source.open(member) as input_handle, target.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, COPY_CHUNK_BYTES)


def sync_boa_model(
    weights_root: str | Path,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> BoaAssetReport:
    """Install the exact official archive atomically and idempotently."""

    existing = check_boa_model(weights_root)
    if existing.ready:
        return existing
    metadata_errors = {
        "missing_install_manifest",
        "install_manifest_mismatch",
        "invalid_install_manifest",
    }
    if existing.errors and set(existing.errors) <= metadata_errors:
        _write_install_manifest(existing.model_directory)
        repaired = check_boa_model(weights_root)
        if repaired.ready:
            return repaired

    root = Path(weights_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / DATASET_DIRECTORY
    with tempfile.TemporaryDirectory(
        prefix=f".{DATASET_DIRECTORY}-sync-",
        dir=root,
    ) as temporary_name:
        temporary = Path(temporary_name)
        archive = temporary / ARCHIVE_NAME
        extracted = temporary / "extracted"
        extracted.mkdir()
        try:
            request = urllib.request.Request(
                DOWNLOAD_URL,
                headers={"User-Agent": "BodyComposition-model-sync/1"},
            )
            with opener(request) as response:
                _copy_download(response, archive)
            _verify_archive(archive)
            _safe_extract(archive, extracted)
        except BoaAssetError:
            raise
        except Exception as error:
            raise BoaAssetError(
                f"Unable to synchronize BOA Task 542 from its official release: {error}"
            ) from error

        staged = extracted / DATASET_DIRECTORY
        if not staged.is_dir():
            raise BoaAssetError(
                f"BOA archive does not contain {DATASET_DIRECTORY!r}."
            )
        staged_report = check_boa_model(extracted)
        if set(staged_report.errors) != {"missing_install_manifest"}:
            raise BoaAssetError(
                "Extracted BOA model failed pinned checks: "
                + ", ".join(staged_report.errors)
            )
        _write_install_manifest(staged / TRAINER_DIRECTORY)

        previous = temporary / "previous"
        promoted = False
        try:
            if destination.exists() or destination.is_symlink():
                os.replace(destination, previous)
            os.replace(staged, destination)
            promoted = True
            installed = check_boa_model(root)
            if not installed.ready or not installed.install_manifest_verified:
                raise BoaAssetError("Promoted BOA model failed final verification.")
        except Exception:
            if promoted and (destination.exists() or destination.is_symlink()):
                os.replace(destination, temporary / "failed-promotion")
            if previous.exists() or previous.is_symlink():
                os.replace(previous, destination)
            raise
    return check_boa_model(root)
