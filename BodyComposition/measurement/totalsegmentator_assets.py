"""Pinned, checksum-verified TotalSegmentator assets used by measurement stage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Mapping


UPSTREAM_PROJECT = "TotalSegmentator"
UPSTREAM_REPOSITORY = "https://github.com/wasserth/TotalSegmentator"
UPSTREAM_RELEASE = "v2.0.0-weights"
UPSTREAM_RELEASE_COMMIT = "d0eb302278588eade4dcc33093d141fe57a3ca90"
UPSTREAM_VERSION = "2.15.0"
CITATION_DOI = "https://doi.org/10.1148/ryai.230024"
CODE_LICENSE = "Apache-2.0"
WEIGHT_LICENSE_STATUS = (
    "upstream_lists_total_and_body_as_open_for_any_usage_under_Apache-2.0"
)
REDISTRIBUTION_MODE = "user_model_sync"
VALIDATION_SET_VERSION = "measurement-technical-contour-validation-pending"
ASSET_MANIFEST_NAME = ".bodycomposition-asset.json"
COPY_CHUNK_BYTES = 1024 * 1024
MAX_UNCOMPRESSED_ARCHIVE_BYTES = 4 * 1024**3


MEASUREMENT_MODEL_MANIFEST: Mapping[str, dict] = {
    "bodytrunk": {
        "asset_id": "totalsegmentator-body-task299-v2.0.0-weights",
        "task_id": 299,
        "model_title": "TotalSegmentator-body",
        "directory": "Dataset299_body_1559subj",
        "archive_name": "Dataset299_body_1559subj.zip",
        "download_url": (
            "https://github.com/wasserth/TotalSegmentator/releases/download/"
            "v2.0.0-weights/Dataset299_body_1559subj.zip"
        ),
        "archive_bytes": 233_211_222,
        "archive_sha256": (
            "d51d68e7f183fed966c122b0cb3dece7bebcdb685d28aeecc62202168e431674"
        ),
        "files": {
            "nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json": {
                "bytes": 406,
                "sha256": (
                    "e0b704e61fe7d3a38dadff40d3b1712633a965429a10b313efbe705b3b769510"
                ),
            },
            "nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json": {
                "bytes": 11_157,
                "sha256": (
                    "dea1187d1a9b96859b06b828e3cffc74d41fae5fd4c3e2227e617a3579c813c2"
                ),
            },
            "nnUNetTrainer__nnUNetPlans__3d_fullres/dataset_fingerprint.json": {
                "bytes": 234_499,
                "sha256": (
                    "38501be1659cc1698bf9929e62d7455e166b4eb9dcd227c0ba731eb27c8a80d1"
                ),
            },
            "nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth": {
                "bytes": 250_110_587,
                "sha256": (
                    "a7383a312864522a8796e677569c73fdf73adca5ddc2b08613596db9c96d794d"
                ),
            },
        },
    },
    "body_landmarks": {
        "asset_id": "totalsegmentator-total-fast-task297-v2.0.0-weights",
        "task_id": 297,
        "model_title": "TotalSegmentator-total-fast",
        "directory": "Dataset297_TotalSegmentator_total_3mm_1559subj",
        "archive_name": "Dataset297_TotalSegmentator_total_3mm_1559subj.zip",
        "download_url": (
            "https://github.com/wasserth/TotalSegmentator/releases/download/"
            "v2.0.0-weights/Dataset297_TotalSegmentator_total_3mm_1559subj.zip"
        ),
        "archive_bytes": 135_386_075,
        "archive_sha256": (
            "0baa2c8de2975600eb31801dd5c1825cd2b356f794498659cf3348714c073394"
        ),
        "files": {
            "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/dataset.json": {
                "bytes": 3_796,
                "sha256": (
                    "e804e08ad8912e3a6f541741c8c78799d75acbd3f4f3f18ff5f211bbd3da961a"
                ),
            },
            "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/plans.json": {
                "bytes": 7_110,
                "sha256": (
                    "d1cb3c15f53dc36fdb618e0f9082573d1f50b74a9b5fa73094ac8680845839f1"
                ),
            },
            "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/dataset_fingerprint.json": {
                "bytes": 232_834,
                "sha256": (
                    "f1336cd0df44e0eeecfa69143226c05b246c550e690b7b12929f65ca9843260f"
                ),
            },
            "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth": {
                "bytes": 164_939_235,
                "sha256": (
                    "1e38e40356adc2706a662e365405a97f862d62a7da65f6bf81d025aee1b979ac"
                ),
            },
        },
    },
}


class TotalSegmentatorAssetError(RuntimeError):
    """Raised when a pinned measurement asset cannot be synchronized safely."""


class UnsafeModelArchiveError(TotalSegmentatorAssetError):
    """Raised when an upstream archive contains an unsafe member."""


@dataclass(frozen=True)
class TotalSegmentatorAssetReport:
    task: str
    task_id: int
    model_title: str
    model_directory: Path
    ready: bool
    checked_files: Mapping[str, str]
    errors: tuple[str, ...]
    install_manifest_verified: bool

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "task_id": self.task_id,
            "model_title": self.model_title,
            "model_directory": str(self.model_directory),
            "ready": self.ready,
            "checked_files": dict(self.checked_files),
            "errors": list(self.errors),
            "install_manifest_verified": self.install_manifest_verified,
            "asset": model_asset_record(self.task),
        }

    def provenance(self) -> dict:
        """Return path-free, JSON-safe provenance for canonical outputs."""

        return {
            **model_asset_record(self.task),
            "task": self.task,
            "asset_sha256": dict(self.checked_files),
            "integrity_verified": self.ready,
            "install_manifest_verified": self.install_manifest_verified,
        }


def _expected_file_record(value: object) -> tuple[int | None, str]:
    if isinstance(value, str):
        return None, value
    if not isinstance(value, Mapping):
        raise TypeError("Expected-file manifest entries must be digests or mappings.")
    expected_bytes = value.get("bytes")
    expected_sha256 = value.get("sha256")
    if expected_bytes is not None and not isinstance(expected_bytes, int):
        raise TypeError("Expected-file byte sizes must be integers.")
    if not isinstance(expected_sha256, str):
        raise TypeError("Expected-file SHA-256 values must be strings.")
    return expected_bytes, expected_sha256


def model_asset_record(task: str) -> dict:
    """Return the complete, path-free asset and distribution contract."""

    if task not in MEASUREMENT_MODEL_MANIFEST:
        raise ValueError(f"No measurement stage TotalSegmentator manifest exists for task {task!r}.")
    manifest = MEASUREMENT_MODEL_MANIFEST[task]
    expected_files = []
    for relative, expected in manifest["files"].items():
        expected_bytes, expected_sha256 = _expected_file_record(expected)
        record: dict[str, object] = {
            "relative_path": relative,
            "sha256": expected_sha256,
        }
        if expected_bytes is not None:
            record["byte_size"] = expected_bytes
        expected_files.append(record)
    return {
        "asset_id": manifest.get(
            "asset_id",
            f"totalsegmentator-task{manifest['task_id']}-{UPSTREAM_RELEASE}",
        ),
        "name": manifest["model_title"],
        "upstream_project": UPSTREAM_PROJECT,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_version": UPSTREAM_VERSION,
        "upstream_release": UPSTREAM_RELEASE,
        "upstream_release_commit": UPSTREAM_RELEASE_COMMIT,
        "citation_doi": CITATION_DOI,
        "task_id": int(manifest["task_id"]),
        "download_url": manifest.get("download_url"),
        "archive_name": manifest.get("archive_name"),
        "archive_byte_size": manifest.get("archive_bytes"),
        "archive_sha256": manifest.get("archive_sha256"),
        "expected_files": expected_files,
        "code_license": CODE_LICENSE,
        "weight_license_status": WEIGHT_LICENSE_STATUS,
        "redistribution_mode": REDISTRIBUTION_MODE,
        "compatibility": {
            "BodyComposition": ">=0.3.0",
            "TotalSegmentator": f"=={UPSTREAM_VERSION}",
            "storage": "nnUNet_results_dataset_directory",
        },
        "validation_set_version": VALIDATION_SET_VERSION,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_manifest_status(task: str, model_directory: Path) -> tuple[bool, str | None]:
    path = model_directory / ASSET_MANIFEST_NAME
    if not path.exists() and not path.is_symlink():
        # Exact pre-existing upstream installations remain supported. Their
        # pinned inference files are still verified below.
        return False, None
    if not path.is_file():
        return False, "invalid_install_manifest"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "invalid_install_manifest"
    expected = {"schema_version": 1, "asset": model_asset_record(task)}
    if payload != expected:
        return False, "install_manifest_mismatch"
    return True, None


@lru_cache(maxsize=16)
def _check_cached(
    task: str,
    root: str,
    fingerprints: tuple[tuple[str, int, int, int], ...],
) -> TotalSegmentatorAssetReport:
    del fingerprints
    manifest = MEASUREMENT_MODEL_MANIFEST[task]
    model_directory = Path(root) / manifest["directory"]
    checked: dict[str, str] = {}
    errors: list[str] = []
    for relative, expected in manifest["files"].items():
        expected_bytes, expected_sha256 = _expected_file_record(expected)
        path = model_directory / relative
        if not path.is_file():
            errors.append(f"missing:{relative}")
            continue
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            errors.append(f"size_mismatch:{relative}")
            continue
        observed = _sha256(path)
        checked[relative] = observed
        if observed != expected_sha256:
            errors.append(f"sha256_mismatch:{relative}")
    manifest_verified, manifest_error = _install_manifest_status(task, model_directory)
    if manifest_error is not None:
        errors.append(manifest_error)
    return TotalSegmentatorAssetReport(
        task=task,
        task_id=int(manifest["task_id"]),
        model_title=str(manifest["model_title"]),
        model_directory=model_directory,
        ready=not errors,
        checked_files=checked,
        errors=tuple(errors),
        install_manifest_verified=manifest_verified,
    )


def check_measurement_model(
    task: str,
    weights_root: str | Path | None = None,
) -> TotalSegmentatorAssetReport:
    if task not in MEASUREMENT_MODEL_MANIFEST:
        raise ValueError(f"No measurement stage TotalSegmentator manifest exists for task {task!r}.")
    if weights_root is None:
        from totalsegmentator.config import get_weights_dir

        weights_root = get_weights_dir()
    root = Path(weights_root).expanduser().resolve()
    manifest = MEASUREMENT_MODEL_MANIFEST[task]
    model_directory = root / manifest["directory"]
    fingerprints: list[tuple[str, int, int, int]] = []
    tracked_paths = [*manifest["files"], ASSET_MANIFEST_NAME]
    for relative in tracked_paths:
        path = model_directory / relative
        if path.is_file():
            file_stat = path.stat()
            fingerprints.append(
                (relative, file_stat.st_size, file_stat.st_mtime_ns, file_stat.st_ctime_ns)
            )
        else:
            fingerprints.append((relative, -1, -1, -1))
    return _check_cached(task, str(root), tuple(fingerprints))


def require_measurement_model(
    task: str,
    weights_root: str | Path | None = None,
) -> TotalSegmentatorAssetReport:
    report = check_measurement_model(task, weights_root)
    if report.ready:
        return report
    command = f"bodycomposition_download_models --model {report.model_title}"
    details = ", ".join(report.errors)
    raise FileNotFoundError(
        f"Pinned TotalSegmentator task {report.task_id} is not ready in the mounted model "
        f"directory ({details}). Run `{command}` before inference."
    )


def _verify_archive(path: Path, manifest: Mapping[str, object]) -> None:
    expected_bytes = manifest.get("archive_bytes")
    expected_sha256 = manifest.get("archive_sha256")
    if not isinstance(expected_bytes, int) or not isinstance(expected_sha256, str):
        raise TotalSegmentatorAssetError("The model archive has no complete size/hash pin.")
    if not path.is_file() or path.stat().st_size != expected_bytes:
        observed = path.stat().st_size if path.is_file() else None
        raise TotalSegmentatorAssetError(
            f"Model archive size mismatch: expected {expected_bytes}, observed {observed}."
        )
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise TotalSegmentatorAssetError(
            "Model archive SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}."
        )


def _validated_member_path(root: Path, member: zipfile.ZipInfo) -> Path:
    if "\\" in member.filename:
        raise UnsafeModelArchiveError(
            f"Backslash-separated model archive path is not allowed: {member.filename!r}."
        )
    relative = PurePosixPath(member.filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise UnsafeModelArchiveError(
            f"Unsafe path in model archive: {member.filename!r}."
        )
    file_type = (member.external_attr >> 16) & 0o170000
    if stat.S_ISLNK(file_type):
        raise UnsafeModelArchiveError(
            f"Symbolic links are not allowed in model archives: {member.filename!r}."
        )
    if member.flag_bits & 0x1:
        raise UnsafeModelArchiveError(
            f"Encrypted model archive member is not supported: {member.filename!r}."
        )
    return root.joinpath(*relative.parts)


def _is_archive_junk(member: zipfile.ZipInfo) -> bool:
    parts = PurePosixPath(member.filename).parts
    return (
        "__MACOSX" in parts
        or any(part == ".DS_Store" or part.startswith("._") for part in parts)
    )


def _safe_extract_archive(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        total_uncompressed = sum(member.file_size for member in members)
        if total_uncompressed > MAX_UNCOMPRESSED_ARCHIVE_BYTES:
            raise UnsafeModelArchiveError(
                f"Model archive expands to {total_uncompressed} bytes, above the safety limit."
            )
        for member in members:
            target = _validated_member_path(destination, member)
            if _is_archive_junk(member):
                continue
            if member.is_dir():
                if target.exists() and not target.is_dir():
                    raise UnsafeModelArchiveError(
                        f"Archive directory conflicts with a file: {member.filename!r}."
                    )
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise UnsafeModelArchiveError(
                    f"Duplicate model archive member: {member.filename!r}."
                )
            with source.open(member) as input_handle, target.open("wb") as output_handle:
                shutil.copyfileobj(
                    input_handle,
                    output_handle,
                    length=COPY_CHUNK_BYTES,
                )


def _copy_download(source: BinaryIO, destination: Path, expected_bytes: int) -> None:
    written = 0
    with destination.open("wb") as output:
        while True:
            chunk = source.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)
            written += len(chunk)
            if written > expected_bytes:
                raise TotalSegmentatorAssetError(
                    "Downloaded model archive is larger than its pinned size."
                )
        output.flush()
        os.fsync(output.fileno())


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def sync_measurement_model(
    task: str,
    weights_root: str | Path | None = None,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> TotalSegmentatorAssetReport:
    """Synchronize one exact upstream archive and promote it after verification."""

    if task not in MEASUREMENT_MODEL_MANIFEST:
        raise ValueError(f"No measurement stage TotalSegmentator manifest exists for task {task!r}.")
    existing = check_measurement_model(task, weights_root)
    if existing.ready:
        return existing

    if weights_root is None:
        from totalsegmentator.config import get_weights_dir

        weights_root = get_weights_dir()
    root = Path(weights_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = MEASUREMENT_MODEL_MANIFEST[task]
    archive_name = manifest.get("archive_name")
    download_url = manifest.get("download_url")
    expected_bytes = manifest.get("archive_bytes")
    if (
        not isinstance(archive_name, str)
        or not isinstance(download_url, str)
        or not isinstance(expected_bytes, int)
    ):
        raise TotalSegmentatorAssetError("The model manifest has no complete download pin.")

    destination = root / manifest["directory"]
    with tempfile.TemporaryDirectory(
        prefix=f".{manifest['directory']}-sync-",
        dir=root,
    ) as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / archive_name
        extracted = temporary_path / "extracted"
        extracted.mkdir()
        try:
            request = urllib.request.Request(
                download_url,
                headers={"User-Agent": "BodyComposition-model-sync/1"},
            )
            with opener(request) as response:
                _copy_download(response, archive, expected_bytes)
            _verify_archive(archive, manifest)
            _safe_extract_archive(archive, extracted)
        except TotalSegmentatorAssetError:
            raise
        except Exception as error:
            raise TotalSegmentatorAssetError(
                f"Unable to synchronize {manifest['model_title']} from {download_url}: {error}"
            ) from error

        staged = extracted / manifest["directory"]
        if not staged.is_dir():
            raise TotalSegmentatorAssetError(
                f"Archive does not contain the expected directory {manifest['directory']!r}."
            )
        staged_report = check_measurement_model(task, extracted)
        if not staged_report.ready:
            raise TotalSegmentatorAssetError(
                "Extracted model failed its pinned file checks: "
                + ", ".join(staged_report.errors)
            )
        (staged / ASSET_MANIFEST_NAME).write_text(
            json.dumps(
                {"schema_version": 1, "asset": model_asset_record(task)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        previous = temporary_path / "previous"
        promoted = False
        try:
            if _path_exists(destination):
                os.replace(destination, previous)
            os.replace(staged, destination)
            promoted = True
            _check_cached.cache_clear()
            installed = check_measurement_model(task, root)
            if not installed.ready or not installed.install_manifest_verified:
                raise TotalSegmentatorAssetError(
                    "Promoted model failed its final installed-asset verification."
                )
        except Exception:
            if promoted and _path_exists(destination):
                os.replace(destination, temporary_path / "failed-promotion")
            if _path_exists(previous):
                os.replace(previous, destination)
            _check_cached.cache_clear()
            raise

    _check_cached.cache_clear()
    return check_measurement_model(task, root)
