"""Checksum-verified, atomic installation of pinned SPINEPS model archives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from BodyComposition.vertebral.spineps_manifest import (
    MODEL_BUNDLE_VERSION,
    SPINEPS_MODEL_ASSETS,
    VIBESEG_CROP_ASSETS,
    ModelAssetSpec,
    ReleaseAssetPin,
)

ASSET_MANIFEST_NAME = ".bodycomposition-asset.json"
VIBESEG_INSTALL_DIR = "vibeseg_crop_dataset100"
MAX_UNCOMPRESSED_ARCHIVE_BYTES = 8 * 1024**3
COPY_CHUNK_BYTES = 1024**2


class AssetVerificationError(RuntimeError):
    """Raised when a downloaded or installed model does not match its pin."""


class UnsafeArchiveError(AssetVerificationError):
    """Raised when a model archive contains an unsafe member."""


@dataclass(frozen=True)
class ModelSyncReport:
    """Verified local locations for every asset in the pinned bundle."""

    model_root: Path
    spineps_models: Mapping[str, Path]
    vibeseg_trained_model: Path
    downloaded_assets: tuple[str, ...]

    @property
    def bundle_id(self) -> str:
        return MODEL_BUNDLE_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, spec: ModelAssetSpec) -> None:
    if not path.is_file():
        raise AssetVerificationError(f"Model archive not found: {path}.")
    observed_size = path.stat().st_size
    if observed_size != spec.bytes:
        raise AssetVerificationError(
            f"{spec.model_id} archive size mismatch: expected {spec.bytes}, got {observed_size}."
        )
    observed_digest = sha256_file(path)
    if observed_digest != spec.sha256:
        raise AssetVerificationError(
            f"{spec.model_id} archive digest mismatch: expected {spec.sha256}, got {observed_digest}."
        )


def _validated_member_path(root: Path, member: zipfile.ZipInfo) -> Path:
    member_path = PurePosixPath(member.filename)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise UnsafeArchiveError(f"Unsafe path in model archive: {member.filename!r}.")
    file_type = (member.external_attr >> 16) & 0o170000
    if stat.S_ISLNK(file_type):
        raise UnsafeArchiveError(f"Symbolic links are not allowed in model archives: {member.filename!r}.")
    if member.flag_bits & 0x1:
        raise UnsafeArchiveError(f"Encrypted model archive member is not supported: {member.filename!r}.")
    return root.joinpath(*member_path.parts)


def _member_matches_file(source: zipfile.ZipFile, member: zipfile.ZipInfo, path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != member.file_size:
        return False
    with source.open(member) as incoming, path.open("rb") as existing:
        while True:
            incoming_chunk = incoming.read(COPY_CHUNK_BYTES)
            existing_chunk = existing.read(COPY_CHUNK_BYTES)
            if incoming_chunk != existing_chunk:
                return False
            if not incoming_chunk:
                return True


def _safe_extract(
    archive: Path,
    destination: Path,
    *,
    merge_identical_files: bool = False,
) -> None:
    with zipfile.ZipFile(archive) as source:
        total_uncompressed = sum(member.file_size for member in source.infolist())
        if total_uncompressed > MAX_UNCOMPRESSED_ARCHIVE_BYTES:
            raise UnsafeArchiveError(
                f"Model archive expands to {total_uncompressed} bytes, above the safety limit."
            )
        for member in source.infolist():
            target = _validated_member_path(destination, member)
            if member.is_dir():
                if target.exists() and not target.is_dir():
                    raise UnsafeArchiveError(f"Archive directory conflicts with a file: {member.filename!r}.")
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if merge_identical_files and _member_matches_file(source, member, target):
                    continue
                raise UnsafeArchiveError(f"Conflicting duplicate model file: {member.filename!r}.")
            with source.open(member) as input_handle, target.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=COPY_CHUNK_BYTES)


def _model_source_directory(extracted: Path) -> Path:
    config_paths = sorted(extracted.rglob("inference_config.json"))
    if len(config_paths) != 1:
        raise AssetVerificationError(
            "A SPINEPS model archive must contain exactly one inference_config.json; "
            f"found {len(config_paths)}."
        )
    return config_paths[0].parent


def _file_records(model_directory: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(candidate for candidate in model_directory.rglob("*") if candidate.is_file()):
        if path.name == ASSET_MANIFEST_NAME:
            continue
        records.append(
            {
                "path": path.relative_to(model_directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _manifest_payload(spec: ModelAssetSpec, model_directory: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_id": spec.model_id,
        "phase": spec.phase,
        "release": spec.release,
        "asset_name": spec.asset_name,
        "source_url": spec.url,
        "archive_bytes": spec.bytes,
        "archive_sha256": spec.sha256,
        "files": _file_records(model_directory),
    }


def _verify_file_inventory(destination: Path, records: object, label: str) -> None:
    if not isinstance(records, list):
        raise AssetVerificationError(f"Installed {label} manifest has no valid file inventory.")
    for record in records:
        if not isinstance(record, dict):
            raise AssetVerificationError(f"Installed {label} manifest has an invalid file record.")
        relative = record.get("path")
        expected_bytes = record.get("bytes")
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(expected_bytes, int)
            or not isinstance(expected_sha256, str)
        ):
            raise AssetVerificationError(f"Installed {label} manifest has an invalid file record.")
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise AssetVerificationError(f"Installed {label} manifest contains an unsafe file path.")
    if records != _file_records(destination):
        raise AssetVerificationError(f"Installed {label} bundle does not match its complete file inventory.")


def verify_installed_asset(
    model_root: Path,
    spec: ModelAssetSpec,
    *,
    full: bool = False,
) -> Path:
    destination = model_root / spec.install_dir
    manifest_path = destination / ASSET_MANIFEST_NAME
    config_paths = sorted(destination.rglob("inference_config.json")) if destination.is_dir() else []
    if not manifest_path.is_file() or len(config_paths) != 1:
        raise AssetVerificationError(f"Installed {spec.model_id} bundle is incomplete: {destination}.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssetVerificationError(f"Invalid asset manifest: {manifest_path}.") from error
    expected = {
        "schema_version": 1,
        "model_id": spec.model_id,
        "phase": spec.phase,
        "release": spec.release,
        "asset_name": spec.asset_name,
        "source_url": spec.url,
        "archive_bytes": spec.bytes,
        "archive_sha256": spec.sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssetVerificationError(
                f"Installed {spec.model_id} manifest has unexpected {key}: {payload.get(key)!r}."
            )
    if full:
        _verify_file_inventory(destination, payload.get("files"), spec.model_id)
    return destination


def install_archive(archive: Path, model_root: Path, spec: ModelAssetSpec) -> Path:
    """Verify and atomically install an already downloaded model archive."""

    model_root.mkdir(parents=True, exist_ok=True)
    destination = model_root / spec.install_dir
    if destination.exists():
        return verify_installed_asset(model_root, spec, full=True)

    verify_archive(archive, spec)
    with tempfile.TemporaryDirectory(prefix=f".{spec.install_dir}-", dir=model_root) as temporary:
        temporary_path = Path(temporary)
        extracted = temporary_path / "extracted"
        staged = temporary_path / "staged"
        extracted.mkdir()
        _safe_extract(archive, extracted)
        source_directory = _model_source_directory(extracted)
        shutil.copytree(source_directory, staged)
        payload = _manifest_payload(spec, staged)
        (staged / ASSET_MANIFEST_NAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staged, destination)
    return verify_installed_asset(model_root, spec, full=True)


def _verify_release_archive(path: Path, asset: ReleaseAssetPin) -> None:
    if not path.is_file():
        raise AssetVerificationError(f"Model archive not found: {path}.")
    if path.stat().st_size != asset.bytes:
        raise AssetVerificationError(f"{asset.asset_name} archive size does not match its pin.")
    if sha256_file(path) != asset.sha256:
        raise AssetVerificationError(f"{asset.asset_name} archive digest does not match its pin.")


def _find_vibeseg_dataset_root(extracted: Path) -> tuple[Path, Path]:
    trained_models = sorted(
        plans.parent
        for plans in extracted.rglob("plans.json")
        if (plans.parent / "dataset.json").is_file()
        and any(plans.parent.glob("fold_*/checkpoint_final.pth"))
    )
    if len(trained_models) != 1:
        raise AssetVerificationError(
            "The VibeSeg bundle must contain exactly one trained model with dataset.json, "
            f"plans.json, and fold checkpoints; found {len(trained_models)}."
        )
    trained_model = trained_models[0]
    dataset_root = trained_model.parent
    return dataset_root, trained_model


def _release_asset_records(assets: Sequence[ReleaseAssetPin]) -> list[dict[str, object]]:
    return [
        {
            "asset_name": asset.asset_name,
            "source_url": asset.url,
            "archive_bytes": asset.bytes,
            "archive_sha256": asset.sha256,
        }
        for asset in assets
    ]


def verify_installed_vibeseg(
    model_root: Path,
    assets: Sequence[ReleaseAssetPin],
    *,
    full: bool = False,
) -> Path:
    destination = model_root / VIBESEG_INSTALL_DIR
    manifest_path = destination / ASSET_MANIFEST_NAME
    if not manifest_path.is_file():
        raise AssetVerificationError(f"Installed VibeSeg bundle is incomplete: {destination}.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssetVerificationError(f"Invalid asset manifest: {manifest_path}.") from error
    expected = {
        "schema_version": 1,
        "model_id": "vibeseg_crop_dataset100",
        "dataset_id": 100,
        "assets": _release_asset_records(assets),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssetVerificationError(f"Installed VibeSeg manifest has unexpected {key}.")
    relative = payload.get("trained_model_path")
    if not isinstance(relative, str):
        raise AssetVerificationError("Installed VibeSeg manifest has no trained-model path.")
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise AssetVerificationError("Installed VibeSeg manifest contains an unsafe trained-model path.")
    trained_model = destination.joinpath(*relative_path.parts)
    if not (trained_model / "dataset.json").is_file() or not (trained_model / "plans.json").is_file():
        raise AssetVerificationError("Installed VibeSeg trained model is incomplete.")
    if not any(trained_model.glob("fold_*/checkpoint_final.pth")):
        raise AssetVerificationError("Installed VibeSeg trained model has no fold checkpoint.")
    if full:
        _verify_file_inventory(destination, payload.get("files"), "VibeSeg")
    return trained_model


def install_vibeseg_archives(
    archive_paths: Mapping[str, Path],
    model_root: Path,
    assets: Sequence[ReleaseAssetPin],
) -> Path:
    """Verify and atomically merge the pinned multipart VibeSeg Dataset100 bundle."""

    expected_names = {asset.asset_name for asset in assets}
    if set(archive_paths) != expected_names:
        raise AssetVerificationError(
            f"VibeSeg archives must exactly match the pinned set: {sorted(expected_names)}."
        )
    model_root.mkdir(parents=True, exist_ok=True)
    destination = model_root / VIBESEG_INSTALL_DIR
    if destination.exists():
        return verify_installed_vibeseg(model_root, assets, full=True)

    for asset in assets:
        _verify_release_archive(Path(archive_paths[asset.asset_name]), asset)
    with tempfile.TemporaryDirectory(prefix=f".{VIBESEG_INSTALL_DIR}-", dir=model_root) as temporary:
        temporary_path = Path(temporary)
        extracted = temporary_path / "extracted"
        staged = temporary_path / "staged"
        extracted.mkdir()
        for asset in assets:
            _safe_extract(
                Path(archive_paths[asset.asset_name]),
                extracted,
                merge_identical_files=True,
            )
        dataset_root, trained_model = _find_vibeseg_dataset_root(extracted)
        shutil.copytree(dataset_root, staged)
        trained_model_relative = trained_model.relative_to(dataset_root).as_posix()
        payload = {
            "schema_version": 1,
            "model_id": "vibeseg_crop_dataset100",
            "dataset_id": 100,
            "assets": _release_asset_records(assets),
            "trained_model_path": trained_model_relative,
            "files": _file_records(staged),
        }
        (staged / ASSET_MANIFEST_NAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staged, destination)
    return verify_installed_vibeseg(model_root, assets, full=True)


def _copy_download(
    source: BinaryIO,
    destination: Path,
    expected_bytes: int,
) -> None:
    written = 0
    with destination.open("wb") as output:
        while True:
            chunk = source.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)
            written += len(chunk)
            if written > expected_bytes:
                raise AssetVerificationError("Downloaded model archive is larger than its pinned size.")
        output.flush()
        os.fsync(output.fileno())


def download_and_install(
    model_root: Path,
    spec: ModelAssetSpec,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> Path:
    """Download one exact upstream asset, verify it, and install it atomically."""

    model_root.mkdir(parents=True, exist_ok=True)
    destination = model_root / spec.install_dir
    if destination.exists():
        return verify_installed_asset(model_root, spec, full=True)

    with tempfile.TemporaryDirectory(prefix=f".{spec.install_dir}-download-", dir=model_root) as temporary:
        archive = Path(temporary) / spec.asset_name
        request = urllib.request.Request(spec.url, headers={"User-Agent": "BodyComposition-model-sync/1"})
        with opener(request) as response:
            _copy_download(response, archive, spec.bytes)
        return install_archive(archive, model_root, spec)


def _download_release_archive(
    destination: Path,
    asset: ReleaseAssetPin,
    *,
    opener: Callable[..., BinaryIO],
) -> None:
    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "BodyComposition-model-sync/1"},
    )
    with opener(request) as response:
        _copy_download(response, destination, asset.bytes)
    _verify_release_archive(destination, asset)


def verify_model_bundle(model_root: Path, *, full: bool = False) -> ModelSyncReport:
    """Verify the complete pinned bundle without accessing the network."""

    root = Path(model_root)
    installed = {
        asset.model_id: verify_installed_asset(root, asset, full=full)
        for asset in SPINEPS_MODEL_ASSETS
    }
    vibeseg = verify_installed_vibeseg(root, VIBESEG_CROP_ASSETS, full=full)
    return ModelSyncReport(
        model_root=root,
        spineps_models=installed,
        vibeseg_trained_model=vibeseg,
        downloaded_assets=(),
    )


def sync_models(
    model_root: Path,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> ModelSyncReport:
    """Synchronize all exact upstream assets and atomically install them.

    Existing verified assets are reused. Downloads are temporary and every
    archive is size/hash checked before extraction and promotion.
    """

    root = Path(model_root)
    root.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    installed: dict[str, Path] = {}
    for asset in SPINEPS_MODEL_ASSETS:
        try:
            installed[asset.model_id] = verify_installed_asset(root, asset, full=True)
        except AssetVerificationError as exc:
            destination = root / asset.install_dir
            if destination.exists():
                raise AssetVerificationError(
                    f"Existing {asset.model_id} bundle is invalid: {destination}. "
                    "Move it aside before synchronizing again."
                ) from exc
            installed[asset.model_id] = download_and_install(root, asset, opener=opener)
            downloaded.append(asset.asset_name)

    try:
        vibeseg = verify_installed_vibeseg(root, VIBESEG_CROP_ASSETS, full=True)
    except AssetVerificationError as exc:
        destination = root / VIBESEG_INSTALL_DIR
        if destination.exists():
            raise AssetVerificationError(
                f"Existing VibeSeg bundle is invalid: {destination}. "
                "Move it aside before synchronizing again."
            ) from exc
        with tempfile.TemporaryDirectory(prefix=".vibeseg-download-", dir=root) as temporary:
            archive_paths: dict[str, Path] = {}
            for asset in VIBESEG_CROP_ASSETS:
                archive_path = Path(temporary) / asset.asset_name
                _download_release_archive(archive_path, asset, opener=opener)
                archive_paths[asset.asset_name] = archive_path
                downloaded.append(asset.asset_name)
            vibeseg = install_vibeseg_archives(
                archive_paths,
                root,
                VIBESEG_CROP_ASSETS,
            )

    verified = verify_model_bundle(root, full=True)
    return ModelSyncReport(
        model_root=verified.model_root,
        spineps_models=verified.spineps_models,
        vibeseg_trained_model=vibeseg,
        downloaded_assets=tuple(downloaded),
    )
