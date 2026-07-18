"""Unified, cache-only model inventory, synchronization, and verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from BodyComposition.config import PipelineConfig


class ModelAssetError(RuntimeError):
    """Raised when a pinned model asset is missing, modified, or unavailable."""


@dataclass(frozen=True)
class ModelStatus:
    model_id: str
    ready: bool
    model_root: Path
    errors: tuple[str, ...]
    checked_files: Mapping[str, str]
    asset: Mapping[str, Any]
    release_issues: tuple[str, ...] = ()

    def as_dict(self, *, include_path: bool = True) -> dict[str, Any]:
        result = {
            "model_id": self.model_id,
            "ready": self.ready,
            "errors": list(self.errors),
            "checked_files": dict(self.checked_files),
            "asset": dict(self.asset),
            "release_issues": list(self.release_issues),
        }
        if include_path:
            result["model_root"] = str(self.model_root)
        return result

    def provenance(self) -> dict[str, Any]:
        """Return verified, path-free information safe for analysis manifests."""

        return self.as_dict(include_path=False)


def _file(sha256: str, size: int) -> dict[str, Any]:
    return {"sha256": sha256, "byte_size": size}


INTERNAL_MODELS: Mapping[str, dict[str, Any]] = {
    "bodycomposition_resenc_l_v1": {
        "name": "BodyCompositionCT-ResEncL",
        "directory": "Dataset611_BodyComposition",
        "trainer": "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
        "repository": "https://huggingface.co/fhofmann/BodyCompositionCT-ResEncL",
        "repo_id": "fhofmann/BodyCompositionCT-ResEncL",
        "revision": None,
        "license": "CC-BY-SA-4.0",
        "weight_license_status": "declared_locally_but_original_repository_not_public",
        "files": {
            "dataset.json": _file("9841afba9b5f2180d870f68e12eb3e5695c5c0e4f64b91dabc52d849d67c340c", 786),
            "plans.json": _file("1790c1825529e29de536c73f66f5fd8ee12b9812271e93434e76a000c1e7cae0", 10757),
            "fold_0/checkpoint_final.pth": _file("d4eb90614fef10c2eb9aad574d4cb3517511d224f2f2da897fe4e4618513cd77", 819931462),
            "fold_1/checkpoint_final.pth": _file("f520d3293d667986dd89fae115e3602af9f40c98413a08808d1cbb0ce08e074c", 819931142),
            "fold_2/checkpoint_final.pth": _file("eeca2b9d95426147c6d533ee82ea43c215e33a785faaeb2443e5def4328bf42c", 819931718),
            "fold_3/checkpoint_final.pth": _file("8a2d7fc6a32dccad3a43cd7fcad392b38df490303baf3842e8ee57587bcfbf4a", 819931142),
            "fold_4/checkpoint_final.pth": _file("9d4c69c505961643bc97490e7a162258ac0bd3d2d17a6f58b8900d2143946231", 819931462),
        },
    },
    "bodycomposition_resenc_m_v1": {
        "name": "BodyCompositionCT-ResEncM",
        "directory": "Dataset611_BodyComposition",
        "trainer": "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
        "repository": "https://huggingface.co/fhofmann/BodyCompositionCT-ResEncM",
        "repo_id": "fhofmann/BodyCompositionCT-ResEncM",
        "revision": None,
        "license": "CC-BY-SA-4.0",
        "weight_license_status": "declared_locally_but_original_repository_not_public",
        "files": {
            "dataset.json": _file("9841afba9b5f2180d870f68e12eb3e5695c5c0e4f64b91dabc52d849d67c340c", 786),
            "plans.json": _file("d80992884b678fae2e6e09ad4147a30a59bcf39d6fd75f40da6a6bb57ef09afd", 15985),
            "fold_all/checkpoint_final.pth": _file("91ca20c9bd2674cbc0c25a5e56c70e3285b8e8f6bdf064430ad4a1d76de65f90", 819932358),
        },
    },
    "vertebral_bodies_resenc_l": {
        "name": "VertebralBodiesCT-ResEncL",
        "directory": "Dataset601_VertebralBodies",
        "trainer": "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
        "repository": "https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncL",
        "repo_id": "fhofmann/VertebralBodiesCT-ResEncL",
        "revision": "b5c3025abe6285832645b242f4abaf79ec65a8aa",
        "license": "CC-BY-SA-4.0",
        "weight_license_status": "published_with_model_card",
        "files": {
            "dataset.json": _file("5fa8ef393148d555064b8f335bcf4f2fe90bcd7d60b23f5b4e30f5281f1a5061", 1136),
            "plans.json": _file("5e65f241dd7760516495e5947f3ebee5ffbbe54114ecf85b8cb0173153ff302f", 10767),
            "fold_0/checkpoint_final.pth": _file("d7edae71a61cad518d870b1d367b7657c618872eb930758aadd44d6623857682", 820934534),
            "fold_1/checkpoint_final.pth": _file("abe017ce2fec85a1b8e98de1e2bf00925f11c763871571d9a7ee2fdbc43cd1f0", 820935302),
            "fold_2/checkpoint_final.pth": _file("002bfe99c365e6a44095be338d67ea0c0fc630a9ca9d4aaa5fd88cd61e40f02a", 820939462),
            "fold_3/checkpoint_final.pth": _file("d85fce4f8eddc21ba4be8b9c9de9752581d4fe93582bfaf281b5a4ca95b8ff1d", 820933958),
            "fold_4/checkpoint_final.pth": _file("1592d00829bc9f3f88e0bac62727fde72056095e3f39e8eb8219c9d0a3347461", 820934214),
        },
    },
    "vertebral_bodies_resenc_m": {
        "name": "VertebralBodiesCT-ResEncM",
        "directory": "Dataset601_VertebralBodies",
        "trainer": "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
        "repository": "https://huggingface.co/fhofmann/VertebralBodiesCT-ResEncM",
        "repo_id": "fhofmann/VertebralBodiesCT-ResEncM",
        "revision": "c7f2d09ebe12d54b9413d8cf6c6f4bd1cc7ef9f0",
        "license": "CC-BY-SA-4.0",
        "weight_license_status": "published_with_model_card",
        "files": {
            "dataset.json": _file("5fa8ef393148d555064b8f335bcf4f2fe90bcd7d60b23f5b4e30f5281f1a5061", 1136),
            "plans.json": _file("48179675760a922ed388fba505816bf7a1305a12b4b4e8f34aaefa66c5f2721e", 10766),
            "fold_all/checkpoint_final.pth": _file("0ebcfa65b7c3acc145839403db80676deca494ee5a0d46fa2db3a6607fdf2862", 820934982),
        },
    },
}


MODEL_IDS = (
    "ctdeeprot_2d_v1",
    "spineps_veridah_ct_v1",
    "bodycomposition_resenc_l_v1",
    "bodycomposition_resenc_m_v1",
    "vertebral_bodies_resenc_l",
    "vertebral_bodies_resenc_m",
    "totalsegmentator_total_task297_landmarks_v1",
    "totalsegmentator_body_task299_v1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _internal_asset(model_id: str) -> dict[str, Any]:
    model = INTERNAL_MODELS[model_id]
    files = [
        {"relative_path": relative, **expected}
        for relative, expected in model["files"].items()
    ]
    return {
        "asset_id": model_id,
        "name": model["name"],
        "upstream_project": model["repository"],
        "upstream_repository": model["repository"],
        "upstream_revision": model["revision"],
        "citation": "BodyComposition model card and associated validation report",
        "download_url": model["repository"],
        "expected_files": files,
        "code_license": "Apache-2.0 (nnU-Net)",
        "weight_license": model["license"],
        "weight_license_status": model["weight_license_status"],
        "redistribution_mode": "user_model_sync",
        "compatibility": {"nnunetv2": "2.5.2", "python": "3.11"},
        "validation_set_version": "bodycomposition-resenc-real-model-smoke-v1",
    }


def model_asset(model_id: str) -> dict[str, Any]:
    if model_id in INTERNAL_MODELS:
        return _internal_asset(model_id)
    if model_id == "ctdeeprot_2d_v1":
        from BodyComposition.orientation.ctdeeprot import model_asset_record

        return model_asset_record()
    if model_id == "spineps_veridah_ct_v1":
        from BodyComposition.vertebral.spineps_manifest import (
            MODEL_BUNDLE_VERSION,
            MODEL_WEIGHT_REDISTRIBUTION_MODE,
            SPINEPS_MODEL_ASSETS,
            SPINEPS_SOURCE_COMMIT,
            SPINEPS_SOURCE_LICENSE,
            SPINEPS_VERSION,
            TPTBOX_COMPAT_VERSION,
            TPTBOX_LICENSE_STATUS,
            VIBESEG_CROP_ASSETS,
            VIBESEG_CROP_DATASET_ID,
            VIBESEG_CROP_RELEASE,
            VIBESEG_SOURCE_LICENSE,
            VIBESEG_WEIGHT_LICENSE_STATUS,
        )

        return {
            "asset_id": MODEL_BUNDLE_VERSION,
            "name": "SPINEPS/VERIDAH CT with VibeSeg Dataset100",
            "upstream_project": "SPINEPS, VERIDAH, TPTBox, and VIBESegmentator",
            "upstream_version": SPINEPS_VERSION,
            "upstream_commit": SPINEPS_SOURCE_COMMIT,
            "compatibility": {"SPINEPS": SPINEPS_VERSION, "TPTBox": TPTBOX_COMPAT_VERSION},
            "code_license": SPINEPS_SOURCE_LICENSE,
            "tptbox_license_status": TPTBOX_LICENSE_STATUS,
            "vibeseg_code_license": VIBESEG_SOURCE_LICENSE,
            "weight_license_status": VIBESEG_WEIGHT_LICENSE_STATUS,
            "redistribution_mode": MODEL_WEIGHT_REDISTRIBUTION_MODE,
            "spineps_assets": [
                {
                    "model_id": asset.model_id,
                    "phase": asset.phase,
                    "release": asset.release,
                    "download_url": asset.url,
                    "archive_byte_size": asset.bytes,
                    "archive_sha256": asset.sha256,
                }
                for asset in SPINEPS_MODEL_ASSETS
            ],
            "required_crop_model": {
                "dataset_id": VIBESEG_CROP_DATASET_ID,
                "release": VIBESEG_CROP_RELEASE,
                "archives": [
                    {
                        "download_url": asset.url,
                        "archive_byte_size": asset.bytes,
                        "archive_sha256": asset.sha256,
                    }
                    for asset in VIBESEG_CROP_ASSETS
                ],
            },
            "validation_set_version": "spineps-veridah-adapter-validation-v1",
        }
    if model_id in {
        "totalsegmentator_total_task297_landmarks_v1",
        "totalsegmentator_body_task299_v1",
    }:
        from BodyComposition.measurement.totalsegmentator_assets import model_asset_record

        task = "body_landmarks" if "task297" in model_id else "bodytrunk"
        return model_asset_record(task)
    raise ValueError(f"Unknown model ID {model_id!r}.")


def list_models() -> tuple[dict[str, Any], ...]:
    return tuple(
        {"model_id": model_id, **model_asset(model_id)}
        for model_id in MODEL_IDS
    )


def required_model_ids(config: PipelineConfig) -> tuple[str, ...]:
    value = config.normalized()
    required = ["ctdeeprot_2d_v1", config.vertebral_backend, config.tissue_backend]
    if value["body_surface"]["backend"] == "totalsegmentator_body_task299_v1":
        required.append("totalsegmentator_body_task299_v1")
    if value["measurements"]["landmarks"]["enabled"]:
        required.append("totalsegmentator_total_task297_landmarks_v1")
    return tuple(dict.fromkeys(required))


def _check_internal(model_id: str, root: Path) -> ModelStatus:
    model = INTERNAL_MODELS[model_id]
    model_root = root / model["directory"] / model["trainer"]
    checked: dict[str, str] = {}
    errors: list[str] = []
    for relative, expected in model["files"].items():
        path = model_root / relative
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
    release_issues = (
        ("upstream_revision_unresolved",) if model["revision"] is None else ()
    )
    return ModelStatus(
        model_id=model_id,
        ready=not errors,
        model_root=model_root,
        errors=tuple(errors),
        checked_files=checked,
        asset=_internal_asset(model_id),
        release_issues=release_issues,
    )


def verify_model(model_id: str, root: str | Path) -> ModelStatus:
    model_root = Path(root).expanduser()
    if model_id in INTERNAL_MODELS:
        return _check_internal(model_id, model_root)
    if model_id == "ctdeeprot_2d_v1":
        from BodyComposition.orientation.ctdeeprot import (
            CHECKPOINT_FILENAME,
            UPSTREAM_COMMIT,
            verify_checkpoint,
        )

        path = model_root / "CTDeepRot" / UPSTREAM_COMMIT / CHECKPOINT_FILENAME
        errors: list[str] = []
        checked = {}
        try:
            verify_checkpoint(path)
            checked[CHECKPOINT_FILENAME] = _sha256(path)
        except Exception as error:
            errors.append(str(error))
        return ModelStatus(model_id, not errors, path.parent, tuple(errors), checked, model_asset(model_id))
    if model_id == "spineps_veridah_ct_v1":
        import importlib.metadata

        from BodyComposition.vertebral.spineps_assets import (
            AssetVerificationError,
            verify_model_bundle,
        )
        from BodyComposition.vertebral.spineps_manifest import (
            SPINEPS_VERSION,
            TPTBOX_COMPAT_VERSION,
        )
        path = model_root / "SPINEPS" / "spineps-veridah-ct-v1"
        model_errors: list[str] = []
        for distribution, expected in (
            ("SPINEPS", SPINEPS_VERSION),
            ("TPTBox", TPTBOX_COMPAT_VERSION),
        ):
            try:
                observed = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                observed = None
            if observed != expected:
                model_errors.append(f"package_version:{distribution}:{observed or 'missing'}")
        bundle_verified = False
        try:
            verify_model_bundle(path, full=True)
            bundle_verified = True
        except (AssetVerificationError, OSError) as error:
            model_errors.append(str(error))
        asset = model_asset(model_id)
        checked = {}
        if bundle_verified:
            checked = {
                item["model_id"]: item["archive_sha256"]
                for item in asset["spineps_assets"]
            }
            checked.update(
                {
                    f"vibeseg:{index}": item["archive_sha256"]
                    for index, item in enumerate(asset["required_crop_model"]["archives"])
                }
            )
        return ModelStatus(
            model_id,
            not model_errors,
            path,
            tuple(model_errors),
            checked,
            asset,
        )
    if model_id in {
        "totalsegmentator_total_task297_landmarks_v1",
        "totalsegmentator_body_task299_v1",
    }:
        from BodyComposition.measurement.totalsegmentator_assets import check_measurement_model

        task = "body_landmarks" if "task297" in model_id else "bodytrunk"
        report = check_measurement_model(task, model_root)
        return ModelStatus(
            model_id,
            report.ready,
            report.model_directory,
            tuple(report.errors),
            dict(report.checked_files),
            model_asset(model_id),
        )
    raise ValueError(f"Unknown model ID {model_id!r}.")


def verify_models(
    config: PipelineConfig,
    model_ids: Sequence[str] | None = None,
) -> tuple[ModelStatus, ...]:
    selected = tuple(required_model_ids(config) if model_ids is None else model_ids)
    return tuple(verify_model(model_id, config.model_root) for model_id in selected)


def require_models(config: PipelineConfig) -> tuple[ModelStatus, ...]:
    reports = verify_models(config)
    failed = [report for report in reports if not report.ready]
    if failed:
        details = "; ".join(
            f"{report.model_id}: {', '.join(report.errors)}" for report in failed
        )
        raise ModelAssetError(
            f"Required pinned model assets are not ready ({details}). Run "
            "`bodycomposition models sync --config CONFIG` and then "
            "`bodycomposition models verify --config CONFIG`."
        )
    return reports


def release_model_issues(
    config: PipelineConfig,
    statuses: Sequence[ModelStatus] | None = None,
    *,
    include_all_selectable: bool = False,
) -> tuple[str, ...]:
    """Return publication blockers separately from local cache readiness."""

    checked = tuple(verify_models(config) if statuses is None else statuses)
    issues: list[str] = []
    for report in checked:
        issues.extend(
            f"{report.model_id}:{issue}" for issue in report.release_issues
        )
    if include_all_selectable:
        for model_id, model in INTERNAL_MODELS.items():
            issue = f"{model_id}:upstream_revision_unresolved"
            if model["revision"] is None and issue not in issues:
                issues.append(issue)
    return tuple(issues)


def _sync_internal(model_id: str, root: Path) -> ModelStatus:
    model = INTERNAL_MODELS[model_id]
    existing = _check_internal(model_id, root)
    if existing.ready:
        return existing
    revision = model["revision"]
    if revision is None:
        raise ModelAssetError(
            f"{model['name']} cannot be synchronized reproducibly because its intended "
            "original Hugging Face repository is not public and has no immutable revision. "
            "Publish/freeze that original model source before the release can pass."
        )
    from huggingface_hub import snapshot_download

    destination = root / model["directory"]
    trainer = model["trainer"]
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model-sync-", dir=root) as temporary:
        staged_root = Path(temporary)
        staged_dataset = staged_root / model["directory"]
        snapshot_download(
            repo_id=model["repo_id"],
            revision=revision,
            local_dir=staged_dataset,
            allow_patterns=[f"{trainer}/**"],
        )
        staged = _check_internal(model_id, staged_root)
        if not staged.ready:
            raise ModelAssetError(
                f"Downloaded {model_id} failed verification: {', '.join(staged.errors)}."
            )
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / trainer
        replacement = staged_dataset / trainer
        if not replacement.is_dir():
            raise ModelAssetError(
                f"Pinned upstream snapshot did not contain expected trainer {trainer}."
            )
        previous = destination / f".{trainer}.replaced-{uuid4().hex}"
        if target.exists():
            os.replace(target, previous)
        try:
            os.replace(replacement, target)
        except Exception:
            if previous.exists() and not target.exists():
                os.replace(previous, target)
            raise
        finally:
            if previous.exists():
                shutil.rmtree(previous)
    report = _check_internal(model_id, root)
    if not report.ready:
        raise ModelAssetError(
            f"Synchronized {model_id} failed verification: {', '.join(report.errors)}."
        )
    return report


def sync_model(model_id: str, root: str | Path) -> ModelStatus:
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown model ID {model_id!r}.")
    model_root = Path(root).expanduser()
    model_root.mkdir(parents=True, exist_ok=True)
    try:
        if model_id in INTERNAL_MODELS:
            return _sync_internal(model_id, model_root)
        if model_id == "ctdeeprot_2d_v1":
            from BodyComposition.orientation.ctdeeprot import (
                CHECKPOINT_FILENAME,
                UPSTREAM_COMMIT,
                sync_checkpoint,
            )

            sync_checkpoint(model_root / "CTDeepRot" / UPSTREAM_COMMIT / CHECKPOINT_FILENAME)
        elif model_id == "spineps_veridah_ct_v1":
            from BodyComposition.vertebral.spineps_assets import sync_models

            sync_models(model_root / "SPINEPS" / "spineps-veridah-ct-v1")
        elif model_id in {
            "totalsegmentator_total_task297_landmarks_v1",
            "totalsegmentator_body_task299_v1",
        }:
            from BodyComposition.measurement.totalsegmentator_assets import sync_measurement_model

            task = "body_landmarks" if "task297" in model_id else "bodytrunk"
            sync_measurement_model(task, model_root)
    except ModelAssetError:
        raise
    except Exception as error:
        raise ModelAssetError(
            f"Could not synchronize pinned model {model_id}: {type(error).__name__}."
        ) from error
    report = verify_model(model_id, model_root)
    if not report.ready:
        raise ModelAssetError(
            f"Synchronized {model_id} failed verification: {', '.join(report.errors)}."
        )
    return report


def sync_models(
    config: PipelineConfig,
    model_ids: Sequence[str] | None = None,
) -> tuple[ModelStatus, ...]:
    selected = tuple(required_model_ids(config) if model_ids is None else model_ids)
    unresolved_missing = [
        model_id
        for model_id in selected
        if model_id in INTERNAL_MODELS
        and INTERNAL_MODELS[model_id]["revision"] is None
        and not _check_internal(model_id, config.model_root).ready
    ]
    if unresolved_missing:
        raise ModelAssetError(
            "Model synchronization cannot start because these selected assets have no "
            "public immutable upstream revision: "
            f"{', '.join(unresolved_missing)}."
        )
    return tuple(sync_model(model_id, config.model_root) for model_id in selected)


def model_bundle_digest(statuses: Sequence[ModelStatus]) -> str:
    payload = [
        status.provenance()
        for status in sorted(statuses, key=lambda status: status.model_id)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
