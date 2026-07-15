import hashlib
import json
import zipfile

import pytest

from BodyComposition.vertebral.spineps_assets import (
    AssetVerificationError,
    UnsafeArchiveError,
    install_archive,
    install_vibeseg_archives,
    verify_installed_asset,
    verify_installed_vibeseg,
)
from BodyComposition.vertebral.spineps_manifest import ModelAssetSpec, ReleaseAssetPin


def _archive(tmp_path, members):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def _spec(path, *, digest=None, size=None):
    return ModelAssetSpec(
        model_id="test_model",
        phase="semantic",
        release="v0",
        asset_name="model.zip",
        url="https://github.com/Hendrik-code/spineps/releases/download/v0/model.zip",
        bytes=path.stat().st_size if size is None else size,
        sha256=digest or hashlib.sha256(path.read_bytes()).hexdigest(),
        install_dir="test_model",
    )


def test_archive_install_is_verified_normalized_and_reusable(tmp_path):
    archive = _archive(
        tmp_path,
        {
            "nested/inference_config.json": json.dumps({"modeltype": "nnunet"}),
            "nested/fold_0/checkpoint_final.pth": b"weights",
        },
    )
    spec = _spec(archive)
    model_root = tmp_path / "models"

    installed = install_archive(archive, model_root, spec)

    assert installed == model_root / "test_model"
    assert (installed / "inference_config.json").is_file()
    assert (installed / "fold_0" / "checkpoint_final.pth").read_bytes() == b"weights"
    assert verify_installed_asset(model_root, spec, full=True) == installed
    assert install_archive(archive, model_root, spec) == installed


@pytest.mark.parametrize("field", ["size", "digest"])
def test_archive_install_rejects_wrong_pin(tmp_path, field):
    archive = _archive(tmp_path, {"inference_config.json": "{}"})
    spec = _spec(
        archive,
        size=archive.stat().st_size + 1 if field == "size" else None,
        digest="0" * 64 if field == "digest" else None,
    )

    with pytest.raises(AssetVerificationError, match="mismatch"):
        install_archive(archive, tmp_path / "models", spec)


def test_archive_install_rejects_path_traversal(tmp_path):
    archive = _archive(
        tmp_path,
        {
            "../outside": "unsafe",
            "model/inference_config.json": "{}",
        },
    )

    with pytest.raises(UnsafeArchiveError, match="Unsafe path"):
        install_archive(archive, tmp_path / "models", _spec(archive))


def test_archive_requires_exactly_one_inference_config(tmp_path):
    archive = _archive(
        tmp_path,
        {
            "one/inference_config.json": "{}",
            "two/inference_config.json": "{}",
        },
    )

    with pytest.raises(AssetVerificationError, match="exactly one"):
        install_archive(archive, tmp_path / "models", _spec(archive))


def test_full_verification_rejects_unsafe_manifest_path(tmp_path):
    archive = _archive(tmp_path, {"inference_config.json": "{}"})
    spec = _spec(archive)
    installed = install_archive(archive, tmp_path / "models", spec)
    manifest_path = installed / ".bodycomposition-asset.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"][0]["path"] = "../outside"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssetVerificationError, match="unsafe file path"):
        verify_installed_asset(tmp_path / "models", spec, full=True)


def test_full_verification_rejects_unlisted_file(tmp_path):
    archive = _archive(tmp_path, {"inference_config.json": "{}"})
    spec = _spec(archive)
    installed = install_archive(archive, tmp_path / "models", spec)
    (installed / "unexpected.bin").write_bytes(b"not in the signed inventory")

    with pytest.raises(AssetVerificationError, match="complete file inventory"):
        verify_installed_asset(tmp_path / "models", spec, full=True)


def _release_pin(path):
    return ReleaseAssetPin(
        asset_name=path.name,
        url=(
            "https://github.com/robert-graf/VIBESegmentator/"
            f"releases/download/test/{path.name}"
        ),
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_multipart_vibeseg_install_merges_verified_archives(tmp_path):
    first = _archive(
        tmp_path / "one",
        {
            "Dataset100/Trainer__nnUNetPlans/dataset.json": "{}",
            "Dataset100/Trainer__nnUNetPlans/plans.json": "{}",
            "Dataset100/Trainer__nnUNetPlans/fold_0/checkpoint_final.pth": "fold zero",
        },
    )
    second = _archive(
        tmp_path / "two",
        {
            "Dataset100/Trainer__nnUNetPlans/plans.json": "{}",
            "Dataset100/Trainer__nnUNetPlans/fold_1/checkpoint_final.pth": "fold one",
        },
    )
    first = first.rename(first.with_name("100.zip"))
    second = second.rename(second.with_name("100_1.zip"))
    assets = (_release_pin(first), _release_pin(second))
    archives = {first.name: first, second.name: second}

    trained_model = install_vibeseg_archives(archives, tmp_path / "models", assets)

    assert trained_model.name == "Trainer__nnUNetPlans"
    assert (trained_model / "fold_0/checkpoint_final.pth").is_file()
    assert (trained_model / "fold_1/checkpoint_final.pth").is_file()
    assert verify_installed_vibeseg(tmp_path / "models", assets, full=True) == trained_model


def test_multipart_vibeseg_install_rejects_conflicting_overlap(tmp_path):
    first = _archive(
        tmp_path / "one",
        {
            "Dataset100/Trainer__nnUNetPlans/dataset.json": "first",
            "Dataset100/Trainer__nnUNetPlans/plans.json": "{}",
            "Dataset100/Trainer__nnUNetPlans/fold_0/checkpoint_final.pth": "weights",
        },
    )
    second = _archive(
        tmp_path / "two",
        {"Dataset100/Trainer__nnUNetPlans/dataset.json": "different"},
    )
    first = first.rename(first.with_name("100.zip"))
    second = second.rename(second.with_name("100_1.zip"))
    assets = (_release_pin(first), _release_pin(second))

    with pytest.raises(UnsafeArchiveError, match="Conflicting duplicate"):
        install_vibeseg_archives(
            {first.name: first, second.name: second},
            tmp_path / "models",
            assets,
        )
