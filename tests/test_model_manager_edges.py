from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import BodyComposition.model_manager as models
from BodyComposition.config import PipelineConfig
from BodyComposition.model_manager import ModelAssetError, ModelStatus


def _asset(*, revision: str | None = "revision") -> dict:
    content = b"expected"
    return {
        "name": "Synthetic model",
        "directory": "Dataset999_Test",
        "trainer": "SyntheticTrainer__Plans__3d_fullres",
        "repository": "https://example.invalid/model",
        "repo_id": "example/model",
        "revision": revision,
        "license": "CC-BY-SA-4.0",
        "weight_license_status": "published_with_model_card",
        "files": {
            "weights.bin": {
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_size": len(content),
            }
        },
    }


def _status(model_id: str, root: Path, *, ready: bool = True) -> ModelStatus:
    return ModelStatus(
        model_id=model_id,
        ready=ready,
        model_root=root,
        errors=() if ready else ("missing:weights.bin",),
        checked_files={"weights.bin": "a" * 64} if ready else {},
        asset={"asset_id": model_id},
    )


def test_internal_model_verification_distinguishes_missing_size_hash_and_release_pin(
    monkeypatch,
    tmp_path,
):
    model_id = "synthetic_internal"
    asset = _asset(revision=None)
    monkeypatch.setitem(models.INTERNAL_MODELS, model_id, asset)
    target = tmp_path / asset["directory"] / asset["trainer"] / "weights.bin"

    missing = models._check_internal(model_id, tmp_path)
    assert missing.errors == ("missing:weights.bin",)
    assert missing.release_issues == ("upstream_revision_unresolved",)
    assert "model_root" in missing.as_dict()
    assert "model_root" not in missing.provenance()

    target.parent.mkdir(parents=True)
    target.write_bytes(b"short")
    assert models._check_internal(model_id, tmp_path).errors == (
        "size_mismatch:weights.bin",
    )

    target.write_bytes(b"mismatch")
    mismatch = models._check_internal(model_id, tmp_path)
    assert mismatch.errors == ("sha256_mismatch:weights.bin",)
    assert mismatch.checked_files["weights.bin"] == hashlib.sha256(b"mismatch").hexdigest()

    target.write_bytes(b"expected")
    ready = models.verify_model(model_id, tmp_path)
    assert ready.ready
    assert ready.checked_files["weights.bin"] == asset["files"]["weights.bin"]["sha256"]


def test_internal_model_sync_uses_pinned_snapshot_and_replaces_bad_cache(
    monkeypatch,
    tmp_path,
):
    model_id = "synthetic_sync"
    asset = _asset()
    monkeypatch.setitem(models.INTERNAL_MODELS, model_id, asset)
    calls = []

    import huggingface_hub

    def snapshot_download(*, repo_id, revision, local_dir, allow_patterns):
        calls.append((repo_id, revision, tuple(allow_patterns)))
        target = Path(local_dir) / asset["trainer"] / "weights.bin"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"expected")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    report = models._sync_internal(model_id, tmp_path)
    assert report.ready
    assert calls == [
        (
            asset["repo_id"],
            asset["revision"],
            (f"{asset['trainer']}/weights.bin",),
        )
    ]

    target = tmp_path / asset["directory"] / asset["trainer"] / "weights.bin"
    target.write_bytes(b"mismatch")
    assert models._sync_internal(model_id, tmp_path).ready
    assert target.read_bytes() == b"expected"
    assert len(calls) == 2

    assert models._sync_internal(model_id, tmp_path).ready
    assert len(calls) == 2


def test_unresolved_internal_model_refuses_network_sync(monkeypatch, tmp_path):
    model_id = "synthetic_unresolved"
    monkeypatch.setitem(models.INTERNAL_MODELS, model_id, _asset(revision=None))
    with pytest.raises(ModelAssetError, match="no immutable revision"):
        models._sync_internal(model_id, tmp_path)


def test_public_tissue_models_are_immutable_original_sources():
    expected = {
        "bodycomposition_resenc_l_v1": (
            "b355aa7254f5307b6d18005dfbabae8c439d24fb",
            "fhofmann/BodyCompositionCT-ResEncL",
        ),
        "bodycomposition_resenc_m_v1": (
            "9c60c8f59a99442b9b8cc1a45abf6c4d81690c0d",
            "fhofmann/BodyCompositionCT-ResEncM",
        ),
    }

    for model_id, (revision, repo_id) in expected.items():
        definition = models.INTERNAL_MODELS[model_id]
        asset = models.model_asset(model_id)
        assert definition["repo_id"] == repo_id
        assert definition["revision"] == revision
        assert definition["license"] == "CC-BY-4.0"
        assert definition["weight_license_status"] == "published_with_model_card_and_license"
        assert asset["upstream_revision"] == revision
        assert asset["download_url"].endswith(f"/tree/{revision}")
        assert definition["files"]["dataset.json"] == {
            "sha256": "8de2ff6b2b4229312517acf9bfd3dd4d23cc7ebedc48c0ad8e36bd3ba10e7f1d",
            "byte_size": 844,
        }

    assert len(models.INTERNAL_MODELS["bodycomposition_resenc_l_v1"]["files"]) == 7
    assert len(models.INTERNAL_MODELS["bodycomposition_resenc_m_v1"]["files"]) == 3


def test_verify_model_routes_ctdeeprot_and_captures_checkpoint_errors(
    monkeypatch,
    tmp_path,
):
    from BodyComposition.orientation import ctdeeprot

    checkpoint = (
        tmp_path
        / "CTDeepRot"
        / ctdeeprot.UPSTREAM_COMMIT
        / ctdeeprot.CHECKPOINT_FILENAME
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(ctdeeprot, "verify_checkpoint", lambda path: None)

    ready = models.verify_model("ctdeeprot_2d_v1", tmp_path)
    assert ready.ready
    assert ready.checked_files[ctdeeprot.CHECKPOINT_FILENAME] == hashlib.sha256(
        b"checkpoint"
    ).hexdigest()

    def fail(_path):
        raise RuntimeError("checkpoint mismatch")

    monkeypatch.setattr(ctdeeprot, "verify_checkpoint", fail)
    failed = models.verify_model("ctdeeprot_2d_v1", tmp_path)
    assert not failed.ready
    assert failed.errors == ("checkpoint mismatch",)


def test_verify_model_routes_spineps_and_checks_dependency_versions(
    monkeypatch,
    tmp_path,
):
    from BodyComposition.vertebral import spineps_assets, spineps_manifest

    expected = {
        "SPINEPS": spineps_manifest.SPINEPS_VERSION,
        "TPTBox": spineps_manifest.TPTBOX_COMPAT_VERSION,
    }
    monkeypatch.setattr(importlib.metadata, "version", lambda name: expected[name])
    monkeypatch.setattr(spineps_assets, "verify_model_bundle", lambda *_args, **_kwargs: None)

    ready = models.verify_model("spineps_veridah_ct_v1", tmp_path)
    assert ready.ready
    assert ready.checked_files

    def missing_or_wrong(name):
        if name == "SPINEPS":
            raise importlib.metadata.PackageNotFoundError
        return "wrong"

    monkeypatch.setattr(importlib.metadata, "version", missing_or_wrong)

    def invalid_bundle(*_args, **_kwargs):
        raise OSError("bundle unreadable")

    monkeypatch.setattr(spineps_assets, "verify_model_bundle", invalid_bundle)
    failed = models.verify_model("spineps_veridah_ct_v1", tmp_path)
    assert not failed.ready
    assert "package_version:SPINEPS:missing" in failed.errors
    assert "package_version:TPTBox:wrong" in failed.errors
    assert "bundle unreadable" in failed.errors
    assert not failed.checked_files


@pytest.mark.parametrize(
    ("model_id", "task"),
    [
        ("totalsegmentator_total_task297_landmarks_v1", "body_landmarks"),
        ("totalsegmentator_body_task299_v1", "bodytrunk"),
    ],
)
def test_verify_model_routes_totalsegmentator_reports(
    monkeypatch,
    tmp_path,
    model_id,
    task,
):
    from BodyComposition.measurement import totalsegmentator_assets

    calls = []
    report = SimpleNamespace(
        ready=True,
        model_directory=tmp_path / task,
        errors=(),
        checked_files={"weights": "b" * 64},
    )
    monkeypatch.setattr(
        totalsegmentator_assets,
        "check_measurement_model",
        lambda selected, root: calls.append((selected, root)) or report,
    )

    status = models.verify_model(model_id, tmp_path)
    assert status.ready
    assert calls == [(task, tmp_path)]


def test_required_and_release_model_sets_are_explicit(monkeypatch, tmp_path):
    config_data = PipelineConfig.model_validate({}).normalized()
    config_data["models"]["root"] = tmp_path
    config_data["body_surface"]["backend"] = "totalsegmentator_body_task299_v1"
    config_data["measurements"]["landmarks"]["enabled"] = False
    config = PipelineConfig.model_validate(config_data)
    required = models.required_model_ids(config)
    assert "totalsegmentator_body_task299_v1" in required
    assert "totalsegmentator_total_task297_landmarks_v1" not in required
    assert len(required) == len(set(required))

    unresolved = _status("bodycomposition_resenc_l_v1", tmp_path)
    unresolved = replace(
        unresolved,
        release_issues=("upstream_revision_unresolved",),
    )
    assert models.release_model_issues(config, (unresolved,)) == (
        "bodycomposition_resenc_l_v1:upstream_revision_unresolved",
    )
    all_issues = models.release_model_issues(
        config,
        (),
        include_all_selectable=True,
    )
    assert all_issues == ()


def test_require_models_reports_all_failed_assets(monkeypatch, tmp_path):
    config = PipelineConfig.model_validate({"models": {"root": tmp_path}})
    ready = _status("ready", tmp_path)
    failed = _status("failed", tmp_path, ready=False)
    monkeypatch.setattr(models, "verify_models", lambda _config: (ready,))
    assert models.require_models(config) == (ready,)

    monkeypatch.setattr(models, "verify_models", lambda _config: (ready, failed))
    with pytest.raises(ModelAssetError, match="failed: missing:weights.bin"):
        models.require_models(config)


def test_sync_model_routes_each_upstream_adapter_and_reverifies(
    monkeypatch,
    tmp_path,
):
    from BodyComposition.measurement import totalsegmentator_assets
    from BodyComposition.orientation import ctdeeprot
    from BodyComposition.vertebral import spineps_assets

    calls = []
    original_verify_model = models.verify_model
    monkeypatch.setattr(
        ctdeeprot,
        "sync_checkpoint",
        lambda path: calls.append(("ctdeeprot", Path(path))),
    )
    monkeypatch.setattr(
        spineps_assets,
        "sync_models",
        lambda path: calls.append(("spineps", Path(path))),
    )
    monkeypatch.setattr(
        totalsegmentator_assets,
        "sync_measurement_model",
        lambda task, root: calls.append((task, Path(root))),
    )
    monkeypatch.setattr(
        models,
        "verify_model",
        lambda model_id, root: _status(model_id, Path(root)),
    )

    for model_id in (
        "ctdeeprot_2d_v1",
        "spineps_veridah_ct_v1",
        "totalsegmentator_total_task297_landmarks_v1",
        "totalsegmentator_body_task299_v1",
    ):
        assert models.sync_model(model_id, tmp_path).ready

    assert [name for name, _path in calls] == [
        "ctdeeprot",
        "spineps",
        "body_landmarks",
        "bodytrunk",
    ]

    with pytest.raises(ValueError, match="Unknown model ID"):
        models.sync_model("unknown", tmp_path)
    with pytest.raises(ValueError, match="Unknown model ID"):
        original_verify_model("unknown", tmp_path)
    with pytest.raises(ValueError, match="Unknown model ID"):
        models.model_asset("unknown")


def test_sync_model_sanitizes_upstream_errors_and_failed_verification(
    monkeypatch,
    tmp_path,
):
    from BodyComposition.orientation import ctdeeprot

    def fail(_path):
        raise RuntimeError("private upstream detail")

    monkeypatch.setattr(ctdeeprot, "sync_checkpoint", fail)
    with pytest.raises(ModelAssetError, match="RuntimeError") as captured:
        models.sync_model("ctdeeprot_2d_v1", tmp_path)
    assert "private upstream detail" not in str(captured.value)

    monkeypatch.setattr(ctdeeprot, "sync_checkpoint", lambda _path: None)
    monkeypatch.setattr(
        models,
        "verify_model",
        lambda model_id, root: _status(model_id, Path(root), ready=False),
    )
    with pytest.raises(ModelAssetError, match="failed verification"):
        models.sync_model("ctdeeprot_2d_v1", tmp_path)


def test_sync_order_is_preserved_and_bundle_digest_is_canonical(
    monkeypatch,
    tmp_path,
):
    config = PipelineConfig.model_validate({"models": {"root": tmp_path}})
    selected = ("ctdeeprot_2d_v1", "spineps_veridah_ct_v1")
    calls = []

    def sync(model_id, root):
        calls.append((model_id, Path(root)))
        return _status(model_id, Path(root))

    monkeypatch.setattr(models, "sync_model", sync)
    statuses = models.sync_models(config, selected)
    assert [status.model_id for status in statuses] == list(selected)
    assert [model_id for model_id, _root in calls] == list(selected)
    relocated = tuple(
        replace(status, model_root=tmp_path / "relocated" / status.model_id)
        for status in statuses
    )
    assert models.model_bundle_digest(statuses) == models.model_bundle_digest(relocated)
    assert models.model_bundle_digest(statuses) == models.model_bundle_digest(
        tuple(reversed(statuses))
    )
    modified = (
        replace(statuses[0], checked_files={"weights.bin": "b" * 64}),
        statuses[1],
    )
    assert models.model_bundle_digest(statuses) != models.model_bundle_digest(modified)


def test_explicit_empty_model_sequences_do_not_expand_to_defaults(monkeypatch, tmp_path):
    config = PipelineConfig.model_validate({"models": {"root": tmp_path}})

    def unexpected(*_args, **_kwargs):
        raise AssertionError("an explicit empty selection must remain empty")

    monkeypatch.setattr(models, "verify_model", unexpected)
    monkeypatch.setattr(models, "sync_model", unexpected)

    assert models.verify_models(config, ()) == ()
    assert models.sync_models(config, ()) == ()

    monkeypatch.setattr(models, "verify_models", unexpected)
    assert models.release_model_issues(config, ()) == ()
