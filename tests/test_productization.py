from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
from jsonschema import Draft202012Validator

import BodyComposition.service as service_module
from BodyComposition import cli
from BodyComposition.config import ConfigError, PipelineConfig, configuration_schema
from BodyComposition.measurement.contracts import MEASUREMENT_SCHEMA_VERSION
from BodyComposition.model_manager import (
    INTERNAL_MODELS,
    ModelAssetError,
    ModelStatus,
    list_models,
    release_model_issues,
    required_model_ids,
    sync_model,
    sync_models,
    verify_model,
)
from BodyComposition.provenance import package_lock_digest
from BodyComposition.reporting.contracts import CaseReportResult
from BodyComposition.results import CaseResult, ExecutionStatus, QCStatus
from BodyComposition.schema_validation import SchemaValidationError
from BodyComposition.service import (
    CaseInput,
    DirtySourceError,
    ExistingRunError,
    PipelineService,
    inspect_result,
    load_batch_manifest,
)

SOURCE_CLEAN = {
    "package_version": "1.0.0rc1",
    "git_commit": "a" * 40,
    "source_dirty": False,
    "source_tree_sha256": "b" * 64,
    "container_image_digest": None,
    "environment_path_variables": [],
}


def _config(**runtime):
    return PipelineConfig.model_validate(
        {
            "measurements": {"landmarks": {"enabled": False}},
            "runtime": {"allow_dirty": True, **runtime},
        }
    )


def _model_status(tmp_path: Path) -> ModelStatus:
    return ModelStatus(
        model_id="test-model",
        ready=True,
        model_root=tmp_path,
        errors=(),
        checked_files={"weights": "c" * 64},
        asset={"asset_id": "test-model", "license": "test-only"},
    )


def _write_ct(path: Path, *, value: int = 0) -> Path:
    image = sitk.GetImageFromArray(np.full((3, 4, 5), value, dtype=np.int16))
    image.SetSpacing((0.8, 1.2, 3.0))
    image.SetOrigin((12.0, -5.0, 20.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


def test_installed_build_lock_digest_uses_valid_embedded_value(monkeypatch):
    import BodyComposition.provenance as provenance

    monkeypatch.setattr(provenance, "__file__", "/installed/BodyComposition/provenance.py")
    monkeypatch.setenv("BODYCOMPOSITION_UV_LOCK_SHA256", "A" * 64)

    assert package_lock_digest() == "a" * 64


def test_installed_build_lock_digest_rejects_invalid_embedded_value(monkeypatch):
    import BodyComposition.provenance as provenance

    monkeypatch.setattr(provenance, "__file__", "/installed/BodyComposition/provenance.py")
    monkeypatch.setenv("BODYCOMPOSITION_UV_LOCK_SHA256", "not-a-sha256")

    assert package_lock_digest() is None


def test_elapsed_run_period_uses_latest_terminal_timestamp():
    ended_at, duration_seconds = service_module._elapsed_run_period(
        "2026-07-29T08:00:00+00:00",
        [
            "2026-07-29T08:01:30+00:00",
            "2026-07-29T10:04:00+02:00",
        ],
    )

    assert ended_at == "2026-07-29T08:04:00+00:00"
    assert duration_seconds == pytest.approx(240.0)
    assert service_module._elapsed_run_period(
        "2026-07-29T08:00:00+00:00",
        ["2026-07-29T07:59:00+00:00"],
    ) == (None, None)


class SuccessfulPipeline:
    instances = 0

    def __init__(self, config, timestamp):
        type(self).instances += 1

    def __call__(self, memory):
        artifact = Path(memory["workspace"]) / "tables" / "slices.parquet"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"case_id": [memory["id"]]}).to_parquet(artifact, index=False)
        memory["tmp/orientation_result"] = SimpleNamespace(
            qc_status="review",
            review_flags=(
                {
                    "code": "orientation_changed",
                    "stage": "orientation",
                    "severity": "warning",
                    "reason": "The input was losslessly reoriented.",
                    "observed": {
                        "orientation_changed": True,
                        "native_label": np.int64(24),
                        "support_labels": np.asarray([23, 24], dtype=np.int64),
                    },
                    "threshold": {"minimum_votes": np.int64(23)},
                    "suggested_action": "Review the orientation scout.",
                },
            ),
            to_dict=lambda: {
                "state": "MISMATCH_REPAIRED",
                "orientation_changed": True,
                "manual_review_required": True,
            },
        )
        memory["tmp/vertebral_result"] = SimpleNamespace(
            backend_id="spineps_veridah_ct_v1",
            qc_status=SimpleNamespace(value="pass"),
            qc_flags=(),
        )
        memory["tmp/measurement_bundle"] = SimpleNamespace(
            qc_status=SimpleNamespace(value="pass"),
            qc_flags=(),
            slices=pd.DataFrame({"schema_version": [MEASUREMENT_SCHEMA_VERSION]}),
        )
        memory["tmp/stage_events"] = [
            {"stage": "Synthetic", "result_code": "succeeded", "duration_seconds": 0.01}
        ]


class InformationalPipeline(SuccessfulPipeline):
    def __call__(self, memory):
        super().__call__(memory)
        memory["tmp/orientation_result"] = SimpleNamespace(
            qc_status="pass",
            review_flags=(),
            to_dict=lambda: {
                "state": "PASS_METADATA_MATCH",
                "orientation_changed": False,
                "manual_review_required": False,
            },
        )
        memory["tmp/vertebral_result"] = SimpleNamespace(
            backend_id="spineps_veridah_ct_v1",
            qc_status=SimpleNamespace(value="pass"),
            qc_flags=(
                {
                    "code": "vertebra_touches_cranial_fov",
                    "stage": "vertebral",
                    "severity": "info",
                    "reason": "The observed anatomy reaches the acquisition boundary.",
                },
            ),
        )


class FailingPipeline:
    def __init__(self, config, timestamp):
        pass

    def __call__(self, memory):
        partial = Path(memory["workspace"]) / "masks" / "partial.nii.gz"
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"partial")
        memory["tmp/current_action"] = "SyntheticInference"
        raise RuntimeError("private path should never enter the manifest")


class InterruptedPipeline:
    def __init__(self, config, timestamp):
        pass

    def __call__(self, memory):
        memory["tmp/current_action"] = "SyntheticInference"
        raise KeyboardInterrupt


def _service(tmp_path, pipeline=SuccessfulPipeline, *, config=None, source=None):
    return PipelineService(
        config or _config(),
        pipeline_factory=pipeline,
        model_provider=lambda value: (_model_status(tmp_path),),
        source_provider=lambda: source or SOURCE_CLEAN,
    )


def test_public_config_is_strict_roundtrippable_and_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.delenv("BODYCOMPOSITION_MODEL_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    config = PipelineConfig.load()
    assert config.vertebral_backend == "spineps_veridah_ct_v1"
    assert config.tissue_backend == "bodycomposition_resenc_l_v1"
    assert config.model_root == Path.home() / ".cache/bodycomposition/models"
    assert config.normalized()["runtime"]["timeout_seconds"] == 14400
    assert config.normalized()["output"]["save_csv_tables"] is True
    assert config.to_runtime_dict()["measurements"]["export"] == {
        "parquet": True,
        "csv": True,
    }
    assert PipelineConfig.model_validate(config.normalized()).digest() == config.digest()
    with pytest.raises(ConfigError, match="Unknown configuration value"):
        PipelineConfig.model_validate({"method": "BodyCompositionFast"})
    with pytest.raises(ConfigError, match="fixed to 1"):
        PipelineConfig.model_validate({"runtime": {"max_workers": 2}})

    schema = configuration_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(config.normalized())


def test_scientific_identity_ignores_runtime_paths_and_reporting():
    first = _config()
    value = first.normalized()
    value["models"]["root"] = "/different/cache"
    value["runtime"]["device"] = "cpu"
    value["output"]["log_level"] = "DEBUG"
    value["reporting"]["enabled"] = True
    second = PipelineConfig.model_validate(value)
    assert first.digest() != second.digest()
    assert first.scientific_digest() == second.scientific_digest()
    value["measurements"]["tissue_definitions"]["vat_sensitivity_hu_m150_m50"] = {
        "enabled": True,
        "source_labels": ["aVAT", "tVAT", "VAT"],
        "hu_range": [-150, -50],
    }
    assert first.scientific_digest() != PipelineConfig.model_validate(value).scientific_digest()


def test_queue_identity_allows_worker_resources_but_not_output_changes():
    first = _config()
    value = first.normalized()
    value["models"]["root"] = "/worker-specific/model-cache"
    value["runtime"].update(
        {
            "device": "cpu",
            "cpu_threads": 12,
            "timeout_seconds": 7200,
            "allow_dirty": False,
        }
    )
    value["vertebrae"]["device"] = "cpu"
    value["output"]["log_level"] = "DEBUG"
    worker_variant = PipelineConfig.model_validate(value)

    assert first.digest() != worker_variant.digest()
    assert first.queue_digest() == worker_variant.queue_digest()

    value["output"]["save_body_surface"] = False
    assert first.queue_digest() != PipelineConfig.model_validate(value).queue_digest()

    csv_variant = first.normalized()
    csv_variant["output"]["save_csv_tables"] = False
    csv_config = PipelineConfig.model_validate(csv_variant)
    assert first.queue_digest() != csv_config.queue_digest()
    assert first.scientific_digest() == csv_config.scientific_digest()
    assert csv_config.to_runtime_dict()["measurements"]["export"] == {
        "parquet": True,
        "csv": False,
    }


def test_model_inventory_and_required_defaults_are_explicit():
    assets = list_models()
    model_ids = [asset["model_id"] for asset in assets]
    assert model_ids == list(dict.fromkeys(model_ids))
    required = required_model_ids(PipelineConfig.load())
    assert required == (
        "ctdeeprot_2d_v1",
        "spineps_veridah_ct_v1",
        "bodycomposition_resenc_l_v1",
        "totalsegmentator_total_task297_landmarks_v1",
    )


def test_internal_model_verification_separates_local_readiness_from_release_pin(
    tmp_path,
    monkeypatch,
):
    payload = b"model"
    model_id = "bodycomposition_resenc_l_v1"
    definition = {
        **INTERNAL_MODELS[model_id],
        "directory": "DatasetTest",
        "trainer": "trainer",
        "revision": None,
        "files": {
            "checkpoint.pth": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_size": len(payload),
            }
        },
    }
    monkeypatch.setitem(INTERNAL_MODELS, model_id, definition)
    checkpoint = tmp_path / "DatasetTest" / "trainer" / "checkpoint.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(payload)
    report = verify_model(model_id, tmp_path)
    assert report.ready
    assert report.release_issues == ("upstream_revision_unresolved",)
    config = PipelineConfig.model_validate(
        {
            "models": {"root": str(tmp_path)},
            "measurements": {"landmarks": {"enabled": False}},
        }
    )
    assert release_model_issues(config, (report,)) == (
        "bodycomposition_resenc_l_v1:upstream_revision_unresolved",
    )
    assert release_model_issues(
        config,
        (report,),
        include_all_selectable=True,
    ) == ("bodycomposition_resenc_l_v1:upstream_revision_unresolved",)
    assert sync_model(model_id, tmp_path).ready


def test_missing_spineps_bundle_does_not_report_unverified_files_as_checked(
    tmp_path,
    monkeypatch,
):
    from BodyComposition.vertebral.spineps_assets import AssetVerificationError

    monkeypatch.setattr(
        "BodyComposition.vertebral.spineps_assets.verify_model_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssetVerificationError("bundle is missing")),
    )

    report = verify_model("spineps_veridah_ct_v1", tmp_path)

    assert not report.ready
    assert report.checked_files == {}


def test_model_sync_preflights_unresolved_assets_before_any_download(
    tmp_path,
    monkeypatch,
):
    import BodyComposition.model_manager as manager

    calls = []
    monkeypatch.setitem(
        manager.INTERNAL_MODELS,
        "bodycomposition_resenc_l_v1",
        {
            **manager.INTERNAL_MODELS["bodycomposition_resenc_l_v1"],
            "revision": None,
        },
    )
    monkeypatch.setattr(manager, "sync_model", lambda *args: calls.append(args))
    config = PipelineConfig.model_validate(
        {
            "models": {"root": str(tmp_path)},
            "measurements": {"landmarks": {"enabled": False}},
        }
    )
    with pytest.raises(ModelAssetError, match="cannot start"):
        sync_models(
            config,
            ("ctdeeprot_2d_v1", "bodycomposition_resenc_l_v1"),
        )
    assert calls == []


def test_service_atomic_resume_aggregate_and_manifest_privacy(tmp_path):
    SuccessfulPipeline.instances = 0
    source = _write_ct(tmp_path / "MRN-123_DOB-1970.nii.gz")
    service = _service(tmp_path)
    first = service.analyze_case(
        source,
        tmp_path / "outputs",
        case_id="pseudonym-1",
        run_id="cohort-1",
    )
    assert first.execution_status == ExecutionStatus.SUCCEEDED
    assert first.qc_status == QCStatus.REVIEW
    assert (
        first.output_path
        == tmp_path / "outputs/runs/cohort-1/cases/pseudonym-1" / first.analysis_id
    )
    assert (first.output_path / "tables/slices.parquet").is_file()
    assert not any((tmp_path / "outputs/runs/cohort-1/.attempts").glob("*"))
    manifest_text = first.manifest_path.read_text(encoding="utf-8")
    assert "MRN-123" not in manifest_text
    assert str(tmp_path) not in manifest_text
    assert "<mounted-model-cache>" in manifest_text
    run = inspect_result(tmp_path / "outputs/runs/cohort-1")
    assert run.aggregate_paths["cases"].is_file()
    assert {
        "cases",
        "failures",
        "review_queue",
        "cases_csv",
        "failures_csv",
        "review_queue_csv",
    } == set(run.aggregate_paths)
    assert all(path.is_file() for path in run.aggregate_paths.values())
    review_queue = pd.read_parquet(run.aggregate_paths["review_queue"])
    review_queue_csv = pd.read_csv(run.aggregate_paths["review_queue_csv"])
    assert len(review_queue) == 1
    assert review_queue_csv.columns.tolist() == review_queue.columns.tolist()
    assert len(review_queue_csv) == len(review_queue)
    assert json.loads(review_queue.iloc[0]["observed_json"]) == {
        "native_label": 24,
        "orientation_changed": True,
        "support_labels": [23, 24],
    }
    assert json.loads(review_queue.iloc[0]["thresholds_json"]) == {
        "minimum_votes": 23,
    }
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["qc_flags"][0]["observed"]["native_label"] == 24
    assert manifest["qc_flags"][0]["thresholds"]["minimum_votes"] == 23
    json.dumps(first.as_dict(), allow_nan=False)
    reloaded_paths = service_module.aggregate_results(run.output_path)
    assert {
        "cases",
        "failures",
        "review_queue",
        "cases_csv",
        "failures_csv",
        "review_queue_csv",
    } == set(reloaded_paths)
    reloaded_queue = pd.read_parquet(reloaded_paths["review_queue"])
    assert json.loads(reloaded_queue.iloc[0]["observed_json"]) == {
        "native_label": 24,
        "orientation_changed": True,
        "support_labels": [23, 24],
    }

    second = service.analyze_case(
        source,
        tmp_path / "outputs",
        case_id="pseudonym-1",
        run_id="cohort-1",
    )
    assert second.execution_status == ExecutionStatus.SKIPPED_IDENTICAL
    assert SuccessfulPipeline.instances == 1


def test_batch_can_omit_csv_mirrors_without_omitting_parquet(tmp_path):
    source = _write_ct(tmp_path / "case.nii.gz")
    value = _config().normalized()
    value["output"]["save_csv_tables"] = False
    result = _service(
        tmp_path,
        config=PipelineConfig.model_validate(value),
    ).analyze_case(
        source,
        tmp_path / "outputs",
        case_id="case-1",
        run_id="without-csv",
    )

    run = inspect_result(result.output_path.parents[2])
    assert set(run.aggregate_paths) == {"cases", "failures", "review_queue"}
    assert all(path.suffix == ".parquet" for path in run.aggregate_paths.values())
    assert not list((run.output_path / "aggregate").glob("*.csv"))
    regenerated = service_module.aggregate_results(run.output_path)
    assert set(regenerated) == {"cases", "failures", "review_queue"}
    assert not list((run.output_path / "aggregate").glob("*.csv"))


def test_qc_evidence_json_normalization_is_strict_and_private(tmp_path):
    flag = {
        "observed": {np.int64(24): np.int64(2)},
        "thresholds": {"values": np.asarray([1.5, 2.5])},
    }
    normalized = service_module._normalise_flag(flag, default_stage="measurement")
    assert normalized["observed"] == {"24": 2}
    assert normalized["thresholds"] == {"values": [1.5, 2.5]}

    sensitive_path = tmp_path / "MRN-123"
    with pytest.raises(TypeError) as unsupported:
        service_module._normalise_flag(
            {"observed": {"source": sensitive_path}},
            default_stage="measurement",
        )
    assert str(sensitive_path) not in str(unsupported.value)

    with pytest.raises(ValueError, match="non-finite"):
        service_module._normalise_flag(
            {"observed": {"value": np.inf}},
            default_stage="measurement",
        )

    with pytest.raises(TypeError, match="collide"):
        service_module._normalise_flag(
            {"observed": {"1": "string", 1: "integer"}},
            default_stage="measurement",
        )


def test_case_qc_preserves_distinct_observations_and_error_severity():
    warning = {
        "code": "vertebral_body_extent_invalid",
        "stage": "measurement",
        "severity": "warning",
        "reason": "A detected vertebral-body extent is invalid or truncated.",
        "observed": {"vertebral_level": "T1", "missing_reason": "truncated_vertebra"},
    }
    error = {
        **warning,
        "severity": "error",
        "observed": {"vertebral_level": "SACRUM", "missing_reason": "invalid_extent"},
    }
    memory = {
        "tmp/measurement_bundle": SimpleNamespace(
            qc_status=SimpleNamespace(value="fail"),
            qc_flags=(warning, warning, error),
        )
    }

    status, flags = service_module._case_qc(memory)

    assert status == QCStatus.FAIL
    assert len(flags) == 2
    assert [flag["observed"]["vertebral_level"] for flag in flags] == ["T1", "SACRUM"]
    assert [flag["severity"] for flag in flags] == ["warning", "error"]


def test_case_qc_retains_information_without_requesting_manual_review():
    information = {
        "code": "vertebral_body_extent_truncated",
        "stage": "measurement",
        "severity": "info",
        "reason": "The observed territory reaches the acquisition boundary.",
        "observed": {"vertebral_level": "T1"},
    }
    memory = {
        "tmp/measurement_bundle": SimpleNamespace(
            qc_status=SimpleNamespace(value="pass"),
            qc_flags=(information,),
        )
    }

    status, flags = service_module._case_qc(memory)

    assert status == QCStatus.PASS
    assert flags == (service_module._normalise_flag(information, default_stage="measurement"),)


def test_information_remains_in_case_manifest_but_not_review_queue(tmp_path):
    source = _write_ct(tmp_path / "case.nii.gz")
    result = _service(tmp_path, pipeline=InformationalPipeline).analyze_case(
        source,
        tmp_path / "outputs",
        case_id="case-1",
        run_id="information-only",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    run = inspect_result(result.output_path.parents[2])
    review_queue = pd.read_parquet(run.aggregate_paths["review_queue"])

    assert result.qc_status == QCStatus.PASS
    assert manifest["manual_review_required"] is False
    assert manifest["qc_flags"][0]["severity"] == "info"
    assert review_queue.empty


def test_service_rejects_loaded_pixels_that_differ_from_preflight(
    tmp_path,
    monkeypatch,
):
    source = _write_ct(tmp_path / "case.nii.gz")

    class ChangedPixelContainer(service_module.NiftiDataContainer):
        def load_from_file(self):
            super().load_from_file()
            changed = self.data.copy()
            changed[0, 0, 0] += 1
            self.data = changed

    monkeypatch.setattr(service_module, "NiftiDataContainer", ChangedPixelContainer)
    result = _service(tmp_path).analyze_case(
        source,
        tmp_path / "outputs",
        case_id="case-1",
        run_id="input-integrity",
    )

    assert result.execution_status == ExecutionStatus.FAILED
    assert result.failure["code"] == "InputChangedError"


def test_service_rejects_nifti_file_mutated_after_preflight(tmp_path, monkeypatch):
    source = _write_ct(tmp_path / "case.nii.gz")
    service = _service(tmp_path)
    drain = service._drain_shared_queue

    def mutate_before_execution(**kwargs):
        _write_ct(source, value=1)
        return drain(**kwargs)

    monkeypatch.setattr(service, "_drain_shared_queue", mutate_before_execution)
    result = service.analyze_case(
        source,
        tmp_path / "outputs",
        case_id="case-1",
        run_id="mutated-input",
    )

    assert result.execution_status == ExecutionStatus.FAILED
    assert result.failure["code"] == "InputChangedError"


def test_result_inspection_rejects_artifact_tampering_and_path_traversal(tmp_path):
    source = _write_ct(tmp_path / "case.nii.gz")
    result = _service(tmp_path).analyze_case(
        source,
        tmp_path / "outputs",
        case_id="case-1",
        run_id="integrity",
    )
    artifact = result.output_path / "artifact.txt"
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact (size|digest) changed"):
        inspect_result(result.manifest_path)

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["relative_path"] = "../../outside.txt"
    payload["artifacts"][0]["byte_size"] = 0
    payload["artifacts"][0]["sha256"] = hashlib.sha256(b"").hexdigest()
    result.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="escapes"):
        inspect_result(result.manifest_path)


def test_batch_records_bad_input_without_dropping_good_case(tmp_path):
    good = _write_ct(tmp_path / "good.nii.gz")
    result = _service(tmp_path).analyze_batch(
        [
            CaseInput(tmp_path / "missing.nii.gz", "bad"),
            CaseInput(good, "good"),
        ],
        tmp_path / "outputs",
        run_id="run-1",
    )
    assert [case.execution_status for case in result.cases] == [
        ExecutionStatus.FAILED,
        ExecutionStatus.SUCCEEDED,
    ]
    failures = pd.read_parquet(result.aggregate_paths["failures"])
    assert failures["case_id"].tolist() == ["bad"]


def test_fail_fast_records_unstarted_cases_as_cancelled(tmp_path):
    first = _write_ct(tmp_path / "first.nii.gz", value=1)
    second = _write_ct(tmp_path / "second.nii.gz", value=2)
    config = PipelineConfig.model_validate(
        {
            "measurements": {"landmarks": {"enabled": False}},
            "runtime": {"allow_dirty": True, "fail_fast": True},
        }
    )
    service = PipelineService(
        config,
        pipeline_factory=FailingPipeline,
        model_provider=lambda value: (_model_status(tmp_path),),
        source_provider=lambda: SOURCE_CLEAN,
    )
    result = service.analyze_batch(
        [CaseInput(first, "first"), CaseInput(second, "second")],
        tmp_path / "outputs",
        run_id="fail-fast",
    )

    assert [case.execution_status for case in result.cases] == [
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
    ]
    assert result.cases[1].failure["code"] == "not_started_after_stop"


def test_failed_attempt_is_not_promoted_and_error_text_is_sanitized(tmp_path):
    source = _write_ct(tmp_path / "private-name.nii.gz")
    result = _service(tmp_path, FailingPipeline).analyze_case(
        source,
        tmp_path / "outputs",
        case_id="case-1",
        run_id="run-1",
    )
    assert result.execution_status == ExecutionStatus.FAILED
    assert "failed/case-1" in result.output_path.as_posix()
    assert not (tmp_path / "outputs/runs/run-1/cases/case-1" / result.analysis_id).exists()
    text = result.manifest_path.read_text(encoding="utf-8")
    assert "private path" not in text
    assert "private-name" not in text


def test_cancellation_produces_complete_ordered_run_manifest(tmp_path):
    first = _write_ct(tmp_path / "one.nii.gz", value=1)
    second = _write_ct(tmp_path / "two.nii.gz", value=2)
    result = _service(tmp_path, InterruptedPipeline).analyze_batch(
        [CaseInput(first, "one"), CaseInput(second, "two")],
        tmp_path / "outputs",
        run_id="cancelled-run",
    )
    assert [case.execution_status for case in result.cases] == [
        ExecutionStatus.CANCELLED,
        ExecutionStatus.CANCELLED,
    ]
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["execution_status"] == "cancelled"
    assert [case["case_id"] for case in payload["cases"]] == ["one", "two"]


def test_dirty_source_requires_explicit_development_flag(tmp_path):
    dirty = {**SOURCE_CLEAN, "source_dirty": True}
    with pytest.raises(DirtySourceError):
        PipelineService(
            PipelineConfig.load(),
            pipeline_factory=SuccessfulPipeline,
            model_provider=lambda value: (_model_status(tmp_path),),
            source_provider=lambda: dirty,
        )
    service = _service(tmp_path, source=dirty)
    assert service.config.allow_dirty


def test_run_id_cannot_be_reused_for_a_different_plan(tmp_path):
    first = _write_ct(tmp_path / "first.nii.gz", value=1)
    second = _write_ct(tmp_path / "second.nii.gz", value=2)
    service = _service(tmp_path)
    service.analyze_case(first, tmp_path / "outputs", case_id="case", run_id="same-run")
    with pytest.raises(ExistingRunError):
        service.analyze_case(second, tmp_path / "outputs", case_id="case", run_id="same-run")


def test_batch_manifest_is_explicit_ordered_and_strict(tmp_path):
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "cases": [
                    {"case_id": "b", "input_path": "b.nii.gz"},
                    {
                        "case_id": "a",
                        "input_path": "dicom/a",
                        "series_uid": "1.2.826.0.1.3680043.10.999.1",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cases = load_batch_manifest(manifest)
    assert [case.case_id for case in cases] == ["b", "a"]
    assert cases[0].input_path == tmp_path / "b.nii.gz"
    assert cases[1].input_path == tmp_path / "dicom/a"
    assert cases[0].series_uid is None
    assert cases[1].series_uid == "1.2.826.0.1.3680043.10.999.1"


def test_cli_analyze_is_a_thin_json_adapter(monkeypatch, capsys, tmp_path):
    expected = CaseResult(
        case_id="case-1",
        run_id="run-1",
        analysis_id="a" * 64,
        attempt_id="attempt-1",
        execution_status=ExecutionStatus.SUCCEEDED,
        qc_status=QCStatus.PASS,
        output_path=tmp_path / "case",
        manifest_path=tmp_path / "case/case_manifest.json",
    )
    calls = {}

    class ServiceStub:
        def __init__(self, config):
            calls["config"] = config

        def analyze_case(self, input_path, output_root, **kwargs):
            calls.update({"input": input_path, "output": output_root, **kwargs})
            return expected

    monkeypatch.setattr(cli, "PipelineService", ServiceStub)
    code = cli.main(
        [
            "analyze",
            str(tmp_path / "input.nii.gz"),
            "--output",
            str(tmp_path / "output"),
            "--case-id",
            "case-1",
            "--run-id",
            "run-1",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload == json.loads(json.dumps(expected.as_dict(), default=str))
    assert isinstance(calls["config"], PipelineConfig)
    assert calls["config"].normalized()["output"]["save_csv_tables"]
    assert calls["case_id"] == "case-1"
    assert calls["series_uid"] is None

    assert (
        cli.main(
            [
                "analyze",
                str(tmp_path / "input.nii.gz"),
                "--no-csv",
                "--json",
            ]
        )
        == cli.EXIT_OK
    )
    capsys.readouterr()
    assert not calls["config"].normalized()["output"]["save_csv_tables"]


def test_cli_common_paths_use_positional_inputs_and_safe_defaults():
    case = cli.build_parser().parse_args(["analyze", "scan.nii.gz"])
    batch = cli.build_parser().parse_args(["batch", "cohort.json"])
    convert = cli.build_parser().parse_args(["convert", "dicom", "scan.nii.gz"])

    assert case.input == Path("scan.nii.gz")
    assert case.output == service_module.DEFAULT_OUTPUT_ROOT
    assert case.config is None
    assert case.device is None
    assert case.case_id is None
    assert case.series_uid is None
    assert not case.no_csv
    assert batch.manifest == Path("cohort.json")
    assert batch.output == service_module.DEFAULT_OUTPUT_ROOT
    assert not batch.worker
    assert not batch.no_csv
    assert convert.input == Path("dicom")
    assert convert.output == Path("scan.nii.gz")
    assert convert.series_uid is None
    assert not convert.overwrite


def test_cli_convert_is_a_thin_adapter(monkeypatch, capsys, tmp_path):
    calls = {}
    output = tmp_path / "ct.nii.gz"
    metadata = tmp_path / "ct.bodycomposition.json"
    expected = {
        "output_path": output,
        "metadata_path": metadata,
        "series_instance_uid": "1.2.3",
        "input_format": "dicom",
    }

    def fake_convert(input_path, output_path, **kwargs):
        calls.update({"input": input_path, "output": output_path, **kwargs})
        return SimpleNamespace(
            output_path=output,
            metadata_path=metadata,
            as_dict=lambda: expected,
        )

    monkeypatch.setattr(cli, "convert_dicom", fake_convert)
    code = cli.main(
        [
            "convert",
            str(tmp_path / "dicom"),
            str(output),
            "--series",
            "1.2.3",
            "--overwrite",
            "--json",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(expected, default=str))
    assert calls == {
        "input": tmp_path / "dicom",
        "output": output,
        "series_uid": "1.2.3",
        "overwrite": True,
    }


def test_python_convenience_api_needs_only_an_input(monkeypatch):
    expected = object()
    calls = {}

    class ServiceStub:
        def __init__(self, config=None):
            calls["config"] = config

        def analyze_case(self, input_path, output_root, **kwargs):
            calls.update({"input": input_path, "output": output_root, **kwargs})
            return expected

    monkeypatch.setattr(service_module, "PipelineService", ServiceStub)

    result = service_module.analyze_case("scan.nii.gz")

    assert result is expected
    assert calls == {
        "config": None,
        "input": "scan.nii.gz",
        "output": service_module.DEFAULT_OUTPUT_ROOT,
        "case_id": None,
        "run_id": None,
        "series_uid": None,
    }


def test_doctor_defaults_to_operational_gate_and_offers_release_gate(
    monkeypatch,
    capsys,
    tmp_path,
):
    ready_model = _model_status(tmp_path)
    monkeypatch.setattr(cli, "verify_models", lambda config: (ready_model,))
    monkeypatch.setattr(cli, "release_model_issues", lambda *args, **kwargs: ("pin",))
    monkeypatch.setattr(cli, "source_state", lambda: SOURCE_CLEAN)
    monkeypatch.setattr(cli, "package_lock_digest", lambda: "a" * 64)
    monkeypatch.setattr(cli, "_distribution_has_notices", lambda: True)

    assert cli.main(["doctor", "-o", str(tmp_path), "--json"]) == 0
    routine = json.loads(capsys.readouterr().out)
    assert routine["operational_ready"]
    assert not routine["release_ready"]

    assert cli.main(["doctor", "-o", str(tmp_path), "--release", "--json"]) == 3
    release = json.loads(capsys.readouterr().out)
    assert release["operational_ready"]
    assert not release["release_ready"]


def test_cli_exposes_only_the_one_release_command_tree():
    help_text = cli.build_parser().format_help()
    for command in (
        "analyze",
        "convert",
        "batch",
        "models",
        "config",
        "results",
        "reports",
        "doctor",
        "version",
    ):
        assert command in help_text
    assert "BodyCompositionFast" not in help_text


def test_reporting_enabled_batch_collates_current_run_order(
    tmp_path,
    monkeypatch,
):
    import BodyComposition.reporting.service as reporting_service

    first = _write_ct(tmp_path / "first.nii.gz", value=1)
    second = _write_ct(tmp_path / "second.nii.gz", value=2)
    config_data = _config().normalized()
    config_data["reporting"]["enabled"] = True
    config = PipelineConfig.model_validate(config_data)
    calls = []
    snapshots = []
    cover_summaries = []

    def load_report(path):
        path = Path(path)
        calls.append(path)
        case_id = path.parents[2].name
        return CaseReportResult(
            case_id=case_id,
            report_id=hashlib.sha256(case_id.encode()).hexdigest(),
            layout="spine_overview_v1",
            status="succeeded",
            pdf_path=tmp_path / "individual.pdf",
            manifest_path=Path(path),
            pdf_sha256="e" * 64,
            manual_review_required=False,
        )

    def collate(reports, *, export_id, output_directory, settings, run_summary):
        snapshots.append([report.case_id for report in reports])
        cover_summaries.append(dict(run_summary))
        assert export_id == "report-run"
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        pdf = output / "case_reports.pdf"
        manifest = output / "report_manifest.json"
        pdf.write_bytes(b"pdf")
        manifest.write_text("{}", encoding="utf-8")
        return {"pdf_path": pdf, "manifest_path": manifest, "manifest": {}}

    monkeypatch.setattr(reporting_service, "load_case_report_result", load_report)
    monkeypatch.setattr(reporting_service, "collate_reports", collate)
    result = _service(tmp_path, config=config).analyze_batch(
        [CaseInput(first, "first"), CaseInput(second, "second")],
        tmp_path / "outputs",
        run_id="report-run",
    )

    run = inspect_result(tmp_path / "outputs/runs/report-run")
    assert result.execution_status == ExecutionStatus.SUCCEEDED
    assert run.execution_status == ExecutionStatus.SUCCEEDED
    assert run.reporting["status"] == "succeeded"
    assert run.reporting["combined_pdf"].endswith("case_reports.pdf")
    assert snapshots == [
        ["first"],
        ["first", "second"],
        ["first", "second"],
    ]
    assert [summary["document_state"] for summary in cover_summaries] == [
        "in_progress",
        "complete",
        "complete",
    ]
    assert [summary["included_case_count"] for summary in cover_summaries] == [1, 2, 2]
    assert all(
        summary["paths"]["input_directories"] == [tmp_path.name]
        for summary in cover_summaries
    )
    assert all(
        summary["paths"]["output_directory"] == "report-run"
        for summary in cover_summaries
    )
    assert len(calls) == 5
