from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
import SimpleITK as sitk

from BodyComposition.actions.vertebral import (
    SPINEPS_BODY_MASK,
    SPINEPS_RESULT_JSON,
    SPINEPS_WHOLE_MASK,
    SegmSpinepsVeridah,
    vertebral_backend_actions,
)
from BodyComposition.orientation.core import OrientationOutcome, ReviewFlag
from BodyComposition.pipelines.bodycomposition import canonical_actions
from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.utils.nifti import NiftiDataContainer
from BodyComposition.vertebral import spineps_assets
from BodyComposition.vertebral.contracts import ExecutionStatus, VertebralResult
from BodyComposition.vertebral.internal_adapter import adapt_internal_vertebral_bodies
from BodyComposition.vertebral.review import write_spine_review
from BodyComposition.vertebral.spineps_assets import AssetVerificationError
from BodyComposition.vertebral.spineps_backend import SpinepsVeridahAdapter
from BodyComposition.vertebral.spineps_manifest import ModelAssetSpec, ReleaseAssetPin


def _zip_bytes(members: dict[str, bytes | str]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return output.getvalue()


def _model_pin(model_id: str, phase: str, archive: bytes) -> ModelAssetSpec:
    import hashlib

    return ModelAssetSpec(
        model_id=model_id,
        phase=phase,
        release="vtest",
        asset_name=f"{model_id}.zip",
        url=(
            "https://github.com/Hendrik-code/spineps/releases/"
            f"download/vtest/{model_id}.zip"
        ),
        bytes=len(archive),
        sha256=hashlib.sha256(archive).hexdigest(),
        install_dir=model_id,
    )


def _release_pin(name: str, archive: bytes) -> ReleaseAssetPin:
    import hashlib

    return ReleaseAssetPin(
        asset_name=name,
        url=(
            "https://github.com/robert-graf/VIBESegmentator/"
            f"releases/download/vtest/{name}"
        ),
        bytes=len(archive),
        sha256=hashlib.sha256(archive).hexdigest(),
    )


def test_sync_models_downloads_six_pins_once_and_is_idempotent(tmp_path, monkeypatch):
    model_archives = {
        phase: _zip_bytes(
            {
                "model/inference_config.json": "{}",
                "model/fold_0/checkpoint_final.pth": phase,
            }
        )
        for phase in ("semantic", "instance", "labeling")
    }
    model_pins = tuple(
        _model_pin(model_id, phase, model_archives[phase])
        for model_id, phase in (
            ("ct", "semantic"),
            ("ct_instance", "instance"),
            ("ct_labeling", "labeling"),
        )
    )
    vibeseg_archives = {
        "100.zip": _zip_bytes(
            {
                "Dataset100/Trainer/dataset.json": "{}",
                "Dataset100/Trainer/plans.json": "{}",
                "Dataset100/Trainer/fold_0/checkpoint_final.pth": "fold0",
            }
        ),
        "100_1.zip": _zip_bytes(
            {"Dataset100/Trainer/fold_1/checkpoint_final.pth": "fold1"}
        ),
        "100_2.zip": _zip_bytes(
            {"Dataset100/Trainer/fold_2/checkpoint_final.pth": "fold2"}
        ),
    }
    vibeseg_pins = tuple(
        _release_pin(name, archive) for name, archive in vibeseg_archives.items()
    )
    monkeypatch.setattr(spineps_assets, "SPINEPS_MODEL_ASSETS", model_pins)
    monkeypatch.setattr(spineps_assets, "VIBESEG_CROP_ASSETS", vibeseg_pins)
    payloads = {
        **{pin.url: model_archives[pin.phase] for pin in model_pins},
        **{pin.url: vibeseg_archives[pin.asset_name] for pin in vibeseg_pins},
    }
    requests: list[str] = []

    def opener(request):
        requests.append(request.full_url)
        return io.BytesIO(payloads[request.full_url])

    first = spineps_assets.sync_models(tmp_path / "models", opener=opener)
    second = spineps_assets.sync_models(
        tmp_path / "models",
        opener=lambda request: (_ for _ in ()).throw(AssertionError("must not download")),
    )

    assert len(requests) == 6
    assert set(first.downloaded_assets) == {
        "ct.zip",
        "ct_instance.zip",
        "ct_labeling.zip",
        "100.zip",
        "100_1.zip",
        "100_2.zip",
    }
    assert second.downloaded_assets == ()
    assert second.vibeseg_trained_model.name == "Trainer"


def test_check_reports_missing_bundle_without_creating_or_downloading(
    tmp_path,
    monkeypatch,
):
    model_root = tmp_path / "missing-model-bundle"

    def missing_bundle(*args, **kwargs):
        raise AssetVerificationError("pinned bundle is absent")

    monkeypatch.setattr(
        "BodyComposition.vertebral.spineps_backend.verify_model_bundle",
        missing_bundle,
    )
    monkeypatch.setattr(
        "BodyComposition.vertebral.spineps_backend.sync_pinned_models",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("check must never synchronize models")
        ),
    )

    report = SpinepsVeridahAdapter(model_root, device="cpu").check(
        output_root=tmp_path,
    )

    model_check = next(item for item in report.checks if item.component == "model_bundle")
    assert not report.ready
    assert model_check.ready is False
    assert model_check.remediation == (
        "bodycomposition models sync --model spineps_veridah_ct_v1"
    )
    assert not model_root.exists()


def test_check_accepts_a_verified_read_only_model_bundle(tmp_path, monkeypatch):
    model_root = tmp_path / "read-only-models"
    model_root.mkdir()
    model_root.chmod(0o555)
    monkeypatch.setattr(
        "BodyComposition.vertebral.spineps_backend.verify_model_bundle",
        lambda *args, **kwargs: object(),
    )
    try:
        report = SpinepsVeridahAdapter(model_root, device="cpu").check(
            output_root=tmp_path,
        )
    finally:
        model_root.chmod(0o755)

    assert report.ready
    assert "model_cache" not in {item.component for item in report.checks}


def _image(shape=(7, 9, 11)) -> sitk.Image:
    array = np.linspace(-1000, 500, np.prod(shape), dtype=np.int16).reshape(shape)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.7, 1.1, 2.5))
    image.SetOrigin((12.0, -5.0, 33.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    return image


def _prepared(image: sitk.Image, *, changed: bool = False) -> OrientationOutcome:
    result = SimpleNamespace(
        state=SimpleNamespace(value="MISMATCH_REPAIRED" if changed else "PASS_METADATA_MATCH"),
        orientation_changed=changed,
        manual_review_required=changed,
        original_orientation_code="RIP",
        prepared_orientation_code="LPS",
        prepared_pixel_sha256="prepared-digest",
        review_flags=(
            ReviewFlag(
                code="severe_misorientation",
                severity="high",
                reason="Orientation required repair.",
            ),
        )
        if changed
        else (),
        to_dict=lambda: {
            "state": "MISMATCH_REPAIRED" if changed else "PASS_METADATA_MATCH",
            "orientation_changed": changed,
            "manual_review_required": changed,
            "prepared_pixel_sha256": "prepared-digest",
        },
    )
    return OrientationOutcome(result=result, prepared_image=image)


def _geometry(image: sitk.Image) -> ImageGeometry:
    return ImageGeometry(
        size_xyz=image.GetSize(),
        spacing_xyz=image.GetSpacing(),
        origin_lps_xyz=image.GetOrigin(),
        direction_lps=image.GetDirection(),
    )


def test_adapter_run_uses_orientation_outcome_and_carries_review_state(
    tmp_path, monkeypatch
):
    image = _image()
    prepared = _prepared(image, changed=True)
    geometry = _geometry(image)
    body = np.zeros(tuple(reversed(image.GetSize())), dtype=np.uint8)
    body[2:5, 2:7, 3:8] = 22
    base_result = VertebralResult(
        backend_id="spineps_veridah_ct_v1",
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=geometry,
        whole_vertebra_labels=body.copy(),
        vertebral_body_labels=body,
        label_schema={22: "L3"},
    )
    calls = []

    class Runtime:
        def run(self, received, *, attempt_directory, case_id):
            calls.append((received, attempt_directory, case_id))
            return base_result

    monkeypatch.setattr(
        "BodyComposition.vertebral.spineps_backend.verify_model_bundle",
        lambda *args, **kwargs: object(),
    )
    adapter = SpinepsVeridahAdapter(
        tmp_path / "models",
        device="cpu",
        runtime_factory=lambda root, use_cpu: Runtime(),
    )
    result = adapter.run(
        prepared,
        attempt_directory=tmp_path / "attempt",
        case_id="case001",
    )

    assert calls == [(prepared, tmp_path / "attempt", "case001")]
    assert result.provenance["orientation_changed"] is True
    assert result.provenance["orientation_manual_review_required"] is True
    assert "orientation_severe_misorientation" in {flag.code for flag in result.qc_flags}


def test_adapter_returns_path_free_failed_result_for_runtime_exception(
    tmp_path,
    monkeypatch,
    caplog,
):
    prepared = _prepared(_image())

    class Runtime:
        def run(self, *args, **kwargs):
            raise RuntimeError("failure at /private/sensitive/model/path")

    monkeypatch.setattr(
        "BodyComposition.vertebral.spineps_backend.verify_model_bundle",
        lambda *args, **kwargs: object(),
    )
    adapter = SpinepsVeridahAdapter(
        tmp_path / "models",
        device="cpu",
        runtime_factory=lambda root, use_cpu: Runtime(),
    )

    result = adapter.run(
        prepared,
        attempt_directory=tmp_path / "attempt",
        case_id="case001",
    )
    serialized = str(result.summary())

    assert result.execution_status == ExecutionStatus.FAILED
    assert {flag.code for flag in result.qc_flags} == {
        "upstream_inference_exception"
    }
    assert "RuntimeError" in serialized
    assert "/private/sensitive" not in serialized
    assert "/private/sensitive/model/path" in caplog.text


def test_adapter_returns_path_free_failed_result_when_backend_is_not_ready(
    tmp_path,
    monkeypatch,
):
    prepared = _prepared(_image())

    def invalid_bundle(*args, **kwargs):
        raise AssetVerificationError("invalid file /private/sensitive/model/path")

    monkeypatch.setattr(
        "BodyComposition.vertebral.spineps_backend.verify_model_bundle",
        invalid_bundle,
    )
    adapter = SpinepsVeridahAdapter(
        tmp_path / "models",
        device="cpu",
        runtime_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("an unready backend must not initialize a runtime")
        ),
    )

    result = adapter.run(
        prepared,
        attempt_directory=tmp_path / "attempt",
        case_id="case001",
    )
    serialized = str(result.summary())

    assert result.execution_status == ExecutionStatus.FAILED
    assert {flag.code for flag in result.qc_flags} == {"backend_not_ready"}
    assert "model_bundle" in serialized
    assert "/private/sensitive" not in serialized


def test_internal_adapter_allows_body_only_result():
    image = _image((4, 5, 6))
    geometry = _geometry(image)
    labels = np.zeros((4, 5, 6), dtype=np.uint8)
    labels[1:3, 1:4, 2:5] = 15

    result = adapt_internal_vertebral_bodies(
        labels,
        geometry,
        {15: "L3"},
        model_preset="ResEncL",
    )

    assert result.execution_status == ExecutionStatus.SUCCEEDED
    assert result.whole_vertebra_labels is None
    assert result.vertebral_body_labels[1, 1, 2] == 15
    assert result.centroids[0].anatomical_label == "L3"


def test_review_image_contains_both_views_and_review_annotation(tmp_path):
    image = _image()
    prepared = _prepared(image, changed=True)
    geometry = _geometry(image)
    labels = np.zeros(tuple(reversed(image.GetSize())), dtype=np.uint8)
    labels[1:3, 2:7, 2:8] = 21
    labels[4:6, 2:7, 2:8] = 22
    result = VertebralResult(
        backend_id="spineps_veridah_ct_v1",
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=geometry,
        whole_vertebra_labels=labels.copy(),
        vertebral_body_labels=labels,
        label_schema={21: "L2", 22: "L3"},
    )

    output = write_spine_review(
        prepared,
        result,
        tmp_path / "spine_review.png",
        case_id="case001",
    )
    rendered = cv2.imread(str(output))

    assert rendered.shape == (1000, 1800, 3)
    assert np.std(rendered[:, :1200]) > 5
    assert np.std(rendered[:, 1220:]) > 5


def test_pipeline_action_promotes_validated_body_mask_and_summary(
    pipeline_stub,
    tmp_path,
):
    pipeline_stub.config["vertebrae"]["spineps"]["save_native_outputs"] = False
    pipeline_stub.config["vertebrae"]["spineps"]["review_enabled"] = False
    image = _image((4, 5, 6))
    prepared = _prepared(image)
    geometry = _geometry(image)
    labels = np.zeros((4, 5, 6), dtype=np.uint8)
    labels[1:3, 1:4, 2:5] = 22
    result = VertebralResult(
        backend_id="spineps_veridah_ct_v1",
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=geometry,
        whole_vertebra_labels=labels.copy(),
        vertebral_body_labels=labels,
        label_schema={22: "L3"},
    )
    action = SegmSpinepsVeridah(pipeline_stub)
    action.adapter = SimpleNamespace(run=lambda *args, **kwargs: result)
    memory = {
        "id": "case001",
        "workspace": tmp_path,
        "tmp/prepared_image": prepared,
    }

    action(memory)
    action.validate_outputs(memory)

    mask_path = tmp_path / SPINEPS_BODY_MASK.format(caseid="case001")
    summary_path = tmp_path / SPINEPS_RESULT_JSON.format(caseid="case001")
    stored = NiftiDataContainer(mask_path)
    stored.load_from_file()
    assert stored.geometry.equivalent_to(geometry)
    assert np.array_equal(stored.data, labels)
    assert summary_path.is_file()
    assert memory["tmp/vertebral_result"].backend_id == "spineps_veridah_ct_v1"
    assert not any((tmp_path / ".attempts").rglob("*.nii.gz"))


def test_pipeline_resume_revalidates_corpus_intersection_against_prepared_ct(
    pipeline_stub,
    tmp_path,
):
    pipeline_stub.config["run"]["skip"] = True
    pipeline_stub.config["vertebrae"]["spineps"]["save_native_outputs"] = True
    pipeline_stub.config["vertebrae"]["spineps"]["review_enabled"] = False
    image = _image((4, 5, 6))
    prepared = _prepared(image)
    geometry = _geometry(image)
    whole = np.zeros((4, 5, 6), dtype=np.uint8)
    whole[1:3, 1:4, 1:5] = 22
    body = np.zeros_like(whole)
    body[1:3, 2:4, 2:4] = 22
    semantic = np.zeros_like(whole)
    semantic[body != 0] = 49
    semantic_path = tmp_path / "upstream" / "semantic.nii.gz"
    semantic_path.parent.mkdir(parents=True)
    semantic_image = sitk.GetImageFromArray(semantic)
    semantic_image.CopyInformation(image)
    sitk.WriteImage(semantic_image, str(semantic_path))
    result = VertebralResult(
        backend_id="spineps_veridah_ct_v1",
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=geometry,
        whole_vertebra_labels=whole,
        vertebral_body_labels=body,
        label_schema={22: "L3"},
        native_outputs={"out_spine": semantic_path},
        provenance={"orientation": prepared.result.to_dict()},
    )
    action = SegmSpinepsVeridah(pipeline_stub)
    calls = []

    def run_once(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) > 1:
            raise AssertionError("identical validated outputs must be reused")
        return result

    action.adapter = SimpleNamespace(run=run_once)
    memory = {
        "id": "case001",
        "workspace": tmp_path,
        "tmp/prepared_image": prepared,
    }

    action(memory)
    action(memory)

    assert len(calls) == 1
    assert memory["tmp/vertebral_result"].execution_status == ExecutionStatus.SKIPPED_IDENTICAL
    assert np.array_equal(memory[SPINEPS_BODY_MASK].data, body)
    assert not np.array_equal(memory[SPINEPS_BODY_MASK].data, whole)


def test_canonical_pipeline_routes_only_vertebral_bodies_downstream(
    pipeline_stub,
):
    actions = canonical_actions(pipeline_stub)
    downstream_keys = {
        value
        for action in actions[1:]
        for attribute in (
            "input_label_name",
            "input_mask_name",
            "input_name",
            "vertebral_body_source_name",
        )
        if (value := getattr(action, attribute, None))
        in {SPINEPS_BODY_MASK, SPINEPS_WHOLE_MASK}
    }

    assert downstream_keys == {SPINEPS_BODY_MASK}


def test_backend_selection_never_falls_back(pipeline_stub):
    pipeline_stub.config["vertebrae"]["backend"] = "unknown"

    try:
        vertebral_backend_actions(pipeline_stub)
    except ValueError as error:
        assert "No alternative backend was attempted" in str(error)
    else:
        raise AssertionError("Unknown backends must fail explicitly.")
