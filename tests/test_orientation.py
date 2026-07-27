from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
import SimpleITK as sitk
from jsonschema import Draft202012Validator
from skimage.transform import resize

from BodyComposition.actions.orientation import AssessOrientation
from BodyComposition.orientation.core import (
    OrientationIntegrityError,
    OrientationState,
    apply_model_rotation_to_sitk,
    assess_orientation,
    write_orientation_failure_artifacts,
)
from BodyComposition.orientation.ctdeeprot import (
    CHECKPOINT_BYTES,
    CHECKPOINT_SHA256,
    CHECKPOINT_URL,
    CODE_LICENSE,
    DEFAULT_CHECKPOINT_PATH,
    LPS_YXZ_REFERENCE_CLASS_INDEX,
    MODEL_CITATION_DOI,
    REDISTRIBUTION_MODE,
    UPSTREAM_REPOSITORY,
    CTDeepRotPrediction,
    CTDeepRotPredictor,
    ModelAssetError,
    model_asset_record,
    projection_feature_groups,
    sync_checkpoint,
    verify_checkpoint,
)
from BodyComposition.orientation.rotations import (
    ROTATION_CLASSES,
    apply_volume_rotation,
    compose_class_indices,
    identity_class_index,
    inverse_class_index,
    is_axial_half_turn,
    signed_permutation_matrix,
)
from BodyComposition.pipeline import PipelineAction
from BodyComposition.utils.nifti import NiftiDataContainer

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "BodyComposition/schemas/orientation_report.schema.json"
)


class FakePredictor:
    def __init__(self, class_index: int, agreement_count: int = 24):
        self.class_index = class_index
        self.agreement_count = agreement_count

    def predict(self, image: sitk.Image) -> CTDeepRotPrediction:
        runner_up = (self.class_index + 1) % 24
        recovered = [self.class_index] * self.agreement_count
        recovered.extend([runner_up] * (24 - self.agreement_count))
        counts = [0] * 24
        for value in recovered:
            counts[value] += 1
        return CTDeepRotPrediction(
            winning_class_index=self.class_index,
            winning_angles_deg=ROTATION_CLASSES[self.class_index].angles_deg,
            agreement_count=self.agreement_count,
            vote_counts=tuple(counts),
            recovered_votes=tuple(recovered),
            augmented_predictions=tuple(recovered),
            vote_entropy=0.0 if self.agreement_count == 24 else 0.5,
            vote_margin=max(0.0, (2 * self.agreement_count - 24) / 24),
            mean_support_probability=0.99,
            checkpoint_sha256=CHECKPOINT_SHA256,
        )


def _ct_image(
    *,
    shape_zyx=(300, 20, 20),
    spacing_xyz=(1.0, 1.0, 1.0),
    direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
) -> sitk.Image:
    array = np.zeros(shape_zyx, dtype=np.int16)
    array[0, 0, 0] = 10
    array[-1, -1, -1] = 20
    array[shape_zyx[0] // 3, 3, 7] = 30
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing_xyz)
    image.SetOrigin((12.5, -8.0, 30.0))
    image.SetDirection(direction_lps)
    return image


def _config(base_config, **orientation_updates):
    config = deepcopy(base_config)
    for key, value in orientation_updates.items():
        section, setting = key.split("__", 1)
        config["orientation"][section][setting] = value
    return config


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_ctdeeprot_rotation_group_contains_exactly_24_proper_rotations():
    probe = np.arange(2 * 3 * 5, dtype=np.int16).reshape(2, 3, 5)
    transformed = {
        (apply_volume_rotation(probe, index).shape, apply_volume_rotation(probe, index).tobytes())
        for index in range(24)
    }

    assert len(ROTATION_CLASSES) == 24
    assert len(transformed) == 24
    assert identity_class_index() == 6
    for index in range(24):
        inverse = inverse_class_index(index)
        assert compose_class_indices(inverse, index) == identity_class_index()
        assert compose_class_indices(index, inverse) == identity_class_index()
        assert np.linalg.det(signed_permutation_matrix(index)) == pytest.approx(1.0)
        assert np.array_equal(
            apply_volume_rotation(apply_volume_rotation(probe, index), inverse),
            probe,
        )
    assert [index for index in range(24) if is_axial_half_turn(index)] == [4]


def test_ctdeeprot_lps_reference_calibration_recovers_relative_rotation():
    reference_inverse = inverse_class_index(LPS_YXZ_REFERENCE_CLASS_INDEX)

    for applied_class in range(24):
        raw_prediction = compose_class_indices(
            applied_class,
            LPS_YXZ_REFERENCE_CLASS_INDEX,
        )
        relative_prediction = compose_class_indices(
            raw_prediction,
            reference_inverse,
        )
        assert relative_prediction == applied_class


def test_ctdeeprot_projection_preprocessing_matches_pinned_upstream_order():
    rng = np.random.default_rng(20260715)
    data = rng.uniform(0, 3500, size=(31, 25, 17)).astype(np.float32)
    expected_groups = []
    for function, window in (
        (lambda values, axis: np.mean(values, axis=axis), (50.0, 1800.0)),
        (lambda values, axis: np.max(values, axis=axis), (100.0, 3200.0)),
        (lambda values, axis: np.std(values, axis=axis, ddof=1), (0.0, 1000.0)),
    ):
        channels = []
        for axis in range(3):
            projected = function(data, axis)
            normalized = (
                np.clip(projected, window[0], window[1]) - window[0]
            ) / (window[1] - window[0])
            resized = resize(normalized, (224, 224), order=3)
            channels.append((resized * 255.0).astype(np.uint8))
        expected_groups.append(np.stack(channels, axis=2))

    observed_groups = projection_feature_groups(data)
    assert len(expected_groups) == len(observed_groups)
    for expected, observed in zip(expected_groups, observed_groups, strict=True):
        observed_uint8 = np.rint((observed + 0.5) * 255.0).astype(np.uint8)
        assert np.array_equal(observed_uint8, expected)


@pytest.mark.parametrize("inverse", [False, True])
def test_every_model_rotation_matches_simpleitk_xyz_transform(inverse):
    array_zyx = np.arange(5 * 2 * 3, dtype=np.int16).reshape(5, 2, 3)
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing((1.1, 2.2, 3.3))
    image.SetOrigin((4.0, 5.0, 6.0))
    model_array_yxz = np.transpose(array_zyx, (1, 2, 0))

    for class_index in range(24):
        selected = inverse_class_index(class_index) if inverse else class_index
        expected_yxz = apply_volume_rotation(model_array_yxz, selected)
        transformed = apply_model_rotation_to_sitk(
            image,
            class_index,
            inverse=inverse,
        )
        observed_yxz = np.transpose(sitk.GetArrayFromImage(transformed), (1, 2, 0))
        assert np.array_equal(observed_yxz, expected_yxz), class_index


def test_simpleitk_rotations_roundtrip_array_and_physical_geometry():
    image = _ct_image(shape_zyx=(7, 9, 11), spacing_xyz=(0.7, 1.2, 3.4))
    center = np.asarray(
        image.TransformContinuousIndexToPhysicalPoint(
            tuple((length - 1) / 2 for length in image.GetSize())
        )
    )
    for class_index in range(24):
        rotated = apply_model_rotation_to_sitk(
            image,
            class_index,
            inverse=True,
            target_direction_lps=image.GetDirection(),
            target_center_lps=center,
        )
        restored = apply_model_rotation_to_sitk(
            rotated,
            class_index,
            inverse=False,
            target_direction_lps=image.GetDirection(),
            target_center_lps=center,
        )
        assert np.array_equal(
            sitk.GetArrayViewFromImage(restored),
            sitk.GetArrayViewFromImage(image),
        )
        assert restored.GetSize() == image.GetSize()
        assert np.allclose(restored.GetSpacing(), image.GetSpacing(), atol=1e-12)
        assert np.allclose(restored.GetOrigin(), image.GetOrigin(), atol=1e-12)
        assert np.allclose(restored.GetDirection(), image.GetDirection(), atol=1e-12)


def test_matching_metadata_is_retained_without_geometry_change(base_config):
    original = _ct_image()
    outcome = assess_orientation(
        original,
        config=base_config,
        predictor=FakePredictor(identity_class_index()),
    )

    assert outcome.result.state is OrientationState.PASS_METADATA_MATCH
    assert outcome.result.qc_status == "pass"
    assert not outcome.result.manual_review_required
    assert not outcome.result.orientation_changed
    assert outcome.result.applied_transform is None
    assert np.array_equal(
        sitk.GetArrayViewFromImage(outcome.prepared_image),
        sitk.GetArrayViewFromImage(original),
    )
    assert outcome.prepared_image.GetSpacing() == original.GetSpacing()
    assert outcome.prepared_image.GetOrigin() == original.GetOrigin()
    assert outcome.prepared_image.GetDirection() == original.GetDirection()


def test_direct_api_preserves_nibabel_xyz_to_simpleitk_zyx_boundary(
    base_config,
    tmp_path,
):
    original = _ct_image(
        shape_zyx=(301, 17, 23),
        spacing_xyz=(0.7, 1.2, 2.5),
    )
    container = NiftiDataContainer(tmp_path / "boundary.nii.gz")
    container.img = original

    outcome = assess_orientation(
        container.imgNifti1,
        config=base_config,
        predictor=FakePredictor(identity_class_index()),
    )

    assert np.array_equal(
        sitk.GetArrayFromImage(outcome.prepared_image),
        sitk.GetArrayFromImage(original),
    )
    assert outcome.prepared_image.GetSize() == original.GetSize()
    assert outcome.prepared_image.GetSpacing() == pytest.approx(original.GetSpacing())
    assert outcome.prepared_image.GetOrigin() == pytest.approx(original.GetOrigin())
    assert outcome.prepared_image.GetDirection() == pytest.approx(original.GetDirection())


def test_high_confidence_mismatch_is_lossless_and_always_requires_review(base_config):
    original = _ct_image(
        direction_lps=(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    )
    original_pixels = sitk.GetArrayFromImage(original)
    outcome = assess_orientation(
        original,
        config=base_config,
        predictor=FakePredictor(0),
    )

    assert outcome.result.state is OrientationState.MISMATCH_REPAIRED
    assert outcome.result.orientation_changed
    assert outcome.result.severe_misorientation
    assert outcome.result.manual_review_required
    assert outcome.result.qc_status == "review"
    assert (
        outcome.result.prepared_orientation_code
        == outcome.result.original_orientation_code
    )
    assert outcome.result.applied_transform["interpolation"] == "none"
    assert outcome.result.applied_transform["final_orientation_code"] == "RAS"
    assert [
        step["operation"]
        for step in outcome.result.applied_transform["transform_steps"]
    ] == ["DICOMOrient", "PermuteAxes+Flip", "DICOMOrient"]
    assert outcome.result.applied_transform["inverse_array_exact"]
    assert outcome.result.applied_transform["inverse_physical_point_max_error_mm"] < 1e-4
    assert {flag.code for flag in outcome.result.review_flags} >= {
        "ORIENTATION_CHANGED",
        "SEVERE_METADATA_ANATOMY_MISMATCH",
    }
    assert np.array_equal(sitk.GetArrayFromImage(original), original_pixels)
    assert not np.array_equal(
        sitk.GetArrayFromImage(outcome.prepared_image),
        original_pixels,
    )


def test_axial_half_turn_is_advisory_only_by_default(base_config):
    original = _ct_image(
        direction_lps=(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    )
    original_pixels = sitk.GetArrayFromImage(original)

    outcome = assess_orientation(
        original,
        config=base_config,
        predictor=FakePredictor(4),
    )

    assert outcome.result.state is OrientationState.MISMATCH_UNCERTAIN
    assert outcome.result.manual_review_required
    assert not outcome.result.orientation_changed
    assert outcome.result.applied_transform is None
    assert {flag.code for flag in outcome.result.review_flags} >= {
        "ORIENTATION_AMBIGUOUS_AXIAL_180",
        "ORIENTATION_MISMATCH_NOT_REPAIRED",
    }
    assert np.array_equal(
        sitk.GetArrayFromImage(outcome.prepared_image),
        original_pixels,
    )


def test_axial_half_turn_repair_requires_explicit_opt_in(base_config):
    original = _ct_image(
        direction_lps=(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    )
    config = _config(
        base_config,
        confidence__allow_axial_180_repair=True,
    )

    outcome = assess_orientation(
        original,
        config=config,
        predictor=FakePredictor(4),
    )

    assert outcome.result.state is OrientationState.MISMATCH_REPAIRED
    assert outcome.result.orientation_changed
    assert "ORIENTATION_AMBIGUOUS_AXIAL_180" not in {
        flag.code for flag in outcome.result.review_flags
    }


def test_repair_restores_upright_ras_pixels_and_geometry(base_config):
    upright_ras = _ct_image(
        shape_zyx=(300, 17, 23),
        spacing_xyz=(0.7, 1.2, 2.5),
        direction_lps=(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
    )
    upright_lps = sitk.DICOMOrient(upright_ras, "LPS")
    rotated_lps = apply_model_rotation_to_sitk(
        upright_lps,
        8,
        inverse=False,
    )
    rotated_ras = sitk.DICOMOrient(rotated_lps, "RAS")

    outcome = assess_orientation(
        rotated_ras,
        config=base_config,
        predictor=FakePredictor(8),
    )

    assert outcome.result.state is OrientationState.MISMATCH_REPAIRED
    assert outcome.result.prepared_orientation_code == "RAS"
    assert np.array_equal(
        sitk.GetArrayFromImage(outcome.prepared_image),
        sitk.GetArrayFromImage(upright_ras),
    )
    assert outcome.prepared_image.GetSize() == upright_ras.GetSize()
    assert outcome.prepared_image.GetSpacing() == pytest.approx(upright_ras.GetSpacing())
    assert outcome.prepared_image.GetOrigin() == pytest.approx(upright_ras.GetOrigin())
    assert outcome.prepared_image.GetDirection() == pytest.approx(upright_ras.GetDirection())


def test_oblique_but_same_discrete_orientation_is_retained(base_config):
    angle = np.deg2rad(10.0)
    direction = (
        np.cos(angle), -np.sin(angle), 0.0,
        np.sin(angle), np.cos(angle), 0.0,
        0.0, 0.0, 1.0,
    )
    original = _ct_image(direction_lps=direction)
    outcome = assess_orientation(
        original,
        config=base_config,
        predictor=FakePredictor(identity_class_index()),
    )

    assert outcome.result.state is OrientationState.PASS_METADATA_MATCH
    assert outcome.result.residual_obliquity_deg == pytest.approx(10.0)
    assert outcome.prepared_image.GetDirection() == original.GetDirection()


def test_severe_obliquity_mismatch_is_header_uncertain_and_reports_candidate(
    base_config,
    tmp_path,
):
    angle = np.deg2rad(20.0)
    direction = (
        np.cos(angle), -np.sin(angle), 0.0,
        np.sin(angle), np.cos(angle), 0.0,
        0.0, 0.0, 1.0,
    )
    outcome = assess_orientation(
        _ct_image(direction_lps=direction),
        config=base_config,
        predictor=FakePredictor(0),
        output_directory=tmp_path,
    )

    assert outcome.result.state is OrientationState.HEADER_UNCERTAIN
    assert outcome.result.qc_status == "review"
    assert not outcome.result.orientation_changed
    assert outcome.review_png.is_file()
    assert "ORIENTATION_SEVERE_OBLIQUITY" in {
        flag.code for flag in outcome.result.review_flags
    }


@pytest.mark.parametrize(
    ("shape_zyx", "agreement_count", "force_reason", "expected_flag"),
    [
        ((50, 20, 20), 24, None, "ORIENTATION_SHORT_FOV"),
        ((300, 20, 20), 22, None, "ORIENTATION_LOW_EQUIVARIANCE"),
        ((300, 20, 20), 24, "reflection-like control", "ORIENTATION_FORCED_REVIEW"),
    ],
)
def test_unsafe_mismatches_continue_without_repair(
    base_config,
    shape_zyx,
    agreement_count,
    force_reason,
    expected_flag,
):
    config = deepcopy(base_config)
    if force_reason:
        config["orientation"]["force_review_reasons"] = [force_reason]
    outcome = assess_orientation(
        _ct_image(shape_zyx=shape_zyx),
        config=config,
        predictor=FakePredictor(0, agreement_count=agreement_count),
    )

    assert outcome.result.state is OrientationState.MISMATCH_UNCERTAIN
    assert outcome.result.qc_status == "review"
    assert not outcome.result.orientation_changed
    assert expected_flag in {flag.code for flag in outcome.result.review_flags}


def test_missing_dicom_orientation_provenance_uses_header_uncertain_state(base_config):
    config = deepcopy(base_config)
    config["orientation"]["header_uncertain_reasons"] = [
        "ImageOrientationPatient missing"
    ]
    outcome = assess_orientation(
        _ct_image(),
        config=config,
        predictor=FakePredictor(0),
    )

    assert outcome.result.state is OrientationState.HEADER_UNCERTAIN
    assert not outcome.result.orientation_changed
    assert "ORIENTATION_HEADER_UNCERTAIN" in {
        flag.code for flag in outcome.result.review_flags
    }


def test_empty_body_extent_prevents_automatic_repair(base_config):
    image = sitk.GetImageFromArray(
        np.full((300, 20, 20), -1000, dtype=np.int16)
    )
    outcome = assess_orientation(
        image,
        config=base_config,
        predictor=FakePredictor(0),
    )

    assert outcome.result.state is OrientationState.MISMATCH_UNCERTAIN
    assert not outcome.result.orientation_changed
    assert outcome.result.body_extent_mm == 0
    assert "ORIENTATION_EMPTY_BODY_EXTENT" in {
        flag.code for flag in outcome.result.review_flags
    }


def test_metal_burden_prevents_automatic_repair(base_config):
    array = np.zeros((300, 20, 20), dtype=np.int16)
    array.reshape(-1)[:100] = 3000
    outcome = assess_orientation(
        sitk.GetImageFromArray(array),
        config=base_config,
        predictor=FakePredictor(0),
    )

    assert outcome.result.state is OrientationState.MISMATCH_UNCERTAIN
    assert not outcome.result.orientation_changed
    assert outcome.result.metal_fraction > 0.0005
    assert "ORIENTATION_METAL_ARTEFACT" in {
        flag.code for flag in outcome.result.review_flags
    }


def test_configured_transverse_truncation_gate_prevents_repair(base_config):
    config = deepcopy(base_config)
    config["orientation"]["artifact_rules"]["review_on_possible_truncation"] = True
    outcome = assess_orientation(
        _ct_image(),
        config=config,
        predictor=FakePredictor(0),
    )

    assert outcome.result.state is OrientationState.MISMATCH_UNCERTAIN
    assert outcome.result.possible_fov_truncation
    assert not outcome.result.orientation_changed
    assert "ORIENTATION_POSSIBLE_TRUNCATION" in {
        flag.code for flag in outcome.result.review_flags
    }


def test_invalid_pixels_are_a_hard_failure(base_config):
    image = sitk.GetImageFromArray(np.full((3, 4, 5), np.nan, dtype=np.float32))
    with pytest.raises(OrientationIntegrityError, match="non-finite"):
        assess_orientation(
            image,
            config=base_config,
            predictor=FakePredictor(identity_class_index()),
        )


def test_nonorthonormal_geometry_is_a_hard_failure(base_config):
    image = _ct_image()
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0))

    with pytest.raises(OrientationIntegrityError, match="not orthonormal"):
        assess_orientation(
            image,
            config=base_config,
            predictor=FakePredictor(identity_class_index()),
        )


def test_direct_orientation_api_rejects_truthy_string_configuration():
    with pytest.raises(ValueError, match="orientation.enabled must be boolean"):
        assess_orientation(
            _ct_image(),
            config={"orientation": {"enabled": "false"}},
            predictor=FakePredictor(identity_class_index()),
        )


def test_reports_validate_against_schema_and_are_deterministic(base_config, tmp_path):
    original = _ct_image(
        direction_lps=(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    )
    directories = [tmp_path / "first", tmp_path / "second"]
    reports = []
    images = []
    for directory in directories:
        outcome = assess_orientation(
            original,
            config=base_config,
            predictor=FakePredictor(0),
            output_directory=directory,
            case_id="pseudonymous-001",
        )
        payload = json.loads(outcome.report_json.read_text(encoding="utf-8"))
        expected_attribution = {
            "asset_id": model_asset_record()["asset_id"],
            "name": "CTDeepRot-2D",
            "required_for": "default_orientation_integrity_stage",
            "upstream_repository": UPSTREAM_REPOSITORY,
            "code_license": CODE_LICENSE,
            "citation_doi": MODEL_CITATION_DOI,
            "download_url": CHECKPOINT_URL,
            "byte_size": CHECKPOINT_BYTES,
            "redistribution_mode": REDISTRIBUTION_MODE,
        }
        for key, value in expected_attribution.items():
            assert payload["model"][key] == value
        Draft202012Validator(_schema()).validate(payload)
        persisted = sitk.ReadImage(str(outcome.corrected_input))
        assert np.array_equal(
            sitk.GetArrayViewFromImage(persisted),
            sitk.GetArrayViewFromImage(outcome.prepared_image),
        )
        review = cv2.imread(str(outcome.review_png))
        assert review.shape[0] >= 1140
        assert review.shape[1:] == (1400, 3)
        reports.append(outcome.report_json.read_bytes())
        images.append(outcome.review_png.read_bytes())

    assert reports[0] == reports[1]
    assert images[0] == images[1]
    assert b"pseudonymous-001" not in images[0]


def test_review_png_renders_every_flag_reason(base_config, tmp_path, monkeypatch):
    config = deepcopy(base_config)
    reasons = ["first explicit review reason", "second explicit review reason"]
    config["orientation"]["force_review_reasons"] = reasons
    rendered_text = []
    original_put_text = cv2.putText

    def capture_text(image, text, *args, **kwargs):
        rendered_text.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    outcome = assess_orientation(
        _ct_image(),
        config=config,
        predictor=FakePredictor(identity_class_index()),
        output_directory=tmp_path,
        case_id="pseudonymous-001",
    )

    assert outcome.review_png.is_file()
    for reason in reasons:
        assert f"ORIENTATION_FORCED_REVIEW: {reason}" in rendered_text


def test_failure_report_validates_against_schema(tmp_path):
    private_path = "/private/patient/source/case-001.nii.gz"
    report, review = write_orientation_failure_artifacts(
        output_directory=tmp_path,
        case_id="pseudonymous-001",
        error=OrientationIntegrityError(f"Invalid CT geometry at {private_path}"),
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    Draft202012Validator(_schema()).validate(payload)
    assert payload["state"] == "ORIENTATION_FAILED"
    assert payload["review_flags"][0]["stage"] == "orientation"
    assert private_path not in report.read_text(encoding="utf-8")
    assert private_path.encode() not in review.read_bytes()
    assert review.is_file()


def test_pipeline_action_replaces_only_repaired_input_and_persists_review(
    pipeline_stub,
    container_factory,
    tmp_path,
):
    source = container_factory(
        tmp_path / "source/images/case-001.nii.gz",
        sitk.GetArrayFromImage(_ct_image()),
    )
    source_pixels = source.data.copy()
    action = AssessOrientation(pipeline_stub)
    action._predictor = FakePredictor(0)
    memory = {
        "id": "case-001",
        "workspace": tmp_path / "workspace",
        "tmp/index": source,
    }

    action(memory)
    action.validate_outputs(memory)

    assert memory["tmp/original_index"] is source
    assert np.array_equal(source.data, source_pixels)
    assert memory["tmp/index"] is not source
    assert memory["tmp/index"].path == (
        tmp_path / "workspace/orientation/corrected_input.nii.gz"
    )
    assert memory["tmp/orientation_result"].manual_review_required
    prepared = memory["tmp/prepared_image"]
    assert prepared.result is memory["tmp/orientation_result"]
    assert prepared.prepared_image.GetSize() == memory["tmp/index"].img.GetSize()
    assert (tmp_path / "workspace/orientation/orientation_report.json").is_file()
    assert (tmp_path / "workspace/orientation/orientation_review.png").is_file()


def test_uncertain_readable_case_reaches_next_pipeline_action(
    pipeline_stub,
    container_factory,
    tmp_path,
):
    source = container_factory(
        tmp_path / "source/images/case-001.nii.gz",
        sitk.GetArrayFromImage(_ct_image(shape_zyx=(50, 20, 20))),
    )
    action = AssessOrientation(pipeline_stub)
    action._predictor = FakePredictor(0)
    memory = {
        "id": "case-001",
        "workspace": tmp_path / "workspace",
        "tmp/index": source,
    }
    action(memory)

    class RecordsPreparedInput(PipelineAction):
        def __init__(self, pipeline):
            super().__init__(pipeline)
            self.io_inputs = ["tmp/index", "tmp/orientation_result"]
            self.io_outputs = ["tmp/reached"]

        def __call__(self, selected_memory):
            super().__call__(selected_memory)
            selected_memory["tmp/reached"] = True

    next_action = RecordsPreparedInput(pipeline_stub)
    next_action(memory)
    next_action.validate_outputs(memory)

    assert memory["tmp/reached"]
    assert memory["tmp/index"] is source
    assert memory["tmp/orientation_result"].state is OrientationState.MISMATCH_UNCERTAIN


def test_orientation_action_receives_incomplete_dicom_header_provenance(
    pipeline_stub,
    container_factory,
    tmp_path,
):
    source = container_factory(
        tmp_path / "attempt/input/converted_input.nii.gz",
        sitk.GetArrayFromImage(_ct_image()),
    )
    action = AssessOrientation(pipeline_stub)
    action._predictor = FakePredictor(identity_class_index())
    memory = {
        "id": "case-001",
        "workspace": tmp_path / "workspace",
        "tmp/index": source,
        "tmp/input_summary": {
            "input_format": "dicom",
            "dicom": {
                "image_orientation_patient_complete": False,
                "image_position_patient_complete": True,
            },
        },
    }

    action(memory)

    result = memory["tmp/orientation_result"]
    assert result.state is OrientationState.HEADER_UNCERTAIN
    assert not result.orientation_changed
    assert "ORIENTATION_HEADER_UNCERTAIN" in {flag.code for flag in result.review_flags}


def test_orientation_action_uses_canonical_case_bundle_paths(pipeline_stub):
    action = AssessOrientation(pipeline_stub)
    assert action.io_persisted_outputs == [
        "orientation/orientation_report.json",
        "orientation/orientation_review.png",
    ]
    assert action.corrected_name == "orientation/corrected_input.nii.gz"


def test_checkpoint_verification_rejects_modified_assets(tmp_path):
    checkpoint = tmp_path / "net2d.pt"
    checkpoint.write_bytes(b"not the pinned model")
    with pytest.raises(ModelAssetError, match="digest mismatch"):
        verify_checkpoint(checkpoint, expected_bytes=None)

    observed = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert verify_checkpoint(checkpoint, observed, checkpoint.stat().st_size) == checkpoint


def test_ctdeeprot_inference_never_downloads_missing_assets(tmp_path, monkeypatch):
    def unexpected_request(*args, **kwargs):
        raise AssertionError("inference attempted a network request")

    monkeypatch.setattr("requests.get", unexpected_request)
    with pytest.raises(ModelAssetError, match="bodycomposition models sync"):
        CTDeepRotPredictor(tmp_path / "missing" / "net2d.pt")


def test_ctdeeprot_model_loader_uses_only_safe_modern_apis(tmp_path, monkeypatch):
    import torch
    from torchvision import models as torchvision_models

    from BodyComposition.orientation import ctdeeprot

    checkpoint = tmp_path / "net2d.pt"
    checkpoint.write_bytes(b"verified in test")
    resnet_calls = []
    load_calls = []
    original_resnet18 = torchvision_models.resnet18

    def observe_resnet18(*args, **kwargs):
        resnet_calls.append((args, kwargs))
        return original_resnet18(*args, **kwargs)

    def reject_after_observation(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise TypeError("simulated unsupported safe loader")

    ctdeeprot.release_cached_models()
    monkeypatch.setattr(ctdeeprot, "verify_checkpoint", lambda *args: checkpoint)
    monkeypatch.setattr(torchvision_models, "resnet18", observe_resnet18)
    monkeypatch.setattr(torch, "load", reject_after_observation)

    with pytest.raises(TypeError, match="unsupported safe loader"):
        ctdeeprot._load_model(str(checkpoint), "expected", "cpu")

    assert resnet_calls == [((), {"weights": None})]
    assert len(load_calls) == 1
    assert load_calls[0][1]["weights_only"] is True
    ctdeeprot.release_cached_models()


def test_ctdeeprot_model_loader_does_not_retry_legacy_resnet_api(
    tmp_path, monkeypatch
):
    from torchvision import models as torchvision_models

    from BodyComposition.orientation import ctdeeprot

    checkpoint = tmp_path / "net2d.pt"
    checkpoint.write_bytes(b"verified in test")
    calls = []

    def reject_modern_api(*args, **kwargs):
        calls.append((args, kwargs))
        raise TypeError("simulated unsupported torchvision")

    ctdeeprot.release_cached_models()
    monkeypatch.setattr(ctdeeprot, "verify_checkpoint", lambda *args: checkpoint)
    monkeypatch.setattr(torchvision_models, "resnet18", reject_modern_api)

    with pytest.raises(TypeError, match="unsupported torchvision"):
        ctdeeprot._load_model(str(checkpoint), "expected", "cpu")

    assert calls == [((), {"weights": None})]
    ctdeeprot.release_cached_models()


def test_ctdeeprot_model_sync_is_atomic_verified_and_idempotent(tmp_path, monkeypatch):
    payload = b"pinned checkpoint fixture"
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = tmp_path / "version" / "net2d.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"invalid old checkpoint")
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            assert chunk_size == 1024 * 1024
            yield payload

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("requests.get", fake_get)
    first = sync_checkpoint(
        checkpoint,
        download_url="https://example.invalid/net2d.pt",
        expected_sha256=digest,
        expected_bytes=len(payload),
    )
    second = sync_checkpoint(
        checkpoint,
        download_url="https://example.invalid/net2d.pt",
        expected_sha256=digest,
        expected_bytes=len(payload),
    )

    assert first == second == checkpoint
    assert checkpoint.read_bytes() == payload
    assert len(calls) == 1
    assert not list(checkpoint.parent.glob("*.part"))


def test_ctdeeprot_asset_record_is_complete_and_versioned():
    asset = model_asset_record()

    assert asset["byte_size"] == CHECKPOINT_BYTES
    assert asset["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert asset["download_url"] == CHECKPOINT_URL
    assert asset["redistribution_mode"] == "user_model_sync"
    assert asset["expected_files"] == [
        {
            "relative_path": "net2d.pt",
            "byte_size": CHECKPOINT_BYTES,
            "sha256": CHECKPOINT_SHA256,
        }
    ]
    assert str(DEFAULT_CHECKPOINT_PATH.parent).endswith(asset["upstream_commit"])


def test_ctdeeprot_sync_cli_uses_the_unified_model_service(tmp_path, monkeypatch, capsys):
    from BodyComposition import cli
    from BodyComposition.model_manager import ModelStatus

    selected = []

    def fake_sync(config, model_ids):
        selected.extend(model_ids)
        return (
            ModelStatus(
                model_id="ctdeeprot_2d_v1",
                ready=True,
                model_root=tmp_path,
                errors=(),
                checked_files={"net2d.pt": "a" * 64},
                asset=model_asset_record(),
            ),
        )

    monkeypatch.setattr(cli, "sync_models", fake_sync)
    assert cli.main(["models", "sync", "--model", "ctdeeprot_2d_v1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ready"]
    assert selected == ["ctdeeprot_2d_v1"]
