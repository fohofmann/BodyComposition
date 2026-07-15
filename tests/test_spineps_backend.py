from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.vertebral import (
    ExecutionStatus,
    QCStatus,
    adapt_spineps_outputs,
    adapt_spineps_sitk_outputs,
)
from BodyComposition.vertebral.contracts import VertebralResult
from BodyComposition.vertebral.spineps_qc import construct_vertebral_body_labels
from BodyComposition.vertebral import spineps_session


def _geometry(shape_zyx=(12, 8, 6), direction=None):
    return ImageGeometry(
        size_xyz=tuple(reversed(shape_zyx)),
        spacing_xyz=(2.0, 2.0, 2.0),
        origin_lps_xyz=(0.0, 0.0, 0.0),
        direction_lps=direction or (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def _ordinary_outputs():
    vertebra = np.zeros((12, 8, 6), dtype=np.uint8)
    semantic = np.zeros_like(vertebra)
    vertebra[2:5, 1:7, 1:5] = 22
    vertebra[6:9, 1:7, 1:5] = 21
    semantic[vertebra != 0] = 49
    return semantic, vertebra


def test_corpus_intersection_preserves_native_labels_and_inputs():
    semantic, vertebra = _ordinary_outputs()
    semantic_before = semantic.copy()
    vertebra_before = vertebra.copy()

    result = construct_vertebral_body_labels(semantic, vertebra, _geometry())

    assert set(np.unique(result)) == {0, 21, 22}
    assert np.array_equal(semantic, semantic_before)
    assert np.array_equal(vertebra, vertebra_before)


def test_adapter_returns_native_labels_centroids_and_provenance():
    semantic, vertebra = _ordinary_outputs()

    result = adapt_spineps_outputs(
        semantic,
        vertebra,
        _geometry(),
        native_outputs={"seg_spine": Path("seg-spine.nii.gz")},
    )

    assert result.execution_status == ExecutionStatus.SUCCEEDED
    assert result.qc_status == QCStatus.PASS
    assert [centroid.anatomical_label for centroid in result.centroids] == ["L2", "L3"]
    assert result.provenance["spineps_version"] == "2.0.0"
    assert result.provenance["model_assets"]["ct"]["redistribution_mode"] == "user_model_sync"
    assert (
        result.provenance["required_crop_model"]["redistribution_mode"]
        == "user_model_sync"
    )
    assert result.native_outputs["seg_spine"] == Path("seg-spine.nii.gz")


def test_adapter_flags_t13_l6_and_physical_fov_boundaries():
    shape = (8, 6, 4)
    geometry = _geometry(
        shape,
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
    )
    vertebra = np.zeros(shape, dtype=np.uint8)
    semantic = np.zeros(shape, dtype=np.uint8)
    vertebra[0:2, 1:5, 1:3] = 28
    vertebra[3:5, 1:5, 1:3] = 20
    vertebra[6:8, 1:5, 1:3] = 25
    semantic[vertebra != 0] = 49

    result = adapt_spineps_outputs(semantic, vertebra, geometry)
    codes = {flag.code for flag in result.qc_flags}

    assert result.qc_status == QCStatus.REVIEW
    assert {"variant_t13", "variant_l6"}.issubset(codes)
    assert "vertebra_touches_cranial_fov" in codes
    assert "vertebra_touches_caudal_fov" in codes


def test_adapter_flags_non_monotonic_physical_sequence():
    semantic, vertebra = _ordinary_outputs()
    vertebra = vertebra.copy()
    vertebra[2:5][vertebra[2:5] == 22] = 21
    vertebra[6:9][vertebra[6:9] == 21] = 22
    semantic = np.where(vertebra != 0, 49, 0).astype(np.uint8)

    result = adapt_spineps_outputs(semantic, vertebra, _geometry())

    assert "non_monotonic_vertebral_order" in {flag.code for flag in result.qc_flags}
    assert result.qc_status == QCStatus.FAIL


def test_adapter_flags_unassigned_semantic_corpus_voxels():
    semantic, vertebra = _ordinary_outputs()
    semantic = semantic.copy()
    semantic[10:12, 1:4, 1:4] = 49

    result = adapt_spineps_outputs(semantic, vertebra, _geometry())

    assert "semantic_corpus_without_vertebra_label" in {
        flag.code for flag in result.qc_flags
    }


def test_adapter_flags_enumeration_variants_and_missing_transition_anchors():
    shape = (24, 10, 10)
    vertebra = np.zeros(shape, dtype=np.uint8)
    for start, label in (
        (20, 18),
        (17, 20),
        (14, 21),
        (11, 22),
        (8, 23),
        (4, 26),
    ):
        vertebra[start : start + 2, 2:8, 2:8] = label
    semantic = np.where(vertebra != 0, 49, 0).astype(np.uint8)

    result = adapt_spineps_outputs(semantic, vertebra, _geometry(shape))
    codes = {flag.code for flag in result.qc_flags}

    assert "variant_four_lumbar_or_missing_l5" in codes
    assert "variant_eleven_thoracic_or_missing_t12" in codes
    assert "incomplete_caudal_anchor_set" in codes
    assert "incomplete_thoracolumbar_anchor_set" in codes


def test_adapter_flags_implausibly_large_corpus_against_adjacent_levels():
    shape = (18, 20, 20)
    vertebra = np.zeros(shape, dtype=np.uint8)
    vertebra[13:16, 6:14, 6:14] = 21
    vertebra[7:12, 1:19, 1:19] = 22
    vertebra[2:5, 6:14, 6:14] = 23
    semantic = np.where(vertebra != 0, 49, 0).astype(np.uint8)

    result = adapt_spineps_outputs(semantic, vertebra, _geometry(shape))

    assert "implausibly_large_corpus" in {flag.code for flag in result.qc_flags}


def test_upstream_failure_is_explicit_and_does_not_fabricate_masks():
    result = adapt_spineps_outputs(None, None, None, upstream_error_code="EMPTY")

    assert result.execution_status == ExecutionStatus.FAILED
    assert result.qc_status == QCStatus.FAIL
    assert result.whole_vertebra_labels is None
    assert result.error_summary == "SPINEPS failed with error code EMPTY."


def test_success_result_rejects_a_body_mask_that_changes_native_labels():
    geometry = _geometry((2, 2, 2))
    whole = np.ones((2, 2, 2), dtype=np.uint8)
    body = np.full((2, 2, 2), 2, dtype=np.uint8)

    with pytest.raises(ValueError, match="labeled subset"):
        VertebralResult(
            backend_id="test",
            execution_status=ExecutionStatus.SUCCEEDED,
            geometry=geometry,
            whole_vertebra_labels=whole,
            vertebral_body_labels=body,
        )


def test_model_session_verifies_paths_and_loads_once(tmp_path, monkeypatch):
    verified = []
    loaded = []

    def fake_verify(root, asset, *, full):
        assert full is False
        verified.append(asset.model_id)
        return root / asset.install_dir

    def fake_loader(path, use_cpu):
        loaded.append((path.name, use_cpu))
        return object()

    monkeypatch.setattr(spineps_session, "verify_installed_asset", fake_verify)
    session = spineps_session.SpinepsModelSession(tmp_path, use_cpu=True, loader=fake_loader)

    first = session.load()
    second = session.load()

    assert first is second
    assert verified == ["ct", "ct_instance", "ct_labeling"]
    assert loaded == [("semantic_ct", True), ("instance_ct", True), ("labeling_ct", True)]
    assert session.provenance["required_crop_model"]["dataset_id"] == 100


def _sitk_label(array_zyx, geometry):
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(geometry.spacing_xyz)
    image.SetOrigin(geometry.origin_lps_xyz)
    image.SetDirection(geometry.direction_lps)
    return image


def test_sitk_boundary_preserves_non_cubic_zyx_array_and_geometry():
    shape_zyx = (3, 5, 7)
    geometry = ImageGeometry(
        size_xyz=(7, 5, 3),
        spacing_xyz=(0.8, 1.2, 2.5),
        origin_lps_xyz=(12.0, -4.0, 31.0),
        direction_lps=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    vertebra = np.zeros(shape_zyx, dtype=np.uint8)
    semantic = np.zeros(shape_zyx, dtype=np.uint8)
    vertebra[1, 2, 6] = 22
    semantic[1, 2, 6] = 49

    result = adapt_spineps_sitk_outputs(
        _sitk_label(semantic, geometry),
        _sitk_label(vertebra, geometry),
        geometry,
    )

    assert result.execution_status == ExecutionStatus.SUCCEEDED
    assert result.whole_vertebra_labels.shape == shape_zyx
    assert result.whole_vertebra_labels[1, 2, 6] == 22
    assert result.centroids[0].index_zyx == (1.0, 2.0, 6.0)


def test_sitk_boundary_returns_failed_result_for_geometry_mismatch():
    semantic, vertebra = _ordinary_outputs()
    reference = _geometry()
    semantic_image = _sitk_label(semantic, reference)
    vertebra_image = _sitk_label(vertebra, reference)
    vertebra_image.SetOrigin((1.0, 0.0, 0.0))

    result = adapt_spineps_sitk_outputs(
        semantic_image,
        vertebra_image,
        reference,
    )

    assert result.execution_status == ExecutionStatus.FAILED
    assert result.qc_status == QCStatus.FAIL
    assert {flag.code for flag in result.qc_flags} == {"physical_domain_mismatch"}
