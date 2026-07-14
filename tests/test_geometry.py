from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from BodyComposition.measurements import calculate_slice_measurements
from BodyComposition.utils.geometry import (
    GeometryError,
    ImageGeometry,
    assert_same_physical_domain,
)
from BodyComposition.utils.nifti import NiftiDataContainer, resample_label_to_reference


def test_anisotropic_pixel_area_and_voxel_volume():
    geometry = ImageGeometry(
        size_xyz=(9, 7, 5),
        spacing_xyz=(0.7, 1.3, 4.0),
        origin_lps_xyz=(10.0, -20.0, 30.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    assert geometry.in_plane_area_mm2 == pytest.approx(0.91)
    assert geometry.voxel_volume_mm3 == pytest.approx(3.64)

    mask = np.zeros((5, 7, 9), dtype=np.uint8)
    mask[2, 1:4, 2:6] = 1
    measured = calculate_slice_measurements(mask, geometry, {1: "SM"})
    assert measured.loc[2, "Vx_SM"] == 12
    assert measured.loc[2, "CSA_SM"] == pytest.approx(12 * 0.91 / 100)


@pytest.mark.parametrize(
    "direction_lps",
    [
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    ],
)
def test_measurements_are_invariant_to_permuted_and_flipped_directions(direction_lps):
    geometry = ImageGeometry(
        size_xyz=(40, 30, 1),
        spacing_xyz=(0.7, 1.3, 4.0),
        origin_lps_xyz=(5.0, -2.0, 9.0),
        direction_lps=direction_lps,
    )
    mask = np.zeros((1, 30, 40), dtype=np.uint8)
    mask[0, 5:15, 7:27] = 1
    measured = calculate_slice_measurements(
        mask,
        geometry,
        {1: "SM"},
        contour_mask_zyx=mask,
    )
    baseline_geometry = ImageGeometry(
        size_xyz=(40, 30, 1),
        spacing_xyz=(0.7, 1.3, 4.0),
        origin_lps_xyz=(5.0, -2.0, 9.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    baseline = calculate_slice_measurements(
        mask,
        baseline_geometry,
        {1: "SM"},
        contour_mask_zyx=mask,
    )
    assert measured.loc[0, "CSA_SM"] == pytest.approx(1.82)
    assert measured.loc[0, "CSA_contour"] == pytest.approx(baseline.loc[0, "CSA_contour"])
    assert measured.loc[0, "CIR_contour"] == pytest.approx(baseline.loc[0, "CIR_contour"])


def test_sitk_nifti_round_trip_preserves_array_and_physical_points(
    tmp_path,
    sitk_image_factory,
):
    array_zyx = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
    image = sitk_image_factory(
        array_zyx,
        spacing_xyz=(0.8, 1.2, 3.5),
        origin_lps_xyz=(14.0, -7.0, 22.0),
        direction_lps=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    original = NiftiDataContainer(tmp_path / "images" / "original.nii.gz")
    original.img = image
    restored = NiftiDataContainer(tmp_path / "images" / "restored.nii.gz")
    restored.img = original.imgNifti1

    assert np.array_equal(restored.data, original.data)
    assert restored.geometry.equivalent_to(original.geometry)
    for index_xyz in ((0, 0, 0), (4, 3, 2), (2, 1, 1)):
        assert restored.img.TransformIndexToPhysicalPoint(index_xyz) == pytest.approx(
            original.img.TransformIndexToPhysicalPoint(index_xyz),
            abs=1e-4,
        )


def test_canonicalization_preserves_each_voxels_physical_point(tmp_path, sitk_image_factory):
    array_zyx = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    image = sitk_image_factory(
        array_zyx,
        spacing_xyz=(0.7, 1.2, 3.4),
        origin_lps_xyz=(10.0, -20.0, 30.0),
        direction_lps=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    )
    expected_points = {}
    for z in range(array_zyx.shape[0]):
        for y in range(array_zyx.shape[1]):
            for x in range(array_zyx.shape[2]):
                expected_points[int(array_zyx[z, y, x])] = image.TransformIndexToPhysicalPoint((x, y, z))

    container = NiftiDataContainer(tmp_path / "images" / "input.nii.gz")
    container.img = image
    container.as_closest_canonical()
    canonical_image = container.img
    for z in range(container.data.shape[0]):
        for y in range(container.data.shape[1]):
            for x in range(container.data.shape[2]):
                value = int(container.data[z, y, x])
                assert canonical_image.TransformIndexToPhysicalPoint((x, y, z)) == pytest.approx(
                    expected_points[value],
                    abs=1e-4,
                )


def test_label_resampling_restores_permuted_grid_without_interpolation(sitk_image_factory):
    array_zyx = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    reference_label = sitk_image_factory(
        array_zyx,
        spacing_xyz=(0.7, 1.2, 3.4),
        origin_lps_xyz=(10.0, -20.0, 30.0),
        direction_lps=(0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    )
    canonical_label = sitk.DICOMOrient(reference_label, "RAS")

    restored = resample_label_to_reference(canonical_label, reference_label)

    assert restored.GetSize() == reference_label.GetSize()
    assert restored.GetSpacing() == pytest.approx(reference_label.GetSpacing())
    assert restored.GetOrigin() == pytest.approx(reference_label.GetOrigin())
    assert restored.GetDirection() == pytest.approx(reference_label.GetDirection())
    assert np.array_equal(sitk.GetArrayFromImage(restored), array_zyx)
    assert set(np.unique(sitk.GetArrayFromImage(restored))) == set(np.unique(array_zyx))


def test_bounding_box_origin_uses_xyz_spacing(tmp_path, sitk_image_factory):
    image = sitk_image_factory(
        np.zeros((5, 8, 10), dtype=np.int16),
        spacing_xyz=(0.5, 1.5, 4.0),
        origin_lps_xyz=(3.0, 5.0, 7.0),
        direction_lps=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    container = NiftiDataContainer(tmp_path / "images" / "input.nii.gz")
    container.img = image
    container.bbox = [1, 4, 2, 7, 3, 9]
    assert container.origin == pytest.approx(image.TransformIndexToPhysicalPoint((3, 2, 1)))


def test_domain_mismatch_fails_early():
    reference = ImageGeometry(
        (10, 10, 3),
        (1.0, 1.0, 2.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    candidate = ImageGeometry(
        (10, 11, 3),
        (1.0, 1.0, 2.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    with pytest.raises(GeometryError, match="sizes differ"):
        assert_same_physical_domain(reference, candidate)
