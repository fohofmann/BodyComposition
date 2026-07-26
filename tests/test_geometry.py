import numpy as np
import pytest
import SimpleITK as sitk

from BodyComposition.utils.geometry import (
    GeometryError,
    ImageGeometry,
    assert_same_physical_domain,
)
from BodyComposition.utils.nifti import NiftiDataContainer


def test_anisotropic_pixel_area_and_voxel_volume():
    geometry = ImageGeometry(
        size_xyz=(9, 7, 5),
        spacing_xyz=(0.7, 1.3, 4.0),
        origin_lps_xyz=(10.0, -20.0, 30.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    assert geometry.in_plane_area_mm2 == pytest.approx(0.91)
    assert geometry.voxel_volume_mm3 == pytest.approx(3.64)

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


def test_ct_dtype_and_values_are_preserved_in_a_mask_named_directory(tmp_path):
    array_zyx = np.array(
        [[[-1024.0, -100.25], [31.5, 847.75]]],
        dtype=np.float32,
    )
    path = tmp_path / "patient_masks_and_labels" / "ct.nii.gz"
    path.parent.mkdir()
    sitk.WriteImage(sitk.GetImageFromArray(array_zyx), str(path))

    container = NiftiDataContainer(path)
    container.load_from_file()

    assert container.data.dtype == np.float32
    assert np.array_equal(container.data, array_zyx)


def test_explicit_label_dtype_is_validated_before_conversion(tmp_path):
    container = NiftiDataContainer(
        tmp_path / "masks" / "labels.nii.gz",
        dtype=np.uint8,
    )
    container.img = sitk.GetImageFromArray(
        np.array([[[0, 1], [2, 255]]], dtype=np.int16)
    )

    assert container.data.dtype == np.uint8
    assert np.array_equal(
        container.data,
        np.array([[[0, 1], [2, 255]]], dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="do not fit"):
        container.data = np.array([[[0, 1], [2, 256]]], dtype=np.int16)
    with pytest.raises(ValueError, match="real numeric"):
        container.data = np.full((1, 2, 2), 1 + 1j, dtype=np.complex64)
    with pytest.raises(TypeError, match="integer label dtype"):
        NiftiDataContainer(tmp_path / "labels.nii.gz", dtype=np.float32)


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
