import numpy as np
import pytest

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
