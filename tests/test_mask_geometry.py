import numpy as np
import pytest

from BodyComposition.actions.masks_int import MasksInternalTissue
from BodyComposition.utils.geometry import GeometryError


def _disable_tissue_filters(config):
    for tissue in ("imat", "sm", "vat", "sat"):
        config["tissue"][tissue]["filter_hu"] = False


def test_internal_postprocessing_rejects_misaligned_image_and_label(
    tmp_path,
    pipeline_stub,
    container_factory,
):
    _disable_tissue_filters(pipeline_stub.config)
    action = MasksInternalTissue(pipeline_stub, image="tmp/index")
    array = np.zeros((2, 3, 4), dtype=np.uint8)
    image = container_factory(tmp_path / "images" / "ct.nii.gz", array.astype(np.int16))
    label = container_factory(
        tmp_path / "masks" / "tissue_compartments.nii.gz",
        array,
        origin_lps_xyz=(1.0, 0.0, 0.0),
    )
    memory = {
        "id": "case",
        "workspace": tmp_path,
        "tmp/index": image,
        action.input_label_tissue_name: label,
    }

    with pytest.raises(GeometryError, match="origin_lps_xyz differ"):
        action(memory)


def test_internal_size_filter_removes_only_small_eligible_islands(
    tmp_path,
    pipeline_stub,
    container_factory,
):
    _disable_tissue_filters(pipeline_stub.config)
    pipeline_stub.config["tissue"]["sm"].update(
        {
            "filter_hu": True,
            "filter_size": True,
            "filter_size_version": "2D",
            "filter_size_2D": 2,
            "filter_size_connectivity": 8,
        }
    )
    action = MasksInternalTissue(pipeline_stub, image="tmp/index")
    labels = np.zeros((1, 10, 10), dtype=np.uint8)
    labels[0, 2, 2] = action.LBL_TISSUE_R["SM"]
    labels[0, 5:7, 5:7] = action.LBL_TISSUE_R["SM"]
    image_array = np.zeros_like(labels, dtype=np.int16)
    image_array[labels != 0] = 40
    image = container_factory(tmp_path / "images" / "ct.nii.gz", image_array)
    label = container_factory(tmp_path / "masks" / "tissue_compartments.nii.gz", labels)
    memory = {
        "id": "case",
        "workspace": tmp_path,
        "tmp/index": image,
        action.input_label_tissue_name: label,
    }

    action(memory)

    output = memory[action.output_mask_name].data
    assert label.path.exists()
    assert output[0, 2, 2] == 0
    assert np.all(output[0, 5:7, 5:7] == action.LBL_TISSUE_R["SM"])


def test_imat_filter_cannot_connect_through_non_muscle_pixels(
    tmp_path,
    pipeline_stub,
    container_factory,
):
    _disable_tissue_filters(pipeline_stub.config)
    pipeline_stub.config["tissue"]["imat"].update(
        {
            "filter_hu": True,
            "filter_size": True,
            "filter_size_version": "2D",
            "filter_size_2D": 2,
            "filter_size_connectivity": 8,
        }
    )
    action = MasksInternalTissue(pipeline_stub, image="tmp/index")
    labels = np.zeros((1, 8, 8), dtype=np.uint8)
    labels[0, 4, 4] = action.LBL_TISSUE_R["SM"]
    image_array = np.zeros_like(labels, dtype=np.int16)
    image_array[0, 3:6, 3:6] = -100
    image = container_factory(tmp_path / "images" / "ct.nii.gz", image_array)
    label = container_factory(tmp_path / "masks" / "tissue_compartments.nii.gz", labels)
    memory = {
        "id": "case",
        "workspace": tmp_path,
        "tmp/index": image,
        action.input_label_tissue_name: label,
    }

    action(memory)

    assert memory[action.output_mask_name].data[0, 4, 4] == action.LBL_TISSUE_R["SM"]
