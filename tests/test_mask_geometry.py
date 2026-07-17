import numpy as np
import pytest

from BodyComposition.actions.masks_int import MasksInternalTissue
from BodyComposition.actions.masks_stanford import MasksStanfordTissue
from BodyComposition.actions.masks_totalsegmentator import MasksTotalSegmentatorSpine
from BodyComposition.utils.geometry import GeometryError


def _disable_tissue_filters(config):
    for tissue in ("imat", "sm", "vat", "sat"):
        config["tissue"][tissue]["filter_hu"] = False


def test_stanford_postprocessing_does_not_mutate_raw_label(
    tmp_path,
    pipeline_stub,
    container_factory,
):
    _disable_tissue_filters(pipeline_stub.config)
    action = MasksStanfordTissue(pipeline_stub, image="tmp/index")
    raw = np.zeros((2, 3, 4), dtype=np.uint8)
    raw[0, 1, 2] = 3
    original = raw.copy()
    image = container_factory(
        tmp_path / "images" / "ct.nii.gz",
        np.zeros_like(raw, dtype=np.int16),
    )
    label = container_factory(tmp_path / "labels" / "tissue.nii.gz", raw)
    memory = {
        "id": "case",
        "workspace": tmp_path,
        "tmp/index": image,
        action.input_label_tissue_name: label,
    }

    action(memory)

    output = memory[action.output_mask_name]
    assert np.array_equal(label.data, original)
    assert output.data[0, 1, 2] == action.LBL_TISSUE_R["SM"]
    assert output.geometry.equivalent_to(label.geometry)


def test_tissue_postprocessing_rejects_misaligned_image_and_label(
    tmp_path,
    pipeline_stub,
    container_factory,
):
    _disable_tissue_filters(pipeline_stub.config)
    action = MasksStanfordTissue(pipeline_stub, image="tmp/index")
    array = np.zeros((2, 3, 4), dtype=np.uint8)
    image = container_factory(tmp_path / "images" / "ct.nii.gz", array.astype(np.int16))
    label = container_factory(
        tmp_path / "labels" / "tissue.nii.gz",
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


def test_internal_size_filter_removes_small_eligible_tissue_islands_after_intersection(
    tmp_path,
    pipeline_stub,
    container_factory,
):
    _disable_tissue_filters(pipeline_stub.config)
    settings = pipeline_stub.config["tissue"]
    settings["sm"].update(
        {
            "filter_hu": True,
            "filter_size": True,
            "filter_size_version": "2D",
            "filter_size_2D": 2,
        }
    )
    action = MasksInternalTissue(pipeline_stub, image="tmp/index")
    labels = np.zeros((1, 10, 10), dtype=np.uint8)
    labels[0, 2, 2] = action.LBL_TISSUE_R["SM"]
    labels[0, 5:7, 5:7] = action.LBL_TISSUE_R["SM"]
    image_array = np.zeros_like(labels, dtype=np.int16)
    image_array[labels != 0] = 40
    image = container_factory(tmp_path / "images" / "ct.nii.gz", image_array)
    label = container_factory(tmp_path / "labels" / "tissue.nii.gz", labels)
    memory = {
        "id": "case",
        "workspace": tmp_path,
        "tmp/index": image,
        action.input_label_tissue_name: label,
    }

    action(memory)

    output = memory[action.output_mask_name].data
    assert output[0, 2, 2] == 0
    assert np.all(output[0, 5:7, 5:7] == action.LBL_TISSUE_R["SM"])


def test_imat_size_filter_does_not_connect_through_pixels_outside_muscle_compartment(
    tmp_path,
    pipeline_stub,
    container_factory,
):
    _disable_tissue_filters(pipeline_stub.config)
    settings = pipeline_stub.config["tissue"]
    settings["imat"].update(
        {
            "filter_hu": True,
            "filter_size": True,
            "filter_size_version": "2D",
            "filter_size_2D": 2,
        }
    )
    action = MasksInternalTissue(pipeline_stub, image="tmp/index")
    labels = np.zeros((1, 8, 8), dtype=np.uint8)
    labels[0, 4, 4] = action.LBL_TISSUE_R["SM"]
    image_array = np.zeros_like(labels, dtype=np.int16)
    image_array[0, 3:6, 3:6] = -100
    image = container_factory(tmp_path / "images" / "ct.nii.gz", image_array)
    label = container_factory(tmp_path / "labels" / "tissue.nii.gz", labels)
    memory = {
        "id": "case",
        "workspace": tmp_path,
        "tmp/index": image,
        action.input_label_tissue_name: label,
    }

    action(memory)

    assert memory[action.output_mask_name].data[0, 4, 4] == action.LBL_TISSUE_R["SM"]


def test_total_segmentator_spine_preserves_raw_labels_and_sacrum(
    tmp_path,
    pipeline_stub,
    container_factory,
):
    action = MasksTotalSegmentatorSpine(pipeline_stub, reduce_to_vb=True)
    raw = np.array([[[29, 25, 0]]], dtype=np.uint8)
    original = raw.copy()
    vertebral_bodies = np.array([[[1, 0, 0]]], dtype=np.uint8)
    spine = container_factory(tmp_path / "labels" / "spine.nii.gz", raw)
    bodies = container_factory(
        tmp_path / "labels" / "bodies.nii.gz",
        vertebral_bodies,
    )
    memory = {
        "id": "case",
        "workspace": tmp_path,
        action.input_label_name: spine,
        action.input_label_vb_name: bodies,
    }

    action(memory)

    output = memory[action.output_mask_name]
    assert np.array_equal(spine.data, original)
    assert np.array_equal(
        output.data,
        np.array([[[15, action.LBL_VERTEBRALBODIES_SACRUM, 0]]], dtype=np.uint8),
    )
