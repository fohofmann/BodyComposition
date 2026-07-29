from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from BodyComposition.actions.masks_int import MasksInternalTissue
from BodyComposition.config import PipelineConfig
from BodyComposition.utils.geometry import GeometryError


def test_consensus_materialization_rejects_misaligned_image_and_label(
    tmp_path,
    pipeline_stub,
    container_factory,
):
    action = MasksInternalTissue(pipeline_stub, image="tmp/index")
    array = np.zeros((2, 3, 4), dtype=np.uint8)
    image = container_factory(
        tmp_path / "images" / "ct.nii.gz",
        array.astype(np.int16),
    )
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


def test_consensus_materialization_rejects_non_native_label(
    tmp_path,
    pipeline_stub,
    container_factory,
):
    action = MasksInternalTissue(pipeline_stub, image="tmp/index")
    image = container_factory(
        tmp_path / "images" / "ct.nii.gz",
        np.zeros((1, 3, 3), dtype=np.int16),
    )
    labels = np.zeros((1, 3, 3), dtype=np.uint8)
    labels[0, 1, 1] = 8
    label = container_factory(
        tmp_path / "masks" / "tissue_compartments.nii.gz",
        labels,
    )

    with pytest.raises(ValueError, match="outside the 0-7 contract: \\[8\\]"):
        action(
            {
                "id": "case",
                "workspace": tmp_path,
                "tmp/index": image,
                action.input_label_tissue_name: label,
            }
        )


def _run_consensus_materialization(
    *,
    tmp_path,
    pipeline_stub,
    container_factory,
    public_config,
    suffix,
):
    pipeline_stub.public_config = public_config
    pipeline_stub.config = public_config.to_runtime_dict()
    action = MasksInternalTissue(pipeline_stub, image="tmp/index")
    labels = np.zeros((1, 5, 6), dtype=np.uint8)
    labels[0, 1, 1:5] = 1
    labels[0, 2, 1:5] = 3
    labels[0, 3, 1:5] = 4
    labels[0, 4, 1:4] = (2, 6, 7)
    image_array = np.zeros_like(labels, dtype=np.int16)
    image_array[0, 1, 1:5] = (-30, -29, 150, 151)
    image_array[0, 2, 1:5] = (-191, -190, -30, -29)
    image_array[0, 3, 1:5] = (-191, -190, -30, -29)
    image = container_factory(
        tmp_path / suffix / "images" / "ct.nii.gz",
        image_array,
    )
    label = container_factory(
        tmp_path / suffix / "masks" / "tissue_compartments.nii.gz",
        labels,
    )
    workspace = tmp_path / suffix
    workspace.mkdir(parents=True, exist_ok=True)
    memory = {
        "id": "case",
        "workspace": workspace,
        "tmp/index": image,
        action.input_label_tissue_name: label,
    }

    action(memory)
    return memory[action.output_mask_name].data.copy(), labels


def test_consensus_materialization_uses_inclusive_consensus_windows(
    tmp_path,
    pipeline_stub,
    container_factory,
):
    output, raw = _run_consensus_materialization(
        tmp_path=tmp_path,
        pipeline_stub=pipeline_stub,
        container_factory=container_factory,
        public_config=PipelineConfig.model_validate(
            {"runtime": {"allow_dirty": True}}
        ),
        suffix="consensus",
    )

    assert np.array_equal(output[0, 1, 1:5], [0, 1, 1, 0])
    assert np.array_equal(output[0, 2, 1:5], [0, 3, 3, 0])
    assert np.array_equal(output[0, 3, 1:5], [0, 4, 4, 0])
    assert np.array_equal(output[0, 4, 1:4], [0, 0, 0])
    assert np.array_equal(
        raw[0, 1:5, 1:5],
        np.array(
            [
                [1, 1, 1, 1],
                [3, 3, 3, 3],
                [4, 4, 4, 4],
                [2, 6, 7, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_sensitivity_profile_does_not_change_consensus_label_artifact(
    tmp_path,
    pipeline_stub,
    container_factory,
):
    baseline, _ = _run_consensus_materialization(
        tmp_path=tmp_path,
        pipeline_stub=pipeline_stub,
        container_factory=container_factory,
        public_config=PipelineConfig.model_validate(
            {"runtime": {"allow_dirty": True}}
        ),
        suffix="baseline",
    )
    profiled_config = PipelineConfig.load(
        Path("config/tissue_profiles/boa_median_3x3.yaml")
    )
    profiled = profiled_config.model_validate(
        {
            **deepcopy(profiled_config.normalized()),
            "runtime": {
                **profiled_config.normalized()["runtime"],
                "allow_dirty": True,
            },
        }
    )
    sensitivity, _ = _run_consensus_materialization(
        tmp_path=tmp_path,
        pipeline_stub=pipeline_stub,
        container_factory=container_factory,
        public_config=profiled,
        suffix="sensitivity",
    )

    assert np.array_equal(sensitivity, baseline)
