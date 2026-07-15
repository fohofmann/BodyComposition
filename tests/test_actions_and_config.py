from copy import deepcopy
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from BodyComposition.actions.calc_measures import CalcMeasures
from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
from BodyComposition.actions.crop import ApplyBoundingBox, CreateBoundingBox
from BodyComposition.actions.data_postprocessing import DataCombine, L3MeanCSA
from BodyComposition.actions.segm_int import SegmIntVertebrae
from BodyComposition.pipeline import ActionContractError, PipelineAction
from BodyComposition.utils.config import ConfigError, validate_config
from BodyComposition.utils.geometry import GeometryError, ImageGeometry
from BodyComposition.utils.logging import log_license


class WritesDeclaredOutput(PipelineAction):
    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.io_inputs = ["tmp/input"]
        self.io_outputs = ["tmp/output"]

    def __call__(self, memory):
        super().__call__(memory)
        memory["tmp/output"] = memory["tmp/input"]


class MisspellsDeclaredOutput(WritesDeclaredOutput):
    def __call__(self, memory):
        PipelineAction.__call__(self, memory)
        memory["tmp/ouptut"] = memory["tmp/input"]


def test_action_contract_catches_missing_input_and_misspelled_output(pipeline_stub, tmp_path):
    memory = {"id": "case-001", "workspace": tmp_path}
    action = WritesDeclaredOutput(pipeline_stub)
    with pytest.raises(ActionContractError, match="tmp/input"):
        action(memory)

    memory["tmp/input"] = 42
    action(memory)
    action.validate_outputs(memory)
    assert memory["tmp/output"] == 42

    misspelled = MisspellsDeclaredOutput(pipeline_stub)
    memory.pop("tmp/output")
    misspelled(memory)
    with pytest.raises(ActionContractError, match="tmp/output"):
        misspelled.validate_outputs(memory)


def test_empty_vertebral_mask_emits_schema_valid_status(
    pipeline_stub,
    tmp_path,
    container_factory,
):
    container = container_factory(
        tmp_path / "masks" / "vertebrae.nii.gz",
        np.zeros((4, 6, 8), dtype=np.uint8),
        spacing_xyz=(0.8, 1.2, 3.0),
    )
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "masks/{caseid}_vertebrae.nii.gz": container,
    }
    action = CalcVertebralLevel(pipeline_stub, mask="masks/{caseid}_vertebrae.nii.gz")
    action(memory)
    action.validate_outputs(memory)

    output = memory["tmp/vertebrae_values"]
    assert len(output) == 4
    assert output["Level"].isna().all()
    assert output["VertebraStatus"].eq("empty_mask").all()


def test_missing_l3_produces_one_schema_valid_summary_row(pipeline_stub, tmp_path):
    input_df = pd.DataFrame(
        {
            "Level": ["L2", "L4"],
            "CSA_SM": [10.0, 12.0],
            "CSA_SAT": [20.0, 24.0],
        }
    )
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "tmp/bodycomposition": input_df,
    }
    action = L3MeanCSA(pipeline_stub)
    action(memory)
    action.validate_outputs(memory)

    output = memory["res/L3MeanCSA"]
    assert len(output) == 1
    assert output.loc[0, "status"] == "not_available"
    assert output.loc[0, "reason"] == "missing_L3"
    assert np.isnan(output.loc[0, "CSA_SM"])


def test_calc_measures_rejects_misaligned_ct_and_mask(
    pipeline_stub,
    tmp_path,
    container_factory,
):
    mask = container_factory(
        tmp_path / "masks" / "tissue.nii.gz",
        np.ones((2, 4, 5), dtype=np.uint8),
        spacing_xyz=(1.0, 1.0, 2.0),
    )
    image = container_factory(
        tmp_path / "images" / "ct.nii.gz",
        np.ones((2, 4, 5), dtype=np.int16),
        spacing_xyz=(1.1, 1.0, 2.0),
    )
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "masks/{caseid}_tissue.nii.gz": mask,
        "tmp/index": image,
    }
    action = CalcMeasures(
        pipeline_stub,
        mask="masks/{caseid}_tissue.nii.gz",
        image="tmp/index",
        calculate_hu=True,
    )
    with pytest.raises(GeometryError, match="spacing_xyz differ"):
        action(memory)


def test_data_combine_rejects_different_physical_domains(pipeline_stub, tmp_path):
    direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "tmp/tissue_values": pd.DataFrame({"CSA_SM": [1.0]}),
        "tmp/vertebrae_values": pd.DataFrame({"Slice": [0], "Level": ["L3"]}),
        "tmp/tissue_geometry": ImageGeometry((5, 4, 1), (1, 1, 2), (0, 0, 0), direction),
        "tmp/vertebrae_geometry": ImageGeometry((6, 4, 1), (1, 1, 2), (0, 0, 0), direction),
    }
    with pytest.raises(GeometryError, match="sizes differ"):
        DataCombine(pipeline_stub)(memory)


def test_bounding_box_converts_zyx_margins_from_xyz_spacing(
    pipeline_stub,
    tmp_path,
    container_factory,
):
    label_array = np.zeros((10, 20, 30), dtype=np.uint8)
    label_array[3:5, 6:10, 8:12] = 15
    label = container_factory(
        tmp_path / "masks" / "vertebrae.nii.gz",
        label_array,
        spacing_xyz=(0.5, 1.0, 4.0),
    )
    pipeline_stub.config["crop"]["Test"] = {
        "roi": [15],
        "axes": [True, True, False, False, False, False],
        "margin": [4, 8, 0, 0, 0, 0],
    }
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "masks/{caseid}_vertebrae.nii.gz": label,
    }
    action = CreateBoundingBox(
        pipeline_stub,
        label="masks/{caseid}_vertebrae.nii.gz",
        task="Test",
    )
    action(memory)
    action.validate_outputs(memory)
    assert memory["bbox"] == [2, 7, 0, 20, 0, 30]


def test_bounding_box_rejects_a_different_physical_domain(
    pipeline_stub,
    tmp_path,
    container_factory,
):
    label_array = np.zeros((3, 4, 5), dtype=np.uint8)
    label_array[1, 1:3, 1:4] = 15
    label = container_factory(tmp_path / "masks" / "vertebrae.nii.gz", label_array)
    image = container_factory(
        tmp_path / "images" / "ct.nii.gz",
        np.zeros_like(label_array, dtype=np.int16),
        origin_lps_xyz=(1.0, 0.0, 0.0),
    )
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "masks/{caseid}_vertebrae.nii.gz": label,
        "tmp/index": image,
    }
    CreateBoundingBox(
        pipeline_stub,
        label="masks/{caseid}_vertebrae.nii.gz",
        task="L3CranioCaudal",
    )(memory)

    with pytest.raises(GeometryError, match="origin_lps_xyz differ"):
        ApplyBoundingBox(pipeline_stub, input="tmp/index")(memory)


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (("tissue", "sm", "filter_size"), "Trze"),
        (("run", "skip"), "True"),
        (("run", "timeout"), None),
        (("orientation", "confidence", "allow_axial_180_repair"), "False"),
    ],
)
def test_configuration_rejects_typographical_and_untyped_values(
    base_config,
    path,
    invalid_value,
):
    config = deepcopy(base_config)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value
    with pytest.raises(ConfigError):
        validate_config(config)


def test_ctdeeprot_runtime_acknowledgement_loads_from_installed_package(
    caplog,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        log_license(["ctdeeprot"])

    assert (
        "CTDeepRot (required model for the default orientation-integrity stage)"
        in caplog.text
    )
    assert "BSD-3-Clause code" in caplog.text
    assert "Jakubicek R, Vicar T, Chmelik J" in caplog.text


def test_internal_segmentation_skip_does_not_load_predictor(
    pipeline_stub,
    tmp_path,
    container_factory,
):
    input_image = container_factory(
        tmp_path / "images" / "ct.nii.gz",
        np.ones((2, 4, 5), dtype=np.int16),
        spacing_xyz=(0.7, 1.2, 3.4),
    )
    existing_output = container_factory(
        tmp_path / "labels" / "case-001_int-vertebrae.nii.gz",
        np.zeros((2, 4, 5), dtype=np.uint8),
        spacing_xyz=(0.7, 1.2, 3.4),
    )
    existing_output.save_to_file()
    pipeline_stub.config["run"]["skip"] = True
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "tmp/index": input_image,
    }

    action = SegmIntVertebrae(pipeline_stub, image="tmp/index")

    def fail_if_loaded():
        raise AssertionError("predictor must remain lazy when an output is skipped")

    action._get_predictor = fail_if_loaded
    action(memory)
    action.validate_outputs(memory)

    assert action.predictor is None
    assert memory[action.output_label_name].path == existing_output.path


def test_internal_segmentation_passes_numpy_axis_spacing_to_nnunet(
    pipeline_stub,
    tmp_path,
    container_factory,
):
    class FakePredictor:
        def __init__(self):
            self.input_shape = None
            self.properties = None

        def predict_single_npy_array(
            self,
            input_array,
            properties,
            segmentation_previous_stage,
            output_file_truncated,
            save_or_return_probabilities,
        ):
            self.input_shape = input_array.shape
            self.properties = properties
            return np.zeros(input_array.shape[1:], dtype=np.uint8)

    input_image = container_factory(
        tmp_path / "images" / "ct.nii.gz",
        np.ones((2, 4, 5), dtype=np.int16),
        spacing_xyz=(0.7, 1.2, 3.4),
    )
    pipeline_stub.config["run"]["skip"] = False
    pipeline_stub.config["segmentation"]["save_label"] = False
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "tmp/index": input_image,
    }
    predictor = FakePredictor()
    action = SegmIntVertebrae(pipeline_stub, image="tmp/index")
    action.predictor = predictor

    action(memory)
    action.validate_outputs(memory)

    output_label = memory[action.output_label_name]
    assert predictor.input_shape == (1, 2, 4, 5)
    assert predictor.properties == {"spacing": (3.4, 1.2, 0.7)}
    assert output_label.geometry.equivalent_to(input_image.geometry)


def test_internal_segmentation_retries_cudnn_engine_failure(
    pipeline_stub,
    tmp_path,
    container_factory,
    monkeypatch,
    caplog,
):
    class FakePredictor:
        def __init__(self):
            self.calls = 0

        def predict_single_npy_array(self, input_array, *args):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(
                    "FIND was unable to find an engine to execute this computation"
                )
            return np.zeros(input_array.shape[1:], dtype=np.uint8)

    cudnn = SimpleNamespace(enabled=True)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(backends=SimpleNamespace(cudnn=cudnn)),
    )
    pipeline_stub.device = SimpleNamespace(type="cuda")
    pipeline_stub.config["run"]["skip"] = False
    pipeline_stub.config["segmentation"]["save_label"] = False
    input_image = container_factory(
        tmp_path / "images" / "ct.nii.gz",
        np.ones((2, 4, 5), dtype=np.int16),
        spacing_xyz=(0.7, 1.2, 3.4),
    )
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "tmp/index": input_image,
    }
    predictor = FakePredictor()
    action = SegmIntVertebrae(pipeline_stub, image="tmp/index")
    action.predictor = predictor

    with caplog.at_level("WARNING"):
        action(memory)

    assert predictor.calls == 2
    assert cudnn.enabled is False
    assert "retrying with cuDNN disabled" in caplog.text
