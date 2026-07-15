from io import StringIO
import subprocess
import sys
import types

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import SimpleITK as sitk
import BodyComposition.python_api as python_api

from BodyComposition.actions.calc_measures import CalcMeasures
from BodyComposition.actions.calc_vertebrallevel import CalcVertebralLevel
from BodyComposition.actions.data_loading import LoadNifti
from BodyComposition.actions.data_postprocessing import DataCombine, DataExport
from BodyComposition.pipeline import (
    PipelineAction,
    PipelineBuilder,
    PipelineExecutionError,
    run_batch,
)
from BodyComposition.pipeline_registry import pipeline_registry
from BodyComposition.pipelines.bodycomposition import BodyCompositionFast
from BodyComposition.utils.datalist import DatalistBuilder


def test_model_free_synthetic_pipeline_loads_measures_combines_and_exports(
    pipeline_stub,
    tmp_path,
    sitk_image_factory,
):
    case_id = "synthetic-001"
    shape_zyx = (4, 12, 14)
    geometry = {
        "spacing_xyz": (0.8, 1.2, 3.0),
        "origin_lps_xyz": (10.0, -5.0, 20.0),
        "direction_lps": (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
    }

    image = np.full(shape_zyx, -100, dtype=np.int16)
    tissue = np.zeros(shape_zyx, dtype=np.uint8)
    tissue[:, 2:8, 3:9] = 1
    tissue[:, 8:10, 2:12] = 3
    image[tissue == 1] = 45
    image[tissue == 3] = -90
    vertebrae = np.zeros(shape_zyx, dtype=np.uint8)
    vertebrae[:, 3:10, 4:11] = 15

    paths = {
        "images/{caseid}.nii.gz": image,
        "masks/{caseid}_tissue.nii.gz": tissue,
        "masks/{caseid}_vertebrae.nii.gz": vertebrae,
    }
    for template, array in paths.items():
        path = tmp_path / template.format(caseid=case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk_image_factory(array, **geometry), str(path))

    pipeline_stub.config["vertebrae"]["min_voxels_per_vertebra"] = 1
    actions = [
        LoadNifti(pipeline_stub, io_inputs=list(paths)),
        CalcVertebralLevel(pipeline_stub, mask="masks/{caseid}_vertebrae.nii.gz"),
        CalcMeasures(
            pipeline_stub,
            mask="masks/{caseid}_tissue.nii.gz",
            image="images/{caseid}.nii.gz",
            contour_mask="masks/{caseid}_tissue.nii.gz",
            calculate_hu=True,
            calculate_contours=True,
        ),
        DataCombine(pipeline_stub),
        DataExport(pipeline_stub),
    ]
    memory = {"id": case_id, "workspace": tmp_path}
    for action in actions:
        action(memory)
        action.validate_outputs(memory)

    result = memory["tmp/bodycomposition"]
    assert len(result) == shape_zyx[0]
    assert result["Slice"].tolist() == [0, 1, 2, 3]
    assert result["Level"].eq("L3").all()
    assert result["HU_SM"].eq(45).all()
    assert result["HU_SAT"].eq(-90).all()

    exported = pd.read_csv(tmp_path / "exports" / f"{case_id}_raw.csv")
    csv_normalized_result = pd.read_csv(StringIO(result.to_csv(index=False)))
    assert_frame_equal(exported, csv_normalized_result, check_dtype=False)


class ModelActionStub(PipelineAction):
    def __init__(self, pipeline, *args, **kwargs):
        super().__init__(pipeline)


def _stub_model_modules(monkeypatch):
    modules = {
        "BodyComposition.actions.segm_int": (
            "SegmIntVertebrae",
            "SegmIntBodyComposition",
        ),
        "BodyComposition.actions.segm_totalsegmentator": (
            "SegmTotalSegmentatorConfig",
            "SegmTotalSegmentator",
        ),
        "BodyComposition.actions.segm_stanford": (
            "SegmStanfordSpine",
            "SegmStanfordTissue",
        ),
        "BodyComposition.actions.segm_boa": ("SegmBOA",),
    }
    for module_name, class_names in modules.items():
        module = types.ModuleType(module_name)
        for class_name in class_names:
            setattr(module, class_name, ModelActionStub)
        monkeypatch.setitem(sys.modules, module_name, module)


def test_every_registered_pipeline_instantiates_with_mocked_model_adapters(
    pipeline_stub,
    monkeypatch,
):
    _stub_model_modules(monkeypatch)
    for name, factory in pipeline_registry.items():
        actions = factory(pipeline_stub)
        assert actions, name
        assert all(isinstance(action, PipelineAction) for action in actions), name


def test_cli_help_and_python_entrypoint_import_without_models():
    completed = subprocess.run(
        [sys.executable, "-m", "BodyComposition.bin.run_batch", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Run batch through pipeline" in completed.stdout

    from BodyComposition.python_api import bodycomposition

    assert callable(bodycomposition)


def test_python_api_accepts_direct_image_when_no_external_inputs_are_required(
    monkeypatch,
    tmp_path,
):
    class PipelineStub:
        def get_io(self):
            return [], []

        def get_licenses(self):
            return []

    pipeline = PipelineStub()
    image = sitk.Image(3, 4, 2, sitk.sitkInt16)
    captured = {}

    monkeypatch.setattr(python_api, "PipelineBuilder", lambda **kwargs: pipeline)
    monkeypatch.setattr(python_api, "init_logging", lambda **kwargs: None)
    monkeypatch.setattr(python_api, "log_license", lambda licenses: None)

    def fake_run_file(selected_pipeline, selected_image, workspace):
        captured["pipeline"] = selected_pipeline
        captured["image"] = selected_image
        captured["workspace"] = workspace
        return "completed"

    monkeypatch.setattr(python_api, "run_file", fake_run_file)

    result = python_api.bodycomposition(image, workspace=tmp_path)

    assert result == "completed"
    assert captured == {
        "pipeline": pipeline,
        "image": image,
        "workspace": tmp_path,
    }


def test_fast_pipeline_completion_markers_only_include_persisted_files(pipeline_stub):
    pipeline_stub.config["segmentation"]["save_label"] = False
    pipeline_stub.config["vertebrae"]["save_mask"] = False
    pipeline_stub.config["tissue"]["save_mask"] = True

    builder = object.__new__(PipelineBuilder)
    builder.actions = BodyCompositionFast(pipeline_stub)

    io_inputs, io_outputs = builder.get_io()

    assert io_inputs == []
    assert io_outputs == [
        "masks/{caseid}_vertebral-bodies.nii.gz",
        "qc/{caseid}_vertebral-result.json",
        "masks/{caseid}_spineps-semantic.nii.gz",
        "masks/{caseid}_spineps-vertebrae.nii.gz",
        "qc/{caseid}_spine-review.png",
        "masks/{caseid}_int-bodycomposition.nii.gz",
        "exports/{caseid}_raw.csv",
    ]
    assert builder.get_reset_outputs() == [
        "masks/{caseid}_vertebral-bodies.nii.gz",
        "masks/{caseid}_spineps-semantic.nii.gz",
        "masks/{caseid}_spineps-vertebrae.nii.gz",
        "qc/{caseid}_vertebral-result.json",
        "qc/{caseid}_spine-review.png",
        "labels/{caseid}_int-bodycomposition.nii.gz",
        "masks/{caseid}_int-bodycomposition.nii.gz",
        "exports/{caseid}_raw.csv",
        "exports/all_L3Mean.csv",
    ]


def test_append_export_declares_output_without_using_it_as_completion_marker(
    pipeline_stub,
    tmp_path,
):
    action = DataExport(pipeline_stub, file="exports/all.csv", append=True)
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "tmp/bodycomposition": pd.DataFrame({"CSA_SM": [12.5]}),
    }

    action(memory)
    action.validate_outputs(memory)

    assert action.io_outputs == ["exports/all.csv"]
    assert action.io_persisted_outputs == []
    assert pd.read_csv(tmp_path / "exports/all.csv").loc[0, "CSA_SM"] == 12.5


def test_batch_without_persisted_outputs_is_not_mistaken_for_complete(tmp_path):
    builder = object.__new__(DatalistBuilder)
    builder.cases = [("case-001", tmp_path / "image.nii.gz", tmp_path)]
    builder.io_outputs = []

    builder.skip_completed()

    assert len(builder) == 1


def test_reset_removes_case_and_shared_outputs_without_changing_completion_markers(
    tmp_path,
):
    case_output = tmp_path / "exports/case-001_raw.csv"
    shared_output = tmp_path / "exports/all.csv"
    case_output.parent.mkdir(parents=True)
    case_output.write_text("case\n", encoding="utf-8")
    shared_output.write_text("all\n", encoding="utf-8")

    builder = object.__new__(DatalistBuilder)
    builder.cases = [("case-001", tmp_path / "image.nii.gz", tmp_path)]
    builder.io_outputs = ["exports/{caseid}_raw.csv"]
    builder.io_reset_outputs = ["exports/{caseid}_raw.csv", "exports/all.csv"]

    builder.reset_outputs()

    assert not case_output.exists()
    assert not shared_output.exists()
    assert builder.io_outputs == ["exports/{caseid}_raw.csv"]


def test_batch_surfaces_case_failures_after_logging(pipeline_stub, tmp_path):
    class FailingPipeline:
        config = pipeline_stub.config

        def __call__(self, memory):
            raise RuntimeError("synthetic case failure")

    datalist = [("case-001", tmp_path / "case-001.nii.gz", tmp_path)]

    with pytest.raises(PipelineExecutionError, match="case-001"):
        run_batch(FailingPipeline(), datalist)
