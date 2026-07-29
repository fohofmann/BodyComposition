from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

import BodyComposition.actions.segm_int as segm_module
from BodyComposition import cli
from BodyComposition.actions.measurement import MeasureCanonicalBodyComposition
from BodyComposition.actions.region import (
    L3_ANALYSIS_REGION,
    L3_TISSUE_INPUT,
    L3RegionUnavailableError,
    PrepareL3TissueRegion,
)
from BodyComposition.actions.segm_int import SegmIntBodyComposition
from BodyComposition.config import ConfigError, low_resource_config
from BodyComposition.measurement.aggregation import aggregate_physical_range
from BodyComposition.measurement.builder import build_measurement_bundle
from BodyComposition.measurement.contracts import (
    BodySurfaceResult,
    MeasurementIdentity,
)
from BodyComposition.measurement.slices import calculate_canonical_slice_measurements
from BodyComposition.model_manager import required_model_ids
from BodyComposition.pipelines.bodycomposition import canonical_actions
from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.utils.regions import ImageRegion
from BodyComposition.vertebral.contracts import ExecutionStatus, VertebralResult


def test_image_region_preserves_zyx_pixels_and_affine_geometry(
    tmp_path,
    container_factory,
):
    angle = np.deg2rad(12.0)
    direction = (
        1.0,
        0.0,
        0.0,
        0.0,
        float(np.cos(angle)),
        float(-np.sin(angle)),
        0.0,
        float(np.sin(angle)),
        float(np.cos(angle)),
    )
    source_array = np.arange(8 * 5 * 4, dtype=np.int16).reshape(8, 5, 4)
    source = container_factory(
        tmp_path / "source.nii.gz",
        source_array,
        spacing_xyz=(1.2, 2.3, 3.4),
        origin_lps_xyz=(11.0, -7.0, 23.0),
        direction_lps=direction,
    )
    region = ImageRegion(
        source_geometry=source.geometry,
        start_xyz=(0, 0, 2),
        size_xyz=(4, 5, 3),
    )

    cropped_array = region.extract_array(source.data)
    cropped_image = region.extract_image(source.img)

    assert np.array_equal(cropped_array, source_array[2:5])
    assert np.array_equal(
        cropped_array,
        sitk.GetArrayFromImage(cropped_image),
    )
    assert ImageGeometry.from_sitk(cropped_image).equivalent_to(region.geometry)
    assert region.geometry.origin_lps_xyz == pytest.approx(
        source.geometry.physical_point((0, 0, 2))
    )
    restored = region.restore_array(cropped_array)
    assert restored.shape == source_array.shape
    assert np.array_equal(restored[2:5], source_array[2:5])
    assert not np.any(restored[:2])
    assert not np.any(restored[5:])


def _l3_inputs(tmp_path, container_factory):
    image_zyx = np.arange(20 * 6 * 8, dtype=np.int16).reshape(20, 6, 8)
    source = container_factory(
        tmp_path / "ct.nii.gz",
        image_zyx,
        spacing_xyz=(1.0, 1.0, 10.0),
        origin_lps_xyz=(3.0, -4.0, 0.0),
    )
    vertebrae = np.zeros_like(image_zyx, dtype=np.uint8)
    vertebrae[12:14, 1:5, 2:6] = 14
    vertebrae[8:11, 1:5, 2:6] = 15
    vertebrae[5:7, 1:5, 2:6] = 16
    result = VertebralResult(
        backend_id="vertebral_bodies_resenc_m",
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=source.geometry,
        whole_vertebra_labels=None,
        vertebral_body_labels=vertebrae,
        label_schema={14: "L2", 15: "L3", 16: "L4"},
    )
    config = low_resource_config(
        {
            "analysis": {"l3": {"inference_context_mm": 20.0}},
            "runtime": {"allow_dirty": True},
        }
    )
    pipeline = SimpleNamespace(
        config=config.to_runtime_dict(),
        public_config=config,
        timestamp=123,
        device="cpu",
    )
    memory = {
        "id": "case",
        "workspace": tmp_path,
        "tmp/index": source,
        "tmp/prepared_image": SimpleNamespace(prepared_image=source.img),
        "tmp/vertebral_result": result,
    }
    return pipeline, memory


def test_l3_region_uses_full_axial_fov_and_restores_only_target_slices(
    tmp_path,
    container_factory,
    monkeypatch,
):
    pipeline, memory = _l3_inputs(tmp_path, container_factory)
    prepare = PrepareL3TissueRegion(pipeline)
    prepare(memory)
    region = memory[L3_ANALYSIS_REGION]
    cropped = memory[L3_TISSUE_INPUT]

    assert cropped.shape[1:] == memory["tmp/index"].shape[1:]
    assert cropped.shape[0] < memory["tmp/index"].shape[0]
    assert region.inference_region.as_dict()["operation"] == "index_crop_no_resampling"
    assert ImageGeometry.from_container(cropped).equivalent_to(
        region.inference_region.geometry
    )

    action = SegmIntBodyComposition(
        pipeline,
        image=L3_TISSUE_INPUT,
        model="ResEncM",
        analysis_region=L3_ANALYSIS_REGION,
        restore_reference="tmp/index",
    )
    monkeypatch.setattr(action, "_get_predictor", lambda: object())
    monkeypatch.setattr(
        action,
        "_predict",
        lambda predictor, input_image, spacing_zyx: np.ones(
            input_image.shape,
            dtype=np.uint8,
        ),
    )
    monkeypatch.setattr(segm_module, "log_gpu_usage", lambda device: None)
    action(memory)

    output = memory["masks/tissue_compartments.nii.gz"]
    available = region.analyzed_slices_mask_z
    assert output.shape == memory["tmp/index"].shape
    assert output.geometry.equivalent_to(memory["tmp/index"].geometry)
    assert np.all(output.data[available] == 1)
    assert not np.any(output.data[~available])


def test_l3_region_fails_explicitly_when_l3_is_not_available(
    tmp_path,
    container_factory,
):
    pipeline, memory = _l3_inputs(tmp_path, container_factory)
    result = memory["tmp/vertebral_result"]
    labels = np.array(result.vertebral_body_labels, copy=True)
    labels[labels == 15] = 0
    memory["tmp/vertebral_result"] = VertebralResult(
        backend_id=result.backend_id,
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=result.geometry,
        whole_vertebra_labels=None,
        vertebral_body_labels=labels,
        label_schema={14: "L2", 16: "L4"},
    )

    with pytest.raises(L3RegionUnavailableError, match="could not identify a valid L3"):
        PrepareL3TissueRegion(pipeline)(memory)

    assert L3_TISSUE_INPUT not in memory
    assert L3_ANALYSIS_REGION not in memory


def test_low_resource_preset_has_only_required_small_model_stack():
    config = low_resource_config()

    assert config.normalized()["analysis"]["scope"] == "l3_vertebral_level"
    assert config.vertebral_backend == "vertebral_bodies_resenc_m"
    assert config.tissue_backend == "bodycomposition_resenc_m_v1"
    assert not config.normalized()["measurements"]["landmarks"]["enabled"]
    assert config.normalized()["runtime"]["unload_models_between_stages"]
    assert required_model_ids(config) == (
        "ctdeeprot_2d_v1",
        "vertebral_bodies_resenc_m",
        "bodycomposition_resenc_m_v1",
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_low_resource_context_must_be_finite_and_nonnegative(value):
    with pytest.raises(ConfigError, match="inference_context_mm"):
        low_resource_config({"analysis": {"l3": {"inference_context_mm": value}}})


def test_cli_low_resource_shortcut_resolves_the_same_preset():
    args = cli.build_parser().parse_args(
        ["analyze", "CT.nii.gz", "--low-resource", "--device", "cpu"]
    )
    config = cli._config(args)

    assert config.normalized()["analysis"]["scope"] == "l3_vertebral_level"
    assert config.vertebral_backend == "vertebral_bodies_resenc_m"
    assert config.tissue_backend == "bodycomposition_resenc_m_v1"
    assert config.device == "cpu"


def test_low_resource_pipeline_localizes_before_scoped_tissue_inference():
    config = low_resource_config({"runtime": {"allow_dirty": True}})
    pipeline = SimpleNamespace(
        config=config.to_runtime_dict(),
        public_config=config,
        timestamp=123,
        device="cpu",
    )

    actions = canonical_actions(pipeline)
    names = [type(action).__name__ for action in actions]
    tissue = next(action for action in actions if isinstance(action, SegmIntBodyComposition))
    measurement = next(
        action for action in actions if isinstance(action, MeasureCanonicalBodyComposition)
    )

    assert names.index("SegmIntVertebrae") < names.index("PrepareL3TissueRegion")
    assert names.index("PrepareL3TissueRegion") < names.index("SegmIntBodyComposition")
    assert "SegmTotalSegmentator" not in names
    assert tissue.input_image_name == L3_TISSUE_INPUT
    assert tissue.analysis_region_name == L3_ANALYSIS_REGION
    assert tissue.restore_reference_name == "tmp/index"
    assert measurement.analysis_scope == "l3_vertebral_level"
    assert L3_ANALYSIS_REGION in measurement.io_inputs


def test_scoped_slice_measurements_keep_geometry_but_not_false_zero_values():
    geometry = ImageGeometry(
        size_xyz=(7, 7, 4),
        spacing_xyz=(1.0, 1.0, 2.0),
        origin_lps_xyz=(0.0, 0.0, 0.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    image = np.full((4, 7, 7), 40, dtype=np.int16)
    labels = np.zeros_like(image, dtype=np.uint8)
    labels[1:3, 2:5, 2:5] = 1
    body = labels != 0
    surface = BodySurfaceResult(
        backend_id="test",
        geometry=geometry,
        body_mask_zyx=body,
        trunk_mask_zyx=body,
    )
    available = np.asarray([False, True, True, False], dtype=bool)
    table = calculate_canonical_slice_measurements(
        image,
        labels,
        geometry,
        {1: "SM"},
        surface,
        MeasurementIdentity("case", "run", "analysis"),
        compartment_labels_zyx=labels,
        compartment_label_schema={1: "SM"},
        analysis_scope="l3_vertebral_level",
        analyzed_slices_z=available,
    )

    assert len(table) == 4
    unavailable = ~table["body_composition_analysis_available"]
    assert table.loc[unavailable, "sm_compartment_area_cm2"].eq(0).all()
    assert not table.loc[unavailable, "sm_compartment_area_valid"].any()
    assert table.loc[unavailable, "sm_compartment_area_reason"].eq(
        "outside_analysis_region"
    ).all()
    aggregate = aggregate_physical_range(
        table,
        float(table["slice_slab_inferior_mm"].min()),
        float(table["slice_slab_superior_mm"].max()),
    )
    assert aggregate["acquisition_coverage_fraction"] == pytest.approx(1.0)
    assert aggregate["coverage_fraction"] == pytest.approx(0.5)
    assert aggregate["range_missing_reason"] == "partial_analysis_region"


def test_l3_measurement_bundle_retains_full_grid_with_scoped_coverage(
    tmp_path,
    container_factory,
):
    pipeline, memory = _l3_inputs(tmp_path, container_factory)
    PrepareL3TissueRegion(pipeline)(memory)
    region = memory[L3_ANALYSIS_REGION]
    geometry = memory["tmp/index"].geometry
    image = memory["tmp/index"].data
    compartments = np.zeros_like(image, dtype=np.uint8)
    available = region.analyzed_slices_mask_z
    compartments[available, 1:5, 2:6] = 1
    surface_mask = compartments != 0
    bundle = build_measurement_bundle(
        image_zyx=image,
        tissue_labels_zyx=compartments,
        geometry=geometry,
        tissue_label_schema=pipeline.config["LBL_TISSUE"],
        tissue_backend_id="bodycomposition_resenc_m_v1",
        tissue_preprocessing={"analysis_scope": "l3_vertebral_level"},
        compartment_labels_zyx=compartments,
        compartment_label_schema=pipeline.config["LBL_TISSUE_COMPARTMENTS"],
        body_surface=BodySurfaceResult(
            backend_id="test",
            geometry=geometry,
            body_mask_zyx=surface_mask,
            trunk_mask_zyx=surface_mask,
        ),
        vertebral_result=memory["tmp/vertebral_result"],
        identity=MeasurementIdentity("case", "run", "analysis"),
        landmarks=None,
        settings=pipeline.config["measurements"],
        analysis_scope="l3_vertebral_level",
        analyzed_slices_z=available,
        analysis_region_provenance=region.as_dict(),
    )

    assert len(bundle.slices) == image.shape[0]
    assert bundle.slices["body_composition_analysis_available"].sum() == int(available.sum())
    assert bundle.signature["coverage_fraction"].lt(1).any()
    assert (
        bundle.hu_distributions["scope_superior_position_superior_mm"]
        - bundle.hu_distributions["scope_inferior_position_superior_mm"]
    ).max() < geometry.size_xyz[2] * geometry.spacing_xyz[2]
    assert "l3_scoped_tissue_analysis" in {
        flag.code for flag in bundle.qc_flags
    }
    flags = {flag.code: flag for flag in bundle.qc_flags}
    assert flags["l3_scoped_tissue_analysis"].severity.value == "info"
    for code in (
        "l3_200mm_slab_unavailable",
        "minimum_waist_unavailable",
        "pelvic_maximum_unavailable",
    ):
        flag = flags.get(code)
        if flag is not None and flag.observed.get("reason") in {
            "outside_analysis_region",
            "partial_analysis_region",
        }:
            assert flag.severity.value == "info"
    assert "minimum_waist_at_search_boundary" not in flags
    assert "pelvic_maximum_at_search_boundary" not in flags
