import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import jsonschema
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import SimpleITK as sitk

from BodyComposition.actions.measurement import (
    CreateBodySurface,
    ExportMeasurementBundle,
    MeasureCanonicalBodyComposition,
    _scientific_measurement_configuration,
)
from BodyComposition.actions.segm_int import SegmIntBodyComposition
from BodyComposition.actions.segm_totalsegmentator import (
    SegmTotalSegmentator,
    _measurement_model_download_guard,
    _require_measurement_model,
)
from BodyComposition.actions.vertebral import SPINEPS_BODY_MASK
from BodyComposition.bin import measurement_view
from BodyComposition.bin.pre_download_models import (
    configured_pipeline_models,
    definition_pipelines,
)
from BodyComposition.measurement.api import (
    TABLE_NAMES,
    l3_measurements,
    load_measurement_tables,
    range_measurements,
)
from BodyComposition.measurement.body_surface import body_surface_from_totalsegmentator
from BodyComposition.measurement.body_surface import TISSUE_ENVELOPE_BACKEND
from BodyComposition.measurement.builder import (
    build_measurement_bundle,
    measurement_analysis_id,
)
from BodyComposition.measurement.contracts import Landmark, LandmarkSet, MeasurementIdentity
from BodyComposition.measurement import totalsegmentator_assets
from BodyComposition.measurement.review import write_measurement_review
from BodyComposition.pipelines.bodycomposition import BodyCompositionFast
from BodyComposition.utils.config import ConfigError, validate_config
from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.vertebral.contracts import (
    ExecutionStatus,
    QCFlag,
    QCSeverity,
    QCStatus,
    VertebralResult,
)


def make_measurement_inputs(config, *, empty_vertebrae=False):
    shape = (80, 24, 28)
    geometry = ImageGeometry(
        size_xyz=tuple(reversed(shape)),
        spacing_xyz=(0.9, 1.2, 4.0),
        origin_lps_xyz=(10.0, -20.0, 0.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    image = np.zeros(shape, dtype=np.int16)
    tissues = np.zeros(shape, dtype=np.uint8)
    tissues[:, 5:9, 5:9] = 1
    tissues[:, 9:11, 6:10] = 3
    tissues[:, 11:13, 7:9] = 4
    tissues[:, 13:15, 8:10] = 5
    tissues[:, 15:17, 9:11] = 8
    image[tissues == 1] = 40
    image[tissues == 3] = -100
    image[tissues == 4] = -90
    image[tissues == 5] = -70
    image[tissues == 8] = -50

    body_labels = np.zeros(shape, dtype=np.uint8)
    body_labels[:, 2:-2, 2:-2] = 1
    body_labels[20:60, 7:11, 22:26] = 2
    body_surface = body_surface_from_totalsegmentator(body_labels, geometry)

    vertebral = np.zeros(shape, dtype=np.uint8)
    if not empty_vertebrae:
        for label, start in (
            (19, 5),
            (17, 15),
            (16, 22),
            (15, 30),
            (14, 38),
            (13, 46),
            (12, 54),
            (10, 65),
        ):
            vertebral[start : start + 5, 9:15, 11:17] = label
    vertebral_result = VertebralResult(
        backend_id="synthetic_body_only",
        execution_status=ExecutionStatus.SUCCEEDED,
        geometry=geometry,
        whole_vertebra_labels=vertebral.copy(),
        vertebral_body_labels=vertebral,
        label_schema=config["LBL_VERTEBRALBODIES"],
        provenance={"model": "synthetic"},
    )
    return image, tissues, geometry, body_surface, vertebral_result


def make_bundle(config, *, empty_vertebrae=False):
    image, tissues, geometry, body_surface, vertebral_result = make_measurement_inputs(
        config,
        empty_vertebrae=empty_vertebrae,
    )
    identity = MeasurementIdentity("case-001", "run-001", "analysis-001")
    bundle = build_measurement_bundle(
        image_zyx=image,
        tissue_labels_zyx=tissues,
        geometry=geometry,
        tissue_label_schema=config["LBL_TISSUE"],
        tissue_backend_id="synthetic",
        tissue_preprocessing={"hu_denoise": False},
        body_surface=body_surface,
        vertebral_result=vertebral_result,
        identity=identity,
        landmarks=None,
        settings=config["measurements"],
        orientation_changed=False,
        orientation_provenance={
            "state": "PASS_METADATA_MATCH",
            "qc_status": "pass",
            "manual_review_required": False,
            "orientation_changed": False,
        },
    )
    return bundle, image, geometry, tissues, vertebral_result


def sitk_image(array_zyx, geometry):
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(geometry.spacing_xyz)
    image.SetOrigin(geometry.origin_lps_xyz)
    image.SetDirection(geometry.direction_lps)
    return image


def test_complete_bundle_validates_three_native_territory_bins(base_config):
    bundle, image, _, _, _ = make_bundle(base_config)

    assert len(bundle.slices) == image.shape[0]
    assert np.all(np.diff(bundle.slices["position_superior_mm"]) > 0)
    assert bundle.vertebrae.groupby("vertebral_level").size().eq(3).all()
    assert set(bundle.vertebrae["territory_bin"]) == {1, 2, 3}
    assert "SACRUM" in set(bundle.vertebrae["vertebral_level"])
    assert "signatures" not in bundle.__dict__
    assert bundle.summaries.iloc[0]["l3_200mm_slab_valid"]
    assert "orientation_changed" not in bundle.slices
    assert bundle.provenance["orientation"]["state"] == "PASS_METADATA_MATCH"
    assert set(TABLE_NAMES) == {"slices", "vertebrae", "summaries"}


def test_empty_vertebral_segmentation_preserves_slices_and_explicit_qc(base_config):
    bundle, image, _, _, _ = make_bundle(base_config, empty_vertebrae=True)

    assert len(bundle.slices) == image.shape[0]
    assert bundle.vertebrae.empty
    assert bundle.slices["assigned_vertebral_level"].isna().all()
    assert set(bundle.slices["vertebral_assignment_status"]) == {
        "no_valid_territory"
    }
    assert any(flag.code == "vertebral_body_segmentation_empty" for flag in bundle.qc_flags)


def test_bundle_surfaces_fragmented_trunk_and_backend_neutral_variant_qc(base_config):
    image, tissues, geometry, _, vertebral_result = make_measurement_inputs(base_config)
    body_labels = np.zeros(image.shape, dtype=np.uint8)
    body_labels[:, 3:-3, 3:-3] = 1
    body_labels[10, 1:2, 8:11] = 1
    body_surface = body_surface_from_totalsegmentator(body_labels, geometry)
    vertebral = vertebral_result.vertebral_body_labels.copy()
    vertebral[72:77, 9:15, 11:17] = 18
    vertebral_result = replace(
        vertebral_result,
        whole_vertebra_labels=vertebral.copy(),
        vertebral_body_labels=vertebral,
    )

    bundle = build_measurement_bundle(
        image_zyx=image,
        tissue_labels_zyx=tissues,
        geometry=geometry,
        tissue_label_schema=base_config["LBL_TISSUE"],
        tissue_backend_id="synthetic",
        tissue_preprocessing={"hu_denoise": False},
        body_surface=body_surface,
        vertebral_result=vertebral_result,
        identity=MeasurementIdentity("case-001", "run-001", "analysis-001"),
        landmarks=None,
        settings=base_config["measurements"],
    )

    codes = {flag.code for flag in bundle.qc_flags}
    assert "trunk_surface_fragmented_slices" in codes
    assert "anatomical_variant_present" in codes
    fragmented = bundle.slices.loc[bundle.slices["trunk_mask_fragmented"]]
    assert fragmented["slice_qc_status"].eq("review").all()
    assert not fragmented["trunk_area_valid"].any()


def test_ambiguous_sequence_variant_stays_continuous_and_is_flagged(
    base_config,
):
    image, tissues, geometry, body_surface, vertebral_result = make_measurement_inputs(
        base_config
    )
    vertebral = vertebral_result.vertebral_body_labels.copy()
    vertebral[vertebral == 17] = 0
    vertebral_result = replace(
        vertebral_result,
        whole_vertebra_labels=vertebral.copy(),
        vertebral_body_labels=vertebral,
        qc_flags=(),
    )

    bundle = build_measurement_bundle(
        image_zyx=image,
        tissue_labels_zyx=tissues,
        geometry=geometry,
        tissue_label_schema=base_config["LBL_TISSUE"],
        tissue_backend_id="synthetic",
        tissue_preprocessing={"hu_denoise": False},
        body_surface=body_surface,
        vertebral_result=vertebral_result,
        identity=MeasurementIdentity("case-001", "run-001", "analysis-001"),
        landmarks=None,
        settings=base_config["measurements"],
    )

    assert "L5" not in set(bundle.vertebrae["vertebral_level"])
    l4 = bundle.vertebrae.loc[bundle.vertebrae["vertebral_level"].eq("L4")]
    assert l4["bin_valid"].all()
    assert l4["sequence_gap_caudal"].all()
    assert l4["territory_qc_status"].eq("review").all()
    assert any(flag.code == "anatomical_variant_present" for flag in bundle.qc_flags)
    assert any(flag.code == "vertebral_territory_sequence_gap" for flag in bundle.qc_flags)


def test_parquet_export_round_trip_api_and_cli_views(base_config, tmp_path, monkeypatch, capsys):
    bundle, _, _, _, _ = make_bundle(base_config)
    action = ExportMeasurementBundle(
        SimpleNamespace(config=base_config, timestamp=123, device="cpu")
    )
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "tmp/measurement_bundle": bundle,
    }

    action(memory)
    action.validate_outputs(memory)

    table_directory = tmp_path / "tables" / "case-001"
    tables = load_measurement_tables(table_directory)
    assert set(tables) == set(TABLE_NAMES)
    assert len(tables["slices"]) == len(bundle.slices)
    assert tables["vertebrae"][["vertebral_level", "territory_bin"]].to_numpy().tolist() == (
        bundle.vertebrae[["vertebral_level", "territory_bin"]].to_numpy().tolist()
    )
    assert l3_measurements(table_directory, aggregation="territory_mean").iloc[0][
        "aggregation"
    ] == "territory_mean"
    assert l3_measurements(table_directory, aggregation="slice").iloc[0]["aggregation"] == "slice"
    named = range_measurements(
        table_directory,
        start_level="T12",
        end_level="L5",
    )
    assert named.iloc[0]["range_valid"]
    assert named.iloc[0]["analysis_id"] == bundle.identity.analysis_id
    assert named.iloc[0]["aggregation"] == "physical_range"
    qc = json.loads((tmp_path / "qc" / "case-001_measurement-qc.json").read_text())
    schema = json.loads(
        Path("BodyComposition/schemas/measurement_qc.schema.json").read_text()
    )
    jsonschema.validate(qc, schema)
    assert qc["provenance"]["vertebral"]["backend_id"] == "synthetic_body_only"
    assert qc["provenance"]["orientation"]["state"] == "PASS_METADATA_MATCH"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bodycomposition_measurements",
            "--tables",
            str(table_directory),
            "--view",
            "l3",
            "--aggregation",
            "slice",
        ],
    )
    measurement_view.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["aggregation"] == "slice"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bodycomposition_measurements",
            "--tables",
            str(table_directory),
            "--view",
            "l3",
            "--aggregation",
            "territory_mean",
        ],
    )
    measurement_view.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["aggregation"] == "territory_mean"
    assert isinstance(payload[0]["contributing_slice_ids"], list)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bodycomposition_measurements",
            "--tables",
            str(table_directory),
            "--view",
            "range",
            "--start-level",
            "T12",
            "--end-level",
            "L5",
        ],
    )
    measurement_view.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["aggregation"] == "physical_range"
    assert payload[0]["analysis_id"] == bundle.identity.analysis_id
    assert not list(tmp_path.rglob("*.partial*"))

    summaries_path = table_directory / "summaries.parquet"
    parquet_schema = pq.read_schema(summaries_path)
    metadata = parquet_schema.metadata
    summaries = pd.read_parquet(summaries_path)
    summaries.loc[:, "analysis_id"] = "analysis-from-another-case"
    corrupted = pa.Table.from_pandas(
        summaries,
        schema=parquet_schema.remove_metadata(),
        preserve_index=False,
        safe=True,
    )
    pq.write_table(corrupted.replace_schema_metadata(metadata), summaries_path)
    with pytest.raises(ValueError, match="inconsistent analysis_id"):
        load_measurement_tables(table_directory)


def test_parquet_reader_rejects_missing_vertebral_territory_bin(base_config, tmp_path):
    bundle, _, _, _, _ = make_bundle(base_config)
    action = ExportMeasurementBundle(
        SimpleNamespace(config=base_config, timestamp=123, device="cpu")
    )
    action(
        {
            "id": "case-001",
            "workspace": tmp_path,
            "tmp/measurement_bundle": bundle,
        }
    )

    vertebrae_path = tmp_path / "tables" / "case-001" / "vertebrae.parquet"
    vertebrae = pq.read_table(vertebrae_path)
    frame = vertebrae.to_pandas()
    removed_row = frame.index[
        frame["vertebral_level"].eq("L3") & frame["territory_bin"].eq(2)
    ][0]
    corrupted = vertebrae.take(
        pa.array(
            [index for index in range(len(frame)) if index != removed_row],
            type=pa.int64(),
        )
    )
    pq.write_table(corrupted, vertebrae_path)

    with pytest.raises(ValueError, match="unstable bins for 'L3'"):
        load_measurement_tables(vertebrae_path.parent)


def test_parquet_reader_rejects_schema_metadata_drift(base_config, tmp_path):
    bundle, _, _, _, _ = make_bundle(base_config)
    action = ExportMeasurementBundle(
        SimpleNamespace(config=base_config, timestamp=123, device="cpu")
    )
    action(
        {
            "id": "case-001",
            "workspace": tmp_path,
            "tmp/measurement_bundle": bundle,
        }
    )

    slices_path = tmp_path / "tables" / "case-001" / "slices.parquet"
    slices = pq.read_table(slices_path)
    metadata = dict(slices.schema.metadata or {})
    metadata[b"bodycomposition.measurement_schema_version"] = b"0.0.0"
    pq.write_table(slices.replace_schema_metadata(metadata), slices_path)

    with pytest.raises(ValueError, match="inconsistent measurement schema metadata"):
        load_measurement_tables(slices_path.parent)


def test_missing_anatomy_keeps_identical_nullable_parquet_schemas(
    base_config,
    tmp_path,
):
    complete, *_ = make_bundle(base_config)
    missing, *_ = make_bundle(base_config, empty_vertebrae=True)
    schemas = {}
    for state, bundle in (("complete", complete), ("missing", missing)):
        workspace = tmp_path / state
        action = ExportMeasurementBundle(
            SimpleNamespace(config=base_config, timestamp=123, device="cpu")
        )
        action(
            {
                "id": "case-001",
                "workspace": workspace,
                "tmp/measurement_bundle": bundle,
            }
        )
        table_directory = workspace / "tables" / "case-001"
        schemas[state] = {
            name: pq.read_schema(table_directory / f"{name}.parquet")
            for name in TABLE_NAMES
        }
        load_measurement_tables(table_directory)

    for name in TABLE_NAMES:
        assert schemas["complete"][name].equals(
            schemas["missing"][name],
            check_metadata=True,
        )
    assert schemas["missing"]["vertebrae"].field("territory_bin").type == pa.int64()
    assert schemas["missing"]["slices"].field(
        "assigned_vertebral_level"
    ).type == pa.string()
    assert "sm_volume_cm3" not in missing.vertebrae
    assert "sm_mean_csa_cm2" in missing.vertebrae


def test_range_api_reuses_the_analysis_coverage_tolerance(base_config, tmp_path):
    config = deepcopy(base_config)
    config["measurements"]["full_coverage_tolerance"] = 0.99
    bundle, *_ = make_bundle(config)
    slices = bundle.slices.copy()
    vertebrae = bundle.vertebrae.copy()
    acquisition_lower = float(slices["slice_slab_inferior_mm"].min())
    acquisition_upper = float(slices["slice_slab_superior_mm"].max())
    observed_length = acquisition_upper - acquisition_lower
    target_coverage = 0.995
    extension = observed_length / target_coverage - observed_length
    vertebrae.loc[
        vertebrae["vertebral_level"].eq("L5"),
        "territory_inferior_mm",
    ] = acquisition_lower - extension
    vertebrae.loc[
        vertebrae["vertebral_level"].eq("T12"),
        "territory_superior_mm",
    ] = acquisition_upper
    bundle = replace(bundle, slices=slices, vertebrae=vertebrae)
    action = ExportMeasurementBundle(
        SimpleNamespace(config=config, timestamp=123, device="cpu")
    )
    action(
        {
            "id": "case-001",
            "workspace": tmp_path,
            "tmp/measurement_bundle": bundle,
        }
    )

    result = range_measurements(
        tmp_path / "tables" / "case-001",
        start_level="L5",
        end_level="T12",
    ).iloc[0]

    assert result["coverage_fraction"] == pytest.approx(target_coverage)
    assert result["full_coverage_tolerance"] == pytest.approx(0.99)
    assert result["range_valid"]


def test_informational_coverage_flag_does_not_request_manual_review(
    base_config,
    tmp_path,
):
    bundle, _, _, _, _ = make_bundle(base_config)
    bundle = replace(
        bundle,
        qc_flags=(
            QCFlag(
                code="expected_outside_fov",
                stage="measurement",
                severity=QCSeverity.INFO,
                reason="Optional vertebral territories are outside the acquired FOV.",
            ),
        ),
    )
    assert bundle.qc_status == QCStatus.PASS
    action = ExportMeasurementBundle(
        SimpleNamespace(config=base_config, timestamp=123, device="cpu")
    )
    memory = {
        "id": "case-001",
        "workspace": tmp_path,
        "tmp/measurement_bundle": bundle,
    }

    action(memory)

    qc = json.loads((tmp_path / "qc" / "case-001_measurement-qc.json").read_text())
    assert qc["qc_status"] == "pass"
    assert not qc["manual_review_required"]
    assert qc["qc_flags"][0]["severity"] == "info"


def test_measurement_review_contains_axial_and_longitudinal_panels(base_config, tmp_path):
    bundle, image, geometry, _, _ = make_bundle(base_config)
    output = write_measurement_review(
        sitk_image(image, geometry),
        bundle,
        tmp_path / "measurement-review.png",
    )

    rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert rendered is not None
    assert rendered.shape == (790, 1280, 3)
    assert rendered.std() > 5


def test_measurement_analysis_identity_is_order_stable_and_content_sensitive(base_config):
    image, tissues, _, body_surface, vertebral_result = make_measurement_inputs(base_config)
    first = measurement_analysis_id(
        image,
        tissues,
        body_surface,
        vertebral_result,
        {"alpha": 1, "beta": 2},
    )
    reordered = measurement_analysis_id(
        image,
        tissues,
        body_surface,
        vertebral_result,
        {"beta": 2, "alpha": 1},
    )
    changed_tissues = tissues.copy()
    changed_tissues[0, 0, 0] = 1
    changed = measurement_analysis_id(
        image,
        changed_tissues,
        body_surface,
        vertebral_result,
        {"alpha": 1, "beta": 2},
    )
    changed_provenance = measurement_analysis_id(
        image,
        tissues,
        replace(body_surface, provenance={"asset_sha256": "changed"}),
        vertebral_result,
        {"alpha": 1, "beta": 2},
    )
    landmarks = LandmarkSet(
        lowest_rib_inferior=Landmark("rib", 250.0, "synthetic", True),
        iliac_crest_superior=Landmark("crest", 150.0, "synthetic", True),
    )
    with_landmarks = measurement_analysis_id(
        image,
        tissues,
        body_surface,
        vertebral_result,
        {"alpha": 1, "beta": 2},
        landmarks=landmarks,
    )
    with_tissue_contract = measurement_analysis_id(
        image,
        tissues,
        body_surface,
        vertebral_result,
        {"alpha": 1, "beta": 2},
        tissue_label_schema={1: "SM", 3: "SAT"},
        tissue_preprocessing={"filter_size": False},
    )
    with_orientation_provenance = measurement_analysis_id(
        image,
        tissues,
        body_surface,
        vertebral_result,
        {"alpha": 1, "beta": 2},
        orientation_provenance={
            "state": "PASS_METADATA_MATCH",
            "orientation_changed": False,
        },
    )

    assert first == reordered
    assert first.startswith("analysis-")
    assert first != changed
    assert first != changed_provenance
    assert first != with_landmarks
    assert first != with_tissue_contract
    assert first != with_orientation_provenance


def test_measurement_identity_configuration_excludes_output_and_review_policy(base_config):
    settings = json.loads(json.dumps(base_config["measurements"]))
    output_only = json.loads(json.dumps(settings))
    output_only["enabled"] = not output_only["enabled"]
    output_only["review"]["enabled"] = not output_only["review"]["enabled"]
    output_only["export"]["parquet"] = not output_only["export"]["parquet"]
    output_only["body_surface"]["save_mask"] = not output_only["body_surface"][
        "save_mask"
    ]

    assert _scientific_measurement_configuration(
        settings
    ) == _scientific_measurement_configuration(output_only)

    output_only["body_surface"]["smoothing_sigma_mm"] += 1
    assert _scientific_measurement_configuration(
        settings
    ) != _scientific_measurement_configuration(output_only)


def test_fast_pipeline_uses_full_prepared_domain_and_explicit_body_label_source(pipeline_stub):
    actions = BodyCompositionFast(pipeline_stub)
    names = [type(action).__name__ for action in actions]

    assert "CreateBoundingBox" not in names
    assert "ApplyBoundingBox" not in names
    tissue = next(action for action in actions if isinstance(action, SegmIntBodyComposition))
    measurement = next(
        action for action in actions if isinstance(action, MeasureCanonicalBodyComposition)
    )
    surface = next(action for action in actions if isinstance(action, CreateBodySurface))
    assert tissue.input_image_name == "tmp/index"
    assert tissue.model_preset == "ResEncM"
    assert measurement.vertebral_body_source_name == SPINEPS_BODY_MASK
    assert "tmp/vertebral_result" in measurement.io_inputs
    assert surface.backend == TISSUE_ENVELOPE_BACKEND
    assert names.index("MasksInternalTissue") < names.index("CreateBodySurface")


def test_tissue_envelope_pipeline_can_run_without_totalsegmentator(
    pipeline_stub,
):
    pipeline_stub.config["measurements"]["landmarks"]["enabled"] = False

    actions = BodyCompositionFast(pipeline_stub)
    names = [type(action).__name__ for action in actions]

    assert "SegmTotalSegmentatorConfig" not in names
    assert "SegmTotalSegmentator" not in names
    assert "CreateMeasurementLandmarks" not in names
    assert "CreateBodySurface" in names
    assert "MeasureCanonicalBodyComposition" in names


def test_measurement_totalsegmentator_assets_are_checked_without_download(
    tmp_path,
    monkeypatch,
):
    files = {
        "trainer/dataset.json": b"dataset",
        "trainer/plans.json": b"plans",
        "trainer/fold_0/checkpoint_final.pth": b"checkpoint",
    }
    manifest = {
        "bodytrunk": {
            "task_id": 299,
            "model_title": "TotalSegmentator-body",
            "directory": "Dataset299_body_1559subj",
            "files": {
                relative: hashlib.sha256(content).hexdigest()
                for relative, content in files.items()
            },
        }
    }
    monkeypatch.setattr(
        totalsegmentator_assets,
        "MEASUREMENT_MODEL_MANIFEST",
        manifest,
    )
    totalsegmentator_assets._check_cached.cache_clear()
    monkeypatch.setattr("totalsegmentator.config.get_weights_dir", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="Run `bodycomposition_download_models"):
        _require_measurement_model("bodytrunk")

    model_directory = tmp_path / "Dataset299_body_1559subj"
    for relative, content in files.items():
        path = model_directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _require_measurement_model("bodytrunk")
    _require_measurement_model("spine")

    (model_directory / "trainer/plans.json").write_bytes(b"corrupted")
    report = totalsegmentator_assets.check_measurement_model("bodytrunk")
    assert not report.ready
    assert report.errors == ("sha256_mismatch:trainer/plans.json",)


def test_measurement_totalsegmentator_inference_is_pinned_and_cache_only(
    pipeline_stub,
    monkeypatch,
):
    checks = []
    monkeypatch.setattr(
        totalsegmentator_assets,
        "require_measurement_model",
        lambda task: checks.append(task),
    )
    import totalsegmentator.python_api as upstream_api

    upstream_download = upstream_api.download_pretrained_weights
    upstream_usage_stats = upstream_api.send_usage_stats
    with _measurement_model_download_guard("body_landmarks"):
        assert upstream_api.download_pretrained_weights is not upstream_download
        assert upstream_api.send_usage_stats is not upstream_usage_stats
        upstream_api.download_pretrained_weights(297)
        assert upstream_api.send_usage_stats({}, {}) is None
        with pytest.raises(RuntimeError, match="unexpected model"):
            upstream_api.download_pretrained_weights(298)
    assert upstream_api.download_pretrained_weights is upstream_download
    assert upstream_api.send_usage_stats is upstream_usage_stats
    assert checks == ["body_landmarks", "body_landmarks"]

    action = SegmTotalSegmentator(
        pipeline_stub,
        image="tmp/index",
        task="body_landmarks",
    )
    assert action.task_config == {
        "task": "total",
        "fast": True,
        "roi_subset": None,
        "license_nc": False,
    }


def test_default_model_sync_omits_optional_body_model_but_keeps_landmarks():
    for pipeline_name in ("BodyComposition", "BodyCompositionFast"):
        models = definition_pipelines[pipeline_name]
        assert "TotalSegmentator-body" not in models
        assert "TotalSegmentator-total-fast" in models


def test_configured_pipeline_model_sync_respects_optional_measurement_support(
    base_config,
):
    config = deepcopy(base_config)
    config["measurements"]["landmarks"]["enabled"] = False
    for pipeline_name in ("BodyComposition", "BodyCompositionFast"):
        models = configured_pipeline_models(pipeline_name, config)
        assert "TotalSegmentator-body" not in models
        assert "TotalSegmentator-total-fast" not in models

    config["measurements"]["body_surface"]["backend"] = (
        "totalsegmentator_body_task299_v1"
    )
    models = configured_pipeline_models("BodyComposition", config)
    assert "TotalSegmentator-body" in models
    assert "TotalSegmentator-total-fast" not in models

    config["measurements"]["landmarks"]["enabled"] = True
    models = configured_pipeline_models("BodyCompositionFast", config)
    assert "TotalSegmentator-body" in models
    assert "TotalSegmentator-total-fast" in models


def test_canonical_config_rejects_duplicate_csv_export(base_config):
    base_config["measurements"]["export"]["csv"] = True
    with pytest.raises(ConfigError, match="do not permit duplicate CSV"):
        validate_config(base_config)
