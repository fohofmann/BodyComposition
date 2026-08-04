from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

import BodyComposition.service as service_module
from BodyComposition import (
    cli,
    convert_dicom,
    discover_case_inputs,
    discover_dicom_series,
)
from BodyComposition.config import PipelineConfig
from BodyComposition.dicom import (
    DicomConversionBatchResult,
    DicomConversionMetadataError,
    DicomConversionResult,
    DicomSeriesSelectionError,
    dicom_report_patient_metadata,
)
from BodyComposition.model_manager import ModelStatus
from BodyComposition.provenance import image_summary
from BodyComposition.results import ExecutionStatus
from BodyComposition.service import InputDiscoveryError, PipelineService
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain


def _write_dicom_series(
    directory: Path,
    array_zyx: np.ndarray,
    *,
    series_uid: str,
    modality: str = "CT",
    spacing_xyz: tuple[float, float, float] = (0.8, 0.9, 2.5),
    origin_lps_xyz: tuple[float, float, float] = (10.0, 20.0, -30.0),
    direction_lps: tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ),
) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array_zyx.astype(np.int16, copy=False))
    image.SetSpacing(spacing_xyz)
    image.SetOrigin(origin_lps_xyz)
    image.SetDirection(direction_lps)
    direction = np.asarray(direction_lps, dtype=float).reshape(3, 3)
    orientation = (*direction[:, 0], *direction[:, 1])
    study_uid = "1.2.826.0.1.3680043.10.999.900"
    paths = []
    for index in range(image.GetSize()[2]):
        image_slice = image[:, :, index]
        position = image.TransformIndexToPhysicalPoint((0, 0, index))
        metadata = {
            "0008|0008": "ORIGINAL\\PRIMARY\\AXIAL",
            "0008|0018": f"{series_uid}.{index + 1}",
            "0008|0060": modality,
            "0008|0070": "Research Test Imaging",
            "0008|0020": "20260718",
            "0008|0022": "20260719",
            "0008|1090": "Synthetic CT 1.0",
            "0010|0010": "Identifying^Patient",
            "0010|0020": "MRN-123456",
            "0010|0030": "19841203",
            "0010|0040": "F",
            "0018|0050": str(spacing_xyz[2]),
            "0020|000d": study_uid,
            "0020|000e": series_uid,
            "0020|0013": str(index + 1),
            "0020|0032": "\\".join(f"{value:.12g}" for value in position),
            "0020|0037": "\\".join(f"{value:.12g}" for value in orientation),
            "0028|0030": f"{spacing_xyz[1]}\\{spacing_xyz[0]}",
        }
        for key, value in metadata.items():
            image_slice.SetMetaData(key, value)
        path = directory / f"slice-{index:04d}.dcm"
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        writer.SetFileName(str(path))
        writer.Execute(image_slice)
        paths.append(path)
    return tuple(paths)


def test_dicom_conversion_preserves_pixels_geometry_and_excludes_identifiers(tmp_path):
    source = tmp_path / "MRN-123456" / "series"
    uid = "1.2.826.0.1.3680043.10.999.1"
    array = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6) - 500
    direction = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    _write_dicom_series(
        source,
        array,
        series_uid=uid,
        direction_lps=direction,
    )

    series = discover_dicom_series(source)
    assert [item.series_instance_uid for item in series] == [uid]
    assert series[0].modality == "CT"
    assert series[0].instance_count == array.shape[0]

    dicom_image, summary = image_summary(source)
    output = tmp_path / "converted.nii.gz"
    result = convert_dicom(source, output)
    nifti_image, nifti_summary = image_summary(output)

    assert np.array_equal(sitk.GetArrayFromImage(dicom_image), array)
    assert np.array_equal(sitk.GetArrayFromImage(nifti_image), array)
    assert np.allclose(dicom_image.GetDirection(), direction)
    assert_same_physical_domain(
        ImageGeometry.from_sitk(dicom_image),
        ImageGeometry.from_sitk(nifti_image),
    )
    assert result.series_instance_uid == uid
    assert result.input_summary == summary
    assert result.metadata_path == tmp_path / "converted.bodycomposition.json"
    assert result.metadata_path.is_file()
    assert nifti_summary["input_pixel_sha256"] == summary["input_pixel_sha256"]
    assert nifti_summary["input_format"] == "nifti"
    assert nifti_summary["dicom"] == summary["dicom"]
    assert nifti_summary["prestage"]["source_format"] == "dicom"
    assert (
        nifti_summary["prestage"]["source_content_sha256"]
        == summary["source_content_sha256"]
    )
    assert summary["input_format"] == "dicom"
    assert summary["dicom"]["image_orientation_patient_complete"]
    assert summary["dicom"]["image_position_patient_complete"]
    assert summary["dicom"]["scanner_manufacturer"] == "Research Test Imaging"
    assert summary["dicom"]["scanner_model"] == "Synthetic CT 1.0"
    assert dicom_report_patient_metadata(source) == {
        "patient_name": "Patient Identifying",
        "date_of_birth": "1984-12-03",
        "scan_date": "2026-07-19",
        "sex": "F",
    }
    serialized = json.dumps(summary)
    serialized_metadata = result.metadata_path.read_text(encoding="utf-8")
    assert uid not in serialized
    assert uid not in serialized_metadata
    assert "MRN-123456" not in serialized
    assert "MRN-123456" not in serialized_metadata
    assert str(tmp_path) not in serialized
    assert str(tmp_path) not in serialized_metadata
    assert not any(key.startswith("0010|") for key in nifti_image.GetMetaDataKeys())


def test_conversion_metadata_is_optional_but_must_match_when_present(tmp_path):
    source = tmp_path / "dicom"
    _write_dicom_series(
        source,
        np.zeros((2, 3, 4), dtype=np.int16),
        series_uid="1.2.826.0.1.3680043.10.999.2",
    )
    output = tmp_path / "ct.nii.gz"
    result = convert_dicom(source, output)
    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    payload["nifti"]["content_sha256"] = "0" * 64
    result.metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DicomConversionMetadataError, match="file hash"):
        image_summary(output)

    result.metadata_path.unlink()
    _, summary = image_summary(output)
    assert summary["input_format"] == "nifti"
    assert "dicom" not in summary
    assert "prestage" not in summary


def test_multiple_ct_series_require_explicit_selection_but_a_file_is_unambiguous(tmp_path):
    root = tmp_path / "dicom"
    uid_a = "1.2.826.0.1.3680043.10.999.11"
    uid_b = "1.2.826.0.1.3680043.10.999.12"
    files_a = _write_dicom_series(
        root / "a",
        np.full((3, 4, 5), 11, dtype=np.int16),
        series_uid=uid_a,
    )
    _write_dicom_series(
        root / "b",
        np.full((3, 4, 5), 22, dtype=np.int16),
        series_uid=uid_b,
    )

    with pytest.raises(DicomSeriesSelectionError, match="Multiple CT series"):
        image_summary(root)

    selected, _ = image_summary(root, series_uid=uid_b)
    assert np.unique(sitk.GetArrayFromImage(selected)).tolist() == [22]
    selected_file, _ = image_summary(files_a[0])
    assert np.unique(sitk.GetArrayFromImage(selected_file)).tolist() == [11]


def test_convert_dicom_refuses_implicit_overwrite_and_non_nifti_output(tmp_path):
    source = tmp_path / "dicom"
    _write_dicom_series(
        source,
        np.zeros((2, 3, 4), dtype=np.int16),
        series_uid="1.2.826.0.1.3680043.10.999.21",
    )
    output = tmp_path / "ct.nii.gz"
    convert_dicom(source, output)

    with pytest.raises(FileExistsError, match="already exists"):
        convert_dicom(source, output)
    replacement = convert_dicom(source, output, overwrite=True)
    assert replacement.output_path == output
    with pytest.raises(ValueError, match="must end"):
        convert_dicom(source, tmp_path / "ct.mha")


def test_standalone_cli_converts_one_dicom_series(tmp_path, capsys):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.22"
    _write_dicom_series(
        source,
        np.full((2, 3, 4), 42, dtype=np.int16),
        series_uid=uid,
    )
    output = tmp_path / "ct.nii.gz"

    code = cli.main(["convert", str(source), str(output), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["output_path"] == str(output)
    assert payload["metadata_path"] == str(tmp_path / "ct.bodycomposition.json")
    assert payload["series_instance_uid"] == uid
    assert payload["input_format"] == "dicom"
    assert output.is_file()


def test_nested_dicom_cohort_conversion_is_transfer_ready_and_idempotent(tmp_path):
    source = tmp_path / "import"
    uid_a = "1.2.826.0.1.3680043.10.999.231"
    uid_b = "1.2.826.0.1.3680043.10.999.232"
    _write_dicom_series(
        source / "patient-one" / "DICOMS",
        np.full((2, 3, 4), 21, dtype=np.int16),
        series_uid=uid_a,
    )
    _write_dicom_series(
        source / "year" / "patient-two" / "DICOM",
        np.full((3, 3, 4), 42, dtype=np.int16),
        series_uid=uid_b,
    )
    output = tmp_path / "converted"

    result = convert_dicom(source, output)

    assert isinstance(result, DicomConversionBatchResult)
    assert result.succeeded
    assert result.discovered_series_count == 2
    assert len(result.conversions) == 2
    assert result.failures == ()
    assert result.manifest_path == output / "bodycomposition-batch.json"
    assert result.report_path == output / "bodycomposition-conversion.json"
    cases = discover_case_inputs(output)
    assert len(cases) == 2
    assert all(case.input_path.parent == output for case in cases)
    assert [case.case_id for case in cases] == [
        conversion.case_id for conversion in result.conversions
    ]
    assert all(case.series_uid is None for case in cases)
    for conversion in result.conversions:
        assert conversion.output_path.is_file()
        assert conversion.metadata_path.is_file()

    public_text = result.manifest_path.read_text(encoding="utf-8")
    public_text += result.report_path.read_text(encoding="utf-8")
    assert "patient-one" not in public_text
    assert "patient-two" not in public_text
    assert uid_a not in public_text
    assert uid_b not in public_text

    repeated = convert_dicom(source, output)
    assert isinstance(repeated, DicomConversionBatchResult)
    assert repeated.succeeded
    assert [item.output_content_sha256 for item in repeated.conversions] == [
        item.output_content_sha256 for item in result.conversions
    ]


def test_cli_converts_nested_dicom_cohort_to_directory(tmp_path, capsys):
    source = tmp_path / "import"
    for index in range(2):
        _write_dicom_series(
            source / f"case-{index}" / "DICOMS",
            np.full((2, 3, 4), index + 1, dtype=np.int16),
            series_uid=f"1.2.826.0.1.3680043.10.999.24{index}",
        )

    code = cli.main(["convert", str(source), str(tmp_path / "converted"), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == cli.EXIT_OK
    assert payload["execution_status"] == "succeeded"
    assert payload["converted_count"] == 2
    assert payload["failed_count"] == 0
    assert Path(payload["manifest_path"]).name == "bodycomposition-batch.json"


def test_directory_analysis_discovers_dicom_series_and_rejects_unsafe_nifti_batch(
    tmp_path,
):
    dicom_root = tmp_path / "dicom"
    uid_a = "1.2.826.0.1.3680043.10.999.251"
    uid_b = "1.2.826.0.1.3680043.10.999.252"
    _write_dicom_series(
        dicom_root / "patient-a" / "DICOMS",
        np.full((2, 3, 4), 1, dtype=np.int16),
        series_uid=uid_a,
    )
    _write_dicom_series(
        dicom_root / "patient-b" / "DICOMS",
        np.full((2, 3, 4), 2, dtype=np.int16),
        series_uid=uid_b,
    )

    cases = discover_case_inputs(dicom_root)
    assert [case.series_uid for case in cases] == [uid_a, uid_b]
    assert len(discover_case_inputs(dicom_root, series_uid=uid_b)) == 1

    nifti_root = tmp_path / "nifti"
    nifti_root.mkdir()
    sitk.WriteImage(sitk.Image((3, 4, 2), sitk.sitkInt16), str(nifti_root / "ct.nii.gz"))
    sitk.WriteImage(sitk.Image((3, 4, 2), sitk.sitkUInt8), str(nifti_root / "mask.nii.gz"))
    with pytest.raises(InputDiscoveryError, match="not every file has"):
        discover_case_inputs(nifti_root)

    with pytest.raises(InputDiscoveryError, match="cannot be used with a NIfTI"):
        discover_case_inputs(nifti_root / "ct.nii.gz", series_uid=uid_a)


def test_single_file_conversion_requires_a_series_for_ambiguous_dicom(tmp_path):
    source = tmp_path / "dicom"
    uid_a = "1.2.826.0.1.3680043.10.999.253"
    uid_b = "1.2.826.0.1.3680043.10.999.254"
    _write_dicom_series(
        source / "series-a",
        np.full((2, 3, 4), 1, dtype=np.int16),
        series_uid=uid_a,
    )
    _write_dicom_series(
        source / "series-b",
        np.full((2, 3, 4), 2, dtype=np.int16),
        series_uid=uid_b,
    )

    with pytest.raises(DicomSeriesSelectionError, match="Multiple CT series"):
        convert_dicom(source, tmp_path / "ct.nii.gz")

    selected = convert_dicom(
        source,
        tmp_path / "ct.nii.gz",
        series_uid=uid_b,
    )
    assert isinstance(selected, DicomConversionResult)
    assert selected.series_instance_uid == uid_b


def test_cohort_conversion_records_case_identity_collisions_without_stopping(tmp_path):
    source = tmp_path / "import"
    pixels = np.full((2, 3, 4), 7, dtype=np.int16)
    _write_dicom_series(
        source / "first" / "DICOM",
        pixels,
        series_uid="1.2.826.0.1.3680043.10.999.261",
    )
    _write_dicom_series(
        source / "second" / "DICOM",
        pixels,
        series_uid="1.2.826.0.1.3680043.10.999.262",
    )

    result = convert_dicom(source, tmp_path / "converted")

    assert isinstance(result, DicomConversionBatchResult)
    assert not result.succeeded
    assert len(result.conversions) == 1
    assert len(result.failures) == 1
    assert result.failures[0].code == "DicomInputError"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["execution_status"] == "failed"
    assert report["converted_count"] == 1
    assert report["failed_count"] == 1


def test_automatic_conversion_manifest_cannot_escape_its_directory(tmp_path):
    root = tmp_path / "converted"
    root.mkdir()
    (root / "bodycomposition-batch.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "cases": [{"input_path": "../outside.nii.gz"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InputDiscoveryError, match="outside its directory"):
        discover_case_inputs(root)


def test_pipeline_accepts_dicom_without_persisting_the_temporary_conversion(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "private-patient-name" / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.31"
    array = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)
    _write_dicom_series(source, array, series_uid=uid)
    captured = {}

    class CapturePipeline:
        def __init__(self, config, timestamp):
            pass

        def __call__(self, memory):
            image = memory["tmp/index"]
            captured["path"] = image.path
            captured["array"] = image.data.copy()
            captured["input_summary"] = memory["tmp/input_summary"]
            captured["patient_metadata"] = memory["tmp/report_patient_metadata"]

    model = ModelStatus(
        model_id="test-model",
        ready=True,
        model_root=tmp_path,
        errors=(),
        checked_files={"weights": "c" * 64},
        asset={"asset_id": "test-model", "license": "test-only"},
    )
    source_state = {
        "package_version": "1.0.0rc1",
        "git_commit": "a" * 40,
        "source_dirty": False,
        "source_tree_sha256": "b" * 64,
        "container_image_digest": None,
        "environment_path_variables": [],
    }
    config = PipelineConfig.model_validate(
        {
            "measurements": {"landmarks": {"enabled": False}},
            "runtime": {"allow_dirty": True},
        }
    )
    service = PipelineService(
        config,
        pipeline_factory=CapturePipeline,
        model_provider=lambda value: (model,),
        source_provider=lambda: source_state,
    )

    result = service.analyze_case(
        source,
        tmp_path / "output",
        case_id="case-1",
        run_id="dicom-run",
    )

    assert result.execution_status == ExecutionStatus.SUCCEEDED
    assert np.array_equal(captured["array"], array)
    assert captured["path"].name == "converted_input.nii.gz"
    assert not captured["path"].exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["input"]["input_format"] == "dicom"
    assert manifest["input"]["dicom"]["series_instance_uid_sha256"]
    assert captured["patient_metadata"] == {
        "patient_name": "Patient Identifying",
        "date_of_birth": "1984-12-03",
        "scan_date": "2026-07-19",
        "sex": "F",
    }
    manifest_text = json.dumps(manifest)
    assert uid not in manifest_text
    assert "private-patient-name" not in manifest_text
    assert "Patient Identifying" not in manifest_text
    assert "1984-12-03" not in manifest_text
    assert not any(
        item["relative_path"].endswith("converted_input.nii.gz") for item in manifest["artifacts"]
    )

    staged_input = tmp_path / "transfer" / "case-1.nii.gz"
    staged_conversion = convert_dicom(source, staged_input)
    staged = service.analyze_case(
        staged_input,
        tmp_path / "staged-output",
        case_id="case-1",
        run_id="staged-nifti",
    )
    staged_manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert staged.execution_status == ExecutionStatus.SUCCEEDED
    assert captured["path"] == staged_input
    assert staged_manifest["input"]["input_format"] == "nifti"
    assert staged_manifest["input"]["dicom"]["scanner_model"] == "Synthetic CT 1.0"
    assert staged_manifest["input"]["prestage"]["source_format"] == "dicom"
    assert (
        staged_manifest["input"]["prestage"]["sidecar_content_sha256"]
        == staged_conversion.metadata_content_sha256
    )
    assert captured["patient_metadata"] == {}

    convert = service_module.convert_dicom

    def changed_conversion(*args, **kwargs):
        conversion = convert(*args, **kwargs)
        changed_summary = dict(conversion.input_summary)
        changed_summary["input_pixel_sha256"] = "0" * 64
        return replace(conversion, input_summary=changed_summary)

    monkeypatch.setattr(service_module, "convert_dicom", changed_conversion)
    changed = service.analyze_case(
        source,
        tmp_path / "changed-output",
        case_id="case-1",
        run_id="changed-dicom",
    )
    assert changed.execution_status == ExecutionStatus.FAILED
    assert changed.failure["code"] == "InputChangedError"
