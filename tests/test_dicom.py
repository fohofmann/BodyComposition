from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from BodyComposition import cli, convert_dicom, discover_dicom_series
from BodyComposition.config import PipelineConfig
from BodyComposition.dicom import DicomSeriesSelectionError
from BodyComposition.model_manager import ModelStatus
from BodyComposition.provenance import image_summary
from BodyComposition.results import ExecutionStatus
from BodyComposition.service import PipelineService
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
            "0008|1090": "Synthetic CT 1.0",
            "0010|0010": "Identifying^Patient",
            "0010|0020": "MRN-123456",
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
    assert nifti_summary["input_pixel_sha256"] == summary["input_pixel_sha256"]
    assert summary["input_format"] == "dicom"
    assert summary["dicom"]["image_orientation_patient_complete"]
    assert summary["dicom"]["image_position_patient_complete"]
    assert summary["dicom"]["scanner_manufacturer"] == "Research Test Imaging"
    assert summary["dicom"]["scanner_model"] == "Synthetic CT 1.0"
    serialized = json.dumps(summary)
    assert uid not in serialized
    assert "MRN-123456" not in serialized
    assert str(tmp_path) not in serialized
    assert not any(key.startswith("0010|") for key in nifti_image.GetMetaDataKeys())


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
    assert payload["series_instance_uid"] == uid
    assert payload["input_format"] == "dicom"
    assert output.is_file()


def test_pipeline_accepts_dicom_without_persisting_the_temporary_conversion(tmp_path):
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
    manifest_text = json.dumps(manifest)
    assert uid not in manifest_text
    assert "private-patient-name" not in manifest_text
    assert not any(
        item["relative_path"].endswith("converted_input.nii.gz") for item in manifest["artifacts"]
    )
