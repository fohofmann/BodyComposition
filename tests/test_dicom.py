from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
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
    NIFTI_ROUNDTRIP_GEOMETRY_ATOL,
    DicomConversionBatchResult,
    DicomConversionMetadataError,
    DicomConversionResult,
    DicomInputError,
    DicomSeriesSelectionError,
    dicom_report_patient_metadata,
)
from BodyComposition.model_manager import ModelStatus
from BodyComposition.provenance import file_sha256, image_summary
from BodyComposition.results import ExecutionStatus
from BodyComposition.service import InputDiscoveryError, PipelineService
from BodyComposition.utils.geometry import GeometryError, ImageGeometry, assert_same_physical_domain


def _write_dicom_series(
    directory: Path,
    array_zyx: np.ndarray,
    *,
    series_uid: str,
    modality: str = "CT",
    spacing_xyz: tuple[float, float, float] = (0.8, 0.9, 2.5),
    slice_thickness_mm: float | None = None,
    study_uid: str = "1.2.826.0.1.3680043.10.999.900",
    patient_id: str = "SYNTHETIC-MRN-001",
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
    extra_metadata: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array_zyx.astype(np.int16, copy=False))
    image.SetSpacing(spacing_xyz)
    image.SetOrigin(origin_lps_xyz)
    image.SetDirection(direction_lps)
    direction = np.asarray(direction_lps, dtype=float).reshape(3, 3)
    orientation = (*direction[:, 0], *direction[:, 1])
    paths = []
    for index in range(image.GetSize()[2]):
        image_slice = image[:, :, index]
        position = image.TransformIndexToPhysicalPoint((0, 0, index))
        metadata = {
            "0008|0008": "ORIGINAL\\PRIMARY\\AXIAL",
            "0008|0018": f"{series_uid}.{index + 1}",
            "0008|0012": "20260810",
            "0008|0013": "153045",
            "0008|0021": "20260719",
            "0008|0060": modality,
            "0008|0070": "Research Test Imaging",
            "0008|0020": "20260718",
            "0008|0022": "20260719",
            "0008|0023": "20260719",
            "0008|002a": "20260719101530",
            "0008|1030": "Synthetic abdomen study",
            "0008|103e": "Portal venous 2.5 mm",
            "0008|1090": "Synthetic CT 1.0",
            "0010|0010": "Synthetic^Subject",
            "0010|0020": patient_id,
            "0010|0030": "19841203",
            "0010|0040": "F",
            "0018|0050": str(spacing_xyz[2] if slice_thickness_mm is None else slice_thickness_mm),
            "0018|0015": "ABDOMEN",
            "0018|0060": "120",
            "0018|0088": str(spacing_xyz[2]),
            "0018|1030": "Abdomen portal venous",
            "0018|5100": "HFS",
            "0020|000d": study_uid,
            "0020|000e": series_uid,
            "0020|0011": "3",
            "0020|0013": str(index + 1),
            "0020|0032": "\\".join(f"{value:.12g}" for value in position),
            "0020|0037": "\\".join(f"{value:.12g}" for value in orientation),
            "0028|0030": f"{spacing_xyz[1]}\\{spacing_xyz[0]}",
        }
        if extra_metadata is not None:
            metadata.update(extra_metadata)
        for key, value in metadata.items():
            image_slice.SetMetaData(key, value)
        path = directory / f"slice-{index:04d}.dcm"
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        writer.SetFileName(str(path))
        writer.Execute(image_slice)
        paths.append(path)
    return tuple(paths)


def _write_dicom_instance(
    path: Path,
    array_yx: np.ndarray,
    *,
    series_uid: str,
    sop_instance_suffix: int,
    orientation_lps: tuple[float, ...],
    position_lps_xyz: tuple[float, ...],
    image_type: str,
    acquisition_number: str | None = None,
    study_uid: str = "1.2.826.0.1.3680043.10.999.900",
    patient_id: str = "SYNTHETIC-MRN-001",
    extra_metadata: Mapping[str, str] | None = None,
) -> Path:
    image = sitk.GetImageFromArray(array_yx.astype(np.int16, copy=False))
    metadata = {
        "0008|0008": image_type,
        "0008|0018": f"{series_uid}.{sop_instance_suffix}",
        "0008|0012": "20260810",
        "0008|0013": "153045",
        "0008|0021": "20260719",
        "0008|0060": "CT",
        "0008|0070": "Research Test Imaging",
        "0008|0020": "20260718",
        "0008|0022": "20260719",
        "0008|0023": "20260719",
        "0008|002a": "20260719101530",
        "0008|1030": "Synthetic abdomen study",
        "0008|103e": "Portal venous 2.5 mm",
        "0008|1090": "Synthetic CT 1.0",
        "0018|0050": "2.5",
        "0018|0015": "ABDOMEN",
        "0018|0060": "120",
        "0018|0088": "2.5",
        "0018|1030": "Abdomen portal venous",
        "0018|5100": "HFS",
        "0010|0020": patient_id,
        "0020|000d": study_uid,
        "0020|000e": series_uid,
        "0020|0011": "3",
        "0020|0013": str(sop_instance_suffix),
        "0020|0032": "\\".join(f"{value:.12g}" for value in position_lps_xyz),
        "0020|0037": "\\".join(f"{value:.12g}" for value in orientation_lps),
        "0028|0030": "0.9\\0.8",
    }
    if acquisition_number is not None:
        metadata["0020|0012"] = acquisition_number
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    pixel_spacing = metadata["0028|0030"].split("\\")
    image.SetSpacing((float(pixel_spacing[1]), float(pixel_spacing[0])))
    for key, value in metadata.items():
        image.SetMetaData(key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(str(path))
    writer.Execute(image)
    return path


def test_dicom_conversion_preserves_pixels_geometry_and_excludes_identifiers(tmp_path):
    source = tmp_path / "synthetic-subject-001" / "series"
    uid = "1.2.826.0.1.3680043.10.999.1"
    array = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6) - 500
    direction = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    _write_dicom_series(
        source,
        array,
        series_uid=uid,
        direction_lps=direction,
        slice_thickness_mm=1.25,
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
    converted_at = datetime.fromisoformat(result.conversion_created_at)
    assert converted_at.tzinfo is not None
    assert nifti_summary["input_pixel_sha256"] == summary["input_pixel_sha256"]
    assert nifti_summary["input_format"] == "nifti"
    assert nifti_summary["dicom"] == summary["dicom"]
    assert nifti_summary["prestage"]["source_format"] == "dicom"
    assert nifti_summary["prestage"]["schema_version"] == "1.5.0"
    assert nifti_summary["prestage"]["conversion_created_at"]
    assert nifti_summary["prestage"]["source_content_sha256"] == summary["source_content_sha256"]
    assert summary["input_format"] == "dicom"
    assert summary["dicom"]["image_orientation_patient_complete"]
    assert summary["dicom"]["image_position_patient_complete"]
    assert summary["dicom"]["scanner_manufacturer"] == "Research Test Imaging"
    assert summary["dicom"]["scanner_model"] == "Synthetic CT 1.0"
    assert summary["dicom"]["slice_thickness_mm"] == pytest.approx(1.25)
    assert summary["dicom"]["acquisition_date"] == "2026-07-19"
    assert summary["dicom"]["acquisition_datetime"] == "20260719101530"
    assert summary["dicom"]["study_date"] == "2026-07-18"
    assert summary["dicom"]["series_date"] == "2026-07-19"
    assert summary["dicom"]["content_date"] == "2026-07-19"
    assert summary["dicom"]["instance_creation_date"] == "2026-08-10"
    assert summary["dicom"]["instance_creation_time"] == "153045"
    assert summary["dicom"]["series_description"] == "Portal venous 2.5 mm"
    assert summary["dicom"]["protocol_name"] == "Abdomen portal venous"
    assert summary["dicom"]["study_description"] == "Synthetic abdomen study"
    assert summary["dicom"]["body_part_examined"] == "ABDOMEN"
    assert summary["dicom"]["patient_position"] == "HFS"
    assert summary["dicom"]["pixel_spacing_row_column_mm"] == pytest.approx([0.9, 0.8])
    assert len(summary["dicom"]["sop_instance_uid_sha256"]) == array.shape[0]
    assert summary["dicom"]["selection"]["series_strategy"] == "single-ct-series"
    assert not summary["dicom"]["selection"]["manual_review_recommended"]
    assert summary["geometry"]["spacing_xyz"][2] == pytest.approx(2.5)
    assert dicom_report_patient_metadata(source) == {
        "patient_name": "Subject Synthetic",
        "date_of_birth": "1984-12-03",
        "scan_date": "2026-07-19",
        "sex": "F",
    }
    serialized = json.dumps(summary)
    serialized_metadata = result.metadata_path.read_text(encoding="utf-8")
    assert uid not in serialized
    assert uid not in serialized_metadata
    assert "SYNTHETIC-MRN-001" not in serialized
    assert "SYNTHETIC-MRN-001" not in serialized_metadata
    assert str(tmp_path) not in serialized
    assert str(tmp_path) not in serialized_metadata
    assert not any(key.startswith("0010|") for key in nifti_image.GetMetaDataKeys())


def test_dicom_conversion_accepts_nifti_float32_affine_rounding(tmp_path):
    source = tmp_path / "dicom"
    array = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)
    _write_dicom_series(
        source,
        array,
        series_uid="1.2.826.0.1.3680043.10.999.101",
        origin_lps_xyz=(-198.599609375, -375.599609375, -1027.9),
    )

    dicom_image, _ = image_summary(source)
    output = tmp_path / "converted.nii.gz"
    convert_dicom(source, output)
    nifti_image = sitk.ReadImage(str(output))
    reference = ImageGeometry.from_sitk(dicom_image)
    converted = ImageGeometry.from_sitk(nifti_image)

    origin_delta = np.max(np.abs(np.asarray(reference.origin_lps_xyz) - converted.origin_lps_xyz))
    assert 1e-5 < origin_delta < NIFTI_ROUNDTRIP_GEOMETRY_ATOL
    assert_same_physical_domain(
        reference,
        converted,
        atol=NIFTI_ROUNDTRIP_GEOMETRY_ATOL,
    )
    assert np.array_equal(sitk.GetArrayFromImage(nifti_image), array)

    shifted = replace(
        converted,
        origin_lps_xyz=(
            converted.origin_lps_xyz[0],
            converted.origin_lps_xyz[1],
            converted.origin_lps_xyz[2] + 1e-3,
        ),
    )
    with pytest.raises(GeometryError, match="origin_lps_xyz differ"):
        assert_same_physical_domain(
            reference,
            shifted,
            atol=NIFTI_ROUNDTRIP_GEOMETRY_ATOL,
        )


def test_conversion_automatically_excludes_tagged_accessory_image(
    tmp_path,
    capsys,
):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.102"
    array = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    _write_dicom_series(source, array, series_uid=uid)
    localizer = _write_dicom_instance(
        source / "orthogonal-localizer.dcm",
        np.full((5, 6), 900, dtype=np.int16),
        series_uid=uid,
        sop_instance_suffix=999,
        orientation_lps=(1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        position_lps_xyz=(10.0, 20.0, -30.0),
        image_type="DERIVED\\SECONDARY\\REFORMATTED",
    )
    assert discover_dicom_series(source)[0].instance_count == array.shape[0] + 1

    summarized, summary = image_summary(source)
    assert np.array_equal(sitk.GetArrayFromImage(summarized), array)
    assert summary["dicom"]["excluded_instance_count"] == 1

    output = tmp_path / "filtered.nii.gz"
    code = cli.main(
        [
            "convert",
            str(source),
            str(output),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    converted = sitk.ReadImage(str(output))
    assert code == cli.EXIT_OK
    assert np.array_equal(sitk.GetArrayFromImage(converted), array)
    assert payload["dicom"]["instance_count"] == array.shape[0]
    assert payload["dicom"]["discovered_instance_count"] == array.shape[0] + 1
    assert payload["dicom"]["excluded_instance_count"] == 1
    assert payload["dicom"]["instance_selection"] == ("exclude-tagged-accessory-instances")
    assert "excluded_instance_content_sha256" not in payload["dicom"]
    assert payload["dicom"]["excluded_instance_content_sha256_count"] == 1
    metadata = json.loads((tmp_path / "filtered.bodycomposition.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "1.5.0"
    assert metadata["conversion_created_at"]
    assert metadata["source_dicom"]["dicom"]["excluded_instance_content_sha256"] == [
        file_sha256(localizer)
    ]


@pytest.mark.parametrize(
    ("spacing_between_slices", "expected_values", "expected_positions", "expected_axis"),
    (
        ("1", [1, 2, 3], [0.0, 1.0, 2.0], 1.0),
        ("-1", [3, 2, 1], [2.0, 1.0, 0.0], -1.0),
    ),
)
def test_selected_instances_follow_gdcm_physical_slice_axis(
    tmp_path,
    spacing_between_slices,
    expected_values,
    expected_positions,
    expected_axis,
):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.139"
    for index, z_position in enumerate((0.0, 1.0, 2.0), start=1):
        path = _write_dicom_instance(
            source / f"slice-{index:04d}.dcm",
            np.full((5, 6), index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, z_position),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
        )
        if spacing_between_slices == "-1":
            payload = bytearray(path.read_bytes())
            tag = b"\x18\x00\x88\x00\x02\x00\x00\x00"
            offset = payload.index(tag) + len(tag)
            assert payload[offset : offset + 2] == b"1 "
            payload[offset : offset + 2] = b"-1"
            path.write_bytes(payload)
    _write_dicom_instance(
        source / "orthogonal-localizer.dcm",
        np.full((5, 6), 900, dtype=np.int16),
        series_uid=uid,
        sop_instance_suffix=999,
        orientation_lps=(1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        position_lps_xyz=(10.0, 20.0, 0.0),
        image_type="DERIVED\\SECONDARY\\REFORMATTED",
    )

    result = convert_dicom(source, tmp_path / "ct.nii.gz")
    image = sitk.ReadImage(str(result.output_path))
    array = sitk.GetArrayFromImage(image)

    assert array[:, 0, 0].tolist() == expected_values
    assert image.GetDirection()[8] == pytest.approx(expected_axis)
    assert [
        image.TransformIndexToPhysicalPoint((0, 0, index))[2] for index in range(image.GetSize()[2])
    ] == pytest.approx(expected_positions)
    assert result.input_summary["dicom"]["excluded_instance_count"] == 1


def test_axial_selection_fails_when_orientation_outlier_is_not_tagged(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.103"
    _write_dicom_series(
        source,
        np.zeros((4, 5, 6), dtype=np.int16),
        series_uid=uid,
    )
    _write_dicom_instance(
        source / "orthogonal-original.dcm",
        np.zeros((5, 6), dtype=np.int16),
        series_uid=uid,
        sop_instance_suffix=999,
        orientation_lps=(1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        position_lps_xyz=(10.0, 20.0, -30.0),
        image_type="ORIGINAL\\PRIMARY\\AXIAL",
    )

    with pytest.raises(DicomInputError, match="not tagged LOCALIZER"):
        convert_dicom(source, tmp_path / "ct.nii.gz")


def test_conversion_excludes_multiple_tagged_accessory_images(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.104"
    array = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    _write_dicom_series(source, array, series_uid=uid)
    excluded = {
        file_sha256(
            _write_dicom_instance(
                source / "ap-localizer.dcm",
                np.full((5, 6), 700, dtype=np.int16),
                series_uid=uid,
                sop_instance_suffix=998,
                orientation_lps=(1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
                position_lps_xyz=(10.0, 20.0, -30.0),
                image_type="LOCALIZER",
            )
        ),
        file_sha256(
            _write_dicom_instance(
                source / "derived-reformat.dcm",
                np.full((5, 6), 800, dtype=np.int16),
                series_uid=uid,
                sop_instance_suffix=999,
                orientation_lps=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
                position_lps_xyz=(10.0, 20.0, -30.0),
                image_type="DERIVED\\SECONDARY\\REFORMATTED",
            )
        ),
    }

    result = convert_dicom(source, tmp_path / "ct.nii.gz")

    assert np.array_equal(
        sitk.GetArrayFromImage(sitk.ReadImage(str(result.output_path))),
        array,
    )
    assert result.input_summary["dicom"]["excluded_instance_count"] == 2
    assert set(result.input_summary["dicom"]["excluded_instance_content_sha256"]) == excluded


def test_conversion_excludes_secondary_capture_without_localizer_tag(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.130"
    array = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    _write_dicom_series(source, array, series_uid=uid)
    screenshot = _write_dicom_instance(
        source / "screen-capture.dcm",
        np.full((5, 6), 900, dtype=np.int16),
        series_uid=uid,
        sop_instance_suffix=999,
        orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        position_lps_xyz=(10.0, 20.0, 90.0),
        image_type="DERIVED\\SECONDARY",
        extra_metadata={"0008|0016": "1.2.840.10008.5.1.4.1.1.7"},
    )

    result = convert_dicom(source, tmp_path / "ct.nii.gz")

    assert np.array_equal(
        sitk.GetArrayFromImage(sitk.ReadImage(str(result.output_path))),
        array,
    )
    assert result.input_summary["dicom"]["excluded_instance_count"] == 1
    assert result.input_summary["dicom"]["excluded_instance_content_sha256"] == [
        file_sha256(screenshot)
    ]


def test_conversion_excludes_explicit_scout_image(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.133"
    array = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    _write_dicom_series(source, array, series_uid=uid)
    scout = _write_dicom_instance(
        source / "scout.dcm",
        np.full((5, 6), 700, dtype=np.int16),
        series_uid=uid,
        sop_instance_suffix=999,
        orientation_lps=(1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        position_lps_xyz=(10.0, 20.0, -30.0),
        image_type="DERIVED\\SECONDARY\\SCOUT",
    )

    result = convert_dicom(source, tmp_path / "ct.nii.gz")

    assert result.input_summary["dicom"]["excluded_instance_count"] == 1
    assert result.input_summary["dicom"]["excluded_instance_content_sha256"] == [file_sha256(scout)]


def test_conversion_retains_one_uniform_derived_axial_stack(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.109"
    for index, z_position in enumerate((0.0, 2.5, 5.0), start=1):
        _write_dicom_instance(
            source / f"slice-{index:04d}.dcm",
            np.full((5, 6), index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, z_position),
            image_type="DERIVED\\SECONDARY\\REFORMATTED",
        )

    result = convert_dicom(source, tmp_path / "ct.nii.gz")

    assert result.input_summary["dicom"]["instance_count"] == 3
    assert result.input_summary["dicom"]["excluded_instance_count"] == 0


def test_conversion_rejects_two_valid_axial_orientation_groups(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.110"
    _write_dicom_series(
        source,
        np.zeros((4, 5, 6), dtype=np.int16),
        series_uid=uid,
    )
    angle = np.deg2rad(5.0)
    orientation = (1.0, 0.0, 0.0, 0.0, float(np.cos(angle)), float(np.sin(angle)))
    normal = np.cross(np.asarray(orientation[:3]), np.asarray(orientation[3:]))
    for index in range(3):
        position = np.asarray((30.0, 40.0, 50.0)) + index * 2.5 * normal
        _write_dicom_instance(
            source / f"derived-{index:04d}.dcm",
            np.full((5, 6), 500 + index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=900 + index,
            orientation_lps=orientation,
            position_lps_xyz=tuple(float(value) for value in position),
            image_type="DERIVED\\SECONDARY\\REFORMATTED",
        )

    with pytest.raises(DicomInputError, match="exactly one uniformly spaced axial"):
        convert_dicom(source, tmp_path / "ct.nii.gz")


def test_conversion_rejects_a_standalone_localizer_series(tmp_path):
    source = tmp_path / "dicom"
    _write_dicom_instance(
        source / "localizer.dcm",
        np.zeros((5, 6), dtype=np.int16),
        series_uid="1.2.826.0.1.3680043.10.999.105",
        sop_instance_suffix=1,
        orientation_lps=(1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        position_lps_xyz=(10.0, 20.0, -30.0),
        image_type="LOCALIZER",
    )

    with pytest.raises(DicomInputError, match="only localizer, secondary-capture"):
        convert_dicom(source, tmp_path / "ct.nii.gz")


def test_conversion_rejects_single_file_enhanced_ct_with_actionable_error(tmp_path):
    source = tmp_path / "dicom"
    _write_dicom_instance(
        source / "enhanced-ct.dcm",
        np.zeros((5, 6), dtype=np.int16),
        series_uid="1.2.826.0.1.3680043.10.999.135",
        sop_instance_suffix=1,
        orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        position_lps_xyz=(10.0, 20.0, -30.0),
        image_type="ORIGINAL\\PRIMARY\\AXIAL",
        extra_metadata={
            "0008|0016": "1.2.840.10008.5.1.4.1.1.2.1",
            "0028|0008": "3",
        },
    )

    with pytest.raises(DicomInputError, match="Single-file Enhanced CT multi-frame"):
        convert_dicom(source, tmp_path / "ct.nii.gz")


def test_conversion_rejects_a_non_axial_primary_stack(tmp_path):
    source = tmp_path / "dicom"
    _write_dicom_series(
        source,
        np.zeros((3, 5, 6), dtype=np.int16),
        series_uid="1.2.826.0.1.3680043.10.999.106",
        direction_lps=(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0),
    )

    with pytest.raises(DicomInputError, match="not axial"):
        convert_dicom(source, tmp_path / "ct.nii.gz")


def test_conversion_rejects_missing_slice_positions(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.107"
    for index, z_position in enumerate((0.0, 2.5, 7.5), start=1):
        _write_dicom_instance(
            source / f"slice-{index:04d}.dcm",
            np.zeros((5, 6), dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, z_position),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
        )

    with pytest.raises(DicomInputError, match="not uniformly spaced"):
        convert_dicom(source, tmp_path / "ct.nii.gz")


def test_conversion_accepts_small_dicom_position_rounding(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.108"
    for index, z_position in enumerate((0.0, 0.799, 1.599, 2.4), start=1):
        _write_dicom_instance(
            source / f"slice-{index:04d}.dcm",
            np.full((5, 6), index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, z_position),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
        )

    result = convert_dicom(source, tmp_path / "ct.nii.gz")

    assert result.input_summary["dicom"]["instance_count"] == 4
    assert result.input_summary["dicom"]["excluded_instance_count"] == 0
    repair_codes = {
        item["code"] for item in result.input_summary["dicom"]["selection"]["header_repairs"]
    }
    assert "normalize_slice_position_rounding" in repair_codes
    assert result.input_summary["dicom"]["selection"]["manual_review_recommended"]


def test_conversion_records_small_header_normalizations(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.131"
    orientations = (
        (1.0, 0.0, 0.0, 0.0, 1.00001, 0.0),
        (1.0, 0.0, 0.0, 0.00001, 1.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, 0.99999, 0.0),
    )
    pixel_spacings = ("0.9000\\0.8000", "0.9002\\0.8001", "0.8999\\0.7999")
    thicknesses = ("2.50", "2.51", "2.49")
    for index, (orientation, pixel_spacing, thickness) in enumerate(
        zip(orientations, pixel_spacings, thicknesses, strict=True),
        start=1,
    ):
        _write_dicom_instance(
            source / f"slice-{index:04d}.dcm",
            np.full((5, 6), index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=index,
            orientation_lps=orientation,
            position_lps_xyz=(10.0, 20.0, float(index - 1) * 2.5),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
            extra_metadata={
                "0028|0030": pixel_spacing,
                "0018|0050": thickness,
            },
        )

    result = convert_dicom(source, tmp_path / "ct.nii.gz")

    repairs = result.input_summary["dicom"]["selection"]["header_repairs"]
    assert {item["code"] for item in repairs} == {
        "normalize_orientation_rounding",
        "normalize_pixel_spacing_rounding",
        "normalize_declared_slice_thickness",
    }
    assert result.input_summary["dicom"]["slice_thickness_mm"] == pytest.approx(2.5)
    assert result.input_summary["geometry"]["spacing_xyz"] == pytest.approx([0.8, 0.9, 2.5])


def test_header_normalization_preserves_negative_slice_axis(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.134"
    for index, (z_position, column_y) in enumerate(
        ((0.0, -1.00001), (-2.5, -1.0), (-5.0, -0.99999)),
        start=1,
    ):
        _write_dicom_instance(
            source / f"slice-{index:04d}.dcm",
            np.full((5, 6), index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, column_y, 0.0),
            position_lps_xyz=(10.0, 20.0, z_position),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
        )

    result = convert_dicom(source, tmp_path / "ct.nii.gz")

    assert result.input_summary["geometry"]["direction_lps"] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]
    )


def test_conversion_rejects_material_pixel_spacing_inconsistency(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.132"
    for index, row_spacing in enumerate((0.9, 0.9, 1.1), start=1):
        _write_dicom_instance(
            source / f"slice-{index:04d}.dcm",
            np.full((5, 6), index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, float(index - 1) * 2.5),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
            extra_metadata={"0028|0030": f"{row_spacing}\\0.8"},
        )

    with pytest.raises(DicomInputError, match="inconsistent in-plane pixel spacing"):
        convert_dicom(source, tmp_path / "ct.nii.gz")


@pytest.mark.parametrize(
    ("rescale_values", "message"),
    (
        ((("0", "1"), ("1", "1"), ("0", "1")), "inconsistent Rescale Intercept"),
        ((("0", "1"), ("0", "2"), ("0", "1")), "inconsistent Rescale Slope"),
        ((("0", "1"), ("0", "-1"), ("0", "1")), "positive Rescale Slope"),
    ),
)
def test_conversion_rejects_invalid_or_inconsistent_hu_rescale(
    tmp_path,
    rescale_values,
    message,
):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.136"
    for index, (intercept, slope) in enumerate(rescale_values, start=1):
        _write_dicom_instance(
            source / f"slice-{index:04d}.dcm",
            np.full((5, 6), index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, float(index - 1) * 2.5),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
            extra_metadata={"0028|1052": intercept, "0028|1053": slope},
        )

    with pytest.raises(DicomInputError, match=message):
        convert_dicom(source, tmp_path / "ct.nii.gz")


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


def test_multiple_ct_series_automatically_select_best_axial_stack(tmp_path):
    root = tmp_path / "dicom"
    uid_coarse = "1.2.826.0.1.3680043.10.999.121"
    uid_fine = "1.2.826.0.1.3680043.10.999.122"
    _write_dicom_series(
        root / "coarse",
        np.full((3, 4, 5), 11, dtype=np.int16),
        series_uid=uid_coarse,
        spacing_xyz=(0.8, 0.9, 5.0),
    )
    _write_dicom_series(
        root / "fine",
        np.full((5, 4, 5), 22, dtype=np.int16),
        series_uid=uid_fine,
        spacing_xyz=(0.8, 0.9, 2.5),
    )

    result = convert_dicom(root, tmp_path / "selected.nii.gz")

    selected = sitk.ReadImage(str(result.output_path))
    assert np.unique(sitk.GetArrayFromImage(selected)).tolist() == [22]
    assert result.series_instance_uid == uid_fine
    selection = result.input_summary["dicom"]["selection"]
    assert selection["series_strategy"] == "automatic-best-series"
    assert selection["evaluated_ct_series_count"] == 2
    assert selection["eligible_stack_count"] == 2
    assert selection["manual_review_recommended"]
    assert selection["selected_metrics"]["coverage_mm"] == pytest.approx(10.0)
    assert selection["selected_metrics"]["slice_spacing_mm"] == pytest.approx(2.5)


def test_best_series_selection_rejects_identity_change_in_middle_instance(tmp_path):
    root = tmp_path / "dicom"
    uid_mixed = "1.2.826.0.1.3680043.10.999.126"
    for index, patient_id in enumerate(
        ("SYNTHETIC-MRN-A", "SYNTHETIC-MRN-B", "SYNTHETIC-MRN-A"),
        start=1,
    ):
        _write_dicom_instance(
            root / "mixed" / f"slice-{index:04d}.dcm",
            np.full((4, 5), index, dtype=np.int16),
            series_uid=uid_mixed,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, float(index - 1) * 2.5),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
            patient_id=patient_id,
        )
    _write_dicom_series(
        root / "consistent",
        np.full((3, 4, 5), 9, dtype=np.int16),
        series_uid="1.2.826.0.1.3680043.10.999.127",
        patient_id="SYNTHETIC-MRN-A",
    )

    with pytest.raises(DicomInputError, match="inconsistent patient, study, or series"):
        convert_dicom(root, tmp_path / "selected.nii.gz")


def test_conversion_rejects_inconsistent_frame_of_reference(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.137"
    for index, frame_uid in enumerate(("1.2.3.1", "1.2.3.2", "1.2.3.1"), start=1):
        _write_dicom_instance(
            source / f"slice-{index:04d}.dcm",
            np.full((5, 6), index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, float(index - 1) * 2.5),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
            extra_metadata={"0020|0052": frame_uid},
        )

    with pytest.raises(DicomInputError, match="inconsistent Frame of Reference UID"):
        convert_dicom(source, tmp_path / "ct.nii.gz")


def test_conversion_rejects_duplicate_sop_instance_uids(tmp_path):
    source = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.138"
    for index in range(3):
        _write_dicom_instance(
            source / f"slice-{index:04d}.dcm",
            np.full((5, 6), index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=1,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, float(index) * 2.5),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
        )

    with pytest.raises(DicomInputError, match="unique SOP Instance UIDs"):
        convert_dicom(source, tmp_path / "ct.nii.gz")


def test_multiple_acquisitions_are_ranked_and_support_explicit_override(tmp_path):
    root = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.123"
    for index, z_position in enumerate((0.0, 5.0, 10.0), start=1):
        _write_dicom_instance(
            root / f"acq-1-{index:04d}.dcm",
            np.full((4, 5), 10 + index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, z_position),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
            acquisition_number="1",
        )
    for index, z_position in enumerate((-10.0, -5.0, 0.0, 5.0, 10.0), start=1):
        _write_dicom_instance(
            root / f"acq-2-{index:04d}.dcm",
            np.full((4, 5), 20 + index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=100 + index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, z_position),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
            acquisition_number="2",
        )

    automatic = convert_dicom(root, tmp_path / "automatic.nii.gz")
    explicit = convert_dicom(
        root,
        tmp_path / "explicit.nii.gz",
        series_uid=uid,
        acquisition_number="1",
    )

    automatic_dicom = automatic.input_summary["dicom"]
    assert automatic_dicom["instance_count"] == 5
    assert automatic_dicom["excluded_instance_count"] == 3
    assert automatic_dicom["acquisition_number"] == "2"
    assert automatic_dicom["instance_selection"] == "automatic-best-axial-stack"
    assert automatic_dicom["selection"]["eligible_stack_count"] == 2
    assert automatic_dicom["selection"]["acquisition_strategy"] == ("automatic-best-acquisition")
    explicit_dicom = explicit.input_summary["dicom"]
    assert explicit_dicom["instance_count"] == 3
    assert explicit_dicom["acquisition_number"] == "1"
    assert explicit_dicom["instance_selection"] == "explicit-acquisition-number"
    assert explicit_dicom["selection"]["acquisition_strategy"] == ("explicit-acquisition-number")

    with pytest.raises(ValueError, match="requires an exact Series Instance UID"):
        convert_dicom(
            root,
            tmp_path / "unsafe-override.nii.gz",
            acquisition_number="1",
        )


def test_complete_acquisition_is_retained_when_another_acquisition_has_a_gap(tmp_path):
    root = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.128"
    for index, z_position in enumerate((0.0, 2.5, 5.0), start=1):
        _write_dicom_instance(
            root / f"complete-{index:04d}.dcm",
            np.full((4, 5), 10 + index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, z_position),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
            acquisition_number="1",
        )
    for index, z_position in enumerate((0.0, 2.5, 7.5), start=1):
        _write_dicom_instance(
            root / f"gapped-{index:04d}.dcm",
            np.full((4, 5), 20 + index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=100 + index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, z_position),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
            acquisition_number="2",
        )

    result = convert_dicom(root, tmp_path / "selected.nii.gz")

    dicom = result.input_summary["dicom"]
    assert dicom["acquisition_number"] == "1"
    assert dicom["instance_count"] == 3
    assert dicom["excluded_instance_count"] == 3
    assert dicom["selection"]["eligible_stack_count"] == 1
    assert dicom["selection"]["rejected_candidates"] == [
        {
            "series_instance_uid_sha256": dicom["series_instance_uid_sha256"],
            "acquisition_number": "2",
            "code": "DicomInputError",
            "summary": "The retained axial DICOM stack is not uniformly spaced.",
        }
    ]


def test_contiguous_acquisition_numbers_remain_one_coherent_stack(tmp_path):
    root = tmp_path / "dicom"
    uid = "1.2.826.0.1.3680043.10.999.129"
    for index, (z_position, acquisition_number) in enumerate(
        ((0.0, "1"), (2.5, "1"), (5.0, "1"), (7.5, "2"), (10.0, "2")),
        start=1,
    ):
        _write_dicom_instance(
            root / f"slice-{index:04d}.dcm",
            np.full((4, 5), index, dtype=np.int16),
            series_uid=uid,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, z_position),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
            acquisition_number=acquisition_number,
        )

    result = convert_dicom(root, tmp_path / "selected.nii.gz")

    dicom = result.input_summary["dicom"]
    assert dicom["instance_count"] == 5
    assert dicom["excluded_instance_count"] == 0
    assert dicom["instance_selection"] == "all-series-instances"
    assert dicom["acquisition_number"] is None
    assert dicom["selection"]["acquisition_strategy"] == "whole-series"
    assert not dicom["selection"]["manual_review_recommended"]


def test_best_series_selection_rejects_incomplete_geometry(tmp_path):
    root = tmp_path / "dicom"
    uid_incomplete = "1.2.826.0.1.3680043.10.999.124"
    uid_complete = "1.2.826.0.1.3680043.10.999.125"
    for index, z_position in enumerate((0.0, 2.5, 7.5), start=1):
        _write_dicom_instance(
            root / "incomplete" / f"slice-{index:04d}.dcm",
            np.full((4, 5), 10 + index, dtype=np.int16),
            series_uid=uid_incomplete,
            sop_instance_suffix=index,
            orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            position_lps_xyz=(10.0, 20.0, z_position),
            image_type="ORIGINAL\\PRIMARY\\AXIAL",
        )
    _write_dicom_series(
        root / "complete",
        np.full((3, 4, 5), 22, dtype=np.int16),
        series_uid=uid_complete,
    )

    result = convert_dicom(root, tmp_path / "selected.nii.gz")

    assert result.series_instance_uid == uid_complete
    selection = result.input_summary["dicom"]["selection"]
    assert selection["eligible_stack_count"] == 1
    assert len(selection["rejected_candidates"]) == 1
    assert selection["rejected_candidates"][0]["code"] == "DicomInputError"


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
    assert datetime.fromisoformat(payload["conversion_created_at"]).tzinfo is not None
    assert payload["series_instance_uid"] == uid
    assert payload["input_format"] == "dicom"
    assert payload["dicom"]["instance_creation_date"] == "2026-08-10"
    assert output.is_file()


def test_nested_dicom_cohort_conversion_is_transfer_ready_and_idempotent(tmp_path):
    source = tmp_path / "import"
    uid_a = "1.2.826.0.1.3680043.10.999.231"
    uid_b = "1.2.826.0.1.3680043.10.999.232"
    _write_dicom_series(
        source / "synthetic-subject-one" / "DICOMS",
        np.full((2, 3, 4), 21, dtype=np.int16),
        series_uid=uid_a,
        study_uid="1.2.826.0.1.3680043.10.999.931",
        patient_id="SYNTHETIC-MRN-A",
    )
    _write_dicom_series(
        source / "nested" / "synthetic-subject-two" / "DICOM",
        np.full((3, 3, 4), 42, dtype=np.int16),
        series_uid=uid_b,
        study_uid="1.2.826.0.1.3680043.10.999.932",
        patient_id="SYNTHETIC-MRN-B",
    )
    output = tmp_path / "converted"

    result = convert_dicom(source, output)

    assert isinstance(result, DicomConversionBatchResult)
    assert result.succeeded
    assert result.discovered_series_count == 2
    assert result.discovered_study_count == 2
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
    assert "synthetic-subject-one" not in public_text
    assert "synthetic-subject-two" not in public_text
    assert uid_a not in public_text
    assert uid_b not in public_text

    repeated = convert_dicom(source, output)
    assert isinstance(repeated, DicomConversionBatchResult)
    assert repeated.succeeded
    assert [item.output_content_sha256 for item in repeated.conversions] == [
        item.output_content_sha256 for item in result.conversions
    ]


def test_cohort_conversion_selects_one_best_stack_per_study(tmp_path):
    source = tmp_path / "import"
    study_a = "1.2.826.0.1.3680043.10.999.971"
    coarse_uid = "1.2.826.0.1.3680043.10.999.271"
    fine_uid = "1.2.826.0.1.3680043.10.999.272"
    study_b_uid = "1.2.826.0.1.3680043.10.999.273"
    _write_dicom_series(
        source / "synthetic-subject-a" / "coarse",
        np.full((3, 4, 5), 1, dtype=np.int16),
        series_uid=coarse_uid,
        study_uid=study_a,
        patient_id="SYNTHETIC-MRN-A",
        spacing_xyz=(0.8, 0.9, 5.0),
    )
    _write_dicom_series(
        source / "synthetic-subject-a" / "fine",
        np.full((5, 4, 5), 2, dtype=np.int16),
        series_uid=fine_uid,
        study_uid=study_a,
        patient_id="SYNTHETIC-MRN-A",
        spacing_xyz=(0.8, 0.9, 2.5),
    )
    _write_dicom_series(
        source / "synthetic-subject-b" / "DICOM",
        np.full((3, 4, 5), 3, dtype=np.int16),
        series_uid=study_b_uid,
        study_uid="1.2.826.0.1.3680043.10.999.972",
        patient_id="SYNTHETIC-MRN-B",
    )

    result = convert_dicom(source, tmp_path / "converted")

    assert isinstance(result, DicomConversionBatchResult)
    assert result.succeeded
    assert result.discovered_series_count == 3
    assert result.discovered_study_count == 2
    assert {item.series_instance_uid for item in result.conversions} == {
        fine_uid,
        study_b_uid,
    }


def test_cli_converts_nested_dicom_cohort_to_directory(tmp_path, capsys):
    source = tmp_path / "import"
    for index in range(2):
        _write_dicom_series(
            source / f"case-{index}" / "DICOMS",
            np.full((2, 3, 4), index + 1, dtype=np.int16),
            series_uid=f"1.2.826.0.1.3680043.10.999.24{index}",
            study_uid=f"1.2.826.0.1.3680043.10.999.94{index}",
            patient_id=f"SYNTHETIC-MRN-{index}",
        )

    code = cli.main(["convert", str(source), str(tmp_path / "converted"), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == cli.EXIT_OK
    assert payload["execution_status"] == "succeeded"
    assert payload["converted_count"] == 2
    assert payload["discovered_study_count"] == 2
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
        study_uid="1.2.826.0.1.3680043.10.999.950",
        patient_id="SYNTHETIC-MRN-A",
    )
    _write_dicom_series(
        dicom_root / "patient-b" / "DICOMS",
        np.full((2, 3, 4), 2, dtype=np.int16),
        series_uid=uid_b,
        study_uid="1.2.826.0.1.3680043.10.999.951",
        patient_id="SYNTHETIC-MRN-B",
    )

    cases = discover_case_inputs(dicom_root)
    assert [case.series_uid for case in cases] == [None, None]
    assert [case.study_uid for case in cases] == [
        "1.2.826.0.1.3680043.10.999.950",
        "1.2.826.0.1.3680043.10.999.951",
    ]
    assert len(discover_case_inputs(dicom_root, series_uid=uid_b)) == 1

    nifti_root = tmp_path / "nifti"
    nifti_root.mkdir()
    sitk.WriteImage(sitk.Image((3, 4, 2), sitk.sitkInt16), str(nifti_root / "ct.nii.gz"))
    sitk.WriteImage(sitk.Image((3, 4, 2), sitk.sitkUInt8), str(nifti_root / "mask.nii.gz"))
    with pytest.raises(InputDiscoveryError, match="not every file has"):
        discover_case_inputs(nifti_root)

    with pytest.raises(InputDiscoveryError, match="cannot be used with a NIfTI"):
        discover_case_inputs(nifti_root / "ct.nii.gz", series_uid=uid_a)


def test_directory_analysis_resolves_one_multiseries_study_as_one_case(tmp_path):
    root = tmp_path / "dicom"
    _write_dicom_series(
        root / "coarse",
        np.full((3, 4, 5), 1, dtype=np.int16),
        series_uid="1.2.826.0.1.3680043.10.999.255",
        spacing_xyz=(0.8, 0.9, 5.0),
    )
    _write_dicom_series(
        root / "fine",
        np.full((5, 4, 5), 2, dtype=np.int16),
        series_uid="1.2.826.0.1.3680043.10.999.256",
        spacing_xyz=(0.8, 0.9, 2.5),
    )

    cases = discover_case_inputs(root)

    assert cases == (
        service_module.CaseInput(
            root,
            study_uid="1.2.826.0.1.3680043.10.999.900",
        ),
    )


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


def test_cohort_conversion_uses_study_identity_when_pixels_match(tmp_path):
    source = tmp_path / "import"
    pixels = np.full((2, 3, 4), 7, dtype=np.int16)
    _write_dicom_series(
        source / "first" / "DICOM",
        pixels,
        series_uid="1.2.826.0.1.3680043.10.999.261",
        study_uid="1.2.826.0.1.3680043.10.999.961",
    )
    _write_dicom_series(
        source / "second" / "DICOM",
        pixels,
        series_uid="1.2.826.0.1.3680043.10.999.262",
        study_uid="1.2.826.0.1.3680043.10.999.962",
    )

    result = convert_dicom(source, tmp_path / "converted")

    assert isinstance(result, DicomConversionBatchResult)
    assert result.succeeded
    assert len(result.conversions) == 2
    assert not result.failures
    assert len({value.case_id for value in result.conversions}) == 2
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["execution_status"] == "succeeded"
    assert report["converted_count"] == 2
    assert report["failed_count"] == 0


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
    source = tmp_path / "synthetic-private-subject" / "dicom"
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
        "patient_name": "Subject Synthetic",
        "date_of_birth": "1984-12-03",
        "scan_date": "2026-07-19",
        "sex": "F",
    }
    manifest_text = json.dumps(manifest)
    assert uid not in manifest_text
    assert "synthetic-private-subject" not in manifest_text
    assert "Subject Synthetic" not in manifest_text
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
