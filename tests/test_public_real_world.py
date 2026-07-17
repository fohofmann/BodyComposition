from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import requests
import SimpleITK as sitk

from BodyComposition.orientation.ctdeeprot import (
    CHECKPOINT_SHA256,
    UPSTREAM_COMMIT as CTDEEPROT_COMMIT,
    verify_checkpoint,
)
from BodyComposition.measurement.totalsegmentator_assets import (
    check_measurement_model,
)
from BodyComposition.vertebral.spineps_assets import verify_model_bundle
from BodyComposition.vertebral.spineps_manifest import (
    MODEL_BUNDLE_VERSION,
    SPINEPS_MODEL_ASSETS,
    VIBESEG_CROP_ASSETS,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "tests/fixtures/public_ct/ct_org_volume_0.json"
RUN_ENVIRONMENT_VARIABLE = "BODYCOMPOSITION_RUN_REAL_WORLD_TEST"

TISSUE_MODEL_ASSETS = {
    "Dataset611_BodyComposition/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/dataset.json": "9841afba9b5f2180d870f68e12eb3e5695c5c0e4f64b91dabc52d849d67c340c",
    "Dataset611_BodyComposition/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/plans.json": "d80992884b678fae2e6e09ad4147a30a59bcf39d6fd75f40da6a6bb57ef09afd",
    "Dataset611_BodyComposition/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_all/checkpoint_final.pth": "91ca20c9bd2674cbc0c25a5e56c70e3285b8e8f6bdf064430ad4a1d76de65f90",
}

EXPECTED_OUTPUTS = (
    "orientation/{case_id}/orientation_report.json",
    "orientation/{case_id}/orientation_review.png",
    "masks/{case_id}_vertebral-bodies.nii.gz",
    "masks/{case_id}_spineps-semantic.nii.gz",
    "masks/{case_id}_spineps-vertebrae.nii.gz",
    "qc/{case_id}_vertebral-result.json",
    "qc/{case_id}_spine-review.png",
    "labels/{case_id}_int-bodycomposition.nii.gz",
    "masks/{case_id}_int-bodycomposition.nii.gz",
    "labels/{case_id}_tseg-body_landmarks.nii.gz",
    "masks/{case_id}_body-surface.nii.gz",
    "tables/{case_id}/slices.parquet",
    "tables/{case_id}/vertebrae.parquet",
    "tables/{case_id}/summaries.parquet",
    "qc/{case_id}_measurement-qc.json",
    "qc/{case_id}_measurement-review.png",
)


def _load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_file_integrity(path: Path, integrity: dict) -> None:
    assert path.is_file(), f"Required file not found: {path}"
    assert path.stat().st_size == integrity["bytes"], (
        f"Unexpected byte count for {path}: {path.stat().st_size} != "
        f"{integrity['bytes']}"
    )
    assert _sha256(path) == integrity["sha256"], f"SHA-256 mismatch for {path}"


def _download_public_ct(manifest: dict, cache_directory: Path) -> Path:
    cache_directory.mkdir(parents=True, exist_ok=True)
    destination = cache_directory / manifest["file_name"]
    if destination.exists():
        _assert_file_integrity(destination, manifest["integrity"])
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        with requests.get(
            manifest["mirror"]["download_url"],
            stream=True,
            timeout=(30, 300),
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        _assert_file_integrity(partial, manifest["integrity"])
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def _public_ct_path(manifest: dict) -> Path:
    supplied_path = os.environ.get("BODYCOMPOSITION_PUBLIC_CT_PATH")
    if supplied_path:
        path = Path(supplied_path).expanduser().resolve()
        _assert_file_integrity(path, manifest["integrity"])
        return path

    cache_root = Path(
        os.environ.get(
            "BODYCOMPOSITION_TEST_DATA_CACHE",
            Path.home() / ".cache/bodycomposition/tests",
        )
    ).expanduser()
    return _download_public_ct(manifest, cache_root)


def _model_bundle() -> dict[str, Path]:
    configured = os.environ.get("BODYCOMPOSITION_MODEL_ROOT")
    assert configured, (
        "BODYCOMPOSITION_MODEL_ROOT must point to the directory containing "
        "Dataset611_BodyComposition, SPINEPS/spineps-veridah-ct-v1, CTDeepRot, "
        "and the pinned TotalSegmentator task-297 directory."
    )
    root = Path(configured).expanduser().resolve()
    assert root.is_dir(), f"Model root does not exist: {root}"
    for relative_path, expected_hash in TISSUE_MODEL_ASSETS.items():
        path = root / relative_path
        assert path.is_file(), f"Pinned model asset not found: {path}"
        assert _sha256(path) == expected_hash, f"Pinned model asset changed: {path}"
    spineps_root = root / "SPINEPS" / MODEL_BUNDLE_VERSION
    verify_model_bundle(spineps_root, full=True)
    checkpoint = Path(
        os.environ.get(
            "BODYCOMPOSITION_CTDEEPROT_CHECKPOINT",
            root / "CTDeepRot" / CTDEEPROT_COMMIT / "net2d.pt",
        )
    ).expanduser().resolve()
    verify_checkpoint(checkpoint)
    for task in ("body_landmarks",):
        report = check_measurement_model(task, root)
        assert report.ready, (
            f"Pinned {report.model_title} assets are not ready: {report.errors}"
        )
    return {
        "tissue": root / "Dataset611_BodyComposition",
        "spineps": spineps_root,
        "ctdeeprot": checkpoint,
        "totalsegmentator": root,
    }


def _assert_cuda_available() -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "assert torch.cuda.is_available(), 'CUDA is not available'; "
                "print(json.dumps({'torch': torch.__version__, "
                "'cuda': torch.version.cuda, "
                "'gpu': torch.cuda.get_device_name(0)}))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "The real-world model test requires a CUDA-capable controlled environment.\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _snapshot(paths: list[Path], root: Path) -> dict:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    }


def _same_geometry(left: sitk.Image, right: sitk.Image) -> bool:
    return (
        left.GetSize() == right.GetSize()
        and np.allclose(left.GetSpacing(), right.GetSpacing())
        and np.allclose(left.GetOrigin(), right.GetOrigin())
        and np.allclose(left.GetDirection(), right.GetDirection())
    )


def _run_cli(command: list[str]) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=2400,
    )
    assert completed.returncode == 0, (
        f"BodyComposition command failed with exit code {completed.returncode}.\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "Traceback (most recent call last)" not in completed.stdout
    assert "Traceback (most recent call last)" not in completed.stderr
    return completed


def test_public_ct_manifest_is_complete_and_keeps_scan_outside_package():
    manifest = _load_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["asset_id"] == "ct_org_volume-0_0000"
    assert manifest["source"]["license_spdx"] == "CC-BY-3.0"
    assert manifest["source"]["doi"].startswith("https://doi.org/")
    assert manifest["source"]["required_citation"]
    assert manifest["source"]["attribution"]
    assert manifest["changes"].startswith("None by BodyComposition")
    assert len(manifest["mirror"]["revision"]) == 40
    assert len(manifest["integrity"]["sha256"]) == 64
    assert manifest["integrity"]["bytes"] > 0
    assert manifest["distribution"]["bundled_with_bodycomposition"] is False
    assert not (MANIFEST_PATH.parent / manifest["file_name"]).exists()


@pytest.mark.model_integration
@pytest.mark.real_world
def test_bodycomposition_fast_on_pinned_public_ct(tmp_path):
    if os.environ.get(RUN_ENVIRONMENT_VARIABLE) != "1":
        pytest.skip(f"set {RUN_ENVIRONMENT_VARIABLE}=1 to run the public CT test")

    manifest = _load_manifest()
    public_ct = _public_ct_path(manifest)
    models = _model_bundle()
    runtime = _assert_cuda_available()

    nifti = nib.load(public_ct)
    expected_nifti = manifest["expected_nifti"]
    assert list(nifti.shape) == expected_nifti["shape_xyz"]
    assert np.allclose(nifti.header.get_zooms()[:3], expected_nifti["spacing_xyz_mm"])
    assert "".join(nib.aff2axcodes(nifti.affine)) == expected_nifti["orientation"]
    assert str(nifti.get_data_dtype()) == expected_nifti["dtype"]
    pixels = np.asanyarray(nifti.dataobj)
    assert np.isfinite(pixels).all()
    assert [int(pixels.min()), int(pixels.max())] == expected_nifti["voxel_value_range"]
    assert len(nifti.header.extensions) == expected_nifti["extensions"]
    header_text = {
        field: bytes(nifti.header[field]).rstrip(b"\x00").decode("utf-8")
        for field in expected_nifti["empty_header_text_fields"]
    }
    assert not any(header_text.values())

    workspace = tmp_path / "workspace"
    config_path = tmp_path / "real_world_config.json"
    config = {
        "paths": {
            "workspace": str(workspace),
            "logs": str(workspace / "logs/BodyCompositionFast_{timestamp}.log"),
            "weights": {
                "spineps": str(models["spineps"]),
                "int-bodycomposition": str(models["tissue"]),
                "totalsegmentator": str(models["totalsegmentator"]),
            },
            "cache": str(workspace / "cache"),
        },
        "logging_level": {"file": "INFO", "console": "WARNING"},
        "run": {"reset": False, "skip": True, "timeout": 1800},
        "orientation": {
            "model": {"checkpoint_path": str(models["ctdeeprot"])},
        },
        "segmentation": {"save_label": True},
        "vertebrae": {
            "backend": "spineps_veridah_ct_v1",
            "save_mask": True,
            "spineps": {
                "device": "cuda",
                "save_native_outputs": True,
                "review_enabled": True,
            },
        },
        "tissue": {"save_mask": True},
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "BodyComposition.bin.run_batch",
        "--input",
        str(public_ct),
        "--method",
        manifest["test_scope"]["pipeline"],
        "--config",
        str(config_path),
    ]
    _run_cli(command)

    case_id = manifest["asset_id"]
    output_paths = [
        workspace / template.format(case_id=case_id) for template in EXPECTED_OUTPUTS
    ]
    assert all(path.is_file() for path in output_paths)

    orientation_report = json.loads(
        (workspace / f"orientation/{case_id}/orientation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert orientation_report["state"] == "PASS_METADATA_MATCH"
    assert orientation_report["prediction"]["raw_winning_class_index"] == 0
    assert orientation_report["prediction"]["winning_class_index"] == 6
    assert not orientation_report["orientation_changed"]
    assert not orientation_report["manual_review_required"]

    ct_image = sitk.ReadImage(str(public_ct))
    vertebral_body_image = sitk.ReadImage(
        str(workspace / f"masks/{case_id}_vertebral-bodies.nii.gz")
    )
    spineps_semantic_image = sitk.ReadImage(
        str(workspace / f"masks/{case_id}_spineps-semantic.nii.gz")
    )
    spineps_whole_image = sitk.ReadImage(
        str(workspace / f"masks/{case_id}_spineps-vertebrae.nii.gz")
    )
    tissue_label_image = sitk.ReadImage(
        str(workspace / f"labels/{case_id}_int-bodycomposition.nii.gz")
    )
    tissue_mask_image = sitk.ReadImage(
        str(workspace / f"masks/{case_id}_int-bodycomposition.nii.gz")
    )
    body_surface_image = sitk.ReadImage(
        str(workspace / f"masks/{case_id}_body-surface.nii.gz")
    )

    assert _same_geometry(ct_image, vertebral_body_image)
    assert _same_geometry(ct_image, spineps_semantic_image)
    assert _same_geometry(ct_image, spineps_whole_image)
    assert _same_geometry(tissue_label_image, tissue_mask_image)
    assert _same_geometry(ct_image, tissue_label_image)
    assert _same_geometry(ct_image, body_surface_image)

    vertebral_body = sitk.GetArrayFromImage(vertebral_body_image)
    spineps_semantic = sitk.GetArrayFromImage(spineps_semantic_image)
    spineps_whole = sitk.GetArrayFromImage(spineps_whole_image)
    expected_vertebral_body = np.where(spineps_semantic == 49, spineps_whole, 0)
    assert np.array_equal(vertebral_body, expected_vertebral_body)
    assert np.all((vertebral_body == 0) | (vertebral_body == spineps_whole))
    assert np.count_nonzero(vertebral_body) <= np.count_nonzero(spineps_whole)

    vertebrae_values = np.unique(vertebral_body).tolist()
    tissue_values = np.unique(sitk.GetArrayViewFromImage(tissue_mask_image)).tolist()
    assert {21, 22, 23}.issubset(vertebrae_values)
    assert {1, 3}.issubset(tissue_values)

    vertebral_result = json.loads(
        (workspace / f"qc/{case_id}_vertebral-result.json").read_text(encoding="utf-8")
    )
    assert vertebral_result["backend_id"] == "spineps_veridah_ct_v1"
    assert vertebral_result["execution_status"] == "succeeded"
    assert vertebral_result["label_schema"]["22"] == "L3"
    assert vertebral_result["provenance"]["vibeseg_precomputed"] is True
    assert str(models["spineps"]) not in json.dumps(vertebral_result)

    table_directory = workspace / f"tables/{case_id}"
    slices = pd.read_parquet(table_directory / "slices.parquet")
    vertebrae = pd.read_parquet(table_directory / "vertebrae.parquet")
    summaries = pd.read_parquet(table_directory / "summaries.parquet")
    assert len(slices) == ct_image.GetSize()[2]
    assert slices["slice_id"].is_unique
    assert np.all(np.diff(slices["position_superior_mm"].to_numpy()) > 0)
    assert {"L2", "L3", "L4"}.issubset(set(vertebrae["vertebral_level"]))
    l3_rows = vertebrae.loc[vertebrae["vertebral_level"].eq("L3")]
    assert len(l3_rows) == 3
    assert l3_rows["territory_bin"].tolist() == [1, 2, 3]
    assert l3_rows["aggregation"].eq("vertebral_territory_third_mean").all()
    for column in ("sm_mean_csa_cm2", "sat_mean_csa_cm2"):
        values = l3_rows[column].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert ((values > 0) & (values < 1000)).all()
    assert len(summaries) == 1
    assert summaries.loc[0, "case_id"] == case_id

    measurement_qc = json.loads(
        (workspace / f"qc/{case_id}_measurement-qc.json").read_text(
            encoding="utf-8"
        )
    )
    assert measurement_qc["case_id"] == case_id
    assert measurement_qc["body_surface_backend"] == "tissue_segmentation_envelope_v1"
    assert measurement_qc["qc_status"] in {"pass", "review", "fail"}

    log_paths_after_first_run = set((workspace / "logs").glob("*.log"))
    assert log_paths_after_first_run
    first_log_text = "\n".join(
        path.read_text(encoding="utf-8") for path in log_paths_after_first_run
    )
    assert " - ERROR - " not in first_log_text
    assert "Traceback (most recent call last)" not in first_log_text

    first_snapshot = _snapshot(output_paths, workspace)
    _run_cli(command)
    second_snapshot = _snapshot(output_paths, workspace)
    assert second_snapshot == first_snapshot

    all_log_paths = set((workspace / "logs").glob("*.log"))
    second_run_logs = all_log_paths - log_paths_after_first_run
    assert second_run_logs, "The resume run did not create a separate audit log."
    second_log_text = "\n".join(
        path.read_text(encoding="utf-8") for path in second_run_logs
    )
    assert "final datalist: 0 cases" in second_log_text
    assert " - ERROR - " not in second_log_text

    report = {
        "asset_id": case_id,
        "input_sha256": manifest["integrity"]["sha256"],
        "input_bytes": manifest["integrity"]["bytes"],
        "ctdeeprot_sha256": CHECKPOINT_SHA256,
        "tissue_model_sha256": TISSUE_MODEL_ASSETS,
        "spineps_model_sha256": {
            asset.asset_name: asset.sha256 for asset in SPINEPS_MODEL_ASSETS
        },
        "vibeseg_model_sha256": {
            asset.asset_name: asset.sha256 for asset in VIBESEG_CROP_ASSETS
        },
        "totalsegmentator_model_sha256": {
            task: dict(check_measurement_model(task, models["totalsegmentator"]).checked_files)
            for task in ("body_landmarks",)
        },
        "runtime": runtime,
        "ct_size_xyz": list(ct_image.GetSize()),
        "ct_spacing_xyz_mm": list(ct_image.GetSpacing()),
        "prepared_size_xyz": list(tissue_label_image.GetSize()),
        "vertebrae_values": vertebrae_values,
        "tissue_values": tissue_values,
        "slice_rows": len(slices),
        "vertebra_rows": len(vertebrae),
        "vertebral_bins_per_level": 3,
        "output_snapshot": first_snapshot,
        "resume_unchanged": second_snapshot == first_snapshot,
    }
    (workspace / "real_world_test_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
