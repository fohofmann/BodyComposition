from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import requests
import SimpleITK as sitk

from BodyComposition.config import PipelineConfig
from BodyComposition.model_manager import required_model_ids, verify_models
from BodyComposition.service import inspect_result

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "tests/fixtures/public_ct/ct_org_volume_0.json"
RUN_ENVIRONMENT_VARIABLE = "BODYCOMPOSITION_RUN_REAL_WORLD_TEST"

TISSUE_MODEL_ASSETS = {
    "dataset.json": "8de2ff6b2b4229312517acf9bfd3dd4d23cc7ebedc48c0ad8e36bd3ba10e7f1d",
    "plans.json": "1790c1825529e29de536c73f66f5fd8ee12b9812271e93434e76a000c1e7cae0",
    "fold_0/checkpoint_final.pth": "d4eb90614fef10c2eb9aad574d4cb3517511d224f2f2da897fe4e4618513cd77",
    "fold_1/checkpoint_final.pth": "f520d3293d667986dd89fae115e3602af9f40c98413a08808d1cbb0ce08e074c",
    "fold_2/checkpoint_final.pth": "eeca2b9d95426147c6d533ee82ea43c215e33a785faaeb2443e5def4328bf42c",
    "fold_3/checkpoint_final.pth": "8a2d7fc6a32dccad3a43cd7fcad392b38df490303baf3842e8ee57587bcfbf4a",
    "fold_4/checkpoint_final.pth": "9d4c69c505961643bc97490e7a162258ac0bd3d2d17a6f58b8900d2143946231",
}

EXPECTED_OUTPUTS = (
    "orientation/orientation_report.json",
    "orientation/orientation_review.png",
    "masks/vertebral_bodies.nii.gz",
    "masks/spine_semantic.nii.gz",
    "masks/vertebra_labels.nii.gz",
    "masks/tissue_compartments.nii.gz",
    "masks/tissue_labels.nii.gz",
    "masks/totalsegmentator_landmarks.nii.gz",
    "masks/body_surface.nii.gz",
    "tables/slices.parquet",
    "tables/vertebrae.parquet",
    "tables/summaries.parquet",
    "tables/signature.parquet",
    "qc/vertebral_result.json",
    "qc/spine_review.png",
    "qc/qc.json",
    "qc/measurement_review.png",
    "logs/stages.jsonl",
    "case_manifest.json",
)


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_file_integrity(path: Path, integrity: dict) -> None:
    assert path.is_file(), f"Required file not found: {path}"
    assert path.stat().st_size == integrity["bytes"]
    assert _sha256(path) == integrity["sha256"]


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


def _release_config() -> PipelineConfig:
    configured = os.environ.get("BODYCOMPOSITION_MODEL_ROOT")
    assert configured, "BODYCOMPOSITION_MODEL_ROOT must select the mounted model cache"
    config = PipelineConfig.model_validate(
        {
            "models": {"root": configured},
            "orientation": {"model_device": "cpu"},
            "vertebrae": {"device": "cuda"},
            "runtime": {
                "device": "cuda",
                "timeout_seconds": 2400,
                "allow_dirty": True,
            },
        }
    )
    reports = verify_models(config)
    assert tuple(report.model_id for report in reports) == required_model_ids(config)
    assert all(report.ready for report in reports), {
        report.model_id: report.errors for report in reports if not report.ready
    }
    tissue = next(
        report for report in reports if report.model_id == "bodycomposition_resenc_l_v1"
    )
    assert dict(tissue.checked_files) == TISSUE_MODEL_ASSETS
    return config


def _assert_cuda_available() -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "assert torch.cuda.is_available(), 'CUDA is not available'; "
                "print(json.dumps({'torch': torch.__version__, "
                "'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _run_cli(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=3000,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "Traceback (most recent call last)" not in completed.stderr
    # JSON mode is an automation contract: upstream banners may not leak into stdout.
    return json.loads(completed.stdout)


def _snapshot(paths: list[Path], root: Path) -> dict:
    return {
        path.relative_to(root).as_posix(): {
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
def test_canonical_pipeline_on_pinned_public_ct(tmp_path):
    if os.environ.get(RUN_ENVIRONMENT_VARIABLE) != "1":
        pytest.skip(f"set {RUN_ENVIRONMENT_VARIABLE}=1 to run the public CT test")

    manifest = _load_manifest()
    public_ct = _public_ct_path(manifest)
    config = _release_config()
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

    output = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config.to_yaml(), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "BodyComposition.cli",
        "analyze",
        str(public_ct),
        "--output",
        str(output),
        "--config",
        str(config_path),
        "--case-id",
        manifest["asset_id"],
        "--run-id",
        "public-real-world",
        "--json",
    ]
    first = _run_cli(command)
    assert first["execution_status"] == "succeeded"
    bundle = Path(first["output_path"])
    output_paths = [bundle / relative for relative in EXPECTED_OUTPUTS]
    assert all(path.is_file() for path in output_paths)

    orientation = json.loads(
        (bundle / "orientation/orientation_report.json").read_text(encoding="utf-8")
    )
    assert orientation["state"] == "PASS_METADATA_MATCH"
    assert orientation["prediction"]["raw_winning_class_index"] == 0
    assert not orientation["orientation_changed"]
    assert not orientation["manual_review_required"]

    ct_image = sitk.ReadImage(str(public_ct))
    images = {
        name: sitk.ReadImage(str(bundle / relative))
        for name, relative in {
            "body": "masks/vertebral_bodies.nii.gz",
            "semantic": "masks/spine_semantic.nii.gz",
            "whole": "masks/vertebra_labels.nii.gz",
            "compartments": "masks/tissue_compartments.nii.gz",
            "tissues": "masks/tissue_labels.nii.gz",
            "surface": "masks/body_surface.nii.gz",
        }.items()
    }
    assert all(_same_geometry(ct_image, image) for image in images.values())
    body = sitk.GetArrayFromImage(images["body"])
    semantic = sitk.GetArrayFromImage(images["semantic"])
    whole = sitk.GetArrayFromImage(images["whole"])
    assert np.array_equal(body, np.where(semantic == 49, whole, 0))
    assert np.count_nonzero(body) <= np.count_nonzero(whole)

    slices = pd.read_parquet(bundle / "tables/slices.parquet")
    vertebrae = pd.read_parquet(bundle / "tables/vertebrae.parquet")
    summaries = pd.read_parquet(bundle / "tables/summaries.parquet")
    signature = pd.read_parquet(bundle / "tables/signature.parquet")
    assert len(slices) == ct_image.GetSize()[2]
    assert slices["slice_id"].is_unique
    assert np.all(np.diff(slices["position_superior_mm"].to_numpy()) > 0)
    assert {"L2", "L3", "L4"}.issubset(set(vertebrae["vertebral_level"]))
    l3 = vertebrae.loc[vertebrae["vertebral_level"].eq("L3")]
    assert l3["territory_bin"].tolist() == [1, 2, 3]
    assert len(summaries) == 1
    assert len(signature) == 100
    assert signature["bin_width_mm"].eq(20.0).all()
    assert signature["signature_bin"].tolist() == list(range(100))
    assert summaries.loc[0, "case_id"] == manifest["asset_id"]

    inspected = inspect_result(bundle)
    assert inspected.analysis_id == first["analysis_id"]
    manifest_text = inspected.manifest_path.read_text(encoding="utf-8")
    assert str(public_ct) not in manifest_text
    assert str(config.model_root) not in manifest_text
    stage_events = [
        json.loads(line)
        for line in (bundle / "logs/stages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert stage_events
    assert all(event["result_code"] == "succeeded" for event in stage_events)
    assert not list((output / "runs/public-real-world/.attempts").glob("**/*"))

    snapshot = _snapshot(output_paths, bundle)
    second = _run_cli(command)
    assert second["execution_status"] == "skipped_identical"
    assert _snapshot(output_paths, bundle) == snapshot

    report = {
        "asset_id": manifest["asset_id"],
        "input_sha256": manifest["integrity"]["sha256"],
        "runtime": runtime,
        "analysis_id": first["analysis_id"],
        "required_models": list(required_model_ids(config)),
        "slice_rows": len(slices),
        "vertebra_rows": len(vertebrae),
        "resume_unchanged": True,
        "output_snapshot": snapshot,
    }
    (tmp_path / "real_world_test_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
