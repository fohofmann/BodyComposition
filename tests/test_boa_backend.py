from __future__ import annotations

import hashlib
import io
import zipfile
from itertools import permutations, product
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage

import BodyComposition.service as service_module
from BodyComposition import cli
from BodyComposition.actions.masks_int import MasksInternalTissue
from BodyComposition.actions.segm_int import SegmBoaBodyRegions
from BodyComposition.config import PipelineConfig, low_resource_config
from BodyComposition.measurement.contracts import MeasurementIdentity
from BodyComposition.measurement.distributions import (
    build_compartment_hu_distributions,
)
from BodyComposition.model_manager import (
    model_asset,
    required_model_ids,
    sync_model,
    verify_model,
)
from BodyComposition.pipelines.bodycomposition import canonical_actions
from BodyComposition.tissue_backends import boa_assets
from BodyComposition.tissue_backends.boa import (
    BOA_BACKEND_ID,
    BOA_NATIVE_COMPARTMENT_LABELS,
    postprocess_boa_body_regions,
    prepare_boa_inference,
    restore_boa_prediction,
)
from BodyComposition.utils.geometry import ImageGeometry
from BodyComposition.utils.nifti import NiftiDataContainer


def _archive(directory: str, files: dict[str, bytes], *, unsafe: bool = False) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(f"{directory}/", b"")
        for relative, content in files.items():
            archive.writestr(f"{directory}/{relative}", content)
        if unsafe:
            archive.writestr("../escaped", b"unsafe")
    return output.getvalue()


def _pin_test_asset(monkeypatch, payload: bytes, files: dict[str, bytes]) -> str:
    directory = "Dataset999_BOA_test"
    trainer = "trainer"
    monkeypatch.setattr(boa_assets, "DATASET_DIRECTORY", directory)
    monkeypatch.setattr(boa_assets, "TRAINER_DIRECTORY", trainer)
    monkeypatch.setattr(boa_assets, "ARCHIVE_NAME", f"{directory}.zip")
    monkeypatch.setattr(boa_assets, "DOWNLOAD_URL", "https://github.com/UMEssen/test.zip")
    monkeypatch.setattr(boa_assets, "ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(
        boa_assets,
        "ARCHIVE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(
        boa_assets,
        "EXPECTED_FILES",
        {
            relative.removeprefix(f"{trainer}/"): {
                "byte_size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for relative, content in files.items()
        },
    )
    return directory


def test_boa_asset_is_pinned_to_the_official_weight_release():
    asset = model_asset(BOA_BACKEND_ID)

    assert asset["upstream_repository"] == ("https://github.com/UMEssen/Body-and-Organ-Analysis")
    assert asset["upstream_commit"] == "6e761702a738ba681e6f7a3a7c944b00b186c3a8"
    assert asset["upstream_release"] == "v1.0.0-weights"
    assert asset["archive_byte_size"] == 1_150_464_739
    assert asset["archive_sha256"] == (
        "096d6cdb45524273026d30e3bd290e8fd10990dec02f0aa0669d214b71bf6d4f"
    )
    assert asset["code_license"] == "Apache-2.0"
    assert asset["weight_license"] == "Apache-2.0"
    assert asset["redistribution_mode"] == "user_model_sync"


def test_boa_sync_is_atomic_idempotent_and_uses_no_inference_download(
    tmp_path,
    monkeypatch,
):
    files = {
        "trainer/dataset.json": b"dataset",
        "trainer/plans.json": b"plans",
        "trainer/fold_0/checkpoint_final.pth": b"checkpoint",
    }
    payload = _archive("Dataset999_BOA_test", files)
    directory = _pin_test_asset(monkeypatch, payload, files)
    target = tmp_path / directory
    target.mkdir()
    (target / "partial").write_bytes(b"old")
    requests: list[str] = []

    def opener(request):
        requests.append(request.full_url)
        return io.BytesIO(payload)

    first = boa_assets.sync_boa_model(tmp_path, opener=opener)
    second = boa_assets.sync_boa_model(
        tmp_path,
        opener=lambda request: (_ for _ in ()).throw(
            AssertionError("verified BOA assets must not be downloaded again")
        ),
    )

    assert requests == ["https://github.com/UMEssen/test.zip"]
    assert first.ready and first.install_manifest_verified
    assert second.ready and second.install_manifest_verified
    assert not (target / "partial").exists()


def test_boa_sync_rejects_unsafe_archive_without_replacing_existing_data(
    tmp_path,
    monkeypatch,
):
    files = {"trainer/fold_0/checkpoint_final.pth": b"checkpoint"}
    payload = _archive("Dataset999_BOA_test", files, unsafe=True)
    directory = _pin_test_asset(monkeypatch, payload, files)
    target = tmp_path / directory
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_bytes(b"keep")

    with pytest.raises(boa_assets.UnsafeBoaArchiveError, match="Unsafe path"):
        boa_assets.sync_boa_model(
            tmp_path,
            opener=lambda request: io.BytesIO(payload),
        )

    assert sentinel.read_bytes() == b"keep"
    assert not (tmp_path.parent / "escaped").exists()


def test_model_manager_routes_boa_sync_and_verification(tmp_path, monkeypatch):
    ready = boa_assets.BoaAssetReport(
        model_directory=tmp_path / "boa",
        ready=True,
        checked_files={"checkpoint": "a" * 64},
        errors=(),
        install_manifest_verified=True,
    )
    calls: list[Path] = []
    monkeypatch.setattr(boa_assets, "check_boa_model", lambda root: ready)
    monkeypatch.setattr(
        boa_assets,
        "sync_boa_model",
        lambda root: calls.append(Path(root)) or ready,
    )

    assert verify_model(BOA_BACKEND_ID, tmp_path).ready
    assert sync_model(BOA_BACKEND_ID, tmp_path).ready
    assert calls == [tmp_path]


def test_boa_postprocessing_matches_the_upstream_retained_component_policy():
    prediction = np.zeros((8, 12, 12), dtype=np.uint8)
    prediction[1:7, 2:10, 2:10] = 1
    prediction[2:6, 3:9, 3:9] = 3
    prediction[3:5, 4:8, 4:8] = 4
    prediction[3:5, 5:7, 5:7] = 7
    prediction[0, 0, 0] = 3
    prediction[7, 11, 11] = 7

    result = postprocess_boa_body_regions(prediction)

    assert result[0, 0, 0] == 0
    assert result[7, 11, 11] == 0
    assert result[2, 3, 3] == 3
    assert result[3, 4, 4] == 4
    assert result[3, 5, 5] == 7
    with pytest.raises(ValueError, match="unknown body-region labels"):
        postprocess_boa_body_regions(np.full((1, 1, 1), 12, dtype=np.uint8))


def _signed_axis_directions():
    for order in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            direction = np.zeros((3, 3), dtype=float)
            for image_axis, physical_axis in enumerate(order):
                direction[physical_axis, image_axis] = signs[image_axis]
            yield direction


def test_boa_ras_model_copy_roundtrips_all_signed_axis_orientations():
    source_zyx = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6)
    for direction in _signed_axis_directions():
        superior_image_axis = int(np.flatnonzero(np.abs(direction[2]) == 1.0)[0])
        spacing = [0.8, 1.2, 1.6]
        spacing[superior_image_axis] = 5.0
        image = sitk.GetImageFromArray(source_zyx)
        image.SetSpacing(tuple(spacing))
        image.SetOrigin((12.0, -34.0, 56.0))
        image.SetDirection(tuple(direction.ravel()))

        model_zyx, context = prepare_boa_inference(image)
        restored_zyx = restore_boa_prediction(model_zyx.astype(np.uint8), context)

        assert not context.thickness_resampled
        assert np.array_equal(restored_zyx, source_zyx)


@pytest.mark.parametrize("slice_thickness_mm", [0.625, 1.0, 1.25, 2.5, 5.0, 7.5])
def test_boa_thickness_staging_matches_the_upstream_nibabel_oracle(
    tmp_path,
    slice_thickness_mm,
):
    source_zyx = np.arange(17 * 9 * 11, dtype=np.int16).reshape(17, 9, 11) % 700 - 300
    image = sitk.GetImageFromArray(source_zyx)
    image.SetSpacing((0.7, 0.9, slice_thickness_mm))
    image.SetOrigin((12.0, -34.0, 56.0))
    container = NiftiDataContainer(tmp_path / "source.nii.gz")
    container.img = image

    canonical = nib.as_closest_canonical(container.imgNifti1)
    spacing_xyz = np.asarray(canonical.header.get_zooms()[:3], dtype=float)
    zoom_xyz = spacing_xyz / np.asarray((spacing_xyz[0], spacing_xyz[1], 5.0))
    if np.array_equal(spacing_xyz, np.asarray((spacing_xyz[0], spacing_xyz[1], 5.0))):
        expected_xyz = canonical.get_fdata().astype(np.int32)
    else:
        expected_xyz = ndimage.zoom(
            canonical.get_fdata(),
            zoom=zoom_xyz,
            order=3,
            mode="nearest",
        ).astype(np.int32)

    model_zyx, context = prepare_boa_inference(image)

    assert np.array_equal(model_zyx, expected_xyz.transpose(2, 1, 0))
    assert context.model_spacing_zyx == pytest.approx((5.0, 0.9, 0.7))
    assert context.thickness_resampled is (slice_thickness_mm != 5.0)


def test_boa_backend_preserves_native_compartments_and_fixed_tissue_labels(
    tmp_path,
    container_factory,
    monkeypatch,
):
    config = PipelineConfig.model_validate(
        {
            "tissue": {"backend": BOA_BACKEND_ID},
            "runtime": {"device": "cpu", "allow_dirty": True},
        }
    )
    runtime = config.to_runtime_dict()
    assert runtime["LBL_TISSUE_COMPARTMENTS"] == BOA_NATIVE_COMPARTMENT_LABELS
    pipeline = SimpleNamespace(
        config=runtime,
        public_config=config,
        timestamp=1,
        device="cpu",
    )
    image = np.asarray([[[-100, 40, -80, -70, 500]]], dtype=np.int16)
    compartments = np.asarray([[[1, 2, 3, 4, 5]]], dtype=np.uint8)
    source = container_factory(tmp_path / "ct.nii.gz", image)
    compartment_container = container_factory(
        tmp_path / "compartments.nii.gz",
        compartments,
    )
    memory = {
        "id": "case",
        "workspace": tmp_path,
        "tmp/index": source,
        "masks/tissue_compartments.nii.gz": compartment_container,
    }
    MasksInternalTissue(pipeline, image="tmp/index")(memory)

    tissues = memory["masks/tissue_labels.nii.gz"].data
    assert tissues.tolist() == [[[3, 1, 4, 5, 0]]]

    action = SegmBoaBodyRegions(pipeline, image="tmp/index")
    monkeypatch.setattr(action, "_get_predictor", lambda: object())
    monkeypatch.setattr(action, "_predict", lambda *args: compartments)
    monkeypatch.setattr("BodyComposition.actions.segm_int.log_gpu_usage", lambda device: None)
    action(memory)
    assert np.array_equal(memory["masks/tissue_compartments.nii.gz"].data, compartments)
    assert action.model_folds == [0, 1, 2, 3, 4]


def test_boa_hu_distributions_do_not_relabel_body_regions_as_adipose_compartments():
    image = np.asarray([[[-100, 40, -80, -70]]], dtype=np.int16)
    compartments = np.asarray([[[1, 2, 3, 4]]], dtype=np.uint8)
    distributions = build_compartment_hu_distributions(
        image_zyx=image,
        compartment_labels_zyx=compartments,
        compartment_label_schema=BOA_NATIVE_COMPARTMENT_LABELS,
        geometry=ImageGeometry(
            size_xyz=(4, 1, 1),
            spacing_xyz=(1.0, 1.0, 1.0),
            origin_lps_xyz=(0.0, 0.0, 0.0),
            direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        ),
        identity=MeasurementIdentity(
            run_id="run",
            analysis_id="analysis",
            case_id="case",
        ),
    )

    source_ids = (
        distributions.groupby("compartment_key", sort=False)["source_label_ids"].first().to_dict()
    )
    assert source_ids == {"sm": "2", "sat": "1", "avat": "", "tvat": ""}
    missing = distributions.loc[distributions["compartment_key"].isin(("avat", "tvat"))]
    assert missing["distribution_valid"].eq(False).all()
    assert missing["distribution_reason"].eq("missing_compartment").all()


def test_default_and_low_resource_model_choices_remain_fixed():
    standard = PipelineConfig.model_validate({})
    low_resource = low_resource_config()

    assert standard.vertebral_backend == "spineps_veridah_ct_v1"
    assert standard.tissue_backend == "bodycomposition_resenc_l_v1"
    assert low_resource.vertebral_backend == "vertebral_bodies_resenc_m"
    assert low_resource.tissue_backend == "bodycomposition_resenc_m_v1"
    assert "spineps_veridah_ct_v1" not in required_model_ids(low_resource)


def test_boa_is_selectable_from_cli_and_builds_the_boa_action():
    args = cli.build_parser().parse_args(
        ["analyze", "CT.nii.gz", "--tissue-backend", "boa"]
    )
    config = cli._config(args)
    pipeline = SimpleNamespace(
        config=config.to_runtime_dict(),
        public_config=config,
        timestamp=1,
        device="cpu",
    )

    actions = canonical_actions(pipeline)

    assert config.tissue_backend == BOA_BACKEND_ID
    assert any(isinstance(action, SegmBoaBodyRegions) for action in actions)
    assert BOA_BACKEND_ID in required_model_ids(config)


def test_boa_is_selectable_from_the_simple_python_api(monkeypatch):
    expected = object()
    observed = {}

    class ServiceStub:
        def __init__(self, config):
            observed["config"] = config

        def analyze_case(self, *args, **kwargs):
            return expected

    monkeypatch.setattr(service_module, "PipelineService", ServiceStub)

    result = service_module.analyze_case(
        "CT.nii.gz",
        tissue_backend="boa",
    )

    assert result is expected
    assert observed["config"].tissue_backend == BOA_BACKEND_ID


def test_low_resource_rejects_a_conflicting_tissue_backend_shortcut():
    args = cli.build_parser().parse_args(
        [
            "analyze",
            "CT.nii.gz",
            "--low-resource",
            "--tissue-backend",
            "boa",
        ]
    )
    with pytest.raises(ValueError, match="fixed ResEncM tissue backend"):
        cli._config(args)
