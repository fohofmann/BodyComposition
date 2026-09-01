from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from BodyComposition.orientation.core import OrientationOutcome
from BodyComposition.utils.geometry import GeometryError, ImageGeometry
from BodyComposition.vertebral import ExecutionStatus, SpinepsRuntime, spineps_runtime


def _geometry(shape_zyx=(4, 5, 6)):
    return ImageGeometry(
        size_xyz=tuple(reversed(shape_zyx)),
        spacing_xyz=(0.8, 1.2, 2.5),
        origin_lps_xyz=(3.0, -2.0, 17.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def _image(array, geometry):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(geometry.spacing_xyz)
    image.SetOrigin(geometry.origin_lps_xyz)
    image.SetDirection(geometry.direction_lps)
    return image


def _write_image(path, array, geometry):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(_image(array, geometry), str(path))


def _prepared(path):
    return OrientationOutcome(
        result=SimpleNamespace(to_dict=lambda: {"state": "PASS_METADATA_MATCH"}),
        prepared_image=sitk.ReadImage(str(path)),
    )


def _rotated_geometry(reference, angle_radians):
    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    return ImageGeometry(
        size_xyz=reference.size_xyz,
        spacing_xyz=reference.spacing_xyz,
        origin_lps_xyz=reference.origin_lps_xyz,
        direction_lps=(
            cosine,
            -sine,
            0.0,
            sine,
            cosine,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )


class FakeSession:
    def __init__(self, use_cpu=False):
        self.use_cpu = use_cpu
        self.loads = 0
        self.provenance = {"spineps_version": "2.0.0"}

    def load(self):
        self.loads += 1
        return SimpleNamespace(semantic=object(), instance=object(), labeling=object())


def test_runtime_precomputes_vibeseg_and_adapts_sitk_outputs(tmp_path):
    geometry = _geometry()
    ct = np.zeros((4, 5, 6), dtype=np.int16)
    input_path = tmp_path / "sub-test_ct.nii.gz"
    _write_image(input_path, ct, geometry)
    paths = {
        "out_vibeseg": tmp_path / "derivatives/vibeseg.nii.gz",
        "out_spine": tmp_path / "derivatives/spine.nii.gz",
        "out_vert": tmp_path / "derivatives/vert.nii.gz",
    }
    img_ref = SimpleNamespace(format="ct", open_nii=lambda: "input-nii")
    calls = []

    def make_bids_file(path):
        assert path.name == "sub-case001_ct.nii.gz"
        return img_ref

    def run_vibeseg(model, input_nii, output, device):
        calls.append((model, input_nii, output, device))
        _write_image(output, np.zeros(ct.shape, dtype=np.uint8), geometry)

    def run_spineps(received_ref, models, derivative_name):
        assert received_ref is img_ref
        assert paths["out_vibeseg"].is_file()
        semantic = np.zeros(ct.shape, dtype=np.uint8)
        vertebra = np.zeros(ct.shape, dtype=np.uint8)
        semantic[1:3, 1:4, 1:5] = 49
        vertebra[1:3, 1:4, 1:5] = 22
        _write_image(paths["out_spine"], semantic, geometry)
        _write_image(paths["out_vert"], vertebra, geometry)
        return paths, SimpleNamespace(name="OK")

    session = FakeSession(use_cpu=True)
    runtime = SpinepsRuntime(
        session,
        Path("/models/vibeseg/trainer"),
        make_bids_file=make_bids_file,
        derive_output_paths=lambda ref, name: paths,
        run_vibeseg=run_vibeseg,
        run_spineps=run_spineps,
        vibeseg_bundle_provenance={
            "release": "v1.0.0",
            "fold_selection": ["0", "1", "2", "3", "4"],
            "trained_model_inventory": {"sha256": "inventory-digest"},
        },
    )

    result = runtime.run(
        _prepared(input_path),
        attempt_directory=tmp_path / "attempt",
        case_id="case-001",
    )

    assert result.execution_status == ExecutionStatus.SUCCEEDED
    assert result.whole_vertebra_labels.shape == ct.shape
    assert set(np.unique(result.vertebral_body_labels)) == {0, 22}
    assert session.loads == 1
    crop_provenance = result.provenance["required_crop_model"]
    assert crop_provenance["release"] == "v1.0.0"
    assert crop_provenance["fold_selection"] == ["0", "1", "2", "3", "4"]
    assert crop_provenance["trained_model_inventory"]["sha256"] == "inventory-digest"
    assert len(crop_provenance["output"]["sha256"]) == 64
    assert calls == [
        (
            Path("/models/vibeseg/trainer"),
            "input-nii",
            paths["out_vibeseg"],
            "cpu",
        )
    ]


def test_runtime_normalizes_bounded_upstream_header_rounding_without_resampling(
    tmp_path,
):
    geometry = _geometry((4, 512, 512))
    rounded_geometry = _rotated_geometry(geometry, 5e-4)
    ct = np.zeros((4, 512, 512), dtype=np.int16)
    input_path = tmp_path / "sub-test_ct.nii.gz"
    _write_image(input_path, ct, geometry)
    paths = {
        "out_vibeseg": tmp_path / "derivatives/vibeseg.nii.gz",
        "out_spine": tmp_path / "derivatives/spine.nii.gz",
        "out_vert": tmp_path / "derivatives/vert.nii.gz",
    }
    crop = np.zeros(ct.shape, dtype=np.uint8)
    crop[1:3, 64:448, 80:432] = 1

    def run_vibeseg(_model, _input_nii, output, _device):
        _write_image(output, crop, rounded_geometry)

    def run_spineps(_img_ref, _models, _derivative_name):
        persisted_crop = sitk.ReadImage(str(paths["out_vibeseg"]))
        assert persisted_crop.GetDirection() == pytest.approx(geometry.direction_lps)
        assert np.array_equal(sitk.GetArrayFromImage(persisted_crop), crop)
        semantic = np.zeros(ct.shape, dtype=np.uint8)
        vertebra = np.zeros(ct.shape, dtype=np.uint8)
        semantic[1:3, 128:384, 144:368] = 49
        vertebra[1:3, 128:384, 144:368] = 22
        return (
            _image(semantic, rounded_geometry),
            _image(vertebra, rounded_geometry),
            None,
            SimpleNamespace(name="OK"),
        )

    runtime = SpinepsRuntime(
        FakeSession(),
        tmp_path / "verified-dataset100",
        make_bids_file=lambda path: SimpleNamespace(
            format="ct",
            open_nii=lambda: "input-nii",
        ),
        derive_output_paths=lambda ref, name: paths,
        run_vibeseg=run_vibeseg,
        run_spineps=run_spineps,
    )

    result = runtime.run(
        _prepared(input_path),
        attempt_directory=tmp_path / "attempt",
        case_id="case-rounded",
    )

    assert result.execution_status == ExecutionStatus.SUCCEEDED
    crop_record = result.provenance["required_crop_model"]["geometry_normalization"]
    assert crop_record["id"] == "tptbox-nifti-header-rounding-v1"
    assert crop_record["pixel_data_modified"] is False
    assert crop_record["maximum_corner_displacement_mm"] <= crop_record[
        "maximum_allowed_corner_displacement_mm"
    ]
    output_records = result.provenance["output_geometry_normalizations"]
    assert [record["component"] for record in output_records] == [
        "SPINEPS semantic output",
        "SPINEPS vertebra output",
    ]
    assert set(np.unique(result.vertebral_body_labels)) == {0, 22}


def test_runtime_rejects_material_upstream_direction_change(tmp_path):
    geometry = _geometry((4, 32, 32))
    changed_geometry = _rotated_geometry(geometry, 0.05)
    ct = np.zeros((4, 32, 32), dtype=np.int16)
    input_path = tmp_path / "sub-test_ct.nii.gz"
    output_path = tmp_path / "derivatives/vibeseg.nii.gz"
    _write_image(input_path, ct, geometry)

    def run_vibeseg(_model, _input_nii, output, _device):
        _write_image(output, np.zeros(ct.shape, dtype=np.uint8), changed_geometry)

    runtime = SpinepsRuntime(
        FakeSession(),
        tmp_path / "verified-dataset100",
        make_bids_file=lambda path: SimpleNamespace(
            format="ct",
            open_nii=lambda: "input-nii",
        ),
        derive_output_paths=lambda ref, name: {"out_vibeseg": output_path},
        run_vibeseg=run_vibeseg,
        run_spineps=lambda *args: (_ for _ in ()).throw(
            AssertionError("SPINEPS must not run after a material geometry change")
        ),
    )

    with pytest.raises(GeometryError, match="Header-only normalization was rejected"):
        runtime.run(
            _prepared(input_path),
            attempt_directory=tmp_path / "attempt",
            case_id="case-material-change",
        )

    persisted = sitk.ReadImage(str(output_path))
    assert persisted.GetDirection() == pytest.approx(changed_geometry.direction_lps)


def test_runtime_returns_explicit_failure_for_upstream_error(tmp_path):
    geometry = _geometry()
    input_path = tmp_path / "sub-test_ct.nii.gz"
    _write_image(input_path, np.zeros((4, 5, 6), dtype=np.int16), geometry)
    vibeseg_path = tmp_path / "derivatives/vibeseg.nii.gz"
    _write_image(vibeseg_path, np.zeros((4, 5, 6), dtype=np.uint8), geometry)
    img_ref = SimpleNamespace(format="ct", open_nii=lambda: "not-used")
    paths = {"out_vibeseg": vibeseg_path}
    runtime = SpinepsRuntime(
        FakeSession(),
        Path("/models/vibeseg/trainer"),
        make_bids_file=lambda path: img_ref,
        derive_output_paths=lambda ref, name: paths,
        run_vibeseg=lambda *args: (_ for _ in ()).throw(AssertionError("must reuse crop")),
        run_spineps=lambda ref, models, name: (paths, SimpleNamespace(name="COMPATIBILITY")),
    )

    result = runtime.run(
        _prepared(input_path),
        attempt_directory=tmp_path / "attempt",
        case_id="case-002",
    )

    assert result.execution_status == ExecutionStatus.FAILED
    assert result.error_summary == "SPINEPS failed with error code COMPATIBILITY."


def test_upstream_wrappers_disable_downloads_and_use_saved_outputs(monkeypatch, tmp_path):
    import TPTBox.segmentation.nnUnet_utils.inference_api as inference_api

    vibeseg_call = {}
    spineps_call = {}

    def fake_vibeseg(*args, **kwargs):
        vibeseg_call.update({"args": args, "kwargs": kwargs})

    def fake_spineps(**kwargs):
        spineps_call.update(kwargs)
        return "response"

    monkeypatch.setattr(
        "TPTBox.segmentation.VibeSeg.inference_nnunet.run_inference_on_file",
        fake_vibeseg,
    )
    monkeypatch.setattr(inference_api, "_interop", False)
    monkeypatch.setattr("spineps.seg_run.process_img_nii", fake_spineps)
    trained_model = tmp_path / "verified-dataset100"
    output = tmp_path / "attempt" / "vibeseg.nii.gz"

    runtime_adapter = spineps_runtime._run_explicit_vibeseg(
        trained_model,
        "input-nii",
        output,
        "cuda",
    )
    models = SimpleNamespace(semantic="semantic", instance="instance", labeling="labeling")
    response = spineps_runtime._run_spineps("img-ref", models, "derivatives_spineps")

    assert vibeseg_call["args"] == (trained_model, ["input-nii"])
    assert vibeseg_call["kwargs"]["out_file"] == output
    assert vibeseg_call["kwargs"]["auto_download"] is False
    assert vibeseg_call["kwargs"]["cache_model"] is True
    assert vibeseg_call["kwargs"]["ddevice"] == "cuda"
    assert vibeseg_call["kwargs"]["max_folds"] is None
    assert inference_api._interop is True
    assert runtime_adapter is None
    assert response == "response"
    assert spineps_call["img_ref"] == "img-ref"
    assert spineps_call["model_semantic"] == "semantic"
    assert spineps_call["model_instance"] == "instance"
    assert spineps_call["model_labeling"] == "labeling"
    assert spineps_call["return_output_instead_of_save"] is False
    assert spineps_call["ignore_compatibility_issues"] is True
    assert spineps_call["save_raw"] is True
    assert spineps_call["save_softmax_logits"] is False
    assert spineps_call["save_debug_data"] is False


def test_upstream_vibeseg_cpu_guard_is_scoped_and_records_provenance(
    monkeypatch,
    tmp_path,
):
    import TPTBox.segmentation.nnUnet_utils.predictor as predictor_module

    original_util = predictor_module.get_gpu_util
    original_memory = predictor_module.get_gpu_memory_MB
    monkeypatch.setattr(
        spineps_runtime,
        "virtual_memory",
        lambda: SimpleNamespace(available=12 * 1024**2),
    )

    def fake_vibeseg(*args, **kwargs):
        assert predictor_module.get_gpu_util("cpu") == 0.0
        assert predictor_module.get_gpu_memory_MB("cpu") == 12.0

    monkeypatch.setattr(
        "TPTBox.segmentation.VibeSeg.inference_nnunet.run_inference_on_file",
        fake_vibeseg,
    )

    adapter = spineps_runtime._run_explicit_vibeseg(
        tmp_path / "verified-dataset100",
        "input-nii",
        tmp_path / "vibeseg.nii.gz",
        "cpu",
    )

    assert adapter is not None
    assert adapter["id"] == "tptbox-0.7.5-cpu-memory-telemetry"
    assert adapter["upstream_revision"] == "acaaf16f74fb0fe8fc555b23cf4e0230efc49753"
    assert predictor_module.get_gpu_util is original_util
    assert predictor_module.get_gpu_memory_MB is original_memory


def test_upstream_vibeseg_cpu_guard_restores_telemetry_after_failure(
    monkeypatch,
    tmp_path,
):
    import TPTBox.segmentation.nnUnet_utils.predictor as predictor_module

    original_util = predictor_module.get_gpu_util
    original_memory = predictor_module.get_gpu_memory_MB

    def fail_vibeseg(*args, **kwargs):
        assert predictor_module.get_gpu_util("cpu") == 0.0
        raise RuntimeError("upstream failure")

    monkeypatch.setattr(
        "TPTBox.segmentation.VibeSeg.inference_nnunet.run_inference_on_file",
        fail_vibeseg,
    )

    with pytest.raises(RuntimeError, match="upstream failure"):
        spineps_runtime._run_explicit_vibeseg(
            tmp_path / "verified-dataset100",
            "input-nii",
            tmp_path / "vibeseg.nii.gz",
            "cpu",
        )

    assert predictor_module.get_gpu_util is original_util
    assert predictor_module.get_gpu_memory_MB is original_memory


def test_spineps_runtime_serializes_cpu_adapter_across_instances(
    monkeypatch,
    tmp_path,
):
    geometry = _geometry()
    array = np.zeros((4, 5, 6), dtype=np.int16)
    input_path = tmp_path / "source.nii.gz"
    _write_image(input_path, array, geometry)
    cpu_paths = {"out_vibeseg": tmp_path / "cpu/vibeseg.nii.gz"}
    cuda_paths = {"out_vibeseg": tmp_path / "cuda/vibeseg.nii.gz"}
    cpu_patch_active = Event()
    release_cpu = Event()
    cuda_spineps_entered = Event()

    monkeypatch.setattr(
        spineps_runtime,
        "_configure_upstream_citation_reminder",
        lambda: None,
    )

    def successful_response():
        semantic = np.zeros(array.shape, dtype=np.uint8)
        vertebra = np.zeros(array.shape, dtype=np.uint8)
        semantic[1:3, 1:4, 1:5] = 49
        vertebra[1:3, 1:4, 1:5] = 22
        return (
            _image(semantic, geometry),
            _image(vertebra, geometry),
            None,
            SimpleNamespace(name="OK"),
        )

    def fake_upstream_vibeseg(*args, **kwargs):
        import TPTBox.segmentation.nnUnet_utils.predictor as predictor_module

        assert predictor_module.get_gpu_util("cpu") == 0.0
        _write_image(
            Path(kwargs["out_file"]),
            np.zeros(array.shape, dtype=np.uint8),
            geometry,
        )
        cpu_patch_active.set()
        assert release_cpu.wait(timeout=5)

    def cuda_vibeseg(model, input_nii, output, device):
        assert device == "cuda"
        _write_image(output, np.zeros(array.shape, dtype=np.uint8), geometry)

    def cuda_spineps(img_ref, models, derivative_name):
        cuda_spineps_entered.set()
        return successful_response()

    monkeypatch.setattr(
        "TPTBox.segmentation.VibeSeg.inference_nnunet.run_inference_on_file",
        fake_upstream_vibeseg,
    )
    cpu_runtime = SpinepsRuntime(
        FakeSession(use_cpu=True),
        tmp_path / "verified-dataset100",
        make_bids_file=lambda path: SimpleNamespace(
            format="ct",
            open_nii=lambda: "cpu-input",
        ),
        derive_output_paths=lambda ref, name: cpu_paths,
        run_spineps=lambda ref, models, name: successful_response(),
    )
    cuda_runtime = SpinepsRuntime(
        FakeSession(use_cpu=False),
        tmp_path / "verified-dataset100",
        make_bids_file=lambda path: SimpleNamespace(
            format="ct",
            open_nii=lambda: "cuda-input",
        ),
        derive_output_paths=lambda ref, name: cuda_paths,
        run_vibeseg=cuda_vibeseg,
        run_spineps=cuda_spineps,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        cpu_future = executor.submit(
            cpu_runtime.run,
            _prepared(input_path),
            attempt_directory=tmp_path / "cpu-attempt",
            case_id="cpu-case",
        )
        assert cpu_patch_active.wait(timeout=5)
        cuda_future = executor.submit(
            cuda_runtime.run,
            _prepared(input_path),
            attempt_directory=tmp_path / "cuda-attempt",
            case_id="cuda-case",
        )
        cuda_blocked = not cuda_spineps_entered.wait(timeout=0.25)
        release_cpu.set()

        assert cuda_blocked
        assert cpu_future.result(timeout=10).execution_status == ExecutionStatus.SUCCEEDED
        assert cuda_future.result(timeout=10).execution_status == ExecutionStatus.SUCCEEDED
        assert cuda_spineps_entered.is_set()


def test_upstream_crop_reuses_precomputed_vibeseg_without_downloading(
    monkeypatch,
    tmp_path,
):
    from spineps.phase_pre import compute_crop
    from TPTBox import to_nii
    from TPTBox.core.vert_constants import Full_Body_Instance_Vibe

    geometry = _geometry((8, 9, 10))
    input_path = tmp_path / "input.nii.gz"
    output_path = tmp_path / "precomputed-vibeseg.nii.gz"
    _write_image(input_path, np.zeros((8, 9, 10), dtype=np.int16), geometry)
    segmentation = np.zeros((8, 9, 10), dtype=np.uint8)
    segmentation[2:6, 2:7, 3:8] = Full_Body_Instance_Vibe.vertebra_body.value
    _write_image(output_path, segmentation, geometry)

    monkeypatch.setattr(
        "TPTBox.segmentation.VibeSeg.inference_nnunet.download_weights",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the upstream crop must reuse the precomputed output")
        ),
    )

    crop = compute_crop(
        to_nii(input_path, seg=False),
        output_path,
        dataset_id=100,
        ddevice="cpu",
    )

    assert len(crop) == 3
    assert all(isinstance(axis, slice) for axis in crop)
