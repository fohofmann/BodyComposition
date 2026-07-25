from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from BodyComposition.orientation.core import OrientationOutcome
from BodyComposition.utils.geometry import ImageGeometry
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
    monkeypatch.setattr("spineps.seg_run.process_img_nii", fake_spineps)
    trained_model = tmp_path / "verified-dataset100"
    output = tmp_path / "attempt" / "vibeseg.nii.gz"

    spineps_runtime._run_explicit_vibeseg(
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
