from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

import BodyComposition.pipeline as pipeline_module
from BodyComposition.config import PipelineConfig
from BodyComposition.pipeline import (
    ActionContractError,
    InternalPipeline,
    PipelineAction,
    _allocated_cpu_count,
    _resolve_device_name,
)
from BodyComposition.provenance import runtime_provenance


class _OutputAction(PipelineAction):
    def __init__(self, pipeline, *, fail: bool = False):
        super().__init__(pipeline)
        self.io_inputs = ["tmp/input"]
        self.io_outputs = ["tmp/output"]
        self.io_reset_outputs = ["result/{caseid}.json"]
        self.fail = fail
        self.released = 0
        self.low_memory = []

    def __call__(self, memory, task=None):
        super().__call__(memory, task)
        if self.fail:
            raise RuntimeError("synthetic stage failure")
        memory["tmp/output"] = memory["tmp/input"]

    def release_model(self):
        self.released += 1

    def set_low_memory_mode(self, enabled):
        self.low_memory.append(enabled)


def _runtime(actions, *, timeout=30):
    runtime = object.__new__(InternalPipeline)
    runtime.actions = list(actions)
    runtime.device = "cpu"
    runtime.low_memory_mode = False
    runtime.public_config = SimpleNamespace(
        model_dump=lambda: {"runtime": {"timeout_seconds": timeout}}
    )
    return runtime


def test_allocated_cpu_count_prefers_affinity_then_slurm_then_host(monkeypatch):
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda _pid: {0, 1, 2},
        raising=False,
    )
    assert _allocated_cpu_count() == 3

    def unavailable(_pid):
        raise OSError("not available")

    monkeypatch.setattr(os, "sched_getaffinity", unavailable, raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert _allocated_cpu_count() == 4

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "invalid")
    monkeypatch.setattr(pipeline_module.multiprocessing, "cpu_count", lambda: 6)
    assert _allocated_cpu_count() == 6

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "0")
    monkeypatch.setattr(pipeline_module.multiprocessing, "cpu_count", lambda: 0)
    assert _allocated_cpu_count() == 1


@pytest.mark.parametrize(
    ("requested", "cuda_available", "expected"),
    [
        ("cpu", False, "cpu"),
        ("cuda", True, "cuda"),
    ],
)
def test_device_resolution_is_explicit(requested, cuda_available, expected):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available))
    assert _resolve_device_name(torch, requested) == expected


def test_device_resolution_rejects_unavailable_or_unknown_backends():
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _resolve_device_name(torch, "cuda")
    with pytest.raises(ValueError, match="auto, cpu, or cuda"):
        _resolve_device_name(torch, "mps")


def test_explicit_cpu_does_not_probe_cuda():
    def fail_if_queried():
        raise AssertionError("explicit CPU execution must not initialize CUDA")

    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=fail_if_queried))
    assert _resolve_device_name(torch, "cpu") == "cpu"


def test_explicit_cpu_provenance_does_not_probe_cuda(monkeypatch):
    def fail_if_queried():
        raise AssertionError("explicit CPU provenance must not initialize CUDA")

    torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda="test"),
        cuda=SimpleNamespace(is_available=fail_if_queried),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    provenance = runtime_provenance(device="cpu")
    assert provenance["device_type"] == "cpu"
    assert provenance["gpu_name"] is None


def test_action_contract_accepts_memory_or_files_and_rejects_missing(
    pipeline_stub,
    tmp_path,
):
    action = _OutputAction(pipeline_stub)
    memory = {"id": "case-1", "workspace": tmp_path}
    with pytest.raises(ActionContractError, match="tmp/input"):
        action(memory)

    memory["tmp/input"] = 42
    action(memory)
    action.validate_outputs(memory)
    assert repr(action) == "_OutputAction"

    file_action = PipelineAction(pipeline_stub)
    file_action.io_inputs = ["inputs/{caseid}.nii.gz"]
    path = tmp_path / "inputs" / "case-1.nii.gz"
    path.parent.mkdir()
    path.write_bytes(b"ct")
    file_action.validate_inputs(memory)

    file_action.io_outputs = ["tmp/missing"]
    with pytest.raises(ActionContractError, match="tmp/missing"):
        file_action.validate_outputs(memory)
    assert not PipelineAction._is_available({}, "relative/file")


def test_internal_pipeline_runs_actions_and_records_stage_events(pipeline_stub):
    action = _OutputAction(pipeline_stub)
    runtime = _runtime([action])
    stages = []
    lease = SimpleNamespace(
        state=SimpleNamespace(stop_requested=lambda: False),
        set_stage=stages.append,
    )
    memory = {
        "id": "case-1",
        "tmp/input": "value",
        "tmp/return": {"ok": True},
        "tmp/case_lease": lease,
    }

    assert runtime(memory) == {"ok": True}
    assert memory["tmp/output"] == "value"
    assert stages == ["_OutputAction"]
    assert memory["tmp/stage_events"][0]["result_code"] == "succeeded"
    assert runtime.get_reset_outputs() == ["result/{caseid}.json"]


def test_internal_pipeline_records_failure_and_honors_stop(pipeline_stub):
    failing = _OutputAction(pipeline_stub, fail=True)
    runtime = _runtime([failing])
    memory = {"id": "case-1", "tmp/input": "value"}
    with pytest.raises(RuntimeError, match="synthetic stage failure"):
        runtime(memory)
    assert memory["tmp/stage_events"][0]["result_code"] == "RuntimeError"

    stopped = _runtime([_OutputAction(pipeline_stub)])
    stop_lease = SimpleNamespace(state=SimpleNamespace(stop_requested=lambda: True))
    with pytest.raises(KeyboardInterrupt):
        stopped({"id": "case-2", "tmp/input": 1, "tmp/case_lease": stop_lease})


def test_internal_pipeline_enforces_before_and_after_stage_timeouts(
    pipeline_stub,
    monkeypatch,
):
    before = _runtime([_OutputAction(pipeline_stub)], timeout=0)
    with pytest.raises(TimeoutError, match="before _OutputAction"):
        before({"id": "case-1", "tmp/input": 1})

    after = _runtime([_OutputAction(pipeline_stub)], timeout=10)
    times = iter([0.0, 0.0, 0.0, 0.1, 10.0])
    monkeypatch.setattr(pipeline_module, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="after _OutputAction"):
        after({"id": "case-2", "tmp/input": 1})


def test_low_memory_mode_releases_models_and_is_idempotent(
    pipeline_stub,
    monkeypatch,
):
    first = _OutputAction(pipeline_stub)
    second = _OutputAction(pipeline_stub)
    runtime = _runtime([first, second])
    cache_releases = []
    monkeypatch.setattr(
        pipeline_module,
        "release_accelerator_cache",
        lambda device: cache_releases.append(device),
    )

    runtime.enable_low_memory_mode()
    runtime.enable_low_memory_mode()
    assert first.low_memory == [True]
    assert second.low_memory == [True]
    assert (first.released, second.released) == (1, 1)
    assert cache_releases == ["cpu"]

    runtime.release_models()
    assert (first.released, second.released) == (2, 2)
    assert cache_releases == ["cpu", "cpu"]


def test_exclusive_model_stage_releases_only_other_model_bundles(
    pipeline_stub,
    monkeypatch,
):
    first = _OutputAction(pipeline_stub)
    support = _OutputAction(pipeline_stub)
    third = _OutputAction(pipeline_stub)
    runtime = _runtime([first, support, third])
    cache_releases = []
    monkeypatch.setattr(
        pipeline_module,
        "release_accelerator_cache",
        lambda device: cache_releases.append(device),
    )

    runtime.prepare_exclusive_model(support)

    assert (first.released, support.released, third.released) == (1, 0, 1)
    assert cache_releases == ["cpu"]


def test_model_release_failure_is_contained(pipeline_stub, caplog):
    action = _OutputAction(pipeline_stub)

    def fail():
        raise RuntimeError("release failed")

    action.release_model = fail
    InternalPipeline._release_action_model(action)
    assert "Could not release model state" in caplog.text


def test_internal_pipeline_initialization_uses_validated_runtime(
    monkeypatch,
):
    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            get_device_name=lambda: "unused",
        ),
        set_num_threads=lambda value: calls.append(("threads", value)),
        set_num_interop_threads=lambda value: calls.append(("interop", value)),
        use_deterministic_algorithms=lambda *args, **kwargs: calls.append(
            ("deterministic", args, kwargs)
        ),
        device=lambda name: SimpleNamespace(type=name),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from BodyComposition.actions import orientation as orientation_action
    from BodyComposition.pipelines import bodycomposition

    class DummyAction(PipelineAction):
        pass

    monkeypatch.setattr(orientation_action, "AssessOrientation", DummyAction)
    monkeypatch.setattr(
        bodycomposition,
        "canonical_actions",
        lambda runtime: [DummyAction(runtime)],
    )
    monkeypatch.setattr(
        pipeline_module,
        "_allocated_cpu_count",
        lambda: 2,
    )

    config = PipelineConfig.model_validate(
        {"runtime": {"device": "cpu", "cpu_threads": 0, "allow_dirty": True}}
    )
    runtime = InternalPipeline(config, timestamp=123)
    assert runtime.device.type == "cpu"
    assert runtime.cpu_threads == 2
    assert len(runtime.actions) == 2
    assert ("threads", 2) in calls


def test_internal_pipeline_requires_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    config = PipelineConfig.model_validate(
        {"runtime": {"device": "cpu", "allow_dirty": True}}
    )
    with pytest.raises(RuntimeError, match="PyTorch is required"):
        InternalPipeline(config, timestamp=123)
