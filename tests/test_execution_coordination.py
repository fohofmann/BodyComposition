from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
from pypdf import PdfReader

import BodyComposition.service as service_module
from BodyComposition.config import PipelineConfig
from BodyComposition.execution import (
    ClaimLostError,
    ExecutionPolicy,
    RestartAfterOutOfMemory,
    RestartAfterWorkerTimeout,
    SharedExecutionState,
    hardware_profile_id,
)
from BodyComposition.model_manager import ModelStatus
from BodyComposition.pipeline import InternalPipeline, _resolve_device_name
from BodyComposition.reporting.contracts import ReportingSettings
from BodyComposition.reporting.service import (
    render_failed_case_report,
    validate_report_manifest,
)
from BodyComposition.results import (
    BatchResult,
    BatchWorkerResult,
    ExecutionStatus,
)
from BodyComposition.service import CaseInput, PipelineService

SOURCE_CLEAN = {
    "package_version": "test",
    "git_commit": "a" * 40,
    "source_dirty": False,
    "source_tree_sha256": "b" * 64,
    "container_image_digest": None,
    "environment_path_variables": [],
}

A100_40GB = {
    "schema_version": 1,
    "backend": "cuda",
    "accelerator_name": "NVIDIA A100-SXM4-40GB",
    "compute_capability": [8, 0],
    "total_memory_bytes": 40 * 1024**3,
    "multiprocessor_count": 108,
}
B200 = {
    "schema_version": 1,
    "backend": "cuda",
    "accelerator_name": "NVIDIA B200",
    "compute_capability": [10, 0],
    "total_memory_bytes": 180 * 1024**3,
    "multiprocessor_count": 148,
}


def _policy(*, max_attempts: int = 3) -> ExecutionPolicy:
    return ExecutionPolicy(
        heartbeat_seconds=0.1,
        stale_after_seconds=1.0,
        max_claim_seconds=10.0,
        poll_seconds=0.01,
        max_attempts=max_attempts,
    )


def _expire(lease) -> None:
    lease.stop_heartbeat()
    for name in ("owner.json", "heartbeat.json"):
        path = lease.claim_directory / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["created_at"] = 0.0
        value["updated_at"] = 0.0
        path.write_text(json.dumps(value), encoding="utf-8")


def _write_ct(path: Path, value: int = 0) -> Path:
    image = sitk.GetImageFromArray(np.full((3, 4, 5), value, dtype=np.int16))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


def _model_status(root: Path) -> ModelStatus:
    return ModelStatus(
        model_id="test-model",
        ready=True,
        model_root=root,
        errors=(),
        checked_files={"weights": "c" * 64},
        asset={"asset_id": "test-model", "license": "test-only"},
    )


def _service(
    root: Path,
    pipeline_factory,
    *,
    hardware_profile=None,
    runtime=None,
) -> PipelineService:
    config = PipelineConfig.model_validate(
        {
            "measurements": {"landmarks": {"enabled": False}},
            "runtime": {"allow_dirty": True, **(runtime or {})},
        }
    )
    options = {}
    if hardware_profile is not None:
        options["hardware_provider"] = lambda device: dict(hardware_profile)
    return PipelineService(
        config,
        pipeline_factory=pipeline_factory,
        model_provider=lambda value: (_model_status(root),),
        source_provider=lambda: SOURCE_CLEAN,
        **options,
    )


def test_case_claim_is_atomic_and_stale_owner_is_fenced(tmp_path):
    first = SharedExecutionState(tmp_path, "run-1", policy=_policy())
    second = SharedExecutionState(tmp_path, "run-1", policy=_policy())

    original = first.try_claim("case-1")
    assert original is not None
    assert second.try_claim("case-1") is None

    _expire(original)
    replacement = second.try_claim("case-1")
    assert replacement is not None
    assert not original.owns_claim()
    with pytest.raises(ClaimLostError):
        original.assert_owned()
    replacement.release()


def test_run_clock_is_initialized_once_for_every_compatible_worker(tmp_path):
    first = SharedExecutionState(tmp_path, "run-1", policy=_policy())
    second = SharedExecutionState(tmp_path, "run-1", policy=_policy())

    started_at = first.initialize_run_clock(
        started_at="2026-07-29T08:00:00+00:00",
        plan_sha256="a" * 64,
    )

    assert started_at == "2026-07-29T08:00:00+00:00"
    assert (
        second.initialize_run_clock(
            started_at="2026-07-29T08:01:00+00:00",
            plan_sha256="a" * 64,
        )
        == started_at
    )
    runtime = json.loads(
        (
            tmp_path
            / ".bodycomposition/execution/run-1/runtime/run.json"
        ).read_text(encoding="utf-8")
    )
    assert runtime["started_at"] == started_at
    assert runtime["plan_sha256"] == "a" * 64

    with pytest.raises(RuntimeError, match="immutable plan"):
        second.initialize_run_clock(
            started_at="2026-07-29T08:02:00+00:00",
            plan_sha256="b" * 64,
        )


def test_repeated_stale_claims_become_terminal_instead_of_looping(tmp_path):
    policy = _policy(max_attempts=2)
    first = SharedExecutionState(tmp_path, "run-1", policy=policy)
    second = SharedExecutionState(tmp_path, "run-1", policy=policy)

    original = first.try_claim("case-1")
    assert original is not None
    _expire(original)
    replacement = second.try_claim("case-1")
    assert replacement is not None
    _expire(replacement)

    assert first.try_claim("case-1") is None
    terminal = first.terminal_failure_record("case-1")
    assert terminal is not None
    assert terminal["attempts"] == 2
    assert terminal["failure_code"] == "stale_worker_limit_exceeded"


def test_claim_uses_owner_timeout_when_workers_have_different_policies(tmp_path):
    long_policy = ExecutionPolicy(
        heartbeat_seconds=0.1,
        stale_after_seconds=2.0,
        max_claim_seconds=100.0,
        poll_seconds=0.01,
        max_attempts=3,
    )
    short_policy = ExecutionPolicy(
        heartbeat_seconds=0.1,
        stale_after_seconds=1.0,
        max_claim_seconds=10.0,
        poll_seconds=0.01,
        max_attempts=3,
    )
    owner = SharedExecutionState(tmp_path, "run-1", policy=long_policy)
    observer = SharedExecutionState(tmp_path, "run-1", policy=short_policy)
    lease = owner.try_claim("case-1")
    assert lease is not None
    lease.stop_heartbeat()

    owner_path = lease.claim_directory / "owner.json"
    owner_record = json.loads(owner_path.read_text(encoding="utf-8"))
    owner_record["created_at"] -= 20.0
    owner_path.write_text(json.dumps(owner_record), encoding="utf-8")
    heartbeat_path = lease.claim_directory / "heartbeat.json"
    heartbeat_record = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    heartbeat_record["updated_at"] = owner_record["created_at"] + 19.5
    heartbeat_path.write_text(json.dumps(heartbeat_record), encoding="utf-8")

    assert observer.try_claim("case-1") is None
    lease.release()


class SyntheticOutOfMemoryError(RuntimeError):
    pass


class OOMThenLowMemoryPipeline:
    instances: list[OOMThenLowMemoryPipeline] = []

    def __init__(self, config, timestamp):
        self.low_memory = False
        self.device = "cuda"
        self.released = False
        type(self).instances.append(self)

    def enable_low_memory_mode(self):
        self.low_memory = True

    def release_models(self):
        self.released = True

    def __call__(self, memory):
        memory["tmp/current_action"] = "SyntheticSegmentation"
        if not self.low_memory:
            raise SyntheticOutOfMemoryError("CUDA out of memory")
        artifact = Path(memory["workspace"]) / "low-memory-success.txt"
        artifact.write_text("ok", encoding="utf-8")


def test_oom_is_persisted_and_next_invocation_starts_low_memory(tmp_path):
    OOMThenLowMemoryPipeline.instances = []
    source = _write_ct(tmp_path / "case.nii.gz")
    output = tmp_path / "output"

    with pytest.raises(RestartAfterOutOfMemory):
        _service(
            tmp_path,
            OOMThenLowMemoryPipeline,
            hardware_profile=A100_40GB,
        ).analyze_case(
            source,
            output,
            case_id="case-1",
            run_id="oom-run",
        )

    marker = (
        output
        / ".bodycomposition/execution/oom-run/runtime/low_memory"
        / f"{hardware_profile_id(A100_40GB)}.json"
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["strategy_on_restart"] == "single_segmentation_bundle"
    assert payload["hardware_profile"] == A100_40GB
    assert OOMThenLowMemoryPipeline.instances[0].released

    result = _service(
        tmp_path,
        OOMThenLowMemoryPipeline,
        hardware_profile=A100_40GB,
    ).analyze_case(
        source,
        output,
        case_id="case-1",
        run_id="oom-run",
    )
    assert result.execution_status == ExecutionStatus.SUCCEEDED
    assert OOMThenLowMemoryPipeline.instances[-1].low_memory
    assert result.provenance["execution"]["strategy"] == "single_segmentation_bundle"


class StrategyProbePipeline:
    instances: list[StrategyProbePipeline] = []

    def __init__(self, config, timestamp):
        self.low_memory = False
        self.device = "cuda"
        type(self).instances.append(self)

    def enable_low_memory_mode(self):
        self.low_memory = True

    def __call__(self, memory):
        artifact = Path(memory["workspace"]) / "strategy.txt"
        artifact.write_text(
            "low" if self.low_memory else "normal",
            encoding="utf-8",
        )


def test_oom_strategy_is_scoped_to_equivalent_hardware(tmp_path):
    OOMThenLowMemoryPipeline.instances = []
    StrategyProbePipeline.instances = []
    source = _write_ct(tmp_path / "case.nii.gz")
    output = tmp_path / "output"

    with pytest.raises(RestartAfterOutOfMemory):
        _service(
            tmp_path,
            OOMThenLowMemoryPipeline,
            hardware_profile=A100_40GB,
        ).analyze_case(
            source,
            output,
            case_id="case-1",
            run_id="mixed-hardware-run",
        )

    result = _service(
        tmp_path,
        StrategyProbePipeline,
        hardware_profile=B200,
    ).analyze_case(
        source,
        output,
        case_id="case-1",
        run_id="mixed-hardware-run",
    )

    assert result.execution_status == ExecutionStatus.SUCCEEDED
    assert not StrategyProbePipeline.instances[-1].low_memory
    assert result.provenance["execution"]["strategy"] == "persistent_models"
    assert result.provenance["execution"]["hardware_profile_id"] == hardware_profile_id(
        B200
    )


class ParallelPipeline:
    barrier = threading.Barrier(2)
    calls: dict[str, int] = {}
    lock = threading.Lock()

    def __init__(self, config, timestamp):
        self.device = "cpu"

    def __call__(self, memory):
        case_id = str(memory["id"])
        with self.lock:
            self.calls[case_id] = self.calls.get(case_id, 0) + 1
        self.barrier.wait(timeout=5)
        (Path(memory["workspace"]) / "done.txt").write_text("ok", encoding="utf-8")


class CountingPipeline:
    calls = 0

    def __init__(self, config, timestamp):
        self.device = "cpu"

    def __call__(self, memory):
        type(self).calls += 1
        (Path(memory["workspace"]) / "done.txt").write_text("ok", encoding="utf-8")


class TimeoutPipeline:
    def __init__(self, config, timestamp):
        self.device = "cpu"

    def __call__(self, memory):
        memory["tmp/current_action"] = "SyntheticSlowStage"
        raise TimeoutError("synthetic worker timeout")


class MultiprocessReportingPipeline:
    """Create a valid case page while forcing two OS processes to overlap."""

    def __init__(self, config, timestamp):
        self.device = "cpu"
        self.settings = ReportingSettings.from_mapping(
            config.normalized()["reporting"]
        )

    def __call__(self, memory):
        lease = memory["tmp/case_lease"]
        rendezvous = (
            lease.state.root
            / "runtime"
            / "multiprocess-reporting-test"
        )
        rendezvous.mkdir(parents=True, exist_ok=True)
        (rendezvous / lease.state.worker_id).write_text("ready", encoding="utf-8")
        deadline = time.monotonic() + 20.0
        while len(tuple(rendezvous.iterdir())) < 2:
            if time.monotonic() >= deadline:
                raise TimeoutError("second reporting worker did not reach rendezvous")
            time.sleep(0.01)
        render_failed_case_report(
            case_id=str(memory["id"]),
            analysis_id=str(memory["analysis_id"]),
            output_directory=Path(memory["workspace"]) / "reports",
            settings=self.settings,
            failure_stage="multiprocess_regression_fixture",
            failure_code="synthetic_status_page",
        )
        (Path(memory["workspace"]) / "done.txt").write_text(
            lease.state.worker_id,
            encoding="utf-8",
        )


def _run_multiprocess_reporting_worker(
    cases,
    output,
    run_id,
    result_queue,
):
    os.environ["BODYCOMPOSITION_POLL_SECONDS"] = "0.01"
    try:
        config = PipelineConfig.model_validate(
            {
                "measurements": {"landmarks": {"enabled": False}},
                "runtime": {
                    "allow_dirty": True,
                    "device": "cpu",
                    "cpu_threads": 1,
                },
                "reporting": {
                    "enabled": True,
                    "combined_pdf": True,
                    "layout": "spine_overview_v1",
                },
            }
        )
        service = PipelineService(
            config,
            pipeline_factory=MultiprocessReportingPipeline,
            model_provider=lambda value: (_model_status(Path(output)),),
            source_provider=lambda: SOURCE_CLEAN,
        )
        result = service.analyze_batch(cases, output, run_id=run_id)
        run_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        result_queue.put(
            {
                "ok": True,
                "run_manifest": str(result.manifest_path),
                "started_at": run_manifest["started_at"],
                "ended_at": run_manifest["ended_at"],
                "reporting": result.reporting,
            }
        )
    except BaseException as error:
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )


def test_default_run_identity_requires_no_worker_specific_arguments(tmp_path):
    CountingPipeline.calls = 0
    source = _write_ct(tmp_path / "case.nii.gz")
    output = tmp_path / "output"

    first = _service(tmp_path, CountingPipeline).analyze_case(
        source,
        output,
        case_id="case-1",
    )
    second = _service(tmp_path, CountingPipeline).analyze_case(
        source,
        output,
        case_id="case-1",
    )

    assert first.run_id == second.run_id
    assert first.run_id.startswith("run-")
    assert CountingPipeline.calls == 1
    assert second.execution_status == ExecutionStatus.SKIPPED_IDENTICAL


def test_timeout_can_resume_with_a_different_worker_configuration(tmp_path):
    CountingPipeline.calls = 0
    source = _write_ct(tmp_path / "case.nii.gz")
    output = tmp_path / "output"

    with pytest.raises(RestartAfterWorkerTimeout):
        _service(
            tmp_path,
            TimeoutPipeline,
            runtime={"timeout_seconds": 60, "cpu_threads": 1},
        ).analyze_case(source, output, case_id="case-1")

    result = _service(
        tmp_path,
        CountingPipeline,
        runtime={"timeout_seconds": 3600, "cpu_threads": 8},
    ).analyze_case(source, output, case_id="case-1")

    assert result.execution_status == ExecutionStatus.SUCCEEDED
    assert CountingPipeline.calls == 1
    assert len(list((output / ".bodycomposition/execution").iterdir())) == 1
    assert len(list((output / "runs").iterdir())) == 1
    assert result.provenance["configuration"]["runtime"]["timeout_seconds"] == 3600


def test_identical_workers_split_whole_cases_and_exit_when_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("BODYCOMPOSITION_POLL_SECONDS", "0.01")
    ParallelPipeline.barrier = threading.Barrier(2)
    ParallelPipeline.calls = {}
    summary_calls = 0
    summary_lock = threading.Lock()
    original_image_summary = service_module.image_summary

    def counted_image_summary(*args, **kwargs):
        nonlocal summary_calls
        with summary_lock:
            summary_calls += 1
        return original_image_summary(*args, **kwargs)

    monkeypatch.setattr(service_module, "image_summary", counted_image_summary)
    cases = (
        CaseInput(_write_ct(tmp_path / "one.nii.gz", 1), "one"),
        CaseInput(_write_ct(tmp_path / "two.nii.gz", 2), "two"),
    )
    output = tmp_path / "output"

    def run_worker():
        return _service(tmp_path, ParallelPipeline).analyze_batch(
            cases,
            output,
            run_id="shared-run",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run_worker(), range(2)))

    assert ParallelPipeline.calls == {"one": 1, "two": 1}
    assert summary_calls == len(cases)
    assert all(len(result.cases) == 2 for result in results)
    assert all(result.execution_status == ExecutionStatus.SUCCEEDED for result in results)
    preflight_files = tuple(
        (output / ".bodycomposition/preflight").glob("*.json")
    )
    assert len(preflight_files) == 1
    assert str(tmp_path) not in preflight_files[0].read_text(encoding="utf-8")


def test_scheduler_worker_exits_when_only_live_claims_remain(tmp_path):
    CountingPipeline.calls = 0
    cases = (
        CaseInput(_write_ct(tmp_path / "one.nii.gz", 1), "one"),
        CaseInput(_write_ct(tmp_path / "two.nii.gz", 2), "two"),
    )
    output = tmp_path / "output"
    run_id = "worker-tail-run"
    state = SharedExecutionState(output, run_id, policy=_policy())
    external_lease = state.try_claim("one")
    assert external_lease is not None

    with external_lease:
        worker = _service(tmp_path, CountingPipeline).analyze_batch(
            cases,
            output,
            run_id=run_id,
            worker_mode=True,
        )

        assert isinstance(worker, BatchWorkerResult)
        assert worker.execution_status == ExecutionStatus.RUNNING
        assert worker.active_case_count == 1
        assert worker.remaining_case_count == 1
        assert [case.case_id for case in worker.cases] == ["two"]
        assert worker.manifest_path is None
        assert not (worker.output_path / "run_manifest.json").exists()

    completed = _service(tmp_path, CountingPipeline).analyze_batch(
        cases,
        output,
        run_id=run_id,
        worker_mode=True,
    )

    assert isinstance(completed, BatchResult)
    assert completed.execution_status == ExecutionStatus.SUCCEEDED
    assert [case.case_id for case in completed.cases] == ["one", "two"]
    assert completed.manifest_path.is_file()
    assert CountingPipeline.calls == 2



def test_scheduler_sigterm_drain_finishes_current_case_without_claiming_another(
    tmp_path,
):
    requested = threading.Event()

    class DrainAfterCurrentCasePipeline:
        calls: list[str] = []

        def __init__(self, config, timestamp):
            self.device = "cpu"

        def __call__(self, memory):
            case_id = str(memory["id"])
            type(self).calls.append(case_id)
            (Path(memory["workspace"]) / "done.txt").write_text(
                "ok",
                encoding="utf-8",
            )
            requested.set()

    cases = (
        CaseInput(_write_ct(tmp_path / "one.nii.gz", 1), "one"),
        CaseInput(_write_ct(tmp_path / "two.nii.gz", 2), "two"),
    )
    output = tmp_path / "output"
    run_id = "sigterm-drain-run"

    drained = _service(tmp_path, DrainAfterCurrentCasePipeline).analyze_batch(
        cases,
        output,
        run_id=run_id,
        worker_mode=True,
        drain_requested=requested.is_set,
        drain_reason="sigterm",
    )

    assert isinstance(drained, BatchWorkerResult)
    assert [case.case_id for case in drained.cases] == ["one"]
    assert DrainAfterCurrentCasePipeline.calls == ["one"]
    assert drained.drain_reason == "sigterm"
    assert drained.active_case_count == 0
    assert drained.remaining_case_count == 1
    assert drained.as_dict()["drain_reason"] == "sigterm"
    assert not (drained.output_path / "run_manifest.json").exists()

    completed = _service(tmp_path, DrainAfterCurrentCasePipeline).analyze_batch(
        cases,
        output,
        run_id=run_id,
        worker_mode=True,
    )

    assert isinstance(completed, BatchResult)
    assert completed.execution_status == ExecutionStatus.SUCCEEDED
    assert [case.case_id for case in completed.cases] == ["one", "two"]
    assert DrainAfterCurrentCasePipeline.calls == ["one", "two"]


def test_scheduler_sigterm_before_first_claim_starts_no_case(tmp_path):
    CountingPipeline.calls = 0
    requested = threading.Event()
    requested.set()
    cases = (
        CaseInput(_write_ct(tmp_path / "one.nii.gz", 1), "one"),
        CaseInput(_write_ct(tmp_path / "two.nii.gz", 2), "two"),
    )

    drained = _service(tmp_path, CountingPipeline).analyze_batch(
        cases,
        tmp_path / "output",
        run_id="sigterm-before-claim-run",
        worker_mode=True,
        drain_requested=requested.is_set,
        drain_reason="sigterm",
    )

    assert isinstance(drained, BatchWorkerResult)
    assert drained.cases == ()
    assert drained.active_case_count == 0
    assert drained.remaining_case_count == 2
    assert CountingPipeline.calls == 0

def test_shared_preflight_cache_invalidates_when_input_changes(tmp_path, monkeypatch):
    CountingPipeline.calls = 0
    source = _write_ct(tmp_path / "case.nii.gz", 1)
    output = tmp_path / "output"
    summary_calls = 0
    original_image_summary = service_module.image_summary

    def counted_image_summary(*args, **kwargs):
        nonlocal summary_calls
        summary_calls += 1
        return original_image_summary(*args, **kwargs)

    monkeypatch.setattr(service_module, "image_summary", counted_image_summary)
    first = _service(tmp_path, CountingPipeline).analyze_case(
        source,
        output,
        case_id="case",
    )
    previous_mtime = source.stat().st_mtime_ns
    _write_ct(source, 2)
    if source.stat().st_mtime_ns == previous_mtime:
        os.utime(source, ns=(previous_mtime + 1, previous_mtime + 1))
    second = _service(tmp_path, CountingPipeline).analyze_case(
        source,
        output,
        case_id="case",
    )

    assert first.run_id != second.run_id
    assert summary_calls == 2
    assert len(tuple((output / ".bodycomposition/preflight").glob("*.json"))) == 2


def test_process_workers_publish_one_complete_transactional_report(tmp_path):
    cases = tuple(
        CaseInput(
            _write_ct(tmp_path / f"case-{index}.nii.gz", index),
            f"case-{index}",
        )
        for index in range(4)
    )
    output = tmp_path / "output"
    run_id = "multiprocess-report-run"
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    workers = [
        context.Process(
            target=_run_multiprocess_reporting_worker,
            args=(cases, output, run_id, result_queue),
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)

    assert [worker.exitcode for worker in workers] == [0, 0]
    messages = [result_queue.get(timeout=5) for _ in workers]
    assert all(message["ok"] for message in messages), messages
    assert len({message["run_manifest"] for message in messages}) == 1
    assert len({message["started_at"] for message in messages}) == 1
    assert len({message["ended_at"] for message in messages}) == 1

    run_root = output / "runs" / run_id
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_clock = json.loads(
        (
            output
            / f".bodycomposition/execution/{run_id}/runtime/run.json"
        ).read_text(encoding="utf-8")
    )
    assert run_manifest["started_at"] == run_clock["started_at"]
    elapsed = (
        datetime.fromisoformat(run_manifest["ended_at"])
        - datetime.fromisoformat(run_manifest["started_at"])
    ).total_seconds()
    assert run_manifest["duration_seconds"] == pytest.approx(elapsed)
    assert [item["case_id"] for item in run_manifest["cases"]] == [
        item.case_id for item in cases
    ]

    worker_ids = {
        json.loads(
            (
                run_root
                / str(item["manifest"])
            ).read_text(encoding="utf-8")
        )["provenance"]["execution"]["worker_id"]
        for item in run_manifest["cases"]
    }
    assert len(worker_ids) == 2

    report_directory = run_root / f"aggregate/reports/{run_id}"
    pdf_path = report_directory / "case_reports.pdf"
    report_manifest_path = report_directory / "report_manifest.json"
    current = report_directory / ".current"
    assert current.is_symlink()
    assert pdf_path.is_symlink()
    assert report_manifest_path.is_symlink()
    report_manifest = validate_report_manifest(
        report_manifest_path,
        pdf_path=pdf_path,
    )
    assert report_manifest["publication"] == {
        "strategy": "atomic_snapshot_pointer_v1",
        "snapshot_id": report_manifest["report_id"],
        "pointer": ".current",
    }
    assert [item["case_id"] for item in report_manifest["cases"]] == [
        item.case_id for item in cases
    ]
    assert len(PdfReader(pdf_path, strict=True).pages) == len(cases) + 1
    snapshots = [
        path
        for path in (report_directory / ".snapshots").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    assert 1 <= len(snapshots) <= 2
    assert not tuple(report_directory.rglob("*.partial*"))


@pytest.mark.parametrize(
    ("cuda", "expected"),
    ((True, "cuda"), (False, "cpu")),
)
def test_auto_device_uses_only_the_released_backend(cuda, expected):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
    )
    assert _resolve_device_name(torch, "auto") == expected


def test_public_configuration_rejects_unvalidated_mps_inference():
    with pytest.raises(ValueError, match="runtime.device"):
        PipelineConfig.model_validate(
            {"runtime": {"device": "mps", "allow_dirty": True}}
        )


def test_low_memory_pipeline_releases_each_stage_model():
    class Action:
        def __init__(self):
            self.releases = 0
            self.low_memory = False

        def set_low_memory_mode(self, enabled):
            self.low_memory = enabled

        def release_model(self):
            self.releases += 1

        def __call__(self, memory):
            memory["ran"] = True

        def validate_outputs(self, memory):
            assert memory["ran"]

    action = Action()
    pipeline = object.__new__(InternalPipeline)
    pipeline.actions = [action]
    pipeline.device = "cpu"
    pipeline.low_memory_mode = False
    pipeline.public_config = SimpleNamespace(
        model_dump=lambda: {"runtime": {"timeout_seconds": 10}}
    )

    pipeline.enable_low_memory_mode()
    pipeline({"id": "case-1", "workspace": Path(".")})

    assert action.low_memory
    assert action.releases == 2  # activation plus release after the stage
