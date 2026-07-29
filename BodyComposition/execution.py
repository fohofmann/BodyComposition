"""Lightweight filesystem coordination for whole-case pipeline workers.

Every process runs the same complete pipeline.  Coordination is deliberately
limited to per-case claims and small JSON state files so the same command can
be used locally, in a container, or by every element of a Slurm array.
"""

from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

_FILE_GUARDS_LOCK = threading.Lock()
_FILE_GUARDS: dict[str, threading.Lock] = {}


@contextmanager
def shared_file_guard(path: str | Path):
    """Serialize a short operation across local threads and POSIX processes."""

    lock_path = Path(path)
    key = str(lock_path.resolve())
    with _FILE_GUARDS_LOCK:
        local_guard = _FILE_GUARDS.setdefault(key, threading.Lock())
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with local_guard, lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ClaimLostError(RuntimeError):
    """Raised when a stale worker no longer owns the case it processed."""


class RestartAfterOutOfMemory(RuntimeError):
    """The run must restart so the persisted low-memory strategy takes effect."""

    def __init__(
        self,
        case_id: str,
        original_error: BaseException,
        *,
        result: Any | None = None,
    ) -> None:
        super().__init__(
            f"Accelerator memory was exhausted while processing {case_id!r}. "
            "Low-memory execution has been recorded for the next invocation "
            "on equivalent hardware."
        )
        self.case_id = str(case_id)
        self.original_error = original_error
        self.result = result


class RestartAfterWorkerTimeout(RuntimeError):
    """The current worker timed out, but another compatible worker may retry."""

    def __init__(
        self,
        case_id: str,
        original_error: BaseException,
        *,
        result: Any | None = None,
    ) -> None:
        super().__init__(
            f"The worker time budget expired while processing {case_id!r}. "
            "The attempt was recorded and the case may be resumed by another "
            "compatible worker or by a restart with a larger timeout."
        )
        self.case_id = str(case_id)
        self.original_error = original_error
        self.result = result


@dataclass(frozen=True)
class ExecutionPolicy:
    """Small operational policy; environment overrides avoid a new CLI surface."""

    heartbeat_seconds: float = 30.0
    stale_after_seconds: float = 300.0
    max_claim_seconds: float = 1800.0
    poll_seconds: float = 2.0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive.")
        if self.stale_after_seconds <= self.heartbeat_seconds:
            raise ValueError("stale_after_seconds must exceed heartbeat_seconds.")
        if self.max_claim_seconds <= self.stale_after_seconds:
            raise ValueError("max_claim_seconds must exceed stale_after_seconds.")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive.")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")

    @classmethod
    def from_environment(
        cls,
        *,
        default_max_claim_seconds: float = 1800.0,
    ) -> ExecutionPolicy:
        heartbeat = float(os.environ.get("BODYCOMPOSITION_HEARTBEAT_SECONDS", "30"))
        stale = float(os.environ.get("BODYCOMPOSITION_STALE_AFTER_SECONDS", "300"))
        maximum = float(
            os.environ.get(
                "BODYCOMPOSITION_MAX_CLAIM_SECONDS",
                str(max(default_max_claim_seconds, stale + heartbeat)),
            )
        )
        return cls(
            heartbeat_seconds=heartbeat,
            stale_after_seconds=stale,
            max_claim_seconds=maximum,
            poll_seconds=float(os.environ.get("BODYCOMPOSITION_POLL_SECONDS", "2")),
            max_attempts=int(os.environ.get("BODYCOMPOSITION_MAX_ATTEMPTS", "3")),
        )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _case_key(case_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(case_id)).strip("-.")[:48]
    digest = hashlib.sha256(str(case_id).encode("utf-8")).hexdigest()[:12]
    return f"{readable or 'case'}-{digest}"


def is_out_of_memory_error(error: BaseException) -> bool:
    """Recognize accelerator allocation failures without importing Torch."""

    current: BaseException | None = error
    while current is not None:
        name = current.__class__.__name__.lower()
        message = str(current).lower()
        if "outofmemory" in name or "out of memory" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def execution_hardware_profile(device: object | None = None) -> dict[str, Any]:
    """Describe OOM-relevant hardware without binding state to a GPU UUID.

    CUDA profiles intentionally omit the visible ordinal and physical UUID.  A
    marker produced on one A100 can therefore protect another equivalent A100
    Slurm worker, while a B200 or a different-memory GPU receives a different
    profile.
    """

    requested = str(getattr(device, "type", device) or "auto")
    backend = requested.split(":", 1)[0].lower()
    try:
        import torch
    except ImportError:
        torch = None

    if backend == "auto":
        backend = (
            "cuda"
            if torch is not None and bool(torch.cuda.is_available())
            else "cpu"
        )

    profile: dict[str, Any] = {
        "schema_version": 1,
        "backend": backend,
    }
    if backend == "cuda":
        if torch is None or not bool(torch.cuda.is_available()):
            profile["accelerator_name"] = "unavailable"
            return profile
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        profile.update(
            {
                "accelerator_name": str(properties.name),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "total_memory_bytes": int(properties.total_memory),
                "multiprocessor_count": int(properties.multi_processor_count),
            }
        )
        return profile

    profile.update(
        {
            "machine": platform.machine() or "unknown",
            "processor": platform.processor() or "unknown",
            "total_memory_bytes": _host_memory_bytes(),
        }
    )
    return profile


def _host_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return page_size * page_count if page_size > 0 and page_count > 0 else None


def hardware_profile_id(profile: Mapping[str, Any]) -> str:
    """Return the stable identifier used for per-hardware runtime state."""

    encoded = json.dumps(
        dict(profile),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def release_accelerator_cache(device: object | None = None) -> None:
    """Release only the accelerator cache used by the current executor."""

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    device_name = str(device or "").split(":", 1)[0].lower()
    if device_name == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


class CaseLease:
    """Renewable ownership of one case claim."""

    def __init__(
        self,
        state: SharedExecutionState,
        case_id: str,
        claim_directory: Path,
        token: str,
    ) -> None:
        self.state = state
        self.case_id = str(case_id)
        self.claim_directory = claim_directory
        self.token = token
        self._stage = "claimed"
        self._stage_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"bodycomposition-heartbeat-{_case_key(self.case_id)}",
            daemon=True,
        )
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.state.policy.heartbeat_seconds):
            if not self.owns_claim():
                return
            try:
                self.state._write_heartbeat(self.claim_directory, self.token)
            except OSError as error:
                logging.warning(
                    "Could not renew the claim heartbeat for %s (%s); retrying.",
                    self.case_id,
                    type(error).__name__,
                )

    def owns_claim(self) -> bool:
        owner = _read_json(self.claim_directory / "owner.json")
        return bool(owner and owner.get("token") == self.token)

    def assert_owned(self) -> None:
        if not self.owns_claim():
            raise ClaimLostError(
                f"Worker lost the claim for case {self.case_id!r}; outputs were not committed."
            )

    def set_stage(self, stage: str) -> None:
        with self._stage_lock:
            self._stage = str(stage)
        try:
            self.state._write_heartbeat(
                self.claim_directory,
                self.token,
                stage=self._stage,
            )
        except OSError as error:
            logging.warning(
                "Could not persist the current stage for %s (%s).",
                self.case_id,
                type(error).__name__,
            )

    @property
    def stage(self) -> str:
        with self._stage_lock:
            return self._stage

    def stop_heartbeat(self) -> None:
        """Stop renewal without releasing ownership (primarily useful for recovery)."""

        self._stop.set()
        self._thread.join(timeout=max(1.0, self.state.policy.heartbeat_seconds * 2))

    def release(self) -> None:
        self.stop_heartbeat()
        if not self.owns_claim():
            return
        try:
            shutil.rmtree(self.claim_directory)
        except FileNotFoundError:
            pass
        except OSError as error:
            logging.warning(
                "Could not release the case claim for %s (%s); stale recovery will handle it.",
                self.case_id,
                type(error).__name__,
            )

    def __enter__(self) -> CaseLease:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class SharedExecutionState:
    """Filesystem-backed state for one run queue within one workspace."""

    _local_guards_lock = threading.Lock()
    _local_guards: dict[str, threading.Lock] = {}

    def __init__(
        self,
        workspace: str | Path,
        queue_id: str,
        *,
        policy: ExecutionPolicy | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.queue_id = str(queue_id)
        self.policy = policy or ExecutionPolicy.from_environment()
        self.root = self.workspace / ".bodycomposition" / "execution" / self.queue_id
        self.worker_id = self._worker_id()

    @staticmethod
    def _worker_id() -> str:
        # A random operational identifier is sufficient for fencing.  Hostnames,
        # usernames, process IDs, and scheduler IDs do not belong in run state.
        return f"worker-{uuid4().hex}"

    def _path(self, category: str, case_id: str) -> Path:
        return self.root / category / _case_key(case_id)

    @classmethod
    def _local_guard(cls, path: Path) -> threading.Lock:
        key = str(path.resolve())
        with cls._local_guards_lock:
            return cls._local_guards.setdefault(key, threading.Lock())

    @contextmanager
    def _claim_guard(self, case_id: str):
        """Serialize claim/takeover decisions across threads and POSIX processes."""

        path = self._path("guards", case_id).with_suffix(".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._local_guard(path), path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def coordination_guard(self, name: str):
        """Serialize one small run-level coordination operation."""

        normalized = str(name).strip()
        if not normalized:
            raise ValueError("Coordination guard name must not be empty.")
        with self._claim_guard(f"coordination-{normalized}"):
            yield

    def initialize_run_clock(
        self,
        *,
        started_at: str,
        plan_sha256: str,
    ) -> str:
        """Return one durable start time shared by every compatible worker."""

        candidate = str(started_at).strip()
        plan_digest = str(plan_sha256).strip()
        if not candidate:
            raise ValueError("Run start time must not be empty.")
        if not re.fullmatch(r"[a-f0-9]{64}", plan_digest):
            raise ValueError("Run plan digest must be a lowercase SHA-256 value.")
        path = self.root / "runtime" / "run.json"
        with self.coordination_guard("run-initialize"):
            record = _read_json(path)
            if path.exists() and record is None:
                raise RuntimeError("Shared run initialization state is unreadable.")
            if record is None:
                record = {
                    "schema_version": 1,
                    "queue_id": self.queue_id,
                    "plan_sha256": plan_digest,
                    "started_at": candidate,
                }
                _atomic_write_json(path, record)
            if (
                record.get("queue_id") != self.queue_id
                or record.get("plan_sha256") != plan_digest
                or not isinstance(record.get("started_at"), str)
                or not str(record["started_at"]).strip()
            ):
                raise RuntimeError(
                    "Shared run initialization state differs from the immutable plan."
                )
            return str(record["started_at"])

    def completed(self, case_id: str) -> bool:
        record = self.completed_record(case_id)
        raw_path = record.get("manifest") if record else None
        if not isinstance(raw_path, str) or not raw_path:
            return False
        path = Path(raw_path)
        return (path if path.is_absolute() else self.workspace / path).is_file()

    def terminal_failed(self, case_id: str) -> bool:
        return self._path("terminal_failed", case_id).with_suffix(".json").is_file()

    def completed_record(self, case_id: str) -> dict[str, Any] | None:
        return _read_json(self._path("completed", case_id).with_suffix(".json"))

    def active_claim_record(self, case_id: str) -> dict[str, Any] | None:
        return _read_json(self._path("claims", case_id) / "owner.json")

    def active_claim_is_stale(self, case_id: str) -> bool:
        claim = self._path("claims", case_id)
        return not claim.is_dir() or self._claim_stale(claim)

    def terminal_failure_record(self, case_id: str) -> dict[str, Any] | None:
        return _read_json(self._path("terminal_failed", case_id).with_suffix(".json"))

    def attach_terminal_manifest(self, case_id: str, manifest_path: str | Path) -> None:
        marker = self._path("terminal_failed", case_id).with_suffix(".json")
        record = _read_json(marker)
        if record is None:
            return
        _atomic_write_json(
            marker,
            {**record, "manifest": self._manifest_reference(manifest_path)},
        )

    def result_manifest(self, case_id: str) -> Path | None:
        record = self.completed_record(case_id) or self.terminal_failure_record(case_id)
        raw_path = record.get("manifest") if record else None
        if not isinstance(raw_path, str) or not raw_path:
            return None
        path = Path(raw_path)
        return path if path.is_absolute() else self.workspace / path

    def _manifest_reference(self, manifest_path: str | Path) -> str:
        path = Path(manifest_path)
        try:
            return path.relative_to(self.workspace).as_posix()
        except ValueError:
            return str(path)

    def _low_memory_marker(self, hardware_profile: Mapping[str, Any]) -> Path:
        return (
            self.root
            / "runtime"
            / "low_memory"
            / f"{hardware_profile_id(hardware_profile)}.json"
        )

    def low_memory_enabled(self, hardware_profile: Mapping[str, Any]) -> bool:
        marker = _read_json(self._low_memory_marker(hardware_profile))
        return bool(
            marker
            and marker.get("hardware_profile_id")
            == hardware_profile_id(hardware_profile)
            and marker.get("hardware_profile") == dict(hardware_profile)
        )

    def stop_requested(self) -> bool:
        return (self.root / "runtime" / "stop.json").is_file()

    def request_stop(self, *, reason: str) -> None:
        marker = self.root / "runtime" / "stop.json"
        if marker.is_file():
            return
        _atomic_write_json(
            marker,
            {
                "queue_id": self.queue_id,
                "requested_at": time.time(),
                "reason": str(reason),
            },
        )

    def record_out_of_memory(
        self,
        case_id: str,
        error: BaseException,
        *,
        hardware_profile: Mapping[str, Any],
    ) -> None:
        profile_id = hardware_profile_id(hardware_profile)
        _atomic_write_json(
            self._low_memory_marker(hardware_profile),
            {
                "queue_id": self.queue_id,
                "case_id": str(case_id),
                "created_at": time.time(),
                "reason": "accelerator_out_of_memory",
                "exception_type": error.__class__.__name__,
                "hardware_profile_id": profile_id,
                "hardware_profile": dict(hardware_profile),
                "strategy_on_restart": "single_segmentation_bundle",
            },
        )

    def _claim_stale(self, claim_directory: Path) -> bool:
        owner = _read_json(claim_directory / "owner.json")
        heartbeat = _read_json(claim_directory / "heartbeat.json")
        token = owner.get("token") if owner else None
        if token and heartbeat and heartbeat.get("token") == token:
            updated_at = heartbeat.get("updated_at")
        else:
            updated_at = owner.get("created_at") if owner else None
        if not isinstance(updated_at, (int, float)):
            try:
                updated_at = claim_directory.stat().st_mtime
            except OSError:
                return True
        now = time.time()
        stale_after_seconds = owner.get("stale_after_seconds") if owner else None
        if not isinstance(stale_after_seconds, (int, float)) or stale_after_seconds <= 0:
            stale_after_seconds = self.policy.stale_after_seconds
        heartbeat_expired = now - float(updated_at) > float(stale_after_seconds)
        created_at = owner.get("created_at") if owner else None
        max_claim_seconds = owner.get("max_claim_seconds") if owner else None
        if not isinstance(max_claim_seconds, (int, float)) or max_claim_seconds <= 0:
            max_claim_seconds = self.policy.max_claim_seconds
        maximum_age_reached = isinstance(created_at, (int, float)) and (
            now - float(created_at) > float(max_claim_seconds)
        )
        return heartbeat_expired or maximum_age_reached

    def _archive_stale_claim(self, case_id: str, claim_directory: Path) -> Path | None:
        archive = (
            self._path("stale_claims", case_id)
            / f"{time.time_ns()}-{uuid4().hex}"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        try:
            claim_directory.rename(archive)
        except (FileNotFoundError, FileExistsError, OSError):
            return None
        return archive

    def _write_heartbeat(
        self,
        claim_directory: Path,
        token: str,
        *,
        stage: str | None = None,
    ) -> None:
        owner = _read_json(claim_directory / "owner.json")
        if not owner or owner.get("token") != token:
            return
        previous = _read_json(claim_directory / "heartbeat.json") or {}
        _atomic_write_json(
            claim_directory / "heartbeat.json",
            {
                "token": token,
                "updated_at": time.time(),
                "stage": str(stage or previous.get("stage") or "claimed"),
            },
        )

    def _record_attempt(
        self,
        case_id: str,
        payload: Mapping[str, Any],
    ) -> int:
        attempt_directory = self._path("attempts", case_id)
        attempt_directory.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            attempt_directory / f"{time.time_ns()}-{uuid4().hex}.json",
            {
                "queue_id": self.queue_id,
                "case_id": str(case_id),
                **dict(payload),
            },
        )
        return self.attempt_count(case_id)

    def _mark_stale_terminal(
        self,
        case_id: str,
        *,
        attempts: int,
        stage: str,
    ) -> None:
        _atomic_write_json(
            self._path("terminal_failed", case_id).with_suffix(".json"),
            {
                "queue_id": self.queue_id,
                "case_id": str(case_id),
                "failed_at": time.time(),
                "attempts": attempts,
                "last_exception_type": "StaleClaim",
                "last_stage": stage,
                "failure_code": "stale_worker_limit_exceeded",
                "failure_summary": (
                    "The maximum number of abandoned worker claims was reached; "
                    "automatic retry stopped."
                ),
            },
        )

    def try_claim(self, case_id: str) -> CaseLease | None:
        with self._claim_guard(case_id):
            return self._try_claim_locked(case_id)

    def _try_claim_locked(self, case_id: str) -> CaseLease | None:
        if self.completed(case_id) or self.terminal_failed(case_id):
            return None
        attempts = self.attempt_count(case_id)
        if attempts >= self.policy.max_attempts:
            attempt_paths = sorted(self._path("attempts", case_id).glob("*.json"))
            last = _read_json(attempt_paths[-1]) if attempt_paths else {}
            last = last or {}
            _atomic_write_json(
                self._path("terminal_failed", case_id).with_suffix(".json"),
                {
                    "queue_id": self.queue_id,
                    "case_id": str(case_id),
                    "failed_at": time.time(),
                    "attempts": attempts,
                    "last_exception_type": last.get("exception_type", "UnknownFailure"),
                    "last_stage": last.get("stage", "unknown"),
                    "manifest": last.get("manifest"),
                    "failure_code": "worker_attempt_limit_exceeded",
                    "failure_summary": "The bounded case-attempt limit was reached.",
                },
            )
            return None
        claim_directory = self._path("claims", case_id)
        claim_directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            claim_directory.mkdir()
        except FileExistsError:
            if not self._claim_stale(claim_directory):
                return None
            archive = self._archive_stale_claim(case_id, claim_directory)
            if archive is None:
                return None
            owner = _read_json(archive / "owner.json") or {}
            heartbeat = _read_json(archive / "heartbeat.json") or {}
            attempts = self._record_attempt(
                case_id,
                {
                    "claim_token": owner.get("token"),
                    "worker_id": owner.get("worker_id"),
                    "failed_at": time.time(),
                    "stage": str(heartbeat.get("stage") or "unknown"),
                    "exception_type": "StaleClaim",
                    "out_of_memory": False,
                    "abandoned": True,
                },
            )
            if attempts >= self.policy.max_attempts:
                self._mark_stale_terminal(
                    case_id,
                    attempts=attempts,
                    stage=str(heartbeat.get("stage") or "unknown"),
                )
                return None
            try:
                claim_directory.mkdir()
            except FileExistsError:
                return None

        token = uuid4().hex
        created_at = time.time()
        _atomic_write_json(
            claim_directory / "owner.json",
            {
                "queue_id": self.queue_id,
                "case_id": str(case_id),
                "token": token,
                "worker_id": self.worker_id,
                "created_at": created_at,
                "stale_after_seconds": self.policy.stale_after_seconds,
                "max_claim_seconds": self.policy.max_claim_seconds,
            },
        )
        self._write_heartbeat(claim_directory, token, stage="claimed")
        return CaseLease(self, case_id, claim_directory, token)

    def attempt_count(self, case_id: str) -> int:
        directory = self._path("attempts", case_id)
        return sum(1 for path in directory.glob("*.json") if path.is_file())

    def record_failure(
        self,
        lease: CaseLease,
        error: BaseException,
        *,
        stage: str,
        out_of_memory: bool,
        manifest_path: str | Path,
    ) -> bool:
        lease.assert_owned()
        attempts = self._record_attempt(
            lease.case_id,
            {
                "claim_token": lease.token,
                "worker_id": self.worker_id,
                "failed_at": time.time(),
                "stage": str(stage),
                "exception_type": error.__class__.__name__,
                "out_of_memory": bool(out_of_memory),
                "manifest": self._manifest_reference(manifest_path),
            },
        )
        terminal = attempts >= self.policy.max_attempts
        if terminal:
            _atomic_write_json(
                self._path("terminal_failed", lease.case_id).with_suffix(".json"),
                {
                    "queue_id": self.queue_id,
                    "case_id": lease.case_id,
                    "failed_at": time.time(),
                    "attempts": attempts,
                    "last_exception_type": error.__class__.__name__,
                    "last_stage": str(stage),
                    "manifest": self._manifest_reference(manifest_path),
                },
            )
        return terminal

    def mark_terminal_failure(
        self,
        lease: CaseLease,
        *,
        manifest_path: str | Path,
        stage: str,
        exception_type: str,
    ) -> None:
        """Record a deterministic failure that should not be retried."""

        lease.assert_owned()
        attempts = self._record_attempt(
            lease.case_id,
            {
                "claim_token": lease.token,
                "worker_id": self.worker_id,
                "failed_at": time.time(),
                "stage": str(stage),
                "exception_type": str(exception_type),
                "out_of_memory": False,
                "manifest": self._manifest_reference(manifest_path),
                "non_retryable": True,
            },
        )
        _atomic_write_json(
            self._path("terminal_failed", lease.case_id).with_suffix(".json"),
            {
                "queue_id": self.queue_id,
                "case_id": lease.case_id,
                "failed_at": time.time(),
                "attempts": attempts,
                "last_exception_type": str(exception_type),
                "last_stage": str(stage),
                "manifest": self._manifest_reference(manifest_path),
            },
        )

    def mark_completed(
        self,
        lease: CaseLease,
        *,
        manifest_path: str | Path,
        reused_outputs: bool = False,
    ) -> None:
        lease.assert_owned()
        _atomic_write_json(
            self._path("completed", lease.case_id).with_suffix(".json"),
            {
                "queue_id": self.queue_id,
                "case_id": lease.case_id,
                "claim_token": lease.token,
                "worker_id": self.worker_id,
                "completed_at": time.time(),
                "reused_outputs": bool(reused_outputs),
                "manifest": self._manifest_reference(manifest_path),
            },
        )
