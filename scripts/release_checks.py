#!/usr/bin/env python3
"""Run the reproducible local release-candidate gate."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_OUTPUT_NAMES = {
    "artifact-checksums.json",
    "clean-install-doctor.json",
    "coverage.xml",
    "distribution-audit.json",
    "release-checks.json",
    "reproducibility-report.json",
    "runtime-requirements.txt",
    "vulnerability-report.json",
}


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clear_stale_outputs(output: Path) -> None:
    """Remove prior release evidence before starting a new gate."""

    output.mkdir(parents=True, exist_ok=True)
    candidates = {output / name for name in RELEASE_OUTPUT_NAMES}
    candidates.update(output.glob("bodycomposition-1.0.0rc1*"))
    for path in candidates:
        if path.is_file():
            path.unlink()


def _prepare_release_gate(root: Path, *, allow_dirty: bool) -> None:
    """Invalidate prior evidence, then enforce committed source when requested."""

    _clear_stale_outputs(root / "dist")
    (root / "coverage.xml").unlink(missing_ok=True)
    if allow_dirty:
        return
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise SystemExit("release gate requires a clean reviewed source tree")


def _source_date_epoch() -> int:
    """Return the explicit or committed source timestamp used for builds."""

    value = os.environ.get("SOURCE_DATE_EPOCH")
    if value is None:
        value = subprocess.run(
            ["git", "show", "-s", "--format=%ct", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    try:
        epoch = int(value)
    except ValueError as error:
        raise SystemExit("SOURCE_DATE_EPOCH must be an integer Unix timestamp") from error
    if not 315532800 <= epoch <= 4294967295:
        raise SystemExit("SOURCE_DATE_EPOCH must fit the reproducible gzip timestamp range")
    os.environ["SOURCE_DATE_EPOCH"] = str(epoch)
    return epoch


def _git_commit() -> str:
    """Return the exact committed source revision covered by the receipt."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit.lower()
    ):
        raise SystemExit("git returned an invalid release commit")
    return commit.lower()


def _normalize_sdist(path: Path, *, epoch: int) -> None:
    """Rewrite an sdist with stable ordering, ownership, and timestamps."""

    temporary = path.with_name(f".{path.name}.normalized")
    try:
        with (
            tarfile.open(path, mode="r:gz") as source,
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed,
            tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as destination,
        ):
            for original in sorted(source.getmembers(), key=lambda item: item.name):
                member = copy.copy(original)
                member.uid = 0
                member.gid = 0
                member.uname = "root"
                member.gname = "root"
                member.mtime = epoch
                member.pax_headers = {}
                content = source.extractfile(original) if original.isfile() else None
                destination.addfile(member, content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_distributions(output: Path, *, epoch: int) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(output),
        ]
    )
    artifacts = {
        path.name: path
        for path in output.glob("bodycomposition-1.0.0rc1*")
        if path.suffix == ".whl" or path.name.endswith(".tar.gz")
    }
    if len(artifacts) != 2:
        raise SystemExit(f"expected one wheel and one sdist, found {sorted(artifacts)}")
    for path in artifacts.values():
        if path.name.endswith(".tar.gz"):
            _normalize_sdist(path, epoch=epoch)
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run development checks but record that the release cleanliness gate was skipped",
    )
    args = parser.parse_args()
    _prepare_release_gate(ROOT, allow_dirty=args.allow_dirty)
    source_date_epoch = _source_date_epoch()
    git_commit = _git_commit()

    _run(["uv", "lock", "--check"])
    _run(["uv", "sync", "--frozen", "--extra", "test", "--extra", "release"])
    _run(["uv", "run", "--no-sync", "ruff", "check", "BodyComposition", "tests", "scripts"])
    _run(["uv", "run", "--no-sync", "mypy"])
    _run(
        [
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "--cov",
            "--cov-report=term",
            "--cov-report=xml",
        ]
    )
    built = _build_distributions(ROOT / "dist", epoch=source_date_epoch)
    with tempfile.TemporaryDirectory(prefix="bodycomposition-rebuild-") as temporary:
        rebuilt = _build_distributions(Path(temporary), epoch=source_date_epoch)
        built_hashes = {name: _sha256(path) for name, path in sorted(built.items())}
        rebuilt_hashes = {name: _sha256(path) for name, path in sorted(rebuilt.items())}
        if built_hashes != rebuilt_hashes:
            raise SystemExit(
                "release artifacts are not reproducible: "
                f"first={built_hashes}, second={rebuilt_hashes}"
            )
    (ROOT / "dist/reproducibility-report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "passed": True,
                "source_date_epoch": source_date_epoch,
                "artifact_checksums": built_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    distributions = [built[name] for name in sorted(built)]
    _run(
        [
            "uv", "run", "--no-sync", "python", "scripts/audit_distribution.py",
            *[str(path) for path in distributions],
            "--output", "dist/distribution-audit.json",
        ]
    )
    _run(["uv", "run", "--no-sync", "python", "scripts/generate_sbom.py"])
    runtime_requirements = ROOT / "dist/runtime-requirements.txt"
    _run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--output-file",
            str(runtime_requirements),
        ]
    )
    _run(
        [
            "uv",
            "run",
            "--no-sync",
            "pip-audit",
            "--requirement",
            str(runtime_requirements),
            "--no-deps",
            "--disable-pip",
            "--strict",
            "--format",
            "json",
            "--output",
            "dist/vulnerability-report.json",
        ]
    )
    wheel = next(path for path in distributions if path.suffix == ".whl")
    doctor_payload: dict[str, object]
    with tempfile.TemporaryDirectory(prefix="bodycomposition-wheel-") as temporary:
        environment = Path(temporary) / "venv"
        _run(["uv", "venv", str(environment), "--python", "3.11.14"])
        interpreter = environment / "bin/python"
        executable = environment / "bin/bodycomposition"
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(interpreter),
                "--requirement",
                str(runtime_requirements),
            ]
        )
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(interpreter),
                "--no-deps",
                str(wheel),
            ]
        )
        _run(["uv", "pip", "check", "--python", str(interpreter)])
        _run([str(executable), "version", "--json"])
        _run([str(executable), "convert", "--help"])
        config_result = subprocess.run(
            [str(executable), "config", "show-default", "--json"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if json.loads(config_result.stdout)["schema_version"] != "1.0.0":
            raise SystemExit("clean wheel returned an unexpected configuration schema")
        _run(
            [
                str(interpreter),
                "-c",
                "from BodyComposition import PipelineConfig, convert_dicom, discover_dicom_series; "
                "assert PipelineConfig.load().normalized()['schema_version'] == '1.0.0'; "
                "assert callable(convert_dicom) and callable(discover_dicom_series)",
            ]
        )
        isolated_home = Path(temporary) / "home"
        isolated_home.mkdir()
        doctor = subprocess.run(
            [
                str(executable),
                "doctor",
                "--output",
                str(Path(temporary) / "output"),
                "--json",
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "HOME": str(isolated_home),
                "BODYCOMPOSITION_MODEL_ROOT": str(Path(temporary) / "models"),
            },
            check=False,
            capture_output=True,
            text=True,
        )
        if doctor.returncode not in {0, 3}:
            raise subprocess.CalledProcessError(
                doctor.returncode,
                doctor.args,
                output=doctor.stdout,
                stderr=doctor.stderr,
            )
        doctor_payload = json.loads(doctor.stdout)
        expected_operational_errors = {
            "package_lock_digest_unavailable",
            "required_models_not_ready",
        }
        if set(doctor_payload["operational_errors"]) != expected_operational_errors:
            raise SystemExit(
                "clean wheel doctor returned unexpected operational errors: "
                f"{doctor_payload['operational_errors']}"
            )
        (ROOT / "dist/clean-install-doctor.json").write_text(
            json.dumps(doctor_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    checksums = {path.name: _sha256(path) for path in sorted(distributions)}
    (ROOT / "dist/artifact-checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ROOT / "dist/release-checks.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "passed": True,
                "git_commit": git_commit,
                "source_cleanliness_enforced": not args.allow_dirty,
                "source_date_epoch": source_date_epoch,
                "package_lock_sha256": _sha256(ROOT / "uv.lock"),
                "reproducible_builds": True,
                "artifact_checksums": checksums,
                "clean_install_expected_operational_errors": doctor_payload[
                    "operational_errors"
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("BodyComposition local release checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
