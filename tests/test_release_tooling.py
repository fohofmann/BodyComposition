from __future__ import annotations

import gzip
import hashlib
import io
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from BodyComposition.utils.digests import array_sha256
from scripts.audit_distribution import audit
from scripts.generate_sbom import generate
from scripts.release_checks import (
    _clear_stale_outputs,
    _git_commit,
    _normalize_sdist,
    _prepare_release_gate,
)


def _wheel(path: Path, *, include_weights: bool = False) -> Path:
    members = {
        "bodycomposition-1.0.dist-info/licenses/LICENSE": "Apache-2.0",
        "bodycomposition-1.0.dist-info/licenses/THIRD_PARTY_NOTICES.md": "Notices",
        "bodycomposition-1.0.dist-info/README.md": "Read me",
        "BodyComposition/schemas/case_manifest.schema.json": "{}",
        "BodyComposition/schemas/run_manifest.schema.json": "{}",
    }
    if include_weights:
        members["BodyComposition/models/checkpoint.pth"] = "weights"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return path


def test_distribution_audit_rejects_model_weights(tmp_path):
    clean = audit(_wheel(tmp_path / "clean.whl"))
    dirty = audit(_wheel(tmp_path / "dirty.whl", include_weights=True))
    assert clean["passed"]
    assert not dirty["passed"]
    assert {item["code"] for item in dirty["findings"]} >= {
        "forbidden_path",
        "model_or_image_asset",
    }


def test_sbom_is_deterministic_and_covers_the_frozen_project():
    first = generate(Path("uv.lock"))
    second = generate(Path("uv.lock"))
    assert first == second
    components = {item["name"].lower(): item for item in first["components"]}
    assert components["bodycomposition"]["version"] == "1.0.0rc1"
    assert components["spineps"]["version"] == "2.0.0"
    assert components["tptbox"]["version"] == "0.7.5"


def test_container_definition_is_weight_free_pinned_and_non_root():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
    assert dockerfile.count("@sha256:") >= 2
    assert "USER 10001:10001" in dockerfile
    assert 'BODYCOMPOSITION_MODEL_ROOT=/models' in dockerfile
    assert 'HF_HUB_DISABLE_XET=1' in dockerfile
    assert 'HF_HUB_DOWNLOAD_TIMEOUT=600' in dockerfile
    assert "TOTALSEG_HOME_DIR=/models" not in dockerfile
    assert "TOTALSEG_WEIGHTS_PATH=/models" not in dockerfile
    assert 'BODYCOMPOSITION_OUTPUT_ROOT=/output' in dockerfile
    assert 'VOLUME ["/input", "/output", "/models"]' in dockerfile
    assert "models sync" not in dockerfile
    assert "-name '*.pt'" in dockerfile
    assert "lpips_models/*.pth" in dockerfile
    assert "-name '*.nii.gz'" in dockerfile
    assert "-name '*.dcm'" in dockerfile
    assert "-size +1024c" in dockerfile
    assert dockerfile.index("lpips_models/*.pth") < dockerfile.index(
        "FROM ${PYTHON_IMAGE} AS runtime"
    )
    assert "site-packages/spineps/models" in dockerfile
    assert "-name '*.onnx'" in dockerfile
    assert "--reinstall-package BodyComposition" in dockerfile
    assert "*.pth" in dockerignore and "*.pt" in dockerignore
    assert "output/" in dockerignore


def test_measurement_support_uses_the_pinned_nnunet_engine_directly():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]

    assert "nnunetv2==2.5.2" in dependencies
    assert not any(
        dependency.lower().startswith("totalsegmentator")
        for dependency in dependencies
    )


def test_checkpoint_runtime_uses_the_exact_validated_safe_torch_pair():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]

    assert "torch==2.13.0" in dependencies
    assert "torchvision==0.28.0" in dependencies


def test_ci_and_release_gate_pin_tools_and_run_supply_chain_checks():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    release = Path("scripts/release_checks.py").read_text(encoding="utf-8")
    assert "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683" in workflow
    assert "astral-sh/setup-uv@1e862dfacbd1d6d858c55d9b792c756523627244" in workflow
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in workflow
    assert "scripts/release_checks.py" in workflow
    assert "--no-hashes" not in workflow
    assert "--no-hashes" not in release
    for requirement in (
        "pytest",
        "pip-audit",
        "clean-install-doctor.json",
        "reproducibility-report.json",
        "reproducible_builds",
        "release-checks.json",
        "source_cleanliness_enforced",
        "uv",
        "venv",
    ):
        assert requirement in release


def test_release_coverage_gate_measures_the_complete_package():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    coverage = project["tool"]["coverage"]

    assert coverage["run"]["branch"] is True
    assert coverage["run"]["source"] == ["BodyComposition"]
    assert "omit" not in coverage["run"]
    assert coverage["report"]["fail_under"] >= 80


def test_release_gate_removes_only_known_stale_evidence(tmp_path):
    stale_receipt = tmp_path / "release-checks.json"
    stale_wheel = tmp_path / "bodycomposition-1.0.0rc1-py3-none-any.whl"
    unrelated = tmp_path / "keep-me.txt"
    for path in (stale_receipt, stale_wheel, unrelated):
        path.write_text("old\n", encoding="utf-8")

    _clear_stale_outputs(tmp_path)

    assert not stale_receipt.exists()
    assert not stale_wheel.exists()
    assert unrelated.read_text(encoding="utf-8") == "old\n"


def test_dirty_release_attempt_invalidates_an_older_passing_receipt(tmp_path, monkeypatch):
    stale_receipt = tmp_path / "dist/release-checks.json"
    stale_receipt.parent.mkdir()
    stale_receipt.write_text('{"passed": true}\n', encoding="utf-8")
    monkeypatch.setattr(
        "scripts.release_checks.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M BodyComposition/example.py\n"),
    )

    with pytest.raises(SystemExit, match="clean reviewed source tree"):
        _prepare_release_gate(tmp_path, allow_dirty=False)

    assert not stale_receipt.exists()


def test_release_receipt_commit_is_a_full_lowercase_revision(monkeypatch):
    monkeypatch.setattr(
        "scripts.release_checks.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=("A" * 40) + "\n"),
    )

    assert _git_commit() == "a" * 40


def test_array_digest_matches_c_order_bytes_for_empty_and_fortran_arrays():
    arrays = (
        np.empty((0, 2), dtype=np.int16),
        np.asfortranarray(np.arange(24, dtype=np.float32).reshape(2, 3, 4)),
    )
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        expected = hashlib.sha256()
        expected.update(str(contiguous.dtype).encode("ascii"))
        expected.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        expected.update(contiguous.tobytes())
        assert array_sha256(array) == expected.hexdigest()


def test_public_product_surface_omits_internal_development_residue():
    public_files = [
        *Path("BodyComposition").rglob("*.py"),
        *Path("scripts").rglob("*.py"),
        *Path("docs").glob("*.md"),
        Path("README.md"),
        Path("THIRD_PARTY_NOTICES.md"),
    ]
    forbidden = {f"wp{index:02d}" for index in range(1, 10)}
    task_id = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    )
    findings = {}
    for path in public_files:
        text = path.read_text(encoding="utf-8").lower()
        matched = sorted(token for token in forbidden if token in text)
        if task_id.search(text):
            matched.append("task_id")
        if matched:
            findings[path.as_posix()] = matched
    assert findings == {}


def _source_archive(path: Path, *, mtime: int) -> Path:
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=mtime) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        directory = tarfile.TarInfo("example-1.0")
        directory.type = tarfile.DIRTYPE
        directory.mtime = mtime
        archive.addfile(directory)
        payload = b"stable source\n"
        member = tarfile.TarInfo("example-1.0/module.py")
        member.size = len(payload)
        member.mtime = mtime
        member.uid = mtime % 100
        archive.addfile(member, io.BytesIO(payload))
    return path


def test_sdist_normalization_is_bit_reproducible(tmp_path):
    first = _source_archive(tmp_path / "first.tar.gz", mtime=1_700_000_000)
    second = _source_archive(tmp_path / "second.tar.gz", mtime=1_800_000_000)

    _normalize_sdist(first, epoch=1_750_000_000)
    _normalize_sdist(second, epoch=1_750_000_000)

    assert first.read_bytes() == second.read_bytes()
