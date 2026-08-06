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

import BodyComposition.provenance as provenance
import scripts.build_container as build_container
import scripts.container_checks as container_checks
from BodyComposition.utils.digests import array_sha256
from scripts.audit_container import (
    _configuration_findings,
    _contains_credential,
    _forbidden_asset,
    _scan_image_archive,
)
from scripts.audit_distribution import audit
from scripts.generate_sbom import generate
from scripts.release_checks import (
    _clear_stale_outputs,
    _git_commit,
    _normalize_sdist,
    _prepare_release_gate,
)
from scripts.sanitize_container_environment import sanitize


def _wheel(
    path: Path,
    *,
    include_weights: bool = False,
    extra_text: str | None = None,
) -> Path:
    members = {
        "bodycomposition-1.0.dist-info/licenses/LICENSE": "Apache-2.0",
        "bodycomposition-1.0.dist-info/licenses/THIRD_PARTY_NOTICES.md": "Notices",
        "bodycomposition-1.0.dist-info/README.md": "Read me",
        "BodyComposition/schemas/case_manifest.schema.json": "{}",
        "BodyComposition/schemas/run_manifest.schema.json": "{}",
    }
    if include_weights:
        members["BodyComposition/models/checkpoint.pth"] = "weights"
    if extra_text is not None:
        members["bodycomposition-1.0.dist-info/extra.txt"] = extra_text
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


def test_distribution_audit_rejects_token_shaped_secret(tmp_path):
    synthetic_token = "hf_" + ("A" * 24)
    result = audit(_wheel(tmp_path / "credential.whl", extra_text=synthetic_token))

    assert not result["passed"]
    assert any(item["code"] == "provider_token" for item in result["findings"])


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
    assert (
        "python:3.11.15-slim-bookworm@sha256:"
        "b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba"
        in dockerfile
    )
    assert "USER 10001:10001" in dockerfile
    assert 'BODYCOMPOSITION_MODEL_ROOT=/models' in dockerfile
    assert 'HF_HUB_DISABLE_XET=1' in dockerfile
    assert 'HF_HUB_DOWNLOAD_TIMEOUT=600' in dockerfile
    assert "TOTALSEG_HOME_DIR=/models" not in dockerfile
    assert "TOTALSEG_WEIGHTS_PATH=/models" not in dockerfile
    assert 'BODYCOMPOSITION_OUTPUT_ROOT=/output' in dockerfile
    assert 'VOLUME ["/input", "/output", "/models"]' in dockerfile
    assert "models sync" not in dockerfile
    assert "scripts/sanitize_container_environment.py" in dockerfile
    assert dockerfile.index("sanitize_container_environment.py") < dockerfile.index(
        "FROM ${PYTHON_IMAGE} AS runtime"
    )
    assert "site-packages/spineps/models" in dockerfile
    assert 'org.bodycomposition.source-dirty="${SOURCE_DIRTY}"' in dockerfile
    assert 'org.bodycomposition.cuda-runtime="13.0"' in dockerfile
    assert "rm -rf /usr/local/lib/python3.11/site-packages" in dockerfile
    assert "--reinstall-package BodyComposition" in dockerfile
    assert "*.pth" in dockerignore and "*.pt" in dockerignore
    assert "output/" in dockerignore
    assert "dev/" in dockerignore


def test_container_sanitizer_removes_only_known_assets_and_rejects_unknown(tmp_path):
    environment = tmp_path / "environment"
    site_packages = environment / "lib/python3.11/site-packages"
    known_scan = site_packages / "TPTBox/tests/sample_ct/example.nii.gz"
    known_weight = (
        site_packages / "torchmetrics/functional/image/lpips_models/alex.pth"
    )
    ordinary_test = site_packages / "example/tests/keep.txt"
    for path, payload in (
        (known_scan, b"ct"),
        (known_weight, b"weight"),
        (ordinary_test, b"ordinary"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    result = sanitize(environment)

    assert result["passed"]
    assert not known_scan.exists()
    assert not known_weight.exists()
    assert ordinary_test.read_bytes() == b"ordinary"

    unexpected = site_packages / "new_dependency/tests/unexpected.dcm"
    unexpected.parent.mkdir(parents=True)
    unexpected.write_bytes(b"dicom")
    with pytest.raises(RuntimeError, match="unexpected.dcm"):
        sanitize(environment)


def test_container_layer_policy_rejects_checkpoints_and_medical_images():
    assert _forbidden_asset("models/checkpoint.pt", 1)
    assert _forbidden_asset("fixtures/case.nii.gz", 1)
    assert _forbidden_asset("package/model.pth", 1025)
    assert not _forbidden_asset("package/typing.pth", 40)
    assert not _forbidden_asset("package/model.py", 100_000)


def test_container_credential_scan_uses_token_boundaries_and_text_only():
    assert _contains_credential(b"HF_TOKEN=hf_" + (b"A" * 24))
    assert _contains_credential(b"API_KEY=sk-proj-" + (b"A" * 40))
    assert not _contains_credential(b"mask-image: linear-gradient(black, white)")
    assert not _contains_credential(b"binary\0hf_" + (b"A" * 24))


def test_container_layer_audit_supports_oci_blob_archives():
    layer_buffer = io.BytesIO()
    with tarfile.open(fileobj=layer_buffer, mode="w") as layer:
        payload = b"weight"
        member = tarfile.TarInfo("opt/package/checkpoint.pt")
        member.size = len(payload)
        layer.addfile(member, io.BytesIO(payload))
        link = tarfile.TarInfo("opt/package/linked-model.pth")
        link.type = tarfile.SYMTYPE
        link.linkname = "checkpoint.bin"
        layer.addfile(link)

    image_buffer = io.BytesIO()
    with tarfile.open(fileobj=image_buffer, mode="w") as image:
        metadata = b'{"schemaVersion": 2}'
        metadata_member = tarfile.TarInfo("blobs/sha256/metadata")
        metadata_member.size = len(metadata)
        image.addfile(metadata_member, io.BytesIO(metadata))
        layer_payload = layer_buffer.getvalue()
        layer_member = tarfile.TarInfo("blobs/sha256/layer")
        layer_member.size = len(layer_payload)
        image.addfile(layer_member, io.BytesIO(layer_payload))
    image_buffer.seek(0)

    layers, files, size, findings = _scan_image_archive(image_buffer)

    assert layers == 1
    assert files == 1
    assert size == len(b"weight")
    assert {finding["path"] for finding in findings} == {
        "opt/package/checkpoint.pt",
        "opt/package/linked-model.pth",
    }


def test_container_configuration_policy_requires_non_root_external_mounts_and_labels():
    labels = {
        "org.opencontainers.image.title": "BodyComposition",
        "org.opencontainers.image.description": "Validated CT body-composition research pipeline",
        "org.opencontainers.image.licenses": "Apache-2.0",
        "org.opencontainers.image.revision": "a" * 40,
        "org.bodycomposition.uv-lock-sha256": "b" * 64,
        "org.bodycomposition.source-sha256": "c" * 64,
        "org.bodycomposition.source-dirty": "false",
        "org.bodycomposition.cuda-runtime": "13.0",
        "org.bodycomposition.model-weights": "not-included",
    }
    inspect = {
        "Config": {
            "User": "10001:10001",
            "Volumes": {"/input": {}, "/models": {}, "/output": {}},
            "Labels": labels,
            "Env": ["BODYCOMPOSITION_MODEL_ROOT=/models"],
        }
    }
    assert _configuration_findings(inspect) == []

    inspect["Config"]["User"] = "root"
    inspect["Config"]["Env"] = ["ACCESS_TOKEN=not-a-real-token"]
    codes = {finding["code"] for finding in _configuration_findings(inspect)}
    assert codes == {"unexpected_user", "sensitive_environment_name"}


def test_container_build_request_is_explicit_reproducible_and_multi_platform(monkeypatch):
    def fake_git(*arguments):
        if arguments[:2] == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments[0] == "status":
            return ""
        if arguments[0] == "show":
            return "1700000000"
        raise AssertionError(arguments)

    monkeypatch.setattr(build_container, "_git", fake_git)
    monkeypatch.setattr(build_container, "source_context_sha256", lambda: "b" * 64)
    monkeypatch.setattr(build_container, "_sha256", lambda path: "c" * 64)
    args = SimpleNamespace(
        allow_dirty=False,
        platform="linux/amd64,linux/arm64",
        push=True,
        load=False,
        tag=["registry.example/bodycomposition:test"],
        metadata_file="dist/metadata.json",
        receipt="dist/request.json",
        no_pull=False,
        no_cache=False,
        no_attestations=False,
    )

    request = build_container.build_request(args)

    command = request["command"]
    assert request["platforms"] == ["linux/amd64", "linux/arm64"]
    assert request["attestations"] is True
    assert "SOURCE_DIRTY=false" in command
    assert "SOURCE_SHA256=" + ("b" * 64) in command
    assert "--provenance=mode=max" in command
    assert "--sbom=true" in command
    annotation = command[command.index("--annotation") + 1]
    assert annotation == (
        "index:org.opencontainers.image.description="
        "Validated CT body-composition research pipeline"
    )
    assert request["oci_description"] == "Validated CT body-composition research pipeline"


def test_container_build_request_rejects_dirty_release_source(monkeypatch):
    monkeypatch.setattr(build_container, "_git", lambda *arguments: " M Dockerfile")
    args = SimpleNamespace(
        allow_dirty=False,
        platform="linux/amd64",
        push=False,
        load=True,
        tag=None,
        metadata_file="dist/metadata.json",
        receipt="dist/request.json",
        no_pull=True,
        no_cache=False,
        no_attestations=False,
    )
    with pytest.raises(RuntimeError, match="clean source tree"):
        build_container.build_request(args)


def test_container_checks_write_auditable_receipts(tmp_path, monkeypatch):
    image_audit = {
        "passed": True,
        "image_id": "sha256:" + ("a" * 64),
        "architecture": "amd64",
        "os": "linux",
        "size": 123,
    }
    monkeypatch.setattr(container_checks, "audit", lambda image: image_audit)

    def fake_run(command):
        if "version" in command:
            return SimpleNamespace(returncode=0, stdout="Docker Scout 1.0\n", stderr="")
        if "--output" in command:
            output = Path(command[command.index("--output") + 1])
            output.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(container_checks, "_run", fake_run)

    result = container_checks.check("bodycomposition:test", tmp_path)

    assert result["passed"]
    assert all(result["checks"].values())
    assert set(result["outputs"]) == {"image_audit", "sbom", "vulnerabilities"}
    assert (tmp_path / "container-checks.json").is_file()


def test_installed_container_source_state_uses_embedded_dirty_label(monkeypatch):
    monkeypatch.setattr(provenance, "_run_git", lambda *args: None)
    monkeypatch.setenv("BODYCOMPOSITION_GIT_SHA", "a" * 40)
    monkeypatch.setenv("BODYCOMPOSITION_SOURCE_SHA256", "b" * 64)
    monkeypatch.setenv("BODYCOMPOSITION_SOURCE_DIRTY", "true")

    state = provenance.source_state()

    assert state["git_commit"] == "a" * 40
    assert state["source_tree_sha256"] == "b" * 64
    assert state["source_dirty"] is True

    monkeypatch.setenv("BODYCOMPOSITION_SOURCE_DIRTY", "not-a-boolean")
    with pytest.raises(RuntimeError, match="must be true or false"):
        provenance.source_state()


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
        *Path("BodyComposition").rglob("*.json"),
        *Path("config").rglob("*.yaml"),
        *Path("scripts").rglob("*.py"),
        *Path("tests").rglob("*.py"),
        *Path("docs").glob("*.md"),
        *Path(".github").rglob("*.yml"),
        Path(".dockerignore"),
        Path(".gitignore"),
        Path("Dockerfile"),
        Path("MANIFEST.in"),
        Path("pyproject.toml"),
        Path("README.md"),
        Path("SECURITY.md"),
        Path("THIRD_PARTY_NOTICES.md"),
    ]
    task_id = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    )
    private_path = re.compile(
        r"/(?:Users/[^/\s]+|home/(?!bodycomposition(?:/|\b))[^/\s]+|"
        r"data[0-9]*/[^/\s]+)/"
    )
    credential_assignment = re.compile(
        r"(?:password|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]+",
        flags=re.IGNORECASE,
    )
    forbidden = {f"wp{index:02d}" for index in range(1, 11)}
    forbidden.update(
        {
            "vibe" + "coding",
            "chat" + "gpt",
            "co" + "dex",
            "gpt-" + "5.6",
            "work" + "package",
        }
    )
    allowed_notice = "Parts of this code were implemented using Codex and GPT-5.6."
    token_shape = re.compile(
        r"\b(?:"
        r"(?:AKIA|ASIA)[0-9A-Z]{16}|"
        r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}|"
        r"github_pat_[A-Za-z0-9_]{20,}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|"
        r"hf_[A-Za-z0-9]{20,}|"
        r"xox[baprs]-[A-Za-z0-9-]{10,}|"
        r"AIza[0-9A-Za-z_-]{30,}"
        r")\b"
    )
    findings = {}
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        searchable = text.replace(allowed_notice, "").lower()
        matched = sorted(token for token in forbidden if token in searchable)
        if task_id.search(text):
            matched.append("task_id")
        if private_path.search(text):
            matched.append("private_path")
        if credential_assignment.search(text):
            matched.append("credential_assignment")
        if token_shape.search(text):
            matched.append("token_shape")
        if matched:
            findings[path.as_posix()] = matched
    assert findings == {}


def test_local_development_material_is_ignored():
    gitignore = Path(".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/dev/" in gitignore


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
