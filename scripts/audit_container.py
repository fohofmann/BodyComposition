#!/usr/bin/env python3
"""Audit a local runtime image for provenance and prohibited assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import PurePosixPath
from typing import BinaryIO

FORBIDDEN_SUFFIXES = (".pt", ".ckpt", ".onnx", ".nii", ".nii.gz", ".nrrd", ".dcm")
TOKEN_SHAPE = re.compile(
    rb"(?<![A-Za-z0-9_-])(?:"
    rb"(?:AKIA|ASIA)[0-9A-Z]{16}|"
    rb"sk-(?:(?:proj-|svcacct-)[A-Za-z0-9_-]{20,}|[A-Za-z0-9]{32,})|"
    rb"github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    rb"hf_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    rb"AIza[0-9A-Za-z_-]{30,}"
    rb")(?![A-Za-z0-9_-])"
)
SENSITIVE_ENV_NAME = re.compile(
    r"(?:^|_)(?:PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|ACCESS_KEY)(?:_|$)",
    flags=re.IGNORECASE,
)
REQUIRED_LABELS = {
    "org.opencontainers.image.title": "BodyComposition",
    "org.opencontainers.image.description": "Validated CT body-composition research pipeline",
    "org.opencontainers.image.licenses": "Apache-2.0",
    "org.bodycomposition.model-weights": "not-included",
    "org.bodycomposition.cuda-runtime": "13.0",
}
REQUIRED_VOLUMES = {"/input", "/models", "/output"}


def _forbidden_asset(name: str, size: int) -> bool:
    lowered = name.lower()
    return lowered.endswith(FORBIDDEN_SUFFIXES) or (
        lowered.endswith(".pth") and size > 1024
    )


def _contains_credential(payload: bytes) -> bool:
    if b"\0" in payload:
        return False
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return TOKEN_SHAPE.search(payload) is not None


def _scan_layer(fileobj: BinaryIO, layer_name: str) -> tuple[int, int, list[dict[str, object]]]:
    files = 0
    bytes_scanned = 0
    findings: list[dict[str, object]] = []
    with tarfile.open(fileobj=fileobj, mode="r|*") as layer:
        for member in layer:
            normalized = PurePosixPath(member.name.lstrip("./")).as_posix()
            if member.issym() or member.islnk():
                if _forbidden_asset(normalized, 1025):
                    findings.append(
                        {
                            "code": "prohibited_asset_link",
                            "layer": layer_name,
                            "path": normalized,
                            "target": member.linkname,
                        }
                    )
                continue
            if not member.isfile():
                continue
            files += 1
            bytes_scanned += member.size
            if _forbidden_asset(normalized, member.size):
                findings.append(
                    {
                        "code": "prohibited_asset",
                        "layer": layer_name,
                        "path": normalized,
                        "size": member.size,
                    }
                )
            if member.size > 16 * 1024 * 1024:
                continue
            payload = layer.extractfile(member)
            if payload is not None and _contains_credential(payload.read()):
                findings.append(
                    {
                        "code": "credential_pattern",
                        "layer": layer_name,
                        "path": normalized,
                        "size": member.size,
                    }
                )
    return files, bytes_scanned, findings


def _scan_image_archive(
    fileobj: BinaryIO,
) -> tuple[int, int, int, list[dict[str, object]]]:
    layer_count = 0
    file_count = 0
    bytes_scanned = 0
    findings: list[dict[str, object]] = []
    with tarfile.open(fileobj=fileobj, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            legacy_layer = member.name.endswith("/layer.tar")
            oci_blob = member.name.startswith("blobs/sha256/")
            if not (legacy_layer or oci_blob):
                continue
            payload = archive.extractfile(member)
            if payload is None:
                continue
            try:
                files, size, layer_findings = _scan_layer(payload, member.name)
            except tarfile.ReadError:
                if legacy_layer:
                    raise
                continue
            layer_count += 1
            file_count += files
            bytes_scanned += size
            findings.extend(layer_findings)
    return layer_count, file_count, bytes_scanned, findings


def _inspect(image: str) -> dict[str, object]:
    completed = subprocess.run(
        ["docker", "image", "inspect", image],
        check=True,
        capture_output=True,
        text=True,
    )
    values = json.loads(completed.stdout)
    if len(values) != 1:
        raise RuntimeError(f"Expected one local image for {image!r}.")
    return values[0]


def _configuration_findings(inspect: dict[str, object]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    config = inspect.get("Config") or {}
    if not isinstance(config, dict):
        return [{"code": "invalid_image_config"}]

    user = config.get("User")
    if user != "10001:10001":
        findings.append({"code": "unexpected_user", "actual": user})

    volumes = config.get("Volumes") or {}
    volume_names = set(volumes) if isinstance(volumes, dict) else set()
    if volume_names != REQUIRED_VOLUMES:
        findings.append(
            {
                "code": "unexpected_volumes",
                "actual": sorted(volume_names),
                "expected": sorted(REQUIRED_VOLUMES),
            }
        )

    labels = config.get("Labels") or {}
    if not isinstance(labels, dict):
        labels = {}
    for name, expected in REQUIRED_LABELS.items():
        if labels.get(name) != expected:
            findings.append(
                {
                    "code": "missing_or_invalid_label",
                    "label": name,
                    "actual": labels.get(name),
                    "expected": expected,
                }
            )
    for name, length in (
        ("org.opencontainers.image.revision", 40),
        ("org.bodycomposition.uv-lock-sha256", 64),
        ("org.bodycomposition.source-sha256", 64),
    ):
        value = labels.get(name)
        if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
            findings.append({"code": "invalid_digest_label", "label": name, "actual": value})
    if labels.get("org.bodycomposition.source-dirty") not in {"true", "false"}:
        findings.append(
            {
                "code": "invalid_source_dirty_label",
                "actual": labels.get("org.bodycomposition.source-dirty"),
            }
        )

    for item in config.get("Env") or []:
        name, _, value = str(item).partition("=")
        if SENSITIVE_ENV_NAME.search(name):
            findings.append({"code": "sensitive_environment_name", "name": name})
        if TOKEN_SHAPE.search(value.encode("utf-8", errors="ignore")):
            findings.append({"code": "credential_pattern_in_environment", "name": name})
    return findings


def audit(image: str) -> dict[str, object]:
    inspect = _inspect(image)
    process = subprocess.Popen(
        ["docker", "image", "save", image],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    findings = _configuration_findings(inspect)
    try:
        layer_count, file_count, bytes_scanned, layer_findings = _scan_image_archive(
            process.stdout
        )
        findings.extend(layer_findings)
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(
            returncode,
            process.args,
            stderr=stderr,
        )
    if layer_count == 0:
        findings.append({"code": "no_layers_scanned"})

    image_id = str(inspect.get("Id") or "")
    digest = hashlib.sha256(image_id.encode("utf-8")).hexdigest()
    return {
        "schema_version": "1.0.0",
        "image": image,
        "image_id": image_id,
        "image_id_receipt_sha256": digest,
        "architecture": inspect.get("Architecture"),
        "os": inspect.get("Os"),
        "size": inspect.get("Size"),
        "layers_scanned": layer_count,
        "files_scanned": file_count,
        "uncompressed_file_bytes_scanned": bytes_scanned,
        "findings": findings,
        "passed": not findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(args.image)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        from pathlib import Path

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
