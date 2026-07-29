#!/usr/bin/env python3
"""Audit built wheel/sdist contents for release-forbidden material."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

FORBIDDEN_SUFFIXES = (
    ".pt", ".pth", ".ckpt", ".onnx", ".nii", ".nii.gz", ".dcm",
)
FORBIDDEN_PARTS = {
    ".env",
    ".git",
    ".DS_Store",
    "__pycache__",
    "data",
    "dev",
    "internal",
    "logs",
    "models",
    "planning",
    "private",
    "tmp",
}
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "provider_token": re.compile(
        r"\b(?:"
        r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}|"
        r"github_pat_[A-Za-z0-9_]{20,}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|"
        r"hf_[A-Za-z0-9]{20,}|"
        r"xox[baprs]-[A-Za-z0-9-]{10,}|"
        r"AIza[0-9A-Za-z_-]{30,}"
        r")\b"
    ),
    "credential_assignment": re.compile(
        r"(?i)(?:password|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]+"
    ),
    "institutional_path": re.compile(r"/(?:Users|home|data[0-9]*)/[^/\s]+/"),
}
MAX_MEMBER_BYTES = 10 * 1024 * 1024


def _members(path: Path) -> list[tuple[str, int, bytes]]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return [
                (item.filename, item.file_size, archive.read(item))
                for item in archive.infolist()
                if not item.is_dir()
            ]
    if path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as archive:
            result = []
            for item in archive.getmembers():
                if not item.isfile():
                    continue
                handle = archive.extractfile(item)
                result.append((item.name, item.size, handle.read() if handle else b""))
            return result
    raise ValueError(f"Unsupported distribution artifact: {path}")


def audit(path: Path) -> dict:
    findings: list[dict[str, str]] = []
    members = _members(path)
    names = []
    for raw_name, size, content in members:
        name = PurePosixPath(raw_name)
        names.append(name.as_posix())
        if name.is_absolute() or ".." in name.parts:
            findings.append({"code": "unsafe_archive_path", "member": raw_name})
        if any(part in FORBIDDEN_PARTS for part in name.parts):
            findings.append({"code": "forbidden_path", "member": raw_name})
        lower = name.as_posix().lower()
        if lower.endswith(FORBIDDEN_SUFFIXES):
            findings.append({"code": "model_or_image_asset", "member": raw_name})
        if size > MAX_MEMBER_BYTES:
            findings.append({"code": "unexpected_large_file", "member": raw_name})
        if size > 2 * 1024 * 1024 or b"\x00" in content[:4096]:
            continue
        text = content.decode("utf-8", errors="ignore")
        for code, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append({"code": code, "member": raw_name})
    required_suffixes = (
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "case_manifest.schema.json",
        "run_manifest.schema.json",
    )
    for required in required_suffixes:
        if not any(name.endswith(required) for name in names):
            findings.append({"code": "required_file_missing", "member": required})
    return {
        "artifact": str(path),
        "member_count": len(members),
        "findings": findings,
        "passed": not findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"schema_version": "1.0.0", "artifacts": [audit(path) for path in args.artifacts]}
    result["passed"] = all(item["passed"] for item in result["artifacts"])
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
