#!/usr/bin/env python3
"""Reject study data, private residue, and credentials from the Git surface."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 10 * 1024 * 1024
MAX_TEXT_SCAN_BYTES = 3 * 1024 * 1024

FORBIDDEN_SUFFIXES = (
    ".dcm",
    ".ima",
    ".nii",
    ".nii.gz",
    ".nrrd",
    ".mha",
    ".mhd",
    ".parquet",
    ".feather",
    ".csv",
    ".tsv",
    ".xls",
    ".xlsx",
    ".sav",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".sif",
    ".img",
    ".ipynb",
    ".rmd",
    ".rdata",
    ".rds",
    ".pdf",
    ".html",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".jsonl",
    ".zip",
)
FORBIDDEN_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
FORBIDDEN_PATH_PARTS = {
    ".env",
    "build",
    "data",
    "dev",
    "downloads",
    "examples",
    "internal",
    "logs",
    "models",
    "output",
    "outputs",
    "planning",
    "private",
    "research",
    "tmp",
    "wheels",
}
FORBIDDEN_BASENAMES = {
    "bodycomposition-batch.json",
    "bodycomposition-conversion.json",
    "case_manifest.json",
    "orientation_report.json",
    "report_manifest.json",
    "run_manifest.json",
}
ALLOWED_BINARY_ASSETS = {
    PurePosixPath("docs/assets/pipeline-overview.png"),
    PurePosixPath("docs/assets/pipeline-report-example.png"),
}

SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
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
}
PRIVATE_PATH = re.compile(
    r"/(?:" + "Users" + r"/[^/\s]+|home/(?!bodycomposition(?:/|\b))[^/\s]+|"
    r"data[0-9]*/[^/\s]+)/"
)


def _git(*arguments: str, input_text: str | None = None) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        input=input_text,
    ).stdout


def _finding(code: str, path: PurePosixPath, **details: Any) -> dict[str, Any]:
    return {"code": code, "path": path.as_posix(), **details}


def audit_path(path: PurePosixPath, size: int) -> list[dict[str, Any]]:
    if path in ALLOWED_BINARY_ASSETS:
        return []
    findings: list[dict[str, Any]] = []
    lowered = path.as_posix().lower()
    lower_parts = {part.lower() for part in path.parts}
    if lower_parts & FORBIDDEN_PATH_PARTS:
        findings.append(_finding("forbidden_path", path))
    if path.name.lower() in FORBIDDEN_BASENAMES or path.name.lower().endswith(
        ".bodycomposition.json"
    ):
        findings.append(_finding("generated_manifest", path))
    if lowered.endswith(FORBIDDEN_SUFFIXES) or lowered.endswith(FORBIDDEN_IMAGE_SUFFIXES):
        findings.append(_finding("data_or_generated_asset", path))
    if size > MAX_TRACKED_BYTES:
        findings.append(_finding("unexpected_large_file", path, byte_size=size))
    return findings


def audit_content(
    path: PurePosixPath,
    content: bytes,
    *,
    current: bool,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if len(content) >= 132 and content[128:132] == b"DICM":
        findings.append(_finding("dicom_content", path))
    if len(content) >= 348 and content[344:348] in {b"n+1\x00", b"ni1\x00"}:
        findings.append(_finding("nifti_content", path))
    if len(content) >= 12 and content[4:8] in {b"n+2\x00", b"ni2\x00"}:
        findings.append(_finding("nifti_content", path))
    if len(content) > MAX_TEXT_SCAN_BYTES or b"\x00" in content[:4096]:
        return findings
    text = content.decode("utf-8", errors="ignore")
    for code, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            findings.append(_finding(code, path))
    if current and PRIVATE_PATH.search(text):
        findings.append(_finding("private_path", path))
    return findings


def _current_paths() -> list[PurePosixPath]:
    raw = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return sorted({PurePosixPath(value.decode()) for value in raw.split(b"\x00") if value})


def _history_blobs() -> list[tuple[str, int, PurePosixPath]]:
    objects = _git("rev-list", "--objects", "--all")
    checked = _git(
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize) %(rest)",
        input_text=objects,
    )
    blobs: list[tuple[str, int, PurePosixPath]] = []
    for line in checked.splitlines():
        object_id, object_type, raw_size, path = line.split(" ", 3)
        if object_type == "blob" and path:
            blobs.append((object_id, int(raw_size), PurePosixPath(path)))
    return blobs


def audit_repository(*, include_history: bool) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    current_paths = _current_paths()
    for path in current_paths:
        disk_path = ROOT / path
        if not disk_path.is_file():
            continue
        size = disk_path.stat().st_size
        findings.extend(audit_path(path, size))
        findings.extend(audit_content(path, disk_path.read_bytes(), current=True))

    history_count = 0
    if include_history:
        shallow = _git("rev-parse", "--is-shallow-repository").strip() == "true"
        if shallow:
            findings.append({"code": "history_unavailable", "path": ".git"})
        else:
            seen_content: set[str] = set()
            for object_id, size, path in _history_blobs():
                history_count += 1
                findings.extend(audit_path(path, size))
                if object_id in seen_content or size > MAX_TEXT_SCAN_BYTES:
                    continue
                seen_content.add(object_id)
                content = subprocess.run(
                    ["git", "cat-file", "blob", object_id],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                ).stdout
                findings.extend(audit_content(path, content, current=False))

    unique = {
        (item["code"], item["path"], item.get("byte_size")): item for item in findings
    }
    ordered = sorted(unique.values(), key=lambda item: (item["path"], item["code"]))
    return {
        "schema_version": "1.0.0",
        "current_file_count": len(current_paths),
        "history_blob_path_count": history_count,
        "history_checked": include_history,
        "findings": ordered,
        "passed": not ordered,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_repository(include_history=args.history)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
