#!/usr/bin/env python3
"""Remove known dependency test assets and reject unexpected runtime assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FORBIDDEN_SUFFIXES = (".pt", ".ckpt", ".onnx", ".nii", ".nii.gz", ".nrrd", ".dcm")
KNOWN_MEDICAL_ASSET_ROOTS = (
    "TPTBox/tests/sample_ct",
    "TPTBox/tests/sample_mri",
    "nibabel/nicom/tests/data",
    "nibabel/tests/data",
    "pydicom/data",
)
KNOWN_WEIGHT_ASSETS = (
    "torchmetrics/functional/image/dists_models/weights.pt",
    "torchmetrics/functional/image/lpips_models/alex.pth",
    "torchmetrics/functional/image/lpips_models/squeeze.pth",
    "torchmetrics/functional/image/lpips_models/vgg.pth",
)


def _site_packages(environment: Path) -> Path:
    candidates = sorted(environment.glob("lib/python*/site-packages"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one site-packages directory below {environment}, found {len(candidates)}."
        )
    return candidates[0]


def _has_forbidden_suffix(path: Path) -> bool:
    return path.name.lower().endswith(FORBIDDEN_SUFFIXES)


def _unexpected_assets(environment: Path) -> list[dict[str, int | str]]:
    findings: list[dict[str, int | str]] = []
    for path in sorted(environment.rglob("*")):
        if not (path.is_file() or path.is_symlink()):
            continue
        size = path.lstat().st_size
        forbidden = _has_forbidden_suffix(path)
        large_pth = path.name.lower().endswith(".pth") and size > 1024
        if forbidden or large_pth:
            findings.append(
                {
                    "path": path.relative_to(environment).as_posix(),
                    "size": size,
                }
            )
    return findings


def sanitize(environment: Path) -> dict[str, object]:
    environment = environment.resolve()
    site_packages = _site_packages(environment)
    removed: list[str] = []

    for relative_root in KNOWN_MEDICAL_ASSET_ROOTS:
        root = site_packages / relative_root
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and _has_forbidden_suffix(path):
                path.unlink()
                removed.append(path.relative_to(environment).as_posix())

    for relative_path in KNOWN_WEIGHT_ASSETS:
        path = site_packages / relative_path
        if path.is_file():
            path.unlink()
            removed.append(path.relative_to(environment).as_posix())

    findings = _unexpected_assets(environment)
    result = {
        "environment": environment.as_posix(),
        "removed_count": len(removed),
        "removed": sorted(removed),
        "unexpected_assets": findings,
        "passed": not findings,
    }
    if findings:
        paths = ", ".join(item["path"] for item in findings[:10])
        raise RuntimeError(f"Unexpected model or medical-image assets remain: {paths}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("environment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = sanitize(args.environment)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
