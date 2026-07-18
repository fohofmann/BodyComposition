#!/usr/bin/env python3
"""Generate a deterministic CycloneDX inventory from the frozen uv lock."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path


def _normalise_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _hashes(package: dict) -> list[dict[str, str]]:
    values = []
    artifacts = list(package.get("wheels", ()))
    if isinstance(package.get("sdist"), dict):
        artifacts.append(package["sdist"])
    for artifact in artifacts:
        digest = artifact.get("hash")
        if isinstance(digest, str) and digest.startswith("sha256:"):
            values.append({"alg": "SHA-256", "content": digest.split(":", 1)[1]})
    return sorted(values, key=lambda item: item["content"])


def generate(lock_path: Path) -> dict:
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    components = []
    for package in sorted(lock["package"], key=lambda item: _normalise_name(item["name"])):
        name = _normalise_name(package["name"])
        version = str(package["version"])
        purl = f"pkg:pypi/{name}@{version}"
        component = {
            "type": "library",
            "bom-ref": purl,
            "name": package["name"],
            "version": version,
            "purl": purl,
            "properties": [
                {"name": "bodycomposition:lock-scope", "value": "uv-universal"}
            ],
        }
        hashes = _hashes(package)
        if hashes:
            component["hashes"] = hashes
        components.append(component)
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "BodyComposition",
                "version": "1.0.0rc1",
            },
            "properties": [
                {"name": "bodycomposition:source", "value": "uv.lock"}
            ],
        },
        "components": components,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/bodycomposition-1.0.0rc1.cdx.json"),
    )
    args = parser.parse_args()
    payload = generate(args.lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
