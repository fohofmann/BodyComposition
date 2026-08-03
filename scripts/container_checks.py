#!/usr/bin/env python3
"""Create auditable SBOM, vulnerability, and image-policy receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

if __package__:
    from .audit_container import audit
else:
    from audit_container import audit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True)


def check(image: str, output_directory: Path) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    audit_path = output_directory / "container-image-audit.json"
    sbom_path = output_directory / "container-sbom.cdx.json"
    cve_path = output_directory / "container-vulnerabilities.sarif.json"

    image_audit = audit(image)
    audit_path.write_text(
        json.dumps(image_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    scout_version = _run(["docker", "scout", "version"])
    sbom = _run(
        [
            "docker",
            "scout",
            "sbom",
            "--format",
            "cyclonedx",
            "--output",
            str(sbom_path),
            f"local://{image}",
        ]
    )
    vulnerabilities = _run(
        [
            "docker",
            "scout",
            "cves",
            "--format",
            "sarif",
            "--output",
            str(cve_path),
            f"local://{image}",
        ]
    )
    fixable_gate = _run(
        [
            "docker",
            "scout",
            "cves",
            "--only-severity",
            "critical,high",
            "--only-fixed",
            "--exit-code",
            f"local://{image}",
        ]
    )

    outputs = {}
    for name, path in (
        ("image_audit", audit_path),
        ("sbom", sbom_path),
        ("vulnerabilities", cve_path),
    ):
        if path.is_file():
            outputs[name] = {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }

    checks = {
        "image_policy": bool(image_audit["passed"]),
        "scout_available": scout_version.returncode == 0,
        "sbom_generated": sbom.returncode == 0 and sbom_path.is_file(),
        "vulnerability_report_generated": (
            vulnerabilities.returncode == 0 and cve_path.is_file()
        ),
        "no_fixable_high_or_critical_vulnerabilities": fixable_gate.returncode == 0,
    }
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "image": image,
        "image_id": image_audit["image_id"],
        "architecture": image_audit["architecture"],
        "os": image_audit["os"],
        "size": image_audit["size"],
        "docker_scout_version": scout_version.stdout.strip(),
        "checks": checks,
        "outputs": outputs,
        "commands": {
            "sbom_returncode": sbom.returncode,
            "sbom_stderr": sbom.stderr.strip(),
            "vulnerability_report_returncode": vulnerabilities.returncode,
            "vulnerability_report_stderr": vulnerabilities.stderr.strip(),
            "fixable_gate_returncode": fixable_gate.returncode,
            "fixable_gate_output": (fixable_gate.stdout + fixable_gate.stderr).strip(),
        },
        "passed": all(checks.values()),
    }
    receipt = output_directory / "container-checks.json"
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--output-directory", type=Path, default=Path("dist/container"))
    args = parser.parse_args()
    result = check(args.image, args.output_directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
