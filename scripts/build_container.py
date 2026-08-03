#!/usr/bin/env python3
"""Build a provenance-labelled BodyComposition image with Docker Buildx."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform as host_platform
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_FILES = (
    ".dockerignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "Dockerfile",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "scripts/sanitize_container_environment.py",
    "uv.lock",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context_paths() -> tuple[Path, ...]:
    paths = [ROOT / relative for relative in CONTEXT_FILES]
    paths.extend(
        path
        for path in (ROOT / "BodyComposition").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Docker build-context file is missing: {missing[0]}")
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("Docker build-context source files must not be symbolic links.")
    return tuple(sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix()))


def source_context_sha256() -> str:
    digest = hashlib.sha256()
    for path in _context_paths():
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _default_platform() -> str:
    machine = host_platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "linux/arm64"
    if machine in {"amd64", "x86_64"}:
        return "linux/amd64"
    raise RuntimeError(f"No default Linux container platform for host architecture {machine!r}.")


def _platforms(value: str | None) -> tuple[str, ...]:
    items = tuple(item.strip() for item in (value or _default_platform()).split(","))
    if not items or any(item not in {"linux/amd64", "linux/arm64"} for item in items):
        raise ValueError("--platform supports linux/amd64 and linux/arm64.")
    if len(set(items)) != len(items):
        raise ValueError("--platform must not repeat a target.")
    return items


def build_request(args: argparse.Namespace) -> dict[str, object]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = str(project["project"]["version"])
    commit = _git("rev-parse", "HEAD").lower()
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status)
    if dirty and not args.allow_dirty:
        raise RuntimeError("Container release builds require a clean source tree.")

    source_digest = source_context_sha256()
    lock_digest = _sha256(ROOT / "uv.lock")
    source_date_epoch = _git("show", "-s", "--format=%ct", "HEAD")
    platforms = _platforms(args.platform)
    push = bool(args.push)
    load = bool(args.load)
    if push and load:
        raise ValueError("--push and --load are mutually exclusive.")
    if len(platforms) > 1 and not push:
        raise ValueError("Multi-platform builds must use --push.")
    if not push and not load:
        load = True

    default_tag = f"bodycomposition:{version}-{commit[:7]}"
    if dirty:
        default_tag += f"-dirty-{source_digest[:8]}"
    tags = tuple(args.tag or (default_tag,))

    metadata_file = Path(args.metadata_file)
    if not metadata_file.is_absolute():
        metadata_file = ROOT / metadata_file
    receipt_file = Path(args.receipt)
    if not receipt_file.is_absolute():
        receipt_file = ROOT / receipt_file

    command = [
        "docker",
        "buildx",
        "build",
        "--platform",
        ",".join(platforms),
        "--build-arg",
        f"VERSION={version}",
        "--build-arg",
        f"GIT_SHA={commit}",
        "--build-arg",
        f"LOCK_SHA256={lock_digest}",
        "--build-arg",
        f"SOURCE_SHA256={source_digest}",
        "--build-arg",
        f"SOURCE_DATE_EPOCH={source_date_epoch}",
        "--build-arg",
        f"SOURCE_DIRTY={'true' if dirty else 'false'}",
        "--metadata-file",
        str(metadata_file),
    ]
    if not args.no_pull:
        command.append("--pull")
    if args.no_cache:
        command.append("--no-cache")
    for tag in tags:
        command.extend(("--tag", tag))
    if push:
        command.append("--push")
        if not args.no_attestations:
            command.extend(("--provenance=mode=max", "--sbom=true"))
    elif load:
        command.append("--load")
    command.append(str(ROOT))

    return {
        "schema_version": "1.0.0",
        "version": version,
        "git_commit": commit,
        "source_dirty": dirty,
        "source_context_sha256": source_digest,
        "package_lock_sha256": lock_digest,
        "source_date_epoch": int(source_date_epoch),
        "platforms": list(platforms),
        "tags": list(tags),
        "push": push,
        "load": load,
        "attestations": push and not args.no_attestations,
        "metadata_file": str(metadata_file),
        "receipt_file": str(receipt_file),
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", action="append", help="image tag; may be repeated")
    parser.add_argument(
        "--platform",
        help="comma-separated target(s): linux/amd64 and/or linux/arm64",
    )
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--no-attestations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--metadata-file",
        default="dist/container-buildkit-metadata.json",
    )
    parser.add_argument("--receipt", default="dist/container-build-request.json")
    args = parser.parse_args()

    request = build_request(args)
    receipt = Path(str(request["receipt_file"]))
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(request, indent=2, sort_keys=True))
        return 0
    Path(str(request["metadata_file"])).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(list(request["command"]), cwd=ROOT, check=True)
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
