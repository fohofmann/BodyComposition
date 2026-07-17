#!/usr/bin/env python
"""CLI for case rendering, explicit export collation, and manifest validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from BodyComposition.reporting.contracts import ReportingSettings
from BodyComposition.reporting.io import load_case_report_input, load_export_manifest
from BodyComposition.reporting.service import (
    collate_reports,
    render_case_report,
    render_failed_case_report,
    validate_report_manifest,
)


def _settings(path: str | None, layout: str | None) -> ReportingSettings:
    value = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Reporting configuration file must contain a mapping.")
        value = loaded
    section = dict(value.get("reporting", value))
    section["enabled"] = True
    if layout is not None:
        section["layout"] = layout
    return ReportingSettings.from_mapping(section)


def _render_case(args) -> int:
    settings = _settings(args.config, args.layout)
    case = load_case_report_input(args.manifest)
    result = render_case_report(case, Path(args.output), settings)
    print(result.manifest_path)
    return 0


def _render_export(args) -> int:
    settings = _settings(args.config, args.layout)
    manifest_path = Path(args.manifest)
    export_id, case_manifests = load_export_manifest(manifest_path)
    output = Path(args.output)
    results = []
    for case_manifest in case_manifests:
        case_payload = json.loads(case_manifest.read_text(encoding="utf-8"))
        case_id = str(case_payload.get("case_id", "invalid-case-id"))
        try:
            case = load_case_report_input(case_manifest)
            result = render_case_report(
                case,
                output / "cases" / case.case_id,
                settings,
            )
        except Exception as error:
            result = render_failed_case_report(
                case_id=case_id,
                analysis_id="analysis_unavailable",
                output_directory=output / "cases" / case_id,
                settings=settings,
                failure_stage="report_input_validation",
                failure_code=error.__class__.__name__,
            )
        results.append(result)
    combined = collate_reports(
        results,
        export_id=export_id,
        output_directory=output,
        settings=settings,
    )
    print(combined["manifest_path"])
    return 0


def _validate(args) -> int:
    validate_report_manifest(args.manifest, pdf_path=args.pdf)
    print("valid")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render or validate deterministic BodyComposition research/QC PDFs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, handler in (("case", _render_case), ("export", _render_export)):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", required=True, type=str)
        child.add_argument("--output", required=True, type=str)
        child.add_argument("--config", type=str)
        child.add_argument(
            "--layout",
            choices=("spine_overview_v1", "spine_profile_v2"),
        )
        child.set_defaults(handler=handler)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=str)
    validate.add_argument("--pdf", type=str)
    validate.set_defaults(handler=_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
