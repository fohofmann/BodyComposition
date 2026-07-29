"""Single automation-friendly command line for BodyComposition 1.x."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stdout, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from BodyComposition import __version__
from BodyComposition.config import ConfigError, PipelineConfig, low_resource_config
from BodyComposition.dicom import convert_dicom
from BodyComposition.model_manager import (
    MODEL_IDS,
    ModelAssetError,
    list_models,
    release_model_issues,
    sync_models,
    verify_models,
)
from BodyComposition.provenance import package_lock_digest, source_state
from BodyComposition.results import ExecutionStatus
from BodyComposition.service import (
    DEFAULT_OUTPUT_ROOT,
    DirtySourceError,
    PipelineService,
    aggregate_results,
    collate_report_export,
    export_result_csv,
    inspect_report,
    inspect_result,
    load_batch_manifest,
    render_report,
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ENVIRONMENT = 3
EXIT_EXECUTION = 4

_JSON_OUTPUT_STREAM: ContextVar[TextIO | None] = ContextVar(
    "bodycomposition_json_output_stream",
    default=None,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return str(value)


def _emit(value: Any, *, json_output: bool, human: str | None = None) -> None:
    if json_output:
        print(
            json.dumps(value, indent=2, sort_keys=True, default=_json_default),
            file=_JSON_OUTPUT_STREAM.get() or sys.stdout,
        )
    elif human is not None:
        print(human)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            print(f"{key}: {_json_default(item)}")
    else:
        print(value)


@contextmanager
def _machine_readable_stdout(enabled: bool) -> Iterator[None]:
    """Reserve stdout for the CLI JSON payload while dependencies run."""

    if not enabled:
        yield
        return
    token = _JSON_OUTPUT_STREAM.set(sys.stdout)
    try:
        with redirect_stdout(sys.stderr):
            yield
    finally:
        _JSON_OUTPUT_STREAM.reset(token)


def _config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig.load(getattr(args, "config", None))
    if getattr(args, "low_resource", False):
        config = low_resource_config(config)
    device = getattr(args, "device", None)
    if device is not None:
        value = config.normalized()
        value["runtime"]["device"] = device
        config = PipelineConfig.model_validate(value)
    if getattr(args, "no_csv", False):
        value = config.normalized()
        value["output"]["save_csv_tables"] = False
        config = PipelineConfig.model_validate(value)
    logging.getLogger().setLevel(config.normalized()["output"]["log_level"])
    return config


def _report_result(value: Any) -> dict[str, Any]:
    return {
        "case_id": value.case_id,
        "report_id": value.report_id,
        "layout": value.layout,
        "status": value.status,
        "pdf_path": value.pdf_path,
        "manifest_path": value.manifest_path,
        "pdf_sha256": value.pdf_sha256,
        "manual_review_required": value.manual_review_required,
        "warnings": list(value.warnings),
    }


def _cmd_analyze(args: argparse.Namespace) -> int:
    result = PipelineService(_config(args)).analyze_case(
        args.input,
        args.output,
        case_id=args.case_id,
        run_id=args.run_id,
        series_uid=args.series_uid,
    )
    _emit(
        result.as_dict(),
        json_output=args.json,
        human=f"{result.execution_status.value}: {result.manifest_path}",
    )
    return EXIT_OK if result.succeeded else EXIT_EXECUTION


def _cmd_convert(args: argparse.Namespace) -> int:
    result = convert_dicom(
        args.input,
        args.output,
        series_uid=args.series_uid,
        overwrite=args.overwrite,
    )
    _emit(
        result.as_dict(),
        json_output=args.json,
        human=f"converted: {result.output_path}\nmetadata: {result.metadata_path}",
    )
    return EXIT_OK


def _cmd_batch(args: argparse.Namespace) -> int:
    cases = load_batch_manifest(args.manifest)
    result = PipelineService(_config(args)).analyze_batch(
        cases,
        args.output,
        run_id=args.run_id,
        worker_mode=args.worker,
    )
    human = (
        f"worker drained; shared run continues: {result.output_path}"
        if result.execution_status == ExecutionStatus.RUNNING
        else f"{result.execution_status.value}: {result.manifest_path}"
    )
    _emit(
        result.as_dict(),
        json_output=args.json,
        human=human,
    )
    return (
        EXIT_OK
        if result.execution_status in {ExecutionStatus.RUNNING, ExecutionStatus.SUCCEEDED}
        else EXIT_EXECUTION
    )


def _cmd_models_list(args: argparse.Namespace) -> int:
    assets = list(list_models())
    _emit(assets, json_output=args.json, human="\n".join(item["model_id"] for item in assets))
    return EXIT_OK


def _selected_models(args: argparse.Namespace) -> Sequence[str] | None:
    return tuple(args.model) if args.model else None


def _cmd_models_verify(args: argparse.Namespace) -> int:
    config = _config(args)
    reports = verify_models(config, _selected_models(args))
    issues = release_model_issues(config, reports)
    payload = {
        "ready": all(report.ready for report in reports),
        "release_ready": all(report.ready for report in reports) and not issues,
        "release_issues": list(issues),
        "models": [report.as_dict() for report in reports],
    }
    _emit(
        payload,
        json_output=args.json,
        human=("models ready" if payload["ready"] else "models not ready"),
    )
    return EXIT_OK if payload["ready"] else EXIT_ENVIRONMENT


def _cmd_models_sync(args: argparse.Namespace) -> int:
    reports = sync_models(_config(args), _selected_models(args))
    payload = {"ready": True, "models": [report.as_dict() for report in reports]}
    _emit(payload, json_output=args.json, human="models synchronized and verified")
    return EXIT_OK


def _cmd_config_validate(args: argparse.Namespace) -> int:
    config = PipelineConfig.load(args.config)
    payload = {
        "valid": True,
        "configuration_sha256": config.digest(),
        "queue_configuration_sha256": config.queue_digest(),
        "scientific_configuration_sha256": config.scientific_digest(),
    }
    _emit(payload, json_output=args.json, human="valid")
    return EXIT_OK


def _cmd_config_default(args: argparse.Namespace) -> int:
    value = PipelineConfig.load().to_yaml()
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(value, encoding="utf-8")
        _emit(
            {"output": destination},
            json_output=args.json,
            human=str(destination),
        )
    elif args.json:
        _emit(PipelineConfig.load().normalized(), json_output=True)
    else:
        print(value, end="")
    return EXIT_OK


def _cmd_results_inspect(args: argparse.Namespace) -> int:
    result = inspect_result(args.path)
    _emit(result.as_dict(), json_output=args.json, human=str(result.manifest_path))
    return EXIT_OK


def _cmd_results_aggregate(args: argparse.Namespace) -> int:
    paths = aggregate_results(args.run)
    _emit(paths, json_output=args.json, human="\n".join(str(path) for path in paths.values()))
    return EXIT_OK


def _cmd_results_export_csv(args: argparse.Namespace) -> int:
    paths = export_result_csv(
        args.manifest,
        args.output,
        overwrite=args.overwrite,
    )
    payload = {
        "source_manifest": args.manifest,
        "output": args.output,
        "tables": paths,
    }
    _emit(
        payload,
        json_output=args.json,
        human="\n".join(str(path) for path in paths.values()),
    )
    return EXIT_OK


def _cmd_results_review(args: argparse.Namespace) -> int:
    run = Path(args.run)
    path = run / "aggregate" / "review_queue.parquet"
    if not path.is_file():
        path = aggregate_results(run)["review_queue"]
    table = pd.read_parquet(path)
    payload = {
        "path": path,
        "count": len(table),
        "records": table.to_dict(orient="records"),
    }
    _emit(payload, json_output=args.json, human=f"{len(table)} review record(s): {path}")
    return EXIT_OK


def _cmd_reports_render(args: argparse.Namespace) -> int:
    result = render_report(
        args.manifest,
        args.output,
        args.config,
        layout=args.layout,
    )
    _emit(
        _report_result(result),
        json_output=args.json,
        human=str(result.manifest_path),
    )
    return EXIT_OK


def _cmd_reports_collate(args: argparse.Namespace) -> int:
    result = collate_report_export(
        args.manifest,
        args.output,
        args.config,
        layout=args.layout,
    )
    payload = {
        "pdf_path": result["pdf_path"],
        "manifest_path": result["manifest_path"],
        "manifest": result["manifest"],
    }
    _emit(payload, json_output=args.json, human=str(result["manifest_path"]))
    return EXIT_OK


def _cmd_reports_inspect(args: argparse.Namespace) -> int:
    payload = inspect_report(args.manifest, pdf_path=args.pdf)
    _emit(payload, json_output=args.json, human="valid")
    return EXIT_OK


def _distribution_has_notices() -> bool:
    try:
        files = importlib.metadata.files("BodyComposition") or ()
    except importlib.metadata.PackageNotFoundError:
        return False
    names = {Path(str(value)).name for value in files}
    return {"LICENSE", "THIRD_PARTY_NOTICES.md"}.issubset(names)


def _output_check(path: str | None) -> tuple[bool, str | None]:
    if path is None:
        return True, None
    root = Path(path)
    probe = root / f".doctor-{os.getpid()}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"")
    except OSError as error:
        return False, type(error).__name__
    finally:
        with suppress(OSError):
            probe.unlink(missing_ok=True)
    return True, None


def _cmd_doctor(args: argparse.Namespace) -> int:
    config = _config(args)
    source = source_state()
    reports = verify_models(config)
    release_issues = release_model_issues(
        config,
        reports,
        include_all_selectable=True,
    )
    lock_digest = package_lock_digest()
    output_ready, output_error = _output_check(args.output)
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        }
    except ImportError:
        torch_info = {
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
        }
    operational_errors = []
    if not all(report.ready for report in reports):
        operational_errors.append("required_models_not_ready")
    if not output_ready:
        operational_errors.append("output_not_writable")
    if lock_digest is None:
        operational_errors.append("package_lock_digest_unavailable")
    if not _distribution_has_notices():
        operational_errors.append("installed_license_notices_missing")
    if source.get("source_dirty") and not config.allow_dirty:
        operational_errors.append("source_tree_dirty")
    release_errors = list(operational_errors)
    if source.get("source_dirty") and "source_tree_dirty" not in release_errors:
        release_errors.append("source_tree_dirty")
    if not source.get("git_commit"):
        release_errors.append("source_commit_unavailable")
    release_errors.extend(release_issues)
    payload = {
        "version": __version__,
        "operational_ready": not operational_errors,
        "release_ready": not release_errors,
        "operational_errors": operational_errors,
        "release_errors": release_errors,
        "configuration_sha256": config.digest(),
        "queue_configuration_sha256": config.queue_digest(),
        "scientific_configuration_sha256": config.scientific_digest(),
        "package_lock_sha256": lock_digest,
        "source": source,
        "torch": torch_info,
        "output": {"ready": output_ready, "error": output_error},
        "models": [report.as_dict() for report in reports],
    }
    if args.release:
        ready = payload["release_ready"]
        human = "release-ready" if ready else "not release-ready"
    else:
        ready = payload["operational_ready"]
        human = (
            "ready"
            if payload["release_ready"]
            else "ready for analysis; release checks remain"
        ) if ready else "not ready"
    _emit(payload, json_output=args.json, human=human)
    return EXIT_OK if ready else EXIT_ENVIRONMENT


def _cmd_version(args: argparse.Namespace) -> int:
    payload = {
        "version": __version__,
        "config_schema_version": "1.0.0",
        "case_result_schema_version": "1.0.0",
        "run_manifest_schema_version": "1.0.0",
    }
    _emit(payload, json_output=args.json, human=__version__)
    return EXIT_OK


def _leaf_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="write only machine-readable JSON to stdout",
    )


def _config_flags(
    parser: argparse.ArgumentParser,
    *,
    device: bool = False,
    low_resource: bool = False,
    csv_output: bool = False,
) -> None:
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="validated YAML override; package defaults are used when omitted",
    )
    if device:
        parser.add_argument(
            "-d",
            "--device",
            choices=("auto", "cpu", "cuda"),
            help="override the configured runtime device",
        )
    if low_resource:
        parser.add_argument(
            "--low-resource",
            action="store_true",
            help=(
                "use ResEncM, unload models between stages, and analyze only "
                "the detected L3 vertebral territory"
            ),
        )
    if csv_output:
        parser.add_argument(
            "--no-csv",
            action="store_true",
            help="omit convenience CSV mirrors; canonical Parquet tables are always retained",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bodycomposition",
        description="Validated CT body-composition research pipeline.",
        epilog=(
            "Start with: bodycomposition analyze CT.nii.gz\n"
            "DICOM also works: bodycomposition analyze DICOM_DIRECTORY\n"
            "Check setup: bodycomposition doctor"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="write machine-readable JSON")
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=__version__,
        help="show the package version and exit",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser(
        "analyze",
        help="analyze one NIfTI or DICOM CT",
        description="Run the complete validated pipeline for one 3-D NIfTI or DICOM CT.",
    )
    analyze.add_argument("input", metavar="CT", type=Path, help="NIfTI file or DICOM path")
    analyze.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_ROOT,
        type=Path,
        help=f"result root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    analyze.add_argument("--case-id", help="optional pseudonymous case identifier")
    analyze.add_argument("--run-id", help="optional stable run identifier")
    analyze.add_argument(
        "--series",
        dest="series_uid",
        help="DICOM Series Instance UID; required only when multiple CT series are present",
    )
    _config_flags(analyze, device=True, low_resource=True, csv_output=True)
    _leaf_json(analyze)
    analyze.set_defaults(handler=_cmd_analyze)

    convert = commands.add_parser(
        "convert",
        help="convert one DICOM CT series to NIfTI",
        description=(
            "Convert one DICOM CT series without changing its physical orientation. "
            "An adjacent .bodycomposition.json sidecar preserves privacy-safe technical "
            "metadata for automatic use by analyze. One CT series is selected automatically; "
            "ambiguous inputs require --series."
        ),
    )
    convert.add_argument("input", metavar="DICOM", type=Path, help="DICOM file or directory")
    convert.add_argument("output", metavar="NIFTI", type=Path, help="output .nii or .nii.gz")
    convert.add_argument(
        "--series",
        dest="series_uid",
        help="DICOM Series Instance UID; required only when multiple CT series are present",
    )
    convert.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing NIfTI and metadata pair",
    )
    _leaf_json(convert)
    convert.set_defaults(handler=_cmd_convert)

    batch = commands.add_parser("batch", help="analyze an ordered input manifest")
    batch.add_argument("manifest", metavar="MANIFEST", type=Path)
    batch.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_ROOT,
        type=Path,
        help=f"result root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    batch.add_argument("--run-id", help="optional stable run identifier")
    batch.add_argument(
        "--worker",
        action="store_true",
        help=(
            "scheduler worker mode: exit successfully when all unfinished cases "
            "are already owned by other workers"
        ),
    )
    _config_flags(batch, device=True, low_resource=True, csv_output=True)
    _leaf_json(batch)
    batch.set_defaults(handler=_cmd_batch)

    models = commands.add_parser("models", help="inspect or hydrate pinned model assets")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_list = model_commands.add_parser("list")
    _leaf_json(model_list)
    model_list.set_defaults(handler=_cmd_models_list)
    for name, handler in (("verify", _cmd_models_verify), ("sync", _cmd_models_sync)):
        child = model_commands.add_parser(name)
        _config_flags(child, low_resource=True)
        child.add_argument("-m", "--model", action="append", choices=MODEL_IDS)
        _leaf_json(child)
        child.set_defaults(handler=handler)

    config = commands.add_parser("config", help="validate or generate configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser("validate")
    validate.add_argument("config", metavar="CONFIG", type=Path)
    _leaf_json(validate)
    validate.set_defaults(handler=_cmd_config_validate)
    default = config_commands.add_parser("show-default")
    default.add_argument("-o", "--output", type=Path)
    _leaf_json(default)
    default.set_defaults(handler=_cmd_config_default)

    results = commands.add_parser("results", help="inspect and aggregate immutable results")
    result_commands = results.add_subparsers(dest="results_command", required=True)
    inspect = result_commands.add_parser("inspect")
    inspect.add_argument("path", type=Path)
    _leaf_json(inspect)
    inspect.set_defaults(handler=_cmd_results_inspect)
    aggregate = result_commands.add_parser("aggregate")
    aggregate.add_argument("run", metavar="RUN", type=Path)
    _leaf_json(aggregate)
    aggregate.set_defaults(handler=_cmd_results_aggregate)
    export_csv = result_commands.add_parser(
        "export-csv",
        help="create CSV copies from one validated immutable case result",
    )
    export_csv.add_argument("manifest", metavar="CASE_MANIFEST", type=Path)
    export_csv.add_argument("-o", "--output", required=True, type=Path)
    export_csv.add_argument(
        "--overwrite",
        action="store_true",
        help="replace differing CSV files in the selected export directory",
    )
    _leaf_json(export_csv)
    export_csv.set_defaults(handler=_cmd_results_export_csv)
    review = result_commands.add_parser("review-queue")
    review.add_argument("run", metavar="RUN", type=Path)
    _leaf_json(review)
    review.set_defaults(handler=_cmd_results_review)

    reports = commands.add_parser("reports", help="render, collate, or inspect reports")
    report_commands = reports.add_subparsers(dest="reports_command", required=True)
    for name, handler in (("render", _cmd_reports_render), ("collate", _cmd_reports_collate)):
        child = report_commands.add_parser(name)
        child.add_argument("manifest", metavar="MANIFEST", type=Path)
        child.add_argument("-o", "--output", required=True, type=Path)
        child.add_argument("-c", "--config", type=Path)
        child.add_argument("--layout", choices=("spine_overview_v1", "spine_profile_v2"))
        _leaf_json(child)
        child.set_defaults(handler=handler)
    report_inspect = report_commands.add_parser("inspect")
    report_inspect.add_argument("manifest", metavar="MANIFEST", type=Path)
    report_inspect.add_argument("--pdf", type=Path)
    _leaf_json(report_inspect)
    report_inspect.set_defaults(handler=_cmd_reports_inspect)

    doctor = commands.add_parser("doctor", help="check runtime, models, lock, and release gates")
    doctor.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_ROOT,
        type=Path,
        help=f"check that this result root is writable (default: {DEFAULT_OUTPUT_ROOT})",
    )
    doctor.add_argument(
        "--release",
        action="store_true",
        help="use full publication readiness, rather than analysis readiness, as the exit gate",
    )
    _config_flags(doctor, device=True, low_resource=True)
    _leaf_json(doctor)
    doctor.set_defaults(handler=_cmd_doctor)

    version = commands.add_parser("version", help="show package and schema versions")
    _leaf_json(version)
    version.set_defaults(handler=_cmd_version)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    with _machine_readable_stdout(args.json):
        try:
            return int(args.handler(args))
        except (ConfigError, ValueError, FileNotFoundError, FileExistsError) as error:
            payload = {
                "status": "error",
                "error_code": type(error).__name__,
                "message": str(error),
            }
            if args.json:
                _emit(payload, json_output=True)
            else:
                print(f"error: {error}", file=sys.stderr)
            return EXIT_USAGE
        except (DirtySourceError, ModelAssetError, PermissionError) as error:
            payload = {
                "status": "error",
                "error_code": type(error).__name__,
                "message": str(error),
            }
            if args.json:
                _emit(payload, json_output=True)
            else:
                print(f"model error: {error}", file=sys.stderr)
            return EXIT_ENVIRONMENT
        except Exception as error:
            payload = {
                "status": "error",
                "error_code": type(error).__name__,
                "message": str(error),
            }
            if args.json:
                _emit(payload, json_output=True)
            else:
                print(f"execution error: {error}", file=sys.stderr)
            return EXIT_EXECUTION


if __name__ == "__main__":
    raise SystemExit(main())
