from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import BodyComposition.cli as cli
from BodyComposition.config import ConfigError, PipelineConfig
from BodyComposition.model_manager import ModelAssetError, ModelStatus
from BodyComposition.results import ExecutionStatus
from BodyComposition.service import DirtySourceError
from BodyComposition.tissue_backends.boa import BOA_BACKEND_ID


def _status(tmp_path: Path, *, ready: bool = True) -> ModelStatus:
    return ModelStatus(
        model_id="ctdeeprot_2d_v1",
        ready=ready,
        model_root=tmp_path / "models",
        errors=() if ready else ("missing:net2d.pt",),
        checked_files={"net2d.pt": "a" * 64} if ready else {},
        asset={"asset_id": "ctdeeprot_2d_v1"},
    )


def test_batch_cli_preserves_ordered_service_result(monkeypatch, capsys, tmp_path):
    calls = {}
    result = SimpleNamespace(
        execution_status=ExecutionStatus.SUCCEEDED,
        manifest_path=tmp_path / "run_manifest.json",
        as_dict=lambda: {"execution_status": "succeeded", "cases": ["a", "b"]},
    )

    class Service:
        def __init__(self, config):
            calls["config"] = config

        def analyze_batch(
            self,
            cases,
            output,
            *,
            run_id,
            update,
            worker_mode,
            drain_requested=None,
            drain_reason="requested",
        ):
            calls.update(
                cases=cases,
                output=output,
                run_id=run_id,
                update=update,
                worker_mode=worker_mode,
                drain_requested=drain_requested,
                drain_reason=drain_reason,
            )
            if worker_mode:
                os.kill(os.getpid(), signal.SIGTERM)
                calls["drain_seen"] = drain_requested()
            return result

    monkeypatch.setattr(cli, "load_batch_manifest", lambda path: ("a", "b"))
    monkeypatch.setattr(cli, "PipelineService", Service)

    code = cli.main(
        [
            "batch",
            str(tmp_path / "cases.json"),
            "--output",
            str(tmp_path / "output"),
            "--run-id",
            "run-1",
            "--device",
            "cpu",
            "--json",
        ]
    )
    assert code == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["cases"] == ["a", "b"]
    assert calls["cases"] == ("a", "b")
    assert calls["output"] == tmp_path / "output"
    assert calls["run_id"] == "run-1"
    assert calls["update"] is False
    assert calls["worker_mode"] is False
    assert isinstance(calls["config"], PipelineConfig)
    assert calls["config"].device == "cpu"
    assert calls["config"].normalized()["output"]["save_csv_tables"]

    assert cli.main(["batch", "cases.json", "--no-csv", "--json"]) == cli.EXIT_OK
    capsys.readouterr()
    assert not calls["config"].normalized()["output"]["save_csv_tables"]

    assert cli.main(["batch", "cases.json", "--worker", "--json"]) == cli.EXIT_OK
    capsys.readouterr()
    assert calls["worker_mode"] is True
    assert callable(calls["drain_requested"])
    assert calls["drain_reason"] == "sigterm"
    assert calls["drain_seen"] is True

    assert (
        cli.main(
            [
                "batch",
                "cases.json",
                "--update",
                "--json",
            ]
        )
        == cli.EXIT_OK
    )
    capsys.readouterr()
    assert calls["run_id"] is None
    assert calls["update"] is True

    result.execution_status = ExecutionStatus.FAILED
    assert cli.main(["batch", "cases.json", "--json"]) == cli.EXIT_EXECUTION
    capsys.readouterr()


def test_worker_sigterm_handler_requests_drain_and_restores_previous_handler():
    previous = signal.getsignal(signal.SIGTERM)

    with cli._worker_sigterm_drain(True) as requested:
        os.kill(os.getpid(), signal.SIGTERM)
        assert requested.is_set()

    assert signal.getsignal(signal.SIGTERM) is previous


def test_model_cli_lists_verifies_and_synchronizes_selected_assets(
    monkeypatch,
    capsys,
    tmp_path,
):
    ready = _status(tmp_path)
    failed = _status(tmp_path, ready=False)
    selections = []
    monkeypatch.setattr(
        cli,
        "list_models",
        lambda: ({"model_id": "ctdeeprot_2d_v1", "name": "CTDeepRot"},),
    )

    assert cli.main(["models", "list", "--json"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)[0]["name"] == "CTDeepRot"

    def verify(_config, selected):
        selections.append(selected)
        return (ready,)

    monkeypatch.setattr(cli, "verify_models", verify)
    monkeypatch.setattr(cli, "release_model_issues", lambda *_args, **_kwargs: ())
    assert (
        cli.main(
            [
                "models",
                "verify",
                "--model",
                "ctdeeprot_2d_v1",
                "--json",
            ]
        )
        == cli.EXIT_OK
    )
    assert json.loads(capsys.readouterr().out)["release_ready"]
    assert selections == [("ctdeeprot_2d_v1",)]

    monkeypatch.setattr(cli, "verify_models", lambda *_args: (failed,))
    assert cli.main(["models", "verify", "--json"]) == cli.EXIT_ENVIRONMENT
    assert not json.loads(capsys.readouterr().out)["ready"]

    synchronized = []

    def sync(config, selected):
        synchronized.append((config.tissue_backend, selected))
        return (ready,)

    monkeypatch.setattr(cli, "sync_models", sync)
    assert cli.main(["models", "sync", "--tissue-backend", "boa", "--json"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["ready"]
    assert synchronized == [(BOA_BACKEND_ID, None)]


def test_config_cli_roundtrips_default_yaml(capsys, tmp_path):
    output = tmp_path / "nested" / "bodycomposition.yaml"
    assert cli.main(["config", "show-default", "--output", str(output), "--json"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["output"] == str(output)
    assert output.is_file()

    assert cli.main(["config", "validate", str(output), "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"]
    assert len(payload["configuration_sha256"]) == 64

    assert cli.main(["config", "show-default", "--json"]) == cli.EXIT_OK
    assert "runtime" in json.loads(capsys.readouterr().out)

    assert cli.main(["config", "show-default"]) == cli.EXIT_OK
    assert "runtime:" in capsys.readouterr().out


def test_result_cli_inspects_aggregates_and_builds_review_queue(
    monkeypatch,
    capsys,
    tmp_path,
):
    manifest = tmp_path / "case_manifest.json"
    inspected = SimpleNamespace(
        manifest_path=manifest,
        as_dict=lambda: {"case_id": "case-1", "manifest_path": manifest},
    )
    monkeypatch.setattr(cli, "inspect_result", lambda path: inspected)
    assert cli.main(["results", "inspect", str(manifest), "--json"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["case_id"] == "case-1"

    run = tmp_path / "run"
    aggregate = run / "aggregate"
    aggregate.mkdir(parents=True)
    cases = aggregate / "cases.parquet"
    review = aggregate / "review_queue.parquet"
    pd.DataFrame([{"case_id": "case-1", "reason": "orientation"}]).to_parquet(review)
    monkeypatch.setattr(
        cli,
        "aggregate_results",
        lambda path: {
            "cases": cases,
            "review_queue": review,
        },
    )

    assert cli.main(["results", "aggregate", str(run), "--json"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["review_queue"] == str(review)

    assert cli.main(["results", "review-queue", str(run), "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["records"][0]["reason"] == "orientation"

    review.unlink()
    aggregate_calls = []

    def rebuild_review(path):
        aggregate_calls.append(Path(path))
        pd.DataFrame([{"case_id": "case-2"}]).to_parquet(review)
        return {"cases": cases, "review_queue": review}

    monkeypatch.setattr(cli, "aggregate_results", rebuild_review)
    assert cli.main(["results", "review-queue", str(run), "--json"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["records"][0]["case_id"] == "case-2"
    assert aggregate_calls == [run]

    csv_paths = {"slices": tmp_path / "csv" / "slices.csv"}
    export_calls = {}

    def export_csv(manifest_path, output_path, *, overwrite=False):
        export_calls.update(
            manifest=Path(manifest_path),
            output=Path(output_path),
            overwrite=overwrite,
        )
        return csv_paths

    monkeypatch.setattr(cli, "export_result_csv", export_csv)
    assert (
        cli.main(
            [
                "results",
                "export-csv",
                str(manifest),
                "--output",
                str(tmp_path / "csv"),
                "--overwrite",
                "--json",
            ]
        )
        == cli.EXIT_OK
    )
    exported = json.loads(capsys.readouterr().out)
    assert exported["tables"]["slices"] == str(csv_paths["slices"])
    assert export_calls == {
        "manifest": manifest,
        "output": tmp_path / "csv",
        "overwrite": True,
    }


def test_report_cli_covers_render_collate_and_inspection(monkeypatch, capsys, tmp_path):
    report = SimpleNamespace(
        case_id="case-1",
        report_id="r" * 64,
        layout="spine_overview_v1",
        status="succeeded",
        pdf_path=tmp_path / "case.pdf",
        manifest_path=tmp_path / "report.json",
        pdf_sha256="a" * 64,
        manual_review_required=False,
        warnings=(),
    )
    monkeypatch.setattr(cli, "render_report", lambda *_args, **_kwargs: report)
    assert (
        cli.main(
            [
                "reports",
                "render",
                "case-input.json",
                "--output",
                str(tmp_path),
                "--layout",
                "spine_overview_v1",
                "--json",
            ]
        )
        == cli.EXIT_OK
    )
    assert json.loads(capsys.readouterr().out)["report_id"] == "r" * 64

    collated = {
        "pdf_path": tmp_path / "export.pdf",
        "manifest_path": tmp_path / "export.json",
        "manifest": {"case_count": 1},
    }
    monkeypatch.setattr(cli, "collate_report_export", lambda *_args, **_kwargs: collated)
    assert (
        cli.main(
            [
                "reports",
                "collate",
                "export-input.json",
                "--output",
                str(tmp_path),
                "--json",
            ]
        )
        == cli.EXIT_OK
    )
    assert json.loads(capsys.readouterr().out)["manifest"]["case_count"] == 1

    monkeypatch.setattr(cli, "inspect_report", lambda *_args, **_kwargs: {"valid": True})
    assert (
        cli.main(
            [
                "reports",
                "inspect",
                "report.json",
                "--pdf",
                "report.pdf",
                "--json",
            ]
        )
        == cli.EXIT_OK
    )
    assert json.loads(capsys.readouterr().out) == {"valid": True}


def test_version_and_human_emit_paths(capsys):
    assert cli.main(["version", "--json"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["config_schema_version"] == "1.0.0"

    assert cli.main(["version"]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip()


def test_json_stdout_excludes_dependency_chatter(monkeypatch, capsys):
    version_handler = cli._cmd_version

    def noisy_version(args):
        print("upstream progress")
        return version_handler(args)

    monkeypatch.setattr(cli, "_cmd_version", noisy_version)
    assert cli.main(["version", "--json"]) == cli.EXIT_OK
    captured = capsys.readouterr()
    assert json.loads(captured.out)["config_schema_version"] == "1.0.0"
    assert "upstream progress" not in captured.out
    assert "upstream progress" in captured.err


def test_json_error_stdout_excludes_dependency_chatter(monkeypatch, capsys):
    def failing_version(_args):
        print("upstream failure details")
        raise RuntimeError("inference failed")

    monkeypatch.setattr(cli, "_cmd_version", failing_version)
    assert cli.main(["version", "--json"]) == cli.EXIT_EXECUTION
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "error_code": "RuntimeError",
        "message": "inference failed",
        "status": "error",
    }
    assert "upstream failure details" not in captured.out
    assert "upstream failure details" in captured.err


def test_output_check_preserves_the_original_filesystem_error(tmp_path):
    assert cli._output_check(None) == (True, None)
    assert cli._output_check(str(tmp_path / "output")) == (True, None)

    existing_file = tmp_path / "not-a-directory"
    existing_file.write_text("blocked", encoding="utf-8")
    assert cli._output_check(str(existing_file)) == (False, "FileExistsError")


def test_distribution_notice_detection_handles_installed_and_source_trees(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "files",
        lambda _name: ["BodyComposition/__init__.py", "LICENSE", "THIRD_PARTY_NOTICES.md"],
    )
    assert cli._distribution_has_notices()

    monkeypatch.setattr(importlib.metadata, "files", lambda _name: None)
    assert not cli._distribution_has_notices()

    def missing(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "files", missing)
    assert not cli._distribution_has_notices()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ConfigError("bad config"), cli.EXIT_USAGE),
        (FileNotFoundError("missing"), cli.EXIT_USAGE),
        (DirtySourceError("dirty"), cli.EXIT_ENVIRONMENT),
        (ModelAssetError("model"), cli.EXIT_ENVIRONMENT),
        (RuntimeError("failed"), cli.EXIT_EXECUTION),
    ],
)
def test_main_maps_public_errors_to_stable_json_exit_codes(
    monkeypatch,
    capsys,
    error,
    expected,
):
    def fail(_args):
        raise error

    parser = SimpleNamespace(parse_args=lambda _argv: argparse.Namespace(handler=fail, json=True))
    monkeypatch.setattr(cli, "build_parser", lambda: parser)

    assert cli.main([]) == expected
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["message"] == str(error)
