from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from BodyComposition.measurement import totalsegmentator_assets as assets
from BodyComposition.model_manager import model_asset


@pytest.fixture(autouse=True)
def clear_asset_check_cache():
    assets._check_cached.cache_clear()
    yield
    assets._check_cached.cache_clear()


def _archive(directory: str, files: dict[str, bytes], *, unsafe: bool = False) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(f"{directory}/", b"")
        archive.writestr(f"{directory}/.DS_Store", b"ignored")
        for relative, content in files.items():
            archive.writestr(f"{directory}/{relative}", content)
        if unsafe:
            archive.writestr("../escaped", b"unsafe")
    return output.getvalue()


def _manifest(directory: str, payload: bytes, files: dict[str, bytes]) -> dict:
    return {
        "asset_id": "totalsegmentator-test-task999-vtest",
        "task_id": 999,
        "model_title": "TotalSegmentator-test",
        "directory": directory,
        "archive_name": f"{directory}.zip",
        "download_url": "https://github.com/wasserth/TotalSegmentator/test.zip",
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "files": {
            relative: {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for relative, content in files.items()
        },
    }


def _pin_test_asset(monkeypatch, payload: bytes, files: dict[str, bytes]):
    directory = "Dataset999_test"
    manifest = _manifest(directory, payload, files)
    monkeypatch.setattr(assets, "MEASUREMENT_MODEL_MANIFEST", {"test": manifest})
    return directory, manifest


def test_sync_repairs_partial_asset_atomically_and_is_idempotent(
    tmp_path,
    monkeypatch,
):
    files = {
        "trainer/dataset.json": b"dataset",
        "trainer/plans.json": b"plans",
        "trainer/fold_0/checkpoint_final.pth": b"checkpoint",
    }
    payload = _archive("Dataset999_test", files)
    directory, manifest = _pin_test_asset(monkeypatch, payload, files)
    target = tmp_path / directory
    target.mkdir()
    (target / "partial-download").write_bytes(b"preserve-until-ready")
    requests = []

    def opener(request):
        requests.append(request.full_url)
        return io.BytesIO(payload)

    first = assets.sync_measurement_model("test", tmp_path, opener=opener)
    second = assets.sync_measurement_model(
        "test",
        tmp_path,
        opener=lambda request: (_ for _ in ()).throw(
            AssertionError("a verified asset must not be downloaded again")
        ),
    )

    assert requests == [manifest["download_url"]]
    assert first.ready and first.install_manifest_verified
    assert second.ready and second.install_manifest_verified
    assert not (target / "partial-download").exists()
    assert not (target / ".DS_Store").exists()
    assert (target / assets.ASSET_MANIFEST_NAME).is_file()
    assert not any(path.name.startswith(f".{directory}-sync-") for path in tmp_path.iterdir())


def test_failed_sync_keeps_existing_partial_asset_untouched(tmp_path, monkeypatch):
    files = {"trainer/checkpoint_final.pth": b"checkpoint"}
    good_payload = _archive("Dataset999_test", files)
    directory, _ = _pin_test_asset(monkeypatch, good_payload, files)
    target = tmp_path / directory
    target.mkdir()
    sentinel = target / "partial-download"
    sentinel.write_bytes(b"original")

    with pytest.raises(assets.TotalSegmentatorAssetError, match="size mismatch"):
        assets.sync_measurement_model(
            "test",
            tmp_path,
            opener=lambda request: io.BytesIO(b"truncated"),
        )

    assert sentinel.read_bytes() == b"original"
    assert not any(path.name.startswith(f".{directory}-sync-") for path in tmp_path.iterdir())


def test_sync_rejects_unsafe_archive_before_promotion(tmp_path, monkeypatch):
    files = {"trainer/checkpoint_final.pth": b"checkpoint"}
    payload = _archive("Dataset999_test", files, unsafe=True)
    directory, _ = _pin_test_asset(monkeypatch, payload, files)

    with pytest.raises(assets.UnsafeModelArchiveError, match="Unsafe path"):
        assets.sync_measurement_model(
            "test",
            tmp_path,
            opener=lambda request: io.BytesIO(payload),
        )

    assert not (tmp_path / directory).exists()
    assert not (tmp_path.parent / "escaped").exists()


def test_asset_records_cover_the_shared_model_contract():
    required = {
        "asset_id",
        "upstream_project",
        "upstream_repository",
        "upstream_version",
        "upstream_release",
        "upstream_release_commit",
        "citation_doi",
        "download_url",
        "archive_byte_size",
        "archive_sha256",
        "expected_files",
        "code_license",
        "weight_license_status",
        "redistribution_mode",
        "compatibility",
        "validation_set_version",
    }
    for task in ("bodytrunk", "body_landmarks"):
        record = assets.model_asset_record(task)
        assert required <= set(record)
        assert record["redistribution_mode"] == "user_model_sync"
        assert len(record["archive_sha256"]) == 64
        assert all(len(item["sha256"]) == 64 for item in record["expected_files"])


def test_release_model_ids_route_to_the_pinned_sync_adapter():
    assert (
        model_asset("totalsegmentator_body_task299_v1")["task_id"] == 299
    )
    assert (
        model_asset("totalsegmentator_total_task297_landmarks_v1")["task_id"]
        == 297
    )
