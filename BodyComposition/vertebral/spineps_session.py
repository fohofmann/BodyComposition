"""Lazy, process-local loading of a verified SPINEPS/VERIDAH model bundle."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from BodyComposition.vertebral.spineps_assets import verify_installed_asset
from BodyComposition.vertebral.spineps_manifest import (
    MODEL_WEIGHT_REDISTRIBUTION_MODE,
    SPINEPS_MODEL_ASSETS,
    SPINEPS_SOURCE_COMMIT,
    SPINEPS_VERSION,
    SPINEPS_WHEEL_SHA256,
    TPTBOX_COMPAT_VERSION,
    TPTBOX_LICENSE_STATUS,
    VIBESEG_CROP_ASSETS,
    VIBESEG_CROP_DATASET_ID,
    VIBESEG_CROP_RELEASE,
    VIBESEG_WEIGHT_LICENSE_STATUS,
)


class SpinepsDependencyError(RuntimeError):
    """Raised when the pinned SPINEPS runtime is unavailable."""


@dataclass(frozen=True)
class LoadedSpinepsModels:
    semantic: Any
    instance: Any
    labeling: Any


def _load_actual_model(path: Path, use_cpu: bool) -> Any:
    try:
        from spineps.get_models import get_actual_model
    except ImportError as error:
        raise SpinepsDependencyError(
            "SPINEPS 2.0.0 is required for the spineps_veridah_ct_v1 backend."
        ) from error
    return get_actual_model(path, use_cpu=use_cpu).load()


class SpinepsModelSession:
    """Load each pinned model once and never invoke upstream auto-download."""

    def __init__(
        self,
        model_root: Path,
        *,
        use_cpu: bool = False,
        loader: Callable[[Path, bool], Any] = _load_actual_model,
    ) -> None:
        self.model_root = Path(model_root)
        self.use_cpu = use_cpu
        self._loader = loader
        self._models: LoadedSpinepsModels | None = None
        self._lock = threading.Lock()

    def load(self) -> LoadedSpinepsModels:
        if self._models is not None:
            return self._models
        with self._lock:
            if self._models is not None:
                return self._models
            installed = {
                # Full inventories are hashed during synchronization/release
                # validation. First inference uses the pinned manifest and
                # cheap structural checks to avoid rehashing multi-GB models.
                asset.phase: verify_installed_asset(self.model_root, asset, full=False)
                for asset in SPINEPS_MODEL_ASSETS
            }
            self._models = LoadedSpinepsModels(
                semantic=self._loader(installed["semantic"], self.use_cpu),
                instance=self._loader(installed["instance"], self.use_cpu),
                labeling=self._loader(installed["labeling"], self.use_cpu),
            )
        return self._models

    def unload(self) -> None:
        """Drop the process-local upstream bundle without touching model files."""

        with self._lock:
            self._models = None

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "spineps_version": SPINEPS_VERSION,
            "spineps_source_commit": SPINEPS_SOURCE_COMMIT,
            "spineps_wheel_sha256": SPINEPS_WHEEL_SHA256,
            "tptbox_compat_version": TPTBOX_COMPAT_VERSION,
            "tptbox_license_status": TPTBOX_LICENSE_STATUS,
            "models": {
                asset.model_id: {
                    "release": asset.release,
                    "asset": asset.asset_name,
                    "sha256": asset.sha256,
                    "redistribution_mode": MODEL_WEIGHT_REDISTRIBUTION_MODE,
                }
                for asset in SPINEPS_MODEL_ASSETS
            },
            "required_crop_model": {
                "dataset_id": VIBESEG_CROP_DATASET_ID,
                "release": VIBESEG_CROP_RELEASE,
                "weight_license_status": VIBESEG_WEIGHT_LICENSE_STATUS,
                "redistribution_mode": MODEL_WEIGHT_REDISTRIBUTION_MODE,
                "archives": [
                    {
                        "asset": asset.asset_name,
                        "sha256": asset.sha256,
                        "bytes": asset.bytes,
                    }
                    for asset in VIBESEG_CROP_ASSETS
                ],
            },
        }
