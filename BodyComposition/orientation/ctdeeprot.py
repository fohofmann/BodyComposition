"""Narrow, checksum-pinned adapter for the CTDeepRot 2D model.

Projection preprocessing follows CTDeepRot at the pinned upstream commit.
Copyright (c) 2020 Jakubicek Roman; distributed under BSD-3-Clause. The full
license and attribution are retained in ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from skimage.transform import resize

from BodyComposition.orientation.rotations import (
    ROTATION_CLASSES,
    apply_projection_rotation,
    compose_class_indices,
    inverse_class_index,
    recover_unaugmented_class,
    rotation_class,
)

UPSTREAM_PROJECT = "CTDeepRot"
UPSTREAM_REPOSITORY = "https://github.com/JakubicekRoman/CTDeepRot"
UPSTREAM_COMMIT = "492114b8f9f3a7f058d4e97c0dd3643fb8d39649"
CODE_LICENSE = "BSD-3-Clause"
MODEL_CITATION_DOI = "https://doi.org/10.1007/978-3-030-64610-3_7"
CHECKPOINT_FILENAME = "net2d.pt"
CHECKPOINT_SHA256 = "a2feb521cfe49c367c4e594f38fa76ba9982ba009e114fbf15fb2aae06c06d96"
CHECKPOINT_BYTES = 44_890_653
CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/JakubicekRoman/CTDeepRot/"
    f"{UPSTREAM_COMMIT}/python/example_prediction/models/{CHECKPOINT_FILENAME}"
)
MODEL_ASSET_ID = f"ctdeeprot-2d-{UPSTREAM_COMMIT}"
DEFAULT_CHECKPOINT_PATH = (
    Path(os.environ.get("BODYCOMPOSITION_MODEL_ROOT", "~/.cache/bodycomposition/models"))
    .expanduser()
    / "CTDeepRot"
    / UPSTREAM_COMMIT
    / CHECKPOINT_FILENAME
)
WEIGHT_LICENSE_STATUS = (
    "published_in_bsd_3_clause_repository_without_separate_checkpoint_terms;"
    "redistribution_not_assumed"
)
REDISTRIBUTION_MODE = "user_model_sync"
VALIDATION_SET_VERSION = (
    "ct-org-140-cads-eee3c0a993636264fd04104f7e2a090c507c0c43-orientation-v1"
)
MODEL_INPUT_SIZE = 224
MODEL_CLASS_COUNT = 24
# CTDeepRot's training convention is not the group-theoretic identity used by
# SimpleITK.  An anatomically upright LPS image converted to model ``(y,x,z)``
# order is represented by raw class 0.  Pipeline decisions use rotations
# relative to this fixed, regression-tested reference frame.
LPS_YXZ_REFERENCE_CLASS_INDEX = 0


class ModelAssetError(RuntimeError):
    """Raised when the pinned CTDeepRot checkpoint cannot be verified."""


@dataclass(frozen=True)
class CTDeepRotPrediction:
    """Equivariance-vote result relative to an upright SimpleITK LPS frame."""

    winning_class_index: int
    winning_angles_deg: tuple[int, int, int]
    agreement_count: int
    vote_counts: tuple[int, ...]
    recovered_votes: tuple[int, ...]
    augmented_predictions: tuple[int, ...]
    vote_entropy: float
    vote_margin: float
    mean_support_probability: float
    checkpoint_sha256: str
    raw_winning_class_index: int | None = None
    raw_winning_angles_deg: tuple[int, int, int] | None = None
    raw_vote_counts: tuple[int, ...] = ()
    raw_recovered_votes: tuple[int, ...] = ()
    model_reference_class_index: int = LPS_YXZ_REFERENCE_CLASS_INDEX
    model_reference_angles_deg: tuple[int, int, int] = ROTATION_CLASSES[
        LPS_YXZ_REFERENCE_CLASS_INDEX
    ].angles_deg
    upstream_commit: str = UPSTREAM_COMMIT

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint_path(configured_path: str | Path) -> Path:
    return Path(configured_path).expanduser()


def verify_checkpoint(
    checkpoint_path: str | Path,
    expected_sha256: str = CHECKPOINT_SHA256,
    expected_bytes: int | None = CHECKPOINT_BYTES,
) -> Path:
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise ModelAssetError(
            f"CTDeepRot checkpoint not found: {path}. Run "
            "`bodycomposition models sync --model ctdeeprot_2d_v1` before inference."
        )
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ModelAssetError(
            "CTDeepRot checkpoint size mismatch: "
            f"expected {expected_bytes} bytes, observed {path.stat().st_size} at {path}."
        )
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ModelAssetError(
            "CTDeepRot checkpoint digest mismatch: "
            f"expected {expected_sha256}, observed {observed} at {path}."
        )
    return path


def sync_checkpoint(
    checkpoint_path: str | Path,
    *,
    download_url: str = CHECKPOINT_URL,
    expected_sha256: str = CHECKPOINT_SHA256,
    expected_bytes: int = CHECKPOINT_BYTES,
) -> Path:
    """Synchronize the pinned model asset with verified atomic promotion."""

    path = resolve_checkpoint_path(checkpoint_path)
    if path.is_file():
        try:
            return verify_checkpoint(path, expected_sha256, expected_bytes)
        except ModelAssetError:
            # Keep the invalid asset in place until a replacement has passed all
            # checks; the final replace remains atomic.
            pass
    elif path.exists():
        raise ModelAssetError(f"CTDeepRot checkpoint path is not a file: {path}.")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".part",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with requests.get(download_url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        verify_checkpoint(temporary_path, expected_sha256, expected_bytes)
        temporary_path.replace(path)
        path.chmod(0o644)
    except Exception as exc:
        if isinstance(exc, ModelAssetError):
            raise
        raise ModelAssetError(
            f"Unable to download the pinned CTDeepRot checkpoint from {download_url}: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return verify_checkpoint(path, expected_sha256, expected_bytes)


def model_asset_record() -> dict:
    """Return the complete pinned CTDeepRot asset and distribution contract."""

    return {
        "asset_id": MODEL_ASSET_ID,
        "name": "CTDeepRot-2D",
        "upstream_project": UPSTREAM_PROJECT,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "citation_doi": MODEL_CITATION_DOI,
        "download_url": CHECKPOINT_URL,
        "byte_size": CHECKPOINT_BYTES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "expected_files": [
            {
                "relative_path": CHECKPOINT_FILENAME,
                "byte_size": CHECKPOINT_BYTES,
                "sha256": CHECKPOINT_SHA256,
            }
        ],
        "code_license": CODE_LICENSE,
        "weight_license_status": WEIGHT_LICENSE_STATUS,
        "redistribution_mode": REDISTRIBUTION_MODE,
        "compatibility": {
            "architecture": "torchvision.resnet18_conv1_9_channels_fc_24_classes",
            "input_shape_chw": [9, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "simpleitk_array_boundary": "array_zyx_to_model_array_yxz",
        },
        "validation_set_version": VALIDATION_SET_VERSION,
    }


def image_to_model_array(image) -> np.ndarray:
    """Convert a SimpleITK image to CTDeepRot's explicit ``array_yxz`` order."""

    import SimpleITK as sitk

    if not isinstance(image, sitk.Image) or image.GetDimension() != 3:
        raise TypeError("CTDeepRot requires a three-dimensional SimpleITK image.")
    array_zyx = sitk.GetArrayFromImage(image)
    if not np.all(np.isfinite(array_zyx)):
        raise ValueError("CTDeepRot input contains non-finite pixels.")

    array_yxz = np.transpose(array_zyx, (1, 2, 0)).astype(np.float32, copy=True)
    below_air = array_yxz < -1024
    minimum = float(np.min(array_yxz))
    if minimum < 0:
        array_yxz += 1024.0
    array_yxz[below_air] = 0.0
    return array_yxz


def _prepare_projection(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Apply the upstream mat2gray-then-resize order exactly."""

    normalized = (np.clip(values, low, high) - low) / (high - low)
    resized = resize(normalized, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), order=3)
    return (resized * 255.0).astype(np.uint8)


def projection_feature_groups(array_yxz: np.ndarray) -> tuple[np.ndarray, ...]:
    """Create CTDeepRot mean, maximum, and standard-deviation triplets."""

    if array_yxz.ndim != 3 or any(length < 2 for length in array_yxz.shape):
        raise ValueError("CTDeepRot requires at least two voxels along every model axis.")
    data = np.asarray(array_yxz, dtype=np.float32)
    statistics: Sequence[
        tuple[Callable[[np.ndarray, int], np.ndarray], tuple[float, float]]
    ] = (
        (lambda values, axis: np.mean(values, axis=axis), (50.0, 1800.0)),
        (lambda values, axis: np.max(values, axis=axis), (100.0, 3200.0)),
        (lambda values, axis: np.std(values, axis=axis, ddof=1), (0.0, 1000.0)),
    )
    groups = []
    for function, window in statistics:
        projections = []
        for axis in range(3):
            projection = function(data, axis)
            projections.append(_prepare_projection(projection, *window))
        group = np.stack(projections, axis=2).astype(np.float32) / 255.0 - 0.5
        groups.append(group)
    return tuple(groups)


def augmented_feature_batch(array_yxz: np.ndarray) -> np.ndarray:
    """Return all 24 CTDeepRot training rotations as one NCHW batch."""

    groups = projection_feature_groups(array_yxz)
    samples = []
    for class_index in range(MODEL_CLASS_COUNT):
        rotated_groups = [
            apply_projection_rotation(group, class_index)
            for group in groups
        ]
        sample_hwc = np.concatenate(rotated_groups, axis=2)
        samples.append(np.transpose(sample_hwc, (2, 0, 1)))
    return np.ascontiguousarray(np.stack(samples), dtype=np.float32)


@lru_cache(maxsize=4)
def _load_model(checkpoint: str, expected_sha256: str, device_name: str):
    import torch
    from torch import nn
    from torchvision.models import resnet18

    verify_checkpoint(Path(checkpoint), expected_sha256)
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, MODEL_CLASS_COUNT)
    state = torch.load(
        checkpoint,
        map_location=torch.device(device_name),
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device_name))
    model.eval()
    return model


def release_cached_models() -> None:
    """Drop process-local CTDeepRot model references for low-memory execution."""

    _load_model.cache_clear()


class CTDeepRotPredictor:
    """Process-local CTDeepRot model with deterministic equivariance voting."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 24,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("CTDeepRot batch_size must be positive.")
        self.checkpoint_path = verify_checkpoint(
            resolve_checkpoint_path(checkpoint_path),
        )
        self.expected_sha256 = CHECKPOINT_SHA256
        self.device = self._resolve_device(device)
        self.batch_size = min(int(batch_size), MODEL_CLASS_COUNT)

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("CTDeepRot device must be 'cpu', 'cuda', or 'auto'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CTDeepRot was configured for CUDA, but CUDA is unavailable.")
        return device

    def predict(self, image) -> CTDeepRotPrediction:
        import torch

        array_yxz = image_to_model_array(image)
        feature_batch = augmented_feature_batch(array_yxz)
        model = _load_model(
            str(self.checkpoint_path.resolve()),
            self.expected_sha256,
            self.device,
        )

        probability_batches = []
        with torch.inference_mode():
            for start in range(0, MODEL_CLASS_COUNT, self.batch_size):
                tensor = torch.from_numpy(
                    feature_batch[start : start + self.batch_size]
                ).to(self.device)
                probability_batches.append(torch.softmax(model(tensor), dim=1).cpu())
        probabilities = torch.cat(probability_batches, dim=0).numpy()
        augmented_predictions = np.argmax(probabilities, axis=1).astype(int)
        raw_recovered = np.asarray(
            [
                recover_unaugmented_class(augmentation, int(prediction))
                for augmentation, prediction in enumerate(augmented_predictions)
            ],
            dtype=int,
        )
        reference_inverse = inverse_class_index(LPS_YXZ_REFERENCE_CLASS_INDEX)
        recovered = np.asarray(
            [
                compose_class_indices(int(raw_class), reference_inverse)
                for raw_class in raw_recovered
            ],
            dtype=int,
        )
        counts = np.bincount(recovered, minlength=MODEL_CLASS_COUNT)
        raw_counts = np.bincount(raw_recovered, minlength=MODEL_CLASS_COUNT)
        order = np.argsort(counts, kind="stable")[::-1]
        winning = int(order[0])
        raw_order = np.argsort(raw_counts, kind="stable")[::-1]
        raw_winning = int(raw_order[0])
        agreement = int(counts[winning])
        runner_up = int(counts[order[1]])
        distribution = counts.astype(float) / MODEL_CLASS_COUNT
        nonzero = distribution[distribution > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        margin = float((agreement - runner_up) / MODEL_CLASS_COUNT)

        support_probabilities = []
        for augmentation, recovered_class in enumerate(recovered):
            if recovered_class != winning:
                continue
            raw_class = int(augmented_predictions[augmentation])
            support_probabilities.append(float(probabilities[augmentation, raw_class]))
        mean_support = float(np.mean(support_probabilities)) if support_probabilities else math.nan

        return CTDeepRotPrediction(
            winning_class_index=winning,
            winning_angles_deg=rotation_class(winning).angles_deg,
            agreement_count=agreement,
            vote_counts=tuple(int(value) for value in counts),
            recovered_votes=tuple(int(value) for value in recovered),
            augmented_predictions=tuple(int(value) for value in augmented_predictions),
            vote_entropy=entropy,
            vote_margin=margin,
            mean_support_probability=mean_support,
            checkpoint_sha256=self.expected_sha256,
            raw_winning_class_index=raw_winning,
            raw_winning_angles_deg=rotation_class(raw_winning).angles_deg,
            raw_vote_counts=tuple(int(value) for value in raw_counts),
            raw_recovered_votes=tuple(int(value) for value in raw_recovered),
        )
