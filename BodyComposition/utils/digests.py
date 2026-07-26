"""Deterministic digests for array-backed pipeline artifacts."""

from __future__ import annotations

import hashlib

import numpy as np


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes without an array-sized bytes copy."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(contiguous.reshape(-1)).cast("B"))
    return digest.hexdigest()
