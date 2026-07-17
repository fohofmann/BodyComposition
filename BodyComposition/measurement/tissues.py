"""Derived tissue classes from raw anatomical compartment labels."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np


TISSUE_NAME_ALIASES = {
    "sm": "sm",
    "skeletalmuscle": "sm",
    "bone": "bone",
    "sat": "sat",
    "avat": "avat",
    "tvat": "tvat",
    "vat": "vat",
    "heart": "heart",
    "lung": "lung",
    "imat": "imat",
    "psoas": "psoas",
    "ssat": "ssat",
    "dsat": "dsat",
    "ipat": "ipat",
    "rpat": "rpat",
}


DERIVED_RATIO_DEFINITIONS = (
    (
        "lama_fraction_of_skeletal_muscle_tissue_hu_m29_150",
        "lama_hu_m29_29",
        ("skeletal_muscle_tissue_hu_m29_150",),
    ),
    (
        "imat_fraction_of_sm_plus_imat_hu_m190_m30",
        "imat_ct_hu_m190_m30",
        (
            "skeletal_muscle_tissue_hu_m29_150",
            "imat_ct_hu_m190_m30",
        ),
    ),
    (
        "vat_to_sat_ratio_hu_m190_m30",
        "vat_total_hu_m190_m30",
        ("sat_total_hu_m190_m30",),
    ),
    (
        "vat_fraction_of_vat_sat_imat_hu_m190_m30",
        "vat_total_hu_m190_m30",
        (
            "vat_total_hu_m190_m30",
            "sat_total_hu_m190_m30",
            "imat_ct_hu_m190_m30",
        ),
    ),
)


def canonical_tissue_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return TISSUE_NAME_ALIASES.get(normalized, normalized)


def derive_configured_tissue_masks(
    image_zyx: np.ndarray,
    compartment_labels_zyx: np.ndarray,
    compartment_label_schema: Mapping[int, str],
    definitions: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Derive configured HU classes from untouched compartment labels.

    HU rules are always evaluated on ``image_zyx`` supplied by the canonical
    prepared-image object. Classification denoising used to build a legacy
    compatibility mask is deliberately not accepted here.
    """

    image = np.asarray(image_zyx)
    labels = np.asarray(compartment_labels_zyx)
    if image.shape != labels.shape or image.ndim != 3:
        raise ValueError(
            "image_zyx and compartment_labels_zyx must be aligned 3D arrays."
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Compartment labels must be integer-valued.")

    labels_by_name: dict[str, list[int]] = {}
    for native_label, name in compartment_label_schema.items():
        if (
            isinstance(native_label, bool)
            or not isinstance(native_label, int)
            or native_label <= 0
        ):
            raise ValueError("Compartment label identifiers must be positive integers.")
        labels_by_name.setdefault(canonical_tissue_name(name), []).append(native_label)

    output: dict[str, np.ndarray] = {}
    for definition_name, definition in definitions.items():
        if not bool(definition["enabled"]):
            continue
        source_names = [
            canonical_tissue_name(name) for name in definition["source_labels"]
        ]
        source_labels = [
            native_label
            for source_name in source_names
            for native_label in labels_by_name.get(source_name, [])
        ]
        if not source_labels:
            raise ValueError(
                f"Enabled tissue definition {definition_name!r} has no source "
                f"label in the compartment schema; requested {source_names}."
            )
        mask = np.isin(labels, source_labels)
        hu_range = definition.get("hu_range")
        if hu_range is not None:
            mask &= image >= min(hu_range)
            mask &= image <= max(hu_range)
        output[str(definition_name)] = mask
    return output
