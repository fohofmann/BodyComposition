"""Derived tissue classes from raw anatomical compartment labels."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from BodyComposition.tissue import cleanup_tissue_mask, prepare_classification_image

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
    "psoas": "psoas",
    "ssat": "ssat",
    "dsat": "dsat",
    "ipat": "ipat",
    "rpat": "rpat",
}


DERIVED_RATIO_DEFINITIONS = (
    (
        "vat_to_sat_ratio_hu_m190_m30",
        "vat_total_hu_m190_m30",
        ("sat_total_hu_m190_m30",),
    ),
)


def _classification_settings(
    preprocessing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Translate one sparse downstream recipe to the classification helper."""

    settings = dict(preprocessing or {"method": "none"})
    method = str(settings.get("method", "none"))
    clip_range = settings.get("clip_hu_range")
    return {
        "method": method,
        "filter_outliers": clip_range is not None,
        "filter_outliers_range": (
            list(clip_range) if clip_range is not None else [-1024.0, 3071.0]
        ),
        "filter_median_kernel": list(settings.get("kernel_zyx", [1, 3, 3])),
        "adaptive_median_min_kernel": list(
            settings.get("minimum_kernel_zyx", [1, 3, 3])
        ),
        "adaptive_median_max_kernel": list(
            settings.get("maximum_kernel_zyx", [1, 7, 7])
        ),
        "anisotropic_diffusion": dict(
            settings.get(
                "anisotropic_diffusion",
                {
                    "dimensionality": "2D",
                    "iterations": 5,
                    "time_step": 0.0625,
                    "conductance": 3.0,
                },
            )
        ),
    }


def _cleanup_settings(cleanup: Mapping[str, Any] | None) -> dict[str, Any]:
    """Translate sparse named cleanup operations to the morphology helper."""

    cleanup = dict(cleanup or {})
    holes = cleanup.get("fill_small_holes")
    objects = cleanup.get("remove_small_objects")

    def value(operation: Mapping[str, Any] | None, key: str, default: Any) -> Any:
        return operation[key] if operation is not None else default

    return {
        "fill_holes": holes is not None,
        "fill_holes_version": value(holes, "dimensionality", "2D"),
        "fill_holes_2D": (
            float(value(holes, "threshold", 0.0))
            if value(holes, "dimensionality", "2D") == "2D"
            else 0.0
        ),
        "fill_holes_3D": (
            float(value(holes, "threshold", 0.0))
            if value(holes, "dimensionality", "2D") == "3D"
            else 0.0
        ),
        "fill_holes_unit": value(holes, "unit", "physical"),
        "fill_holes_connectivity": int(value(holes, "connectivity", 8)),
        "filter_size": objects is not None,
        "filter_size_version": value(objects, "dimensionality", "3D"),
        "filter_size_2D": (
            float(value(objects, "threshold", 0.0))
            if value(objects, "dimensionality", "3D") == "2D"
            else 0.0
        ),
        "filter_size_3D": (
            float(value(objects, "threshold", 0.0))
            if value(objects, "dimensionality", "3D") == "3D"
            else 0.0
        ),
        "filter_size_unit": value(objects, "unit", "physical"),
        "filter_size_connectivity": int(value(objects, "connectivity", 26)),
    }


def canonical_tissue_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return TISSUE_NAME_ALIASES.get(normalized, normalized)


def derive_configured_tissue_masks(
    image_zyx: np.ndarray,
    compartment_labels_zyx: np.ndarray,
    compartment_label_schema: Mapping[int, str],
    definitions: Mapping[str, Mapping[str, Any]],
    *,
    spacing_xyz: tuple[float, float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Apply named downstream definitions to immutable compartment labels.

    Each definition owns its optional classification preprocessing and
    morphology. Mean-HU measurements are calculated later from ``image_zyx``
    itself, so a denoised classification branch never replaces raw CT values.
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

    unknown_labels = sorted(
        int(label)
        for label in np.unique(labels)
        if int(label) != 0 and int(label) not in compartment_label_schema
    )
    if unknown_labels:
        raise ValueError(
            "Compartment labels contain values outside the model-native schema: "
            f"{unknown_labels}."
        )

    output: dict[str, np.ndarray] = {}
    classification_cache: dict[str, np.ndarray] = {}
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
        support = np.isin(labels, source_labels)
        mask = support.copy()
        hu_range = definition.get("hu_range")
        if hu_range is not None:
            preprocessing = definition.get("preprocessing")
            cache_key = json.dumps(
                dict(preprocessing or {"method": "none"}),
                sort_keys=True,
                separators=(",", ":"),
            )
            if cache_key not in classification_cache:
                settings = _classification_settings(preprocessing)
                if (
                    settings["method"] == "curvature_anisotropic_diffusion"
                    and spacing_xyz is None
                ):
                    raise ValueError(
                        "spacing_xyz is required for anisotropic-diffusion "
                        f"definition {definition_name!r}."
                    )
                classification_cache[cache_key] = prepare_classification_image(
                    image,
                    spacing_xyz or (1.0, 1.0, 1.0),
                    settings,
                )
            classification_image = classification_cache[cache_key]
            mask &= classification_image >= min(hu_range)
            mask &= classification_image <= max(hu_range)
        cleanup = definition.get("cleanup")
        if cleanup:
            if spacing_xyz is None:
                raise ValueError(
                    f"spacing_xyz is required for cleanup definition "
                    f"{definition_name!r}."
                )
            cleanup_tissue_mask(
                mask,
                _cleanup_settings(cleanup),
                spacing_xyz,
                support_mask=support,
            )
        output[str(definition_name)] = mask
    return output
