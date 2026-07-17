from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration does not satisfy the runtime schema."""


def update_dict_deep(original_dict: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping):
            existing = original_dict.get(key)
            if not isinstance(existing, dict):
                existing = {}
            original_dict[key] = update_dict_deep(existing, value)
        else:
            original_dict[key] = value
    return original_dict


def update_config(
    config: dict[str, Any] | None = None,
    config_new: dict[str, Any] | Path | None = None,
) -> dict[str, Any]:
    config = {} if config is None else config
    if isinstance(config_new, Path):
        if config_new.exists():
            with config_new.open() as handle:
                loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, Mapping):
                raise ConfigError(f"Configuration file must contain a mapping: {config_new}.")
            config = update_dict_deep(config, loaded)
    elif isinstance(config_new, Mapping):
        config = update_dict_deep(config, config_new)
    else:
        raise TypeError("config_new must be a dict or a Path.")

    for key, value in config.get("paths", {}).items():
        if isinstance(value, str):
            config["paths"][key] = None if value.strip() in {"", "None"} else Path(value)
    for key, value in config.get("logging_level", {}).items():
        if isinstance(value, str):
            level = getattr(logging, value.upper(), None)
            if not isinstance(level, int):
                raise ConfigError(f"Unknown logging level at logging_level.{key}: {value!r}.")
            config["logging_level"][key] = level
    return config


def _require_mapping(config: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError(f"Missing configuration value: {path}.")
        value = value[key]
    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration value {path} must be a mapping.")
    return value


def _value(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError(f"Missing configuration value: {path}.")
        value = value[key]
    return value


def _require_bool(config: Mapping[str, Any], path: str) -> None:
    value = _value(config, path)
    if not isinstance(value, bool):
        raise ConfigError(f"Configuration value {path} must be boolean, got {value!r}.")


def _require_nonnegative_number(config: Mapping[str, Any], path: str) -> None:
    value = _value(config, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigError(f"Configuration value {path} must be a non-negative number.")


def _require_range(config: Mapping[str, Any], path: str) -> None:
    value = _value(config, path)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise ConfigError(f"Configuration value {path} must be a two-number list.")


def validate_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise ConfigError("Configuration must be a mapping.")

    paths = _require_mapping(config, "paths")
    for path_name in ("workspace", "logs", "totalsegmentator_config", "cache"):
        value = paths.get(path_name)
        if value is not None and not isinstance(value, (str, Path)):
            raise ConfigError(f"Configuration value paths.{path_name} must be a path or null.")
    _require_mapping(config, "paths.weights")

    logging_levels = _require_mapping(config, "logging_level")
    for key in ("file", "console"):
        if isinstance(logging_levels.get(key), bool) or not isinstance(logging_levels.get(key), int):
            raise ConfigError(f"Configuration value logging_level.{key} must be a logging level.")

    for path in (
        "run.reset",
        "run.skip",
        "segmentation.save_label",
        "vertebrae.save_mask",
        "vertebrae.fill_undefined_levels",
        "vertebrae.correct_monotonicity",
        "vertebrae.center_of_mass",
        "vertebrae.spineps.save_native_outputs",
        "vertebrae.spineps.review_enabled",
        "tissue.save_mask",
        "tissue.hu_denoise.filter_outliers",
        "tissue.hu_denoise.filter_median",
        "measurements.enabled",
        "measurements.body_surface.save_mask",
        "measurements.landmarks.enabled",
        "measurements.review.enabled",
        "measurements.export.parquet",
        "measurements.export.csv",
    ):
        _require_bool(config, path)

    timeout = _value(config, "run.timeout")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ConfigError("Configuration value run.timeout must be a positive integer.")

    _require_mapping(config, "orientation")
    _require_bool(config, "orientation.enabled")
    _require_bool(config, "orientation.confidence.allow_axial_180_repair")
    _require_bool(config, "orientation.artifact_rules.review_on_possible_truncation")
    _require_bool(config, "orientation.report.enabled")
    for path in (
        "orientation.confidence.min_body_extent_mm",
        "orientation.confidence.max_obliquity_deg",
        "orientation.artifact_rules.max_metal_fraction",
    ):
        _require_nonnegative_number(config, path)
    for path in (
        "orientation.confidence.body_threshold_hu",
        "orientation.artifact_rules.metal_threshold_hu",
    ):
        value = _value(config, path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"Configuration value {path} must be numeric.")
    min_votes = _value(config, "orientation.confidence.min_equivariant_votes")
    if isinstance(min_votes, bool) or not isinstance(min_votes, int) or not 1 <= min_votes <= 24:
        raise ConfigError("orientation.confidence.min_equivariant_votes must be 1..24.")
    min_pixels = _value(config, "orientation.confidence.min_body_pixels_per_slice")
    if isinstance(min_pixels, bool) or not isinstance(min_pixels, int) or min_pixels <= 0:
        raise ConfigError("orientation.confidence.min_body_pixels_per_slice must be positive.")
    metal_fraction = _value(config, "orientation.artifact_rules.max_metal_fraction")
    if metal_fraction > 1:
        raise ConfigError("orientation.artifact_rules.max_metal_fraction must not exceed one.")

    model = _require_mapping(config, "orientation.model")
    if not isinstance(model.get("checkpoint_path"), (str, Path)):
        raise ConfigError("orientation.model.checkpoint_path must be a path.")
    if model.get("device") not in {"cpu", "cuda", "auto"}:
        raise ConfigError("orientation.model.device must be cpu, cuda, or auto.")
    batch_size = model.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 24:
        raise ConfigError("orientation.model.batch_size must be 1..24.")
    for path in (
        "orientation.header_uncertain_reasons",
        "orientation.force_review_reasons",
    ):
        reasons = _value(config, path)
        if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
            raise ConfigError(f"{path} must be a list of strings.")
    for path in (
        "vertebrae.min_voxels_per_vertebra",
        "vertebrae.correct_monotonicity_max_windowsize",
    ):
        value = _value(config, path)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"Configuration value {path} must be a non-negative integer.")

    deprioritized = _value(config, "vertebrae.deprioritize_labels")
    if not isinstance(deprioritized, list) or any(
        isinstance(label, bool) or not isinstance(label, int) for label in deprioritized
    ):
        raise ConfigError("vertebrae.deprioritize_labels must be a list of integer labels.")

    backend = _value(config, "vertebrae.backend")
    supported_backends = {
        "spineps_veridah_ct_v1",
        "vertebral_bodies_resenc_l",
        "vertebral_bodies_resenc_m",
    }
    if backend not in supported_backends:
        raise ConfigError(
            f"vertebrae.backend must be one of {sorted(supported_backends)}, got {backend!r}."
        )
    spineps = _require_mapping(config, "vertebrae.spineps")
    if spineps.get("device") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("vertebrae.spineps.device must be auto, cpu, or cuda.")
    model_root = _value(config, "paths.weights.spineps")
    if not isinstance(model_root, (str, Path)):
        raise ConfigError("paths.weights.spineps must be a path.")
    deprioritized_names = _value(config, "vertebrae.deprioritize_anatomical")
    if not isinstance(deprioritized_names, list) or any(
        not isinstance(name, str) or not name.strip() for name in deprioritized_names
    ):
        raise ConfigError("vertebrae.deprioritize_anatomical must be a list of names.")

    _require_range(config, "tissue.hu_denoise.filter_outliers_range")
    median_kernel = _value(config, "tissue.hu_denoise.filter_median_kernel")
    if not isinstance(median_kernel, list) or len(median_kernel) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in median_kernel
    ):
        raise ConfigError("tissue.hu_denoise.filter_median_kernel must contain three positive integers.")

    for tissue_name in ("imat", "sm", "vat", "sat"):
        base = f"tissue.{tissue_name}"
        _require_bool(config, f"{base}.filter_hu")
        _require_range(config, f"{base}.filter_hu_range")
        _require_bool(config, f"{base}.filter_size")
        version = _value(config, f"{base}.filter_size_version")
        if version not in {"2D", "3D"}:
            raise ConfigError(f"Configuration value {base}.filter_size_version must be 2D or 3D.")
        _require_nonnegative_number(config, f"{base}.filter_size_2D")
        _require_nonnegative_number(config, f"{base}.filter_size_3D")

    measurement = _require_mapping(config, "measurements")
    if measurement.get("totalsegmentator_version") != "2.15.0":
        raise ConfigError("measurements.totalsegmentator_version must be pinned to 2.15.0.")
    body_backend = _value(config, "measurements.body_surface.backend")
    if body_backend not in {
        "tissue_segmentation_envelope_v1",
        "totalsegmentator_body_task299_v1",
        "deterministic_body_mask_v1",
    }:
        raise ConfigError("Unknown measurements.body_surface.backend.")
    landmark_backend = _value(config, "measurements.landmarks.backend")
    if landmark_backend != "totalsegmentator_total_task297_landmarks_v1":
        raise ConfigError("Unknown measurements.landmarks.backend.")
    for path in (
        "measurements.body_surface.min_component_volume_mm3",
        "measurements.body_surface.minimum_component_area_mm2",
        "measurements.landmarks.maximum_side_disagreement_mm",
        "measurements.qc.circumference_jump.maximum_gap_mm",
        "measurements.qc.circumference_jump.minimum_absolute_jump_cm",
        "measurements.qc.circumference_jump.minimum_relative_jump",
    ):
        value = _value(config, path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"Configuration value {path} must be positive.")
    for path in (
        "measurements.body_surface.closing_radius_mm",
        "measurements.body_surface.smoothing_sigma_mm",
    ):
        value = _value(config, path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"Configuration value {path} must be non-negative.")
    maximum_removed_fraction = _value(
        config,
        "measurements.vertebral_extent.maximum_removed_fraction",
    )
    if (
        isinstance(maximum_removed_fraction, bool)
        or not isinstance(maximum_removed_fraction, (int, float))
        or not 0 <= maximum_removed_fraction < 1
    ):
        raise ConfigError(
            "measurements.vertebral_extent.maximum_removed_fraction must be in [0, 1)."
        )
    for path in (
        "measurements.l3_slab_length_mm",
        "measurements.full_coverage_tolerance",
    ):
        value = _value(config, path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"Configuration value {path} must be positive.")
    if _value(config, "measurements.full_coverage_tolerance") > 1:
        raise ConfigError("measurements.full_coverage_tolerance must not exceed one.")
    for path in (
        "measurements.landmarks.minimum_voxels",
        "measurements.vertebral_extent.minimum_component_voxels",
    ):
        value = _value(config, path)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"Configuration value {path} must be a positive integer.")
    threshold_hu = _value(config, "measurements.body_surface.threshold_hu")
    if isinstance(threshold_hu, bool) or not isinstance(threshold_hu, (int, float)):
        raise ConfigError("measurements.body_surface.threshold_hu must be numeric.")
    if measurement["enabled"] and not measurement["export"]["parquet"]:
        raise ConfigError("Canonical measurements require Parquet export.")
    if measurement["enabled"] and measurement["export"]["csv"]:
        raise ConfigError("Canonical measurements do not permit duplicate CSV export.")

    for crop_name, crop in _require_mapping(config, "crop").items():
        if not isinstance(crop, Mapping):
            raise ConfigError(f"crop.{crop_name} must be a mapping.")
        if not isinstance(crop.get("roi"), list) or any(
            isinstance(label, bool) or not isinstance(label, int) for label in crop.get("roi", [])
        ):
            raise ConfigError(f"crop.{crop_name}.roi must be a list of integer labels.")
        anatomical = crop.get("roi_anatomical", [])
        if not isinstance(anatomical, list) or any(
            not isinstance(name, str) or not name.strip() for name in anatomical
        ):
            raise ConfigError(f"crop.{crop_name}.roi_anatomical must be a list of names.")
        axes = crop.get("axes")
        if not isinstance(axes, list) or len(axes) != 6 or any(not isinstance(axis, bool) for axis in axes):
            raise ConfigError(f"crop.{crop_name}.axes must contain six booleans.")
        margin = crop.get("margin")
        if not isinstance(margin, list) or len(margin) != 6 or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) for item in margin
        ):
            raise ConfigError(f"crop.{crop_name}.margin must contain six numbers.")

    for mapping_name in ("LBL_TISSUE", "LBL_VERTEBRALBODIES"):
        mapping = _require_mapping(config, mapping_name)
        if not mapping or any(
            isinstance(label, bool)
            or not isinstance(label, int)
            or label <= 0
            or not isinstance(name, str)
            or not name.strip()
            for label, name in mapping.items()
        ):
            raise ConfigError(f"{mapping_name} must map positive integers to non-empty names.")

    return config
