from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from BodyComposition.config import (
    CANONICAL_TISSUE_DEFINITIONS,
    TISSUE_LABELS,
    ConfigError,
)
from BodyComposition.measurement.tissues import (
    canonical_compartment_name,
    tissue_source_compartment_name,
)


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
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ConfigError(f"Configuration value {path} must be a non-negative number.")


def _require_range(config: Mapping[str, Any], path: str) -> None:
    value = _value(config, path)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
        or value[0] > value[1]
    ):
        raise ConfigError(f"Configuration value {path} must be an ordered two-number list.")


def validate_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise ConfigError("Configuration must be a mapping.")

    paths = _require_mapping(config, "paths")
    for path_name in ("workspace", "logs", "cache"):
        value = paths.get(path_name)
        if value is not None and not isinstance(value, (str, Path)):
            raise ConfigError(f"Configuration value paths.{path_name} must be a path or null.")
    _require_mapping(config, "paths.weights")

    logging_levels = _require_mapping(config, "logging_level")
    for key in ("file", "console"):
        if isinstance(logging_levels.get(key), bool) or not isinstance(
            logging_levels.get(key), int
        ):
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
        "tissue.save_compartment_mask",
        "measurements.enabled",
        "measurements.body_surface.save_mask",
        "measurements.landmarks.enabled",
        "measurements.review.enabled",
        "measurements.export.parquet",
        "measurements.export.csv",
    ):
        _require_bool(config, path)

    analysis = _require_mapping(config, "analysis")
    if analysis.get("scope") not in {"full_ct", "l3_vertebral_level"}:
        raise ConfigError("analysis.scope must be full_ct or l3_vertebral_level.")
    _require_nonnegative_number(
        config,
        "analysis.l3.inference_context_mm",
    )

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

    _require_mapping(config, "tissue")
    measurement = _require_mapping(config, "measurements")
    profile_id = measurement.get("tissue_profile_id")
    if not isinstance(profile_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", profile_id):
        raise ConfigError(
            "measurements.tissue_profile_id must be a non-empty lowercase profile identifier."
        )

    def require_kernel(kernel: Any, path: str) -> list[int]:
        if (
            not isinstance(kernel, list)
            or len(kernel) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value % 2 == 0
                for value in kernel
            )
        ):
            raise ConfigError(f"{path} must contain three positive odd integers.")
        return kernel

    definitions = _require_mapping(config, "measurements.tissue_definitions")
    for definition_name, definition in definitions.items():
        if not isinstance(definition_name, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*",
            definition_name,
        ):
            raise ConfigError("measurements.tissue_definitions keys must use lowercase snake_case.")
        if not isinstance(definition, Mapping):
            raise ConfigError(
                f"measurements.tissue_definitions.{definition_name} must be a mapping."
            )
        definition_path = f"measurements.tissue_definitions.{definition_name}"
        allowed_definition_keys = {
            "enabled",
            "source_labels",
            "hu_range",
            "preprocessing",
            "cleanup",
        }
        unknown_definition_keys = sorted(set(definition) - allowed_definition_keys)
        if unknown_definition_keys:
            raise ConfigError(f"{definition_path} has unknown values: {unknown_definition_keys}.")
        enabled = definition.get("enabled")
        if not isinstance(enabled, bool):
            raise ConfigError(
                f"measurements.tissue_definitions.{definition_name}.enabled must be boolean."
            )
        sources = definition.get("source_labels")
        if (
            not isinstance(sources, list)
            or not sources
            or any(not isinstance(source, str) or not source.strip() for source in sources)
        ):
            raise ConfigError(
                f"measurements.tissue_definitions.{definition_name}.source_labels "
                "must be a non-empty list of names."
            )
        hu_range = definition.get("hu_range")
        if hu_range is not None and (
            not isinstance(hu_range, list)
            or len(hu_range) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in hu_range
            )
            or hu_range[0] > hu_range[1]
        ):
            raise ConfigError(
                f"measurements.tissue_definitions.{definition_name}.hu_range "
                "must be null or a two-number list."
            )
        preprocessing = definition.get("preprocessing")
        if preprocessing is not None:
            if not isinstance(preprocessing, Mapping):
                raise ConfigError(f"{definition_path}.preprocessing must be a mapping.")
            method = preprocessing.get("method")
            allowed_methods = {
                "none",
                "median",
                "adaptive_median",
                "curvature_anisotropic_diffusion",
            }
            if method not in allowed_methods:
                raise ConfigError(f"{definition_path}.preprocessing.method is unsupported.")
            method_keys = {
                "none": set(),
                "median": {"kernel_zyx"},
                "adaptive_median": {
                    "minimum_kernel_zyx",
                    "maximum_kernel_zyx",
                },
                "curvature_anisotropic_diffusion": {
                    "anisotropic_diffusion",
                },
            }[str(method)]
            allowed_preprocessing_keys = {
                "method",
                "clip_hu_range",
                *method_keys,
            }
            unknown_preprocessing = sorted(set(preprocessing) - allowed_preprocessing_keys)
            missing_preprocessing = sorted(method_keys - set(preprocessing))
            if unknown_preprocessing or missing_preprocessing:
                raise ConfigError(
                    f"{definition_path}.preprocessing has unknown="
                    f"{unknown_preprocessing}, missing={missing_preprocessing}."
                )
            clip_range = preprocessing.get("clip_hu_range")
            if clip_range is not None and (
                not isinstance(clip_range, list)
                or len(clip_range) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in clip_range
                )
                or clip_range[0] > clip_range[1]
            ):
                raise ConfigError(
                    f"{definition_path}.preprocessing.clip_hu_range must be "
                    "null or an ordered two-number list."
                )
            if method == "median":
                require_kernel(
                    preprocessing["kernel_zyx"],
                    f"{definition_path}.preprocessing.kernel_zyx",
                )
            elif method == "adaptive_median":
                minimum = require_kernel(
                    preprocessing["minimum_kernel_zyx"],
                    f"{definition_path}.preprocessing.minimum_kernel_zyx",
                )
                maximum = require_kernel(
                    preprocessing["maximum_kernel_zyx"],
                    f"{definition_path}.preprocessing.maximum_kernel_zyx",
                )
                if any(lower > upper for lower, upper in zip(minimum, maximum, strict=True)):
                    raise ConfigError(
                        f"{definition_path}.preprocessing minimum kernel must "
                        "not exceed its maximum."
                    )
            elif method == "curvature_anisotropic_diffusion":
                diffusion = preprocessing["anisotropic_diffusion"]
                if not isinstance(diffusion, Mapping):
                    raise ConfigError(
                        f"{definition_path}.preprocessing.anisotropic_diffusion must be a mapping."
                    )
                expected_diffusion = {
                    "dimensionality",
                    "iterations",
                    "time_step",
                    "conductance",
                }
                if set(diffusion) != expected_diffusion:
                    raise ConfigError(
                        f"{definition_path}.preprocessing.anisotropic_diffusion "
                        f"must contain exactly {sorted(expected_diffusion)}."
                    )
                dimensionality = diffusion["dimensionality"]
                if dimensionality not in {"2D", "3D"}:
                    raise ConfigError(
                        f"{definition_path}.preprocessing.anisotropic_diffusion."
                        "dimensionality must be 2D or 3D."
                    )
                iterations = diffusion["iterations"]
                if (
                    isinstance(iterations, bool)
                    or not isinstance(iterations, int)
                    or iterations <= 0
                ):
                    raise ConfigError(
                        f"{definition_path}.preprocessing.anisotropic_diffusion."
                        "iterations must be positive."
                    )
                for key in ("time_step", "conductance"):
                    value = diffusion[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                        raise ConfigError(
                            f"{definition_path}.preprocessing."
                            f"anisotropic_diffusion.{key} must be positive."
                        )
                stability_limit = 0.125 if dimensionality == "2D" else 0.0625
                if float(diffusion["time_step"]) > stability_limit:
                    raise ConfigError(
                        f"{definition_path}.preprocessing.anisotropic_diffusion."
                        f"time_step exceeds the {dimensionality} stability limit "
                        f"{stability_limit}."
                    )

        cleanup = definition.get("cleanup")
        if cleanup is not None:
            if not isinstance(cleanup, Mapping):
                raise ConfigError(f"{definition_path}.cleanup must be a mapping.")
            unknown_cleanup = sorted(set(cleanup) - {"fill_small_holes", "remove_small_objects"})
            if unknown_cleanup:
                raise ConfigError(
                    f"{definition_path}.cleanup has unknown values: {unknown_cleanup}."
                )
            for operation_name, operation in cleanup.items():
                operation_path = f"{definition_path}.cleanup.{operation_name}"
                if not isinstance(operation, Mapping):
                    raise ConfigError(f"{operation_path} must be a mapping.")
                expected_operation = {
                    "dimensionality",
                    "threshold",
                    "unit",
                    "connectivity",
                }
                if set(operation) != expected_operation:
                    raise ConfigError(
                        f"{operation_path} must contain exactly {sorted(expected_operation)}."
                    )
                dimensionality = operation["dimensionality"]
                if dimensionality not in {"2D", "3D"}:
                    raise ConfigError(f"{operation_path}.dimensionality must be 2D or 3D.")
                threshold = operation["threshold"]
                if (
                    isinstance(threshold, bool)
                    or not isinstance(threshold, (int, float))
                    or threshold < 0
                ):
                    raise ConfigError(f"{operation_path}.threshold must be non-negative.")
                if operation["unit"] not in {"physical", "voxel"}:
                    raise ConfigError(f"{operation_path}.unit must be physical or voxel.")
                allowed_connectivity = {4, 8} if dimensionality == "2D" else {6, 18, 26}
                if operation["connectivity"] not in allowed_connectivity:
                    raise ConfigError(
                        f"{operation_path}.connectivity must be one of "
                        f"{sorted(allowed_connectivity)} for {dimensionality}."
                    )

    changed_canonical = sorted(
        name
        for name, expected in CANONICAL_TISSUE_DEFINITIONS.items()
        if definitions.get(name) != expected
    )
    if changed_canonical:
        raise ConfigError(
            "Canonical consensus tissue definitions are immutable; changed or "
            f"missing={changed_canonical}."
        )
    if measurement.get("measurement_support_nnunet_version") != "2.5.2":
        raise ConfigError(
            "measurements.measurement_support_nnunet_version must be pinned to 2.5.2."
        )
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

    try:
        from BodyComposition.reporting.contracts import ReportingSettings

        reporting = ReportingSettings.from_mapping(config)
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error
    if reporting.enabled and not measurement["enabled"]:
        raise ConfigError("reporting.enabled requires canonical measurements.enabled=true.")
    if reporting.enabled and not _value(config, "tissue.save_mask"):
        raise ConfigError(
            "reporting.enabled requires tissue.save_mask=true so the axial "
            "segmentation view can be regenerated post hoc."
        )

    for mapping_name in (
        "LBL_TISSUE_COMPARTMENTS",
        "LBL_TISSUE",
        "LBL_VERTEBRALBODIES",
    ):
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

    compartment_labels = _require_mapping(config, "LBL_TISSUE_COMPARTMENTS")
    tissue_labels = _require_mapping(config, "LBL_TISSUE")
    if tissue_labels != TISSUE_LABELS:
        raise ConfigError("LBL_TISSUE must match the released HU-filtered tissue label schema.")
    tissue_source_compartments = {
        tissue_source_compartment_name(name) for name in compartment_labels.values()
    }
    canonical_tissues = {canonical_compartment_name(name) for name in tissue_labels.values()}
    if not canonical_tissues.issubset(tissue_source_compartments):
        raise ConfigError("LBL_TISSUE labels must resolve to configured model-native compartments.")

    return config
