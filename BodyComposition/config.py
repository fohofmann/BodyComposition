"""Validated public configuration for the BodyComposition release surface.

The public configuration is intentionally smaller than the dictionary consumed
by the internal action implementations.  :class:`PipelineConfig` is the only
supported configuration boundary; ``to_runtime_dict`` performs the narrow
translation needed while those validated scientific actions are retained.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from BodyComposition.tissue_backends.boa import (
    BOA_BACKEND_ID,
    BOA_NATIVE_COMPARTMENT_LABELS,
)

CONFIG_SCHEMA_VERSION = "1.0.0"

VERTEBRAL_LABELS = {
    1: "T1",
    2: "T2",
    3: "T3",
    4: "T4",
    5: "T5",
    6: "T6",
    7: "T7",
    8: "T8",
    9: "T9",
    10: "T10",
    11: "T11",
    12: "T12",
    13: "L1",
    14: "L2",
    15: "L3",
    16: "L4",
    17: "L5",
    18: "L6",
    19: "SACRUM",
    20: "COCCYX",
    21: "T13",
}

TISSUE_COMPARTMENT_LABELS = {
    1: "SM",
    2: "BONE",
    3: "SAT",
    4: "aVAT",
    5: "tVAT",
    6: "HEART",
    7: "LUNG",
}

TISSUE_BACKEND_IDS = (
    "bodycomposition_resenc_l_v1",
    "bodycomposition_resenc_m_v1",
    BOA_BACKEND_ID,
)

# ``tissue_labels`` contains only the default HU-filtered body-composition
# tissues. Native bone, heart, and lung predictions remain available in the
# immutable compartment artifact.
TISSUE_LABELS = {
    1: "SM",
    3: "SAT",
    4: "aVAT",
    5: "tVAT",
}

CONSENSUS_TISSUE_PROFILE_ID = (
    "consensus_hu_muscle_m29_150_adipose_m190_m30_v1"
)

CANONICAL_TISSUE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "skeletal_muscle_tissue_hu_m29_150": {
        "enabled": True,
        "source_labels": ["SM"],
        "hu_range": [-29, 150],
    },
    "sat_tissue_hu_m190_m30": {
        "enabled": True,
        "source_labels": ["SAT"],
        "hu_range": [-190, -30],
    },
    "avat_tissue_hu_m190_m30": {
        "enabled": True,
        "source_labels": ["aVAT"],
        "hu_range": [-190, -30],
    },
    "tvat_tissue_hu_m190_m30": {
        "enabled": True,
        "source_labels": ["tVAT"],
        "hu_range": [-190, -30],
    },
    "vat_tissue_hu_m190_m30": {
        "enabled": True,
        "source_labels": ["aVAT", "tVAT", "VAT"],
        "hu_range": [-190, -30],
    },
}


class ConfigError(ValueError):
    """Raised when a public configuration is incomplete or inconsistent."""


def _default_mapping() -> dict[str, Any]:
    """Return a new copy of the release defaults.

    Defaults live in code so installed wheels, the API, the CLI, and the
    container cannot accidentally read different working-directory YAML files.
    ``bodycomposition config show-default`` is the canonical YAML generator.
    """

    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "models": {
            "root": os.environ.get(
                "BODYCOMPOSITION_MODEL_ROOT",
                "~/.cache/bodycomposition/models",
            )
        },
        "analysis": {
            "scope": "full_ct",
            "l3": {
                "inference_context_mm": 120.0,
            },
        },
        "orientation": {
            "backend": "ctdeeprot_2d_v1",
            "policy": "check_and_safe_repair",
            "confidence": {
                "min_equivariant_votes": 23,
                "min_body_extent_mm": 250.0,
                "max_obliquity_deg": 15.0,
                "body_threshold_hu": -700.0,
                "min_body_pixels_per_slice": 100,
                "allow_axial_180_repair": False,
            },
            "artifact_rules": {
                "metal_threshold_hu": 2500.0,
                "max_metal_fraction": 0.0005,
                "review_on_possible_truncation": False,
            },
            "model_device": "cpu",
            "batch_size": 24,
            "header_uncertain_reasons": [],
            "force_review_reasons": [],
        },
        "vertebrae": {
            "backend": "spineps_veridah_ct_v1",
            "device": "auto",
            "save_native_outputs": True,
            "review_image": True,
        },
        "tissue": {
            "backend": "bodycomposition_resenc_l_v1",
        },
        "body_surface": {
            "backend": "tissue_segmentation_envelope_v1",
            "minimum_component_area_mm2": 100.0,
            "closing_radius_mm": 6.0,
            "smoothing_sigma_mm": 2.0,
            "threshold_hu": -500.0,
            "min_component_volume_mm3": 10000.0,
        },
        "measurements": {
            "full_coverage_tolerance": 0.999,
            "l3_slab_length_mm": 200.0,
            "tissue_profile_id": CONSENSUS_TISSUE_PROFILE_ID,
            "tissue_definitions": copy.deepcopy(CANONICAL_TISSUE_DEFINITIONS),
            "landmarks": {
                "enabled": True,
                "backend": "totalsegmentator_total_task297_landmarks_v1",
                "minimum_voxels": 20,
                "maximum_side_disagreement_mm": 30.0,
            },
            "vertebral_extent": {
                "minimum_component_voxels": 20,
                "maximum_removed_fraction": 0.05,
            },
            "qc": {
                "circumference_jump": {
                    "maximum_gap_mm": 10.0,
                    "minimum_absolute_jump_cm": 15.0,
                    "minimum_relative_jump": 0.15,
                }
            },
        },
        "reporting": {
            "enabled": False,
            "layout": "spine_profile_v2",
            "individual_pdf": True,
            "combined_pdf": True,
            "page_size": "A4_landscape",
            "spine_view": "sagittal_thick_slab_v1",
            "measure_aggregation": "territory_mean",
            "measurement_columns": [
                "skeletal_muscle_tissue_hu_m29_150_mean_csa_cm2",
                "vat_tissue_hu_m190_m30_mean_csa_cm2",
                "sat_tissue_hu_m190_m30_mean_csa_cm2",
                "trunk_mean_circumference_cm",
            ],
            "vertebral_range": "detected",
            "include_qc_flags": True,
            "manual_review_summary": "auto",
            "numeric_precision": 1,
            "locale": "en",
            "missing_value_symbol": "NA",
            "ct_window": [-450.0, 1050.0],
            "overlay_opacity": 0.42,
            "sagittal_slab_margin_mm": 25.0,
            "projection_spacing_mm": 1.5,
        },
        "runtime": {
            "device": "auto",
            "cpu_threads": 0,
            "max_workers": 1,
            "timeout_seconds": 14400,
            "fail_fast": False,
            "deterministic": True,
            "unload_models_between_stages": False,
            "allow_dirty": False,
        },
        "output": {
            "save_tissue_labels": True,
            "save_tissue_compartments": True,
            "save_body_surface": True,
            "save_review_images": True,
            "save_csv_tables": True,
            "log_level": "INFO",
        },
    }


def _merge_strict(base: dict[str, Any], update: Mapping[str, Any], path: str = "") -> None:
    for key, value in update.items():
        location = f"{path}.{key}" if path else str(key)
        if key not in base:
            if path == "measurements.tissue_definitions":
                if not isinstance(value, Mapping):
                    raise ConfigError(
                        f"Configuration value {location} must be a mapping."
                    )
                base[key] = copy.deepcopy(dict(value))
                continue
            raise ConfigError(f"Unknown configuration value: {location}.")
        if isinstance(base[key], dict):
            if not isinstance(value, Mapping):
                raise ConfigError(f"Configuration value {location} must be a mapping.")
            _merge_strict(base[key], value, location)
        else:
            base[key] = copy.deepcopy(value)


def _require_bool(mapping: Mapping[str, Any], key: str) -> None:
    if not isinstance(mapping.get(key), bool):
        raise ConfigError(f"Configuration value {key} must be boolean.")


def _validate_public(data: Mapping[str, Any]) -> None:
    if data.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}.")
    model_root = data["models"]["root"]
    if not isinstance(model_root, (str, Path)) or not str(model_root).strip():
        raise ConfigError("models.root must be a non-empty path.")

    orientation = data["orientation"]
    if orientation["backend"] != "ctdeeprot_2d_v1":
        raise ConfigError("orientation.backend must be 'ctdeeprot_2d_v1'.")
    if orientation["policy"] != "check_and_safe_repair":
        raise ConfigError("orientation.policy must be 'check_and_safe_repair'.")
    if orientation["model_device"] not in {"auto", "cpu", "cuda"}:
        raise ConfigError("orientation.model_device must be auto, cpu, or cuda.")

    if data["vertebrae"]["backend"] not in {
        "spineps_veridah_ct_v1",
        "vertebral_bodies_resenc_l",
        "vertebral_bodies_resenc_m",
    }:
        raise ConfigError("Unknown vertebrae.backend.")
    if data["vertebrae"]["device"] not in {"auto", "cpu", "cuda"}:
        raise ConfigError("vertebrae.device must be auto, cpu, or cuda.")
    if data["tissue"]["backend"] not in TISSUE_BACKEND_IDS:
        raise ConfigError("Unknown tissue.backend.")
    if data["body_surface"]["backend"] not in {
        "tissue_segmentation_envelope_v1",
        "totalsegmentator_body_task299_v1",
        "deterministic_body_mask_v1",
    }:
        raise ConfigError("Unknown body_surface.backend.")

    analysis = data["analysis"]
    if analysis["scope"] not in {"full_ct", "l3_vertebral_level"}:
        raise ConfigError(
            "analysis.scope must be full_ct or l3_vertebral_level."
        )
    context_mm = analysis["l3"]["inference_context_mm"]
    if (
        isinstance(context_mm, bool)
        or not isinstance(context_mm, (int, float))
        or not math.isfinite(context_mm)
        or context_mm < 0
    ):
        raise ConfigError(
            "analysis.l3.inference_context_mm must be a non-negative number."
        )

    runtime = data["runtime"]
    if runtime["device"] not in {"auto", "cpu", "cuda"}:
        raise ConfigError("runtime.device must be auto, cpu, or cuda.")
    for key in ("cpu_threads", "max_workers", "timeout_seconds"):
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"runtime.{key} must be an integer.")
    if runtime["cpu_threads"] < 0:
        raise ConfigError("runtime.cpu_threads must be non-negative (zero means automatic).")
    if runtime["max_workers"] != 1:
        raise ConfigError(
            "runtime.max_workers is fixed to 1 inside each process; start the same "
            "batch command multiple times to share the filesystem-backed case queue."
        )
    if runtime["timeout_seconds"] <= 0:
        raise ConfigError("runtime.timeout_seconds must be positive.")
    for key in (
        "fail_fast",
        "deterministic",
        "unload_models_between_stages",
        "allow_dirty",
    ):
        _require_bool(runtime, key)
    for key in (
        "save_tissue_labels",
        "save_tissue_compartments",
        "save_body_surface",
        "save_review_images",
        "save_csv_tables",
    ):
        _require_bool(data["output"], key)
    if data["output"]["log_level"] not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError("output.log_level must be DEBUG, INFO, WARNING, or ERROR.")
    if data["reporting"]["enabled"] and not data["output"]["save_tissue_labels"]:
        raise ConfigError(
            "reporting.enabled requires output.save_tissue_labels=true so the axial "
            "segmentation view can be regenerated post hoc."
        )

    reasons = orientation["header_uncertain_reasons"]
    forced = orientation["force_review_reasons"]
    if any(not isinstance(value, str) for value in [*reasons, *forced]):
        raise ConfigError("Orientation review reasons must be strings.")
    profile_id = data["measurements"]["tissue_profile_id"]
    if not isinstance(profile_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", profile_id):
        raise ConfigError("measurements.tissue_profile_id must be a lowercase identifier.")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable validated configuration shared by the Python API and CLI."""

    _data: dict[str, Any]

    @classmethod
    def model_validate(cls, value: PipelineConfig | Mapping[str, Any] | None) -> PipelineConfig:
        if isinstance(value, cls):
            return value
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ConfigError("Pipeline configuration must be a mapping.")
        data = _default_mapping()
        _merge_strict(data, value)
        _validate_public(data)
        instance = cls(copy.deepcopy(data))
        from BodyComposition.utils.config import validate_config

        validate_config(instance.to_runtime_dict())
        return instance

    @classmethod
    def load(cls, path: str | Path | None = None) -> PipelineConfig:
        if path is None:
            return cls.model_validate({})
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Configuration file not found: {source}.")
        loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ConfigError("Configuration file must contain a mapping.")
        return cls.model_validate(loaded)

    @property
    def model_root(self) -> Path:
        return Path(self._data["models"]["root"]).expanduser().resolve()

    @property
    def vertebral_backend(self) -> str:
        return str(self._data["vertebrae"]["backend"])

    @property
    def tissue_backend(self) -> str:
        return str(self._data["tissue"]["backend"])

    @property
    def device(self) -> str:
        return str(self._data["runtime"]["device"])

    @property
    def allow_dirty(self) -> bool:
        return bool(self._data["runtime"]["allow_dirty"])

    @property
    def fail_fast(self) -> bool:
        return bool(self._data["runtime"]["fail_fast"])

    @property
    def reporting_enabled(self) -> bool:
        return bool(self._data["reporting"]["enabled"])

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        value = copy.deepcopy(self._data)
        return _json_safe(value) if mode == "json" else value

    def normalized(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def normalized_json(self) -> str:
        return json.dumps(self.normalized(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def digest(self) -> str:
        return hashlib.sha256(self.normalized_json().encode("utf-8")).hexdigest()

    def scientific_normalized(self) -> dict[str, Any]:
        value = self.normalized()
        value.pop("models")
        runtime = value.pop("runtime")
        value["deterministic"] = runtime["deterministic"]
        value.pop("output")
        value.pop("reporting")
        for key in ("save_native_outputs", "review_image", "device"):
            value["vertebrae"].pop(key)
        return value

    def scientific_digest(self) -> str:
        payload = json.dumps(
            self.scientific_normalized(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def queue_normalized(self) -> dict[str, Any]:
        """Return the configuration contract shared by compatible workers.

        Worker-local resource choices may differ without changing scientific
        results or the requested output bundle.  Reporting, output selection,
        fail-fast behavior, determinism, and every scientific setting remain
        queue-wide and therefore stay in this projection.
        """

        value = self.normalized()
        value.pop("models")
        for key in (
            "device",
            "cpu_threads",
            "timeout_seconds",
            "unload_models_between_stages",
            "allow_dirty",
        ):
            value["runtime"].pop(key)
        value["output"].pop("log_level")
        value["vertebrae"].pop("device")
        return value

    def queue_digest(self) -> str:
        payload = json.dumps(
            self.queue_normalized(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.normalized(), sort_keys=False, allow_unicode=False)

    def to_runtime_dict(self) -> dict[str, Any]:
        """Translate the public schema to the validated internal action mapping."""

        value = self.model_dump()
        root = self.model_root
        tissue = copy.deepcopy(value["tissue"])
        tissue.pop("backend")
        tissue["save_mask"] = value["output"]["save_tissue_labels"]
        tissue["save_compartment_mask"] = value["output"]["save_tissue_compartments"]
        measurements = copy.deepcopy(value["measurements"])
        measurements.update(
            {
                "enabled": True,
                "measurement_support_nnunet_version": "2.5.2",
                "body_surface": {
                    **copy.deepcopy(value["body_surface"]),
                    "save_mask": value["output"]["save_body_surface"],
                },
                "review": {"enabled": value["output"]["save_review_images"]},
                "export": {
                    "parquet": True,
                    "csv": value["output"]["save_csv_tables"],
                },
            }
        )
        orientation = copy.deepcopy(value["orientation"])
        orientation.pop("backend")
        orientation.pop("policy")
        orientation["enabled"] = True
        from BodyComposition.orientation.ctdeeprot import UPSTREAM_COMMIT

        orientation["model"] = {
            "checkpoint_path": root / "CTDeepRot" / UPSTREAM_COMMIT / "net2d.pt",
            "device": orientation.pop("model_device"),
            "batch_size": orientation.pop("batch_size"),
        }
        orientation["report"] = {"enabled": value["output"]["save_review_images"]}
        vertebrae = {
            "backend": value["vertebrae"]["backend"],
            "save_mask": True,
            "min_voxels_per_vertebra": 20,
            "fill_undefined_levels": False,
            "correct_monotonicity": False,
            "correct_monotonicity_max_windowsize": 0,
            "center_of_mass": True,
            "deprioritize_labels": [19],
            "deprioritize_anatomical": ["SACRUM"],
            "spineps": {
                "device": value["vertebrae"]["device"],
                "save_native_outputs": value["vertebrae"]["save_native_outputs"],
                "review_enabled": value["vertebrae"]["review_image"],
            },
        }
        level = getattr(__import__("logging"), value["output"]["log_level"])
        return {
            "paths": {
                "workspace": None,
                "logs": None,
                "weights": {
                    "int-vertebrae": root / "Dataset601_VertebralBodies",
                    "spineps": root / "SPINEPS" / "spineps-veridah-ct-v1",
                    "int-bodycomposition": root / "Dataset611_BodyComposition",
                    "boa-body-regions": root,
                    "totalsegmentator": root,
                },
                "cache": root / ".cache",
            },
            "logging_level": {"file": level, "console": level},
            "run": {
                "reset": False,
                "skip": False,
                "timeout": value["runtime"]["timeout_seconds"],
            },
            "orientation": orientation,
            "analysis": copy.deepcopy(value["analysis"]),
            "segmentation": {"save_label": value["output"]["save_tissue_compartments"]},
            "vertebrae": vertebrae,
            "tissue": tissue,
            "measurements": measurements,
            "reporting": copy.deepcopy(value["reporting"]),
            "LBL_TISSUE_COMPARTMENTS": copy.deepcopy(
                BOA_NATIVE_COMPARTMENT_LABELS
                if value["tissue"]["backend"] == BOA_BACKEND_ID
                else TISSUE_COMPARTMENT_LABELS
            ),
            "LBL_TISSUE": copy.deepcopy(TISSUE_LABELS),
            "LBL_VERTEBRALBODIES": copy.deepcopy(VERTEBRAL_LABELS),
        }


def default_config() -> PipelineConfig:
    return PipelineConfig.model_validate({})


def low_resource_config(
    base: PipelineConfig | Mapping[str, Any] | None = None,
) -> PipelineConfig:
    """Return the supported L3-only ResEncM analysis preset.

    The preset retains orientation assessment and vertebral localization on the
    complete CT. Tissue inference is restricted to the L3 vertebral territory,
    with surrounding slices used only as model context.
    """

    value = PipelineConfig.model_validate(base).normalized()
    value["analysis"]["scope"] = "l3_vertebral_level"
    value["vertebrae"]["backend"] = "vertebral_bodies_resenc_m"
    value["tissue"]["backend"] = "bodycomposition_resenc_m_v1"
    value["measurements"]["landmarks"]["enabled"] = False
    value["runtime"]["unload_models_between_stages"] = True
    return PipelineConfig.model_validate(value)


def _schema_for_default(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            "type": "object",
            "additionalProperties": False,
            "required": list(value),
            "properties": {str(key): _schema_for_default(item) for key, item in value.items()},
        }
    if isinstance(value, list):
        item_schema = _schema_for_default(value[0]) if value else {}
        return {"type": "array", "items": item_schema, "default": value}
    if isinstance(value, bool):
        return {"type": "boolean", "default": value}
    if isinstance(value, int):
        return {"type": "integer", "default": value}
    if isinstance(value, float):
        return {"type": "number", "default": value}
    if value is None:
        return {"type": "null", "default": None}
    return {"type": "string", "default": value}


def configuration_schema() -> dict[str, Any]:
    """Return the complete strict public configuration schema."""

    defaults = default_config().normalized()
    schema = _schema_for_default(defaults)
    schema.update(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/fohofmann/BodyComposition/config.schema.json",
            "title": "BodyComposition PipelineConfig",
        }
    )
    properties = schema["properties"]
    properties["schema_version"] = {"const": CONFIG_SCHEMA_VERSION}
    properties["analysis"]["properties"]["scope"]["enum"] = [
        "full_ct",
        "l3_vertebral_level",
    ]
    properties["analysis"]["properties"]["l3"]["properties"][
        "inference_context_mm"
    ]["minimum"] = 0
    properties["orientation"]["properties"]["backend"]["enum"] = ["ctdeeprot_2d_v1"]
    properties["orientation"]["properties"]["policy"]["enum"] = ["check_and_safe_repair"]
    properties["orientation"]["properties"]["model_device"]["enum"] = ["auto", "cpu", "cuda"]
    properties["vertebrae"]["properties"]["backend"]["enum"] = [
        "spineps_veridah_ct_v1",
        "vertebral_bodies_resenc_l",
        "vertebral_bodies_resenc_m",
    ]
    properties["vertebrae"]["properties"]["device"]["enum"] = ["auto", "cpu", "cuda"]
    properties["tissue"]["properties"]["backend"]["enum"] = list(
        TISSUE_BACKEND_IDS
    )
    properties["body_surface"]["properties"]["backend"]["enum"] = [
        "tissue_segmentation_envelope_v1",
        "totalsegmentator_body_task299_v1",
        "deterministic_body_mask_v1",
    ]
    operation_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["dimensionality", "threshold", "unit", "connectivity"],
        "properties": {
            "dimensionality": {"enum": ["2D", "3D"]},
            "threshold": {"type": "number", "minimum": 0},
            "unit": {"enum": ["physical", "voxel"]},
            "connectivity": {"enum": [4, 6, 8, 18, 26]},
        },
    }
    definition_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["enabled", "source_labels", "hu_range"],
        "properties": {
            "enabled": {"type": "boolean"},
            "source_labels": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "hu_range": {
                "oneOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "prefixItems": [{"type": "number"}, {"type": "number"}],
                        "minItems": 2,
                        "maxItems": 2,
                    },
                ]
            },
            "preprocessing": {
                "type": "object",
                "additionalProperties": False,
                "required": ["method"],
                "properties": {
                    "method": {
                        "enum": [
                            "none",
                            "median",
                            "adaptive_median",
                            "curvature_anisotropic_diffusion",
                        ]
                    },
                    "clip_hu_range": {
                        "oneOf": [
                            {"type": "null"},
                            {
                                "type": "array",
                                "prefixItems": [
                                    {"type": "number"},
                                    {"type": "number"},
                                ],
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        ]
                    },
                    "kernel_zyx": {
                        "type": "array",
                        "prefixItems": [
                            {"type": "integer"},
                            {"type": "integer"},
                            {"type": "integer"},
                        ],
                        "minItems": 3,
                        "maxItems": 3,
                    },
                    "minimum_kernel_zyx": {
                        "type": "array",
                        "prefixItems": [
                            {"type": "integer"},
                            {"type": "integer"},
                            {"type": "integer"},
                        ],
                        "minItems": 3,
                        "maxItems": 3,
                    },
                    "maximum_kernel_zyx": {
                        "type": "array",
                        "prefixItems": [
                            {"type": "integer"},
                            {"type": "integer"},
                            {"type": "integer"},
                        ],
                        "minItems": 3,
                        "maxItems": 3,
                    },
                    "anisotropic_diffusion": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "dimensionality",
                            "iterations",
                            "time_step",
                            "conductance",
                        ],
                        "properties": {
                            "dimensionality": {"enum": ["2D", "3D"]},
                            "iterations": {"type": "integer", "minimum": 1},
                            "time_step": {"type": "number", "exclusiveMinimum": 0},
                            "conductance": {"type": "number", "exclusiveMinimum": 0},
                        },
                    },
                },
            },
            "cleanup": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "fill_small_holes": operation_schema,
                    "remove_small_objects": operation_schema,
                },
            },
        },
    }
    definitions_schema = properties["measurements"]["properties"][
        "tissue_definitions"
    ]
    definitions_schema["propertyNames"] = {"pattern": "^[a-z][a-z0-9_]*$"}
    definitions_schema["additionalProperties"] = definition_schema
    properties["runtime"]["properties"]["device"]["enum"] = ["auto", "cpu", "cuda"]
    properties["output"]["properties"]["log_level"]["enum"] = ["DEBUG", "INFO", "WARNING", "ERROR"]
    properties["models"]["properties"]["root"]["minLength"] = 1
    return schema
