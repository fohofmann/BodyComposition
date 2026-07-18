from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from BodyComposition.config import ConfigError, PipelineConfig
from BodyComposition.utils.config import validate_config


def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
    target = config
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        ("paths.workspace", 1),
        ("paths.weights", "not-a-mapping"),
        ("logging_level.file", True),
        ("run.reset", "true"),
        ("run.timeout", 0),
        ("orientation.confidence.min_body_extent_mm", -1),
        ("orientation.confidence.body_threshold_hu", True),
        ("orientation.confidence.min_equivariant_votes", 0),
        ("orientation.confidence.min_body_pixels_per_slice", 0),
        ("orientation.artifact_rules.max_metal_fraction", 1.1),
        ("orientation.model.checkpoint_path", 1),
        ("orientation.model.device", "mps"),
        ("orientation.model.batch_size", 25),
        ("orientation.header_uncertain_reasons", [1]),
        ("vertebrae.min_voxels_per_vertebra", -1),
        ("vertebrae.deprioritize_labels", [True]),
        ("vertebrae.backend", "unknown"),
        ("vertebrae.spineps.device", "mps"),
        ("paths.weights.spineps", 1),
        ("vertebrae.deprioritize_anatomical", [""]),
        ("tissue.profile_id", "Not Canonical"),
        ("tissue.hu_denoise.filter_outliers_range", [1]),
        ("tissue.hu_denoise.filter_outliers_range", [1, -1]),
        ("tissue.hu_denoise.filter_median_kernel", [2, 3, 3]),
        ("tissue.hu_denoise.method", "unknown"),
        ("tissue.hu_denoise.apply_to", ["sm", "sm"]),
        ("tissue.hu_denoise.apply_to", [["sm"]]),
        ("tissue.hu_denoise.anisotropic_diffusion.dimensionality", "4D"),
        ("tissue.hu_denoise.anisotropic_diffusion.iterations", 0),
        ("tissue.hu_denoise.anisotropic_diffusion.conductance", 0),
        ("tissue.hu_denoise.anisotropic_diffusion.time_step", 1.0),
        ("tissue.sm.filter_size", "true"),
        ("tissue.sm.filter_size_version", "4D"),
        ("tissue.sm.filter_size_unit", "pixels"),
        ("tissue.sm.filter_size_connectivity", 4),
        ("tissue.sm.fill_holes_version", "4D"),
        ("tissue.sm.fill_holes_unit", "pixels"),
        ("tissue.sm.fill_holes_connectivity", 26),
        ("measurements.totalsegmentator_version", "latest"),
        ("measurements.body_surface.backend", "unknown"),
        ("measurements.landmarks.backend", "unknown"),
        ("measurements.body_surface.min_component_volume_mm3", 0),
        ("measurements.body_surface.closing_radius_mm", -1),
        ("measurements.vertebral_extent.maximum_removed_fraction", 1),
        ("measurements.l3_slab_length_mm", 0),
        ("measurements.full_coverage_tolerance", 1.1),
        ("measurements.landmarks.minimum_voxels", 0),
        ("measurements.body_surface.threshold_hu", True),
        ("measurements.export.parquet", False),
        ("measurements.export.csv", True),
        ("LBL_TISSUE", {}),
        ("LBL_VERTEBRALBODIES", {0: "invalid"}),
    ],
)
def test_runtime_configuration_rejects_invalid_safety_values(base_config, path, invalid):
    config = deepcopy(base_config)
    _set_path(config, path, invalid)

    with pytest.raises(ConfigError):
        validate_config(config)


def test_runtime_configuration_rejects_missing_and_non_mapping_sections(base_config):
    missing = deepcopy(base_config)
    del missing["paths"]
    with pytest.raises(ConfigError, match="Missing configuration value: paths"):
        validate_config(missing)

    wrong_type = deepcopy(base_config)
    wrong_type["paths"] = []
    with pytest.raises(ConfigError, match="paths must be a mapping"):
        validate_config(wrong_type)

    with pytest.raises(ConfigError, match="must be a mapping"):
        validate_config([])


def test_runtime_configuration_checks_cross_field_constraints(base_config):
    kernels = deepcopy(base_config)
    kernels["tissue"]["hu_denoise"]["adaptive_median_min_kernel"] = [5, 5, 5]
    kernels["tissue"]["hu_denoise"]["adaptive_median_max_kernel"] = [3, 3, 3]
    with pytest.raises(ConfigError, match="minimum kernel"):
        validate_config(kernels)

    reporting = deepcopy(base_config)
    reporting["reporting"]["enabled"] = True
    reporting["measurements"]["enabled"] = False
    with pytest.raises(ConfigError, match="reporting.enabled requires"):
        validate_config(reporting)


@pytest.mark.parametrize(
    "definition",
    [
        {"Bad-Name": {"enabled": True, "source_labels": ["SM"], "hu_range": None}},
        {"muscle_compartment": "not-a-mapping"},
        {"muscle_compartment": {"enabled": "true", "source_labels": ["SM"]}},
        {"muscle_compartment": {"enabled": True, "source_labels": []}},
        {
            "muscle_compartment": {
                "enabled": True,
                "source_labels": ["SM"],
                "hu_range": [1],
            }
        },
        {
            "muscle_compartment": {
                "enabled": True,
                "source_labels": ["SM"],
                "hu_range": [1, -1],
            }
        },
    ],
)
def test_runtime_configuration_rejects_malformed_tissue_definitions(
    base_config,
    definition,
):
    config = deepcopy(base_config)
    config["measurements"]["tissue_definitions"] = definition

    with pytest.raises(ConfigError):
        validate_config(config)


def test_reporting_requires_persisted_tissue_labels_for_regeneration():
    with pytest.raises(ConfigError, match="save_tissue_labels=true"):
        PipelineConfig.model_validate(
            {
                "reporting": {"enabled": True},
                "output": {"save_tissue_labels": False},
            }
        )


def test_runtime_configuration_keeps_canonical_tissue_definitions_enabled(base_config):
    missing = deepcopy(base_config)
    del missing["measurements"]["tissue_definitions"]["muscle_compartment"]
    with pytest.raises(ConfigError, match="must remain enabled"):
        validate_config(missing)

    disabled = deepcopy(base_config)
    disabled["measurements"]["tissue_definitions"]["muscle_compartment"]["enabled"] = False
    with pytest.raises(ConfigError, match="must remain enabled"):
        validate_config(disabled)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("roi", [True]),
        ("roi_anatomical", [""]),
        ("axes", [True]),
        ("margin", [0]),
    ],
)
def test_runtime_configuration_rejects_malformed_crop_contracts(
    base_config,
    field,
    invalid,
):
    config = deepcopy(base_config)
    config["crop"]["test"] = {
        "roi": [1],
        "roi_anatomical": ["L3"],
        "axes": [True, True, True, True, True, True],
        "margin": [0, 0, 0, 0, 0, 0],
    }
    config["crop"]["test"][field] = invalid

    with pytest.raises(ConfigError):
        validate_config(config)
