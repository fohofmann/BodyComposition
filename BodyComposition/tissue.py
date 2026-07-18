"""Configurable CT tissue-classification preprocessing and cleanup.

Anatomical compartment segmentation, HU classification, and phenotype
interpretation are deliberately separate.  The functions in this module are
used only to reproduce an explicitly configured classification branch; raw-HU
measurement continues to use the untouched prepared CT.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    maximum_filter,
    median_filter,
    minimum_filter,
)

from BodyComposition.utils.masks import fill_small_holes, remove_small_objects


def _adaptive_median_filter(
    image_zyx: np.ndarray,
    minimum_kernel_zyx: tuple[int, int, int],
    maximum_kernel_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Apply the standard two-stage adaptive median algorithm."""

    image = np.asarray(image_zyx)
    output = image.astype(np.float32, copy=True)
    unresolved = np.ones(image.shape, dtype=bool)
    last_median = output
    minimum = np.asarray(minimum_kernel_zyx, dtype=int)
    maximum = np.asarray(maximum_kernel_zyx, dtype=int)
    kernels: list[tuple[int, int, int]] = []
    current = minimum.copy()
    while np.all(current <= maximum):
        kernels.append(tuple(int(value) for value in current))
        next_kernel = current.copy()
        expandable = current < maximum
        next_kernel[expandable] = np.minimum(
            current[expandable] + 2,
            maximum[expandable],
        )
        if np.array_equal(next_kernel, current):
            break
        current = next_kernel

    for kernel in kernels:
        local_min = minimum_filter(image, size=kernel, mode="nearest")
        local_max = maximum_filter(image, size=kernel, mode="nearest")
        local_median = median_filter(image, size=kernel, mode="nearest")
        last_median = local_median
        stage_a = unresolved & (local_median > local_min) & (local_median < local_max)
        stage_b = (image > local_min) & (image < local_max)
        output[stage_a] = np.where(
            stage_b[stage_a],
            image[stage_a],
            local_median[stage_a],
        )
        unresolved[stage_a] = False
    output[unresolved] = last_median[unresolved]
    return output


def _anisotropic_diffusion(
    image_zyx: np.ndarray,
    spacing_xyz,
    settings: Mapping[str, object],
) -> np.ndarray:
    dimensionality = str(settings["dimensionality"])

    def execute(array: np.ndarray, spacing) -> np.ndarray:
        image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
        image.SetSpacing(tuple(float(value) for value in spacing))
        diffusion = sitk.CurvatureAnisotropicDiffusionImageFilter()
        diffusion.SetNumberOfIterations(int(settings["iterations"]))
        diffusion.SetTimeStep(float(settings["time_step"]))
        diffusion.SetConductanceParameter(float(settings["conductance"]))
        return sitk.GetArrayFromImage(diffusion.Execute(image))

    if dimensionality == "2D":
        return np.stack(
            [
                execute(slice_yx, tuple(spacing_xyz[:2]))
                for slice_yx in np.asarray(image_zyx)
            ],
            axis=0,
        )
    if dimensionality == "3D":
        return execute(image_zyx, tuple(spacing_xyz))
    raise ValueError("Anisotropic diffusion dimensionality must be 2D or 3D.")


def prepare_classification_image(
    image_zyx: np.ndarray,
    spacing_xyz,
    settings: Mapping[str, object],
) -> np.ndarray:
    """Return the configured image used for HU *classification* only."""

    image = np.asarray(image_zyx)
    output = image
    if bool(settings["filter_outliers"]):
        hu_range = settings["filter_outliers_range"]
        output = np.clip(output, min(hu_range), max(hu_range))
        logging.debug("  clipped classification HU values, range=%s", hu_range)

    method = str(settings.get("method", "none"))
    if method == "none":
        return output
    if method == "median":
        kernel = tuple(int(value) for value in settings["filter_median_kernel"])
        logging.debug("  applied classification median filter, kernel=%s", kernel)
        return median_filter(output, size=kernel, mode="nearest")
    if method == "adaptive_median":
        minimum_kernel = tuple(
            int(value) for value in settings["adaptive_median_min_kernel"]
        )
        maximum_kernel = tuple(
            int(value) for value in settings["adaptive_median_max_kernel"]
        )
        logging.debug(
            "  applied adaptive classification median filter, kernels=%s..%s",
            minimum_kernel,
            maximum_kernel,
        )
        return _adaptive_median_filter(output, minimum_kernel, maximum_kernel)
    if method == "curvature_anisotropic_diffusion":
        logging.debug(
            "  applied curvature anisotropic diffusion for classification, settings=%s",
            settings["anisotropic_diffusion"],
        )
        return _anisotropic_diffusion(
            output,
            np.asarray(spacing_xyz, dtype=float),
            settings["anisotropic_diffusion"],
        )
    raise ValueError(f"Unknown HU classification preprocessing method {method!r}.")


def prepare_classification_images(
    image_zyx: np.ndarray,
    spacing_xyz,
    settings: Mapping[str, object],
    tissue_names=("imat", "sm", "vat", "sat"),
) -> dict[str, np.ndarray]:
    """Return raw or preprocessed classification images by tissue rule."""

    names = tuple(str(name) for name in tissue_names)
    apply_to = set(settings.get("apply_to", names))
    processed = prepare_classification_image(image_zyx, spacing_xyz, settings)
    raw = np.asarray(image_zyx)
    return {
        name: processed if name in apply_to else raw
        for name in names
    }


def cleanup_tissue_mask(
    mask_zyx: np.ndarray,
    settings: Mapping[str, object],
    spacing_xyz,
    *,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply configured cleanup in-place inside a fixed anatomical support."""

    mask = np.asarray(mask_zyx, dtype=bool)
    support = None
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != mask.shape:
            raise ValueError("support_mask must have the same shape as mask_zyx.")
        mask &= support
    if bool(settings.get("fill_holes", False)):
        fill_small_holes(
            mask,
            spacing_xyz,
            settings["fill_holes_version"],
            settings["fill_holes_2D"],
            settings["fill_holes_3D"],
            size_unit=settings["fill_holes_unit"],
            connectivity=settings["fill_holes_connectivity"],
        )
        if support is not None:
            mask &= support
    if bool(settings["filter_size"]):
        remove_small_objects(
            mask,
            spacing_xyz,
            settings["filter_size_version"],
            settings["filter_size_2D"],
            settings["filter_size_3D"],
            size_unit=settings["filter_size_unit"],
            connectivity=settings["filter_size_connectivity"],
        )
        if support is not None:
            mask &= support
    return mask
