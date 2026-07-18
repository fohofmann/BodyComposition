# libraries
import logging

import numpy as np
from scipy.ndimage import label as ndi_label
from scipy.ndimage import sum as ndi_sum
from skimage import measure


def _connectivity_rank(ndim: int, connectivity: int) -> int:
    mapping = {
        2: {4: 1, 8: 2},
        3: {6: 1, 18: 2, 26: 3},
    }
    try:
        return mapping[ndim][connectivity]
    except KeyError as error:
        allowed = sorted(mapping.get(ndim, {}))
        raise ValueError(
            f"Connectivity for a {ndim}D mask must be one of {allowed}, "
            f"got {connectivity}."
        ) from error


def _component_size_threshold(
    value: float,
    *,
    size_unit: str,
    physical_voxel_size: float,
) -> float:
    if size_unit == "physical":
        return float(value) / physical_voxel_size
    if size_unit == "voxel":
        return float(value)
    raise ValueError("size_unit must be 'physical' or 'voxel'.")

# function to remove small objects, per slice (2d)
def _remove_small_objects_2d(mask, min_size=10, connectivity=8):
    labels = measure.label(
        np.asarray(mask, dtype=bool),
        connectivity=_connectivity_rank(2, connectivity),
    )
    sizes = np.bincount(labels.ravel())
    keep = sizes >= float(min_size)
    keep[0] = False
    return keep[labels]

# function to remove small objects, 3d volume
def _remove_small_objects_3d(
    mask,
    min_size=100,
    min_extent=0.1,
    connectivity=26,
):
    labels = measure.label(
        np.asarray(mask, dtype=bool),
        connectivity=_connectivity_rank(3, connectivity),
    )
    props = measure.regionprops(labels)
    labels_filtered = [
        prop.label
        for prop in props
        if prop.area >= float(min_size) and prop.extent >= min_extent
    ]
    return np.isin(labels, labels_filtered)


# function to filter HU range
def filter_hu(image_np: np.ndarray, hu_range: list):
    logging.info(f"  filter: HU= {min(hu_range)} to {max(hu_range)}")
    return (image_np >= min(hu_range)) & (image_np <= max(hu_range))


# function to remove small objects
def remove_small_objects(
    mask_np,
    spacing_xyz,
    limit_size_version,
    limit_size_2D=0,
    limit_size_3D=0,
    *,
    size_unit="physical",
    connectivity=None,
):
    """Remove connected components below a configured threshold in-place.

    ``size_unit='physical'`` interprets the 2D/3D thresholds as mm2/mm3.
    ``size_unit='voxel'`` interprets them as pixels/voxels, which is retained
    only for reproducing resolution-dependent published implementations.
    """

    spacing_xyz = np.asarray(spacing_xyz, dtype=float)
    if spacing_xyz.shape != (3,) or not np.all(np.isfinite(spacing_xyz)) or np.any(spacing_xyz <= 0):
        raise ValueError(f'spacing_xyz must contain three finite positive values, got {spacing_xyz}.')

    # Array slices are y-x planes; SimpleITK spacing is x-y-z.
    pix_area = spacing_xyz[0] * spacing_xyz[1]
    pix_vol = float(np.prod(spacing_xyz))
    
    # remove small objects, 2d or 3d
    if limit_size_version == '2D' and limit_size_2D > 0:
        connectivity = 8 if connectivity is None else int(connectivity)
        threshold = _component_size_threshold(
            limit_size_2D,
            size_unit=size_unit,
            physical_voxel_size=pix_area,
        )
        for i in range(mask_np.shape[0]):
            mask_np[i, :, :] = _remove_small_objects_2d(
                mask_np[i, :, :],
                threshold,
                connectivity=connectivity,
            )
        unit = "mm^2" if size_unit == "physical" else "pixels"
        logging.info(
            "  removed small objects (2D size < %s %s; connectivity=%s)",
            limit_size_2D,
            unit,
            connectivity,
        )
    elif limit_size_version == '3D' and limit_size_3D > 0:
        connectivity = 26 if connectivity is None else int(connectivity)
        threshold = _component_size_threshold(
            limit_size_3D,
            size_unit=size_unit,
            physical_voxel_size=pix_vol,
        )
        mask_np[:] = _remove_small_objects_3d(mask_np,
                                              min_size=threshold,
                                              min_extent=0,
                                              connectivity=connectivity)
        unit = "mm^3" if size_unit == "physical" else "voxels"
        logging.info(
            "  removed small objects (3D size < %s %s; connectivity=%s)",
            limit_size_3D,
            unit,
            connectivity,
        )


def _fill_small_holes_nd(mask: np.ndarray, maximum_size: float, connectivity: int) -> np.ndarray:
    """Fill enclosed background components smaller than ``maximum_size``."""

    boolean = np.asarray(mask, dtype=bool)
    labels = measure.label(
        ~boolean,
        connectivity=_connectivity_rank(boolean.ndim, connectivity),
    )
    if not np.any(labels):
        return boolean.copy()
    border = np.concatenate(
        [
            labels.take(0, axis=axis).ravel()
            for axis in range(boolean.ndim)
        ]
        + [
            labels.take(-1, axis=axis).ravel()
            for axis in range(boolean.ndim)
        ]
    )
    exterior = np.unique(border)
    sizes = np.bincount(labels.ravel())
    fill = sizes < float(maximum_size)
    fill[0] = False
    fill[exterior] = False
    return boolean | fill[labels]


def fill_small_holes(
    mask_np,
    spacing_xyz,
    limit_size_version,
    limit_size_2D=0,
    limit_size_3D=0,
    *,
    size_unit="physical",
    connectivity=None,
):
    """Fill small enclosed holes in-place using explicit dimensions and units."""

    spacing_xyz = np.asarray(spacing_xyz, dtype=float)
    if spacing_xyz.shape != (3,) or not np.all(np.isfinite(spacing_xyz)) or np.any(spacing_xyz <= 0):
        raise ValueError(f'spacing_xyz must contain three finite positive values, got {spacing_xyz}.')
    pix_area = float(spacing_xyz[0] * spacing_xyz[1])
    pix_vol = float(np.prod(spacing_xyz))

    if limit_size_version == "2D" and limit_size_2D > 0:
        connectivity = 8 if connectivity is None else int(connectivity)
        threshold = _component_size_threshold(
            limit_size_2D,
            size_unit=size_unit,
            physical_voxel_size=pix_area,
        )
        for index in range(mask_np.shape[0]):
            mask_np[index] = _fill_small_holes_nd(
                mask_np[index],
                threshold,
                connectivity,
            )
        unit = "mm^2" if size_unit == "physical" else "pixels"
        logging.info(
            "  filled holes (2D size < %s %s; connectivity=%s)",
            limit_size_2D,
            unit,
            connectivity,
        )
    elif limit_size_version == "3D" and limit_size_3D > 0:
        connectivity = 26 if connectivity is None else int(connectivity)
        threshold = _component_size_threshold(
            limit_size_3D,
            size_unit=size_unit,
            physical_voxel_size=pix_vol,
        )
        mask_np[:] = _fill_small_holes_nd(mask_np, threshold, connectivity)
        unit = "mm^3" if size_unit == "physical" else "voxels"
        logging.info(
            "  filled holes (3D size < %s %s; connectivity=%s)",
            limit_size_3D,
            unit,
            connectivity,
        )



# function to keep only the largest object
# - avoids memory reallocation by modifying the input mask
def filter_keep_largest(mask_np, labels=None):
    if labels is None:
        labels = np.unique(mask_np)[1:]
    for label in labels:
        mask = mask_np == label
        if np.any(mask):
            labeled, num_labels = ndi_label(mask)
            sizes = ndi_sum(mask, labeled, index=range(1, num_labels+1))
            largest_label = np.argmax(sizes) + 1
            mask_np[(labeled != largest_label) & mask] = 0


# function to fill holes in mask of single objects
# - inverts mask, largest component = surrounding / rest = hole, inverts back
# - can make problems with border touching objects, most are adressed by padding
def fill_holes(mask_np, labels=None):
    mask_framed = np.pad(mask_np, pad_width=1, mode='constant', constant_values=0)
    if labels is None:
        labels = np.unique(mask_np)[1:]
    for label in labels:
        mask = mask_framed == label
        if np.any(mask):
            np.invert(mask, out=mask)
            labeled, num_labels = ndi_label(mask)
            if num_labels <= 1:
                continue
            sizes = ndi_sum(mask, labeled, index=range(1, num_labels+1))
            largest_label = np.argmax(sizes) + 1
            mask[labeled != largest_label] = 0
            np.invert(mask, out=mask)
            mask_framed[mask] = label
    mask_np[:] = mask_framed[1:-1, 1:-1, 1:-1]
