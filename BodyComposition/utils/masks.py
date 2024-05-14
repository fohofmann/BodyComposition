# libraries
import cv2
import numpy as np
from skimage import measure
import logging
from scipy.ndimage import label as ndi_label, sum as ndi_sum, median_filter as ndi_median_filter

# function to remove small objects, per slice (2d)
def _remove_small_objects_2d(mask, min_size=10):
    components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    sizes = stats[1:, -1]
    mask = np.zeros(output.shape)
    for i in range(0, components - 1):
        if sizes[i] >= min_size:
            mask[output == i + 1] = 1
    return mask

# function to remove small objects, 3d volume
def _remove_small_objects_3d(mask, min_size=100, min_extent=0.1):
    labels = measure.label(mask, connectivity=3)
    props = measure.regionprops(labels)
    labels_filtered = [prop.label for prop in props if prop.area >= min_size and prop.extent >= min_extent]
    return np.isin(labels, labels_filtered)


# function to filter HU range
def filter_hu(image_np: np.ndarray, hu_range: list):
    logging.info(f"  filter: HU= {min(hu_range)} to {max(hu_range)}")
    return (image_np >= min(hu_range)) & (image_np <= max(hu_range))


# function to remove small objects
def remove_small_objects(mask_np, image_zooms, limit_size_version,
                         limit_size_2D = 0, limit_size_3D = 0):

    # calculate pixel volume (RAS+)
    pix_area = image_zooms[0] * image_zooms[1]
    pix_vol = pix_area * image_zooms[2]
    
    # remove small objects, 2d or 3d
    if limit_size_version == '2D' and limit_size_2D > 0:
        for i in range(mask_np.shape[-1]):
            mask_np[:, :, i][~_remove_small_objects_2d(mask_np[:, :, i], limit_size_2D/pix_area)] = 0
        logging.info(f"  removed small objects (2D size < {limit_size_2D} mm^2)")
    elif limit_size_version == '3D' and limit_size_3D > 0:
        mask_np[:] = _remove_small_objects_3d(mask_np,
                                              min_size=limit_size_3D/pix_vol,
                                              min_extent=0)
        logging.info(f"  removed small objects (3D size < {limit_size_3D} mm^3)")



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