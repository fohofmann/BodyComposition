import math

import numpy as np
import pandas as pd
import pytest

from BodyComposition.measurements import (
    MeasurementError,
    calculate_slice_measurements,
    contour_metrics,
    validate_label_mapping,
    weighted_mean_hu,
)
from BodyComposition.utils.geometry import ImageGeometry


def make_geometry(shape_zyx, spacing_xyz=(1.0, 1.0, 2.0)):
    return ImageGeometry(
        size_xyz=tuple(reversed(shape_zyx)),
        spacing_xyz=spacing_xyz,
        origin_lps_xyz=(2.0, -3.0, 5.0),
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def test_integer_labels_produce_named_columns():
    mask = np.zeros((2, 4, 5), dtype=np.uint8)
    mask[0, :2, :3] = 1
    mask[1, 1:4, 2:5] = 3
    measured = calculate_slice_measurements(mask, make_geometry(mask.shape), {1: "SM", 3: "SAT"})

    assert list(measured.columns) == ["Vx_SM", "CSA_SM", "Vx_SAT", "CSA_SAT"]
    assert measured.loc[0, "Vx_SM"] == 6
    assert measured.loc[1, "Vx_SAT"] == 9


def test_hu_means_are_null_with_reason_for_empty_tissue_and_weighted_by_voxels():
    mask = np.zeros((3, 2, 3), dtype=np.uint8)
    image = np.zeros(mask.shape, dtype=np.int16)
    mask[0, 0, :2] = 1
    image[0, 0, :2] = [10, 20]
    mask[2, 0, :3] = 1
    image[2, 0, :3] = [30, 40, 50]

    measured = calculate_slice_measurements(
        mask,
        make_geometry(mask.shape),
        {1: "SM"},
        image_zyx=image,
    )
    assert measured.loc[0, "HU_SM"] == pytest.approx(15.0)
    assert np.isnan(measured.loc[1, "HU_SM"])
    assert measured.loc[1, "HU_status_SM"] == "empty_tissue"
    value, status = weighted_mean_hu(measured["Vx_SM"], measured["HU_SM"])
    assert value == pytest.approx((2 * 15 + 3 * 40) / 5)
    assert status == "ok"

    empty_value, empty_status = weighted_mean_hu(
        pd.Series([0, 0]),
        pd.Series([np.nan, np.nan]),
    )
    assert np.isnan(empty_value)
    assert empty_status == "empty_tissue"


@pytest.mark.parametrize("spacing_xyz", [(1.0, 1.0, 2.0), (0.7, 1.3, 2.0)])
def test_rectangle_contour_matches_physical_reference(spacing_xyz):
    shape_zyx = (1, 70, 90)
    geometry = make_geometry(shape_zyx, spacing_xyz)
    mask = np.zeros(shape_zyx[1:], dtype=bool)
    height_px, width_px = 30, 40
    mask[15 : 15 + height_px, 20 : 20 + width_px] = True

    perimeter_cm, area_cm2, status = contour_metrics(mask, geometry, 0)
    expected_area_cm2 = width_px * spacing_xyz[0] * height_px * spacing_xyz[1] / 100
    expected_perimeter_cm = 2 * (
        width_px * spacing_xyz[0] + height_px * spacing_xyz[1]
    ) / 10
    assert status == "ok"
    assert area_cm2 == pytest.approx(expected_area_cm2, rel=0.01)
    assert perimeter_cm == pytest.approx(expected_perimeter_cm, rel=0.02)


@pytest.mark.parametrize("spacing_xyz", [(1.0, 1.0, 2.0), (0.8, 1.4, 2.0)])
def test_ellipse_contour_matches_discrete_analytic_reference(spacing_xyz):
    shape_zyx = (1, 100, 120)
    geometry = make_geometry(shape_zyx, spacing_xyz)
    yy, xx = np.ogrid[: shape_zyx[1], : shape_zyx[2]]
    radius_x_px, radius_y_px = 30, 20
    ellipse = ((xx - 60) / radius_x_px) ** 2 + ((yy - 50) / radius_y_px) ** 2 <= 1

    perimeter_cm, area_cm2, status = contour_metrics(ellipse, geometry, 0)
    radius_x_mm = radius_x_px * spacing_xyz[0]
    radius_y_mm = radius_y_px * spacing_xyz[1]
    expected_area_cm2 = math.pi * radius_x_mm * radius_y_mm / 100
    h = ((radius_x_mm - radius_y_mm) / (radius_x_mm + radius_y_mm)) ** 2
    expected_perimeter_cm = (
        math.pi
        * (radius_x_mm + radius_y_mm)
        * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
        / 10
    )
    assert status == "ok"
    assert area_cm2 == pytest.approx(expected_area_cm2, rel=0.04)
    assert perimeter_cm == pytest.approx(expected_perimeter_cm, rel=0.04)


def test_label_mapping_rejects_names_as_keys():
    with pytest.raises(MeasurementError, match="positive integers"):
        validate_label_mapping({"SM": 1})
