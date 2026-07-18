"""Deterministic identifier-free measurement QC image."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import SimpleITK as sitk
from skimage.segmentation import find_boundaries

from BodyComposition.measurement.aggregation import select_l3_view
from BodyComposition.measurement.contracts import MeasurementBundle
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain

PANEL_SIZE = 320


def _window_ct(array_yx: np.ndarray, lower: float = -200.0, upper: float = 300.0) -> np.ndarray:
    clipped = np.clip(array_yx.astype(np.float32), lower, upper)
    scaled = ((clipped - lower) / (upper - lower) * 255.0).astype(np.uint8)
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)


def _fit_panel(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min((PANEL_SIZE - 20) / height, (PANEL_SIZE - 20) / width)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((PANEL_SIZE, PANEL_SIZE, 3), 20, dtype=np.uint8)
    y = (PANEL_SIZE - resized.shape[0]) // 2
    x = (PANEL_SIZE - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _panel(
    image_zyx: np.ndarray,
    bundle: MeasurementBundle,
    slice_id: int | None,
    title: str,
) -> np.ndarray:
    if slice_id is None or not 0 <= slice_id < image_zyx.shape[0]:
        panel = np.full((PANEL_SIZE, PANEL_SIZE, 3), 20, dtype=np.uint8)
        cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
        cv2.putText(panel, "not available", (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 170, 255), 2)
        return panel
    image = _window_ct(image_zyx[slice_id])
    body_boundary = find_boundaries(bundle.body_surface.body_mask_zyx[slice_id], mode="outer")
    trunk_boundary = find_boundaries(bundle.body_surface.trunk_mask_zyx[slice_id], mode="outer")
    image[body_boundary] = (255, 180, 0)
    image[trunk_boundary] = (40, 220, 40)
    panel = _fit_panel(image)
    cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
    selected = bundle.slices.loc[bundle.slices["slice_id"].eq(slice_id)]
    qc_status = str(selected.iloc[0]["slice_qc_status"]) if not selected.empty else "unknown"
    qc_color = (40, 220, 40) if qc_status == "pass" else (40, 170, 255)
    cv2.putText(
        panel,
        f"QC: {qc_status}",
        (10, PANEL_SIZE - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        qc_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"prepared slice {slice_id}",
        (10, PANEL_SIZE - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return panel


def _optional_slice_id(value) -> int | None:
    try:
        if value is None or bool(np.isnan(value)):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _curve_panel(bundle: MeasurementBundle, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 20, dtype=np.uint8)
    table = bundle.slices
    position = table["position_superior_mm"].to_numpy(dtype=float)
    valid = table["trunk_circumference_cm"].notna().to_numpy()
    circumference = table["trunk_circumference_cm"].fillna(0).to_numpy(dtype=float)
    if not np.any(valid):
        cv2.putText(panel, "No valid trunk circumference curve", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 170, 255), 2)
        return panel
    left, right, top, bottom = 70, width - 30, 30, height - 55
    cv2.rectangle(panel, (left, top), (right, bottom), (100, 100, 100), 1)
    x = left + (position - position.min()) / max(np.ptp(position), 1e-9) * (right - left)
    valid_values = circumference[valid]
    y = bottom - (circumference - valid_values.min()) / max(np.ptp(valid_values), 1e-9) * (bottom - top)
    segments: list[np.ndarray] = []
    current: list[tuple[int, int]] = []
    for x_value, y_value, is_valid in zip(x, y, valid, strict=True):
        if is_valid:
            current.append((int(round(x_value)), int(round(y_value))))
        elif current:
            segments.append(np.asarray(current, dtype=np.int32))
            current = []
    if current:
        segments.append(np.asarray(current, dtype=np.int32))
    for segment in segments:
        if len(segment) > 1:
            cv2.polylines(panel, [segment], False, (40, 220, 40), 2, cv2.LINE_AA)
    if "trunk_circumference_jump_flag" in table:
        flagged = table["trunk_circumference_jump_flag"].to_numpy(dtype=bool) & valid
        for x_value, y_value in zip(x[flagged], y[flagged], strict=True):
            cv2.circle(
                panel,
                (int(round(x_value)), int(round(y_value))),
                5,
                (40, 40, 255),
                -1,
                cv2.LINE_AA,
            )
    review = table["slice_qc_status"].ne("pass").to_numpy(dtype=bool) & valid
    for x_value, y_value in zip(x[review], y[review], strict=True):
        point = (int(round(x_value)), int(round(y_value)))
        cv2.drawMarker(
            panel,
            point,
            (220, 80, 220),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=9,
            thickness=1,
            line_type=cv2.LINE_AA,
        )
    cv2.putText(
        panel,
        "red: jump  magenta: slice review",
        (left + 190, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(panel, "inferior", (left, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(panel, "superior", (right - 65, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(panel, "trunk circumference (cm)", (left, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
    return panel


def _tissue_curve_panel(bundle: MeasurementBundle, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 20, dtype=np.uint8)
    table = bundle.slices
    position = table["position_superior_mm"].to_numpy(dtype=float)
    curves = (
        ("sm_area_cm2", "SM", (80, 180, 255)),
        ("imat_area_cm2", "IMAT", (180, 120, 255)),
        ("sat_area_cm2", "SAT", (80, 220, 220)),
        ("total_vat_area_cm2", "total VAT", (40, 80, 230)),
        ("avat_area_cm2", "aVAT", (80, 80, 180)),
        ("tvat_area_cm2", "tVAT", (140, 70, 220)),
    )
    available = [
        (column, label, color)
        for column, label, color in curves
        if column in table and table[column].notna().any()
    ]
    if not available:
        cv2.putText(
            panel,
            "No valid tissue-area curves",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 170, 255),
            2,
        )
        return panel
    left, right, top, bottom = 70, width - 30, 35, height - 50
    cv2.rectangle(panel, (left, top), (right, bottom), (100, 100, 100), 1)
    x = left + (position - position.min()) / max(np.ptp(position), 1e-9) * (right - left)
    maximum = max(float(table[column].max(skipna=True)) for column, _, _ in available)
    for curve_index, (column, label, color) in enumerate(available):
        values = table[column].to_numpy(dtype=float)
        valid = np.isfinite(values)
        y = bottom - values / max(maximum, 1e-9) * (bottom - top)
        segments: list[np.ndarray] = []
        current: list[tuple[int, int]] = []
        for x_value, y_value, is_valid in zip(x, y, valid, strict=True):
            if is_valid:
                current.append((int(round(x_value)), int(round(y_value))))
            elif current:
                segments.append(np.asarray(current, dtype=np.int32))
                current = []
        if current:
            segments.append(np.asarray(current, dtype=np.int32))
        for segment in segments:
            if len(segment) > 1:
                cv2.polylines(panel, [segment], False, color, 2, cv2.LINE_AA)
        legend_x = left + curve_index * 150
        cv2.line(panel, (legend_x, 19), (legend_x + 22, 19), color, 2, cv2.LINE_AA)
        cv2.putText(
            panel,
            label,
            (legend_x + 28, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        panel,
        "tissue area (cm2)",
        (8, top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (230, 230, 230),
        1,
    )
    cv2.putText(
        panel,
        "inferior",
        (left, bottom + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        panel,
        "superior",
        (right - 65, bottom + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )
    return panel


def write_measurement_review(
    prepared_image: sitk.Image,
    bundle: MeasurementBundle,
    output_path: str | Path,
) -> Path:
    """Write axial contour checks and the acquired-slice circumference curve."""

    geometry = ImageGeometry.from_sitk(prepared_image)
    assert_same_physical_domain(
        geometry,
        bundle.body_surface.geometry,
        reference_name="orientation-prepared CT",
        candidate_name="measurement review body surface",
    )
    image_zyx = sitk.GetArrayFromImage(prepared_image)
    summaries = bundle.summaries.iloc[0]
    l3_view = select_l3_view(bundle.slices, bundle.vertebrae, aggregation="slice")
    l3_slice = (
        _optional_slice_id(l3_view.iloc[0].get("slice_id"))
        if not l3_view.empty and bool(l3_view.iloc[0].get("valid", False))
        else None
    )
    selections = (
        (l3_slice, "L3 centroid slice"),
        (
            _optional_slice_id(summaries.get("ct_min_trunk_circumference_t10_l5_slice_id")),
            "T10-L5 minimum waist",
        ),
        (_optional_slice_id(summaries.get("ct_midwaist_slice_id")), "anatomical mid-waist"),
        (
            _optional_slice_id(summaries.get("ct_max_pelvic_circumference_slice_id")),
            "sacral pelvic maximum",
        ),
    )
    width = PANEL_SIZE * 4
    curve_height = 230
    tissue_height = 240
    canvas = np.full(
        (PANEL_SIZE + curve_height + tissue_height, width, 3),
        20,
        dtype=np.uint8,
    )
    for index, (slice_id, title) in enumerate(selections):
        canvas[:PANEL_SIZE, index * PANEL_SIZE : (index + 1) * PANEL_SIZE] = _panel(
            image_zyx,
            bundle,
            slice_id,
            title,
        )
    canvas[PANEL_SIZE : PANEL_SIZE + curve_height] = _curve_panel(
        bundle,
        width,
        curve_height,
    )
    canvas[PANEL_SIZE + curve_height :] = _tissue_curve_panel(
        bundle,
        width,
        tissue_height,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial.png")
    if not cv2.imwrite(str(temporary), canvas):
        raise OSError(f"Could not write measurement review image: {temporary}.")
    os.replace(temporary, destination)
    return destination
