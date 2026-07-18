"""Deterministic deidentified review image for vertebral results."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

from BodyComposition.orientation.core import OrientationOutcome
from BodyComposition.utils.geometry import ImageGeometry, assert_same_physical_domain
from BodyComposition.vertebral.contracts import VertebralResult

CANVAS_WIDTH = 1800
CANVAS_HEIGHT = 1000


def _geometry(image: sitk.Image) -> ImageGeometry:
    return ImageGeometry(
        size_xyz=image.GetSize(),
        spacing_xyz=image.GetSpacing(),
        origin_lps_xyz=image.GetOrigin(),
        direction_lps=image.GetDirection(),
    )


def _label_image(array_zyx: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array_zyx.astype(np.uint16, copy=False))
    image.CopyInformation(reference)
    return image


def _normalize_ct(array: np.ndarray) -> np.ndarray:
    clipped = np.clip(array.astype(np.float32), -1000.0, 1000.0)
    return np.asarray((clipped + 1000.0) * (255.0 / 2000.0), dtype=np.uint8)


def _color(label: int) -> tuple[int, int, int]:
    hue = int((label * 37) % 180)
    hsv = np.uint8([[[hue, 210, 245]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _projection(
    ct_zyx: np.ndarray,
    labels_zyx: np.ndarray,
    *,
    collapse_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.where(np.any(labels_zyx != 0, axis=tuple(i for i in range(3) if i != collapse_axis)))[0]
    center = int(np.median(occupied)) if occupied.size else labels_zyx.shape[collapse_axis] // 2
    radius = max(1, min(8, labels_zyx.shape[collapse_axis] // 20))
    start = max(0, center - radius)
    stop = min(labels_zyx.shape[collapse_axis], center + radius + 1)
    indices = np.arange(start, stop)
    ct = np.mean(np.take(ct_zyx, indices, axis=collapse_axis), axis=collapse_axis)
    labels = np.max(labels_zyx, axis=collapse_axis)
    # Canonical LPS z indices increase toward superior; display superior at top.
    return np.flipud(_normalize_ct(ct)), np.flipud(labels)


def _render_panel(
    ct: np.ndarray,
    labels: np.ndarray,
    size: tuple[int, int],
    schema: dict[int, str],
) -> np.ndarray:
    base = cv2.cvtColor(ct, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    for label in sorted(int(value) for value in np.unique(labels) if value != 0):
        mask = labels == label
        overlay[mask] = _color(label)
    base = cv2.addWeighted(base, 0.62, overlay, 0.38, 0)
    for label in sorted(int(value) for value in np.unique(labels) if value != 0):
        coordinates = np.argwhere(labels == label)
        if coordinates.size == 0:
            continue
        row, column = np.median(coordinates, axis=0).astype(int)
        cv2.putText(
            base,
            schema.get(label, str(label)),
            (max(2, int(column) - 15), max(14, int(row))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return cv2.resize(base, size, interpolation=cv2.INTER_AREA)


def write_spine_review(
    prepared: OrientationOutcome,
    result: VertebralResult,
    output_path: Path,
    *,
    case_id: str,
) -> Path:
    """Write sagittal/coronal overlays plus native labels and every QC flag."""

    if result.vertebral_body_labels is None or result.geometry is None:
        raise ValueError("A successful vertebral result is required for the review image.")
    reference = prepared.prepared_image
    assert_same_physical_domain(
        _geometry(reference),
        result.geometry,
        reference_name="orientation-prepared CT",
        candidate_name="vertebral result",
    )
    label_image = _label_image(result.vertebral_body_labels, reference)
    oriented_ct = sitk.DICOMOrient(reference, "LPS")
    oriented_labels = sitk.DICOMOrient(label_image, "LPS")
    assert_same_physical_domain(
        _geometry(oriented_ct),
        _geometry(oriented_labels),
        reference_name="review CT",
        candidate_name="review label",
    )
    ct_zyx = sitk.GetArrayFromImage(oriented_ct)
    labels_zyx = sitk.GetArrayFromImage(oriented_labels)
    sagittal_ct, sagittal_labels = _projection(ct_zyx, labels_zyx, collapse_axis=2)
    coronal_ct, coronal_labels = _projection(ct_zyx, labels_zyx, collapse_axis=1)

    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 248, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"Vertebral review | case {case_id} | {result.backend_id}",
        (32, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    schema = dict(result.label_schema)
    sagittal = _render_panel(sagittal_ct, sagittal_labels, (650, 820), schema)
    coronal = _render_panel(coronal_ct, coronal_labels, (500, 820), schema)
    canvas[80:900, 25:675] = sagittal
    canvas[80:900, 700:1200] = coronal
    flag_codes = {flag.code for flag in result.qc_flags}
    if "vertebra_touches_cranial_fov" in flag_codes:
        cv2.line(canvas, (25, 80), (1200, 80), (0, 0, 220), 4)
    if "vertebra_touches_caudal_fov" in flag_codes:
        cv2.line(canvas, (25, 899), (1200, 899), (0, 0, 220), 4)
    cv2.putText(canvas, "Sagittal thick slab", (25, 935), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    cv2.putText(canvas, "Coronal thick slab", (700, 935), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)

    present = sorted(
        int(value) for value in np.unique(result.vertebral_body_labels) if value != 0
    )
    lines = [
        f"Execution: {result.execution_status.value}",
        f"QC: {result.qc_status.value}",
        f"Orientation changed: {prepared.result.orientation_changed}",
        f"Orientation review: {prepared.result.manual_review_required}",
        "",
        "Native labels",
    ]
    lines.extend(
        f"{label:>2}  {schema.get(label, 'UNKNOWN'):<8}  {int(np.count_nonzero(result.vertebral_body_labels == label)):,} vox"
        for label in present
    )
    lines.extend(("", "QC flags"))
    lines.extend(
        f"[{flag.stage}] {flag.code}: {flag.reason}" for flag in result.qc_flags
    )
    if not result.qc_flags:
        lines.append("none")
    y = 92
    for line in lines:
        if y > 955:
            break
        wrapped = [line[index:index + 66] for index in range(0, len(line), 66)] or [""]
        for part in wrapped:
            cv2.putText(
                canvas,
                part,
                (1230, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
            y += 19

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial.png")
    if not cv2.imwrite(str(temporary), canvas):
        raise RuntimeError(f"Failed to write vertebral review image: {temporary}.")
    os.replace(temporary, output)
    return output
