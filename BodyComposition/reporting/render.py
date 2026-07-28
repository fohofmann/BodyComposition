"""Headless one-page PDF renderers for both fixed case-report layouts."""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from BodyComposition import __version__
from BodyComposition.measurement.contracts import (
    HU_DISTRIBUTION_BIN_WIDTH_HU,
    HU_DISTRIBUTION_MAX_HU,
    HU_DISTRIBUTION_MIN_HU,
    HU_DISTRIBUTION_TISSUES,
)
from BodyComposition.reporting.contracts import (
    MEASUREMENT_DEFINITIONS,
    CaseReportInput,
    ReportingSettings,
    ReviewEntry,
)
from BodyComposition.reporting.metrics import public_text
from BodyComposition.reporting.projection import (
    AxialSegmentationView,
    SagittalProjection,
    display_vertebral_level,
    tissue_color,
)

PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)
FONT_REGULAR = "BCVera"
FONT_BOLD = "BCVera-Bold"
MIN_FONT_SIZE = 8.0
LAYOUT_REVISION = "scientific_onepager_v24"
_FONT_LOCK = threading.Lock()
PAGE_BACKGROUND = colors.white
CARD_BACKGROUND = colors.white
INK = colors.HexColor("#1D3343")
MUTED = colors.HexColor("#5E6E79")
BORDER = colors.HexColor("#D4DDE3")
CARD_GAP = 8.0
CARD_PADDING = 10.0
CARD_TITLE_OFFSET = 18.0
CARD_SUBTITLE_OFFSET = 31.0
CARD_BORDER_WIDTH = 0.6
CARD_CORNER_RADIUS = 0.0
MIN_TABLE_ROW_HEIGHT = 9.5
PLOT_LEGEND_STYLE = "tissue_swatch_regular_8pt_v1"
PLOT_AXIS_STYLE = "ink_regular_ticks_bold_title_8pt_v1"
PLOT_SWATCH_SIZE = 8.0
PLOT_SWATCH_TEXT_GAP = 3.0
PLOT_LEGEND_ITEM_GAP = 8.0
VERTEBRAL_LABEL_STYLE = "centered_translucent_badge_v1"
VERTEBRAL_LABEL_BACKGROUND_ALPHA = 0.68
VERTEBRAL_LABEL_MIN_GAP_PT = 10.0
ROUTINE_BOUNDARY_NOTE_CODES = frozenset(
    {
        "vertebra_touches_cranial_fov",
        "vertebra_touches_caudal_fov",
    }
)
REVIEW_NOTE_LABELS = {
    "disconnected_vertebral_body": "Disconnected vertebral body segmentation.",
    "vertebral_body_extent_invalid": "Incomplete or truncated vertebral body extent.",
    "slice_measurement_invalid": "One or more slice measurements are invalid.",
    "body_surface_touches_fov": "Body surface reaches the lateral image boundary.",
    "trunk_contour_touches_fov": "Trunk contour reaches the image boundary.",
    "pelvic_maximum_unavailable": "Pelvic max could not be measured.",
    "minimum_waist_unavailable": "Waist min could not be measured.",
    "midwaist_measurement_unavailable": "Mid-waist could not be measured.",
    "severe_misorientation_repaired": (
        "Input CT was automatically reoriented after a substantial orientation mismatch."
    ),
    "orientation_mismatch_uncertain": (
        "Scan orientation remains uncertain; processing continued without orientation repair."
    ),
}


@dataclass(frozen=True)
class ImageBox:
    left: float
    bottom: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def top(self) -> float:
        return self.bottom + self.height


@dataclass(frozen=True)
class _NotesSummary:
    items: tuple[str, ...]
    review_count: int
    displayed_review_count: int
    suppressed_review_codes: tuple[str, ...]


def _font_paths() -> tuple[Path, Path]:
    root = resources.files("reportlab") / "fonts"
    return Path(str(root / "Vera.ttf")), Path(str(root / "VeraBd.ttf"))


def font_manifest() -> dict[str, Any]:
    regular, bold = _font_paths()
    return {
        "family": "Bitstream Vera",
        "license": "Bitstream Vera Fonts Copyright",
        "files": [
            {"name": regular.name, "sha256": hashlib.sha256(regular.read_bytes()).hexdigest()},
            {"name": bold.name, "sha256": hashlib.sha256(bold.read_bytes()).hexdigest()},
        ],
    }


def _register_fonts() -> None:
    with _FONT_LOCK:
        if FONT_REGULAR not in pdfmetrics.getRegisteredFontNames():
            regular, bold = _font_paths()
            pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(regular)))
            pdfmetrics.registerFont(TTFont(FONT_BOLD, str(bold)))


def _new_canvas(destination: Path, *, case_id: str) -> canvas.Canvas:
    _register_fonts()
    pdf = canvas.Canvas(
        str(destination),
        pagesize=(PAGE_WIDTH, PAGE_HEIGHT),
        pageCompression=1,
        invariant=1,
    )
    pdf.setTitle(f"BodyComposition case report {case_id}")
    pdf.setAuthor("BodyComposition")
    pdf.setSubject("Automated research and QC report")
    pdf.setCreator("BodyComposition reporting")
    pdf.setProducer("BodyComposition reporting")
    return pdf


def _text(
    pdf: canvas.Canvas, x: float, y: float, value: Any, *, size: float = 8, bold: bool = False
) -> None:
    if size < MIN_FONT_SIZE:
        raise ValueError("Released PDF layouts do not permit text below 8 pt.")
    pdf.setFont(FONT_BOLD if bold else FONT_REGULAR, size)
    pdf.drawString(x, y, public_text(value, maximum=180))


def _centered(
    pdf: canvas.Canvas, x: float, y: float, value: Any, *, size: float = 8, bold: bool = False
) -> None:
    if size < MIN_FONT_SIZE:
        raise ValueError("Released PDF layouts do not permit text below 8 pt.")
    text = public_text(value, maximum=100)
    font = FONT_BOLD if bold else FONT_REGULAR
    pdf.setFont(font, size)
    pdf.drawCentredString(x, y, text)


def _truncate_to_width(
    text: str, width: float, *, font: str = FONT_REGULAR, size: float = 8
) -> str:
    text = public_text(text, maximum=180)
    if pdfmetrics.stringWidth(text, font, size) <= width:
        return text
    suffix = "..."
    while text and pdfmetrics.stringWidth(text + suffix, font, size) > width:
        text = text[:-1]
    return text + suffix


def _fit_font_size(text: str, width: float, *, maximum: float, font: str) -> float:
    size = float(maximum)
    while size > MIN_FONT_SIZE and pdfmetrics.stringWidth(text, font, size) > width:
        size -= 0.5
    if pdfmetrics.stringWidth(text, font, size) > width:
        raise ValueError("Text cannot fit the released layout at 8 pt.")
    return size


def _draw_metadata_icon(
    pdf: canvas.Canvas,
    kind: str,
    x: float,
    y: float,
    *,
    size: float = 8.0,
) -> None:
    """Draw one small, deterministic line icon without relying on a symbol font."""
    scale = size / 8.0

    def point_x(value: float) -> float:
        return x + value * scale

    def point_y(value: float) -> float:
        return y + value * scale

    pdf.saveState()
    pdf.setStrokeColor(MUTED)
    pdf.setFillColor(MUTED)
    pdf.setLineWidth(0.7)
    pdf.setLineCap(1)
    pdf.setLineJoin(1)

    if kind == "analysis":
        pdf.roundRect(
            point_x(0.5),
            point_y(0.7),
            7.0 * scale,
            6.4 * scale,
            0.7 * scale,
            stroke=1,
            fill=0,
        )
        pdf.line(point_x(0.5), point_y(5.2), point_x(7.5), point_y(5.2))
        pdf.line(point_x(2.0), point_y(6.3), point_x(2.0), point_y(7.7))
        pdf.line(point_x(6.0), point_y(6.3), point_x(6.0), point_y(7.7))
        pdf.circle(point_x(2.2), point_y(3.2), 0.45 * scale, stroke=0, fill=1)
        pdf.circle(point_x(4.0), point_y(3.2), 0.45 * scale, stroke=0, fill=1)
        pdf.circle(point_x(5.8), point_y(3.2), 0.45 * scale, stroke=0, fill=1)
    elif kind == "input":
        page = pdf.beginPath()
        page.moveTo(point_x(1.4), point_y(0.5))
        page.lineTo(point_x(7.1), point_y(0.5))
        page.lineTo(point_x(7.1), point_y(5.6))
        page.lineTo(point_x(5.5), point_y(7.4))
        page.lineTo(point_x(1.4), point_y(7.4))
        page.close()
        pdf.drawPath(page, stroke=1, fill=0)
        pdf.line(point_x(5.5), point_y(7.4), point_x(5.5), point_y(5.6))
        pdf.line(point_x(5.5), point_y(5.6), point_x(7.1), point_y(5.6))
        pdf.line(point_x(0.2), point_y(3.2), point_x(4.3), point_y(3.2))
        pdf.line(point_x(3.1), point_y(4.3), point_x(4.3), point_y(3.2))
        pdf.line(point_x(3.1), point_y(2.1), point_x(4.3), point_y(3.2))
    elif kind == "pipeline":
        pdf.line(point_x(1.6), point_y(4.0), point_x(3.1), point_y(4.0))
        pdf.line(point_x(4.9), point_y(4.0), point_x(6.4), point_y(4.0))
        for center_x in (0.9, 4.0, 7.1):
            pdf.circle(
                point_x(center_x),
                point_y(4.0),
                0.8 * scale,
                stroke=1,
                fill=0,
            )
    elif kind == "runtime":
        pdf.roundRect(
            point_x(1.5),
            point_y(1.5),
            5.0 * scale,
            5.0 * scale,
            0.6 * scale,
            stroke=1,
            fill=0,
        )
        pdf.rect(
            point_x(2.8),
            point_y(2.8),
            2.4 * scale,
            2.4 * scale,
            stroke=1,
            fill=0,
        )
        for offset in (2.6, 5.4):
            pdf.line(point_x(0.5), point_y(offset), point_x(1.5), point_y(offset))
            pdf.line(point_x(6.5), point_y(offset), point_x(7.5), point_y(offset))
            pdf.line(point_x(offset), point_y(0.5), point_x(offset), point_y(1.5))
            pdf.line(point_x(offset), point_y(6.5), point_x(offset), point_y(7.5))
    elif kind == "scanner":
        pdf.roundRect(
            point_x(0.5),
            point_y(0.8),
            7.0 * scale,
            6.4 * scale,
            0.8 * scale,
            stroke=1,
            fill=0,
        )
        pdf.circle(point_x(4.0), point_y(4.0), 2.0 * scale, stroke=1, fill=0)
        pdf.line(point_x(2.4), point_y(1.0), point_x(5.6), point_y(1.0))
        pdf.line(point_x(4.0), point_y(1.0), point_x(4.0), point_y(2.0))
    elif kind == "slice":
        for index in range(3):
            inset = index * 0.45
            pdf.roundRect(
                point_x(0.8 + inset),
                point_y(1.0 + index * 2.1),
                (6.4 - 2 * inset) * scale,
                1.3 * scale,
                0.35 * scale,
                stroke=1,
                fill=0,
            )
    elif kind == "models":
        pdf.line(point_x(4.0), point_y(4.0), point_x(1.1), point_y(4.0))
        pdf.line(point_x(4.0), point_y(4.0), point_x(6.8), point_y(6.5))
        pdf.line(point_x(4.0), point_y(4.0), point_x(6.8), point_y(1.5))
        for center_x, center_y in ((1.0, 4.0), (4.0, 4.0), (7.0, 6.7), (7.0, 1.3)):
            pdf.circle(
                point_x(center_x),
                point_y(center_y),
                0.75 * scale,
                stroke=1,
                fill=0,
            )
    else:
        pdf.restoreState()
        raise ValueError(f"Unsupported technical-metadata icon: {kind}")
    pdf.restoreState()


def _header(
    pdf: canvas.Canvas,
    case: CaseReportInput,
    _report_id: str,
) -> tuple[float, dict[str, Any]]:
    box = ImageBox(24, PAGE_HEIGHT - 38, PAGE_WIDTH - 48, 28)
    _draw_card(pdf, box, border_visible=False)

    right_text = "Page 1 of 1"
    right_width = pdfmetrics.stringWidth(right_text, FONT_REGULAR, 8)
    patient = case.patient_metadata
    present_patient_fields = [
        key
        for key in ("patient_name", "date_of_birth", "scan_date", "sex")
        if patient.get(key)
    ]
    maximum_left_width = box.width - 2 * CARD_PADDING - right_width - 18
    demographic_parts: list[str] = []
    for label, key in (
        ("DOB", "date_of_birth"),
        ("Scan", "scan_date"),
        ("Sex", "sex"),
    ):
        if patient.get(key):
            demographic_parts.append(f"{label} {patient[key]}")

    case_text = f"Case {public_text(case.case_id, maximum=64)}"
    if patient.get("patient_name"):
        reserved_without_case = " | ".join(demographic_parts)
        reserved_width = pdfmetrics.stringWidth(
            f"{reserved_without_case} | " if reserved_without_case else "",
            FONT_BOLD,
            8,
        )
        case_budget = max(55.0, maximum_left_width - reserved_width - 95.0)
        case_text = _truncate_to_width(
            case_text,
            case_budget,
            font=FONT_BOLD,
            size=8,
        )
        suffix = " | ".join((*demographic_parts, case_text))
        suffix_width = pdfmetrics.stringWidth(f" | {suffix}", FONT_BOLD, 8)
        patient_text = _truncate_to_width(
            f"Patient {patient['patient_name']}",
            max(35.0, maximum_left_width - suffix_width),
            font=FONT_BOLD,
            size=8,
        )
        left_text = f"{patient_text} | {suffix}"
    else:
        left_text = _truncate_to_width(
            case_text,
            maximum_left_width,
            font=FONT_BOLD,
            size=8,
        )
    pdf.setFillColor(INK)
    _text(
        pdf,
        box.left + CARD_PADDING,
        box.top - CARD_TITLE_OFFSET,
        _truncate_to_width(
            left_text,
            maximum_left_width,
            font=FONT_BOLD,
            size=8,
        ),
        size=8,
        bold=True,
    )
    pdf.setFillColor(MUTED)
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawRightString(
        box.right - CARD_PADDING,
        box.top - CARD_TITLE_OFFSET,
        right_text,
    )
    return box.bottom - 8, {
        "boxed": False,
        "outer_border_visible": False,
        "identity_only": not bool(present_patient_fields),
        "subheader_visible": False,
        "patient_fields": present_patient_fields,
        "case_id_visible": True,
        "analysis_id_visible": False,
        "report_id_visible": False,
        "page_number_scope": "individual_document",
        "box_height_pt": box.height,
        "box_top_pt": box.top,
        "content_padding_pt": CARD_PADDING,
        "technical_metadata_location": "right_column",
        "technical_metadata_rows": 0,
        "technical_metadata_icons": [],
        "technical_metadata_truncated": False,
    }


def page_number_overlay(page_number: int, page_count: int) -> bytes:
    """Return a deterministic transparent overlay for combined-document numbering."""

    if not 1 <= page_number <= page_count:
        raise ValueError("Page number must be within the combined document.")
    _register_fonts()
    stream = BytesIO()
    pdf = canvas.Canvas(
        stream,
        pagesize=(PAGE_WIDTH, PAGE_HEIGHT),
        pageCompression=1,
        invariant=1,
    )
    box = ImageBox(24, PAGE_HEIGHT - 38, PAGE_WIDTH - 48, 28)
    clear_left = box.right - 96
    pdf.setFillColor(PAGE_BACKGROUND)
    pdf.rect(
        clear_left,
        box.bottom,
        box.right - clear_left,
        box.height,
        stroke=0,
        fill=1,
    )
    pdf.setFillColor(MUTED)
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawRightString(
        box.right - CARD_PADDING,
        box.top - CARD_TITLE_OFFSET,
        f"Page {page_number} of {page_count}",
    )
    pdf.showPage()
    pdf.save()
    return stream.getvalue()


def _draw_card(
    pdf: canvas.Canvas,
    box: ImageBox,
    *,
    border_visible: bool = True,
) -> None:
    pdf.setFillColor(CARD_BACKGROUND)
    pdf.setStrokeColor(BORDER)
    pdf.setLineWidth(CARD_BORDER_WIDTH)
    pdf.rect(
        box.left,
        box.bottom,
        box.width,
        box.height,
        stroke=int(border_visible),
        fill=1,
    )


def _tissue_legend_item_width(label: str) -> float:
    return (
        PLOT_SWATCH_SIZE
        + PLOT_SWATCH_TEXT_GAP
        + pdfmetrics.stringWidth(label, FONT_REGULAR, 8)
        + PLOT_LEGEND_ITEM_GAP
    )


def _draw_tissue_legend_item(
    pdf: canvas.Canvas,
    *,
    x: float,
    y: float,
    tissue_key: str,
    label: str,
) -> float:
    red, green, blue = tissue_color(tissue_key)
    pdf.setFillColor(colors.Color(red / 255, green / 255, blue / 255))
    pdf.rect(x, y, PLOT_SWATCH_SIZE, PLOT_SWATCH_SIZE, stroke=0, fill=1)
    pdf.setFillColor(INK)
    _text(
        pdf,
        x + PLOT_SWATCH_SIZE + PLOT_SWATCH_TEXT_GAP,
        y + 1.0,
        label,
        size=8,
    )
    return _tissue_legend_item_width(label)


def _draw_plot_tick(
    pdf: canvas.Canvas,
    *,
    x: float,
    y: float,
    label: str,
    alignment: str = "center",
) -> None:
    pdf.setFillColor(INK)
    pdf.setFont(FONT_REGULAR, 8)
    if alignment == "left":
        pdf.drawString(x, y, label)
    elif alignment == "right":
        pdf.drawRightString(x, y, label)
    elif alignment == "center":
        pdf.drawCentredString(x, y, label)
    else:
        raise ValueError(f"Unsupported plot tick alignment: {alignment}.")


def _draw_plot_axis_title(
    pdf: canvas.Canvas,
    *,
    x: float,
    y: float,
    label: str,
) -> None:
    pdf.setFillColor(INK)
    _centered(pdf, x, y, label, size=8, bold=True)


def _draw_main_grid(
    pdf: canvas.Canvas,
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    dividers: tuple[float, ...],
) -> dict[str, Any]:
    pdf.setStrokeColor(BORDER)
    pdf.setLineWidth(CARD_BORDER_WIDTH)
    pdf.line(left, top, right, top)
    pdf.line(left, bottom, right, bottom)
    for divider in dividers:
        pdf.line(divider, bottom, divider, top)
    return {
        "style": "open_editorial_grid",
        "divider_count": len(dividers),
        "rule_width_pt": CARD_BORDER_WIDTH,
        "outer_vertical_rules": False,
    }


def _wrap_text(value: str, width: float, *, font: str = FONT_REGULAR, size: float = 8) -> list[str]:
    words = public_text(value, maximum=500).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and pdfmetrics.stringWidth(candidate, font, size) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or ["Not recorded"]


def _image_box(projection: SagittalProjection, panel: ImageBox) -> ImageBox:
    height_px, width_px = projection.rgb.shape[:2]
    scale = min(panel.width / width_px, panel.height / height_px)
    width = width_px * scale
    height = height_px * scale
    return ImageBox(
        left=panel.left + (panel.width - width) / 2,
        bottom=panel.top - height,
        width=width,
        height=height,
    )


def _separate_label_y_positions(
    desired: list[float],
    *,
    lower: float,
    upper: float,
    preferred_gap: float = 11.0,
) -> list[float]:
    """Separate label centers while preserving their physical order."""

    if not desired:
        return []
    if len(desired) == 1:
        return [float(np.clip(desired[0], lower, upper))]
    order = np.argsort(np.asarray(desired, dtype=float), kind="stable")
    ordered = np.clip(np.asarray(desired, dtype=float)[order], lower, upper)
    gap = min(float(preferred_gap), float(upper - lower) / (len(ordered) - 1))
    placed = ordered.copy()
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap)
    if placed[-1] > upper:
        placed[-1] = upper
        for index in range(len(placed) - 2, -1, -1):
            placed[index] = min(placed[index], placed[index + 1] - gap)
    if placed[0] < lower:
        placed += lower - placed[0]
    result = np.empty_like(placed)
    result[order] = placed
    return [float(value) for value in result]


def _anthropometry_spine_markers(
    summary: pd.Series,
    projection: SagittalProjection,
) -> tuple[dict[str, Any], ...]:
    markers: list[dict[str, Any]] = []
    for label, prefix, line_style, label_side in (
        ("Waist min", "ct_min_trunk_circumference_t10_l5", "solid", "left"),
        ("Pelvic max", "ct_max_pelvic_circumference", "dashed", "right"),
    ):
        position_raw = pd.to_numeric(
            summary.get(f"{prefix}_position_superior_mm"),
            errors="coerce",
        )
        value_raw = pd.to_numeric(
            summary.get(f"{prefix}_cm"),
            errors="coerce",
        )
        position = float(position_raw) if not pd.isna(position_raw) else float("nan")
        value = float(value_raw) if not pd.isna(value_raw) else float("nan")
        measured = bool(
            summary.get(f"{prefix}_valid", False)
            and np.isfinite(position)
            and np.isfinite(value)
        )
        within_projection = bool(
            measured
            and projection.inferior_mm <= float(position) <= projection.superior_mm
        )
        markers.append(
            {
                "label": label,
                "position_superior_mm": position if np.isfinite(position) else None,
                "value_cm": value if np.isfinite(value) else None,
                "measured": measured,
                "displayed": within_projection,
                "line_style": line_style,
                "label_side": label_side,
            }
        )
    return tuple(markers)


def _draw_anthropometry_spine_markers(
    pdf: canvas.Canvas,
    projection: SagittalProjection,
    box: ImageBox,
    markers: tuple[Mapping[str, Any], ...],
) -> None:
    for marker in markers:
        if not marker["displayed"]:
            continue
        position = float(marker["position_superior_mm"])
        y = projection.y_for_superior(position, box.bottom, box.top)
        dash = [] if marker["line_style"] == "solid" else [3.0, 2.0]
        pdf.saveState()
        pdf.setDash(dash)
        pdf.setStrokeColor(colors.white)
        pdf.setLineWidth(2.4)
        pdf.line(box.left, y, box.right, y)
        pdf.setStrokeColor(colors.HexColor("#203746"))
        pdf.setLineWidth(0.9)
        pdf.line(box.left, y, box.right, y)
        pdf.restoreState()

        label = str(marker["label"])
        label_width = pdfmetrics.stringWidth(label, FONT_BOLD, 8) + 8
        label_x = (
            box.left + 3
            if marker["label_side"] == "left"
            else box.right - label_width - 3
        )
        preferred_baseline = y - 12 if y > box.top - 22 else y + 3
        label_baseline = float(
            np.clip(preferred_baseline, box.bottom + 4, box.top - 9)
        )
        pdf.setFillColor(colors.HexColor("#203746"))
        pdf.roundRect(
            label_x,
            label_baseline - 4,
            label_width,
            11,
            2,
            stroke=0,
            fill=1,
        )
        pdf.setFillColor(colors.white)
        _text(pdf, label_x + 4, label_baseline - 1, label, size=8, bold=True)


def _draw_projection(
    pdf: canvas.Canvas,
    projection: SagittalProjection,
    panel: ImageBox,
    *,
    ct_window: tuple[float, float],
    rows: list[dict[str, Any]],
    show_title: bool = True,
    anthropometry_markers: tuple[Mapping[str, Any], ...] = (),
) -> ImageBox:
    box = _image_box(projection, panel)
    image = ImageReader(Image.fromarray(projection.rgb, mode="RGB"))
    pdf.drawImage(
        image,
        box.left,
        box.bottom,
        width=box.width,
        height=box.height,
        preserveAspectRatio=True,
        mask="auto",
    )
    pdf.setStrokeColor(colors.HexColor("#3D4B55"))
    pdf.rect(box.left, box.bottom, box.width, box.height, stroke=1, fill=0)
    if show_title:
        pdf.setFillColor(colors.HexColor("#263746"))
        _text(pdf, panel.left, panel.top + 7, "Sagittal thick-slab projection", size=8, bold=True)
    pdf.setFillColor(colors.white)
    _text(pdf, box.left + 4, box.top - 12, "S", size=8, bold=True)
    _text(pdf, box.left + 4, box.bottom + 4, "I", size=8, bold=True)
    _text(pdf, box.left + 18, box.bottom + 4, "A", size=8, bold=True)
    pdf.setFont(FONT_BOLD, 8)
    pdf.drawRightString(box.right - 4, box.bottom + 4, "P")
    if show_title:
        pdf.setFillColor(colors.HexColor("#F7F8F9"))
        pdf.roundRect(box.left + 5, box.top - 28, 125, 13, 2, stroke=0, fill=1)
        pdf.setFillColor(colors.HexColor("#263746"))
        _text(
            pdf,
            box.left + 9,
            box.top - 24,
            f"window {int(ct_window[0])} to {int(ct_window[1])} HU",
            size=8,
        )
    _draw_anthropometry_spine_markers(
        pdf,
        projection,
        box,
        anthropometry_markers,
    )
    scale_y = box.height / (projection.superior_mm - projection.inferior_mm)
    scale_length_mm = 50.0 if scale_y * 50.0 <= box.height / 3 else 20.0
    scale_length = scale_y * scale_length_mm
    # Keep the independent physical scale at the left edge, away from the
    # centered vertebral labels.
    x_scale = box.left + 13
    y_scale = box.bottom + 18
    pdf.setStrokeColor(colors.white)
    pdf.setLineWidth(2)
    pdf.line(x_scale, y_scale, x_scale, y_scale + scale_length)
    pdf.line(x_scale - 3, y_scale, x_scale + 3, y_scale)
    pdf.line(x_scale - 3, y_scale + scale_length, x_scale + 3, y_scale + scale_length)
    pdf.setFillColor(colors.white)
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawString(x_scale + 5, y_scale + scale_length / 2 - 3, f"{int(scale_length_mm)} mm")

    centroids = sorted(
        (
            row
            for row in rows
            if row["native_label"] in projection.native_labels
            and all(value is not None for value in row["centroid_lps_xyz"])
        ),
        key=lambda item: item["centroid_lps_xyz"][2],
        reverse=True,
    )
    desired_y = [
        projection.y_for_superior(row["centroid_lps_xyz"][2], box.bottom, box.top)
        for row in centroids
    ]
    label_y = _separate_label_y_positions(
        desired_y,
        lower=box.bottom + 10,
        upper=box.top - 10,
        preferred_gap=VERTEBRAL_LABEL_MIN_GAP_PT,
    )
    for centroid, y_label in zip(centroids, label_y, strict=True):
        point = centroid["centroid_lps_xyz"]
        x_true = (
            box.left
            + (
                (point[1] - projection.anterior_mm)
                / (projection.posterior_mm - projection.anterior_mm)
            )
            * box.width
        )
        anatomical = centroid["vertebral_level"]
        display_label = display_vertebral_level(anatomical)
        label_width = max(
            14.0,
            pdfmetrics.stringWidth(display_label, FONT_BOLD, 8) + 6.0,
        )
        label_x = float(
            np.clip(
                x_true - label_width / 2,
                box.left + 2,
                box.right - label_width - 2,
            )
        )
        pdf.saveState()
        pdf.setFillColor(
            colors.Color(
                29 / 255,
                51 / 255,
                67 / 255,
                alpha=VERTEBRAL_LABEL_BACKGROUND_ALPHA,
            )
        )
        pdf.roundRect(label_x, y_label - 5, label_width, 10, 2, stroke=0, fill=1)
        pdf.setFillColor(colors.white)
        _text(pdf, label_x + 3, y_label - 2, display_label, size=8, bold=True)
        pdf.restoreState()
    return box


def _format_metric(metric: dict[str, Any], settings: ReportingSettings) -> str:
    if not metric["valid"] or metric["value"] is None:
        return settings.missing_value_symbol + "*"
    marker = "!" if bool(metric.get("value_is_fov_cropped", False)) else ""
    return f"{float(metric['value']):.{settings.numeric_precision}f}{marker}"


def _draw_table(
    pdf: canvas.Canvas,
    rows: list[dict[str, Any]],
    settings: ReportingSettings,
    projection: SagittalProjection,
    panel: ImageBox,
) -> dict[str, Any]:
    displayed_rows = [row for row in rows if int(row["native_label"]) in projection.native_labels]
    if not displayed_rows:
        raise ValueError("The vertebral table has no detected native levels to display.")
    incomplete_rows = [row for row in displayed_rows if not bool(row["territory_complete"])]
    pdf.setFillColor(colors.HexColor("#263746"))
    _text(
        pdf,
        panel.left,
        panel.top - 8,
        "Vertebral summary",
        size=8,
        bold=True,
    )
    pdf.setFillColor(MUTED)
    _text(
        pdf,
        panel.left,
        panel.top - 21,
        "Whole-level mean",
        size=8,
    )
    level_width = 39.0
    metric_width = (panel.width - level_width) / len(settings.measurement_columns)
    grid_top = panel.top - 34
    header_height = 25.0
    header_bottom = grid_top - header_height
    available_body_height = header_bottom - panel.bottom
    row_height = min(18.0, available_body_height / len(displayed_rows))
    row_capacity = int(available_body_height // MIN_TABLE_ROW_HEIGHT)
    if row_height < MIN_TABLE_ROW_HEIGHT:
        raise ValueError("Detected vertebral rows cannot fit the released uniform table at 8 pt.")
    table_bottom = header_bottom - row_height * len(displayed_rows)
    header_y = grid_top - 10
    pdf.setFillColor(colors.HexColor("#F3F7F9"))
    pdf.rect(panel.left, header_bottom, panel.width, header_height, stroke=0, fill=1)
    for row_index in range(len(displayed_rows)):
        if row_index % 2:
            row_top = header_bottom - row_index * row_height
            pdf.setFillColor(colors.HexColor("#FAFCFD"))
            pdf.rect(
                panel.left,
                row_top - row_height,
                panel.width,
                row_height,
                stroke=0,
                fill=1,
            )

    pdf.setFillColor(colors.HexColor("#263746"))
    _centered(pdf, panel.left + level_width / 2, header_y - 4, "Level", size=8, bold=True)
    for index, column in enumerate(settings.measurement_columns):
        definition = MEASUREMENT_DEFINITIONS[column]
        x = panel.left + level_width + (index + 0.5) * metric_width
        _centered(pdf, x, header_y, definition["short_label"], size=8, bold=True)
        _centered(pdf, x, header_y - 10.5, f"({definition['unit']})", size=8)

    pdf.setStrokeColor(BORDER)
    pdf.setLineWidth(0.6)
    pdf.rect(panel.left, table_bottom, panel.width, grid_top - table_bottom, stroke=1, fill=0)
    column_edges = [panel.left, panel.left + level_width]
    column_edges.extend(
        panel.left + level_width + index * metric_width
        for index in range(1, len(settings.measurement_columns) + 1)
    )
    for x in column_edges[1:-1]:
        pdf.line(x, table_bottom, x, grid_top)
    pdf.line(panel.left, header_bottom, panel.right, header_bottom)

    for row_index, row in enumerate(displayed_rows):
        row_top = header_bottom - row_index * row_height
        row_bottom = row_top - row_height
        pdf.setStrokeColor(colors.HexColor("#E7ECEF"))
        pdf.line(panel.left, row_bottom, panel.right, row_bottom)
        baseline = row_bottom + row_height / 2 - 2.8
        marker = "".join(
            (
                "V" if row["anatomical_variant"] else "",
                "!" if not row["territory_complete"] else "",
            )
        )
        _text(
            pdf,
            panel.left + 2,
            baseline,
            f"{display_vertebral_level(row['vertebral_level'])} {marker}".rstrip(),
            size=8,
            bold=True,
        )
        for index, column in enumerate(settings.measurement_columns):
            value = _format_metric(row["metrics"][column], settings)
            _centered(
                pdf,
                panel.left + level_width + (index + 0.5) * metric_width,
                baseline,
                value,
                size=8,
            )
    pdf.setFillColor(MUTED)
    _text(
        pdf,
        panel.left,
        panel.bottom - 13,
        "* unavailable | V variant | ! incomplete/FOV-limited",
        size=8,
    )
    return {
        "row_layout": "uniform_categorical_grid",
        "displayed_levels": [
            display_vertebral_level(str(row["vertebral_level"])) for row in displayed_rows
        ],
        "canonical_levels": [str(row["vertebral_level"]) for row in displayed_rows],
        "measurement_columns": list(settings.measurement_columns),
        "display_labels": [
            MEASUREMENT_DEFINITIONS[column]["short_label"]
            for column in settings.measurement_columns
        ],
        "row_count": len(displayed_rows),
        "row_capacity": row_capacity,
        "row_height_pt": float(row_height),
        "minimum_row_height_pt": MIN_TABLE_ROW_HEIGHT,
        "metric_column_width_pt": float(metric_width),
        "incomplete_level_display_policy": "valid_observed_mean_with_level_marker",
        "incomplete_levels": [str(row["vertebral_level"]) for row in incomplete_rows],
        "fov_limited_metric_count": sum(
            bool(metric.get("value_is_fov_cropped", False))
            for row in displayed_rows
            for metric in row["metrics"].values()
        ),
        "incomplete_observed_metric_count": sum(
            bool(metric["valid"]) and metric["value"] is not None
            for row in incomplete_rows
            for metric in row["metrics"].values()
        ),
        "incomplete_missing_metric_count": sum(
            not bool(metric["valid"]) or metric["value"] is None
            for row in incomplete_rows
            for metric in row["metrics"].values()
        ),
    }


def _raster_box(array: np.ndarray, panel: ImageBox) -> ImageBox:
    height_px, width_px = array.shape[:2]
    scale = min(panel.width / width_px, panel.height / height_px)
    width = width_px * scale
    height = height_px * scale
    return ImageBox(
        left=panel.left + (panel.width - width) / 2,
        bottom=panel.top - height,
        width=width,
        height=height,
    )


def _draw_axial_marker(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    value: str,
    *,
    centered: bool = True,
) -> None:
    pdf.setFillColor(colors.HexColor("#24333D"))
    pdf.roundRect(x - 7, y - 6, 14, 13, 2, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    if centered:
        _centered(pdf, x, y - 2, value, size=8, bold=True)
    else:
        _text(pdf, x - 4, y - 2, value, size=8, bold=True)


def _summary_value(
    summary: pd.Series,
    name: str,
    settings: ReportingSettings,
    *,
    unit: str,
) -> str:
    value = summary.get(name)
    base_name = name[:-3] if name.endswith("_cm") else name
    valid = bool(summary.get(f"{base_name}_valid", False))
    if not valid or pd.isna(value):
        return settings.missing_value_symbol + "*"
    formatted = f"{float(value):.{settings.numeric_precision}f}"
    eligible = summary.get(f"{base_name}_eligible", True)
    marker = "*" if pd.notna(eligible) and not bool(eligible) else ""
    return f"{formatted} {unit}{marker}".rstrip()


def _anthropometry_note(summary: pd.Series) -> str | None:
    reason_text = {
        "missing_anchor": "the anatomical range is incomplete",
        "sequence_gap": "the vertebral sequence is incomplete",
        "partial_fov": "the anatomical range is only partly acquired",
        "partial_contour_coverage": "contour coverage is incomplete",
        "boundary_extremum": "the extremum lies at the search boundary",
        "touching_image_boundary": "the contour reaches the image boundary",
        "invalid_measurement": "no valid closed contour is available",
        "outside_fov": "the anatomical range lies outside the scan",
    }
    statuses: list[str] = []
    for label, prefix in (
        ("Waist min", "ct_min_trunk_circumference_t10_l5"),
        ("Pelvic max", "ct_max_pelvic_circumference"),
    ):
        valid = bool(summary.get(f"{prefix}_valid", False))
        eligible = bool(summary.get(f"{prefix}_eligible", False))
        if valid and eligible:
            continue
        if bool(summary.get(f"{prefix}_value_is_fov_cropped", False)):
            statuses.append(
                f"{label} contour is cropped by the image boundary and may be underestimated"
            )
            continue
        reason = str(summary.get(f"{prefix}_reason") or "invalid_measurement")
        explanation = reason_text.get(reason, reason.replace("_", " "))
        if valid:
            statuses.append(f"{label} observed, but {explanation}")
        else:
            statuses.append(f"{label} unavailable because {explanation}")
    if not statuses:
        return None
    return "; ".join(statuses) + "."


def _draw_key_anthropometry(
    pdf: canvas.Canvas,
    case: CaseReportInput,
    settings: ReportingSettings,
    panel: ImageBox,
) -> None:
    summary = case.measurement_bundle.summaries.iloc[0]
    pdf.setFillColor(INK)
    _text(pdf, panel.left, panel.top, "Key anthropometry", size=8, bold=True)
    rows = (
        (
            "Waist min",
            _summary_value(
                summary,
                "ct_min_trunk_circumference_t10_l5_cm",
                settings,
                unit="cm",
            ),
        ),
        (
            "Pelvic max",
            _summary_value(
                summary,
                "ct_max_pelvic_circumference_cm",
                settings,
                unit="cm",
            ),
        ),
        (
            "Waist / pelvic",
            _summary_value(
                summary,
                "ct_min_waist_to_pelvic_ratio",
                settings,
                unit="",
            ),
        ),
    )
    y = panel.top - 16
    for label, value in rows:
        pdf.setFillColor(MUTED)
        _text(pdf, panel.left, y, _truncate_to_width(label, panel.width - 55), size=8)
        pdf.setFillColor(INK)
        pdf.setFont(FONT_BOLD, 8)
        pdf.drawRightString(panel.right, y, value)
        y -= 13
    pdf.setFillColor(MUTED)
    _text(pdf, panel.left, y, "* unavailable or not fully eligible", size=8)


def _right_column_panels(
    *,
    left: float,
    width: float,
    content_bottom: float,
    content_top: float,
) -> tuple[ImageBox, ImageBox, ImageBox]:
    anthropometry_height = 82.0
    technical_height = 155.0
    available_height = content_top - content_bottom
    visual_height = available_height - anthropometry_height - technical_height
    if visual_height < 120:
        raise ValueError("Right-column report sections cannot fit the released layout.")
    axial_card = ImageBox(
        left,
        content_top - visual_height,
        width,
        visual_height,
    )
    anthropometry_card = ImageBox(
        left,
        axial_card.bottom - anthropometry_height,
        width,
        anthropometry_height,
    )
    technical_card = ImageBox(
        left,
        content_bottom,
        width,
        technical_height,
    )
    return axial_card, anthropometry_card, technical_card


def _draw_section_rule(pdf: canvas.Canvas, panel: ImageBox) -> None:
    pdf.setStrokeColor(BORDER)
    pdf.setLineWidth(CARD_BORDER_WIDTH)
    pdf.line(panel.left, panel.top, panel.right, panel.top)


def _draw_anthropometry_card(
    pdf: canvas.Canvas,
    case: CaseReportInput,
    settings: ReportingSettings,
    panel: ImageBox,
) -> dict[str, Any]:
    summary = case.measurement_bundle.summaries.iloc[0]
    _draw_section_rule(pdf, panel)
    _draw_key_anthropometry(
        pdf,
        case,
        settings,
        ImageBox(
            panel.left + CARD_PADDING,
            panel.bottom + CARD_PADDING,
            panel.width - 2 * CARD_PADDING,
            panel.height - CARD_TITLE_OFFSET - CARD_PADDING,
        ),
    )
    return {
        "separate_card": False,
        "section_style": "shared_grid_rule",
        "top_rule_width_pt": CARD_BORDER_WIDTH,
        "box_width_pt": panel.width,
        "box_height_pt": panel.height,
        "box_top_pt": panel.top,
        "box_bottom_pt": panel.bottom,
        "content_padding_pt": CARD_PADDING,
        "title_baseline_from_top_pt": CARD_TITLE_OFFSET,
        "labels": ["Waist min", "Pelvic max", "Waist / pelvic"],
        "asterisk_explanation": "* unavailable or not fully eligible",
        "body_surface_touches_fov": bool(summary.get("body_surface_touches_fov", False)),
        "body_surface_touches_fov_slice_count": int(
            summary.get("body_surface_touches_fov_slice_count", 0)
        ),
        "trunk_contour_touches_fov": bool(summary.get("trunk_contour_touches_fov", False)),
        "trunk_contour_touches_fov_slice_count": int(
            summary.get("trunk_contour_touches_fov_slice_count", 0)
        ),
        "pelvic_value_is_fov_cropped": bool(
            summary.get(
                "ct_max_pelvic_circumference_value_is_fov_cropped",
                False,
            )
        ),
        "ratio_value_is_fov_cropped": bool(
            summary.get(
                "ct_min_waist_to_pelvic_ratio_value_is_fov_cropped",
                False,
            )
        ),
    }


def _draw_technical_context(
    pdf: canvas.Canvas,
    case: CaseReportInput,
    panel: ImageBox,
) -> dict[str, Any]:
    _draw_section_rule(pdf, panel)
    pdf.setFillColor(INK)
    _text(
        pdf,
        panel.left + CARD_PADDING,
        panel.top - CARD_TITLE_OFFSET,
        "Technical context",
        size=8,
        bold=True,
    )

    items = _technical_metadata_items(case)
    font_size = 8.0
    line_height = 9.0
    row_gap = 1.0
    icon_size = 7.0
    text_x = panel.left + CARD_PADDING + icon_size + 4.0
    text_width = panel.right - CARD_PADDING - text_x
    y = panel.top - 34.0
    rendered_line_counts: list[int] = []
    explicit_line_counts: list[int] = []
    for kind, values in items:
        lines = [
            line
            for value in values
            for line in _wrap_text(
                public_text(value, maximum=180),
                text_width,
                size=font_size,
            )
        ]
        required_height = len(lines) * line_height
        if y - required_height + line_height < panel.bottom + 6.0:
            raise ValueError("Technical context cannot fit the released right column.")
        _draw_metadata_icon(
            pdf,
            kind,
            panel.left + CARD_PADDING,
            y - 1.0,
            size=icon_size,
        )
        for line in lines:
            pdf.setFillColor(MUTED)
            _text(pdf, text_x, y, line, size=font_size)
            y -= line_height
        y -= row_gap
        rendered_line_counts.append(len(lines))
        explicit_line_counts.append(len(values))
    return {
        "location": "right_column",
        "metadata_scope": "complete",
        "input_voxel_size_included": True,
        "separate_slice_row": True,
        "section_style": "shared_grid_rule",
        "top_rule_width_pt": CARD_BORDER_WIDTH,
        "box_left_pt": panel.left,
        "box_right_pt": panel.right,
        "box_width_pt": panel.width,
        "box_height_pt": panel.height,
        "box_top_pt": panel.top,
        "box_bottom_pt": panel.bottom,
        "content_padding_pt": CARD_PADDING,
        "title_baseline_from_top_pt": CARD_TITLE_OFFSET,
        "icons": [kind for kind, _values in items],
        "icon_style": "monochrome_vector",
        "font_size_pt": font_size,
        "rendered_line_counts": rendered_line_counts,
        "explicit_line_counts": explicit_line_counts,
        "separators_suppressed": ["/", "|"],
        "truncated": False,
    }


def _draw_axial_panel(
    pdf: canvas.Canvas,
    view: AxialSegmentationView,
    panel: ImageBox,
) -> dict[str, Any]:
    pdf.setFillColor(INK)
    title = (
        f"Axial tissue segmentation | {view.vertebral_level}"
        if panel.width >= 180
        else f"Axial overlay | {view.vertebral_level}"
    )
    _text(
        pdf,
        panel.left + CARD_PADDING,
        panel.top - CARD_TITLE_OFFSET,
        _truncate_to_width(title, panel.width - 2 * CARD_PADDING),
        size=8,
        bold=True,
    )
    pdf.setFillColor(MUTED)
    subtitle_prefix = "soft-tissue window" if panel.width >= 180 else "window"
    subtitle = (
        f"{subtitle_prefix} {int(view.ct_window_hu[0])} to {int(view.ct_window_hu[1])} HU"
        + ("" if view.l3_available else " | L3 unavailable")
    )
    _text(
        pdf,
        panel.left + CARD_PADDING,
        panel.top - CARD_SUBTITLE_OFFSET,
        _truncate_to_width(subtitle, panel.width - 2 * CARD_PADDING),
        size=8,
    )

    image_region = ImageBox(
        panel.left + CARD_PADDING,
        panel.bottom + 26,
        panel.width - 2 * CARD_PADDING,
        max(45.0, panel.height - 73),
    )
    box = _raster_box(view.rgb, image_region)
    pdf.drawImage(
        ImageReader(Image.fromarray(view.rgb, mode="RGB")),
        box.left,
        box.bottom,
        width=box.width,
        height=box.height,
        preserveAspectRatio=True,
        mask="auto",
    )
    pdf.setStrokeColor(colors.HexColor("#586872"))
    pdf.rect(box.left, box.bottom, box.width, box.height, stroke=1, fill=0)
    _draw_axial_marker(pdf, box.left + 10, box.bottom + box.height / 2, "R")
    _draw_axial_marker(pdf, box.right - 10, box.bottom + box.height / 2, "L")
    _draw_axial_marker(pdf, box.left + box.width / 2, box.top - 10, "A")
    _draw_axial_marker(pdf, box.left + box.width / 2, box.bottom + 10, "P")

    scale_mm = next(
        (
            candidate
            for candidate in (50.0, 20.0, 10.0)
            if box.width / view.field_of_view_mm[0] * candidate <= box.width / 3
        ),
        10.0,
    )
    scale_width = box.width / view.field_of_view_mm[0] * scale_mm
    scale_y = box.bottom + 9
    pdf.setStrokeColor(colors.white)
    pdf.setLineWidth(2)
    pdf.line(box.right - scale_width - 8, scale_y, box.right - 8, scale_y)
    pdf.line(box.right - scale_width - 8, scale_y - 3, box.right - scale_width - 8, scale_y + 3)
    pdf.line(box.right - 8, scale_y - 3, box.right - 8, scale_y + 3)
    pdf.setFillColor(colors.white)
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawRightString(box.right - 8, scale_y + 4, f"{int(scale_mm)} mm")

    legend_y = max(panel.bottom + 18, box.bottom - 17)
    legend_x = panel.left + CARD_PADDING
    palette_lookup = {
        "SM": "sm",
        "SAT": "sat",
        "aVAT": "avat",
        "tVAT": "tvat",
    }
    for name in view.tissue_names:
        palette_name = palette_lookup.get(name)
        if palette_name is None:
            continue
        item_width = _tissue_legend_item_width(name)
        if (
            legend_x > panel.left + CARD_PADDING
            and legend_x + item_width > panel.right - CARD_PADDING
        ):
            legend_x = panel.left + CARD_PADDING
            legend_y -= 10
        legend_x += _draw_tissue_legend_item(
            pdf,
            x=legend_x,
            y=legend_y,
            tissue_key=palette_name,
            label=name,
        )

    return {
        "method": view.method,
        "section_top_pt": panel.top,
        "section_bottom_pt": panel.bottom,
        "section_height_pt": panel.height,
        "vertebral_level": view.vertebral_level,
        "native_label": view.native_label,
        "position_superior_mm": view.position_superior_mm,
        "l3_available": view.l3_available,
        "tissue_names": list(view.tissue_names),
        "pixel_spacing_mm": view.pixel_spacing_mm,
        "field_of_view_mm": list(view.field_of_view_mm),
        "ct_window_hu": list(view.ct_window_hu),
        "legend_style": PLOT_LEGEND_STYLE,
    }


def _analysis_date(value: str | None) -> str:
    if not value:
        return "Not recorded"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return public_text(value, maximum=32)
    if parsed.utcoffset() is not None:
        parsed = parsed.astimezone(UTC)
        suffix = " UTC"
    else:
        suffix = ""
    return parsed.strftime("%Y-%m-%d %H:%M") + suffix


def _split_technical_lines(*values: str | None) -> tuple[str, ...]:
    lines: list[str] = []
    for value in values:
        if not value:
            continue
        lines.extend(
            component.strip()
            for component in re.split(r"\s*[|/]\s*", str(value))
            if component.strip()
        )
    return tuple(lines) or ("Not recorded",)


def _model_lines(case: CaseReportInput) -> tuple[str, ...]:
    orientation = case.orientation_result
    orientation = orientation.to_dict() if hasattr(orientation, "to_dict") else orientation
    orientation_model = "orientation not recorded"
    if isinstance(orientation, Mapping):
        model = orientation.get("model")
        if isinstance(model, Mapping):
            orientation_model = str(model.get("name") or model.get("asset_id") or orientation_model)
    tissue = case.measurement_bundle.provenance.get("tissue", {})
    tissue_model = (
        tissue.get("backend_id", "tissue not recorded")
        if isinstance(tissue, Mapping)
        else "tissue not recorded"
    )
    return _split_technical_lines(
        orientation_model,
        case.vertebral_result.backend_id,
        str(tissue_model),
    )


def _technical_metadata_items(
    case: CaseReportInput,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    metadata = case.technical_metadata
    scanner = (
        " ".join(
            value
            for value in (
                metadata.get("scanner_manufacturer"),
                metadata.get("scanner_model"),
            )
            if value
        )
        or "Not recorded"
    )
    runtime = _split_technical_lines(
        metadata.get("runtime_backend"),
        metadata.get("runtime_hardware"),
    )
    slices = case.measurement_bundle.slices
    thickness = pd.to_numeric(slices["slice_thickness_normal_mm"], errors="coerce")
    thickness = thickness[np.isfinite(thickness)]
    thickness_text = (
        f"{float(thickness.median()):.2f} mm" if not thickness.empty else "Not recorded"
    )
    spacing = case.prepared_image.GetSpacing()
    input_text = (
        f"{str(metadata.get('input_format') or 'unknown').upper()} "
        f"{spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm"
    )
    pipeline_version = metadata.get("pipeline_version") or __version__
    return (
        ("analysis", (_analysis_date(metadata.get("analysis_started_at")),)),
        ("input", (input_text,)),
        ("pipeline", (f"BodyComposition {pipeline_version}",)),
        ("runtime", runtime),
        ("scanner", (scanner,)),
        ("slice", (thickness_text,)),
        ("models", _model_lines(case)),
    )


def _review_note_text(entry: ReviewEntry) -> str:
    readable = REVIEW_NOTE_LABELS.get(entry.code) or entry.reason
    return public_text(readable, maximum=180)


def _report_notes(
    case: CaseReportInput,
    rows: list[dict[str, Any]],
    review_entries: tuple[ReviewEntry, ...],
    warnings: tuple[str, ...],
) -> _NotesSummary:
    displayed_entries = tuple(
        entry for entry in review_entries if entry.code not in ROUTINE_BOUNDARY_NOTE_CODES
    )
    suppressed_codes = tuple(
        sorted(
            {entry.code for entry in review_entries if entry.code in ROUTINE_BOUNDARY_NOTE_CODES}
        )
    )
    items = [_review_note_text(entry) for entry in displayed_entries]
    displayed_codes = {entry.code for entry in displayed_entries}
    orientation = case.orientation_result
    if isinstance(orientation, Mapping):
        orientation_changed = bool(orientation.get("orientation_changed", False))
        orientation_review = bool(orientation.get("manual_review_required", False))
    else:
        orientation_changed = bool(orientation.orientation_changed)
        orientation_review = bool(orientation.manual_review_required)
    unchanged_uncertain = orientation_review and not orientation_changed
    if orientation_changed and "severe_misorientation_repaired" not in displayed_codes:
        items.append("Input CT was automatically reoriented before analysis.")
    elif unchanged_uncertain and "orientation_mismatch_uncertain" not in displayed_codes:
        items.append("Processing continued without orientation repair.")
    if "missing_or_invalid_display_measurement" in warnings:
        missing = sum(not metric["valid"] for row in rows for metric in row["metrics"].values())
        items.append(f"Vertebral summary contains {missing} unavailable values.")
    anthropometry_note = _anthropometry_note(case.measurement_bundle.summaries.iloc[0])
    if anthropometry_note is not None:
        items.append(anthropometry_note)
    if not items:
        items.append("No report-level issues.")
    return _NotesSummary(
        items=tuple(items),
        review_count=len(review_entries),
        displayed_review_count=len(displayed_entries),
        suppressed_review_codes=suppressed_codes,
    )


def _note_item_block(
    item: str,
    width: float,
) -> list[str]:
    return _wrap_text(item, width - 10.0, size=8)


def _balanced_note_item_columns(
    blocks: list[list[str]],
    column_count: int,
    maximum_height: float,
) -> list[list[list[str]]] | None:
    if column_count < 1 or len(blocks) < column_count:
        return None
    line_height = 9.0
    block_gap = 3.0
    block_count = len(blocks)
    line_prefix = [0]
    for block in blocks:
        line_prefix.append(line_prefix[-1] + len(block))

    def group_height(start: int, end: int) -> float:
        return (line_prefix[end] - line_prefix[start]) * line_height + max(
            0, end - start - 1
        ) * block_gap

    infinity = float("inf")
    costs = [[infinity] * (block_count + 1) for _ in range(column_count + 1)]
    splits = [[-1] * (block_count + 1) for _ in range(column_count + 1)]
    costs[0][0] = 0.0
    for columns in range(1, column_count + 1):
        for end in range(columns, block_count + 1):
            for start in range(columns - 1, end):
                cost = max(costs[columns - 1][start], group_height(start, end))
                if cost < costs[columns][end]:
                    costs[columns][end] = cost
                    splits[columns][end] = start
    if costs[column_count][block_count] > maximum_height:
        return None

    boundaries = [block_count]
    end = block_count
    for columns in range(column_count, 0, -1):
        start = splits[columns][end]
        boundaries.append(start)
        end = start
    boundaries.reverse()
    return [blocks[boundaries[index] : boundaries[index + 1]] for index in range(column_count)]


def _draw_notes_card(
    pdf: canvas.Canvas,
    panel: ImageBox,
    notes: _NotesSummary,
) -> dict[str, Any]:
    pdf.setFillColor(INK)
    _text(
        pdf,
        panel.left + CARD_PADDING,
        panel.top - CARD_TITLE_OFFSET,
        "Notes",
        size=8,
        bold=True,
    )
    pdf.setStrokeColor(BORDER)
    pdf.setLineWidth(CARD_BORDER_WIDTH)
    header_rule_y = panel.top - 29.0
    pdf.line(
        panel.left + CARD_PADDING,
        header_rule_y,
        panel.right - CARD_PADDING,
        header_rule_y,
    )

    maximum_columns = 4
    column_gap = 20.0
    preferred_columns = 3
    available_height = panel.height - 45.0
    columns_used = 1
    column_width = panel.width - 2 * CARD_PADDING
    column_blocks: list[list[list[str]]] | None = None

    candidate_column_counts = [preferred_columns]
    if preferred_columns == 3 and len(notes.items) >= 4:
        candidate_column_counts.append(4)
    for candidate_columns in candidate_column_counts:
        candidate_width = (
            panel.width - 2 * CARD_PADDING - column_gap * (candidate_columns - 1)
        ) / candidate_columns
        blocks = [_note_item_block(item, candidate_width) for item in notes.items]
        candidate_blocks = (
            [[block] for block in blocks]
            + [[] for _ in range(candidate_columns - len(blocks))]
            if len(blocks) < candidate_columns
            else _balanced_note_item_columns(
                blocks,
                candidate_columns,
                available_height,
            )
        )
        if candidate_blocks is not None:
            columns_used = candidate_columns
            column_width = candidate_width
            column_blocks = candidate_blocks
            break

    if column_blocks is None:
        raise ValueError(
            "Notes contain too many plain-language issues for the released "
            "three- or four-column register."
        )

    body_top = panel.top - 42.0
    if columns_used > 1:
        pdf.setStrokeColor(BORDER)
        pdf.setLineWidth(CARD_BORDER_WIDTH)
        for column_index in range(1, columns_used):
            divider_x = (
                panel.left
                + CARD_PADDING
                + column_index * column_width
                + (column_index - 0.5) * column_gap
            )
            pdf.line(
                divider_x,
                panel.bottom + CARD_PADDING,
                divider_x,
                header_rule_y - 5,
            )

    column_line_counts: list[int] = []
    column_item_counts: list[int] = []
    for column_index, blocks in enumerate(column_blocks):
        x = panel.left + CARD_PADDING + column_index * (column_width + column_gap)
        y = body_top
        line_count = 0
        for block_index, block in enumerate(blocks):
            if block_index:
                y -= 3
            pdf.setFillColor(INK)
            pdf.circle(x + 2.2, y + 2.3, 1.2, stroke=0, fill=1)
            for line in block:
                pdf.setFillColor(MUTED)
                _text(pdf, x + 9, y, line, size=8)
                y -= 9
                line_count += 1
        column_line_counts.append(line_count)
        column_item_counts.append(len(blocks))
    return {
        "columns_used": columns_used,
        "minimum_columns": 3,
        "maximum_columns": maximum_columns,
        "column_width_pt": column_width,
        "column_gap_pt": column_gap,
        "content_padding_pt": CARD_PADDING,
        "column_item_counts": column_item_counts,
        "column_line_counts": column_line_counts,
        "item_count": len(notes.items),
        "all_issues_rendered": True,
        "list_style": "vector_bullets",
        "subheadings_visible": False,
        "block_split": False,
        "box_top_pt": panel.top,
        "outer_border_visible": False,
        "inner_header_rule_visible": True,
        "inner_column_divider_count": max(0, columns_used - 1),
        "structured": True,
        "review_count": notes.review_count,
        "displayed_review_count": notes.displayed_review_count,
        "suppressed_review_codes": list(notes.suppressed_review_codes),
        "status_visible": False,
        "rendered_line_count": sum(column_line_counts),
        "truncated": False,
    }


def _profile_layers(slices: pd.DataFrame) -> list[tuple[str, str, str]]:
    layers: list[tuple[str, str, str]] = []
    for name, label, column in (
        (
            "sm",
            "SM",
            "skeletal_muscle_tissue_hu_m29_150_area_cm2",
        ),
        (
            "sat",
            "SAT",
            "sat_total_hu_m190_m30_area_cm2",
        ),
        (
            "avat",
            "aVAT",
            "avat_hu_m190_m30_area_cm2",
        ),
        (
            "tvat",
            "tVAT",
            "tvat_hu_m190_m30_area_cm2",
        ),
    ):
        if column in slices:
            layers.append((name, label, column))
    return layers


def _contiguous_runs(valid: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    return [run for run in np.split(indices, splits) if len(run) >= 2]


def _styled_line_runs(
    plotted: np.ndarray,
    boundary_touching: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    if plotted.shape != boundary_touching.shape:
        raise ValueError("Profile line masks must have identical shapes.")
    runs: list[tuple[str, list[int]]] = []
    for index in range(len(plotted) - 1):
        if not (plotted[index] and plotted[index + 1]):
            continue
        style = (
            "boundary_touching"
            if boundary_touching[index] or boundary_touching[index + 1]
            else "strict_valid"
        )
        if runs and runs[-1][0] == style and runs[-1][1][-1] == index:
            runs[-1][1].append(index + 1)
        else:
            runs.append((style, [index, index + 1]))
    return [(style, np.asarray(indices, dtype=int)) for style, indices in runs]


def _display_integral_cm3(
    values_cm2: np.ndarray,
    positions_mm: np.ndarray,
    valid: np.ndarray,
) -> float:
    integral_cm2_mm = sum(
        float(np.trapezoid(values_cm2[run], positions_mm[run])) for run in _contiguous_runs(valid)
    )
    return integral_cm2_mm / 10.0


def _draw_profile(
    pdf: canvas.Canvas,
    case: CaseReportInput,
    projection: SagittalProjection,
    panel: ImageBox,
    shared_y_box: ImageBox,
    card_top: float,
) -> dict[str, Any]:
    slices = case.measurement_bundle.slices
    layers = _profile_layers(slices)
    if not layers:
        raise ValueError("spine_profile_v2 requires canonical per-slice tissue-area columns.")
    positions = pd.to_numeric(slices["position_superior_mm"], errors="coerce").to_numpy(dtype=float)
    values: list[np.ndarray] = []
    layer_validities: list[np.ndarray] = []
    display_valid = (
        np.isfinite(positions)
        & (positions >= projection.inferior_mm)
        & (positions <= projection.superior_mm)
    )
    for _name, _, column in layers:
        layer = pd.to_numeric(slices[column], errors="coerce").to_numpy(dtype=float)
        validity_column = column.replace("_cm2", "_valid")
        layer_valid = (
            slices[validity_column].fillna(False).astype(bool).to_numpy()
            if validity_column in slices
            else np.isfinite(layer)
        )
        layer_validities.append(display_valid & np.isfinite(layer) & (layer >= 0) & layer_valid)
        values.append(layer)
    safe_values = [
        np.where(layer_valid, layer, 0.0)
        for layer, layer_valid in zip(values, layer_validities, strict=True)
    ]
    any_layer_valid = np.any(np.vstack(layer_validities), axis=0)
    stack = np.sum(np.vstack(safe_values), axis=0)
    raw_max = float(np.nanmax(stack[any_layer_valid])) if np.any(any_layer_valid) else 0.0
    trunk = None
    trunk_valid = np.zeros(len(slices), dtype=bool)
    trunk_boundary_touching = np.zeros(len(slices), dtype=bool)
    trunk_plotted = np.zeros(len(slices), dtype=bool)
    trunk_invalid_reasons: list[str] = []
    if "trunk_area_cm2" in slices:
        trunk = pd.to_numeric(slices["trunk_area_cm2"], errors="coerce").to_numpy(dtype=float)
        trunk_observed = display_valid & np.isfinite(trunk) & (trunk >= 0)
        trunk_valid = trunk_observed.copy()
        if "trunk_area_valid" in slices:
            trunk_valid &= slices["trunk_area_valid"].fillna(False).astype(bool).to_numpy()
        if "trunk_area_reason" in slices:
            trunk_reasons = slices["trunk_area_reason"].fillna("").astype(str).to_numpy()
            boundary_only = np.ones(len(slices), dtype=bool)
            if "trunk_touching_fov" in slices:
                boundary_only &= slices["trunk_touching_fov"].fillna(False).astype(bool).to_numpy()
            for exclusion_column in (
                "trunk_mask_fragmented",
                "trunk_mask_internal_gap",
            ):
                if exclusion_column in slices:
                    boundary_only &= ~(
                        slices[exclusion_column].fillna(False).astype(bool).to_numpy()
                    )
            trunk_boundary_touching = (
                trunk_observed
                & ~trunk_valid
                & boundary_only
                & (trunk_reasons == "touching_image_boundary")
            )
        trunk_plotted = trunk_valid | trunk_boundary_touching
        if "trunk_area_reason" in slices:
            trunk_invalid_reasons = sorted(
                {
                    str(reason)
                    for reason in slices.loc[
                        display_valid & ~trunk_valid,
                        "trunk_area_reason",
                    ].dropna()
                    if str(reason).strip()
                }
            )
        if np.any(trunk_plotted):
            raw_max = max(raw_max, float(np.nanmax(trunk[trunk_plotted])))
    if raw_max <= 0:
        raise ValueError("spine_profile_v2 has no valid positive tissue-area profile.")
    magnitude = 10.0 ** np.floor(np.log10(raw_max))
    normalized = raw_max / magnitude
    nice = next(value for value in (1.0, 2.0, 5.0, 10.0) if normalized <= value)
    x_max = float(nice * magnitude)
    pdf.setFillColor(colors.HexColor("#263746"))
    _text(
        pdf,
        panel.left,
        card_top - CARD_TITLE_OFFSET,
        "Stacked tissue-area profile",
        size=8,
        bold=True,
    )
    hu_definition_caption = "SM -29..150 | fat -190..-30 HU"
    pdf.setFillColor(MUTED)
    _text(
        pdf,
        panel.left,
        card_top - CARD_SUBTITLE_OFFSET,
        hu_definition_caption,
        size=8,
    )
    pdf.setStrokeColor(colors.HexColor("#AEB8C0"))
    pdf.rect(panel.left, shared_y_box.bottom, panel.width, shared_y_box.height, stroke=1, fill=0)
    cumulative = np.zeros(len(slices), dtype=float)
    for (name, _label, _), layer_values, layer_valid in zip(
        layers,
        safe_values,
        layer_validities,
        strict=True,
    ):
        lower = cumulative.copy()
        upper = lower + layer_values
        red, green, blue = tissue_color(name)
        pdf.setFillColor(colors.Color(red / 255, green / 255, blue / 255, alpha=0.78))
        pdf.setStrokeColor(colors.Color(red / 255, green / 255, blue / 255))
        for run in _contiguous_runs(layer_valid):
            path = pdf.beginPath()
            first = int(run[0])
            path.moveTo(
                panel.left + lower[first] / x_max * panel.width,
                projection.y_for_superior(positions[first], shared_y_box.bottom, shared_y_box.top),
            )
            for index in run:
                path.lineTo(
                    panel.left + upper[index] / x_max * panel.width,
                    projection.y_for_superior(
                        positions[index], shared_y_box.bottom, shared_y_box.top
                    ),
                )
            for index in run[::-1]:
                path.lineTo(
                    panel.left + lower[index] / x_max * panel.width,
                    projection.y_for_superior(
                        positions[index], shared_y_box.bottom, shared_y_box.top
                    ),
                )
            path.close()
            pdf.drawPath(path, stroke=1, fill=1)
        cumulative = upper
    if trunk is not None:
        pdf.setLineWidth(0.8)
        for style, run in _styled_line_runs(
            trunk_plotted,
            trunk_boundary_touching,
        ):
            if style == "boundary_touching":
                pdf.setStrokeColor(colors.HexColor("#5D6469"))
                pdf.setDash(1.2, 2.4)
            else:
                pdf.setStrokeColor(colors.HexColor("#222222"))
                pdf.setDash()
            path = pdf.beginPath()
            for offset, index in enumerate(run):
                x = panel.left + trunk[index] / x_max * panel.width
                y = projection.y_for_superior(
                    positions[index], shared_y_box.bottom, shared_y_box.top
                )
                (path.moveTo if offset == 0 else path.lineTo)(x, y)
            pdf.drawPath(path, stroke=1, fill=0)
        pdf.setDash()
    tick_step = x_max / 4.0
    tick_precision = 0 if abs(tick_step - round(tick_step)) < 1e-9 else 1
    for value in np.linspace(0.0, x_max, 5):
        x = panel.left + value / x_max * panel.width
        pdf.setStrokeColor(colors.HexColor("#D7DDE1"))
        pdf.line(x, shared_y_box.bottom, x, shared_y_box.top)
        label = f"{value:.{tick_precision}f}"
        _draw_plot_tick(
            pdf,
            x=x,
            y=shared_y_box.bottom - 13,
            label=label,
        )
    _draw_plot_axis_title(
        pdf,
        x=panel.left + panel.width / 2,
        y=shared_y_box.bottom - 27,
        label="Tissue area (cm2)",
    )
    trunk_reference_label = None
    if trunk is not None and np.any(trunk_plotted):
        trunk_reference_label = (
            "Trunk area | FOV edge" if np.any(trunk_boundary_touching) else "Trunk area"
        )
        reference_y = shared_y_box.bottom - 43
        pdf.setStrokeColor(colors.HexColor("#222222"))
        pdf.setLineWidth(0.8)
        line_start = panel.left + 4
        line_end = line_start + 16
        pdf.line(line_start, reference_y + 3, line_end, reference_y + 3)
        pdf.setFillColor(INK)
        label_x = line_end + 4
        _text(pdf, label_x, reference_y, "Trunk area", size=8)
        if np.any(trunk_boundary_touching):
            boundary_line_start = (
                label_x + pdfmetrics.stringWidth("Trunk area", FONT_REGULAR, 8) + 7
            )
            boundary_line_end = boundary_line_start + 12
            pdf.setStrokeColor(colors.HexColor("#5D6469"))
            pdf.setDash(1.2, 2.4)
            pdf.line(
                boundary_line_start,
                reference_y + 3,
                boundary_line_end,
                reference_y + 3,
            )
            pdf.setDash()
            pdf.setFillColor(INK)
            _text(pdf, boundary_line_end + 4, reference_y, "FOV edge", size=8)
    legend_x = panel.left
    legend_y = shared_y_box.top + 7
    legend_rows = 1
    for name, label, _ in layers:
        item_width = _tissue_legend_item_width(label)
        if legend_x > panel.left and legend_x + item_width > panel.right:
            legend_x = panel.left
            legend_y -= 10
            legend_rows += 1
        legend_x += _draw_tissue_legend_item(
            pdf,
            x=legend_x,
            y=legend_y,
            tissue_key=name,
            label=label,
        )
    return {
        "layers": [name for name, _, _ in layers],
        "legend_labels": [label for _, label, _ in layers],
        "legend_rows": legend_rows,
        "legend_style": PLOT_LEGEND_STYLE,
        "axis_style": PLOT_AXIS_STYLE,
        "hu_definition_caption": hu_definition_caption,
        "source_columns": [column for _, _, column in layers],
        "displayed_layer_integrals_cm3": {
            name: _display_integral_cm3(layer, positions, layer_valid)
            for (name, _, _), layer, layer_valid in zip(
                layers,
                values,
                layer_validities,
                strict=True,
            )
        },
        "valid_slice_counts": {
            name: int(np.count_nonzero(layer_valid))
            for (name, _, _), layer_valid in zip(
                layers,
                layer_validities,
                strict=True,
            )
        },
        "x_axis": "area_cm2",
        "y_axis": "position_superior_mm",
        "displayed_superior_range_mm": [
            projection.inferior_mm,
            projection.superior_mm,
        ],
        "x_max_display_cm2": x_max,
        "smoothing": "none",
        "trunk_area_reference": "unfilled" if "trunk_area_cm2" in slices else "unavailable",
        "trunk_area_reference_label": trunk_reference_label,
        "trunk_area_reference_gap_policy": "never_bridge_omitted_slices",
        "trunk_area_reference_style_policy": {
            "strict_valid": "solid",
            "touching_image_boundary": "dotted",
            "other_invalid_or_missing": "omitted",
        },
        "trunk_valid_slice_count": int(np.count_nonzero(trunk_valid)),
        "trunk_touching_boundary_numeric_slice_count": int(
            np.count_nonzero(trunk_boundary_touching)
        ),
        "trunk_plotted_slice_count": int(np.count_nonzero(trunk_plotted)),
        "trunk_omitted_slice_count": int(np.count_nonzero(display_valid & ~trunk_plotted)),
        "trunk_solid_section_count": sum(
            style == "strict_valid"
            for style, _run in _styled_line_runs(
                trunk_plotted,
                trunk_boundary_touching,
            )
        ),
        "trunk_dotted_section_count": sum(
            style == "boundary_touching"
            for style, _run in _styled_line_runs(
                trunk_plotted,
                trunk_boundary_touching,
            )
        ),
        "trunk_invalid_reasons": trunk_invalid_reasons,
        "trunk_reference_integral_scope": "strict_valid_sections_only",
        "displayed_trunk_reference_integral_cm3": (
            _display_integral_cm3(trunk, positions, trunk_valid) if trunk is not None else None
        ),
    }


def _draw_hu_distributions(
    pdf: canvas.Canvas,
    case: CaseReportInput,
    panel: ImageBox,
    card_top: float,
) -> dict[str, Any]:
    table = case.measurement_bundle.hu_distributions
    if table.empty:
        raise ValueError("spine_profile_v2 requires canonical tissue HU distributions.")

    pdf.setFillColor(INK)
    _text(
        pdf,
        panel.left,
        card_top - CARD_TITLE_OFFSET,
        "Tissue HU distributions",
        size=8,
        bold=True,
    )
    pdf.setFillColor(MUTED)
    _text(
        pdf,
        panel.left,
        card_top - CARD_SUBTITLE_OFFSET,
        "Model-native masks",
        size=8,
    )
    _text(
        pdf,
        panel.left,
        card_top - CARD_SUBTITLE_OFFSET - 11.0,
        "median [IQR] | peak scale",
        size=8,
    )

    def x_for_hu(value: float) -> float:
        fraction = (float(value) - HU_DISTRIBUTION_MIN_HU) / (
            HU_DISTRIBUTION_MAX_HU - HU_DISTRIBUTION_MIN_HU
        )
        return panel.left + float(np.clip(fraction, 0.0, 1.0)) * panel.width

    axis_height = 24.0
    chart_bottom = panel.bottom + axis_height
    row_height = (panel.top - chart_bottom) / len(HU_DISTRIBUTION_TISSUES)
    statistics: dict[str, dict[str, Any]] = {}
    total_voxel_counts: dict[str, int] = {}
    contributing_slice_counts: dict[str, int] = {}
    clipped_voxel_counts: dict[str, int] = {}
    clipped_tail_marker_displayed = False

    for tissue_index, (tissue_key, display_label, _definition) in enumerate(
        HU_DISTRIBUTION_TISSUES
    ):
        group = table.loc[table["tissue_key"].eq(tissue_key)].sort_values("bin_index")
        if group.empty:
            raise ValueError(f"Missing canonical HU distribution for {display_label}.")
        first = group.iloc[0]
        row_top = panel.top - tissue_index * row_height
        row_bottom = row_top - row_height
        label_y = row_top - 9.0
        plot_top = row_top - 18.0
        plot_bottom = row_bottom + 5.0
        if plot_top <= plot_bottom:
            raise ValueError("Tissue HU distribution rows cannot fit the released layout.")

        red, green, blue = tissue_color(tissue_key)
        tissue_fill = colors.Color(
            red / 255,
            green / 255,
            blue / 255,
            alpha=0.62,
        )
        _draw_tissue_legend_item(
            pdf,
            x=panel.left,
            y=label_y - 1.0,
            tissue_key=tissue_key,
            label=display_label,
        )

        valid = bool(first["distribution_valid"])
        total_voxels = int(first["total_voxel_count"])
        contributing_slices = int(first["contributing_slice_count"])
        clipped_voxels = int(first["below_histogram_voxel_count"]) + int(
            first["above_histogram_voxel_count"]
        )
        clipped_fraction = clipped_voxels / total_voxels if total_voxels else 0.0
        clipped_tail_marker_displayed |= clipped_voxels > 0
        q1_hu = float(first["q1_hu"]) if pd.notna(first["q1_hu"]) else None
        median_hu = float(first["median_hu"]) if pd.notna(first["median_hu"]) else None
        q3_hu = float(first["q3_hu"]) if pd.notna(first["q3_hu"]) else None
        if not valid or median_hu is None or q1_hu is None or q3_hu is None:
            summary_text = "NA"
        else:
            tail_marker = "*" if clipped_voxels else ""
            summary_text = f"{median_hu:.0f} [{q1_hu:.0f},{q3_hu:.0f}]{tail_marker}"
        pdf.setFillColor(MUTED)
        pdf.setFont(FONT_REGULAR, 8)
        pdf.drawRightString(panel.right, label_y, summary_text)

        pdf.setStrokeColor(colors.HexColor("#D7DDE1"))
        pdf.setLineWidth(0.4)
        for guide_hu in (-30.0, 0.0):
            x = x_for_hu(guide_hu)
            pdf.line(x, plot_bottom, x, plot_top)
        pdf.setStrokeColor(colors.HexColor("#AEB8C0"))
        pdf.setLineWidth(0.6)
        pdf.line(panel.left, plot_bottom, panel.right, plot_bottom)

        fractions = pd.to_numeric(
            group["voxel_fraction"],
            errors="coerce",
        ).to_numpy(dtype=float)
        bin_lower = pd.to_numeric(
            group["bin_lower_hu"],
            errors="coerce",
        ).to_numpy(dtype=float)
        bin_upper = pd.to_numeric(
            group["bin_upper_hu"],
            errors="coerce",
        ).to_numpy(dtype=float)
        peak = float(np.max(fractions)) if fractions.size else 0.0
        if valid and peak > 0:
            for lower_hu, upper_hu, fraction in zip(
                bin_lower,
                bin_upper,
                fractions,
                strict=True,
            ):
                if fraction <= 0:
                    continue
                x0 = x_for_hu(lower_hu)
                x1 = x_for_hu(upper_hu)
                height = fraction / peak * (plot_top - plot_bottom)
                pdf.setFillColor(tissue_fill)
                pdf.rect(
                    x0,
                    plot_bottom,
                    max(0.3, x1 - x0),
                    height,
                    stroke=0,
                    fill=1,
                )
            if median_hu is not None:
                pdf.setStrokeColor(INK)
                pdf.setLineWidth(0.7)
                median_x = x_for_hu(median_hu)
                pdf.line(median_x, plot_bottom, median_x, plot_top)
        total_voxel_counts[tissue_key] = total_voxels
        contributing_slice_counts[tissue_key] = contributing_slices
        clipped_voxel_counts[tissue_key] = clipped_voxels
        reason_value = first["distribution_reason"]
        statistics[tissue_key] = {
            "valid": valid,
            "reason": None if pd.isna(reason_value) else str(reason_value),
            "mean_hu": (float(first["mean_hu"]) if pd.notna(first["mean_hu"]) else None),
            "standard_deviation_hu": (
                float(first["standard_deviation_hu"])
                if pd.notna(first["standard_deviation_hu"])
                else None
            ),
            "q1_hu": q1_hu,
            "median_hu": median_hu,
            "q3_hu": q3_hu,
            "source_compartment": str(first["source_compartment"]),
            "source_label_ids": str(first["source_label_ids"]),
            "source_semantics": str(first["source_semantics"]),
            "outside_display_range_voxel_fraction": clipped_fraction,
        }

    for value in (
        HU_DISTRIBUTION_MIN_HU,
        -30.0,
        HU_DISTRIBUTION_MAX_HU,
    ):
        x = x_for_hu(value)
        label = f"{value:.0f}"
        if value == HU_DISTRIBUTION_MIN_HU:
            alignment = "left"
        elif value == HU_DISTRIBUTION_MAX_HU:
            alignment = "right"
        else:
            alignment = "center"
        _draw_plot_tick(
            pdf,
            x=x,
            y=panel.bottom + 8.0,
            label=label,
            alignment=alignment,
        )
    _draw_plot_axis_title(
        pdf,
        x=panel.left + panel.width / 2,
        y=panel.bottom - 5.0,
        label=("HU | * outside plot" if clipped_tail_marker_displayed else "HU"),
    )
    first_row = table.iloc[0]
    return {
        "tissues": [key for key, _label, _definition in HU_DISTRIBUTION_TISSUES],
        "display_labels": [label for _key, label, _definition in HU_DISTRIBUTION_TISSUES],
        "legend_style": PLOT_LEGEND_STYLE,
        "axis_style": PLOT_AXIS_STYLE,
        "x_axis_label": "HU",
        "source_table": "hu_distributions.parquet",
        "distribution_schema_version": str(first_row["distribution_schema_version"]),
        "distribution_scope": str(first_row["distribution_scope"]),
        "scope_superior_range_mm": [
            float(first_row["scope_inferior_position_superior_mm"]),
            float(first_row["scope_superior_position_superior_mm"]),
        ],
        "histogram_range_hu": [
            HU_DISTRIBUTION_MIN_HU,
            HU_DISTRIBUTION_MAX_HU,
        ],
        "bin_width_hu": HU_DISTRIBUTION_BIN_WIDTH_HU,
        "bin_semantics": str(first_row["bin_semantics"]),
        "normalization": "within_tissue_peak",
        "cross_tissue_magnitude_comparison": False,
        "summary_statistics": "exact_raw_voxel_mean_sd_median_and_iqr",
        "iqr_display": "numeric_summary_only",
        "iqr_band_displayed": False,
        "median_reference_line_displayed": True,
        "source_semantics": "model_native_compartment",
        "quantile_method": str(first_row["quantile_method"]),
        "statistics": statistics,
        "total_voxel_counts": total_voxel_counts,
        "contributing_slice_counts": contributing_slice_counts,
        "clipped_voxel_counts": clipped_voxel_counts,
        "clipped_tail_marker_displayed": clipped_tail_marker_displayed,
        "per_slice_mean_hu_displayed": False,
        "smoothing": "none",
    }


def render_case_page(
    destination: Path,
    *,
    case: CaseReportInput,
    settings: ReportingSettings,
    report_id: str,
    projection: SagittalProjection,
    axial_view: AxialSegmentationView,
    rows: list[dict[str, Any]],
    review_entries: tuple[ReviewEntry, ...],
    warnings: tuple[str, ...],
) -> dict[str, Any]:
    """Render exactly one fixed A4 landscape page and return display audit data."""

    pdf = _new_canvas(destination, case_id=case.case_id)
    pdf.bookmarkPage(case.case_id)
    pdf.setFillColor(PAGE_BACKGROUND)
    pdf.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
    content_top, header_audit = _header(pdf, case, report_id)
    notes_card = ImageBox(24, 18, PAGE_WIDTH - 48, 100)
    content_bottom = notes_card.top + CARD_GAP
    _draw_card(pdf, notes_card, border_visible=False)
    notes_audit = _draw_notes_card(
        pdf,
        notes_card,
        _report_notes(case, rows, review_entries, warnings),
    )
    anthropometry_markers = _anthropometry_spine_markers(
        case.measurement_bundle.summaries.iloc[0],
        projection,
    )
    audit: dict[str, Any] = {
        "layout": settings.layout,
        "layout_revision": LAYOUT_REVISION,
        "displayed_rows": rows,
        "technical_metadata": dict(case.technical_metadata),
        "design": {
            "card_gap_pt": CARD_GAP,
            "content_padding_pt": CARD_PADDING,
            "title_baseline_from_top_pt": CARD_TITLE_OFFSET,
            "subtitle_baseline_from_top_pt": CARD_SUBTITLE_OFFSET,
            "main_card_top_pt": content_top,
            "main_panel_style": "open_editorial_grid",
            "card_corner_radius_pt": CARD_CORNER_RADIUS,
            "rule_width_pt": CARD_BORDER_WIDTH,
        },
        "header": header_audit,
        "notes_qc": {
            **notes_audit,
            "box_width_pt": notes_card.width,
            "box_height_pt": notes_card.height,
        },
        "sagittal_projection": {
            "method": projection.method,
            "crop_basis": projection.crop_basis,
            "display_bounds_lps_mm": {
                "anterior": projection.anterior_mm,
                "posterior": projection.posterior_mm,
                "inferior": projection.inferior_mm,
                "superior": projection.superior_mm,
                "slab_left": projection.slab_left_mm,
                "slab_right": projection.slab_right_mm,
            },
            "crop_margins_mm": {
                "anterior_posterior": projection.anterior_posterior_margin_mm,
                "superior_inferior": projection.superior_inferior_margin_mm,
            },
            "anthropometry_markers": list(anthropometry_markers),
            "vertebral_labels": {
                "style": VERTEBRAL_LABEL_STYLE,
                "placement": "vertebral_centroid",
                "leader_lines_visible": False,
                "background": "translucent_dark_neutral",
                "background_alpha": VERTEBRAL_LABEL_BACKGROUND_ALPHA,
                "minimum_vertical_gap_pt": VERTEBRAL_LABEL_MIN_GAP_PT,
            },
        },
    }
    if settings.layout == "spine_overview_v1":
        sagittal_card = ImageBox(24, content_bottom, 225, content_top - content_bottom)
        table_card = ImageBox(
            sagittal_card.right + CARD_GAP,
            content_bottom,
            PAGE_WIDTH - 494,
            content_top - content_bottom,
        )
        axial_card, anthropometry_card, technical_card = _right_column_panels(
            left=table_card.right + CARD_GAP,
            width=205,
            content_bottom=content_bottom,
            content_top=content_top,
        )
        audit["panel_order"] = [
            "sagittal_spine",
            "vertebral_summary",
            "axial_segmentation",
            "key_anthropometry",
            "technical_context",
        ]
        right_column = ImageBox(
            axial_card.left,
            content_bottom,
            axial_card.width,
            content_top - content_bottom,
        )
        audit["main_grid"] = _draw_main_grid(
            pdf,
            left=sagittal_card.left,
            right=right_column.right,
            bottom=content_bottom,
            top=content_top,
            dividers=(
                sagittal_card.right + CARD_GAP / 2,
                table_card.right + CARD_GAP / 2,
            ),
        )
        audit["right_column_stack"] = {
            "style": "open_editorial_stack",
            "section_order": ["axial_segmentation", "key_anthropometry", "technical_context"],
            "horizontal_divider_count": 2,
            "divider_y_pt": [anthropometry_card.top, technical_card.top],
            "rule_width_pt": CARD_BORDER_WIDTH,
        }
        pdf.setFillColor(INK)
        _text(
            pdf,
            sagittal_card.left + CARD_PADDING,
            sagittal_card.top - CARD_TITLE_OFFSET,
            "Spine localization",
            size=8,
            bold=True,
        )
        pdf.setFillColor(MUTED)
        _text(
            pdf,
            sagittal_card.left + CARD_PADDING,
            sagittal_card.top - CARD_SUBTITLE_OFFSET,
            "Sagittal thick-slab projection",
            size=8,
        )
        left_panel = ImageBox(
            sagittal_card.left + CARD_PADDING,
            sagittal_card.bottom + 24,
            sagittal_card.width - 2 * CARD_PADDING,
            sagittal_card.height - 68,
        )
        image_box = _draw_projection(
            pdf,
            projection,
            left_panel,
            ct_window=settings.ct_window,
            rows=rows,
            show_title=False,
            anthropometry_markers=anthropometry_markers,
        )
        pdf.setFillColor(MUTED)
        _text(
            pdf,
            sagittal_card.left + CARD_PADDING,
            sagittal_card.bottom + 9,
            f"window {int(settings.ct_window[0])} to {int(settings.ct_window[1])} HU",
            size=8,
        )
        audit["axial_view"] = _draw_axial_panel(
            pdf,
            axial_view,
            axial_card,
        )
        audit["anthropometry"] = _draw_anthropometry_card(
            pdf,
            case,
            settings,
            anthropometry_card,
        )
        audit["technical_context"] = _draw_technical_context(
            pdf,
            case,
            technical_card,
        )
        table_panel = ImageBox(
            table_card.left + CARD_PADDING,
            table_card.bottom + 24,
            table_card.width - 2 * CARD_PADDING,
            table_card.height - 34,
        )
        audit["table"] = _draw_table(pdf, rows, settings, projection, table_panel)
    elif settings.layout == "spine_profile_v2":
        sagittal_card = ImageBox(24, content_bottom, 140, content_top - content_bottom)
        profile_card = ImageBox(
            sagittal_card.right + CARD_GAP,
            content_bottom,
            165,
            content_top - content_bottom,
        )
        hu_card = ImageBox(
            profile_card.right + CARD_GAP,
            content_bottom,
            124,
            content_top - content_bottom,
        )
        axial_card, anthropometry_card, technical_card = _right_column_panels(
            left=PAGE_WIDTH - 169,
            width=145,
            content_bottom=content_bottom,
            content_top=content_top,
        )
        table_card = ImageBox(
            hu_card.right + CARD_GAP,
            content_bottom,
            axial_card.left - hu_card.right - 2 * CARD_GAP,
            content_top - content_bottom,
        )
        audit["panel_order"] = [
            "sagittal_spine",
            "tissue_area_profile",
            "tissue_hu_distribution",
            "vertebral_summary",
            "axial_segmentation",
            "key_anthropometry",
            "technical_context",
        ]
        right_column = ImageBox(
            axial_card.left,
            content_bottom,
            axial_card.width,
            content_top - content_bottom,
        )
        audit["main_grid"] = _draw_main_grid(
            pdf,
            left=sagittal_card.left,
            right=right_column.right,
            bottom=content_bottom,
            top=content_top,
            dividers=(
                sagittal_card.right + CARD_GAP / 2,
                profile_card.right + CARD_GAP / 2,
                hu_card.right + CARD_GAP / 2,
                table_card.right + CARD_GAP / 2,
            ),
        )
        audit["right_column_stack"] = {
            "style": "open_editorial_stack",
            "section_order": ["axial_segmentation", "key_anthropometry", "technical_context"],
            "horizontal_divider_count": 2,
            "divider_y_pt": [anthropometry_card.top, technical_card.top],
            "rule_width_pt": CARD_BORDER_WIDTH,
        }
        pdf.setFillColor(INK)
        _text(
            pdf,
            sagittal_card.left + CARD_PADDING,
            sagittal_card.top - CARD_TITLE_OFFSET,
            "Spine localization",
            size=8,
            bold=True,
        )
        pdf.setFillColor(MUTED)
        _text(
            pdf,
            sagittal_card.left + CARD_PADDING,
            sagittal_card.top - CARD_SUBTITLE_OFFSET,
            "Sagittal thick-slab projection",
            size=8,
        )
        profile_axis_space = 54.0
        image_top_space = 51.0
        left_panel = ImageBox(
            sagittal_card.left + CARD_PADDING,
            sagittal_card.bottom + profile_axis_space,
            sagittal_card.width - 2 * CARD_PADDING,
            sagittal_card.height - profile_axis_space - image_top_space,
        )
        image_box = _draw_projection(
            pdf,
            projection,
            left_panel,
            ct_window=settings.ct_window,
            rows=rows,
            show_title=False,
            anthropometry_markers=anthropometry_markers,
        )
        pdf.setFillColor(MUTED)
        _text(
            pdf,
            sagittal_card.left + CARD_PADDING,
            sagittal_card.bottom + 9,
            f"window {int(settings.ct_window[0])} to {int(settings.ct_window[1])} HU",
            size=8,
        )
        audit["axial_view"] = _draw_axial_panel(
            pdf,
            axial_view,
            axial_card,
        )
        audit["anthropometry"] = _draw_anthropometry_card(
            pdf,
            case,
            settings,
            anthropometry_card,
        )
        audit["technical_context"] = _draw_technical_context(
            pdf,
            case,
            technical_card,
        )
        profile_panel = ImageBox(
            profile_card.left + CARD_PADDING,
            profile_card.bottom + profile_axis_space,
            profile_card.width - 2 * CARD_PADDING,
            image_box.height,
        )
        hu_panel = ImageBox(
            hu_card.left + CARD_PADDING,
            hu_card.bottom + profile_axis_space,
            hu_card.width - 2 * CARD_PADDING,
            image_box.height,
        )
        table_panel = ImageBox(
            table_card.left + CARD_PADDING,
            table_card.bottom + 24,
            table_card.width - 2 * CARD_PADDING,
            table_card.height - 34,
        )
        audit["profile"] = _draw_profile(
            pdf,
            case,
            projection,
            profile_panel,
            image_box,
            profile_card.top,
        )
        audit["hu_distributions"] = _draw_hu_distributions(
            pdf,
            case,
            hu_panel,
            hu_card.top,
        )
        audit["table"] = _draw_table(pdf, rows, settings, projection, table_panel)
    else:
        raise ValueError(f"Unsupported report layout {settings.layout!r}.")
    pdf.showPage()
    pdf.save()
    return audit


def render_failure_page(
    destination: Path,
    *,
    case_id: str,
    report_id: str,
    settings: ReportingSettings,
    failure_stage: str,
    failure_code: str,
    projection: SagittalProjection | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render one explicit non-success page without raw exception/path text.

    When the prepared CT and vertebral-body result remain usable, the page
    preserves that available overview. It never fabricates missing measurement values.
    """

    pdf = _new_canvas(destination, case_id=case_id)
    pdf.bookmarkPage(case_id)
    pdf.setFillColor(colors.HexColor("#17324D"))
    pdf.rect(0, PAGE_HEIGHT - 47, PAGE_WIDTH, 47, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    case_title = f"Case {case_id}"
    page_identity = "Page 1 of 1"
    identity_width = pdfmetrics.stringWidth(page_identity, FONT_REGULAR, 8)
    _text(
        pdf,
        28,
        PAGE_HEIGHT - 25,
        case_title,
        size=_fit_font_size(
            case_title,
            PAGE_WIDTH - 56 - identity_width - 18,
            maximum=13,
            font=FONT_BOLD,
        ),
        bold=True,
    )
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawRightString(PAGE_WIDTH - 28, PAGE_HEIGHT - 25, page_identity)
    pdf.setFillColor(colors.HexColor("#A43E25"))
    banner_bottom = PAGE_HEIGHT - (103 if projection is not None else 145)
    banner_height = 36 if projection is not None else 45
    pdf.roundRect(
        55,
        banner_bottom,
        PAGE_WIDTH - 110,
        banner_height,
        5,
        stroke=0,
        fill=1,
    )
    pdf.setFillColor(colors.white)
    banner_y = banner_bottom + banner_height / 2 - 4
    _centered(
        pdf,
        PAGE_WIDTH / 2,
        banner_y,
        "ANALYSIS INCOMPLETE - FAILURE PAGE",
        size=13,
        bold=True,
    )

    audit: dict[str, Any] = {
        "partial_spine_overview": projection is not None,
        "measurement_status": "unavailable",
    }
    if projection is None:
        pdf.setFillColor(colors.HexColor("#263746"))
        _text(
            pdf,
            85,
            PAGE_HEIGHT - 205,
            f"Requested layout: {settings.layout}",
            size=10,
            bold=True,
        )
        _text(
            pdf,
            85,
            PAGE_HEIGHT - 235,
            f"Failure stage: {public_text(failure_stage, maximum=60)}",
            size=10,
        )
        _text(
            pdf,
            85,
            PAGE_HEIGHT - 265,
            f"Stable reason: {public_text(failure_code, maximum=80)}",
            size=10,
        )
        _text(
            pdf,
            85,
            PAGE_HEIGHT - 315,
            "This page records pipeline/report status only. It is not evidence that the scientific analysis succeeded.",
            size=9,
        )
        _text(
            pdf,
            85,
            PAGE_HEIGHT - 340,
            "Inspect the canonical machine-readable QC and log artifacts for authorized technical review.",
            size=9,
        )
    else:
        rows = rows or []
        content_bottom = 58.0
        content_top = PAGE_HEIGHT - 126.0
        if settings.layout == "spine_overview_v1":
            projection_panel = ImageBox(
                28,
                content_bottom,
                300,
                content_top - content_bottom,
            )
            status_panel = ImageBox(
                350,
                content_bottom,
                PAGE_WIDTH - 378,
                content_top - content_bottom,
            )
        else:
            projection_panel = ImageBox(
                28,
                content_bottom,
                230,
                content_top - content_bottom,
            )
            unavailable_panel = ImageBox(
                276,
                content_bottom,
                170,
                content_top - content_bottom,
            )
            pdf.setDash(4, 3)
            pdf.setStrokeColor(colors.HexColor("#A43E25"))
            pdf.rect(
                unavailable_panel.left,
                unavailable_panel.bottom,
                unavailable_panel.width,
                unavailable_panel.height,
                stroke=1,
                fill=0,
            )
            pdf.setDash()
            pdf.setFillColor(colors.HexColor("#A43E25"))
            _centered(
                pdf,
                unavailable_panel.left + unavailable_panel.width / 2,
                unavailable_panel.top - 20,
                "Stacked tissue-area profile",
                size=8,
                bold=True,
            )
            _centered(
                pdf,
                unavailable_panel.left + unavailable_panel.width / 2,
                unavailable_panel.top - 40,
                "unavailable",
                size=8,
                bold=True,
            )
            _centered(
                pdf,
                unavailable_panel.left + unavailable_panel.width / 2,
                unavailable_panel.top - 57,
                "Measurement output incomplete",
                size=8,
            )
            status_panel = ImageBox(
                464,
                content_bottom,
                PAGE_WIDTH - 492,
                content_top - content_bottom,
            )

        _draw_projection(
            pdf,
            projection,
            projection_panel,
            ct_window=settings.ct_window,
            rows=rows,
        )
        pdf.setFillColor(colors.HexColor("#F6F8F9"))
        pdf.setStrokeColor(colors.HexColor("#AEB8C0"))
        pdf.roundRect(
            status_panel.left,
            status_panel.bottom,
            status_panel.width,
            status_panel.height,
            4,
            stroke=1,
            fill=1,
        )
        left = status_panel.left + 14
        y = status_panel.top - 19
        available_width = status_panel.width - 28
        pdf.setFillColor(colors.HexColor("#263746"))
        _text(pdf, left, y, "AVAILABLE SPINE OVERVIEW", size=9, bold=True)
        y -= 23
        _text(
            pdf,
            left,
            y,
            _truncate_to_width(
                f"Failure stage: {public_text(failure_stage, maximum=60)}",
                available_width,
            ),
            size=8,
        )
        y -= 17
        _text(
            pdf,
            left,
            y,
            _truncate_to_width(
                f"Stable reason: {public_text(failure_code, maximum=80)}",
                available_width,
            ),
            size=8,
        )
        y -= 23
        _text(pdf, left, y, "Available inputs", size=8, bold=True)
        y -= 16
        _text(pdf, left + 8, y, "Prepared CT and vertebral-body labels", size=8)
        y -= 16
        levels = ", ".join(str(row["vertebral_level"]) for row in rows) or "none"
        _text(
            pdf,
            left + 8,
            y,
            _truncate_to_width(f"Detected levels: {levels}", available_width - 8),
            size=8,
        )
        y -= 25
        pdf.setFillColor(colors.HexColor("#A43E25"))
        _text(pdf, left, y, "Measurements unavailable", size=8, bold=True)
        pdf.setFillColor(colors.HexColor("#263746"))
        for column in settings.measurement_columns:
            y -= 16
            definition = MEASUREMENT_DEFINITIONS[column]
            _text(
                pdf,
                left + 8,
                y,
                f"{definition['short_label']} ({definition['unit']}): {settings.missing_value_symbol}*",
                size=8,
            )
        y -= 24
        _text(
            pdf,
            left,
            y,
            _truncate_to_width(
                "No values were reconstructed or inferred for this page.",
                available_width,
            ),
            size=8,
            bold=True,
        )
        y -= 19
        _text(
            pdf,
            left,
            y,
            _truncate_to_width(
                "This page is not evidence that scientific analysis succeeded.",
                available_width,
            ),
            size=8,
        )
        audit.update(
            {
                "displayed_native_labels": list(projection.native_labels),
                "measurement_columns": list(settings.measurement_columns),
            }
        )
    pdf.showPage()
    pdf.save()
    return audit


def render_review_summary_page(
    destination: Path,
    *,
    total_cases: int,
    page_count: int,
    rows: list[dict[str, Any]],
) -> None:
    """Render the conditional single-page review index at fixed capacity."""

    if len(rows) > 24:
        raise ValueError("Manual-review summary exceeds the released 24-row capacity.")
    pdf = _new_canvas(destination, case_id="review-summary")
    pdf.bookmarkPage("review-summary")
    pdf.setFillColor(colors.HexColor("#17324D"))
    pdf.rect(0, PAGE_HEIGHT - 52, PAGE_WIDTH, 52, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    _text(pdf, 28, PAGE_HEIGHT - 30, "Manual-review summary", size=14, bold=True)
    _text(
        pdf,
        260,
        PAGE_HEIGHT - 29,
        f"{len(rows)} flagged cases of {total_cases}",
        size=8,
    )
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawRightString(
        PAGE_WIDTH - 28,
        PAGE_HEIGHT - 29,
        f"Page 1 of {page_count}",
    )
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        for entry in row["review_entries"]:
            key = (entry["domain"], entry["code"])
            counts[key] = counts.get(key, 0) + 1
    count_text = "; ".join(
        f"{domain}:{code}={count}" for (domain, code), count in sorted(counts.items())
    )
    pdf.setFillColor(colors.HexColor("#263746"))
    _text(pdf, 28, PAGE_HEIGHT - 73, _truncate_to_width(count_text, PAGE_WIDTH - 56), size=8)
    columns = (
        ("Case", 28, 85),
        ("Domain / code", 119, 145),
        ("Reason", 270, 150),
        ("Automatic action", 426, 180),
        ("Review status", 612, 90),
        ("Page", 708, 50),
    )
    header_y = PAGE_HEIGHT - 96
    for label, x, _ in columns:
        _text(pdf, x, header_y, label, size=8, bold=True)
    pdf.setStrokeColor(colors.HexColor("#9CAAB5"))
    pdf.line(28, header_y - 5, PAGE_WIDTH - 28, header_y - 5)
    row_height = 18.0
    for index, row in enumerate(rows):
        y = header_y - 21 - index * row_height
        entries = row["review_entries"]
        domains = ", ".join(f"{entry['domain']}:{entry['code']}" for entry in entries)
        reasons = "; ".join(entry["reason"] for entry in entries)
        actions = "; ".join(dict.fromkeys(entry["automatic_action"] for entry in entries))
        review_statuses = ", ".join(dict.fromkeys(entry["review_status"] for entry in entries))
        values = (
            row["case_id"],
            domains,
            reasons,
            actions,
            review_statuses,
            str(row["combined_page_number"]),
        )
        for value, (_, x, width) in zip(values, columns, strict=True):
            _text(pdf, x, y, _truncate_to_width(str(value), width), size=8)
        pdf.setStrokeColor(colors.HexColor("#E0E5E8"))
        pdf.line(28, y - 5, PAGE_WIDTH - 28, y - 5)
    _text(
        pdf,
        28,
        45,
        "Automated QC index, not an adjudication. Canonical JSON and QC artifacts remain authoritative.",
        size=8,
    )
    pdf.showPage()
    pdf.save()
