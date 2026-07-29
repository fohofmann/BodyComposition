"""Headless one-page PDF renderers for both fixed case-report layouts."""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Mapping, Sequence
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
    HU_DISTRIBUTION_COMPARTMENTS,
    HU_DISTRIBUTION_MAX_HU,
    HU_DISTRIBUTION_MIN_HU,
)
from BodyComposition.reporting.contracts import (
    MEASUREMENT_DEFINITIONS,
    CaseReportInput,
    ReportingSettings,
    ReviewEntry,
)
from BodyComposition.reporting.metrics import (
    ROUTINE_BOUNDARY_REVIEW_CODES,
    public_text,
)
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
LAYOUT_REVISION = "scientific_onepager_v35"
RUN_COVER_REVISION = "scientific_run_cover_v17"
RUN_COVER_MAX_CONFIGURATION_ROWS = 12
RUN_COVER_MAX_PIPELINE_ROWS = 16
RUN_COVER_HIDDEN_MODEL_ROLES = frozenset({"body surface", "landmarks"})
RUN_COVER_PIPELINE_SEQUENCE = (
    "Analysis scope",
    "Orientation",
    "Orientation policy",
    "Orientation decision",
    "Anatomy evidence",
    "Axial 180 repair",
    "Vertebral bodies",
    "Native compartments",
    "Body surface",
    "Surface cleanup",
    "Measurement coverage",
    "Vertebral cleanup",
    "SM HU filter",
    "AT HU filter",
    "Deterministic execution",
)
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
HU_FILTER_INACTIVE_CAPTION = "HU filter inactive"
HU_FILTER_RANGES = {
    "SM": (-29, 150),
    "AT": (-190, -30),
}
ANTHROPOMETRY_ASTERISK_EXPLANATION = "* incomplete range or FOV edge"
VERTEBRAL_LABEL_STYLE = "centered_translucent_badge_v2"
VERTEBRAL_LABEL_BACKGROUND_ALPHA = 0.55
VERTEBRAL_LABEL_MIN_GAP_PT = 10.0
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
RUN_COVER_ISSUE_LABELS = {
    "disconnected_vertebral_body": "Disconnected vertebral body segmentation",
    "vertebral_body_extent_invalid": "Incomplete vertebral body extent",
    "slice_measurement_invalid": "Invalid slice measurement",
    "body_surface_touches_fov": "Body surface reaches field of view",
    "trunk_contour_touches_fov": "Trunk contour reaches field of view",
    "pelvic_maximum_unavailable": "Pelvic max unavailable",
    "minimum_waist_unavailable": "Waist min unavailable",
    "midwaist_measurement_unavailable": "Mid-waist unavailable",
    "severe_misorientation_repaired": "Substantial orientation repair applied",
    "orientation_mismatch_uncertain": "Orientation remains uncertain",
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


def _display_unit(unit: str) -> str:
    return str(unit)


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


def _draw_filter_icon(
    pdf: canvas.Canvas,
    *,
    x: float,
    y: float,
    size: float = 8.0,
) -> None:
    """Draw a compact funnel icon without relying on a symbol font."""

    scale = size / 8.0
    pdf.saveState()
    pdf.setStrokeColor(MUTED)
    pdf.setLineWidth(0.7)
    pdf.setLineCap(1)
    pdf.setLineJoin(1)
    path = pdf.beginPath()
    path.moveTo(x, y + 7.0 * scale)
    path.lineTo(x + 8.0 * scale, y + 7.0 * scale)
    path.lineTo(x + 5.0 * scale, y + 3.5 * scale)
    path.lineTo(x + 5.0 * scale, y)
    path.lineTo(x + 3.0 * scale, y + 1.0 * scale)
    path.lineTo(x + 3.0 * scale, y + 3.5 * scale)
    path.close()
    pdf.drawPath(path, stroke=1, fill=0)
    pdf.restoreState()


def _hu_filter_display(
    active_ranges: Mapping[str, tuple[int, int]],
    *,
    mixed: bool = False,
) -> dict[str, Any]:
    ordered_ranges = [
        (label, active_ranges[label])
        for label in ("SM", "AT")
        if label in active_ranges
    ]
    if not ordered_ranges:
        return {
            "status": "inactive",
            "caption": HU_FILTER_INACTIVE_CAPTION,
            "active_ranges_hu": {},
        }
    caption = " | ".join(
        f"{label} {lower}..{upper}"
        for label, (lower, upper) in ordered_ranges
    )
    caption = f"{caption} HU"
    if mixed:
        caption = f"{caption} | mixed"
    return {
        "status": "mixed" if mixed else "active",
        "caption": caption,
        "active_ranges_hu": {
            label: [lower, upper]
            for label, (lower, upper) in ordered_ranges
        },
    }


def _table_hu_filter_display(columns: tuple[str, ...]) -> dict[str, Any]:
    active_ranges: dict[str, tuple[int, int]] = {}
    unfiltered_tissue_metric = False
    for column in columns:
        definition = MEASUREMENT_DEFINITIONS[column]
        if definition["short_label"] == "Trunk":
            continue
        source = definition["source"].lower()
        if "hu_m29_150" in source:
            active_ranges["SM"] = HU_FILTER_RANGES["SM"]
        elif "hu_m190_m30" in source:
            active_ranges["AT"] = HU_FILTER_RANGES["AT"]
        else:
            unfiltered_tissue_metric = True
    return _hu_filter_display(
        active_ranges,
        mixed=bool(active_ranges) and unfiltered_tissue_metric,
    )


def _draw_hu_filter_caption(
    pdf: canvas.Canvas,
    *,
    x: float,
    y: float,
    display: Mapping[str, Any],
) -> dict[str, Any]:
    _draw_filter_icon(
        pdf,
        x=x,
        y=y - 1.0,
    )
    pdf.setFillColor(MUTED)
    _text(
        pdf,
        x + 12.0,
        y,
        display["caption"],
        size=8,
    )
    return {
        **dict(display),
        "icon": "funnel",
    }


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
    for label, prefix in (
        ("Waist min", "ct_min_trunk_circumference_t10_l5"),
        ("Pelvic max", "ct_max_pelvic_circumference"),
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
                "line_style": "solid",
                "label_side": "left",
                "label_placement": "below_rule",
                "label_placement_fallback": "above_rule_at_image_edge",
            }
        )
    return tuple(markers)


def _anthropometry_marker_label_box(
    projection: SagittalProjection,
    box: ImageBox,
    marker: Mapping[str, Any],
) -> ImageBox:
    """Return the shared left-aligned label geometry for an anthropometry rule."""

    label = str(marker["label"])
    label_width = pdfmetrics.stringWidth(label, FONT_BOLD, 8) + 8
    label_height = 11.0
    label_gap = 3.0
    position = float(marker["position_superior_mm"])
    y = projection.y_for_superior(position, box.bottom, box.top)
    label_bottom = y - label_gap - label_height
    if label_bottom < box.bottom + 3:
        label_bottom = y + label_gap
    label_bottom = float(
        np.clip(label_bottom, box.bottom + 3, box.top - label_height - 3)
    )
    return ImageBox(
        left=box.left + 3,
        bottom=label_bottom,
        width=label_width,
        height=label_height,
    )


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
        pdf.saveState()
        pdf.setDash([])
        pdf.setStrokeColor(colors.white)
        pdf.setLineWidth(2.4)
        pdf.line(box.left, y, box.right, y)
        pdf.setStrokeColor(colors.HexColor("#203746"))
        pdf.setLineWidth(0.9)
        pdf.line(box.left, y, box.right, y)
        pdf.restoreState()

        label = str(marker["label"])
        label_box = _anthropometry_marker_label_box(projection, box, marker)
        pdf.setFillColor(colors.HexColor("#203746"))
        pdf.roundRect(
            label_box.left,
            label_box.bottom,
            label_box.width,
            label_box.height,
            2,
            stroke=0,
            fill=1,
        )
        pdf.setFillColor(colors.white)
        _text(
            pdf,
            label_box.left + 4,
            label_box.bottom + 3,
            label,
            size=8,
            bold=True,
        )


def _projection_scale_geometry(
    projection: SagittalProjection,
    box: ImageBox,
    markers: tuple[Mapping[str, Any], ...] = (),
) -> dict[str, Any]:
    """Place the vertical physical scale opposite the left marker labels."""

    scale_y = box.height / (projection.superior_mm - projection.inferior_mm)
    length_mm = 50.0 if scale_y * 50.0 <= box.height / 3 else 20.0
    length = scale_y * length_mm
    line_x = box.right - 13
    text = f"{int(length_mm)} mm"
    text_right = line_x - 5
    text_width = pdfmetrics.stringWidth(text, FONT_REGULAR, 8)
    label_boxes = [
        _anthropometry_marker_label_box(projection, box, marker)
        for marker in markers
        if marker["displayed"]
    ]

    def candidate(anchor: str, line_bottom: float) -> dict[str, Any]:
        bounds = {
            "left": min(line_x - 3, text_right - text_width),
            "right": line_x + 3,
            "bottom": line_bottom - 3,
            "top": line_bottom + length + 3,
        }
        overlap_area = 0.0
        collision_count = 0
        for label_box in label_boxes:
            overlap_width = max(
                0.0,
                min(bounds["right"], label_box.right)
                - max(bounds["left"], label_box.left),
            )
            overlap_height = max(
                0.0,
                min(bounds["top"], label_box.top)
                - max(bounds["bottom"], label_box.bottom),
            )
            if overlap_width > 0 and overlap_height > 0:
                collision_count += 1
                overlap_area += overlap_width * overlap_height
        return {
            "placement": "right_clear_of_anthropometry_labels",
            "vertical_anchor": anchor,
            "orientation": "vertical",
            "text_alignment": "inward_left",
            "collision_policy": "avoid_anthropometry_label_boxes",
            "collision_count": collision_count,
            "collision_overlap_area_pt2": overlap_area,
            "length_mm": length_mm,
            "line_x_pt": line_x,
            "line_bottom_pt": line_bottom,
            "line_top_pt": line_bottom + length,
            "text": text,
            "text_right_pt": text_right,
            "text_baseline_pt": line_bottom + length / 2 - 3,
            "bounds_pt": bounds,
        }

    options = [
        candidate("lower", box.bottom + 18),
        candidate("center", box.bottom + (box.height - length) / 2),
        candidate("upper", box.top - 18 - length),
    ]
    return min(
        options,
        key=lambda item: (
            int(item["collision_count"]),
            float(item["collision_overlap_area_pt2"]),
            options.index(item),
        ),
    )


def _draw_projection(
    pdf: canvas.Canvas,
    projection: SagittalProjection,
    panel: ImageBox,
    *,
    rows: list[dict[str, Any]],
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
    pdf.setFillColor(colors.white)
    _text(pdf, box.left + 4, box.top - 12, "S", size=8, bold=True)
    _text(pdf, box.left + 4, box.bottom + 4, "I", size=8, bold=True)
    _text(pdf, box.left + 18, box.bottom + 4, "A", size=8, bold=True)
    pdf.setFont(FONT_BOLD, 8)
    pdf.drawRightString(box.right - 4, box.bottom + 4, "P")
    _draw_anthropometry_spine_markers(
        pdf,
        projection,
        box,
        anthropometry_markers,
    )
    scale = _projection_scale_geometry(projection, box, anthropometry_markers)
    x_scale = float(scale["line_x_pt"])
    y_scale = float(scale["line_bottom_pt"])
    scale_top = float(scale["line_top_pt"])
    pdf.setStrokeColor(colors.white)
    pdf.setLineWidth(2)
    pdf.line(x_scale, y_scale, x_scale, scale_top)
    pdf.line(x_scale - 3, y_scale, x_scale + 3, y_scale)
    pdf.line(x_scale - 3, scale_top, x_scale + 3, scale_top)
    pdf.setFillColor(colors.white)
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawRightString(
        float(scale["text_right_pt"]),
        float(scale["text_baseline_pt"]),
        str(scale["text"]),
    )

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
            12.0,
            pdfmetrics.stringWidth(display_label, FONT_BOLD, 8) + 4.0,
        )
        label_x = float(
            np.clip(
                x_true - label_width / 2,
                box.left + 2,
                box.right - label_width - 2,
            )
        )
        pdf.saveState()
        pdf.setFillColor(colors.white)
        pdf.setFillAlpha(VERTEBRAL_LABEL_BACKGROUND_ALPHA)
        pdf.roundRect(label_x, y_label - 5, label_width, 10, 1.5, stroke=0, fill=1)
        pdf.setFillAlpha(1.0)
        pdf.setFillColor(INK)
        _text(pdf, label_x + 2, y_label - 2, display_label, size=8, bold=True)
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
    hu_filter_display = _table_hu_filter_display(settings.measurement_columns)
    title = "Measurements"
    detail = "Whole-level mean"
    pdf.setFillColor(colors.HexColor("#263746"))
    _text(
        pdf,
        panel.left,
        panel.top - 8,
        title,
        size=8,
        bold=True,
    )
    hu_filter_audit = _draw_hu_filter_caption(
        pdf,
        x=panel.left,
        y=panel.top - 20,
        display=hu_filter_display,
    )
    pdf.setFillColor(MUTED)
    _text(
        pdf,
        panel.left,
        panel.top - 31,
        detail,
        size=8,
    )
    level_width = 39.0
    metric_width = (panel.width - level_width) / len(settings.measurement_columns)
    grid_top = panel.top - 43
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
        _centered(
            pdf,
            x,
            header_y - 10.5,
            f"({_display_unit(definition['unit'])})",
            size=8,
        )

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
    footer_lines = (
        "* unavailable | V variant",
        "! incomplete or FOV-limited",
    )
    _text(pdf, panel.left, panel.bottom - 6, footer_lines[0], size=8)
    _text(pdf, panel.left, panel.bottom - 17, footer_lines[1], size=8)
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
        "display_units": [
            _display_unit(MEASUREMENT_DEFINITIONS[column]["unit"])
            for column in settings.measurement_columns
        ],
        "title": title,
        "detail_lines": [hu_filter_audit["caption"], detail],
        "hu_filter": hu_filter_audit,
        "footer_lines": list(footer_lines),
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
    _text(pdf, panel.left, y, ANTHROPOMETRY_ASTERISK_EXPLANATION, size=8)


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
        "asterisk_explanation": ANTHROPOMETRY_ASTERISK_EXPLANATION,
        "asterisk_semantics": [
            "incomplete_anatomical_range",
            "contour_reaches_field_of_view_edge",
        ],
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
        f"Axial filtered tissues | {view.vertebral_level}"
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
        "ct_window_caption_visible": False,
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
    body_composition = case.measurement_bundle.provenance.get(
        "body_composition",
        {},
    )
    compartments = (
        body_composition.get("compartments", {})
        if isinstance(body_composition, Mapping)
        else {}
    )
    compartment_model = (
        compartments.get("backend_id", "compartment model not recorded")
        if isinstance(compartments, Mapping)
        else "compartment model not recorded"
    )
    return _split_technical_lines(
        orientation_model,
        case.vertebral_result.backend_id,
        str(compartment_model),
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
        entry for entry in review_entries if entry.code not in ROUTINE_BOUNDARY_REVIEW_CODES
    )
    suppressed_codes = tuple(
        sorted(
            {entry.code for entry in review_entries if entry.code in ROUTINE_BOUNDARY_REVIEW_CODES}
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
        items.append(f"Vertebral table contains {missing} unavailable values.")
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
                body_top + 4,
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
        "inner_header_rule_visible": False,
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
            "sat_tissue_hu_m190_m30_area_cm2",
        ),
        (
            "avat",
            "aVAT",
            "avat_tissue_hu_m190_m30_area_cm2",
        ),
        (
            "tvat",
            "tVAT",
            "tvat_tissue_hu_m190_m30_area_cm2",
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


def _profile_plot_support(
    positions_mm: np.ndarray,
    *,
    inferior_mm: float,
    superior_mm: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Close sampled slice cells at the projection bounds for visual alignment."""

    finite = np.isfinite(positions_mm)
    ordered = np.sort(np.unique(positions_mm[finite]))
    positive_steps = np.diff(ordered)
    positive_steps = positive_steps[positive_steps > 0]
    half_step_mm = (
        float(np.median(positive_steps)) / 2.0 if positive_steps.size else 0.0
    )
    tolerance = max(1e-6, half_step_mm * 1e-6)
    support = (
        finite
        & (positions_mm >= inferior_mm - half_step_mm - tolerance)
        & (positions_mm <= superior_mm + half_step_mm + tolerance)
    )
    plotted_positions = positions_mm.copy()
    supported_indices = np.flatnonzero(support)
    if supported_indices.size:
        inferior_index = int(
            supported_indices[np.argmin(positions_mm[supported_indices])]
        )
        superior_index = int(
            supported_indices[np.argmax(positions_mm[supported_indices])]
        )
        if abs(float(positions_mm[inferior_index]) - inferior_mm) <= (
            half_step_mm + tolerance
        ):
            plotted_positions[inferior_index] = inferior_mm
        if abs(superior_mm - float(positions_mm[superior_index])) <= (
            half_step_mm + tolerance
        ):
            plotted_positions[superior_index] = superior_mm
    return plotted_positions, support, half_step_mm


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
    plot_layer_validities: list[np.ndarray] = []
    display_valid = (
        np.isfinite(positions)
        & (positions >= projection.inferior_mm)
        & (positions <= projection.superior_mm)
    )
    plot_positions, plot_support, sampling_half_step_mm = _profile_plot_support(
        positions,
        inferior_mm=projection.inferior_mm,
        superior_mm=projection.superior_mm,
    )
    for _name, _, column in layers:
        layer = pd.to_numeric(slices[column], errors="coerce").to_numpy(dtype=float)
        validity_column = column.replace("_cm2", "_valid")
        source_valid = (
            slices[validity_column]
            .fillna(False)
            .astype(bool)
            .to_numpy(copy=True)
            if validity_column in slices
            else np.isfinite(layer)
        )
        source_valid &= np.isfinite(layer) & (layer >= 0)
        layer_validities.append(display_valid & source_valid)
        plot_layer_validities.append(plot_support & source_valid)
        values.append(layer)
    safe_plot_values = [
        np.where(layer_valid, layer, 0.0)
        for layer, layer_valid in zip(values, plot_layer_validities, strict=True)
    ]
    any_layer_valid = np.any(np.vstack(plot_layer_validities), axis=0)
    stack = np.sum(np.vstack(safe_plot_values), axis=0)
    raw_max = float(np.nanmax(stack[any_layer_valid])) if np.any(any_layer_valid) else 0.0
    trunk = None
    trunk_valid = np.zeros(len(slices), dtype=bool)
    trunk_boundary_touching = np.zeros(len(slices), dtype=bool)
    trunk_plotted = np.zeros(len(slices), dtype=bool)
    trunk_plot_boundary_touching = np.zeros(len(slices), dtype=bool)
    trunk_plot_plotted = np.zeros(len(slices), dtype=bool)
    trunk_invalid_reasons: list[str] = []
    if "trunk_area_cm2" in slices:
        trunk = pd.to_numeric(slices["trunk_area_cm2"], errors="coerce").to_numpy(dtype=float)
        trunk_observed_source = np.isfinite(trunk) & (trunk >= 0)
        trunk_strict_source = trunk_observed_source.copy()
        if "trunk_area_valid" in slices:
            trunk_strict_source &= (
                slices["trunk_area_valid"].fillna(False).astype(bool).to_numpy()
            )
        trunk_valid = display_valid & trunk_strict_source
        trunk_boundary_source = np.zeros(len(slices), dtype=bool)
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
            trunk_boundary_source = (
                trunk_observed_source
                & ~trunk_strict_source
                & boundary_only
                & (trunk_reasons == "touching_image_boundary")
            )
            trunk_boundary_touching = display_valid & trunk_boundary_source
        trunk_plotted = trunk_valid | trunk_boundary_touching
        trunk_plot_boundary_touching = plot_support & trunk_boundary_source
        trunk_plot_plotted = plot_support & (
            trunk_strict_source | trunk_boundary_source
        )
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
        if np.any(trunk_plot_plotted):
            raw_max = max(raw_max, float(np.nanmax(trunk[trunk_plot_plotted])))
    if raw_max <= 0:
        raise ValueError("spine_profile_v2 has no valid positive filtered-tissue profile.")
    magnitude = 10.0 ** np.floor(np.log10(raw_max))
    normalized = raw_max / magnitude
    nice = next(value for value in (1.0, 2.0, 5.0, 10.0) if normalized <= value)
    x_max = float(nice * magnitude)
    pdf.setFillColor(colors.HexColor("#263746"))
    _text(
        pdf,
        panel.left,
        card_top - CARD_TITLE_OFFSET,
        "Tissue area",
        size=8,
        bold=True,
    )
    hu_filter_display = _hu_filter_display(HU_FILTER_RANGES)
    hu_filter_audit = _draw_hu_filter_caption(
        pdf,
        x=panel.left,
        y=card_top - CARD_SUBTITLE_OFFSET,
        display=hu_filter_display,
    )
    cumulative = np.zeros(len(slices), dtype=float)
    for (name, _label, _), layer_values, layer_valid in zip(
        layers,
        safe_plot_values,
        plot_layer_validities,
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
                projection.y_for_superior(
                    plot_positions[first],
                    shared_y_box.bottom,
                    shared_y_box.top,
                ),
            )
            for index in run:
                path.lineTo(
                    panel.left + upper[index] / x_max * panel.width,
                    projection.y_for_superior(
                        plot_positions[index],
                        shared_y_box.bottom,
                        shared_y_box.top,
                    ),
                )
            for index in run[::-1]:
                path.lineTo(
                    panel.left + lower[index] / x_max * panel.width,
                    projection.y_for_superior(
                        plot_positions[index],
                        shared_y_box.bottom,
                        shared_y_box.top,
                    ),
                )
            path.close()
            pdf.drawPath(path, stroke=1, fill=1)
        cumulative = upper
    if trunk is not None:
        pdf.setLineWidth(0.8)
        for style, run in _styled_line_runs(
            trunk_plot_plotted,
            trunk_plot_boundary_touching,
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
                    plot_positions[index],
                    shared_y_box.bottom,
                    shared_y_box.top,
                )
                (path.moveTo if offset == 0 else path.lineTo)(x, y)
            pdf.drawPath(path, stroke=1, fill=0)
        pdf.setDash()
    tick_step = x_max / 4.0
    tick_precision = 0 if abs(tick_step - round(tick_step)) < 1e-9 else 1
    pdf.setStrokeColor(colors.HexColor("#AEB8C0"))
    pdf.setLineWidth(0.6)
    pdf.line(
        panel.left,
        shared_y_box.bottom,
        panel.right,
        shared_y_box.bottom,
    )
    for value in np.linspace(0.0, x_max, 5):
        x = panel.left + value / x_max * panel.width
        pdf.line(
            x,
            shared_y_box.bottom,
            x,
            shared_y_box.bottom - 3,
        )
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
        "title": "Tissue area",
        "detail_lines": [hu_filter_audit["caption"]],
        "layers": [name for name, _, _ in layers],
        "legend_labels": [label for _, label, _ in layers],
        "legend_rows": legend_rows,
        "legend_style": PLOT_LEGEND_STYLE,
        "axis_style": PLOT_AXIS_STYLE,
        "frame_visible": False,
        "vertical_gridlines_visible": False,
        "x_axis_baseline_visible": True,
        "x_axis_tick_marks_visible": True,
        "hu_definition_caption": hu_filter_audit["caption"],
        "filter_icon_visible": True,
        "filter_icon_semantics": "hu_filter_state",
        "hu_filter": hu_filter_audit,
        "x_axis_display_label": "Tissue area (cm2)",
        "unit_rendering": "plain_ascii",
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
        "physical_y_axis_box_pt": {
            "bottom": shared_y_box.bottom,
            "top": shared_y_box.top,
            "height": shared_y_box.height,
        },
        "physical_y_axis_alignment": "shared_exact_projection_image_box",
        "sampling_cell_edge_closure": {
            "method": "half_sampling_interval_clipped_to_projection_bounds",
            "half_step_mm": sampling_half_step_mm,
            "inferior_bound_reached": bool(
                np.any(
                    np.vstack(plot_layer_validities)
                    & np.isclose(
                        plot_positions,
                        projection.inferior_mm,
                        atol=1e-6,
                    )
                )
            ),
            "superior_bound_reached": bool(
                np.any(
                    np.vstack(plot_layer_validities)
                    & np.isclose(
                        plot_positions,
                        projection.superior_mm,
                        atol=1e-6,
                    )
                )
            ),
        },
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
        raise ValueError(
            "spine_profile_v2 requires canonical native-compartment HU distributions."
        )

    pdf.setFillColor(INK)
    _text(
        pdf,
        panel.left,
        card_top - CARD_TITLE_OFFSET,
        "HU distribution",
        size=8,
        bold=True,
    )
    hu_filter_audit = _draw_hu_filter_caption(
        pdf,
        x=panel.left,
        y=card_top - CARD_SUBTITLE_OFFSET,
        display=_hu_filter_display({}),
    )
    pdf.setFillColor(MUTED)
    _text(
        pdf,
        panel.left,
        card_top - CARD_SUBTITLE_OFFSET - 11.0,
        "median [IQR]",
        size=8,
    )

    def x_for_hu(value: float) -> float:
        fraction = (float(value) - HU_DISTRIBUTION_MIN_HU) / (
            HU_DISTRIBUTION_MAX_HU - HU_DISTRIBUTION_MIN_HU
        )
        return panel.left + float(np.clip(fraction, 0.0, 1.0)) * panel.width

    axis_height = 24.0
    chart_bottom = panel.bottom + axis_height
    row_height = (panel.top - chart_bottom) / len(HU_DISTRIBUTION_COMPARTMENTS)
    statistics: dict[str, dict[str, Any]] = {}
    total_voxel_counts: dict[str, int] = {}
    contributing_slice_counts: dict[str, int] = {}
    clipped_voxel_counts: dict[str, int] = {}
    clipped_tail_present = False

    for compartment_index, (compartment_key, display_label, _definition) in enumerate(
        HU_DISTRIBUTION_COMPARTMENTS
    ):
        group = table.loc[
            table["compartment_key"].eq(compartment_key)
        ].sort_values("bin_index")
        if group.empty:
            raise ValueError(f"Missing canonical HU distribution for {display_label}.")
        first = group.iloc[0]
        row_top = panel.top - compartment_index * row_height
        row_bottom = row_top - row_height
        label_y = row_top - 9.0
        plot_top = row_top - 18.0
        plot_bottom = row_bottom + 5.0
        if plot_top <= plot_bottom:
            raise ValueError(
                "Compartment HU distribution rows cannot fit the released layout."
            )

        red, green, blue = tissue_color(compartment_key)
        compartment_fill = colors.Color(
            red / 255,
            green / 255,
            blue / 255,
            alpha=0.62,
        )
        _draw_tissue_legend_item(
            pdf,
            x=panel.left,
            y=label_y - 1.0,
            tissue_key=compartment_key,
            label=display_label,
        )

        valid = bool(first["distribution_valid"])
        total_voxels = int(first["total_voxel_count"])
        contributing_slices = int(first["contributing_slice_count"])
        clipped_voxels = int(first["below_histogram_voxel_count"]) + int(
            first["above_histogram_voxel_count"]
        )
        clipped_fraction = clipped_voxels / total_voxels if total_voxels else 0.0
        clipped_tail_present |= clipped_voxels > 0
        q1_hu = float(first["q1_hu"]) if pd.notna(first["q1_hu"]) else None
        median_hu = float(first["median_hu"]) if pd.notna(first["median_hu"]) else None
        q3_hu = float(first["q3_hu"]) if pd.notna(first["q3_hu"]) else None
        if not valid or median_hu is None or q1_hu is None or q3_hu is None:
            summary_text = "NA"
        else:
            summary_text = f"{median_hu:.0f} [{q1_hu:.0f},{q3_hu:.0f}]"
        pdf.setFillColor(MUTED)
        pdf.setFont(FONT_REGULAR, 8)
        pdf.drawRightString(panel.right, label_y, summary_text)

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
                pdf.setFillColor(compartment_fill)
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
        total_voxel_counts[compartment_key] = total_voxels
        contributing_slice_counts[compartment_key] = contributing_slices
        clipped_voxel_counts[compartment_key] = clipped_voxels
        reason_value = first["distribution_reason"]
        statistics[compartment_key] = {
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
        label="HU",
    )
    first_row = table.iloc[0]
    return {
        "compartments": [key for key, _label, _definition in HU_DISTRIBUTION_COMPARTMENTS],
        "display_labels": [label for _key, label, _definition in HU_DISTRIBUTION_COMPARTMENTS],
        "legend_style": PLOT_LEGEND_STYLE,
        "axis_style": PLOT_AXIS_STYLE,
        "x_axis_label": "HU",
        "title": "HU distribution",
        "detail_lines": [hu_filter_audit["caption"], "median [IQR]"],
        "hu_filter": hu_filter_audit,
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
        "normalization": "within_compartment_peak",
        "normalization_caption": None,
        "normalization_caption_visible": False,
        "normalization_explanation": (
            "Each compartment histogram is divided by its own maximum bin height."
        ),
        "cross_compartment_magnitude_comparison": False,
        "summary_statistics": "exact_raw_voxel_mean_sd_median_and_iqr",
        "iqr_display": "numeric_summary_only",
        "iqr_band_displayed": False,
        "median_reference_line_displayed": True,
        "reference_guides_hu": [],
        "background_band_displayed": False,
        "source_semantics": "model_native_compartment",
        "quantile_method": str(first_row["quantile_method"]),
        "statistics": statistics,
        "total_voxel_counts": total_voxel_counts,
        "contributing_slice_counts": contributing_slice_counts,
        "clipped_voxel_counts": clipped_voxel_counts,
        "clipped_tail_present": clipped_tail_present,
        "clipped_tail_marker_displayed": False,
        "clipped_tail_annotation_policy": "audit_only",
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
                "background": "translucent_white_neutral",
                "background_alpha": VERTEBRAL_LABEL_BACKGROUND_ALPHA,
                "minimum_vertical_gap_pt": VERTEBRAL_LABEL_MIN_GAP_PT,
            },
            "ct_window_hu": list(settings.ct_window),
            "ct_window_caption_visible": False,
            "projection_method_caption_visible": False,
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
        left_panel = ImageBox(
            sagittal_card.left + CARD_PADDING,
            sagittal_card.bottom + 24,
            sagittal_card.width - 2 * CARD_PADDING,
            sagittal_card.height - 75,
        )
        image_box = _draw_projection(
            pdf,
            projection,
            left_panel,
            rows=rows,
            anthropometry_markers=anthropometry_markers,
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
            "compartment_hu_distribution",
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
            rows=rows,
            anthropometry_markers=anthropometry_markers,
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
            image_box.bottom,
            profile_card.width - 2 * CARD_PADDING,
            image_box.height,
        )
        hu_panel = ImageBox(
            hu_card.left + CARD_PADDING,
            image_box.bottom,
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
    scale_audit = _projection_scale_geometry(
        projection,
        image_box,
        anthropometry_markers,
    )
    audit["sagittal_projection"]["image_box_pt"] = {
        "bottom": image_box.bottom,
        "top": image_box.top,
        "height": image_box.height,
    }
    audit["sagittal_projection"]["profile_alignment_policy"] = (
        "shared_exact_physical_y_box"
    )
    audit["sagittal_projection"]["physical_scale"] = scale_audit
    marker_audit = []
    for marker in anthropometry_markers:
        rendered = dict(marker)
        if marker["displayed"]:
            label_box = _anthropometry_marker_label_box(
                projection,
                image_box,
                marker,
            )
            rendered["label_box_pt"] = {
                "left": label_box.left,
                "right": label_box.right,
                "bottom": label_box.bottom,
                "top": label_box.top,
            }
        marker_audit.append(rendered)
    audit["sagittal_projection"]["anthropometry_markers"] = marker_audit
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
                "Tissue area",
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
                (
                    f"{definition['short_label']} "
                    f"({_display_unit(definition['unit'])}): "
                    f"{settings.missing_value_symbol}*"
                ),
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


def _format_cover_duration(value: Any) -> str:
    if value is None:
        return "Not available"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if not np.isfinite(seconds) or seconds < 0:
        return "Not available"
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} min"
    return f"{minutes / 60.0:.1f} h"


def _format_cover_datetime(value: Any) -> str:
    if value is None:
        return "Not recorded"
    return _analysis_date(str(value))


def _format_cover_case_count(count: int, denominator: int) -> str:
    percentage = 100.0 * count / denominator if denominator else 0.0
    rounded = round(percentage)
    percentage_text = f"{rounded:.0f}%" if np.isclose(percentage, rounded) else f"{percentage:.1f}%"
    return f"{count} ({percentage_text})"


def _humanize_cover_code(value: Any) -> str:
    text = re.sub(r"[_-]+", " ", public_text(value, maximum=80)).strip()
    return text[:1].upper() + text[1:] if text else "Unspecified"


def _cover_quality_issue_label(item: Mapping[str, Any]) -> str:
    code = str(item["code"])
    return RUN_COVER_ISSUE_LABELS.get(code) or _humanize_cover_code(code)


def _cover_execution_issue_label(item: Mapping[str, Any]) -> str:
    stage = _humanize_cover_code(item["stage"])
    code = _humanize_cover_code(item["code"])
    return f"Execution - {stage}: {code}"


def _cover_value_lines(label: str, value: str) -> tuple[str, ...]:
    """Return compact reader-facing lines for one cover value."""

    text = public_text(value, maximum=180)
    if label != "Tissue area filters" or text == "HU filter inactive":
        return (text,)
    without_unit = text.removesuffix(" HU")
    filters = tuple(
        f"{part.strip()} HU"
        for part in without_unit.split(" | ")
        if part.strip()
    )
    return filters or (text,)


def _draw_cover_section(
    pdf: canvas.Canvas,
    *,
    box: ImageBox,
    title: str,
    rows: Sequence[tuple[str, str]],
    maximum_rows: int = 8,
    inline_values: bool = False,
) -> int:
    """Draw one compact cover-page key/value section."""

    pdf.setFillColor(INK)
    _text(
        pdf,
        box.left,
        box.top - 13,
        title,
        size=8,
        bold=True,
    )
    pdf.setStrokeColor(BORDER)
    pdf.setLineWidth(CARD_BORDER_WIDTH)
    pdf.line(box.left, box.top - 20, box.right, box.top - 20)
    displayed = list(rows[:maximum_rows])
    y = box.top - 38
    if inline_values:
        row_height = (
            min(18.0, (box.height - 48.0) / (len(displayed) - 1))
            if len(displayed) > 1
            else 18.0
        )
        label_width = max(
            (
                pdfmetrics.stringWidth(public_text(label), FONT_REGULAR, 8)
                for label, _value in displayed
            ),
            default=0.0,
        )
        value_x = box.left + min(label_width + 8.0, box.width * 0.48)
        for label, value in displayed:
            pdf.setFillColor(MUTED)
            _text(pdf, box.left, y, label, size=8)
            pdf.setFillColor(INK)
            _text(
                pdf,
                value_x,
                y,
                _truncate_to_width(value, box.right - value_x),
                size=8,
                bold=True,
            )
            y -= row_height
        return len(displayed)
    row_height = (
        min(24.0, (box.height - 57.0) / (len(displayed) - 1))
        if len(displayed) > 1
        else 24.0
    )
    for label, value in displayed:
        value_lines = _cover_value_lines(label, value)
        pdf.setFillColor(MUTED)
        _text(pdf, box.left, y, label, size=8)
        pdf.setFillColor(INK)
        for line_index, line in enumerate(value_lines):
            _text(
                pdf,
                box.left,
                y - 11 - line_index * 11,
                _truncate_to_width(line, box.width),
                size=8,
                bold=True,
            )
        y -= row_height + (len(value_lines) - 1) * 11
    return len(displayed)


def _ordered_cover_pipeline_rows(
    summary: Mapping[str, Any],
) -> list[tuple[str, str]]:
    """Return reader-facing pipeline rows in canonical execution order."""

    pipeline = dict(summary["pipeline"])
    rows = [
        (
            str(item["role"]),
            str(item["identifier"]),
        )
        for item in list(summary["models"])
        if str(item["role"]).strip().lower() not in RUN_COVER_HIDDEN_MODEL_ROLES
    ]
    rows.extend(
        (str(item["label"]), str(item["value"]))
        for item in summary["configuration_rows"]
    )
    ordered: list[tuple[str, str]] = [
        ("Pipeline", f"{pipeline['name']} {pipeline['version']}"),
    ]
    remaining = list(rows)
    for expected_label in RUN_COVER_PIPELINE_SEQUENCE:
        matching = [
            row
            for row in remaining
            if row[0].strip().casefold() == expected_label.casefold()
        ]
        ordered.extend(matching)
        remaining = [row for row in remaining if row not in matching]
    ordered.extend(remaining)
    return ordered


def render_run_cover_page(
    destination: Path,
    *,
    summary: Mapping[str, Any],
    page_count: int,
) -> dict[str, Any]:
    """Render the always-present one-page run snapshot for a combined PDF."""

    if page_count < 2:
        raise ValueError("A combined run report requires a cover and at least one case page.")
    run_id = str(summary["run_id"])
    pdf = _new_canvas(destination, case_id=run_id)
    pdf.setTitle(f"BodyComposition run summary {run_id}")
    pdf.bookmarkPage("run-summary")
    pdf.setFillColor(PAGE_BACKGROUND)
    pdf.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)

    left = 24.0
    right = PAGE_WIDTH - 24.0
    pdf.setFillColor(INK)
    _text(
        pdf,
        left,
        PAGE_HEIGHT - 38,
        "BodyComposition run summary",
        size=14,
        bold=True,
    )
    document_state = str(summary["document_state"]).replace("_", " ").upper()
    remaining = int(summary["remaining_case_count"])
    state_text = document_state
    if remaining:
        state_text = f"{state_text} | {remaining} remaining"
    pdf.setFillColor(MUTED)
    _text(
        pdf,
        left,
        PAGE_HEIGHT - 56,
        _truncate_to_width(
            f"Run {run_id} | updated {_format_cover_datetime(summary['updated_at'])} | "
            f"{state_text}",
            PAGE_WIDTH - 190,
        ),
        size=8,
    )
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawRightString(right, PAGE_HEIGHT - 38, f"Page 1 of {page_count}")
    pdf.setStrokeColor(BORDER)
    pdf.setLineWidth(CARD_BORDER_WIDTH)
    pdf.line(left, PAGE_HEIGHT - 72, right, PAGE_HEIGHT - 72)

    case_counts = dict(summary["case_counts"])
    failed = int(case_counts.get("failed", 0))
    cancelled = int(case_counts.get("cancelled", 0))
    metrics = (
        ("Planned cases", str(summary["planned_case_count"])),
        ("Included cases", str(summary["included_case_count"])),
        (
            "Successful",
            str(
                int(case_counts.get("succeeded", 0))
                + int(case_counts.get("skipped_identical", 0))
            ),
        ),
        ("Review cases", str(summary["manual_review_case_count"])),
        ("Failed / cancelled", f"{failed} / {cancelled}"),
        ("Mean case runtime", _format_cover_duration(summary.get("mean_runtime_seconds"))),
    )
    metric_top = PAGE_HEIGHT - 88
    metric_bottom = PAGE_HEIGHT - 150
    metric_width = (right - left) / len(metrics)
    for index, (label, value) in enumerate(metrics):
        x = left + index * metric_width
        if index:
            pdf.setStrokeColor(BORDER)
            pdf.line(x, metric_bottom, x, metric_top)
        pdf.setFillColor(MUTED)
        _text(pdf, x + 10, metric_top - 14, label, size=8)
        pdf.setFillColor(INK)
        _text(pdf, x + 10, metric_top - 40, value, size=14, bold=True)
    pdf.setStrokeColor(BORDER)
    pdf.line(left, metric_bottom, right, metric_bottom)

    model_rows = _ordered_cover_pipeline_rows(summary)
    runtime = dict(summary["runtime"])
    run_complete = str(summary["document_state"]) != "in_progress"
    runtime_rows = [
        ("Analysis started", _format_cover_datetime(summary.get("started_at"))),
        (
            "Analysis finished",
            (
                _format_cover_datetime(summary.get("ended_at"))
                if run_complete
                else "In progress"
            ),
        ),
        (
            "Run duration",
            (
                _format_cover_duration(summary.get("run_duration_seconds"))
                if run_complete
                else "In progress"
            ),
        ),
        ("Device", str(runtime.get("device") or "Not recorded")),
        ("Hardware", str(runtime.get("hardware") or "Not recorded")),
        ("CUDA", str(runtime.get("cuda") or "Not recorded")),
        ("Torch", str(runtime.get("torch") or "Not recorded")),
        ("Python", str(runtime.get("python") or "Not recorded")),
    ]
    paths = dict(summary["paths"])
    input_directories = paths.get("input_directories") or ["Not recorded"]
    if isinstance(input_directories, str):
        input_directories = [input_directories]
    input_directories = [str(value) for value in input_directories]
    path_rows = [
        (
            "Input directory" if len(input_directories) == 1 else "Input directories",
            " | ".join(input_directories),
        ),
        (
            "Output directory",
            str(paths.get("output_directory") or paths["run_folder"]),
        ),
    ]
    qc_counts = dict(summary["qc_counts"])
    completed = int(case_counts.get("succeeded", 0)) + int(
        case_counts.get("skipped_identical", 0)
    )
    progress_rows = (
        (
            "Execution",
            f"{completed} complete | {failed} failed | {cancelled} cancelled",
        ),
        (
            "QC",
            (
                f"{int(qc_counts.get('pass', 0))} pass | "
                f"{int(qc_counts.get('review', 0))} review | "
                f"{int(qc_counts.get('fail', 0))} fail"
            ),
        ),
    )
    main_top = PAGE_HEIGHT - 170
    main_bottom = 48.0
    column_gap = 20.0
    usable_width = right - left - 2 * column_gap
    pipeline_width = usable_width * 0.38
    compact_width = usable_width * 0.28
    issue_width = usable_width - pipeline_width - compact_width
    pipeline_box = ImageBox(
        left,
        main_bottom,
        pipeline_width,
        main_top - main_bottom,
    )
    compact_left = pipeline_box.right + column_gap
    issue_left = compact_left + compact_width + column_gap
    issue_box = ImageBox(
        issue_left,
        main_bottom,
        issue_width,
        main_top - main_bottom,
    )
    for divider_x in (
        pipeline_box.right + column_gap / 2,
        compact_left + compact_width + column_gap / 2,
    ):
        pdf.setStrokeColor(BORDER)
        pdf.line(divider_x, main_bottom, divider_x, main_top)

    compact_gap = 15.0
    runtime_height = 168.0
    directory_height = 78.0
    runtime_box = ImageBox(
        compact_left,
        main_top - runtime_height,
        compact_width,
        runtime_height,
    )
    directory_top = runtime_box.bottom - compact_gap
    directory_box = ImageBox(
        compact_left,
        directory_top - directory_height,
        compact_width,
        directory_height,
    )
    progress_top = directory_box.bottom - compact_gap
    progress_box = ImageBox(
        compact_left,
        main_bottom,
        compact_width,
        progress_top - main_bottom,
    )
    section_counts = {
        "pipeline_and_models": _draw_cover_section(
            pdf,
            box=pipeline_box,
            title="Pipeline, models and relevant configuration",
            rows=model_rows,
            maximum_rows=RUN_COVER_MAX_PIPELINE_ROWS,
            inline_values=True,
        ),
        "runtime_context": _draw_cover_section(
            pdf,
            box=runtime_box,
            title="Runtime context",
            rows=runtime_rows,
            inline_values=True,
        ),
        "run_directories": _draw_cover_section(
            pdf,
            box=directory_box,
            title="Run directories",
            rows=path_rows,
        ),
        "case_progress": _draw_cover_section(
            pdf,
            box=progress_box,
            title="Case progress",
            rows=progress_rows,
        ),
    }

    _text(pdf, issue_box.left, issue_box.top - 12, "Issues", size=8, bold=True)
    pdf.setStrokeColor(BORDER)
    pdf.line(issue_box.left, issue_box.top - 19, issue_box.right, issue_box.top - 19)
    technical_errors = list(summary["technical_errors"])
    quality_issues = sorted(
        list(summary["quality_issues"]),
        key=lambda item: (
            -int(item["count"]),
            _cover_quality_issue_label(item).casefold(),
        ),
    )
    issue_rows: list[dict[str, Any]] = []
    if technical_errors:
        issue_rows.extend(
            {
                "label": _cover_execution_issue_label(item),
                "count": int(item["count"]),
                "source": "execution",
            }
            for item in technical_errors
        )
    else:
        issue_rows.append(
            {
                "label": "Execution failures",
                "count": 0,
                "source": "execution_zero",
            }
        )
    if quality_issues:
        issue_rows.extend(
            {
                "label": _cover_quality_issue_label(item),
                "count": int(item["count"]),
                "source": "quality_control",
                "domain": str(item["domain"]),
                "code": str(item["code"]),
            }
            for item in quality_issues
        )
    else:
        issue_rows.append(
            {
                "label": "Case-level quality-control findings",
                "count": 0,
                "source": "quality_control_zero",
            }
        )

    issue_denominator = int(summary["included_case_count"])
    issue_count_header = f"Cases (N={issue_denominator})"
    header_y = issue_box.top - 36
    pdf.setFillColor(MUTED)
    _text(pdf, issue_box.left, header_y, "Finding", size=8, bold=True)
    pdf.setFont(FONT_BOLD, 8)
    pdf.drawRightString(issue_box.right, header_y, issue_count_header)
    pdf.setStrokeColor(BORDER)
    pdf.line(issue_box.left, header_y - 7, issue_box.right, header_y - 7)

    row_height = 18.0
    row_top = header_y - 24.0
    bottom_note_height = 22.0
    maximum_issue_rows = max(
        1,
        int((row_top - (issue_box.bottom + bottom_note_height)) // row_height) + 1,
    )
    displayed_issue_rows = issue_rows[:maximum_issue_rows]
    count_width = 58.0
    for index, issue in enumerate(displayed_issue_rows):
        y = row_top - index * row_height
        pdf.setFillColor(INK)
        _text(
            pdf,
            issue_box.left,
            y,
            _truncate_to_width(
                str(issue["label"]),
                issue_box.width - count_width - 8.0,
            ),
            size=8,
        )
        pdf.setFont(FONT_BOLD, 8)
        pdf.drawRightString(
            issue_box.right,
            y,
            _format_cover_case_count(
                int(issue["count"]),
                issue_denominator,
            ),
        )
    hidden = len(issue_rows) - len(displayed_issue_rows)
    pdf.setFillColor(MUTED)
    issue_note = "Percentages use included cases; findings may overlap."
    if hidden > 0:
        issue_note = f"+ {hidden} additional issue types in manifest."
    _text(
        pdf,
        issue_box.left,
        issue_box.bottom + 5,
        _truncate_to_width(issue_note, issue_box.width),
        size=8,
    )

    pdf.setFillColor(MUTED)
    _text(
        pdf,
        left,
        23,
        (
            "This PDF is updated after each finished case. "
            "Canonical case manifests and machine-readable artifacts remain authoritative."
        ),
        size=8,
    )
    pdf.showPage()
    pdf.save()
    return {
        "revision": RUN_COVER_REVISION,
        "page_number": 1,
        "outer_border_visible": False,
        "background": "white",
        "metric_count": len(metrics),
        "section_rows_displayed": section_counts,
        "technical_issue_type_count": len(technical_errors),
        "technical_issue_row_capacity": maximum_issue_rows,
        "technical_issue_rows_displayed": sum(
            item["source"] == "execution" for item in displayed_issue_rows
        ),
        "technical_issue_rows_truncated": any(
            item["source"] == "execution" for item in issue_rows[maximum_issue_rows:]
        ),
        "quality_issue_type_count": len(quality_issues),
        "quality_issue_rows_displayed": sum(
            item["source"] == "quality_control" for item in displayed_issue_rows
        ),
        "quality_issue_rows_truncated": any(
            item["source"] == "quality_control" for item in issue_rows[maximum_issue_rows:]
        ),
        "issue_row_capacity": maximum_issue_rows,
        "issue_rows_displayed": len(displayed_issue_rows),
        "issue_rows_truncated": hidden > 0,
        "issue_table_columns": ["finding", "cases_percent"],
        "issue_count_header": issue_count_header,
        "issue_percentage_denominator": "included_case_count",
        "issue_denominator_case_count": issue_denominator,
        "issue_counts_may_overlap": True,
        "suppressed_quality_issue_codes": sorted(ROUTINE_BOUNDARY_REVIEW_CODES),
        "hidden_model_roles": [
            str(item["role"])
            for item in summary["models"]
            if str(item["role"]).strip().lower() in RUN_COVER_HIDDEN_MODEL_ROLES
        ],
        "hidden_runtime_fields": ["strategy"],
        "runtime_context_style": "inline_key_value",
        "runtime_period_complete": run_complete,
        "pipeline_row_order": [label for label, _ in model_rows],
        "pipeline_row_order_policy": "canonical_execution_order",
        "configuration_row_count": len(summary["configuration_rows"]),
        "configuration_display_scope": "result_and_reproducibility_relevant",
        "configuration_style": "inline_key_value",
        "configuration_exclusions": [
            "display-only reporting settings",
            "execution throughput settings",
            "output and persistence settings",
            "routine QC-only thresholds",
            "opaque provenance identifiers",
            "non-reader-facing helper backends",
        ],
        "body_column_layout": [
            "pipeline_models_configuration",
            "runtime_directories_progress",
            "issues",
        ],
        "issues_full_height_column": True,
    }


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
