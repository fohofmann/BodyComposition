"""Headless one-page PDF renderers for both fixed case-report layouts."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
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
    tissue_color,
    vertebral_color,
)

PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)
FONT_REGULAR = "BCVera"
FONT_BOLD = "BCVera-Bold"
MIN_FONT_SIZE = 8.0
LAYOUT_REVISION = "scientific_onepager_v16"
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
HU_SCALE_MIN = -190.0
HU_SCALE_MAX = 150.0
HU_PALETTE_RGB = (
    (0x00, 0x20, 0x4C),
    (0x57, 0x5D, 0x6D),
    (0xA5, 0x9C, 0x74),
    (0xFD, 0xE7, 0x37),
)
ROUTINE_BOUNDARY_NOTE_CODES = frozenset(
    {
        "vertebra_touches_cranial_fov",
        "vertebra_touches_caudal_fov",
    }
)
NOTE_DOMAIN_LABELS = {
    "orientation": "Orientation",
    "vertebral": "Spine segmentation",
    "measurement": "Measurements",
    "pipeline": "Processing",
    "reporting": "Report generation",
}
REVIEW_NOTE_LABELS = {
    "disconnected_vertebral_body": "Disconnected vertebral body segmentation.",
    "vertebral_body_extent_invalid": "Incomplete or truncated vertebral body extent.",
    "slice_measurement_invalid": "One or more slice measurements are invalid.",
    "trunk_contour_touches_fov": "Trunk contour reaches the image boundary.",
    "pelvic_maximum_unavailable": "Pelvic maximum could not be measured.",
    "minimum_waist_unavailable": "Minimum waist could not be measured.",
    "midwaist_measurement_unavailable": "Mid-waist could not be measured.",
    "severe_misorientation_repaired": (
        "A substantial orientation mismatch was corrected before analysis."
    ),
    "orientation_mismatch_uncertain": "The scan orientation remains uncertain.",
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
class _NotesSection:
    heading: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _NotesSummary:
    sections: tuple[_NotesSection, ...]
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


def _metadata_text_widths(
    values: tuple[str, ...],
    available_width: float,
) -> tuple[float, ...]:
    natural = tuple(pdfmetrics.stringWidth(value, FONT_REGULAR, 8) for value in values)
    if sum(natural) <= available_width:
        return natural
    minimum = tuple(min(width, 42.0) for width in natural)
    minimum_total = sum(minimum)
    if minimum_total > available_width:
        raise ValueError("Technical metadata cannot fit the released header.")
    flexible = tuple(width - floor for width, floor in zip(natural, minimum, strict=True))
    flexible_total = sum(flexible)
    scale = (available_width - minimum_total) / flexible_total if flexible_total else 0.0
    return tuple(
        floor + extension * scale
        for floor, extension in zip(minimum, flexible, strict=True)
    )


def _draw_metadata_row(
    pdf: canvas.Canvas,
    row: tuple[tuple[str, str], ...],
    x: float,
    y: float,
    width: float,
) -> bool:
    icon_size = 8.0
    icon_text_gap = 3.5
    segment_gap = 14.0
    fixed_width = len(row) * (icon_size + icon_text_gap) + (len(row) - 1) * segment_gap
    values = tuple(public_text(value, maximum=180) for _kind, value in row)
    text_widths = _metadata_text_widths(values, width - fixed_width)
    cursor = x
    truncated = False
    for index, ((kind, value), text_width) in enumerate(zip(row, text_widths, strict=True)):
        if index:
            divider_x = cursor + segment_gap / 2.0
            pdf.setStrokeColor(BORDER)
            pdf.setLineWidth(0.5)
            pdf.line(divider_x, y - 1.0, divider_x, y + 7.0)
            cursor += segment_gap
        _draw_metadata_icon(pdf, kind, cursor, y - 1.0, size=icon_size)
        cursor += icon_size + icon_text_gap
        displayed = _truncate_to_width(value, text_width, size=8)
        truncated |= displayed != value
        pdf.setFillColor(MUTED)
        _text(pdf, cursor, y, displayed, size=8)
        cursor += pdfmetrics.stringWidth(displayed, FONT_REGULAR, 8)
    return truncated


def _header(
    pdf: canvas.Canvas,
    case: CaseReportInput,
    report_id: str,
) -> tuple[float, dict[str, Any]]:
    box = ImageBox(24, PAGE_HEIGHT - 62, PAGE_WIDTH - 48, 52)
    _draw_card(pdf, box)

    right_text = f"case page 1 of 1 | report {report_id[:12]}"
    right_width = pdfmetrics.stringWidth(right_text, FONT_REGULAR, 8)
    left_text = (
        f"CASE {public_text(case.case_id, maximum=64)} | "
        f"analysis {case.measurement_bundle.identity.analysis_id[:12]}"
    )
    pdf.setFillColor(INK)
    _text(
        pdf,
        box.left + CARD_PADDING,
        box.top - CARD_TITLE_OFFSET,
        _truncate_to_width(
            left_text,
            box.width - 2 * CARD_PADDING - right_width - 18,
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
    pdf.setStrokeColor(BORDER)
    pdf.setLineWidth(0.6)
    pdf.line(
        box.left + CARD_PADDING,
        box.top - 25,
        box.right - CARD_PADDING,
        box.top - 25,
    )

    metadata_rows = _technical_metadata_rows(case)
    pdf.setFillColor(MUTED)
    metadata_truncated = False
    for index, row in enumerate(metadata_rows):
        metadata_truncated |= _draw_metadata_row(
            pdf,
            row,
            box.left + CARD_PADDING,
            box.top - 34 - index * 11,
            box.width - 2 * CARD_PADDING,
        )
    return box.bottom - 8, {
        "boxed": True,
        "box_height_pt": box.height,
        "box_top_pt": box.top,
        "content_padding_pt": CARD_PADDING,
        "technical_metadata_location": "header",
        "technical_metadata_rows": len(metadata_rows),
        "technical_metadata_icons": [
            kind for row in metadata_rows for kind, _value in row
        ],
        "technical_metadata_icon_style": "monochrome_vector",
        "technical_metadata_truncated": metadata_truncated,
    }


def _draw_card(pdf: canvas.Canvas, box: ImageBox) -> None:
    pdf.setFillColor(CARD_BACKGROUND)
    pdf.setStrokeColor(BORDER)
    pdf.setLineWidth(CARD_BORDER_WIDTH)
    pdf.rect(box.left, box.bottom, box.width, box.height, stroke=1, fill=1)


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


def _hu_color(value: float) -> colors.Color:
    normalized = float(
        np.clip(
            (float(value) - HU_SCALE_MIN) / (HU_SCALE_MAX - HU_SCALE_MIN),
            0.0,
            1.0,
        )
    )
    scaled = normalized * (len(HU_PALETTE_RGB) - 1)
    lower = min(int(np.floor(scaled)), len(HU_PALETTE_RGB) - 2)
    fraction = scaled - lower
    left = np.asarray(HU_PALETTE_RGB[lower], dtype=float)
    right = np.asarray(HU_PALETTE_RGB[lower + 1], dtype=float)
    red, green, blue = (left + fraction * (right - left)) / 255.0
    return colors.Color(float(red), float(green), float(blue))


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


def _draw_projection(
    pdf: canvas.Canvas,
    projection: SagittalProjection,
    panel: ImageBox,
    *,
    ct_window: tuple[float, float],
    rows: list[dict[str, Any]],
    show_title: bool = True,
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
    scale_y = box.height / (projection.superior_mm - projection.inferior_mm)
    scale_length_mm = 50.0 if scale_y * 50.0 <= box.height / 3 else 20.0
    scale_length = scale_y * scale_length_mm
    # Vertebral labels are preferentially placed to the right of their
    # centroids, so keep the independent physical scale on the opposite edge.
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
    )
    for centroid, y_true, y_label in zip(centroids, desired_y, label_y, strict=True):
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
        red, green, blue = vertebral_color(anatomical)
        pdf.setStrokeColor(colors.Color(red / 255, green / 255, blue / 255))
        pdf.setLineWidth(0.8)
        pdf.line(x_true, y_true, min(box.right - 22, x_true + 12), y_label)
        pdf.setFillColor(colors.white)
        label_width = max(24, pdfmetrics.stringWidth(anatomical, FONT_BOLD, 8) + 7)
        label_x = min(box.right - label_width - 2, max(box.left + 2, x_true + 12))
        pdf.roundRect(label_x, y_label - 5, label_width, 10, 2, stroke=0, fill=1)
        pdf.setFillColor(colors.HexColor("#15232E"))
        _text(pdf, label_x + 3, y_label - 2, anatomical, size=8, bold=True)
    return box


def _format_metric(metric: dict[str, Any], settings: ReportingSettings) -> str:
    if not metric["valid"] or metric["value"] is None:
        return settings.missing_value_symbol + "*"
    return f"{float(metric['value']):.{settings.numeric_precision}f}"


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
        "Whole-territory mean | three physical bins",
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
            f"{row['vertebral_level']} {marker}".rstrip(),
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
        "* missing/invalid | V variant | ! incomplete",
        size=8,
    )
    return {
        "row_layout": "uniform_categorical_grid",
        "displayed_levels": [str(row["vertebral_level"]) for row in displayed_rows],
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
            "Min waist",
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
    y = panel.top - 18
    for label, value in rows:
        pdf.setFillColor(MUTED)
        _text(pdf, panel.left, y, _truncate_to_width(label, panel.width - 55), size=8)
        pdf.setFillColor(INK)
        pdf.setFont(FONT_BOLD, 8)
        pdf.drawRightString(panel.right, y, value)
        y -= 15


def _axial_card_pair(
    *,
    left: float,
    width: float,
    content_bottom: float,
    content_top: float,
) -> tuple[ImageBox, ImageBox]:
    anthropometry_height = 82.0
    preferred_visual_height = width + (57.0 if width >= 180 else 67.0)
    available_height = content_top - content_bottom
    visual_height = min(
        preferred_visual_height,
        available_height - CARD_GAP - anthropometry_height,
    )
    if visual_height < 120:
        raise ValueError("Axial and anthropometry cards cannot fit the released layout.")
    axial_card = ImageBox(
        left,
        content_top - visual_height,
        width,
        visual_height,
    )
    anthropometry_card = ImageBox(
        left,
        axial_card.bottom - CARD_GAP - anthropometry_height,
        width,
        anthropometry_height,
    )
    return axial_card, anthropometry_card


def _draw_anthropometry_card(
    pdf: canvas.Canvas,
    case: CaseReportInput,
    settings: ReportingSettings,
    panel: ImageBox,
) -> dict[str, Any]:
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
        "separate_card": True,
        "box_width_pt": panel.width,
        "box_height_pt": panel.height,
        "box_top_pt": panel.top,
        "content_padding_pt": CARD_PADDING,
        "title_baseline_from_top_pt": CARD_TITLE_OFFSET,
        "labels": ["Min waist", "Pelvic max", "Waist / pelvic"],
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
        item_width = pdfmetrics.stringWidth(name, FONT_REGULAR, 8) + 18
        if (
            legend_x > panel.left + CARD_PADDING
            and legend_x + item_width > panel.right - CARD_PADDING
        ):
            legend_x = panel.left + CARD_PADDING
            legend_y -= 10
        red, green, blue = tissue_color(palette_name)
        pdf.setFillColor(colors.Color(red / 255, green / 255, blue / 255))
        pdf.rect(legend_x, legend_y, 7, 7, stroke=0, fill=1)
        pdf.setFillColor(MUTED)
        _text(pdf, legend_x + 10, legend_y - 1, name, size=8)
        legend_x += item_width

    return {
        "method": view.method,
        "vertebral_level": view.vertebral_level,
        "native_label": view.native_label,
        "position_superior_mm": view.position_superior_mm,
        "l3_available": view.l3_available,
        "tissue_names": list(view.tissue_names),
        "pixel_spacing_mm": view.pixel_spacing_mm,
        "field_of_view_mm": list(view.field_of_view_mm),
        "ct_window_hu": list(view.ct_window_hu),
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


def _model_text(case: CaseReportInput) -> str:
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
    return f"{orientation_model} | {case.vertebral_result.backend_id} | {tissue_model}"


def _technical_metadata_rows(
    case: CaseReportInput,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
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
    runtime = (
        " / ".join(
            value
            for value in (
                metadata.get("runtime_backend"),
                metadata.get("runtime_hardware"),
            )
            if value
        )
        or "Not recorded"
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
        (
            ("analysis", _analysis_date(metadata.get("analysis_started_at"))),
            ("input", input_text),
            ("pipeline", f"BodyComposition {pipeline_version}"),
        ),
        (
            ("runtime", runtime),
            ("scanner", scanner),
            ("slice", thickness_text),
            ("models", _model_text(case)),
        ),
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
    sections: list[_NotesSection] = []
    orientation_changed = False
    unchanged_uncertain = False
    displayed_entries = tuple(
        entry for entry in review_entries if entry.code not in ROUTINE_BOUNDARY_NOTE_CODES
    )
    suppressed_codes = tuple(
        sorted(
            {
                entry.code
                for entry in review_entries
                if entry.code in ROUTINE_BOUNDARY_NOTE_CODES
            }
        )
    )
    orientation = case.orientation_result
    if isinstance(orientation, Mapping):
        orientation_changed = bool(orientation.get("orientation_changed", False))
        orientation_review = bool(orientation.get("manual_review_required", False))
    else:
        orientation_changed = bool(orientation.orientation_changed)
        orientation_review = bool(orientation.manual_review_required)
    unchanged_uncertain = orientation_review and not orientation_changed
    if displayed_entries:
        entries_by_domain: dict[str, list[ReviewEntry]] = {}
        for entry in displayed_entries:
            entries_by_domain.setdefault(entry.domain, []).append(entry)
        compact_domain_summaries = len(displayed_entries) > 8
        for domain, entries in entries_by_domain.items():
            lines = (
                [_review_note_text(entries[0])]
                if compact_domain_summaries
                else [_review_note_text(entry) for entry in entries]
            )
            if compact_domain_summaries and len(entries) > 1:
                lines.append(
                    f"{len(entries) - 1} additional "
                    f"{'finding' if len(entries) == 2 else 'findings'} "
                    "in the canonical quality-control record."
                )
            sections.append(
                _NotesSection(
                    heading=NOTE_DOMAIN_LABELS.get(
                        domain,
                        domain.replace("_", " ").capitalize(),
                    ),
                    lines=tuple(lines),
                )
            )

    report_lines: list[str] = []
    if orientation_changed:
        report_lines.append("Action: input CT automatically reoriented before analysis.")
    elif unchanged_uncertain:
        report_lines.append("Action: processing continued without orientation repair.")
    attribution = case.technical_metadata.get("data_attribution")
    if attribution:
        report_lines.append(f"Source: {attribution}")
    if "missing_or_invalid_display_measurement" in warnings:
        missing = sum(not metric["valid"] for row in rows for metric in row["metrics"].values())
        report_lines.append(f"Table: {missing} values missing or invalid.")
    if displayed_entries:
        report_lines.append("Details: canonical quality-control record.")
    if not report_lines:
        report_lines.append("No report-level issues.")
    sections.append(
        _NotesSection(
            heading="Data and provenance",
            lines=tuple(report_lines),
        )
    )
    return _NotesSummary(
        sections=tuple(sections),
        review_count=len(review_entries),
        displayed_review_count=len(displayed_entries),
        suppressed_review_codes=suppressed_codes,
    )


def _note_section_block(
    section: _NotesSection,
    width: float,
) -> list[tuple[str, bool]]:
    lines: list[tuple[str, bool]] = [(section.heading, True)]
    for value in section.lines:
        lines.extend((line, False) for line in _wrap_text(value, width, size=8))
    return lines


def _balanced_note_section_columns(
    blocks: list[list[tuple[str, bool]]],
    column_count: int,
    maximum_height: float,
) -> list[list[list[tuple[str, bool]]]] | None:
    if column_count < 1 or len(blocks) < column_count:
        return None
    line_height = 10.0
    block_gap = 4.0
    block_count = len(blocks)
    line_prefix = [0]
    for block in blocks:
        line_prefix.append(line_prefix[-1] + len(block))

    def group_height(start: int, end: int) -> float:
        return (
            (line_prefix[end] - line_prefix[start]) * line_height
            + max(0, end - start - 1) * block_gap
        )

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
    return [
        blocks[boundaries[index] : boundaries[index + 1]]
        for index in range(column_count)
    ]


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

    maximum_columns = 3
    column_gap = 20.0
    preferred_columns = min(maximum_columns, len(notes.sections))
    available_height = panel.height - 48.0
    columns_used = 1
    column_width = panel.width - 2 * CARD_PADDING
    column_blocks: list[list[list[tuple[str, bool]]]] | None = None
    truncated = False

    for candidate_columns in range(preferred_columns, 0, -1):
        candidate_width = (
            panel.width
            - 2 * CARD_PADDING
            - column_gap * (candidate_columns - 1)
        ) / candidate_columns
        blocks = [
            _note_section_block(section, candidate_width)
            for section in notes.sections
        ]
        candidate_blocks = _balanced_note_section_columns(
            blocks,
            candidate_columns,
            available_height,
        )
        if candidate_blocks is not None:
            columns_used = candidate_columns
            column_width = candidate_width
            column_blocks = candidate_blocks
            break

    if column_blocks is None:
        columns_used = min(maximum_columns, max(1, len(notes.sections)))
        column_width = (
            panel.width
            - 2 * CARD_PADDING
            - column_gap * (columns_used - 1)
        ) / columns_used
        fallback = _NotesSection(
            heading="Details",
            lines=("See the canonical quality-control record for remaining observations.",),
        )
        column_blocks = [[_note_section_block(fallback, column_width)]]
        columns_used = 1
        truncated = True
    if column_blocks is None:
        raise RuntimeError("Notes/QC layout did not produce a renderable column.")

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
    for column_index, blocks in enumerate(column_blocks):
        x = panel.left + CARD_PADDING + column_index * (column_width + column_gap)
        y = body_top
        line_count = 0
        for block_index, block in enumerate(blocks):
            if block_index:
                y -= 4
            for line, bold in block:
                pdf.setFillColor(INK if bold else MUTED)
                _text(pdf, x, y, line, size=8, bold=bold)
                y -= 10
                line_count += 1
        column_line_counts.append(line_count)
    return {
        "columns_used": columns_used,
        "maximum_columns": maximum_columns,
        "column_width_pt": column_width,
        "column_gap_pt": column_gap,
        "content_padding_pt": CARD_PADDING,
        "column_block_counts": [len(blocks) for blocks in column_blocks],
        "column_line_counts": column_line_counts,
        "block_split": False,
        "box_top_pt": panel.top,
        "structured": True,
        "review_count": notes.review_count,
        "displayed_review_count": notes.displayed_review_count,
        "suppressed_review_codes": list(notes.suppressed_review_codes),
        "status_visible": False,
        "section_headings": [section.heading for section in notes.sections],
        "rendered_line_count": sum(column_line_counts),
        "truncated": truncated,
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
        layer_validities.append(
            display_valid & np.isfinite(layer) & (layer >= 0) & layer_valid
        )
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
    if "trunk_area_cm2" in slices:
        trunk = pd.to_numeric(slices["trunk_area_cm2"], errors="coerce").to_numpy(dtype=float)
        trunk_valid = np.isfinite(trunk)
        if "trunk_area_valid" in slices:
            trunk_valid &= slices["trunk_area_valid"].fillna(False).astype(bool).to_numpy()
        trunk_valid &= (positions >= projection.inferior_mm) & (positions <= projection.superior_mm)
        if np.any(trunk_valid):
            raw_max = max(raw_max, float(np.nanmax(trunk[trunk_valid])))
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
        pdf.setStrokeColor(colors.HexColor("#222222"))
        pdf.setLineWidth(0.8)
        pdf.setDash(3, 2)
        for run in _contiguous_runs(trunk_valid):
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
    pdf.setFillColor(colors.HexColor("#263746"))
    for value in np.linspace(0.0, x_max, 5):
        x = panel.left + value / x_max * panel.width
        pdf.setStrokeColor(colors.HexColor("#D7DDE1"))
        pdf.line(x, shared_y_box.bottom, x, shared_y_box.top)
        label = f"{value:.{tick_precision}f}"
        _centered(pdf, x, shared_y_box.bottom - 13, label, size=8)
    _centered(
        pdf,
        panel.left + panel.width / 2,
        shared_y_box.bottom - 27,
        "Tissue area (cm2)",
        size=8,
        bold=True,
    )
    trunk_reference_label = None
    if trunk is not None and np.any(trunk_valid):
        trunk_reference_label = "Trunk area reference"
        reference_y = shared_y_box.bottom - 43
        pdf.setStrokeColor(colors.HexColor("#222222"))
        pdf.setLineWidth(0.8)
        pdf.setDash(3, 2)
        pdf.line(panel.left + 4, reference_y + 3, panel.left + 24, reference_y + 3)
        pdf.setDash()
        pdf.setFillColor(colors.HexColor("#263746"))
        _text(pdf, panel.left + 29, reference_y, trunk_reference_label, size=8)
    legend_x = panel.left
    legend_y = shared_y_box.top + 7
    legend_rows = 1
    for name, label, _ in layers:
        item_width = pdfmetrics.stringWidth(label, FONT_REGULAR, 8) + 19
        if legend_x > panel.left and legend_x + item_width > panel.right:
            legend_x = panel.left
            legend_y -= 10
            legend_rows += 1
        red, green, blue = tissue_color(name)
        pdf.setFillColor(colors.Color(red / 255, green / 255, blue / 255))
        pdf.rect(legend_x, legend_y, 8, 8, stroke=0, fill=1)
        pdf.setFillColor(colors.HexColor("#263746"))
        _text(pdf, legend_x + 11, legend_y + 1, label, size=8)
        legend_x += item_width
    return {
        "layers": [name for name, _, _ in layers],
        "legend_labels": [label for _, label, _ in layers],
        "legend_rows": legend_rows,
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
        "displayed_trunk_reference_integral_cm3": (
            _display_integral_cm3(trunk, positions, trunk_valid) if trunk is not None else None
        ),
    }


def _draw_hu_heatmap(
    pdf: canvas.Canvas,
    case: CaseReportInput,
    projection: SagittalProjection,
    panel: ImageBox,
    shared_y_box: ImageBox,
    card_top: float,
) -> dict[str, Any]:
    slices = case.measurement_bundle.slices
    layers: list[tuple[str, str, str, str]] = []
    for name, label, area_column in _profile_layers(slices):
        stem = area_column.removesuffix("_area_cm2")
        value_column = f"{stem}_mean_hu"
        validity_column = f"{stem}_hu_valid"
        if value_column in slices and validity_column in slices:
            layers.append((name, label, value_column, validity_column))
    if not layers:
        raise ValueError("spine_profile_v2 requires per-slice tissue mean-HU columns.")

    pdf.setFillColor(INK)
    _text(
        pdf,
        panel.left,
        card_top - CARD_TITLE_OFFSET,
        "Tissue mean HU",
        size=8,
        bold=True,
    )
    pdf.setFillColor(MUTED)
    _text(
        pdf,
        panel.left,
        card_top - CARD_SUBTITLE_OFFSET,
        "white = NA",
        size=8,
    )

    lower = pd.to_numeric(
        slices["slice_slab_inferior_mm"],
        errors="coerce",
    ).to_numpy(dtype=float)
    upper = pd.to_numeric(
        slices["slice_slab_superior_mm"],
        errors="coerce",
    ).to_numpy(dtype=float)
    displayed = (
        np.isfinite(lower)
        & np.isfinite(upper)
        & (upper >= projection.inferior_mm)
        & (lower <= projection.superior_mm)
    )
    column_width = panel.width / len(layers)
    pdf.setFillColor(colors.white)
    pdf.rect(
        panel.left,
        shared_y_box.bottom,
        panel.width,
        shared_y_box.height,
        stroke=0,
        fill=1,
    )
    valid_cell_counts: dict[str, int] = {}
    clipped_cell_counts: dict[str, int] = {}
    for column_index, (name, label, value_column, validity_column) in enumerate(layers):
        display_label = label
        pdf.setFillColor(INK)
        _centered(
            pdf,
            panel.left + (column_index + 0.5) * column_width,
            shared_y_box.top + 7,
            display_label,
            size=8,
            bold=True,
        )
        values = pd.to_numeric(slices[value_column], errors="coerce").to_numpy(dtype=float)
        valid = (
            displayed
            & np.isfinite(values)
            & slices[validity_column].fillna(False).astype(bool).to_numpy()
        )
        valid_cell_counts[name] = int(np.count_nonzero(valid))
        clipped_cell_counts[name] = int(
            np.count_nonzero(valid & ((values < HU_SCALE_MIN) | (values > HU_SCALE_MAX)))
        )
        x = panel.left + column_index * column_width
        for row_index in np.flatnonzero(valid):
            inferior = max(float(lower[row_index]), projection.inferior_mm)
            superior = min(float(upper[row_index]), projection.superior_mm)
            y_inferior = projection.y_for_superior(
                inferior,
                shared_y_box.bottom,
                shared_y_box.top,
            )
            y_superior = projection.y_for_superior(
                superior,
                shared_y_box.bottom,
                shared_y_box.top,
            )
            cell_bottom = min(y_inferior, y_superior)
            cell_height = abs(y_superior - y_inferior)
            if cell_height <= 0:
                continue
            pdf.setFillColor(_hu_color(values[row_index]))
            pdf.rect(
                x,
                cell_bottom,
                column_width + 0.2,
                cell_height + 0.2,
                stroke=0,
                fill=1,
            )

    pdf.setStrokeColor(colors.HexColor("#AEB8C0"))
    pdf.setLineWidth(0.6)
    pdf.rect(
        panel.left,
        shared_y_box.bottom,
        panel.width,
        shared_y_box.height,
        stroke=1,
        fill=0,
    )
    for column_index in range(1, len(layers)):
        x = panel.left + column_index * column_width
        pdf.line(x, shared_y_box.bottom, x, shared_y_box.top)

    _centered(
        pdf,
        panel.left + panel.width / 2,
        shared_y_box.bottom - 14,
        "HU scale",
        size=8,
        bold=True,
    )
    bar_bottom = shared_y_box.bottom - 29
    bar_height = 6.0
    steps = 32
    for index in range(steps):
        fraction = index / (steps - 1)
        value = HU_SCALE_MIN + fraction * (HU_SCALE_MAX - HU_SCALE_MIN)
        pdf.setFillColor(_hu_color(value))
        pdf.rect(
            panel.left + index * panel.width / steps,
            bar_bottom,
            panel.width / steps + 0.2,
            bar_height,
            stroke=0,
            fill=1,
        )
    pdf.setFillColor(MUTED)
    _text(pdf, panel.left, shared_y_box.bottom - 43, "-190", size=8)
    _centered(
        pdf,
        panel.left + panel.width / 2,
        shared_y_box.bottom - 43,
        "-20",
        size=8,
    )
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawRightString(panel.right, shared_y_box.bottom - 43, "150")
    return {
        "tissues": [name for name, _, _, _ in layers],
        "display_labels": [label for _, label, _, _ in layers],
        "source_columns": [value_column for _, _, value_column, _ in layers],
        "validity_columns": [validity_column for _, _, _, validity_column in layers],
        "valid_cell_counts": valid_cell_counts,
        "clipped_cell_counts": clipped_cell_counts,
        "color_scale_hu": [HU_SCALE_MIN, HU_SCALE_MAX],
        "color_scale_midpoint_hu": (HU_SCALE_MIN + HU_SCALE_MAX) / 2.0,
        "palette": "cividis_like_monotonic_v1",
        "aggregation": "per_slice_voxel_mean",
        "smoothing": "none",
        "displayed_superior_range_mm": [
            projection.inferior_mm,
            projection.superior_mm,
        ],
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
    _draw_card(pdf, notes_card)
    notes_audit = _draw_notes_card(
        pdf,
        notes_card,
        _report_notes(case, rows, review_entries, warnings),
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
        axial_card, anthropometry_card = _axial_card_pair(
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
        _draw_card(pdf, anthropometry_card)
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
        axial_card, anthropometry_card = _axial_card_pair(
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
            "tissue_hu_heatmap",
            "vertebral_summary",
            "axial_segmentation",
            "key_anthropometry",
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
        _draw_card(pdf, anthropometry_card)
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
        audit["hu_heatmap"] = _draw_hu_heatmap(
            pdf,
            case,
            projection,
            hu_panel,
            image_box,
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
    page_identity = f"case page 1 of 1 | report {report_id[:12]}"
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
    export_report_id: str,
    total_cases: int,
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
        f"summary page 1 | report {export_report_id[:12]} | "
        f"{len(rows)} flagged cases of {total_cases}",
        size=8,
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
