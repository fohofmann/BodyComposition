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
LAYOUT_REVISION = "scientific_onepager_v5"
_FONT_LOCK = threading.Lock()
PAGE_BACKGROUND = colors.HexColor("#F4F7F9")
CARD_BACKGROUND = colors.white
INK = colors.HexColor("#1D3343")
MUTED = colors.HexColor("#5E6E79")
BORDER = colors.HexColor("#D4DDE3")
ACCENT = colors.HexColor("#16738A")


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


def _header(
    pdf: canvas.Canvas,
    case: CaseReportInput,
    report_id: str,
    review_entries: tuple[ReviewEntry, ...],
) -> float:
    pdf.setFillColor(PAGE_BACKGROUND)
    pdf.rect(0, PAGE_HEIGHT - 64, PAGE_WIDTH, 64, stroke=0, fill=1)
    pdf.setFillColor(ACCENT)
    pdf.rect(0, PAGE_HEIGHT - 4, PAGE_WIDTH, 4, stroke=0, fill=1)
    pdf.setFillColor(INK)
    _text(pdf, 28, PAGE_HEIGHT - 27, "Body composition analysis", size=15, bold=True)
    pdf.setFillColor(MUTED)
    _text(
        pdf, 28, PAGE_HEIGHT - 43, "Automated CT segmentation and quantitative QC summary", size=8
    )

    pdf.setFillColor(MUTED)
    _text(pdf, 434, PAGE_HEIGHT - 20, "CASE", size=8, bold=True)
    pdf.setFillColor(INK)
    case_title = public_text(case.case_id, maximum=64)
    case_size = _fit_font_size(case_title, 205, maximum=13, font=FONT_BOLD)
    _text(pdf, 434, PAGE_HEIGHT - 38, case_title, size=case_size, bold=True)
    pdf.setFillColor(MUTED)
    _text(
        pdf,
        434,
        PAGE_HEIGHT - 52,
        f"analysis {case.measurement_bundle.identity.analysis_id[:12]} | report {report_id[:12]}",
        size=8,
    )

    pdf.setStrokeColor(BORDER)
    pdf.line(28, PAGE_HEIGHT - 64, PAGE_WIDTH - 28, PAGE_HEIGHT - 64)

    content_top = PAGE_HEIGHT - 75
    if review_entries:
        changed = any(
            "lossless orientation transform" in entry.automatic_action.lower()
            for entry in review_entries
        )
        unchanged_uncertain = any(
            "without orientation repair" in entry.automatic_action.lower()
            for entry in review_entries
        )
        pdf.setFillColor(colors.HexColor("#FFF1E8" if changed else "#FFF7DF"))
        pdf.setStrokeColor(colors.HexColor("#D38143" if changed else "#D2A33C"))
        pdf.roundRect(28, content_top - 23, PAGE_WIDTH - 56, 22, 4, stroke=1, fill=1)
        pdf.setFillColor(colors.HexColor("#7B3A13" if changed else "#765400"))
        domains = ", ".join(f"{entry.domain}:{entry.code}" for entry in review_entries[:4])
        prefix = (
            "Manual review required - input CT automatically reoriented before analysis: "
            if changed
            else (
                "Manual review required - processing continued without orientation repair: "
                if unchanged_uncertain
                else "Manual review required: "
            )
        )
        text = _truncate_to_width(
            prefix + domains + " | see canonical QC JSON",
            PAGE_WIDTH - 82,
            font=FONT_BOLD,
            size=8,
        )
        _text(pdf, 39, content_top - 16, text, size=8, bold=True)
        content_top -= 32
    return content_top


def _footer(pdf: canvas.Canvas, report_id: str, *, position: str = "case page 1 of 1") -> None:
    pdf.setStrokeColor(BORDER)
    pdf.line(28, 31, PAGE_WIDTH - 28, 31)
    pdf.setFillColor(MUTED)
    _text(
        pdf,
        28,
        17,
        "Automated research/QC report. Canonical machine-readable outputs are authoritative. Flagged results require review.",
        size=8,
    )
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawRightString(
        PAGE_WIDTH - 28,
        17,
        f"{position} | report {report_id[:16]}",
    )


def _draw_card(pdf: canvas.Canvas, box: ImageBox) -> None:
    pdf.setFillColor(CARD_BACKGROUND)
    pdf.setStrokeColor(BORDER)
    pdf.setLineWidth(0.7)
    pdf.roundRect(box.left, box.bottom, box.width, box.height, 5, stroke=1, fill=1)


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
        pdf.roundRect(label_x, y_label - 5, label_width, 11, 2, stroke=0, fill=1)
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
    displayed_rows = [
        row for row in rows if int(row["native_label"]) in projection.native_labels
    ]
    if not displayed_rows:
        raise ValueError("The vertebral table has no detected native levels to display.")
    pdf.setFillColor(colors.HexColor("#263746"))
    _text(
        pdf,
        panel.left,
        panel.top - 8,
        "Vertebral summary",
        size=9,
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
    if row_height < 9.5:
        raise ValueError(
            "Detected vertebral rows cannot fit the released uniform table at 8 pt."
        )
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
        "* missing/invalid | V variant | ! incomplete territory",
        size=8,
    )
    return {
        "row_layout": "uniform_categorical_grid",
        "displayed_levels": [str(row["vertebral_level"]) for row in displayed_rows],
        "row_height_pt": float(row_height),
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
    valid = bool(summary.get(name.replace("_cm", "") + "_valid", False))
    if name.endswith("_cm"):
        valid = bool(summary.get(f"{name[:-3]}_valid", False))
    elif name.endswith("_ratio"):
        valid = bool(summary.get(f"{name}_valid", False))
    if not valid or pd.isna(value):
        return settings.missing_value_symbol + "*"
    formatted = f"{float(value):.{settings.numeric_precision}f}"
    return f"{formatted} {unit}".rstrip()


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


def _draw_axial_panel(
    pdf: canvas.Canvas,
    case: CaseReportInput,
    settings: ReportingSettings,
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
        panel.left + 10,
        panel.top - 18,
        _truncate_to_width(title, panel.width - 20),
        size=9,
        bold=True,
    )
    pdf.setFillColor(MUTED)
    subtitle_prefix = "soft-tissue window" if panel.width >= 180 else "window"
    subtitle = (
        f"{subtitle_prefix} {int(view.ct_window_hu[0])} to {int(view.ct_window_hu[1])} HU"
        + ("" if view.l3_available else " | L3 unavailable")
    )
    _text(
        pdf, panel.left + 10, panel.top - 31, _truncate_to_width(subtitle, panel.width - 20), size=8
    )

    key_height = 81.0
    image_region = ImageBox(
        panel.left + 10,
        panel.bottom + key_height + 26,
        panel.width - 20,
        max(45.0, panel.height - key_height - 73),
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

    legend_y = max(panel.bottom + key_height + 18, box.bottom - 17)
    legend_x = panel.left + 10
    palette_lookup = {"SM": "sm", "IMAT": "imat", "SAT": "sat", "aVAT": "avat", "tVAT": "tvat"}
    for name in view.tissue_names:
        palette_name = palette_lookup.get(name)
        if palette_name is None:
            continue
        item_width = pdfmetrics.stringWidth(name, FONT_REGULAR, 8) + 18
        if legend_x > panel.left + 10 and legend_x + item_width > panel.right - 10:
            legend_x = panel.left + 10
            legend_y -= 10
        red, green, blue = tissue_color(palette_name)
        pdf.setFillColor(colors.Color(red / 255, green / 255, blue / 255))
        pdf.rect(legend_x, legend_y, 7, 7, stroke=0, fill=1)
        pdf.setFillColor(MUTED)
        _text(pdf, legend_x + 10, legend_y - 1, name, size=8)
        legend_x += item_width

    separator_y = max(panel.bottom + key_height, legend_y - 13)
    pdf.setStrokeColor(BORDER)
    pdf.line(panel.left + 10, separator_y, panel.right - 10, separator_y)
    _draw_key_anthropometry(
        pdf,
        case,
        settings,
        ImageBox(
            panel.left + 10,
            separator_y - key_height + 12,
            panel.width - 20,
            key_height - 24,
        ),
    )
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
            orientation_model = str(model.get("asset_id") or model.get("name") or orientation_model)
    tissue = case.measurement_bundle.provenance.get("tissue", {})
    tissue_model = (
        tissue.get("backend_id", "tissue not recorded")
        if isinstance(tissue, Mapping)
        else "tissue not recorded"
    )
    return f"{orientation_model} | {case.vertebral_result.backend_id} | {tissue_model}"


def _draw_metadata_card(pdf: canvas.Canvas, case: CaseReportInput, panel: ImageBox) -> None:
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
        " | ".join(
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
        f"{str(metadata.get('input_format') or 'unknown').upper()} | "
        f"{spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm"
    )
    pipeline_version = metadata.get("pipeline_version") or __version__
    rows = (
        ("Analysis", _analysis_date(metadata.get("analysis_started_at")), "Input", input_text),
        ("Pipeline", f"BodyComposition {pipeline_version}", "Scanner", scanner),
        ("Runtime", runtime, "Slice", thickness_text),
    )
    pdf.setFillColor(INK)
    _text(pdf, panel.left + 12, panel.top - 18, "Technical metadata", size=9, bold=True)
    column_width = (panel.width - 34) / 2
    y = panel.top - 37
    for left_label, left_value, right_label, right_value in rows:
        for offset, label, value in (
            (0.0, left_label, left_value),
            (column_width + 10, right_label, right_value),
        ):
            x = panel.left + 12 + offset
            pdf.setFillColor(MUTED)
            _text(pdf, x, y, label.upper(), size=8, bold=True)
            pdf.setFillColor(INK)
            _text(
                pdf,
                x + 53,
                y,
                _truncate_to_width(str(value), column_width - 55),
                size=8,
            )
        y -= 15
    pdf.setFillColor(MUTED)
    _text(pdf, panel.left + 12, y, "MODELS", size=8, bold=True)
    pdf.setFillColor(INK)
    _text(
        pdf,
        panel.left + 65,
        y,
        _truncate_to_width(_model_text(case), panel.width - 79),
        size=8,
    )


def _report_notes(
    case: CaseReportInput,
    rows: list[dict[str, Any]],
    review_entries: tuple[ReviewEntry, ...],
    warnings: tuple[str, ...],
) -> list[str]:
    notes: list[str] = []
    attribution = case.technical_metadata.get("data_attribution")
    if attribution:
        notes.append(f"Data: {attribution}")
    orientation_changed = any(
        "lossless orientation transform" in entry.automatic_action.lower()
        for entry in review_entries
    )
    if orientation_changed:
        notes.append("Orientation corrected automatically; manual orientation review is required.")
    for entry in review_entries:
        note = f"{entry.domain}: {entry.reason}"
        if note not in notes:
            notes.append(note)
    if "missing_or_invalid_display_measurement" in warnings:
        missing = sum(not metric["valid"] for row in rows for metric in row["metrics"].values())
        notes.append(
            f"{missing} vertebral table value(s) are missing or invalid; see canonical QC."
        )
    return notes or ["No automated review flags or report-level issues."]


def _draw_notes_card(
    pdf: canvas.Canvas,
    panel: ImageBox,
    notes: list[str],
) -> None:
    pdf.setFillColor(INK)
    _text(pdf, panel.left + 12, panel.top - 18, "Notes / QC", size=9, bold=True)
    available_lines = max(1, int((panel.height - 35) // 11))
    lines: list[str] = []
    for note in notes:
        wrapped = _wrap_text(f"- {note}", panel.width - 24)
        if len(lines) + len(wrapped) > available_lines:
            break
        lines.extend(wrapped)
    if not lines:
        lines = ["- Additional issues are recorded in canonical QC."]
    elif len(lines) < sum(len(_wrap_text(f"- {note}", panel.width - 24)) for note in notes):
        lines[-1] = _truncate_to_width(lines[-1] + " ...", panel.width - 24)
    y = panel.top - 36
    pdf.setFillColor(MUTED)
    for line in lines[:available_lines]:
        _text(pdf, panel.left + 12, y, line, size=8)
        y -= 11


def _profile_layers(slices: pd.DataFrame) -> list[tuple[str, str, str]]:
    layers: list[tuple[str, str, str]] = []
    for name, label in (("sm", "SM"), ("imat", "IMAT"), ("sat", "SAT")):
        if f"{name}_area_cm2" in slices:
            layers.append((name, label, f"{name}_area_cm2"))
    if all(f"{name}_area_cm2" in slices for name in ("avat", "tvat")):
        layers.extend((("avat", "aVAT", "avat_area_cm2"), ("tvat", "tVAT", "tvat_area_cm2")))
    elif "vat_area_cm2" in slices:
        layers.append(("vat", "VAT", "vat_area_cm2"))
    elif "total_vat_area_cm2" in slices:
        layers.append(("vat", "total VAT", "total_vat_area_cm2"))
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
) -> dict[str, Any]:
    slices = case.measurement_bundle.slices
    layers = _profile_layers(slices)
    if not layers:
        raise ValueError("spine_profile_v2 requires canonical per-slice tissue-area columns.")
    positions = pd.to_numeric(slices["position_superior_mm"], errors="coerce").to_numpy(dtype=float)
    values: list[np.ndarray] = []
    valid = np.isfinite(positions)
    for _name, _, column in layers:
        layer = pd.to_numeric(slices[column], errors="coerce").to_numpy(dtype=float)
        validity_column = column.replace("_cm2", "_valid")
        layer_valid = (
            slices[validity_column].fillna(False).astype(bool).to_numpy()
            if validity_column in slices
            else np.isfinite(layer)
        )
        valid &= np.isfinite(layer) & (layer >= 0) & layer_valid
        values.append(layer)
    valid &= (positions >= projection.inferior_mm) & (positions <= projection.superior_mm)
    stack = np.sum(np.vstack(values), axis=0)
    raw_max = float(np.nanmax(stack[valid])) if np.any(valid) else 0.0
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
        shared_y_box.top + 32,
        "Stacked tissue-area profile",
        size=8,
        bold=True,
    )
    pdf.setStrokeColor(colors.HexColor("#AEB8C0"))
    pdf.rect(panel.left, shared_y_box.bottom, panel.width, shared_y_box.height, stroke=1, fill=0)
    cumulative = np.zeros(len(slices), dtype=float)
    for name, _label, _ in layers:
        layer_index = [item[0] for item in layers].index(name)
        layer_values = values[layer_index]
        lower = cumulative.copy()
        upper = lower + layer_values
        red, green, blue = tissue_color(name)
        pdf.setFillColor(colors.Color(red / 255, green / 255, blue / 255, alpha=0.78))
        pdf.setStrokeColor(colors.Color(red / 255, green / 255, blue / 255))
        for run in _contiguous_runs(valid):
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
        "area (cm2)",
        size=8,
        bold=True,
    )
    legend_x = panel.left
    legend_y = shared_y_box.top + 17
    for name, label, _ in layers:
        item_width = pdfmetrics.stringWidth(label, FONT_REGULAR, 8) + 19
        if legend_x > panel.left and legend_x + item_width > panel.right:
            legend_x = panel.left
            legend_y -= 10
        red, green, blue = tissue_color(name)
        pdf.setFillColor(colors.Color(red / 255, green / 255, blue / 255))
        pdf.rect(legend_x, legend_y, 8, 8, stroke=0, fill=1)
        pdf.setFillColor(colors.HexColor("#263746"))
        _text(pdf, legend_x + 11, legend_y + 1, label, size=8)
        legend_x += item_width
    return {
        "layers": [name for name, _, _ in layers],
        "source_columns": [column for _, _, column in layers],
        "displayed_layer_integrals_cm3": {
            name: _display_integral_cm3(layer, positions, valid)
            for (name, _, _), layer in zip(layers, values, strict=True)
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
        "displayed_trunk_reference_integral_cm3": (
            _display_integral_cm3(trunk, positions, trunk_valid) if trunk is not None else None
        ),
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
    content_top = _header(pdf, case, report_id, review_entries)
    content_bottom = 151.0
    metadata_card = ImageBox(24, 42, 526, 99)
    notes_card = ImageBox(560, 42, PAGE_WIDTH - 584, 99)
    _draw_card(pdf, metadata_card)
    _draw_card(pdf, notes_card)
    _draw_metadata_card(pdf, case, metadata_card)
    _draw_notes_card(
        pdf,
        notes_card,
        _report_notes(case, rows, review_entries, warnings),
    )
    audit: dict[str, Any] = {
        "layout": settings.layout,
        "layout_revision": LAYOUT_REVISION,
        "displayed_rows": rows,
        "technical_metadata": dict(case.technical_metadata),
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
        table_card = ImageBox(259, content_bottom, PAGE_WIDTH - 498, content_top - content_bottom)
        axial_card = ImageBox(
            table_card.right + 10, content_bottom, 205, content_top - content_bottom
        )
        audit["panel_order"] = [
            "sagittal_spine",
            "vertebral_summary",
            "axial_segmentation",
        ]
        for card in (sagittal_card, table_card, axial_card):
            _draw_card(pdf, card)
        pdf.setFillColor(INK)
        _text(
            pdf,
            sagittal_card.left + 10,
            sagittal_card.top - 18,
            "Spine localization",
            size=9,
            bold=True,
        )
        pdf.setFillColor(MUTED)
        _text(
            pdf,
            sagittal_card.left + 10,
            sagittal_card.top - 31,
            "Sagittal thick-slab projection",
            size=8,
        )
        left_panel = ImageBox(
            sagittal_card.left + 10,
            sagittal_card.bottom + 24,
            sagittal_card.width - 20,
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
            sagittal_card.left + 10,
            sagittal_card.bottom + 9,
            f"window {int(settings.ct_window[0])} to {int(settings.ct_window[1])} HU",
            size=8,
        )
        audit["axial_view"] = _draw_axial_panel(
            pdf,
            case,
            settings,
            axial_view,
            axial_card,
        )
        table_panel = ImageBox(
            table_card.left + 10,
            table_card.bottom + 24,
            table_card.width - 20,
            table_card.height - 32,
        )
        audit["table"] = _draw_table(pdf, rows, settings, projection, table_panel)
    elif settings.layout == "spine_profile_v2":
        sagittal_card = ImageBox(24, content_bottom, 150, content_top - content_bottom)
        profile_card = ImageBox(
            sagittal_card.right + 8, content_bottom, 160, content_top - content_bottom
        )
        axial_card = ImageBox(
            PAGE_WIDTH - 169, content_bottom, 145, content_top - content_bottom
        )
        table_card = ImageBox(
            profile_card.right + 8,
            content_bottom,
            axial_card.left - profile_card.right - 16,
            content_top - content_bottom,
        )
        audit["panel_order"] = [
            "sagittal_spine",
            "tissue_area_profile",
            "vertebral_summary",
            "axial_segmentation",
        ]
        for card in (sagittal_card, profile_card, table_card, axial_card):
            _draw_card(pdf, card)
        pdf.setFillColor(INK)
        _text(
            pdf,
            sagittal_card.left + 8,
            sagittal_card.top - 18,
            "Spine localization",
            size=9,
            bold=True,
        )
        pdf.setFillColor(MUTED)
        _text(
            pdf,
            sagittal_card.left + 8,
            sagittal_card.top - 31,
            "Sagittal thick-slab projection",
            size=8,
        )
        left_panel = ImageBox(
            sagittal_card.left + 8,
            sagittal_card.bottom + 27,
            sagittal_card.width - 16,
            sagittal_card.height - 70,
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
            sagittal_card.left + 8,
            sagittal_card.bottom + 9,
            f"window {int(settings.ct_window[0])} to {int(settings.ct_window[1])} HU",
            size=8,
        )
        audit["axial_view"] = _draw_axial_panel(
            pdf,
            case,
            settings,
            axial_view,
            axial_card,
        )
        profile_panel = ImageBox(
            profile_card.left + 8,
            profile_card.bottom + 28,
            profile_card.width - 16,
            image_box.height,
        )
        table_panel = ImageBox(
            table_card.left + 10,
            table_card.bottom + 24,
            table_card.width - 20,
            table_card.height - 32,
        )
        audit["profile"] = _draw_profile(pdf, case, projection, profile_panel, image_box)
        audit["table"] = _draw_table(pdf, rows, settings, projection, table_panel)
    else:
        raise ValueError(f"Unsupported report layout {settings.layout!r}.")
    _footer(pdf, report_id)
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
    _text(
        pdf,
        28,
        PAGE_HEIGHT - 25,
        case_title,
        size=_fit_font_size(case_title, PAGE_WIDTH - 56, maximum=13, font=FONT_BOLD),
        bold=True,
    )
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
    _footer(pdf, report_id)
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
        f"export report {export_report_id[:16]} | {len(rows)} flagged cases of {total_cases}",
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
    _footer(pdf, export_report_id, position="summary page 1")
    pdf.showPage()
    pdf.save()
