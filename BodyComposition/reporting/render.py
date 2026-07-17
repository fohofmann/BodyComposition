"""Headless one-page PDF renderers for both fixed reporting stage layouts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
from pathlib import Path
import threading
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from BodyComposition.reporting.contracts import (
    MEASUREMENT_DEFINITIONS,
    CaseReportInput,
    ReportingSettings,
    ReviewEntry,
)
from BodyComposition.reporting.metrics import public_text
from BodyComposition.reporting.projection import (
    SagittalProjection,
    tissue_color,
    vertebral_color,
)


PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)
FONT_REGULAR = "BCVera"
FONT_BOLD = "BCVera-Bold"
MIN_FONT_SIZE = 8.0
_FONT_LOCK = threading.Lock()


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
    pdf.setCreator("BodyComposition reporting stage")
    pdf.setProducer("BodyComposition reporting stage")
    return pdf


def _text(pdf: canvas.Canvas, x: float, y: float, value: Any, *, size: float = 8, bold: bool = False) -> None:
    if size < MIN_FONT_SIZE:
        raise ValueError("Released PDF layouts do not permit text below 8 pt.")
    pdf.setFont(FONT_BOLD if bold else FONT_REGULAR, size)
    pdf.drawString(x, y, public_text(value, maximum=180))


def _centered(pdf: canvas.Canvas, x: float, y: float, value: Any, *, size: float = 8, bold: bool = False) -> None:
    if size < MIN_FONT_SIZE:
        raise ValueError("Released PDF layouts do not permit text below 8 pt.")
    text = public_text(value, maximum=100)
    font = FONT_BOLD if bold else FONT_REGULAR
    pdf.setFont(font, size)
    pdf.drawCentredString(x, y, text)


def _truncate_to_width(text: str, width: float, *, font: str = FONT_REGULAR, size: float = 8) -> str:
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
    settings: ReportingSettings,
    report_id: str,
    review_entries: tuple[ReviewEntry, ...],
    status: str,
) -> float:
    pdf.setFillColor(colors.HexColor("#17324D"))
    pdf.rect(0, PAGE_HEIGHT - 47, PAGE_WIDTH, 47, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    case_title = f"Case {case.case_id}"
    title_size = _fit_font_size(case_title, 590, maximum=13, font=FONT_BOLD)
    _text(pdf, 28, PAGE_HEIGHT - 24, case_title, size=title_size, bold=True)
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawRightString(
        PAGE_WIDTH - 28,
        PAGE_HEIGHT - 20,
        f"{settings.layout} | {settings.measure_aggregation}",
    )
    tissue = case.measurement_bundle.provenance.get("tissue", {}).get("backend_id", "unknown")
    _text(
        pdf,
        28,
        PAGE_HEIGHT - 36,
        _truncate_to_width(
            (
            f"analysis {case.measurement_bundle.identity.analysis_id[:12]} | "
            f"report {report_id[:12]} | vertebral {case.vertebral_result.backend_id} | "
            f"tissue {tissue} | analysis succeeded | QC "
            f"{'review' if review_entries else 'pass'} | report {status}"
            ),
            PAGE_WIDTH - 56,
        ),
        size=8,
    )
    content_top = PAGE_HEIGHT - 61
    if review_entries:
        changed = any(
            "lossless orientation transform" in entry.automatic_action.lower()
            for entry in review_entries
        )
        unchanged_uncertain = any(
            "without orientation repair" in entry.automatic_action.lower()
            for entry in review_entries
        )
        pdf.setFillColor(colors.HexColor("#A43E25" if changed else "#8A5B00"))
        pdf.roundRect(28, content_top - 18, PAGE_WIDTH - 56, 17, 3, stroke=0, fill=1)
        pdf.setFillColor(colors.white)
        domains = ", ".join(
            f"{entry.domain}:{entry.code}" for entry in review_entries[:4]
        )
        prefix = (
            "MANUAL REVIEW - input CT automatically reoriented before analysis: "
            if changed
            else (
                "MANUAL REVIEW - processing continued without orientation repair: "
                if unchanged_uncertain
                else "MANUAL REVIEW: "
            )
        )
        text = _truncate_to_width(
            prefix + domains + " | see canonical QC JSON",
            PAGE_WIDTH - 72,
            font=FONT_BOLD,
            size=8,
        )
        _text(pdf, 36, content_top - 13, text, size=8, bold=True)
        content_top -= 24
    return content_top


def _footer(pdf: canvas.Canvas, report_id: str, *, position: str = "case page 1 of 1") -> None:
    pdf.setStrokeColor(colors.HexColor("#A9B4BE"))
    pdf.line(28, 31, PAGE_WIDTH - 28, 31)
    pdf.setFillColor(colors.HexColor("#4B5963"))
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


def _image_box(projection: SagittalProjection, panel: ImageBox) -> ImageBox:
    height_px, width_px = projection.rgb.shape[:2]
    scale = min(panel.width / width_px, panel.height / height_px)
    width = width_px * scale
    height = height_px * scale
    return ImageBox(
        left=panel.left + (panel.width - width) / 2,
        bottom=panel.bottom + (panel.height - height) / 2,
        width=width,
        height=height,
    )


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
    pdf.drawImage(image, box.left, box.bottom, width=box.width, height=box.height, preserveAspectRatio=True, mask="auto")
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
            row for row in rows
            if row["native_label"] in projection.native_labels
            and all(value is not None for value in row["centroid_lps_xyz"])
        ),
        key=lambda item: item["centroid_lps_xyz"][2],
        reverse=True,
    )
    used_y: list[float] = []
    for index, centroid in enumerate(centroids):
        point = centroid["centroid_lps_xyz"]
        y_true = projection.y_for_superior(point[2], box.bottom, box.top)
        x_true = box.left + (
            (point[1] - projection.anterior_mm)
            / (projection.posterior_mm - projection.anterior_mm)
        ) * box.width
        y_label = y_true
        for previous in used_y:
            if abs(y_label - previous) < 10:
                y_label += 10 if index % 2 == 0 else -10
        y_label = float(np.clip(y_label, box.bottom + 10, box.top - 10))
        used_y.append(y_label)
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


def _validate_row_separation(rows: list[dict[str, Any]], projection: SagittalProjection, box: ImageBox) -> None:
    positions = [
        projection.y_for_superior(float(row["centroid_superior_mm"]), box.bottom, box.top)
        for row in rows
        if row["centroid_superior_mm"] is not None
        and projection.inferior_mm <= float(row["centroid_superior_mm"]) <= projection.superior_mm
    ]
    positions.sort()
    if len(positions) > 1 and min(np.diff(positions)) < 8.0:
        raise ValueError("Physical vertebral rows cannot fit the released layout at 8 pt.")


def _draw_table(
    pdf: canvas.Canvas,
    rows: list[dict[str, Any]],
    settings: ReportingSettings,
    projection: SagittalProjection,
    panel: ImageBox,
    shared_y_box: ImageBox,
) -> None:
    _validate_row_separation(rows, projection, shared_y_box)
    pdf.setFillColor(colors.HexColor("#263746"))
    _text(
        pdf,
        panel.left,
        shared_y_box.top + 28,
        "Whole-territory means from three physical bins",
        size=8,
        bold=True,
    )
    level_width = 43.0
    metric_width = (panel.width - level_width) / len(settings.measurement_columns)
    header_y = shared_y_box.top + 13
    _centered(pdf, panel.left + level_width / 2, header_y, "Level", size=8, bold=True)
    for index, column in enumerate(settings.measurement_columns):
        definition = MEASUREMENT_DEFINITIONS[column]
        _centered(
            pdf,
            panel.left + level_width + (index + 0.5) * metric_width,
            header_y,
            definition["short_label"],
            size=8,
            bold=True,
        )
        _centered(
            pdf,
            panel.left + level_width + (index + 0.5) * metric_width,
            header_y - 9,
            f"({definition['unit']})",
            size=8,
        )
    pdf.setStrokeColor(colors.HexColor("#AEB8C0"))
    pdf.line(panel.left, shared_y_box.top + 1, panel.right, shared_y_box.top + 1)
    for row in rows:
        position = row["centroid_superior_mm"]
        if position is None or not projection.inferior_mm <= float(position) <= projection.superior_mm:
            continue
        y = projection.y_for_superior(float(position), shared_y_box.bottom, shared_y_box.top)
        pdf.setStrokeColor(colors.HexColor("#E1E5E8"))
        pdf.line(panel.left, y - 5, panel.right, y - 5)
        marker = "".join(
            (
                "V" if row["anatomical_variant"] else "",
                "!" if not row["territory_complete"] else "",
            )
        )
        _text(pdf, panel.left + 2, y - 3, f"{row['vertebral_level']} {marker}".rstrip(), size=8, bold=True)
        for index, column in enumerate(settings.measurement_columns):
            value = _format_metric(row["metrics"][column], settings)
            _centered(
                pdf,
                panel.left + level_width + (index + 0.5) * metric_width,
                y - 3,
                value,
                size=8,
            )
    pdf.setFillColor(colors.HexColor("#596771"))
    _text(pdf, panel.left, panel.bottom - 13, "* missing/invalid; V anatomical variant; ! incomplete territory", size=8)


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
        float(np.trapezoid(values_cm2[run], positions_mm[run]))
        for run in _contiguous_runs(valid)
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
    for name, _, column in layers:
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
        shared_y_box.top + 28,
        "Stacked tissue-area profile",
        size=8,
        bold=True,
    )
    pdf.setStrokeColor(colors.HexColor("#AEB8C0"))
    pdf.rect(panel.left, shared_y_box.bottom, panel.width, shared_y_box.height, stroke=1, fill=0)
    cumulative = np.zeros(len(slices), dtype=float)
    for name, label, _ in layers:
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
                    projection.y_for_superior(positions[index], shared_y_box.bottom, shared_y_box.top),
                )
            for index in run[::-1]:
                path.lineTo(
                    panel.left + lower[index] / x_max * panel.width,
                    projection.y_for_superior(positions[index], shared_y_box.bottom, shared_y_box.top),
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
                y = projection.y_for_superior(positions[index], shared_y_box.bottom, shared_y_box.top)
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
        _centered(pdf, x, panel.bottom - 13, label, size=8)
    _centered(pdf, panel.left + panel.width / 2, panel.bottom - 27, "area (cm2)", size=8, bold=True)
    legend_x = panel.left
    for name, label, _ in layers:
        red, green, blue = tissue_color(name)
        pdf.setFillColor(colors.Color(red / 255, green / 255, blue / 255))
        pdf.rect(legend_x, shared_y_box.top + 8, 8, 8, stroke=0, fill=1)
        pdf.setFillColor(colors.HexColor("#263746"))
        _text(pdf, legend_x + 11, shared_y_box.top + 9, label, size=8)
        legend_x += pdfmetrics.stringWidth(label, FONT_REGULAR, 8) + 19
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
            _display_integral_cm3(trunk, positions, trunk_valid)
            if trunk is not None
            else None
        ),
    }


def render_case_page(
    destination: Path,
    *,
    case: CaseReportInput,
    settings: ReportingSettings,
    report_id: str,
    projection: SagittalProjection,
    rows: list[dict[str, Any]],
    review_entries: tuple[ReviewEntry, ...],
    status: str,
) -> dict[str, Any]:
    """Render exactly one fixed A4 landscape page and return display audit data."""

    pdf = _new_canvas(destination, case_id=case.case_id)
    pdf.bookmarkPage(case.case_id)
    content_top = _header(pdf, case, settings, report_id, review_entries, status)
    content_bottom = 57.0
    audit: dict[str, Any] = {"layout": settings.layout, "displayed_rows": rows}
    if settings.layout == "spine_overview_v1":
        left_panel = ImageBox(28, content_bottom, 330, content_top - content_bottom - 48)
        image_box = _draw_projection(
            pdf,
            projection,
            left_panel,
            ct_window=settings.ct_window,
            rows=rows,
        )
        table_panel = ImageBox(378, image_box.bottom, PAGE_WIDTH - 406, image_box.height)
        _draw_table(pdf, rows, settings, projection, table_panel, image_box)
    elif settings.layout == "spine_profile_v2":
        left_panel = ImageBox(28, content_bottom + 20, 245, content_top - content_bottom - 68)
        image_box = _draw_projection(
            pdf,
            projection,
            left_panel,
            ct_window=settings.ct_window,
            rows=rows,
        )
        profile_panel = ImageBox(291, content_bottom + 20, 180, image_box.height)
        table_panel = ImageBox(490, content_bottom + 20, PAGE_WIDTH - 518, image_box.height)
        audit["profile"] = _draw_profile(pdf, case, projection, profile_panel, image_box)
        _draw_table(pdf, rows, settings, projection, table_panel, image_box)
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
    preserves that available overview. It never fabricates missing measurement stage values.
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
                "measurement stage output incomplete",
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
    count_text = "; ".join(f"{domain}:{code}={count}" for (domain, code), count in sorted(counts.items()))
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
        review_statuses = ", ".join(
            dict.fromkeys(entry["review_status"] for entry in entries)
        )
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
