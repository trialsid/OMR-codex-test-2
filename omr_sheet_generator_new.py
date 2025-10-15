"""Generate a modern OMR sheet PDF using fpdf2.

This module mirrors :mod:`omr_sheet_generator` but relies on ``fpdf2`` to
handle the PDF structure instead of building the content stream manually.
It keeps the same geometric calculations so that the generated sheet is
compatible with the OMR processing pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from fpdf import FPDF
from fpdf.enums import RenderStyle

from omr_config import PageGeometry, BubbleLayout, MarkerConfig, SheetLayout
from omr_layout import (
    BubbleCoordinate,
    calculate_questions_label_position,
    calculate_roll_label_position,
    generate_all_bubble_coordinates,
)


class OMRPDF(FPDF):
    """FPDF subclass configured for point units and no automatic margins."""

    def __init__(self, geom: PageGeometry):
        super().__init__(unit="pt", format=(geom.width, geom.height))
        self.set_auto_page_break(False)
        # Disable default margins so all coordinates map 1:1 with layout points.
        self.set_margins(0, 0, 0)
        self.add_page()
        self.set_font("Helvetica", size=12)


def _text(pdf: OMRPDF, geom: PageGeometry, x: float, y: float, text: str) -> None:
    """Render text at the given bottom-left coordinate."""

    pdf.text(x, geom.height - y, text)


def _rect(
    pdf: OMRPDF,
    geom: PageGeometry,
    x: float,
    y: float,
    width: float,
    height: float,
    style: RenderStyle = RenderStyle.D,
) -> None:
    """Draw a rectangle whose input coordinates use bottom-left origin."""

    pdf.rect(x, geom.height - y - height, width, height, style=style)


def _ellipse(
    pdf: OMRPDF,
    geom: PageGeometry,
    x: float,
    y: float,
    width: float,
    height: float,
    style: RenderStyle = RenderStyle.D,
) -> None:
    """Draw an ellipse using bottom-left coordinates for the bounding box."""

    pdf.ellipse(x, geom.height - y - height, width, height, style=style)


def draw_anchor_markers(pdf: OMRPDF, geom: PageGeometry, markers: MarkerConfig) -> None:
    inset = geom.margin / 2
    positions = [
        (inset, geom.height - inset - markers.anchor_size),
        (geom.width - inset - markers.anchor_size, geom.height - inset - markers.anchor_size),
        (inset, inset),
        (geom.width - inset - markers.anchor_size, inset),
    ]

    pdf.set_fill_color(0, 0, 0)
    for x, y in positions:
        _rect(pdf, geom, x, y, markers.anchor_size, markers.anchor_size, style=RenderStyle.F)


def draw_grid_markers(pdf: OMRPDF, geom: PageGeometry, markers: MarkerConfig) -> None:
    start_y = geom.margin + markers.grid_spacing
    end_y = geom.height - geom.margin - markers.grid_spacing
    y = start_y

    pdf.set_fill_color(0, 0, 0)
    while y <= end_y + 1e-6:
        for x in (geom.margin / 2, geom.width - geom.margin / 2 - markers.grid_marker_size):
            _rect(
                pdf,
                geom,
                x,
                y,
                markers.grid_marker_size,
                markers.grid_marker_size,
                style=RenderStyle.F,
            )
        y += markers.grid_spacing


def draw_roll_number_section(
    pdf: OMRPDF,
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    bubbles: List[BubbleCoordinate],
) -> None:
    if not bubbles:
        return

    label_x, label_y = calculate_roll_label_position(geom, layout)
    pdf.set_font("Helvetica", size=12)
    _text(pdf, geom, label_x, label_y, "Roll Number")

    top_y = geom.height - geom.margin - layout.diameter
    left_padding = layout.column_padding / 2
    x_start = geom.margin + layout.label_column_width + left_padding + layout.radius

    box_width = layout.diameter * 1.3
    box_height = layout.diameter * 1.2
    box_y_center = top_y - layout.vertical_gap * 2

    pdf.set_line_width(0.8)
    pdf.set_draw_color(128, 128, 128)
    for col in range(sheet.roll_columns):
        box_x_center = x_start + col * (layout.diameter + layout.option_gap)
        _rect(
            pdf,
            geom,
            box_x_center - box_width / 2,
            box_y_center - box_height / 2,
            box_width,
            box_height,
        )

    pdf.set_line_width(1)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_font("Helvetica", size=12)

    digit_width = 6.5
    gap_before_bubble = 5
    rows_seen = set()

    for bubble in bubbles:
        if bubble.row is not None and bubble.row not in rows_seen:
            label_end_x = geom.margin + layout.label_column_width - gap_before_bubble
            digit_x = label_end_x - digit_width
            digit_y = bubble.y - layout.radius / 2
            _text(pdf, geom, digit_x, digit_y, str(bubble.digit))
            rows_seen.add(bubble.row)

        _ellipse(
            pdf,
            geom,
            bubble.x - bubble.radius,
            bubble.y - bubble.radius,
            bubble.radius * 2,
            bubble.radius * 2,
        )


def draw_question_columns(
    pdf: OMRPDF,
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    bubbles: List[BubbleCoordinate],
    roll_bottom: float,
) -> None:
    if not bubbles:
        return

    label_x, label_y = calculate_questions_label_position(geom, layout, sheet, roll_bottom)
    if label_x > 0 and label_y > 0:
        pdf.set_font("Helvetica", size=12)
        _text(pdf, geom, label_x, label_y, "Questions")

    pdf.set_line_width(1)
    pdf.set_draw_color(0, 0, 0)

    topmost_per_column: dict[int, float] = {}
    for bubble in bubbles:
        if bubble.question_column is None:
            continue
        topmost_per_column[bubble.question_column] = max(
            topmost_per_column.get(bubble.question_column, float("-inf")),
            bubble.y,
        )

    pdf.set_font("Helvetica", size=10)
    for col, top_y in topmost_per_column.items():
        column_origin = geom.margin + col * layout.group_width(sheet.question_options)
        x_base = column_origin + layout.label_column_width + layout.column_padding / 2
        header_y = top_y + layout.vertical_gap * 0.7
        for opt in range(sheet.question_options):
            x = x_base + layout.radius + opt * (layout.diameter + layout.option_gap)
            letter = chr(ord("A") + opt)
            _text(pdf, geom, x - 3.25, header_y, letter)

    max_question_per_column: dict[int, int] = {}
    for bubble in bubbles:
        if bubble.question is None or bubble.question_column is None:
            continue
        max_question_per_column[bubble.question_column] = max(
            max_question_per_column.get(bubble.question_column, 0),
            bubble.question,
        )

    pdf.set_font("Helvetica", size=12)
    digit_width = 6.5
    gap_before_bubble = 5
    questions_seen = set()

    for bubble in bubbles:
        if bubble.question is not None and bubble.option_index == 0:
            if bubble.question not in questions_seen:
                column_origin = geom.margin + (bubble.question_column or 0) * layout.group_width(
                    sheet.question_options
                )
                label_end_x = column_origin + layout.label_column_width - gap_before_bubble
                current_text_width = len(str(bubble.question)) * digit_width
                label_x = label_end_x - current_text_width
                label_y = bubble.y - layout.radius / 2
                _text(pdf, geom, label_x, label_y, str(bubble.question))
                questions_seen.add(bubble.question)

        _ellipse(
            pdf,
            geom,
            bubble.x - bubble.radius,
            bubble.y - bubble.radius,
            bubble.radius * 2,
            bubble.radius * 2,
        )


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_omr_sheet(output_path: Path) -> None:
    geom = PageGeometry()
    layout = BubbleLayout()
    markers = MarkerConfig()
    sheet = SheetLayout()

    ensure_output_directory(output_path.parent)
    pdf = OMRPDF(geom)

    roll_bubbles, question_bubbles, roll_bottom = generate_all_bubble_coordinates(
        geom, layout, sheet
    )

    draw_anchor_markers(pdf, geom, markers)
    draw_grid_markers(pdf, geom, markers)
    draw_roll_number_section(pdf, geom, layout, sheet, roll_bubbles)
    draw_question_columns(pdf, geom, layout, sheet, question_bubbles, roll_bottom)

    pdf.output(str(output_path))


if __name__ == "__main__":
    target_path = Path("sheets") / "omr_sheet_fpdf.pdf"
    generate_omr_sheet(target_path)