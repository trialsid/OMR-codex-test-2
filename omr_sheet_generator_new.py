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
    BoxCoordinate,
    calculate_anchor_positions,
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


def _line(
    pdf: OMRPDF,
    geom: PageGeometry,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> None:
    """Draw a line using bottom-left coordinates."""
    pdf.line(x1, geom.height - y1, x2, geom.height - y2)


def draw_header_section(pdf: OMRPDF, geom: PageGeometry) -> None:
    """Render the header band with title, exam info, and instructions."""

    header_bottom = geom.header_bottom
    header_height = geom.height - header_bottom
    header_top = geom.height - geom.margin

    pdf.set_fill_color(247, 248, 250)
    _rect(pdf, geom, 0, header_bottom, geom.width, header_height, style=RenderStyle.F)

    pdf.set_draw_color(185, 185, 185)
    pdf.set_line_width(1.2)
    _line(pdf, geom, geom.margin, header_bottom, geom.width - geom.margin, header_bottom)
    pdf.set_draw_color(0, 0, 0)

    title_padding = 16
    pdf.set_font("Helvetica", "B", 20)
    title = "OMR Answer Sheet"
    title_width = pdf.get_string_width(title)
    title_x = (geom.width - title_width) / 2
    _text(pdf, geom, title_x, header_top - title_padding, title)

    pdf.set_font("Helvetica", size=12)
    exam_info = "Exam Date: _____________    Max Marks: _____________"
    exam_width = pdf.get_string_width(exam_info)
    exam_x = (geom.width - exam_width) / 2
    _text(pdf, geom, exam_x, header_top - title_padding - 26, exam_info)

    inner_width = geom.inner_width
    gutter = 28
    left_width = inner_width * 0.45
    right_x = geom.margin + left_width + gutter
    left_x = geom.margin
    divider_x = geom.margin + left_width + gutter / 2

    content_top = header_top - title_padding - 58
    line_height = 18

    pdf.set_font("Helvetica", "B", 12)
    _text(pdf, geom, left_x, content_top, "Student Details")
    pdf.set_line_width(0.6)
    pdf.set_draw_color(200, 200, 200)

    pdf.set_font("Helvetica", size=11)
    detail_y = content_top - line_height
    for label in ("Student Name", "Class / Section"):
        _text(pdf, geom, left_x, detail_y, f"{label}:")
        line_y = detail_y - 6
        _line(pdf, geom, left_x, line_y, left_x + left_width, line_y)
        detail_y -= line_height

    pdf.set_line_width(0.8)
    _line(pdf, geom, divider_x, header_bottom + 12, divider_x, content_top + 6)

    pdf.set_line_width(1)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 12)
    _text(pdf, geom, right_x, content_top, "Instructions")

    pdf.set_font("Helvetica", size=11)
    instructions = (
        "Use blue or black pen to fill bubbles.",
        "Shade only one option for each question.",
        "Do not fold or staple the sheet.",
    )
    instr_y = content_top - line_height
    for instruction in instructions:
        _text(pdf, geom, right_x, instr_y, f"- {instruction}")
        instr_y -= line_height

    pdf.set_line_width(1)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_fill_color(0, 0, 0)


def draw_anchor_markers(pdf: OMRPDF, geom: PageGeometry, markers: MarkerConfig) -> None:
    positions = calculate_anchor_positions(geom, markers)

    pdf.set_fill_color(0, 0, 0)
    for key in ("top_left", "top_right", "bottom_left", "bottom_right"):
        x, y = positions[key]
        _rect(pdf, geom, x, y, markers.anchor_size, markers.anchor_size, style=RenderStyle.F)


def draw_grid_markers(
    pdf: OMRPDF,
    geom: PageGeometry,
    markers: MarkerConfig,
    roll_bubbles: List[BubbleCoordinate],
    question_bubbles: List[BubbleCoordinate],
) -> None:
    """Draw one grid marker per bubble row, with clearance from anchors."""
    # Calculate anchor boundaries
    vertical_inset = geom.margin / 2.0

    # Bottom anchor top edge
    bottom_anchor_top = vertical_inset + markers.anchor_size

    # Top anchor bottom edge
    top_anchor_bottom = max(vertical_inset, geom.header_bottom - vertical_inset - markers.anchor_size)

    # Collect all unique bubble row y-coordinates
    bubble_y_positions = set()
    for bubble in roll_bubbles + question_bubbles:
        bubble_y_positions.add(bubble.y)

    # Filter positions that are too close to anchors (5-point clearance)
    clearance = 5
    valid_positions = sorted([
        y for y in bubble_y_positions
        if bottom_anchor_top + clearance <= y <= top_anchor_bottom - clearance
    ])

    # Draw grid markers at valid bubble row positions
    pdf.set_fill_color(0, 0, 0)
    for y in valid_positions:
        for x in (geom.margin / 2, geom.width - geom.margin / 2 - markers.grid_marker_size):
            _rect(
                pdf,
                geom,
                x,
                y - markers.grid_marker_size / 2,  # Center marker on bubble row
                markers.grid_marker_size,
                markers.grid_marker_size,
                style=RenderStyle.F,
            )


def draw_roll_number_section(
    pdf: OMRPDF,
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    bubbles: List[BubbleCoordinate],
    boxes: List[BoxCoordinate],
) -> None:
    if not bubbles:
        return

    # Find topmost bubble Y coordinate to calculate label position
    topmost_bubble_y = max(bubble.y for bubble in bubbles)

    # The roll number grid reserves four rows above the first bubble:
    #   row 0 -> label, row 1 -> spacer, row 2 -> boxes, row 3 -> spacer
    grid_top_y = topmost_bubble_y + layout.vertical_gap * 4

    # Roll label shares the same horizontal alignment as question headers
    label_x, _ = calculate_roll_label_position(geom, layout)
    label_y = grid_top_y - layout.vertical_gap

    pdf.set_font("Helvetica", size=12)
    _text(pdf, geom, label_x, label_y, "Roll Number")

    # Draw write-in boxes using pre-generated coordinates
    pdf.set_line_width(0.8)
    pdf.set_draw_color(128, 128, 128)
    for box in boxes:
        _rect(
            pdf,
            geom,
            box.x - box.width / 2,
            box.y - box.height / 2,
            box.width,
            box.height,
        )

    pdf.set_line_width(1)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_font("Helvetica", size=12)

    digit_width = 6.5
    gap_before_bubble = 5
    rows_seen = set()

    # Find the topmost and bottommost bubble y-coordinates
    min_y = min(bubble.y - bubble.radius for bubble in bubbles)
    max_y = max(bubble.y + bubble.radius for bubble in bubbles)

    # Draw vertical lines on either side of the label column
    left_line_x = geom.margin
    right_line_x = geom.margin + layout.label_column_width
    pdf.set_draw_color(192, 192, 192)  # Light grey
    pdf.set_line_width(1)
    _line(pdf, geom, left_line_x, min_y, left_line_x, max_y)
    _line(pdf, geom, right_line_x, min_y, right_line_x, max_y)
    pdf.set_draw_color(0, 0, 0)  # Reset to black

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
    markers: MarkerConfig,
    bubbles: List[BubbleCoordinate],
    roll_bottom: float,
) -> None:
    if not bubbles:
        return

    label_x, label_y = calculate_questions_label_position(geom, layout, sheet, roll_bottom, markers)
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

    # Calculate min and max y-coordinates for each question column
    column_y_ranges: dict[int, tuple[float, float]] = {}
    for bubble in bubbles:
        if bubble.question_column is None:
            continue
        col = bubble.question_column
        min_y = bubble.y - bubble.radius
        max_y = bubble.y + bubble.radius
        if col in column_y_ranges:
            column_y_ranges[col] = (
                min(column_y_ranges[col][0], min_y),
                max(column_y_ranges[col][1], max_y),
            )
        else:
            column_y_ranges[col] = (min_y, max_y)

    # Draw vertical lines on either side of each label column
    pdf.set_draw_color(192, 192, 192)  # Light grey
    pdf.set_line_width(1)
    for col, (min_y, max_y) in column_y_ranges.items():
        column_origin = geom.margin + col * layout.group_width(sheet.question_options)
        left_line_x = column_origin
        right_line_x = column_origin + layout.label_column_width
        _line(pdf, geom, left_line_x, min_y, left_line_x, max_y)
        _line(pdf, geom, right_line_x, min_y, right_line_x, max_y)
    pdf.set_draw_color(0, 0, 0)  # Reset to black

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

    roll_bubbles, roll_boxes, question_bubbles, roll_bottom = generate_all_bubble_coordinates(
        geom, layout, sheet, markers
    )

    draw_header_section(pdf, geom)
    draw_anchor_markers(pdf, geom, markers)
    draw_grid_markers(pdf, geom, markers, roll_bubbles, question_bubbles)
    draw_roll_number_section(pdf, geom, layout, sheet, roll_bubbles, roll_boxes)
    draw_question_columns(pdf, geom, layout, sheet, markers, question_bubbles, roll_bottom)

    pdf.output(str(output_path))


if __name__ == "__main__":
    target_path = Path("sheets") / "omr_sheet_fpdf.pdf"
    generate_omr_sheet(target_path)