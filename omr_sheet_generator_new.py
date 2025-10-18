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
    BubbleGroup,
    BoxCoordinate,
    calculate_anchor_positions,
    generate_all_bubble_coordinates,
    build_section_specifications,
    allocate_column_rows,
)


class OMRPDF(FPDF):
    """FPDF subclass configured for point units and no automatic margins."""

    def __init__(self, geom: PageGeometry):
        super().__init__(unit="pt", format=(geom.width, geom.height))
        self.set_auto_page_break(False)
        # Disable default margins so all coordinates map 1:1 with layout points.
        self.set_margins(0, 0, 0)

        # Register custom fonts
        self.add_font("Stinger", "B", "fonts/StingerFitTrial-Bold.ttf")
        self.add_font("Noto", "", "fonts/NotoSans-Regular.ttf")
        self.add_font("Noto", "B", "fonts/NotoSans-Bold.ttf")
        self.add_font("Noto", "I", "fonts/NotoSans-Italic.ttf")

        self.add_page()
        self.set_font("Noto", size=12)


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

    # Consistent padding at top and bottom edges of header
    header_padding = 6

    # Account for school name font height so padding is from top of text, not baseline
    school_font_size = 26
    header_top = geom.height - header_padding - school_font_size

    pdf.set_fill_color(247, 248, 250)
    _rect(pdf, geom, 0, header_bottom, geom.width, header_height, style=RenderStyle.F)

    pdf.set_draw_color(185, 185, 185)
    pdf.set_line_width(1.2)
    _line(pdf, geom, geom.margin, header_bottom, geom.width - geom.margin, header_bottom)
    pdf.set_draw_color(0, 0, 0)

    # Title section - hierarchical flow (PaperGen style)
    # 1. School name (large, Stinger font)
    pdf.set_font("Stinger", "B", school_font_size)
    school_name = "St. Xavier's High School"
    school_width = pdf.get_string_width(school_name)
    school_x = (geom.width - school_width) / 2
    _text(pdf, geom, school_x, header_top, school_name)

    # 2. Exam name (small, Noto italic)
    pdf.set_font("Noto", "I", 14)
    exam_name = "Mid-Term Examination 2024"
    exam_width = pdf.get_string_width(exam_name)
    exam_x = (geom.width - exam_width) / 2
    _text(pdf, geom, exam_x, header_top - 22, exam_name)

    # 3. OMR Answer Sheet (medium, Noto regular)
    pdf.set_font("Noto", size=16)
    answer_sheet = "OMR Answer Sheet"
    answer_width = pdf.get_string_width(answer_sheet)
    answer_x = (geom.width - answer_width) / 2
    _text(pdf, geom, answer_x, header_top - 42, answer_sheet)

    # Draw horizontal line after OMR Answer Sheet
    line_y = header_top - 50
    pdf.set_draw_color(185, 185, 185)
    pdf.set_line_width(1.2)
    _line(pdf, geom, geom.margin, line_y, geom.width - geom.margin, line_y)
    pdf.set_draw_color(0, 0, 0)

    inner_width = geom.inner_width
    # Equal width sections with divider in the middle
    divider_x = geom.margin + inner_width / 2
    left_x = geom.margin
    left_width = inner_width / 2 - 8  # Small padding before divider
    right_x = divider_x + 8  # Small padding after divider

    content_top = header_top - 66
    line_height = 14

    # Calculate bottom boundary to match top padding
    content_font_size = 10
    content_bottom = header_bottom + header_padding + content_font_size * 0.3

    pdf.set_font("Noto", "B", 11)
    _text(pdf, geom, left_x, content_top, "Student Details")
    pdf.set_line_width(0.6)
    pdf.set_draw_color(200, 200, 200)

    pdf.set_font("Noto", size=content_font_size)
    # Add empty row after header
    detail_y = content_top - line_height * 2

    # Student Name field
    _text(pdf, geom, left_x, detail_y, "Student Name:")
    line_y = detail_y - 5
    _line(pdf, geom, left_x, line_y, left_x + left_width, line_y)
    detail_y -= line_height * 2  # Extra spacing for writing room

    # Class / Section field
    _text(pdf, geom, left_x, detail_y, "Class / Section:")
    line_y = detail_y - 5
    _line(pdf, geom, left_x, line_y, left_x + left_width, line_y)

    pdf.set_line_width(0.8)
    _line(pdf, geom, divider_x, content_bottom, divider_x, content_top + 4)

    pdf.set_line_width(1)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_font("Noto", "B", 11)
    _text(pdf, geom, right_x, content_top, "Instructions")

    pdf.set_font("Noto", size=10)
    instructions = (
        "Use blue or black pen to fill bubbles.",
        "Shade only one option for each question.",
        "Do not fold or staple the sheet.",
    )
    instr_y = content_top - line_height
    for instruction in instructions:
        _text(pdf, geom, right_x, instr_y, f"• {instruction}")
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
    bubble_groups: List[BubbleGroup],
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
    for group in bubble_groups:
        for bubble in group.bubbles:
            bubble_y_positions.add(bubble.y)

    # Filter positions that are too close to anchors (5-point clearance)
    clearance = 5
    valid_positions = sorted([
        y for y in bubble_y_positions
        if bottom_anchor_top + clearance <= y <= top_anchor_bottom - clearance
    ])

    # Draw grid markers at valid bubble row positions (rectangular, 2x width, growing inward)
    pdf.set_fill_color(0, 0, 0)
    marker_width = markers.grid_marker_size * 2
    marker_height = markers.grid_marker_size

    for y in valid_positions:
        # Left marker: starts at edge, extends right (inward)
        left_x = geom.margin / 2
        # Right marker: positioned to extend left (inward)
        right_x = geom.width - geom.margin / 2 - marker_width

        for x in (left_x, right_x):
            _rect(
                pdf,
                geom,
                x,
                y - marker_height / 2,  # Center marker on bubble row
                marker_width,
                marker_height,
                style=RenderStyle.F,
            )


def draw_unified_bubble_section(
    pdf: OMRPDF,
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers: MarkerConfig,
    groups: List[BubbleGroup],
    boxes: List[BoxCoordinate],
) -> None:
    """Draw all bubble groups with proper section labels, headers, and spacers."""
    if not groups:
        return

    # Get section specifications and calculate row allocations for drawing labels/headers
    section_specs = build_section_specifications(sheet)

    # Calculate content zone for row positions
    from omr_layout import calculate_content_zone
    content_top_y, content_bottom_y = calculate_content_zone(geom, markers, layout)
    top_y = content_top_y + layout.vertical_gap

    # Calculate row centers
    row_centers: List[float] = []
    row_index = 0
    while True:
        y = top_y - row_index * layout.vertical_gap
        if y - layout.radius <= content_bottom_y:
            break
        row_centers.append(y)
        row_index += 1

    # Calculate column widths (same logic as omr_layout.py)
    first_column_options = max(sheet.class_options, sheet.roll_columns, sheet.set_options)
    first_column_width = layout.group_width(first_column_options)
    other_column_width = layout.group_width(sheet.question_options)

    # Group bubbles by column and extract column origins
    columns: dict[int, List[BubbleGroup]] = {}
    column_origins: dict[int, float] = {}

    for group in groups:
        col_idx = group.column_index or 0
        columns.setdefault(col_idx, []).append(group)

        # Extract column origin from the first bubble in the group
        if col_idx not in column_origins and group.bubbles:
            # Calculate back from bubble x to column origin
            first_bubble_x = group.bubbles[0].x
            # Bubble x = column_origin + label_column_width + column_padding/2 + radius
            column_origin = first_bubble_x - layout.radius - layout.column_padding / 2 - layout.label_column_width
            column_origins[col_idx] = column_origin

    # Process each column
    for col_idx in sorted(columns.keys()):
        is_first_column = (col_idx == 0)
        column_origin = column_origins.get(col_idx, geom.margin)

        # Get row allocations for this column
        row_allocations, _ = allocate_column_rows(section_specs, row_centers, is_first_column)

        # Draw labels, headers, and boxes based on row allocations
        for allocation in row_allocations:
            x_base = column_origin + layout.label_column_width + layout.column_padding / 2

            if allocation.row_type == "label":
                # Draw section label
                section_labels = {"class": "Class", "class_section": "Class section", "roll": "Roll Number", "set": "Set", "question": "Questions"}
                label_text = section_labels.get(allocation.section_category, "")
                pdf.set_font("Noto", "B", size=11)
                _text(pdf, geom, x_base, allocation.y_position, label_text)

            elif allocation.row_type == "headers":
                # Draw option headers
                pdf.set_font("Noto", size=10)
                x_bubble_base = x_base + layout.radius

                # Determine headers based on section category
                if allocation.section_category == "class":
                    # Class headers: dynamically generated based on config
                    class_numbers = [6 + i for i in range(sheet.class_options)]
                    for opt_idx, class_num in enumerate(class_numbers):
                        x = x_bubble_base + opt_idx * (layout.diameter + layout.option_gap)
                        # Center single digit under bubble
                        _text(pdf, geom, x - 3.25, allocation.y_position, str(class_num))
                elif allocation.section_category == "class_section":
                    # Class section headers: a b c d (lowercase)
                    section_labels = [chr(ord("a") + i) for i in range(sheet.class_section_options)]
                    for opt_idx, label in enumerate(section_labels):
                        x = x_bubble_base + opt_idx * (layout.diameter + layout.option_gap)
                        # Center single character under bubble
                        _text(pdf, geom, x - 3.25, allocation.y_position, label)
                elif allocation.section_category == "set":
                    # Set headers: A B C D
                    num_options = sheet.set_options
                    ascii_offsets = [chr(ord("A") + opt) for opt in range(num_options)]
                    for opt in range(num_options):
                        x = x_bubble_base + opt * (layout.diameter + layout.option_gap)
                        _text(pdf, geom, x - 3.25, allocation.y_position, ascii_offsets[opt])
                else:  # questions
                    # Question headers: A B C D
                    num_options = sheet.question_options
                    ascii_offsets = [chr(ord("A") + opt) for opt in range(num_options)]
                    for opt in range(num_options):
                        x = x_bubble_base + opt * (layout.diameter + layout.option_gap)
                        _text(pdf, geom, x - 3.25, allocation.y_position, ascii_offsets[opt])

    # Draw write-in boxes for roll number
    if boxes:
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

    # Calculate y-ranges for each column to draw vertical lines
    column_y_ranges: dict[int, tuple[float, float]] = {}
    for group in groups:
        col = group.column_index or 0
        for bubble in group.bubbles:
            min_y = bubble.y - bubble.radius
            max_y = bubble.y + bubble.radius
            if col in column_y_ranges:
                column_y_ranges[col] = (
                    min(column_y_ranges[col][0], min_y),
                    max(column_y_ranges[col][1], max_y),
                )
            else:
                column_y_ranges[col] = (min_y, max_y)

    # Draw vertical lines on either side of label columns
    pdf.set_draw_color(192, 192, 192)
    pdf.set_line_width(1)
    for col, (min_y, max_y) in column_y_ranges.items():
        column_origin = column_origins.get(col, geom.margin)
        left_line_x = column_origin
        right_line_x = column_origin + layout.label_column_width
        _line(pdf, geom, left_line_x, min_y, left_line_x, max_y)
        _line(pdf, geom, right_line_x, min_y, right_line_x, max_y)
    pdf.set_draw_color(0, 0, 0)

    # Draw row labels and bubbles
    pdf.set_font("Noto", size=12)
    digit_width = 6.5
    gap_before_bubble = 5

    for group in groups:
        if not group.bubbles:
            continue

        col_idx = group.column_index or 0
        column_origin = column_origins.get(col_idx, geom.margin)
        label_end_x = column_origin + layout.label_column_width - gap_before_bubble

        # Draw row label
        current_text_width = len(group.display_label) * digit_width
        label_x = label_end_x - current_text_width
        label_y = group.bubbles[0].y - layout.radius / 2
        _text(pdf, geom, label_x, label_y, group.display_label)

        # Draw bubbles
        for bubble in group.bubbles:
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


def generate_omr_sheet(output_path: Path, sheet: SheetLayout | None = None) -> None:
    """Generate an OMR sheet PDF.

    Args:
        output_path: Path where PDF will be saved
        sheet: Sheet layout configuration (if None, uses default)
    """
    geom = PageGeometry()
    layout = BubbleLayout()
    markers = MarkerConfig()
    if sheet is None:
        sheet = SheetLayout()

    ensure_output_directory(output_path.parent)
    pdf = OMRPDF(geom)

    bubble_groups, roll_boxes, roll_bottom = generate_all_bubble_coordinates(geom, layout, sheet, markers)

    draw_header_section(pdf, geom)
    draw_anchor_markers(pdf, geom, markers)
    draw_grid_markers(pdf, geom, markers, bubble_groups)
    draw_unified_bubble_section(pdf, geom, layout, sheet, markers, bubble_groups, roll_boxes)

    pdf.output(str(output_path))


if __name__ == "__main__":
    import argparse
    import json
    from omr_config_loader import load_sheet_config

    parser = argparse.ArgumentParser(
        description="Generate OMR answer sheet PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python omr_sheet_generator_new.py                           # Generate with defaults
  python omr_sheet_generator_new.py --questions 50            # 50 questions
  python omr_sheet_generator_new.py --config midterm.json     # Use config file
  python omr_sheet_generator_new.py --config midterm.json --questions 60  # Config + override
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to JSON configuration file (e.g., sheet_config.json)"
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of questions (overrides config file if both specified)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="sheets/omr_sheet.pdf",
        metavar="PATH",
        help="Output PDF file path (default: sheets/omr_sheet.pdf)"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        # Load from config file
        try:
            sheet = load_sheet_config(args.config)
            print(f"Loaded configuration from: {args.config}")
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"Error loading config file: {e}")
            exit(1)

        # Override max_questions if specified on command line
        if args.questions is not None:
            from dataclasses import replace
            sheet = replace(sheet, max_questions=args.questions)
            print(f"Overriding max_questions to: {args.questions}")
    else:
        # Create sheet layout with max_questions if specified
        sheet = SheetLayout(max_questions=args.questions)

    target_path = Path(args.output)
    generate_omr_sheet(target_path, sheet)