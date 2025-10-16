"""Procedural generation of bubble coordinates for OMR sheets.

This module contains the shared logic for generating bubble positions,
used by both the generator (to draw PDF) and processor (to sample pixels).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from omr_config import PageGeometry, BubbleLayout, MarkerConfig, SheetLayout


@dataclass
class BubbleCoordinate:
    """Represents a bubble's position and metadata."""
    x: float  # Absolute coordinate in PDF points
    y: float  # Absolute coordinate in PDF points
    radius: float  # Radius in PDF points

    # Roll bubble metadata (None for question bubbles)
    row: int | None = None
    column: int | None = None
    digit: int | None = None

    # Question bubble metadata (None for roll bubbles)
    question: int | None = None
    option_index: int | None = None
    question_column: int | None = None


@dataclass
class BoxCoordinate:
    """Represents a write-in box position in the roll number grid."""
    x: float  # Center x-coordinate in PDF points
    y: float  # Center y-coordinate in PDF points
    width: float  # Box width in PDF points
    height: float  # Box height in PDF points
    column: int  # Column index in the roll number grid


def generate_roll_bubble_coordinates(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers: MarkerConfig,
) -> tuple[List[BubbleCoordinate], List[BoxCoordinate], float, float, float]:
    """Generate coordinates for roll number bubbles and write-in boxes.

    Returns:
        (bubbles, boxes, top_y, area_width, bottom_y)
    """
    bubble_span = sheet.roll_columns * layout.diameter
    bubble_span += (sheet.roll_columns - 1) * layout.option_gap
    area_width = layout.label_column_width + layout.column_padding + bubble_span
    left_padding = layout.column_padding / 2
    x_start = geom.margin + layout.label_column_width + left_padding + layout.radius

    # Use centralized content zone calculation for consistency
    content_top_y, content_bottom_y = calculate_content_zone(geom, markers, layout)

    # The label appears at top_y - vertical_gap, so adjust top_y accordingly
    top_y = content_top_y + layout.vertical_gap

    # Generate write-in boxes at row 2 of the grid
    # Grid structure: row 0 = label, row 1 = spacer, row 2 = boxes, row 3 = spacer, row 4+ = bubbles
    box_width = layout.diameter * 1.3
    box_height = layout.diameter * 1.2
    box_y = top_y - 2 * layout.vertical_gap

    boxes = []
    for col in range(sheet.roll_columns):
        x = x_start + col * (layout.diameter + layout.option_gap)
        boxes.append(BoxCoordinate(
            x=x,
            y=box_y,
            width=box_width,
            height=box_height,
            column=col,
        ))

    # Generate bubbles at row 4+ of the grid
    bubbles = []
    for row in range(sheet.roll_rows):
        # Shift bubbles down by 4 rows to make space for label (1 row) and write-in boxes (1 row)
        # with proper spacing: row 0 = label header, row 1 = spacing, row 2 = boxes, row 3 = gap, row 4+ = bubbles
        y = top_y - (row + 4) * layout.vertical_gap
        for col in range(sheet.roll_columns):
            x = x_start + col * (layout.diameter + layout.option_gap)
            bubbles.append(BubbleCoordinate(
                x=x,
                y=y,
                radius=layout.radius,
                row=row,
                column=col,
                digit=row % 10,
            ))

    # Adjust bottom_y to account for the extra rows used by label and write-in boxes
    bottom_y = top_y - (sheet.roll_rows + 3) * layout.vertical_gap - layout.radius
    return bubbles, boxes, top_y, area_width, bottom_y


def generate_question_bubble_coordinates(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers: MarkerConfig,
    top_y: float,
    x_start: float,
    roll_bottom: float,
) -> List[BubbleCoordinate]:
    """Generate coordinates for question bubbles.

    Args:
        markers: Marker configuration for anchor-aware positioning
        top_y: Top y-coordinate to start from
        x_start: Left x-coordinate to start from
        roll_bottom: Bottom edge of roll number section (for overlap avoidance)
    """
    options = sheet.question_options
    column_width = layout.group_width(options)
    available_width = geom.width - geom.margin - x_start
    columns = max(1, int(available_width // column_width))

    # Use centralized content zone for bottom boundary
    content_top_y, content_bottom_y = calculate_content_zone(geom, markers, layout)

    # Determine all candidate row centers
    row_centers: List[float] = []
    row_index = 1
    while True:
        y = top_y - row_index * layout.vertical_gap
        if y - layout.radius <= content_bottom_y:
            break
        row_centers.append(y)
        row_index += 1

    # Find first row below roll number section
    first_column_start = next(
        (idx for idx, y in enumerate(row_centers) if y - layout.radius < roll_bottom),
        len(row_centers),
    )

    bubbles = []
    question_number = 1
    for col in range(columns):
        column_origin = x_start + col * column_width
        x_base = column_origin + layout.label_column_width + layout.column_padding / 2

        if col == 0:
            start_row = first_column_start + 2
            if start_row >= len(row_centers):
                continue
        else:
            start_row = 0
            if not row_centers:
                continue

        for y in row_centers[start_row:]:
            for opt in range(options):
                x = x_base + layout.radius + opt * (layout.diameter + layout.option_gap)
                bubbles.append(BubbleCoordinate(
                    x=x,
                    y=y,
                    radius=layout.radius,
                    question=question_number,
                    option_index=opt,
                    question_column=col,
                ))
            question_number += 1

    return bubbles


def calculate_roll_label_position(
    geom: PageGeometry,
    layout: BubbleLayout,
) -> tuple[float, float]:
    """Calculate the position for 'Roll Number' label.

    Returns:
        (x, y) coordinates for the label
    """
    # Position label horizontally to match Questions label positioning
    label_x = geom.margin + layout.label_column_width + layout.column_padding / 2

    # Label occupies the first row in the grid (row 0)
    # Position at row center Y coordinate (matching Questions label positioning)
    top_y = geom.header_bottom - layout.diameter
    label_y = top_y - layout.vertical_gap

    return label_x, label_y


def calculate_questions_label_position(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    roll_bottom: float,
    markers: MarkerConfig,
) -> tuple[float, float]:
    """Calculate the position for 'Questions' label above option headers.

    Args:
        roll_bottom: Bottom edge of roll number section (passed from generate_roll_bubble_coordinates)
        markers: Marker configuration for anchor-aware positioning

    Returns:
        (x, y) coordinates for the label, or (0, 0) if no valid position
    """
    # Use centralized content zone calculation for consistency
    content_top_y, content_bottom_y = calculate_content_zone(geom, markers, layout)

    # The label appears at top_y - vertical_gap, so adjust accordingly
    top_y = content_top_y + layout.vertical_gap

    # Generate row centers for questions
    row_centers: List[float] = []
    row_index = 1
    while True:
        y = top_y - row_index * layout.vertical_gap
        if y - layout.radius <= content_bottom_y:
            break
        row_centers.append(y)
        row_index += 1

    # Find where first column starts (avoiding roll bubbles)
    first_column_start = next(
        (idx for idx, y in enumerate(row_centers) if y - layout.radius < roll_bottom),
        len(row_centers),
    )

    # First question bubble in column 0 starts at first_column_start + 2
    first_question_row = first_column_start + 2
    if first_question_row >= len(row_centers):
        return 0, 0  # No valid position

    # Position label well above the option headers for clear separation
    # Headers are at topmost_y + 0.7 * vertical_gap
    # Label should be significantly higher to avoid overlap
    topmost_bubble_y = row_centers[first_question_row]
    label_x = geom.margin + layout.label_column_width + layout.column_padding / 2
    label_y = topmost_bubble_y + layout.vertical_gap * 1.6

    return label_x, label_y


def calculate_anchor_positions(
    geom: PageGeometry, markers: MarkerConfig
) -> Dict[str, Tuple[float, float]]:
    """Calculate anchor marker positions shared by generator and processor."""

    horizontal_inset = geom.margin / 2.0
    vertical_inset = geom.margin / 2.0

    left_x = horizontal_inset
    right_x = geom.width - horizontal_inset - markers.anchor_size

    header_bottom = geom.header_bottom
    top_y = max(vertical_inset, header_bottom - vertical_inset - markers.anchor_size)
    bottom_y = vertical_inset

    return {
        "top_left": (left_x, top_y),
        "top_right": (right_x, top_y),
        "bottom_left": (left_x, bottom_y),
        "bottom_right": (right_x, bottom_y),
    }


def calculate_anchor_centers(
    geom: PageGeometry, markers: MarkerConfig
) -> Dict[str, Tuple[float, float]]:
    """Return anchor marker center coordinates for geometric transforms."""

    positions = calculate_anchor_positions(geom, markers)
    half_size = markers.anchor_size / 2.0

    return {
        name: (x + half_size, y + half_size)
        for name, (x, y) in positions.items()
    }


def calculate_content_zone(
    geom: PageGeometry, markers: MarkerConfig, layout: BubbleLayout
) -> Tuple[float, float]:
    """Calculate the content zone boundaries based on anchor positions.

    This ensures symmetric spacing and that content stays within anchor bounds.
    Top spacing accounts for column headers that extend above bubbles.

    Returns:
        (top_y, bottom_y) - Y-coordinates defining the content zone
            top_y: Maximum Y where content can start
            bottom_y: Minimum Y where content can end
    """
    # Calculate anchor boundaries
    vertical_inset = geom.margin / 2.0

    # Top anchor bottom edge
    top_anchor_bottom = max(vertical_inset, geom.header_bottom - vertical_inset - markers.anchor_size)

    # Bottom anchor top edge
    bottom_anchor_top = vertical_inset + markers.anchor_size

    # Top spacing must account for column headers extending above bubbles
    # Headers are positioned at bubble_y + 0.7 * vertical_gap
    top_spacing = 8 + int(0.7 * layout.vertical_gap) + 2  # 8 base + header height + 2 buffer
    bottom_spacing = 8  # Keep symmetric spacing for bottom

    content_top_y = top_anchor_bottom - top_spacing
    content_bottom_y = bottom_anchor_top + bottom_spacing

    return content_top_y, content_bottom_y


def generate_all_bubble_coordinates(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers: MarkerConfig,
) -> tuple[List[BubbleCoordinate], List[BoxCoordinate], List[BubbleCoordinate], float]:
    """Generate all bubble coordinates and box coordinates for an OMR sheet.

    Returns:
        (roll_bubbles, roll_boxes, question_bubbles, roll_bottom)
    """
    roll_bubbles, roll_boxes, roll_top, _, roll_bottom = generate_roll_bubble_coordinates(
        geom, layout, sheet, markers
    )

    question_x_start = geom.margin
    question_bubbles = generate_question_bubble_coordinates(
        geom, layout, sheet, markers, roll_top, question_x_start, roll_bottom
    )

    return roll_bubbles, roll_boxes, question_bubbles, roll_bottom
