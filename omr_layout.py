"""Procedural generation of bubble coordinates for OMR sheets.

This module contains the shared logic for generating bubble positions,
used by both the generator (to draw PDF) and processor (to sample pixels).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

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


def generate_roll_bubble_coordinates(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
) -> tuple[List[BubbleCoordinate], float, float, float]:
    """Generate coordinates for roll number bubbles.

    Returns:
        (bubbles, top_y, area_width, bottom_y)
    """
    bubble_span = sheet.roll_columns * layout.diameter
    bubble_span += (sheet.roll_columns - 1) * layout.option_gap
    area_width = layout.label_column_width + layout.column_padding + bubble_span
    left_padding = layout.column_padding / 2
    x_start = geom.margin + layout.label_column_width + left_padding + layout.radius
    top_y = geom.height - geom.margin - layout.diameter

    bubbles = []
    for row in range(sheet.roll_rows):
        y = top_y - (row + 1) * layout.vertical_gap
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

    bottom_y = top_y - sheet.roll_rows * layout.vertical_gap - layout.radius
    return bubbles, top_y, area_width, bottom_y


def generate_question_bubble_coordinates(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    top_y: float,
    x_start: float,
    roll_bottom: float,
) -> List[BubbleCoordinate]:
    """Generate coordinates for question bubbles.

    Args:
        top_y: Top y-coordinate to start from
        x_start: Left x-coordinate to start from
        roll_bottom: Bottom edge of roll number section (for overlap avoidance)
    """
    options = sheet.question_options
    column_width = layout.group_width(options)
    available_width = geom.width - geom.margin - x_start
    columns = max(1, int(available_width // column_width))

    # Determine all candidate row centers
    row_centers: List[float] = []
    row_index = 1
    while True:
        y = top_y - row_index * layout.vertical_gap
        if y - layout.radius <= geom.margin:
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
    left_padding = layout.column_padding / 2
    label_x = geom.margin + layout.label_column_width + left_padding

    # Label sits above the first bubble row
    top_y = geom.height - geom.margin - layout.diameter
    label_y = top_y + layout.radius / 2

    return label_x, label_y


def calculate_questions_label_position(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
) -> tuple[float, float]:
    """Calculate the position for 'Questions' label.

    Returns:
        (x, y) coordinates for the label, or (0, 0) if no valid position
    """
    # Recalculate the geometry to find the label row
    top_y = geom.height - geom.margin - layout.diameter
    roll_bottom = top_y - sheet.roll_rows * layout.vertical_gap - layout.radius

    # Generate row centers for questions
    row_centers: List[float] = []
    row_index = 1
    while True:
        y = top_y - row_index * layout.vertical_gap
        if y - layout.radius <= geom.margin:
            break
        row_centers.append(y)
        row_index += 1

    # Find where first column starts (avoiding roll bubbles)
    first_column_start = next(
        (idx for idx, y in enumerate(row_centers) if y - layout.radius < roll_bottom),
        len(row_centers),
    )

    # Label goes at the row after first_column_start
    label_row_index = first_column_start + 1
    if label_row_index >= len(row_centers):
        return 0, 0  # No valid position

    label_x = geom.margin + layout.label_column_width + layout.column_padding / 2
    label_y = row_centers[label_row_index]

    return label_x, label_y


def generate_all_bubble_coordinates(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
) -> tuple[List[BubbleCoordinate], List[BubbleCoordinate]]:
    """Generate all bubble coordinates for an OMR sheet.

    Returns:
        (roll_bubbles, question_bubbles)
    """
    roll_bubbles, roll_top, _, roll_bottom = generate_roll_bubble_coordinates(
        geom, layout, sheet
    )

    question_x_start = geom.margin
    question_bubbles = generate_question_bubble_coordinates(
        geom, layout, sheet, roll_top, question_x_start, roll_bottom
    )

    return roll_bubbles, question_bubbles
