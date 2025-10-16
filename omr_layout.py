"""Procedural generation of bubble coordinates for OMR sheets.

This module contains the shared logic for generating bubble positions,
used by both the generator (to draw PDF) and processor (to sample pixels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from omr_config import PageGeometry, BubbleLayout, MarkerConfig, SheetLayout


@dataclass
class BubbleCoordinate:
    """Represents a bubble's position, ordering and label."""

    x: float  # Absolute coordinate in PDF points
    y: float  # Absolute coordinate in PDF points
    radius: float  # Radius in PDF points
    label: str  # Textual label rendered inside the bubble
    index: int  # Order within the group (column/option index)


@dataclass
class BubbleGroup:
    """Collection of bubbles that share a label and behaviour."""

    category: Literal["roll", "question"]
    group_index: int  # Row index for roll, question number for questions
    display_label: str  # Text rendered alongside the group (digit or question number)
    bubbles: List[BubbleCoordinate]
    column_index: int | None = None  # Question column, None for roll rows


@dataclass
class BoxCoordinate:
    """Represents a write-in box position in the roll number grid."""

    x: float  # Center x-coordinate in PDF points
    y: float  # Center y-coordinate in PDF points
    width: float  # Box width in PDF points
    height: float  # Box height in PDF points
    column: int  # Column index in the roll number grid


def generate_roll_bubble_groups(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers: MarkerConfig,
) -> tuple[List[BubbleGroup], List[BoxCoordinate], float, float]:
    """Generate roll number bubble groups and write-in boxes."""

    bubble_span = sheet.roll_columns * layout.diameter
    bubble_span += (sheet.roll_columns - 1) * layout.option_gap
    left_padding = layout.column_padding / 2
    x_start = geom.margin + layout.label_column_width + left_padding + layout.radius

    # Use centralized content zone calculation for consistency
    content_top_y, _ = calculate_content_zone(geom, markers, layout)

    # The label appears at top_y - vertical_gap, so adjust top_y accordingly
    top_y = content_top_y + layout.vertical_gap

    # Generate write-in boxes at row 2 of the grid
    # Grid structure: row 0 = label, row 1 = spacer, row 2 = boxes, row 3 = spacer, row 4+ = bubbles
    box_width = layout.diameter * 1.3
    box_height = layout.diameter * 1.2
    box_y = top_y - 2 * layout.vertical_gap

    boxes: List[BoxCoordinate] = []
    for col in range(sheet.roll_columns):
        x = x_start + col * (layout.diameter + layout.option_gap)
        boxes.append(
            BoxCoordinate(
                x=x,
                y=box_y,
                width=box_width,
                height=box_height,
                column=col,
            )
        )

    groups: List[BubbleGroup] = []
    for row in range(sheet.roll_rows):
        y = top_y - (row + 4) * layout.vertical_gap
        digit = str(row % 10)
        bubbles = [
            BubbleCoordinate(
                x=x_start + col * (layout.diameter + layout.option_gap),
                y=y,
                radius=layout.radius,
                label=digit,
                index=col,
            )
            for col in range(sheet.roll_columns)
        ]
        groups.append(
            BubbleGroup(
                category="roll",
                group_index=row,
                display_label=digit,
                bubbles=bubbles,
                column_index=None,
            )
        )

    bottom_y = top_y - (sheet.roll_rows + 3) * layout.vertical_gap - layout.radius
    return groups, boxes, top_y, bottom_y


def generate_question_bubble_groups(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers: MarkerConfig,
    roll_bottom: float,
) -> List[BubbleGroup]:
    """Generate question bubble groups."""

    options = sheet.question_options
    column_width = layout.group_width(options)
    available_width = geom.width - geom.margin - geom.margin
    columns = max(1, int(available_width // column_width))

    # Use centralized content zone for bottom boundary
    content_top_y, content_bottom_y = calculate_content_zone(geom, markers, layout)
    top_y = content_top_y + layout.vertical_gap

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

    groups: List[BubbleGroup] = []
    question_number = 1
    ascii_offsets = [chr(ord("A") + opt) for opt in range(options)]

    for col in range(columns):
        column_origin = geom.margin + col * column_width
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
            bubbles = [
                BubbleCoordinate(
                    x=x_base + layout.radius + opt * (layout.diameter + layout.option_gap),
                    y=y,
                    radius=layout.radius,
                    label=ascii_offsets[opt],
                    index=opt,
                )
                for opt in range(options)
            ]
            groups.append(
                BubbleGroup(
                    category="question",
                    group_index=question_number,
                    display_label=str(question_number),
                    bubbles=bubbles,
                    column_index=col,
                )
            )
            question_number += 1

    return groups


def calculate_roll_label_position(
    geom: PageGeometry,
    layout: BubbleLayout,
) -> tuple[float, float]:
    """Calculate the position for 'Roll Number' label."""

    label_x = geom.margin + layout.label_column_width + layout.column_padding / 2

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
    """Calculate the position for 'Questions' label above option headers."""

    content_top_y, content_bottom_y = calculate_content_zone(geom, markers, layout)

    top_y = content_top_y + layout.vertical_gap

    row_centers: List[float] = []
    row_index = 1
    while True:
        y = top_y - row_index * layout.vertical_gap
        if y - layout.radius <= content_bottom_y:
            break
        row_centers.append(y)
        row_index += 1

    first_column_start = next(
        (idx for idx, y in enumerate(row_centers) if y - layout.radius < roll_bottom),
        len(row_centers),
    )

    first_question_row = first_column_start + 2
    if first_question_row >= len(row_centers):
        return 0, 0

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
    """Calculate the content zone boundaries based on anchor positions."""

    vertical_inset = geom.margin / 2.0

    top_anchor_bottom = max(vertical_inset, geom.header_bottom - vertical_inset - markers.anchor_size)

    bottom_anchor_top = vertical_inset + markers.anchor_size

    top_spacing = 8 + int(0.7 * layout.vertical_gap) + 2
    bottom_spacing = 8

    content_top_y = top_anchor_bottom - top_spacing
    content_bottom_y = bottom_anchor_top + bottom_spacing

    return content_top_y, content_bottom_y


def generate_all_bubble_coordinates(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers: MarkerConfig,
) -> tuple[List[BubbleGroup], List[BoxCoordinate], float]:
    """Generate all bubble groups and write-in boxes for an OMR sheet."""

    roll_groups, boxes, _, roll_bottom = generate_roll_bubble_groups(
        geom, layout, sheet, markers
    )
    question_groups = generate_question_bubble_groups(
        geom, layout, sheet, markers, roll_bottom
    )

    return roll_groups + question_groups, boxes, roll_bottom
