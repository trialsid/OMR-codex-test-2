"""Procedural generation of bubble coordinates for OMR sheets.

This module contains the shared logic for generating bubble positions,
used by both the generator (to draw PDF) and processor (to sample pixels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Optional

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

    category: Literal["class", "roll", "set", "question"]
    group_index: int  # Row index within category
    display_label: str  # Text rendered alongside the group
    bubbles: List[BubbleCoordinate]
    column_index: int | None = None  # Column index in the unified layout


@dataclass
class BoxCoordinate:
    """Represents a write-in box position in the roll number grid."""

    x: float  # Center x-coordinate in PDF points
    y: float  # Center y-coordinate in PDF points
    width: float  # Box width in PDF points
    height: float  # Box height in PDF points
    column: int  # Column index in the roll number grid


@dataclass
class SectionSpec:
    """Specification for a section's layout requirements."""

    category: Literal["class", "roll", "set", "question"]
    label: str  # Section header text
    bubble_rows: int  # Number of rows of bubbles
    has_label_row: bool = True  # Section has a label row
    has_spacer_after: bool = True  # Empty row after section
    has_special_row: bool = False  # Special row (boxes for roll, headers for questions)
    special_row_position: Literal["before", "after"] = "before"  # Position of special row relative to bubbles

    def total_rows(self) -> int:
        """Calculate total rows needed for this section."""
        rows = 0
        if self.has_label_row:
            rows += 1
        if self.has_special_row and self.special_row_position == "before":
            rows += 1
        rows += self.bubble_rows
        if self.has_special_row and self.special_row_position == "after":
            rows += 1
        if self.has_spacer_after:
            rows += 1
        return rows


@dataclass
class RowAllocation:
    """Represents the allocation of a specific row."""

    row_index: int  # Absolute row index in the column
    y_position: float  # Y-coordinate for this row
    row_type: Literal["label", "boxes", "headers", "bubbles", "spacer"]
    section_category: Optional[Literal["class", "roll", "set", "question"]] = None
    bubble_group_index: Optional[int] = None  # Index within the section (e.g., class 5 = index 0)


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


def build_section_specifications(sheet: SheetLayout) -> List[SectionSpec]:
    """Build section specifications for all sections in order."""
    specs = [
        SectionSpec(
            category="class",
            label="Class",
            bubble_rows=sheet.class_options,
            has_label_row=True,
            has_spacer_after=True,
            has_special_row=False,
        ),
        SectionSpec(
            category="roll",
            label="Roll Number",
            bubble_rows=sheet.roll_rows,
            has_label_row=True,
            has_spacer_after=True,
            has_special_row=True,
            special_row_position="before",
        ),
        SectionSpec(
            category="set",
            label="Set",
            bubble_rows=sheet.set_options,
            has_label_row=True,
            has_spacer_after=True,
            has_special_row=False,
        ),
        SectionSpec(
            category="question",
            label="Questions",
            bubble_rows=0,  # Will be calculated dynamically
            has_label_row=True,
            has_spacer_after=False,
            has_special_row=True,
            special_row_position="before",
        ),
    ]
    return specs


def allocate_column_rows(
    section_specs: List[SectionSpec],
    row_centers: List[float],
    is_first_column: bool,
) -> tuple[List[RowAllocation], int]:
    """Allocate rows for a column based on section specifications.

    Returns:
        - List of row allocations
        - Number of question rows that fit
    """
    allocations: List[RowAllocation] = []
    current_row = 0

    if is_first_column:
        # First column: all sections with labels and spacers
        for spec in section_specs:
            if current_row >= len(row_centers):
                break

            # Label row
            if spec.has_label_row:
                allocations.append(RowAllocation(
                    row_index=current_row,
                    y_position=row_centers[current_row],
                    row_type="label",
                    section_category=spec.category,
                ))
                current_row += 1

            # Special row before bubbles (boxes for roll, headers for questions)
            if spec.has_special_row and spec.special_row_position == "before":
                if current_row >= len(row_centers):
                    break
                row_type = "boxes" if spec.category == "roll" else "headers"
                allocations.append(RowAllocation(
                    row_index=current_row,
                    y_position=row_centers[current_row],
                    row_type=row_type,
                    section_category=spec.category,
                ))
                current_row += 1

            # Bubble rows
            if spec.category == "question":
                # Calculate how many question rows fit
                remaining_rows = len(row_centers) - current_row
                # Account for spacer if needed
                if spec.has_spacer_after:
                    remaining_rows -= 1
                question_rows = max(0, remaining_rows)

                for bubble_idx in range(question_rows):
                    if current_row >= len(row_centers):
                        break
                    allocations.append(RowAllocation(
                        row_index=current_row,
                        y_position=row_centers[current_row],
                        row_type="bubbles",
                        section_category="question",
                        bubble_group_index=bubble_idx,
                    ))
                    current_row += 1
                return allocations, question_rows
            else:
                # Fixed sections
                for bubble_idx in range(spec.bubble_rows):
                    if current_row >= len(row_centers):
                        break
                    allocations.append(RowAllocation(
                        row_index=current_row,
                        y_position=row_centers[current_row],
                        row_type="bubbles",
                        section_category=spec.category,
                        bubble_group_index=bubble_idx,
                    ))
                    current_row += 1

            # Spacer row
            if spec.has_spacer_after:
                if current_row >= len(row_centers):
                    break
                allocations.append(RowAllocation(
                    row_index=current_row,
                    y_position=row_centers[current_row],
                    row_type="spacer",
                    section_category=spec.category,
                ))
                current_row += 1
    else:
        # Subsequent columns: only questions with headers
        if current_row >= len(row_centers):
            return allocations, 0

        # Headers row
        allocations.append(RowAllocation(
            row_index=current_row,
            y_position=row_centers[current_row],
            row_type="headers",
            section_category="question",
        ))
        current_row += 1

        # Question bubbles
        question_rows = len(row_centers) - current_row
        for bubble_idx in range(question_rows):
            if current_row >= len(row_centers):
                break
            allocations.append(RowAllocation(
                row_index=current_row,
                y_position=row_centers[current_row],
                row_type="bubbles",
                section_category="question",
                bubble_group_index=bubble_idx,
            ))
            current_row += 1
        return allocations, question_rows

    return allocations, 0


def generate_class_bubble_groups(sheet: SheetLayout) -> List[BubbleGroup]:
    """Generate class bubble groups (classes 5-10)."""
    groups: List[BubbleGroup] = []
    for class_num in range(5, 11):  # 5, 6, 7, 8, 9, 10
        groups.append(
            BubbleGroup(
                category="class",
                group_index=class_num - 5,  # 0-5
                display_label=str(class_num),
                bubbles=[],  # Will be positioned later
                column_index=None,
            )
        )
    return groups


def generate_roll_bubble_groups_unified(sheet: SheetLayout) -> tuple[List[BubbleGroup], List[BoxCoordinate]]:
    """Generate roll number bubble groups (digits 0-9 with multiple columns)."""
    groups: List[BubbleGroup] = []

    # Create write-in box templates
    box_width = 13 * 1.3
    box_height = 13 * 1.2
    boxes: List[BoxCoordinate] = []
    for col in range(sheet.roll_columns):
        boxes.append(
            BoxCoordinate(x=0, y=0, width=box_width, height=box_height, column=col)
        )

    for row in range(sheet.roll_rows):
        digit = str(row % 10)
        groups.append(
            BubbleGroup(
                category="roll",
                group_index=row,
                display_label=digit,
                bubbles=[],  # Will be positioned later
                column_index=None,
            )
        )

    return groups, boxes


def generate_set_bubble_groups(sheet: SheetLayout) -> List[BubbleGroup]:
    """Generate set bubble groups (A, B, C, D)."""
    groups: List[BubbleGroup] = []
    set_labels = ["A", "B", "C", "D"]
    for idx in range(sheet.set_options):
        groups.append(
            BubbleGroup(
                category="set",
                group_index=idx,
                display_label=set_labels[idx],
                bubbles=[],  # Will be positioned later
                column_index=None,
            )
        )
    return groups


def generate_question_bubble_groups_unified(sheet: SheetLayout, count: int) -> List[BubbleGroup]:
    """Generate question bubble groups."""
    groups: List[BubbleGroup] = []
    for question_num in range(1, count + 1):
        groups.append(
            BubbleGroup(
                category="question",
                group_index=question_num,
                display_label=str(question_num),
                bubbles=[],  # Will be positioned later
                column_index=None,
            )
        )
    return groups


def position_bubble_group(
    group: BubbleGroup,
    column_origin: float,
    y: float,
    layout: BubbleLayout,
    sheet: SheetLayout,
    column_index: int,
) -> BubbleGroup:
    """Position bubbles for a group at specific coordinates."""
    x_base = column_origin + layout.label_column_width + layout.column_padding / 2 + layout.radius

    bubbles: List[BubbleCoordinate] = []

    if group.category == "class":
        # Single bubble for class selection
        bubbles.append(
            BubbleCoordinate(x=x_base, y=y, radius=layout.radius, label="•", index=0)
        )
    elif group.category == "roll":
        # Multiple columns for roll number
        digit = group.display_label
        for col in range(sheet.roll_columns):
            x = x_base + col * (layout.diameter + layout.option_gap)
            bubbles.append(
                BubbleCoordinate(x=x, y=y, radius=layout.radius, label=digit, index=col)
            )
    elif group.category == "set":
        # Single bubble for set selection
        bubbles.append(
            BubbleCoordinate(x=x_base, y=y, radius=layout.radius, label="•", index=0)
        )
    elif group.category == "question":
        # Multiple options (A, B, C, D)
        ascii_offsets = [chr(ord("A") + opt) for opt in range(sheet.question_options)]
        for opt in range(sheet.question_options):
            x = x_base + opt * (layout.diameter + layout.option_gap)
            bubbles.append(
                BubbleCoordinate(x=x, y=y, radius=layout.radius, label=ascii_offsets[opt], index=opt)
            )

    return BubbleGroup(
        category=group.category,
        group_index=group.group_index,
        display_label=group.display_label,
        bubbles=bubbles,
        column_index=column_index,
    )


def generate_all_bubble_coordinates(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers: MarkerConfig,
) -> tuple[List[BubbleGroup], List[BoxCoordinate], float]:
    """Generate all bubble groups using modular section specifications.

    Order: Class → Roll → Set → Questions, flowing through columns.
    """

    # Build section specifications
    section_specs = build_section_specifications(sheet)

    # Calculate content zone and available rows per column
    content_top_y, content_bottom_y = calculate_content_zone(geom, markers, layout)
    top_y = content_top_y + layout.vertical_gap

    # Calculate row centers (y-positions for each row)
    row_centers: List[float] = []
    row_index = 0
    while True:
        y = top_y - row_index * layout.vertical_gap
        if y - layout.radius <= content_bottom_y:
            break
        row_centers.append(y)
        row_index += 1

    if not row_centers:
        return [], [], 0

    # Calculate column width based on the widest section
    max_options = max(
        1,  # class (single option per row)
        sheet.roll_columns,
        1,  # set (single option per row)
        sheet.question_options
    )
    column_width = layout.group_width(max_options)
    available_width = geom.width - 2 * geom.margin
    num_columns = max(1, int(available_width // column_width))

    all_groups: List[BubbleGroup] = []
    all_boxes: List[BoxCoordinate] = []
    question_counter = 1

    # Process each column
    for col_idx in range(num_columns):
        is_first_column = (col_idx == 0)
        column_origin = geom.margin + col_idx * column_width

        # Allocate rows for this column
        row_allocations, question_rows_in_col = allocate_column_rows(
            section_specs, row_centers, is_first_column
        )

        # Generate bubble groups based on row allocations
        for allocation in row_allocations:
            if allocation.row_type == "bubbles":
                # Create bubble group
                group = create_bubble_group_from_allocation(
                    allocation, column_origin, layout, sheet, col_idx, question_counter
                )
                if group:
                    all_groups.append(group)
                    if allocation.section_category == "question":
                        question_counter += 1

            elif allocation.row_type == "boxes":
                # Create roll number write-in boxes
                boxes = create_boxes_from_allocation(
                    allocation, column_origin, layout, sheet
                )
                all_boxes.extend(boxes)

        # If no more questions fit in this column, stop generating columns
        if not is_first_column and question_rows_in_col == 0:
            break

    # Calculate roll_bottom for compatibility
    roll_bottom = 0
    for group in all_groups:
        if group.category == "roll":
            for bubble in group.bubbles:
                roll_bottom = min(roll_bottom, bubble.y - bubble.radius) if roll_bottom else bubble.y - bubble.radius

    return all_groups, all_boxes, roll_bottom


def create_bubble_group_from_allocation(
    allocation: RowAllocation,
    column_origin: float,
    layout: BubbleLayout,
    sheet: SheetLayout,
    column_index: int,
    question_counter: int,
) -> Optional[BubbleGroup]:
    """Create a bubble group from a row allocation."""
    if allocation.section_category is None or allocation.bubble_group_index is None:
        return None

    x_base = column_origin + layout.label_column_width + layout.column_padding / 2 + layout.radius
    y = allocation.y_position
    bubbles: List[BubbleCoordinate] = []

    # Determine group index and label
    if allocation.section_category == "class":
        class_num = 5 + allocation.bubble_group_index  # 5, 6, 7, 8, 9, 10
        display_label = str(class_num)
        group_index = allocation.bubble_group_index
        # Single bubble
        bubbles.append(
            BubbleCoordinate(x=x_base, y=y, radius=layout.radius, label="•", index=0)
        )

    elif allocation.section_category == "roll":
        digit = str(allocation.bubble_group_index % 10)
        display_label = digit
        group_index = allocation.bubble_group_index
        # Multiple columns for roll number
        for col in range(sheet.roll_columns):
            x = x_base + col * (layout.diameter + layout.option_gap)
            bubbles.append(
                BubbleCoordinate(x=x, y=y, radius=layout.radius, label=digit, index=col)
            )

    elif allocation.section_category == "set":
        set_labels = ["A", "B", "C", "D"]
        display_label = set_labels[allocation.bubble_group_index]
        group_index = allocation.bubble_group_index
        # Single bubble
        bubbles.append(
            BubbleCoordinate(x=x_base, y=y, radius=layout.radius, label="•", index=0)
        )

    elif allocation.section_category == "question":
        display_label = str(question_counter)
        group_index = question_counter
        # Multiple options (A, B, C, D)
        ascii_offsets = [chr(ord("A") + opt) for opt in range(sheet.question_options)]
        for opt in range(sheet.question_options):
            x = x_base + opt * (layout.diameter + layout.option_gap)
            bubbles.append(
                BubbleCoordinate(x=x, y=y, radius=layout.radius, label=ascii_offsets[opt], index=opt)
            )
    else:
        return None

    return BubbleGroup(
        category=allocation.section_category,
        group_index=group_index,
        display_label=display_label,
        bubbles=bubbles,
        column_index=column_index,
    )


def create_boxes_from_allocation(
    allocation: RowAllocation,
    column_origin: float,
    layout: BubbleLayout,
    sheet: SheetLayout,
) -> List[BoxCoordinate]:
    """Create write-in boxes for roll number from a row allocation."""
    if allocation.section_category != "roll":
        return []

    boxes: List[BoxCoordinate] = []
    x_base = column_origin + layout.label_column_width + layout.column_padding / 2 + layout.radius
    y = allocation.y_position
    box_width = layout.diameter * 1.3
    box_height = layout.diameter * 1.2

    for col in range(sheet.roll_columns):
        x = x_base + col * (layout.diameter + layout.option_gap)
        boxes.append(
            BoxCoordinate(x=x, y=y, width=box_width, height=box_height, column=col)
        )

    return boxes
