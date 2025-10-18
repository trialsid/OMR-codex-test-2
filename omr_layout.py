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
            bubble_rows=1,  # Single row with all options horizontal
            has_label_row=True,
            has_spacer_after=True,
            has_special_row=True,  # Number headers (6 7 8 9 10) above bubbles
            special_row_position="before",
        ),
    ]

    # Add class section only if configured (skip when 0)
    if sheet.class_section_options > 0:
        specs.append(
            SectionSpec(
                category="class_section",
                label="Class section",
                bubble_rows=1,  # Single row with all options horizontal
                has_label_row=True,
                has_spacer_after=True,
                has_special_row=True,  # Option headers (a b c d) above bubbles
                special_row_position="before",
            )
        )

    specs.extend([
        SectionSpec(
            category="roll",
            label="Roll Number",
            bubble_rows=10,  # Always 10 rows for digits 0-9
            has_label_row=True,
            has_spacer_after=True,
            has_special_row=True,
            special_row_position="before",
        ),
        SectionSpec(
            category="set",
            label="Set",
            bubble_rows=1,  # Single row with all options horizontal
            has_label_row=True,
            has_spacer_after=True,
            has_special_row=True,  # Option headers (A B C D) above bubbles
            special_row_position="before",
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
    ])
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

    # Calculate column widths responsively
    # First column needs to fit all sections: Class, Class Section, Roll, Set, and Questions
    first_column_options = max(sheet.class_options, sheet.class_section_options, sheet.roll_columns, sheet.set_options, sheet.question_options)
    first_column_width = layout.group_width(first_column_options)

    # Subsequent columns only have Questions(4)
    other_column_width = layout.group_width(sheet.question_options)

    available_width = geom.width - 2 * geom.margin

    all_groups: List[BubbleGroup] = []
    all_boxes: List[BoxCoordinate] = []
    question_counter = 1

    # Track column positions dynamically
    column_positions: List[float] = []
    current_x = geom.margin

    # Add first column if it fits
    if first_column_width <= available_width:
        column_positions.append(current_x)
        current_x += first_column_width
    else:
        # If even first column doesn't fit, we can't create any layout
        print(f"Warning: First column width ({first_column_width:.1f}pt) exceeds available width ({available_width:.1f}pt)")
        return [], [], 0

    # Add subsequent columns while they fit
    while current_x + other_column_width <= geom.margin + available_width:
        column_positions.append(current_x)
        current_x += other_column_width

    # Process each column
    reached_max_questions = False
    for col_idx, column_origin in enumerate(column_positions):
        if reached_max_questions:
            break

        is_first_column = (col_idx == 0)

        # Allocate rows for this column
        row_allocations, question_rows_in_col = allocate_column_rows(
            section_specs, row_centers, is_first_column
        )

        # Generate bubble groups based on row allocations
        for allocation in row_allocations:
            if allocation.row_type == "bubbles":
                # Check if we've reached max_questions before creating more question groups
                if allocation.section_category == "question" and sheet.max_questions is not None:
                    if question_counter > sheet.max_questions:
                        reached_max_questions = True
                        break

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

    # Check if max_questions was requested but couldn't all fit
    actual_questions = question_counter - 1  # question_counter is 1-indexed
    if sheet.max_questions is not None and actual_questions < sheet.max_questions:
        print(f"Warning: Requested {sheet.max_questions} questions but only {actual_questions} fit in available space.")
        print(f"Generated sheet with {actual_questions} questions.")

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
        # Generate class numbers dynamically based on config (starting from 6)
        class_numbers = [6 + i for i in range(sheet.class_options)]
        display_label = ""  # No individual display label for horizontal layout
        group_index = 0  # Single group for all class options
        # Multiple bubbles horizontal
        for opt_idx, class_num in enumerate(class_numbers):
            x = x_base + opt_idx * (layout.diameter + layout.option_gap)
            bubbles.append(
                BubbleCoordinate(x=x, y=y, radius=layout.radius, label=str(class_num), index=opt_idx)
            )

    elif allocation.section_category == "class_section":
        # Generate class section labels dynamically based on config (lowercase a, b, c, ...)
        section_labels = [chr(ord("a") + i) for i in range(sheet.class_section_options)]
        display_label = ""  # No individual display label for horizontal layout
        group_index = 0  # Single group for all section options
        # Multiple bubbles horizontal
        for opt_idx, label in enumerate(section_labels):
            x = x_base + opt_idx * (layout.diameter + layout.option_gap)
            bubbles.append(
                BubbleCoordinate(x=x, y=y, radius=layout.radius, label=label, index=opt_idx)
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
        # Generate set labels dynamically based on config (A, B, C, ...)
        set_labels = [chr(ord("A") + i) for i in range(sheet.set_options)]
        display_label = ""  # No individual display label for horizontal layout
        group_index = 0  # Single group for all set options
        # Multiple bubbles horizontal
        for opt_idx, label in enumerate(set_labels):
            x = x_base + opt_idx * (layout.diameter + layout.option_gap)
            bubbles.append(
                BubbleCoordinate(x=x, y=y, radius=layout.radius, label=label, index=opt_idx)
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
