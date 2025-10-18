"""Shared configuration for OMR sheet generation and processing.

This module contains all layout parameters used by both the generator and processor
to ensure they remain synchronized.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PageGeometry:
    """Page dimensions, margins, and header sizing."""

    width: float = 595.276  # A4 width in points
    height: float = 841.89  # A4 height in points
    margin: float = 36
    header_ratio: float = 0.2  # 20% of the page reserved for the header

    @property
    def inner_width(self) -> float:
        return self.width - 2 * self.margin

    @property
    def inner_height(self) -> float:
        return self.height - 2 * self.margin

    @property
    def header_height(self) -> float:
        """Height of the header section in points."""

        return self.height * self.header_ratio

    @property
    def header_bottom(self) -> float:
        """Y coordinate marking the bottom edge of the header."""

        return self.height - self.header_height


@dataclass(frozen=True)
class BubbleLayout:
    """Bubble dimensions and spacing."""
    radius: float = 6.5
    vertical_gap: float = 19
    option_gap: float = 9
    column_padding: float = 18
    label_column_width: float = 28
    fill_threshold: float = 0.4  # Minimum darkness ratio to consider bubble filled

    @property
    def diameter(self) -> float:
        return 2 * self.radius

    def group_width(self, options: int) -> float:
        """Calculate width of a bubble group."""
        bubble_span = options * self.diameter
        gap_span = (options - 1) * self.option_gap if options > 1 else 0
        return self.label_column_width + self.column_padding + bubble_span + gap_span


@dataclass(frozen=True)
class MarkerConfig:
    """Configuration for anchor and grid markers."""
    anchor_size: float = 20
    grid_marker_size: float = 6
    grid_spacing: float = 42


@dataclass(frozen=True)
class SheetLayout:
    """Complete OMR sheet layout configuration."""
    # Class section (first in column)
    class_options: int = 5  # Classes 6-10

    # Class section/division (second in column)
    class_section_options: int = 3  # Class sections (a, b, c)

    # Roll number section (third in column)
    # Note: Roll numbers always have 10 rows (digits 0-9)
    roll_columns: int = 3

    # Set section (fourth in column)
    set_options: int = 4  # Sets A-D

    # Question section (fills remaining rows)
    question_options: int = 4
    max_questions: int | None = None  # Maximum number of questions (None = fill available space)

    def __post_init__(self):
        """Validate configuration."""
        assert self.class_options > 0, "Class options must be positive"
        assert self.class_section_options >= 0, "Class section options must be non-negative"
        assert self.roll_columns > 0, "Roll columns must be positive"
        assert self.set_options > 0, "Set options must be positive"
        assert self.question_options > 0, "Question options must be positive"
        if self.max_questions is not None:
            assert self.max_questions > 0, "Max questions must be positive if specified"


# Default configuration instance
DEFAULT_CONFIG = {
    'geometry': PageGeometry(),
    'layout': BubbleLayout(),
    'markers': MarkerConfig(),
    'sheet': SheetLayout(),
}
