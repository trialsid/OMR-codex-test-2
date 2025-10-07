"""Shared configuration for OMR sheet generation and processing.

This module contains all layout parameters used by both the generator and processor
to ensure they remain synchronized.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PageGeometry:
    """Page dimensions and margins."""
    width: float = 595.276  # A4 width in points
    height: float = 841.89  # A4 height in points
    margin: float = 36

    @property
    def inner_width(self) -> float:
        return self.width - 2 * self.margin

    @property
    def inner_height(self) -> float:
        return self.height - 2 * self.margin


@dataclass(frozen=True)
class BubbleLayout:
    """Bubble dimensions and spacing."""
    radius: float = 8
    vertical_gap: float = 26
    option_gap: float = 12
    column_padding: float = 24
    label_column_width: float = 28

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
    anchor_size: float = 18
    grid_marker_size: float = 6
    grid_spacing: float = 42


@dataclass(frozen=True)
class SheetLayout:
    """Complete OMR sheet layout configuration."""
    # Roll number section
    roll_columns: int = 3
    roll_rows: int = 10

    # Question section
    question_options: int = 4

    def __post_init__(self):
        """Validate configuration."""
        assert self.roll_columns > 0, "Roll columns must be positive"
        assert self.roll_rows > 0, "Roll rows must be positive"
        assert self.question_options > 0, "Question options must be positive"


# Default configuration instance
DEFAULT_CONFIG = {
    'geometry': PageGeometry(),
    'layout': BubbleLayout(),
    'markers': MarkerConfig(),
    'sheet': SheetLayout(),
}
