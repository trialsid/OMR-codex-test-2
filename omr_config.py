"""Shared configuration for OMR sheet generation and processing.

This module contains all layout parameters used by both the generator and processor
to ensure they remain synchronized.
"""
from dataclasses import dataclass, field
from typing import List


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
    grid_calibration_enabled: bool = True
    grid_calibration_min_fraction: float = 0.45
    grid_calibration_min_matches: int = 12
    grid_marker_distance_limit: float = 0.5
    grid_marker_area_tolerance: float = 0.65
    grid_marker_aspect_tolerance: float = 0.45
    grid_marker_outlier_sigma: float = 3.5
    grid_marker_scale_tolerance: float = 0.35


@dataclass(frozen=True)
class AnchorDetectionZones:
    """Configuration for anchor marker detection zones.

    These define the corner regions where anchor markers are expected to be found.
    Detection uses a two-pass approach:
    - First pass: Uses strict zones (corner_expand = 0.0)
    - Second pass: Uses relaxed zones (corner_expand = relaxed_expansion)
    """
    # Base zone dimensions (as fraction of image dimensions)
    corner_band_x_ratio: float = 0.25      # 25% from left/right edges
    corner_band_y_top_ratio: float = 0.35  # 35% from top (accounts for 20% header)
    corner_band_y_bottom_ratio: float = 0.20  # 20% from bottom

    # Relaxed zone expansion (added to base ratios in second pass)
    relaxed_expansion: float = 0.1  # 10% expansion for second pass

    def get_zones(self, img_width: int, img_height: int, corner_expand: float = 0.0):
        """Calculate zone boundaries in pixels.

        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels
            corner_expand: Expansion factor (0.0 for strict, relaxed_expansion for relaxed)

        Returns:
            Tuple of (corner_band_x, corner_band_y_top, corner_band_y_bottom) in pixels
        """
        corner_band_x = img_width * (self.corner_band_x_ratio + corner_expand)
        corner_band_y_top = img_height * (self.corner_band_y_top_ratio + corner_expand)
        corner_band_y_bottom = img_height * (self.corner_band_y_bottom_ratio + corner_expand)
        return corner_band_x, corner_band_y_top, corner_band_y_bottom


@dataclass(frozen=True)
class QuestionOptionRange:
    """Defines a range of questions with a specific number of options."""
    start: int
    end: int
    options: int

    def __post_init__(self):
        """Validate range."""
        assert self.start > 0, "Question start must be positive"
        assert self.end >= self.start, "Question end must be >= start"
        assert self.options > 0, "Question options must be positive"


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
    question_option_ranges: List[QuestionOptionRange] = field(
        default_factory=lambda: [QuestionOptionRange(start=1, end=50, options=4)]
    )

    def __post_init__(self):
        """Validate configuration."""
        assert self.class_options > 0, "Class options must be positive"
        assert self.class_section_options >= 0, "Class section options must be non-negative"
        assert self.roll_columns > 0, "Roll columns must be positive"
        assert self.set_options > 0, "Set options must be positive"

        # Validate question ranges
        if self.question_option_ranges:
            # Check for gaps and overlaps
            sorted_ranges = sorted(self.question_option_ranges, key=lambda r: r.start)
            for i, range_obj in enumerate(sorted_ranges):
                if i == 0:
                    assert range_obj.start == 1, "First question range must start at 1"
                else:
                    prev_range = sorted_ranges[i - 1]
                    assert range_obj.start == prev_range.end + 1, \
                        f"Question ranges must be contiguous: gap between {prev_range.end} and {range_obj.start}"

    @property
    def max_questions(self) -> int | None:
        """Derive maximum number of questions from ranges."""
        if not self.question_option_ranges:
            return None
        return max(r.end for r in self.question_option_ranges)

    def get_question_options(self, question_number: int) -> int | None:
        """Get the number of options for a specific question number.

        Args:
            question_number: The question number to query

        Returns:
            Number of options for that question, or None if not in any range
        """
        if not self.question_option_ranges:
            return None
        for range_obj in self.question_option_ranges:
            if range_obj.start <= question_number <= range_obj.end:
                return range_obj.options
        return None


# Default configuration instance
DEFAULT_CONFIG = {
    'geometry': PageGeometry(),
    'layout': BubbleLayout(),
    'markers': MarkerConfig(),
    'sheet': SheetLayout(),
    'anchor_zones': AnchorDetectionZones(),
}
