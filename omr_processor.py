"""Process scanned OMR sheets to detect and label bubbles.

This script processes scanned OMR sheets by:
- Detecting anchor markers for alignment.
- Correcting skew using perspective transformation.
- Detecting all bubbles in the sheet.
- Identifying roll number bubbles vs question bubbles.
- Overlaying question numbers and option labels (A, B, C, D) on each bubble.

The processed images are saved to the ``processed/`` directory.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Tuple

from omr_config import PageGeometry, BubbleLayout, MarkerConfig, SheetLayout
from omr_layout import generate_all_bubble_coordinates, BubbleCoordinate


@dataclass
class Bubble:
    """Represents a detected bubble."""
    x: int
    y: int
    radius: int
    is_filled: bool = False
    fill_intensity: float = 0.0  # 0.0 = empty, 1.0 = completely filled

    def __lt__(self, other):
        """Sort by y first (top to bottom), then x (left to right)."""
        if abs(self.y - other.y) > 10:  # Same row tolerance
            return self.y < other.y
        return self.x < other.x


def detect_anchor_markers(image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """Detect the four corner anchor markers.

    Returns:
        List of (x, y) coordinates for [top-left, top-right, bottom-left, bottom-right]
        or None if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for square-like contours
    markers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100 or area > 5000:  # Filter by area
            continue

        # Check if it's square-like
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        if 0.7 < aspect_ratio < 1.3:  # Approximately square
            # Use center of bounding box
            markers.append((x + w // 2, y + h // 2))

    if len(markers) < 4:
        return None

    # Sort markers: top-left, top-right, bottom-left, bottom-right
    markers = sorted(markers, key=lambda p: p[1])  # Sort by y
    top_two = sorted(markers[:2], key=lambda p: p[0])  # Top row, sort by x
    bottom_two = sorted(markers[-2:], key=lambda p: p[0])  # Bottom row, sort by x

    return [top_two[0], top_two[1], bottom_two[0], bottom_two[1]]


def correct_skew(image: np.ndarray, markers: List[Tuple[int, int]], geom: PageGeometry) -> np.ndarray:
    """Apply perspective transformation to correct skew.

    Args:
        image: Input image
        markers: Corner markers [top-left, top-right, bottom-left, bottom-right]
        geom: Page geometry configuration

    Returns:
        Corrected image
    """
    # Calculate output dimensions based on A4 aspect ratio
    aspect_ratio = geom.width / geom.height
    output_height = 1400  # Target height in pixels
    output_width = int(output_height * aspect_ratio)

    # Source points (detected markers)
    src_points = np.float32(markers)

    # Destination points (corrected corners)
    dst_points = np.float32([
        [0, 0],
        [output_width - 1, 0],
        [0, output_height - 1],
        [output_width - 1, output_height - 1]
    ])

    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply transformation
    corrected = cv2.warpPerspective(image, matrix, (output_width, output_height))

    return corrected



def _refine_bubble_center(
    gray: np.ndarray,
    inner_x: int,
    inner_y: int,
    radius: int,
) -> Optional[Tuple[float, float, float]]:
    """Attempt to refine the bubble center using image moments."""

    if radius <= 0:
        return None

    search_radius = max(radius * 2, 8)
    x0 = max(inner_x - search_radius, 0)
    y0 = max(inner_y - search_radius, 0)
    x1 = min(inner_x + search_radius + 1, gray.shape[1])
    y1 = min(inner_y + search_radius + 1, gray.shape[0])

    if x1 <= x0 or y1 <= y0:
        return None

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    dark_pixels = cv2.countNonZero(thresh)
    if dark_pixels == 0:
        return None

    # Require a minimum fraction of the expected circle area to reduce noise
    expected_area = np.pi * (radius ** 2)
    if dark_pixels < expected_area * 0.1:
        return None

    moments = cv2.moments(thresh)
    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    refined_inner_x = x0 + cx
    refined_inner_y = y0 + cy
    refined_radius = max(1.0, np.sqrt(dark_pixels / np.pi))

    return refined_inner_x, refined_inner_y, refined_radius


def _coordinate_to_bubble(
    coord: BubbleCoordinate,
    geom: PageGeometry,
    markers: MarkerConfig,
    page_width: int,
    page_height: int,
    margin_x: int,
    margin_y: int,
    inner_gray: np.ndarray,
    layout: BubbleLayout,
    ) -> Optional[Tuple[Bubble, Optional[Tuple[float, float]]]]:
    """Convert a PDF coordinate to pixel coordinates and sample fill state."""
    # Calculate transformation from PDF to pixel space
    anchor_inset_x = geom.margin / 2.0 + markers.anchor_size / 2.0
    anchor_inset_y = geom.margin / 2.0 + markers.anchor_size / 2.0
    effective_width = geom.width - 2.0 * anchor_inset_x
    effective_height = geom.height - 2.0 * anchor_inset_y

    if effective_width <= 0 or effective_height <= 0:
        return None

    scale_x = page_width / effective_width

    # Transform PDF coordinates to relative position within effective area
    rel_x = (coord.x - anchor_inset_x) / effective_width
    rel_y = ((geom.height - anchor_inset_y) - coord.y) / effective_height

    # Clamp to valid range
    rel_x = min(max(rel_x, 0.0), 1.0)
    rel_y = min(max(rel_y, 0.0), 1.0)

    # Convert to absolute pixel coordinates
    abs_x = int(round(rel_x * page_width))
    abs_y = int(round(rel_y * page_height))
    abs_x = min(max(abs_x, 0), max(page_width - 1, 0))
    abs_y = min(max(abs_y, 0), max(page_height - 1, 0))

    # Convert to inner image coordinates (after margin crop)
    inner_x = abs_x - margin_x
    inner_y = abs_y - margin_y

    # Scale radius
    radius = max(1, int(round(coord.radius * scale_x)))

    # Check bounds
    if (
        inner_x < 0
        or inner_y < 0
        or inner_x >= inner_gray.shape[1]
        or inner_y >= inner_gray.shape[0]
    ):
        return None

    # Analyze fill state
    is_filled, intensity = analyze_bubble_fill(
        inner_gray,
        inner_x,
        inner_y,
        radius,
        layout.fill_threshold,
    )

    bubble = Bubble(
        x=abs_x,
        y=abs_y,
        radius=radius,
        is_filled=is_filled,
        fill_intensity=intensity,
    )

    refinement = _refine_bubble_center(inner_gray, inner_x, inner_y, radius)
    offset: Optional[Tuple[float, float]] = None
    if refinement is not None:
        refined_inner_x, refined_inner_y, refined_radius = refinement
        refined_abs_x = refined_inner_x + margin_x
        refined_abs_y = refined_inner_y + margin_y
        offset = (refined_abs_x - abs_x, refined_abs_y - abs_y)
        bubble.x = int(round(refined_abs_x))
        bubble.y = int(round(refined_abs_y))
        bubble.radius = max(1, int(round(refined_radius)))

    return bubble, offset


def sample_bubbles_from_coordinates(
    corrected: np.ndarray,
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers: MarkerConfig,
) -> Tuple[List[List[Bubble]], List[List[Bubble]]]:
    """Generate bubble coordinates procedurally and sample their fill states."""

    if corrected.size == 0:
        return [], []

    # Generate bubble coordinates using shared logic
    roll_coords, question_coords = generate_all_bubble_coordinates(geom, layout, sheet)

    height, width = corrected.shape[:2]
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if corrected.ndim == 3 else corrected

    # Calculate transformation parameters
    anchor_inset_x = geom.margin / 2.0 + markers.anchor_size / 2.0
    anchor_inset_y = geom.margin / 2.0 + markers.anchor_size / 2.0
    effective_width = geom.width - 2.0 * anchor_inset_x
    effective_height = geom.height - 2.0 * anchor_inset_y

    if effective_width <= 0 or effective_height <= 0:
        return [], []

    scale_x = width / effective_width
    scale_y = height / effective_height

    margin_after_crop_x = max(0.0, geom.margin - anchor_inset_x)
    margin_after_crop_y = max(0.0, geom.margin - anchor_inset_y)
    margin_x = int(round(margin_after_crop_x * scale_x))
    margin_y = int(round(margin_after_crop_y * scale_y))

    inner_gray = gray[margin_y: height - margin_y, margin_x: width - margin_x]
    if inner_gray.size == 0:
        return [[] for _ in range(sheet.roll_rows)], []

    def _median_offset(offsets: List[Tuple[float, float]]) -> Tuple[float, float]:
        if not offsets:
            return 0.0, 0.0
        arr = np.array(offsets, dtype=float)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))

    # Process roll bubbles
    roll_groups: List[List[Bubble]] = [[] for _ in range(sheet.roll_rows)]
    roll_offsets = defaultdict(list)
    roll_results: List[Tuple[BubbleCoordinate, Bubble, Optional[Tuple[float, float]]]] = []
    for coord in roll_coords:
        result = _coordinate_to_bubble(
            coord, geom, markers, width, height, margin_x, margin_y, inner_gray, layout
        )
        if result is None or coord.row is None:
            continue
        bubble, offset = result
        roll_results.append((coord, bubble, offset))
        if offset is not None:
            roll_offsets[coord.row].append(offset)

    roll_offset_cache = {
        row: _median_offset(offsets) for row, offsets in roll_offsets.items() if offsets
    }

    for coord, bubble, offset in roll_results:
        if offset is None:
            cached = roll_offset_cache.get(coord.row)
            if cached is not None:
                bubble.x = int(round(bubble.x + cached[0]))
                bubble.y = int(round(bubble.y + cached[1]))
        roll_groups[coord.row].append(bubble)

    for group in roll_groups:
        group.sort(key=lambda b: b.x)

    # Process question bubbles
    question_groups: List[List[Bubble]] = []
    question_map: dict[int, List[tuple[int, Bubble]]] = {}
    question_offsets_by_question = defaultdict(list)
    question_offsets_by_column = defaultdict(list)
    question_results: List[
        Tuple[BubbleCoordinate, Bubble, Optional[Tuple[float, float]]]
    ] = []

    for coord in question_coords:
        result = _coordinate_to_bubble(
            coord, geom, markers, width, height, margin_x, margin_y, inner_gray, layout
        )
        if (
            result is None
            or coord.question is None
            or coord.option_index is None
        ):
            continue
        bubble, offset = result
        question_results.append((coord, bubble, offset))
        if offset is not None:
            question_offsets_by_question[coord.question].append(offset)
            if coord.question_column is not None:
                question_offsets_by_column[coord.question_column].append(offset)
        question_map.setdefault(coord.question, []).append((coord.option_index, bubble))

    question_offset_cache = {
        key: _median_offset(offsets)
        for key, offsets in question_offsets_by_question.items()
        if offsets
    }
    question_column_cache = {
        key: _median_offset(offsets)
        for key, offsets in question_offsets_by_column.items()
        if offsets
    }

    # Apply cached offsets to bubbles without a reliable refinement
    for coord, bubble, offset in question_results:
        if offset is None:
            cached = None
            if coord.question is not None:
                cached = question_offset_cache.get(coord.question)
            if cached is None and coord.question_column is not None:
                cached = question_column_cache.get(coord.question_column)
            if cached is not None:
                bubble.x = int(round(bubble.x + cached[0]))
                bubble.y = int(round(bubble.y + cached[1]))

    for question_number in sorted(question_map):
        ordered = sorted(question_map[question_number], key=lambda item: item[0])
        question_groups.append([bubble for _, bubble in ordered])

    return roll_groups, question_groups

def analyze_bubble_fill(gray: np.ndarray, x: int, y: int, radius: int, threshold: float) -> Tuple[bool, float]:
    """Analyze whether a bubble is filled by comparing interior to background ring.

    Args:
        gray: Grayscale image
        x, y: Bubble center coordinates
        radius: Bubble radius
        threshold: Fill threshold (0.0 to 1.0)

    Returns:
        (is_filled, fill_intensity) tuple
    """
    h, w = gray.shape

    # Create masks for interior and background ring
    y_grid, x_grid = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)

    # Interior: pixels inside radius * 0.7 (avoid edge artifacts)
    interior_mask = distance_from_center <= (radius * 0.7)

    # Background ring: pixels in ring slightly outside the bubble
    ring_inner_radius = radius * 1.2
    ring_outer_radius = radius * 1.8
    background_mask = (distance_from_center >= ring_inner_radius) & (distance_from_center <= ring_outer_radius)

    # Calculate mean intensity (lower = darker = more filled)
    interior_pixels = gray[interior_mask]
    background_pixels = gray[background_mask]

    if len(interior_pixels) == 0 or len(background_pixels) == 0:
        return False, 0.0

    interior_mean = np.mean(interior_pixels)
    background_mean = np.mean(background_pixels)

    # Calculate darkness ratio: how much darker is interior compared to background
    # Normalize to 0-1 scale where 1.0 means interior is completely black
    if background_mean > 0:
        darkness_ratio = (background_mean - interior_mean) / background_mean
    else:
        darkness_ratio = 0.0

    # Clamp to [0, 1] range
    fill_intensity = max(0.0, min(1.0, darkness_ratio))
    is_filled = fill_intensity >= threshold

    return is_filled, fill_intensity


def overlay_labels(image: np.ndarray, roll_groups: List[List[Bubble]],
                   question_groups: List[List[Bubble]], sheet: SheetLayout) -> np.ndarray:
    """Overlay question numbers and option labels on bubbles.

    Args:
        image: Image to draw on
        roll_groups: Roll number bubble groups
        question_groups: Question bubble groups
        sheet: Sheet layout configuration

    Returns:
        Image with labels overlaid
    """
    output = image.copy()

    # First pass: Draw pink highlights for all filled bubbles
    pink_color = (255, 0, 255)  # Bright magenta/pink in BGR
    for row_bubbles in roll_groups:
        for bubble in row_bubbles:
            if bubble.is_filled:
                cv2.circle(output, (bubble.x, bubble.y), bubble.radius + 2, pink_color, 2)

    for question_bubbles in question_groups:
        for bubble in question_bubbles:
            if bubble.is_filled:
                cv2.circle(output, (bubble.x, bubble.y), bubble.radius + 2, pink_color, 2)

    # Second pass: Label roll numbers (digits 0-9 for each of 3 columns)
    for row_idx, row_bubbles in enumerate(roll_groups):
        digit = row_idx % 10
        for bubble in row_bubbles:
            # Draw digit inside bubble
            text = str(digit)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = bubble.x - text_size[0] // 2
            text_y = bubble.y + text_size[1] // 2
            cv2.putText(output, text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness)

    # Third pass: Label questions (A, B, C, ...)
    option_labels = [chr(ord('A') + idx) for idx in range(sheet.question_options)]
    for q_idx, question_bubbles in enumerate(question_groups):
        question_num = q_idx + 1

        for opt_idx, bubble in enumerate(question_bubbles):
            if opt_idx >= len(option_labels):
                continue

            # Draw question number above first bubble
            if opt_idx == 0:
                q_text = f"Q{question_num}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.35
                thickness = 1
                text_size = cv2.getTextSize(q_text, font, font_scale, thickness)[0]
                text_x = bubble.x - text_size[0] // 2
                text_y = bubble.y - bubble.radius - 5
                cv2.putText(output, q_text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

            # Draw option label inside bubble
            option_text = option_labels[opt_idx]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            text_size = cv2.getTextSize(option_text, font, font_scale, thickness)[0]
            text_x = bubble.x - text_size[0] // 2
            text_y = bubble.y + text_size[1] // 2
            cv2.putText(output, option_text, (text_x, text_y), font, font_scale, (0, 128, 0), thickness)

    return output


def process_omr_sheet(
    input_path: Path,
    output_path: Path,
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers_cfg: MarkerConfig,
) -> bool:
    """Process a single OMR sheet image.

    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        geom: Page geometry configuration
        layout: Bubble layout configuration
        sheet: Sheet layout configuration
        markers_cfg: Marker configuration shared with generator

    Returns:
        True if processing succeeded, False otherwise
    """
    # Read image
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Failed to read image: {input_path}")
        return False

    print(f"Processing {input_path.name}...")

    # Detect anchor markers
    anchor_points = detect_anchor_markers(image)
    if anchor_points is None or len(anchor_points) != 4:
        print("Failed to detect anchor markers")
        return False

    print(f"Detected {len(anchor_points)} anchor markers")

    # Correct skew
    corrected = correct_skew(image, anchor_points, geom)
    print("Applied perspective correction")

    # Sample bubbles at procedurally generated coordinates
    roll_groups, question_groups = sample_bubbles_from_coordinates(
        corrected,
        geom,
        layout,
        sheet,
        markers_cfg,
    )
    total_roll_bubbles = sum(len(group) for group in roll_groups)
    total_questions = len(question_groups)
    print(
        f"Sampled {total_roll_bubbles} roll bubbles "
        f"and {total_questions} questions"
    )

    if total_roll_bubbles == 0 and total_questions == 0:
        print("No bubble samples evaluated")
        return False

    # Overlay labels
    labeled = overlay_labels(corrected, roll_groups, question_groups, sheet)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), labeled)
    print(f"Saved processed image to {output_path}")

    return True


def main():
    """Main entry point."""
    geom = PageGeometry()
    layout = BubbleLayout()
    sheet = SheetLayout()
    markers_cfg = MarkerConfig()

    # Process all images in sheets/ directory
    sheets_dir = Path("sheets")
    processed_dir = Path("processed")

    if not sheets_dir.exists():
        print(f"Directory {sheets_dir} not found")
        return

    # Find image files (png, jpg, jpeg)
    image_files = list(sheets_dir.glob("*.png")) + \
                  list(sheets_dir.glob("*.jpg")) + \
                  list(sheets_dir.glob("*.jpeg"))

    if not image_files:
        print(f"No image files found in {sheets_dir}")
        return

    print(f"Found {len(image_files)} image(s) to process\n")

    for image_file in image_files:
        output_file = processed_dir / f"processed_{image_file.name}"
        success = process_omr_sheet(
            image_file,
            output_file,
            geom,
            layout,
            sheet,
            markers_cfg,
        )
        print()


if __name__ == "__main__":
    main()
