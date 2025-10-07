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
from typing import Callable, Dict, List, Optional, Tuple

from omr_config import PageGeometry, BubbleLayout, MarkerConfig, SheetLayout


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


def detect_bubbles(image: np.ndarray, layout: BubbleLayout) -> List[Bubble]:
    """Detect all bubbles in the corrected image.

    Args:
        image: Corrected grayscale or color image
        layout: Bubble layout configuration

    Returns:
        List of detected bubbles with fill information
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,
        param1=50,
        param2=30,
        minRadius=8,
        maxRadius=20
    )

    bubbles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
            # Analyze fill state
            is_filled, fill_intensity = analyze_bubble_fill(gray, x, y, r, layout.fill_threshold)
            bubbles.append(Bubble(
                x=x, y=y, radius=r,
                is_filled=is_filled,
                fill_intensity=fill_intensity
            ))

    # Sort bubbles top-to-bottom, left-to-right
    bubbles.sort()

    return bubbles


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


def detect_grid_markers(image: np.ndarray, geom: PageGeometry,
                        markers: MarkerConfig) -> Dict[str, List[Tuple[int, int]]]:
    """Detect vertical grid markers used to align bubble rows."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape
    scale_x = width / geom.width
    scale_y = height / geom.height

    expected_size = markers.grid_marker_size * (scale_x + scale_y) / 2
    min_area = (expected_size * 0.4) ** 2
    max_area = (expected_size * 1.8) ** 2

    left_markers: List[Tuple[int, int]] = []
    right_markers: List[Tuple[int, int]] = []
    edge_margin = int(geom.margin * scale_x * 1.2)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue
        aspect_ratio = float(w) / h
        if not (0.6 <= aspect_ratio <= 1.4):
            continue

        center = (x + w // 2, y + h // 2)
        if center[0] < edge_margin:
            left_markers.append(center)
        elif center[0] > width - edge_margin:
            right_markers.append(center)

    left_markers.sort(key=lambda p: p[1])
    right_markers.sort(key=lambda p: p[1])

    return {"left": left_markers, "right": right_markers}


def compute_grid_marker_rows(geom: PageGeometry, markers: MarkerConfig) -> List[float]:
    """Return the template y-positions of vertical grid markers."""

    start_y = geom.margin + markers.grid_spacing
    end_y = geom.height - geom.margin - markers.grid_spacing

    y_positions: List[float] = []
    current = start_y
    while current <= end_y + 1e-6:
        y_positions.append(current)
        current += markers.grid_spacing

    return y_positions


def build_coordinate_mappers(
    image: np.ndarray,
    geom: PageGeometry,
    markers: MarkerConfig,
    detected_markers: Dict[str, List[Tuple[int, int]]]
) -> Tuple[Callable[[float], int], Callable[[float], int]]:
    """Create conversion functions from PDF coordinates to image pixels."""

    height, width = image.shape[:2]
    scale_x = width / geom.width

    def map_x(x_value: float) -> int:
        return int(round(x_value * scale_x))

    left_markers = detected_markers.get("left", []) if detected_markers else []
    expected_rows = compute_grid_marker_rows(geom, markers)

    if left_markers and len(left_markers) >= 2:
        usable = min(len(left_markers), len(expected_rows))
        pdf_top = [geom.height - expected_rows[i] for i in range(usable)]
        image_y = [left_markers[i][1] for i in range(usable)]

        A = np.vstack([np.ones(usable), pdf_top]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, np.array(image_y, dtype=float), rcond=None)
        intercept, slope = coeffs.tolist()

        def map_y(y_value: float) -> int:
            top_origin = geom.height - y_value
            return int(round(intercept + slope * top_origin))

        return map_x, map_y

    scale_y = height / geom.height

    def fallback_map_y(y_value: float) -> int:
        top_origin = geom.height - y_value
        return int(round(top_origin * scale_y))

    return map_x, fallback_map_y


def compute_template_positions(
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
) -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    """Calculate bubble centers for roll rows and questions in template space."""

    left_padding = layout.column_padding / 2
    roll_x_start = geom.margin + layout.label_column_width + left_padding + layout.radius
    roll_top_y = geom.height - geom.margin - layout.diameter

    roll_positions: List[List[Tuple[float, float]]] = []
    for row in range(sheet.roll_rows):
        y = roll_top_y - (row + 1) * layout.vertical_gap
        row_positions: List[Tuple[float, float]] = []
        for col in range(sheet.roll_columns):
            x = roll_x_start + col * (layout.diameter + layout.option_gap)
            row_positions.append((x, y))
        roll_positions.append(row_positions)

    roll_bottom = roll_top_y - sheet.roll_rows * layout.vertical_gap - layout.radius

    options = sheet.question_options
    column_width = layout.group_width(options)
    question_x_start = geom.margin
    available_width = geom.width - geom.margin - question_x_start
    columns = max(1, int(available_width // column_width))

    row_centers: List[float] = []
    row_index = 1
    while True:
        y = roll_top_y - row_index * layout.vertical_gap
        if y - layout.radius <= geom.margin:
            break
        row_centers.append(y)
        row_index += 1

    first_column_start = next(
        (idx for idx, y in enumerate(row_centers) if y - layout.radius < roll_bottom),
        len(row_centers)
    )

    question_positions: List[List[Tuple[float, float]]] = []
    for col in range(columns):
        column_origin = question_x_start + col * column_width
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
            group: List[Tuple[float, float]] = []
            for opt in range(options):
                x = x_base + layout.radius + opt * (layout.diameter + layout.option_gap)
                group.append((x, y))
            question_positions.append(group)

    return roll_positions, question_positions


def map_detected_bubbles_to_template(
    image: np.ndarray,
    detected_bubbles: List[Bubble],
    roll_template: List[List[Tuple[float, float]]],
    question_template: List[List[Tuple[float, float]]],
    geom: PageGeometry,
    layout: BubbleLayout,
    markers: MarkerConfig,
    detected_markers: Dict[str, List[Tuple[int, int]]]
) -> Tuple[List[List[Bubble]], List[List[Bubble]]]:
    """Fuse circle detection with the template grid to recover missing bubbles."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    map_x, map_y = build_coordinate_mappers(image, geom, markers, detected_markers)

    scale_x = image.shape[1] / geom.width
    scale_y = image.shape[0] / geom.height
    template_radius = int(round(layout.radius * (scale_x + scale_y) / 2))
    tolerance = template_radius * 1.5

    used_indices: set[int] = set()

    def nearest_detected(x_pos: int, y_pos: int) -> Optional[Tuple[int, Bubble]]:
        best_idx: Optional[int] = None
        best_distance = tolerance
        for idx, bubble in enumerate(detected_bubbles):
            if idx in used_indices:
                continue
            distance = np.hypot(bubble.x - x_pos, bubble.y - y_pos)
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
        if best_idx is None:
            return None
        used_indices.add(best_idx)
        return best_idx, detected_bubbles[best_idx]

    def instantiate_bubble(x_pdf: float, y_pdf: float) -> Bubble:
        expected_x = map_x(x_pdf)
        expected_y = map_y(y_pdf)

        match = nearest_detected(expected_x, expected_y)
        if match is not None:
            _, detected = match
            center_x, center_y = detected.x, detected.y
            radius = detected.radius
        else:
            center_x, center_y = expected_x, expected_y
            radius = template_radius

        is_filled, fill_intensity = analyze_bubble_fill(gray, center_x, center_y, radius, layout.fill_threshold)
        return Bubble(x=center_x, y=center_y, radius=radius,
                      is_filled=is_filled, fill_intensity=fill_intensity)

    roll_groups: List[List[Bubble]] = []
    for row in roll_template:
        roll_groups.append([instantiate_bubble(x, y) for x, y in row])

    question_groups: List[List[Bubble]] = []
    for group in question_template:
        question_groups.append([instantiate_bubble(x, y) for x, y in group])

    return roll_groups, question_groups


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

    # Third pass: Label questions (A, B, C, D)
    option_labels = ['A', 'B', 'C', 'D']
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


def process_omr_sheet(input_path: Path, output_path: Path,
                      geom: PageGeometry, layout: BubbleLayout,
                      marker_config: MarkerConfig, sheet: SheetLayout) -> bool:
    """Process a single OMR sheet image.

    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        geom: Page geometry configuration
        layout: Bubble layout configuration
        marker_config: Marker layout configuration
        sheet: Sheet layout configuration

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
    marker_positions = detect_anchor_markers(image)
    if marker_positions is None or len(marker_positions) != 4:
        print("Failed to detect anchor markers")
        return False

    print(f"Detected {len(marker_positions)} anchor markers")

    # Correct skew
    corrected = correct_skew(image, marker_positions, geom)
    print("Applied perspective correction")

    # Detect bubbles via Hough transform
    bubbles = detect_bubbles(corrected, layout)
    print(f"Detected {len(bubbles)} candidate bubbles")

    # Detect grid markers to refine vertical alignment
    grid_markers = detect_grid_markers(corrected, geom, marker_config)
    print(f"Detected {len(grid_markers.get('left', []))} left grid markers and {len(grid_markers.get('right', []))} right grid markers")

    # Build template-aligned bubbles using configuration
    roll_template, question_template = compute_template_positions(geom, layout, sheet)
    roll_groups, question_groups = map_detected_bubbles_to_template(
        corrected, bubbles, roll_template, question_template, geom, layout, marker_config, grid_markers
    )
    print(f"Mapped {len(roll_groups)} roll number rows and {len(question_groups)} questions using template")

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
    markers = MarkerConfig()
    sheet = SheetLayout()

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
        success = process_omr_sheet(image_file, output_file, geom, layout, markers, sheet)
        print()


if __name__ == "__main__":
    main()
