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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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



def load_bubble_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Load bubble placement metadata exported by the generator."""

    try:
        raw_text = metadata_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}") from exc

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse metadata file {metadata_path}: {exc}") from exc


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _entry_to_bubble(
    entry: Dict[str, Any],
    page_width: int,
    page_height: int,
    margin_x: int,
    margin_y: int,
    inner_gray: np.ndarray,
    layout: BubbleLayout,
) -> Optional[Bubble]:
    center = entry.get("center")
    if not isinstance(center, dict):
        return None

    try:
        norm_x = float(center["x"])
        norm_y = float(center["y"])
        norm_radius = float(entry["radius"])
    except (KeyError, TypeError, ValueError):
        return None

    abs_x = int(round(norm_x * page_width))
    abs_y = int(round(norm_y * page_height))
    inner_x = abs_x - margin_x
    inner_y = abs_y - margin_y
    radius = max(1, int(round(norm_radius * page_width)))

    if (
        inner_x < 0
        or inner_y < 0
        or inner_x >= inner_gray.shape[1]
        or inner_y >= inner_gray.shape[0]
    ):
        return None

    is_filled, intensity = analyze_bubble_fill(
        inner_gray,
        inner_x,
        inner_y,
        radius,
        layout.fill_threshold,
    )

    return Bubble(
        x=abs_x,
        y=abs_y,
        radius=radius,
        is_filled=is_filled,
        fill_intensity=intensity,
    )


def sample_bubbles_from_metadata(
    corrected: np.ndarray,
    metadata: Dict[str, Any],
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
) -> Tuple[List[List[Bubble]], List[List[Bubble]]]:
    """Evaluate bubble fill states at known coordinates."""

    if corrected.size == 0:
        return [], []

    height, width = corrected.shape[:2]
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if corrected.ndim == 3 else corrected

    scale_x = width / geom.width
    scale_y = height / geom.height
    margin_x = int(round(geom.margin * scale_x))
    margin_y = int(round(geom.margin * scale_y))

    inner_gray = gray[margin_y: height - margin_y, margin_x: width - margin_x]
    if inner_gray.size == 0:
        return [[] for _ in range(sheet.roll_rows)], []

    roll_groups: List[List[Bubble]] = [[] for _ in range(sheet.roll_rows)]
    roll_entries = metadata.get("roll_bubbles", [])
    if isinstance(roll_entries, list):
        for entry in roll_entries:
            bubble = _entry_to_bubble(entry, width, height, margin_x, margin_y, inner_gray, layout)
            if bubble is None:
                continue
            row_index = _safe_int(entry.get("row"))
            if row_index is None or not (0 <= row_index < sheet.roll_rows):
                continue
            roll_groups[row_index].append(bubble)

    for group in roll_groups:
        group.sort(key=lambda b: b.x)

    question_groups: List[List[Bubble]] = []
    question_entries = metadata.get("question_bubbles", [])
    if isinstance(question_entries, list):
        question_map: Dict[int, List[Tuple[int, Bubble]]] = {}
        for entry in question_entries:
            bubble = _entry_to_bubble(entry, width, height, margin_x, margin_y, inner_gray, layout)
            if bubble is None:
                continue
            question_number = _safe_int(entry.get("question"))
            option_index = _safe_int(entry.get("option_index"))
            if question_number is None or option_index is None:
                continue
            question_map.setdefault(question_number, []).append((option_index, bubble))

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
    metadata: Dict[str, Any],
) -> bool:
    """Process a single OMR sheet image.

    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        geom: Page geometry configuration
        layout: Bubble layout configuration
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
    markers = detect_anchor_markers(image)
    if markers is None or len(markers) != 4:
        print("Failed to detect anchor markers")
        return False

    print(f"Detected {len(markers)} anchor markers")

    # Correct skew
    corrected = correct_skew(image, markers, geom)
    print("Applied perspective correction")

    # Evaluate predefined bubble locations from metadata
    roll_groups, question_groups = sample_bubbles_from_metadata(corrected, metadata, geom, layout, sheet)
    total_roll_bubbles = sum(len(group) for group in roll_groups)
    total_questions = len(question_groups)
    print(
        f"Evaluated {total_roll_bubbles} roll bubbles "
        f"and {total_questions} questions using metadata"
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

    # Process all images in sheets/ directory
    sheets_dir = Path("sheets")
    processed_dir = Path("processed")

    if not sheets_dir.exists():
        print(f"Directory {sheets_dir} not found")
        return

    metadata_candidates = sorted(sheets_dir.glob("*.json"))
    if not metadata_candidates:
        print(f"No metadata JSON file found in {sheets_dir}")
        return

    metadata_path = metadata_candidates[0]
    print(f"Loading bubble metadata from {metadata_path.name}")
    metadata = load_bubble_metadata(metadata_path)

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
        success = process_omr_sheet(image_file, output_file, geom, layout, sheet, metadata)
        print()


if __name__ == "__main__":
    main()
