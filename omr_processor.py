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


def detect_anchor_markers(
    image: np.ndarray,
    geom: PageGeometry,
    markers_cfg: MarkerConfig
) -> Optional[List[Tuple[int, int]]]:
    """Detect the four corner anchor markers with validation.

    Args:
        image: Input image
        geom: Page geometry configuration
        markers_cfg: Marker configuration with expected anchor size

    Returns:
        List of (x, y) coordinates for [top-left, top-right, bottom-left, bottom-right]
        or None if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Use Otsu's automatic thresholding to handle varying lighting conditions
    # This automatically finds the optimal threshold between dark markers and light background
    # Works for underexposed, overexposed, and normal lighting scenarios
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    img_height, img_width = gray.shape

    # Calculate expected anchor size in pixels
    # Assume image corresponds roughly to the page geometry
    scale = img_width / geom.width
    expected_anchor_size = markers_cfg.anchor_size * scale
    expected_area = expected_anchor_size ** 2

    # Allow 50% tolerance for size variation
    min_area = expected_area * 0.5
    max_area = expected_area * 2.0

    # Define corner bands (anchors should be in outer 25% of image)
    # Allows for realistic mobile captures with some margin around the sheet
    corner_band_x = img_width * 0.25
    corner_band_y = img_height * 0.25

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for square-like contours with expected size and corner positions
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Check area against expected anchor size
        if area < min_area or area > max_area:
            continue

        # Check if it's square-like
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        if not (0.6 < aspect_ratio < 1.4):  # Approximately square
            continue

        # Check if bounding box is in a corner region
        center_x = x + w // 2
        center_y = y + h // 2

        in_left_band = center_x < corner_band_x
        in_right_band = center_x > (img_width - corner_band_x)
        in_top_band = center_y < corner_band_y
        in_bottom_band = center_y > (img_height - corner_band_y)

        # Must be in a corner (left/right AND top/bottom)
        in_corner = (in_left_band or in_right_band) and (in_top_band or in_bottom_band)

        if not in_corner:
            continue

        candidates.append((center_x, center_y))

    if len(candidates) < 4:
        print(f"Warning: Only found {len(candidates)} anchor candidates (expected 4)")
        return None

    # Sort candidates to find the four corners
    candidates = sorted(candidates, key=lambda p: p[1])  # Sort by y
    top_two = sorted(candidates[:2], key=lambda p: p[0])  # Top row, sort by x
    bottom_two = sorted(candidates[-2:], key=lambda p: p[0])  # Bottom row, sort by x

    markers = [top_two[0], top_two[1], bottom_two[0], bottom_two[1]]

    # Validate geometric layout: check if markers form a reasonable rectangle
    if not _validate_rectangle_geometry(markers, img_width, img_height):
        print("Warning: Anchor markers do not form a valid rectangular layout")
        return None

    return markers


def _validate_rectangle_geometry(
    markers: List[Tuple[int, int]],
    img_width: int,
    img_height: int
) -> bool:
    """Validate that four markers form a reasonable rectangle.

    Args:
        markers: [top-left, top-right, bottom-left, bottom-right] coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        True if geometry is valid, False otherwise
    """
    if len(markers) != 4:
        return False

    tl, tr, bl, br = markers

    # Check horizontal alignment: top two should have similar y, bottom two similar y
    # Allow 15% tolerance for realistic perspective distortion from mobile captures
    top_y_diff = abs(tl[1] - tr[1])
    bottom_y_diff = abs(bl[1] - br[1])
    max_y_tolerance = img_height * 0.15  # 15% tolerance for perspective (was 0.12)

    if top_y_diff > max_y_tolerance or bottom_y_diff > max_y_tolerance:
        print(f"Warning: Horizontal alignment check failed (top_diff={top_y_diff}, bottom_diff={bottom_y_diff})")
        return False

    # Check vertical alignment: left two should have similar x, right two similar x
    # Allow 15% tolerance for realistic perspective distortion from mobile captures
    left_x_diff = abs(tl[0] - bl[0])
    right_x_diff = abs(tr[0] - br[0])
    max_x_tolerance = img_width * 0.15  # 15% tolerance for perspective (was 0.12)

    if left_x_diff > max_x_tolerance or right_x_diff > max_x_tolerance:
        print(f"Warning: Vertical alignment check failed (left_diff={left_x_diff}, right_diff={right_x_diff})")
        return False

    # Check that widths are consistent (top and bottom should be similar)
    # Trapezoid shape from perspective can make widths differ by ~30%
    top_width = tr[0] - tl[0]
    bottom_width = br[0] - bl[0]
    width_ratio = min(top_width, bottom_width) / max(top_width, bottom_width) if max(top_width, bottom_width) > 0 else 0

    if width_ratio < 0.70:  # Widths should be within 30% of each other (was 0.75)
        print(f"Warning: Width consistency check failed (ratio={width_ratio:.2f})")
        return False

    # Check that heights are consistent (left and right should be similar)
    # Trapezoid shape from perspective can make heights differ by ~30%
    left_height = bl[1] - tl[1]
    right_height = br[1] - tr[1]
    height_ratio = min(left_height, right_height) / max(left_height, right_height) if max(left_height, right_height) > 0 else 0

    if height_ratio < 0.70:  # Heights should be within 30% of each other (was 0.75)
        print(f"Warning: Height consistency check failed (ratio={height_ratio:.2f})")
        return False

    # Check aspect ratio is reasonable (should be close to expected image aspect)
    # Allow 30% deviation for perspective distortion
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2
    aspect_ratio = avg_width / avg_height if avg_height > 0 else 0
    expected_aspect = img_width / img_height

    # Allow 30% deviation from expected aspect ratio for realistic captures
    if abs(aspect_ratio - expected_aspect) > expected_aspect * 0.3:
        print(f"Warning: Aspect ratio check failed (found={aspect_ratio:.2f}, expected={expected_aspect:.2f})")
        return False

    return True


def correct_skew(image: np.ndarray, markers: List[Tuple[int, int]], geom: PageGeometry) -> np.ndarray:
    """Apply perspective transformation to correct skew.

    Args:
        image: Input image
        markers: Corner markers [top-left, top-right, bottom-left, bottom-right]
        geom: Page geometry configuration

    Returns:
        Corrected image
    """
    # Calculate adaptive output dimensions based on detected marker distances
    # This preserves original resolution quality instead of forcing 1400px
    tl, tr, bl, br = markers

    # Measure actual pixel distances between markers
    top_edge = np.linalg.norm(np.array(tr) - np.array(tl))
    bottom_edge = np.linalg.norm(np.array(br) - np.array(bl))
    left_edge = np.linalg.norm(np.array(bl) - np.array(tl))
    right_edge = np.linalg.norm(np.array(br) - np.array(tr))

    # Average the opposing edges to get robust scale estimate
    avg_width_pixels = (top_edge + bottom_edge) / 2.0
    avg_height_pixels = (left_edge + right_edge) / 2.0

    # Calculate scale factors from physical dimensions
    scale_x = avg_width_pixels / geom.width
    scale_y = avg_height_pixels / geom.height

    # Use average scale to maintain aspect ratio
    scale = (scale_x + scale_y) / 2.0

    # Compute output dimensions preserving the detected resolution
    output_width = int(round(geom.width * scale))
    output_height = int(round(geom.height * scale))

    # Clamp to reasonable bounds while preserving aspect ratio
    img_height, img_width = image.shape[:2]

    # First apply maximum bounds (don't exceed source)
    output_width = min(output_width, img_width)
    output_height = min(output_height, img_height)

    # Then apply minimum bounds while preserving aspect ratio
    min_width = 800
    min_height = 1000
    aspect_ratio = geom.width / geom.height

    if output_width < min_width:
        output_width = min_width
        output_height = int(round(output_width / aspect_ratio))

    if output_height < min_height:
        output_height = min_height
        output_width = int(round(output_height * aspect_ratio))

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

    print(f"Applied perspective correction: {img_width}x{img_height} -> {output_width}x{output_height} (scale: {scale:.2f}x)")

    return corrected



def calculate_adaptive_fill_threshold(gray: np.ndarray, base_threshold: float = 0.4) -> float:
    """Calculate adaptive fill threshold based on image contrast profile.

    Args:
        gray: Grayscale image of the bubble region (should exclude anchor markers
              and margins to avoid skewing percentiles)
        base_threshold: Base threshold for normal contrast conditions

    Returns:
        Adjusted threshold that accounts for compressed contrast ranges
    """
    # Analyze the image's intensity distribution
    # Use percentiles to ignore outliers (very dark/bright pixels)
    percentile_low = np.percentile(gray, 5)   # Darkest 5% (likely filled bubbles/markers)
    percentile_high = np.percentile(gray, 95)  # Lightest 95% (background/unfilled)

    # Calculate available contrast range
    available_range = percentile_high - percentile_low
    max_range = 255.0  # Maximum possible range

    # Contrast factor: how much contrast is available (0.0 to 1.0)
    contrast_factor = available_range / max_range if max_range > 0 else 1.0

    # Adjust threshold based on available contrast
    # Less contrast = lower threshold needed to detect filled bubbles
    # More contrast = can use higher threshold for better discrimination
    # More aggressive reduction for low contrast to handle overexposed images
    # where printed text may skew percentiles
    if contrast_factor < 0.3:  # Very low contrast (e.g., overexposed)
        adjusted_threshold = base_threshold * 0.3  # More aggressive (was 0.5)
    elif contrast_factor < 0.5:  # Moderate low contrast
        adjusted_threshold = base_threshold * 0.6  # More aggressive (was 0.7)
    elif contrast_factor < 0.7:  # Slightly reduced contrast
        adjusted_threshold = base_threshold * 0.8  # More aggressive (was 0.85)
    else:  # Normal or high contrast
        adjusted_threshold = base_threshold

    return adjusted_threshold


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
    adaptive_threshold: float,
) -> Optional[Bubble]:
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

    # Analyze fill state using adaptive threshold
    is_filled, intensity = analyze_bubble_fill(
        inner_gray,
        inner_x,
        inner_y,
        radius,
        adaptive_threshold,
    )

    return Bubble(
        x=abs_x,
        y=abs_y,
        radius=radius,
        is_filled=is_filled,
        fill_intensity=intensity,
    )


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
    roll_coords, question_coords, _ = generate_all_bubble_coordinates(geom, layout, sheet)

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

    # Calculate adaptive fill threshold based on inner region's contrast profile
    # Use inner_gray to exclude anchor markers which would skew percentiles
    adaptive_threshold = calculate_adaptive_fill_threshold(inner_gray, layout.fill_threshold)

    # Process roll bubbles
    roll_groups: List[List[Bubble]] = [[] for _ in range(sheet.roll_rows)]
    for coord in roll_coords:
        bubble = _coordinate_to_bubble(
            coord, geom, markers, width, height, margin_x, margin_y, inner_gray, layout, adaptive_threshold
        )
        if bubble is not None and coord.row is not None:
            roll_groups[coord.row].append(bubble)

    for group in roll_groups:
        group.sort(key=lambda b: b.x)

    # Process question bubbles
    question_groups: List[List[Bubble]] = []
    question_map: dict[int, List[tuple[int, Bubble]]] = {}

    for coord in question_coords:
        bubble = _coordinate_to_bubble(
            coord, geom, markers, width, height, margin_x, margin_y, inner_gray, layout, adaptive_threshold
        )
        if bubble is not None and coord.question is not None and coord.option_index is not None:
            question_map.setdefault(coord.question, []).append((coord.option_index, bubble))

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

    # Extract local region around bubble (optimization: only process nearby pixels)
    # Region size: enough to include background ring
    region_size = int(radius * 2.0) + 1
    x_min = max(0, x - region_size)
    x_max = min(w, x + region_size + 1)
    y_min = max(0, y - region_size)
    y_max = min(h, y + region_size + 1)

    # Extract local region
    local_region = gray[y_min:y_max, x_min:x_max]
    if local_region.size == 0:
        return False, 0.0

    # Adjust center coordinates to local region
    local_x = x - x_min
    local_y = y - y_min

    # Create masks for interior and background ring (only for local region)
    local_h, local_w = local_region.shape
    y_grid, x_grid = np.ogrid[:local_h, :local_w]
    distance_from_center = np.sqrt((x_grid - local_x)**2 + (y_grid - local_y)**2)

    # Interior: pixels inside radius * 0.7 (avoid edge artifacts)
    interior_mask = distance_from_center <= (radius * 0.7)

    # Background ring: pixels in ring slightly outside the bubble
    ring_inner_radius = radius * 1.2
    ring_outer_radius = radius * 1.8
    background_mask = (distance_from_center >= ring_inner_radius) & (distance_from_center <= ring_outer_radius)

    # Calculate mean intensity (lower = darker = more filled)
    interior_pixels = local_region[interior_mask]
    background_pixels = local_region[background_mask]

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

    # Detect anchor markers with validation
    anchor_points = detect_anchor_markers(image, geom, markers_cfg)
    if anchor_points is None or len(anchor_points) != 4:
        print(f"ERROR: Failed to detect valid anchor markers for {input_path.name}")
        print("Possible causes:")
        print("  - Anchors are outside expected corner regions")
        print("  - Anchor size doesn't match configuration")
        print("  - Detected anchors don't form a proper rectangle")
        print("  - Strong perspective distortion or misaligned scan")
        print("Skipping this sheet.")
        return False

    print(f"Detected and validated {len(anchor_points)} anchor markers")

    # Correct skew
    corrected = correct_skew(image, anchor_points, geom)

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
