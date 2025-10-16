"""Validate OMR sheet bubble positions using Hough circle detection.

This script compares expected bubble positions from the layout configuration
with actual circles detected in the rendered OMR sheet image.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Tuple, Optional
import numpy as np
import cv2

from omr_config import PageGeometry, BubbleLayout, SheetLayout, MarkerConfig
from omr_layout import generate_all_bubble_coordinates, BubbleCoordinate


def pdf_to_image_coords(
    pdf_x: float,
    pdf_y: float,
    pdf_height: float,
    img_height: int,
    img_width: int,
    pdf_width: float,
) -> Tuple[int, int]:
    """Convert PDF coordinates (points) to image pixel coordinates.

    Args:
        pdf_x, pdf_y: PDF coordinates (origin at bottom-left)
        pdf_height: PDF page height in points
        img_height, img_width: Image dimensions in pixels
        pdf_width: PDF page width in points

    Returns:
        (img_x, img_y) in pixel coordinates (origin at top-left)
    """
    # Calculate scaling factor
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    # Convert coordinates (flip Y axis)
    img_x = int(pdf_x * scale_x)
    img_y = int((pdf_height - pdf_y) * scale_y)

    return img_x, img_y


def match_circles(
    expected: List[Tuple[int, int, float]],
    detected: List[Tuple[int, int, float]],
    max_distance: float = 50,
) -> Tuple[List[Tuple[int, int, float]], List[int]]:
    """Match detected circles to expected positions.

    Args:
        expected: List of (x, y, radius) for expected bubbles
        detected: List of (x, y, radius) for detected circles
        max_distance: Maximum distance in pixels to consider a match

    Returns:
        (matched_pairs, unmatched_expected_indices)
        matched_pairs: List of (expected_idx, detected_idx, drift_distance)
        unmatched_expected_indices: Indices of expected bubbles with no match
    """
    matched_pairs = []
    used_detected = set()
    unmatched_expected = []

    for exp_idx, (ex, ey, er) in enumerate(expected):
        best_dist = float('inf')
        best_det_idx = -1

        for det_idx, (dx, dy, dr) in enumerate(detected):
            if det_idx in used_detected:
                continue

            dist = np.sqrt((ex - dx) ** 2 + (ey - dy) ** 2)

            if dist < best_dist:
                best_dist = dist
                best_det_idx = det_idx

        if best_det_idx >= 0 and best_dist <= max_distance:
            matched_pairs.append((exp_idx, best_det_idx, best_dist))
            used_detected.add(best_det_idx)
        else:
            unmatched_expected.append(exp_idx)

    return matched_pairs, unmatched_expected


def create_visualization(
    image: np.ndarray,
    expected_bubbles: List[Tuple[int, int, float]],
    detected_circles: List[Tuple[int, int, float]],
    matched_pairs: List[Tuple[int, int, float]],
    unmatched_expected: List[int],
    output_path: Path,
) -> None:
    """Create visualization with expected and detected circles.

    Args:
        image: Input image
        expected_bubbles: List of (x, y, radius) for expected positions
        detected_circles: List of (x, y, radius) for detected circles
        matched_pairs: List of (expected_idx, detected_idx, drift)
        unmatched_expected: Indices of unmatched expected bubbles
        output_path: Where to save the visualization
    """
    # Create color copy
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw all expected circles in blue
    for x, y, r in expected_bubbles:
        cv2.circle(vis, (x, y), int(r), (255, 0, 0), 1)  # Blue

    # Draw all detected circles in green
    for x, y, r in detected_circles:
        cv2.circle(vis, (x, y), int(r), (0, 255, 0), 1)  # Green

    # Draw drift lines and labels for matched pairs
    for exp_idx, det_idx, drift in matched_pairs:
        exp_x, exp_y, exp_r = expected_bubbles[exp_idx]
        det_x, det_y, det_r = detected_circles[det_idx]

        # Draw line connecting expected to detected
        if drift > 0.1:  # Only draw line if there's noticeable drift
            cv2.line(vis, (exp_x, exp_y), (det_x, det_y), (0, 255, 255), 1)  # Yellow

        # Add drift text below bubble
        text = f"{drift:.2f}"
        text_x = exp_x - 10
        text_y = exp_y + int(exp_r) + 12

        # Add small background for text readability
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.rectangle(vis, (text_x - 1, text_y - text_h - 1),
                     (text_x + text_w + 1, text_y + 2), (255, 255, 255), -1)

        cv2.putText(vis, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)  # Red text

    # Mark unmatched expected bubbles with red X
    for exp_idx in unmatched_expected:
        x, y, r = expected_bubbles[exp_idx]
        cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS,
                      int(r * 2), 2)  # Red X

    # Save visualization
    cv2.imwrite(str(output_path), vis)


def validate_bubbles(image_path: Path, output_path: Path) -> None:
    """Validate bubble positions in an OMR sheet image.

    Args:
        image_path: Path to the OMR sheet image
        output_path: Path to save visualization
    """
    # Load configuration
    geom = PageGeometry()
    layout = BubbleLayout()
    sheet = SheetLayout()
    markers = MarkerConfig()

    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img_height, img_width = image.shape
    print(f"Image dimensions: {img_width}x{img_height}")
    print(f"PDF dimensions: {geom.width}x{geom.height} points")

    # Generate expected bubble coordinates
    roll_bubbles, question_bubbles, _ = generate_all_bubble_coordinates(geom, layout, sheet, markers)
    all_bubbles = roll_bubbles + question_bubbles

    # Convert PDF coordinates to image coordinates
    expected_circles = []
    for bubble in all_bubbles:
        img_x, img_y = pdf_to_image_coords(
            bubble.x, bubble.y, geom.height, img_height, img_width, geom.width
        )
        # Convert radius to pixels
        radius_pixels = bubble.radius * (img_width / geom.width)
        expected_circles.append((img_x, img_y, radius_pixels))

    print(f"\nExpected bubbles: {len(expected_circles)}")

    # Detect circles using Hough transform
    # Estimate radius range in pixels
    expected_radius = layout.radius * (img_width / geom.width)
    min_radius = int(expected_radius * 0.8)
    max_radius = int(expected_radius * 1.2)

    print(f"Detecting circles with radius range: {min_radius}-{max_radius} pixels")

    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(expected_radius * 1.5),  # Minimum distance between centers
        param1=50,  # Canny edge detection threshold
        param2=15,  # Accumulator threshold (lower = more false positives)
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    detected_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        detected_circles = [(x, y, r) for x, y, r in circles]
        print(f"Detected circles: {len(detected_circles)}")
    else:
        print("No circles detected!")
        return

    # Match expected to detected
    matched_pairs, unmatched_expected = match_circles(
        expected_circles, detected_circles, max_distance=expected_radius * 2
    )

    print(f"Matched bubbles: {len(matched_pairs)}")
    print(f"Unmatched expected bubbles: {len(unmatched_expected)}")

    # Calculate drift statistics
    if matched_pairs:
        drifts = [drift for _, _, drift in matched_pairs]
        avg_drift = np.mean(drifts)
        max_drift = np.max(drifts)
        min_drift = np.min(drifts)
        std_drift = np.std(drifts)

        print(f"\nDrift Statistics:")
        print(f"  Average drift: {avg_drift:.3f} pixels")
        print(f"  Min drift: {min_drift:.3f} pixels")
        print(f"  Max drift: {max_drift:.3f} pixels")
        print(f"  Std deviation: {std_drift:.3f} pixels")

        # Find bubble with max drift
        max_drift_idx = drifts.index(max_drift)
        max_exp_idx, max_det_idx, _ = matched_pairs[max_drift_idx]
        max_bubble = all_bubbles[max_exp_idx]

        # Identify the bubble
        if max_bubble.row is not None:
            bubble_id = f"Roll(row={max_bubble.row}, col={max_bubble.column})"
        else:
            bubble_id = f"Question {max_bubble.question}, option {max_bubble.option_index}"

        print(f"  Max drift bubble: {bubble_id}")

    # Create visualization
    print(f"\nCreating visualization: {output_path}")
    create_visualization(
        image, expected_circles, detected_circles,
        matched_pairs, unmatched_expected, output_path
    )
    print("Done!")


if __name__ == "__main__":
    sheets_dir = Path("sheets")
    input_image = sheets_dir / "omr_sheet.png"
    output_image = sheets_dir / "bubble_validation.png"

    validate_bubbles(input_image, output_image)
