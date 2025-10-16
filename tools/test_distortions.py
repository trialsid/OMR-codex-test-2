"""Test OMR processing robustness with various image distortions.

This script simulates real-world camera capture issues:
- Perspective distortions (camera angles)
- Contrast variations (lighting issues)
- Aspect ratio distortions (squashing/stretching)
- Combined distortions (realistic scenarios)

Generates an HTML report showing success/failure for each distortion.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from omr_config import PageGeometry, BubbleLayout, MarkerConfig, SheetLayout
from omr_processor import (
    detect_anchor_markers,
    correct_skew,
    sample_bubbles_from_coordinates,
    overlay_labels,
)


def crop_to_content_and_resize(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Crop image to non-white content and resize to target dimensions.

    Args:
        image: Input image with potential white borders
        target_shape: (height, width) to resize to

    Returns:
        Cropped and resized image filling target dimensions
    """
    # Convert to grayscale for thresholding
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Threshold to find non-white content (anything darker than 250)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Find contours of non-white regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # No content found, return original resized
        return cv2.resize(image, (target_shape[1], target_shape[0]))

    # Get bounding box of all content
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    x_min, y_min = int(x_min), int(y_min)
    x_max, y_max = int(x_max), int(y_max)

    # Add bleed/margin around content (8% of content dimensions)
    # This leaves realistic white space like a mobile camera capture
    content_width = x_max - x_min
    content_height = y_max - y_min
    bleed_x = int(content_width * 0.08)
    bleed_y = int(content_height * 0.08)

    # Expand bounding box with bleed, staying within image bounds
    img_h, img_w = image.shape[:2]
    x_min = max(0, x_min - bleed_x)
    y_min = max(0, y_min - bleed_y)
    x_max = min(img_w, x_max + bleed_x)
    y_max = min(img_h, y_max + bleed_y)

    # Crop to content with bleed
    cropped = image[y_min:y_max, x_min:x_max]

    # Resize to target dimensions
    target_h, target_w = target_shape
    resized = cv2.resize(cropped, (target_w, target_h))

    return resized


@dataclass
class DistortionResult:
    """Result of processing a distorted image."""
    name: str
    description: str
    success: bool
    error_message: str = ""
    anchors_detected: int = 0
    roll_bubbles: int = 0
    question_bubbles: int = 0
    processing_time_ms: float = 0.0
    variation_path: str = ""
    anchor_viz_path: str = ""
    processed_path: str = ""


def apply_perspective_distortion(
    image: np.ndarray,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    rotation: float = 0.0
) -> np.ndarray:
    """Apply perspective distortion to simulate camera angle.

    Simulates viewing the paper from a camera at an angle. The paper appears
    as a trapezoid within the camera frame, with all 4 corners visible.

    Args:
        image: Input image
        tilt_x: Horizontal tilt in degrees (positive = viewing from right side)
        tilt_y: Vertical tilt in degrees (positive = viewing from bottom)
        rotation: Rotation angle in degrees (paper rotated on table)

    Returns:
        Distorted image with all corners visible
    """
    h, w = image.shape[:2]
    # Save original dimensions to resize back to at the end
    original_h, original_w = h, w

    # For rotation, we need a larger canvas to avoid cutting off corners
    if rotation != 0:
        # Calculate bounding box for rotated image
        angle_rad = np.radians(abs(rotation))
        new_w = int(abs(w * np.cos(angle_rad)) + abs(h * np.sin(angle_rad)))
        new_h = int(abs(h * np.cos(angle_rad)) + abs(w * np.sin(angle_rad)))

        # Ensure we have enough padding
        new_w = int(new_w * 1.2)
        new_h = int(new_h * 1.2)

        # Center the rotation
        center_x = new_w / 2
        center_y = new_h / 2

        # Create rotation matrix
        rot_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation, 1.0)

        # Adjust translation to center the original image
        rot_matrix[0, 2] += center_x - w / 2
        rot_matrix[1, 2] += center_y - h / 2

        # Apply rotation with larger canvas
        rotated = cv2.warpAffine(image, rot_matrix, (new_w, new_h),
                                 borderValue=(255, 255, 255))

        # Now apply perspective distortion to the rotated image
        image = rotated
        h, w = image.shape[:2]

    # Apply perspective tilt to simulate camera viewing angle
    # The paper appears as a trapezoid within the frame

    # Shrink factor: how much smaller the paper appears (0.0 to 1.0)
    # Larger tilt = paper appears smaller to fit in frame with perspective
    shrink_factor = 0.15  # Base shrink to ensure all corners are visible

    # Calculate trapezoid corner positions
    # Start with centered, shrunk rectangle
    margin = int(min(w, h) * shrink_factor)

    src_corners = np.float32([
        [0, 0],           # top-left
        [w-1, 0],         # top-right
        [0, h-1],         # bottom-left
        [w-1, h-1]        # bottom-right
    ])

    # Destination corners (trapezoid shape within the frame)
    dst_corners = np.float32([
        [margin, margin],                    # top-left
        [w - margin - 1, margin],            # top-right
        [margin, h - margin - 1],            # bottom-left
        [w - margin - 1, h - margin - 1]     # bottom-right
    ])

    # Apply perspective tilt by modifying the trapezoid
    # tilt_x: positive = viewing from right (left edge appears farther/compressed vertically)
    # tilt_y: positive = viewing from bottom (top edge appears farther/compressed horizontally)

    if tilt_x != 0:
        # Horizontal tilt: left edge appears farther away
        # Left edge gets compressed vertically (trapezoid effect)
        # The two left corners move toward the vertical center
        compression = tilt_x * 0.015  # Compression factor

        # Calculate how much to move left edge corners toward center
        left_compression = int(h * compression)

        # Compress left edge vertically
        dst_corners[0][1] += left_compression  # top-left: move down
        dst_corners[2][1] -= left_compression  # bottom-left: move up

        # Right edge expands slightly (closer to viewer)
        right_expansion = int(left_compression * 0.3)
        dst_corners[1][1] -= right_expansion  # top-right: move up slightly
        dst_corners[3][1] += right_expansion  # bottom-right: move down slightly

    if tilt_y != 0:
        # Vertical tilt: top edge appears farther away
        # Top edge gets compressed horizontally (trapezoid effect)
        # The two top corners move toward the horizontal center
        compression = tilt_y * 0.015  # Compression factor

        # Calculate how much to move top edge corners toward center
        top_compression = int(w * compression)

        # Compress top edge horizontally
        dst_corners[0][0] += top_compression  # top-left: move right
        dst_corners[1][0] -= top_compression  # top-right: move left

        # Bottom edge expands slightly (closer to viewer)
        bottom_expansion = int(top_compression * 0.3)
        dst_corners[2][0] -= bottom_expansion  # bottom-left: move left slightly
        dst_corners[3][0] += bottom_expansion  # bottom-right: move right slightly

    # Calculate and apply perspective transform
    matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
    distorted = cv2.warpPerspective(image, matrix, (w, h),
                                    borderValue=(255, 255, 255))

    # Crop whitespace and resize to original dimensions to fill the frame
    result = crop_to_content_and_resize(distorted, (original_h, original_w))
    return result


def apply_contrast_variation(
    image: np.ndarray,
    black_point: int = 0,
    white_point: int = 255,
    output_black: int = 0,
    output_white: int = 255
) -> np.ndarray:
    """Apply contrast and brightness variations by remapping histogram.

    Simulates real-world lighting conditions by remapping the input range
    [black_point, white_point] to output range [output_black, output_white].

    Args:
        image: Input image
        black_point: Input value to treat as black (0-255)
        white_point: Input value to treat as white (0-255)
        output_black: Output value for blacks (0-255)
        output_white: Output value for whites (0-255)

    Returns:
        Modified image

    Examples:
        Underexposed (dark): black_point=0, white_point=255, output_black=10, output_white=120
            -> Markers go 0→10, background 255→120
        Overexposed (washed): black_point=0, white_point=255, output_black=180, output_white=250
            -> Markers go 0→180, background 255→250
        Low contrast: black_point=0, white_point=255, output_black=80, output_white=180
            -> Everything compressed to narrow range
    """
    # Build lookup table for histogram remapping
    table = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        # Normalize input to [0, 1] range
        if white_point > black_point:
            normalized = (i - black_point) / (white_point - black_point)
        else:
            normalized = 0.0

        # Clamp to [0, 1]
        normalized = max(0.0, min(1.0, normalized))

        # Map to output range
        output_value = output_black + normalized * (output_white - output_black)
        table[i] = np.clip(int(output_value), 0, 255)

    # Apply lookup table
    result = cv2.LUT(image, table)

    return result


def apply_aspect_distortion(
    image: np.ndarray,
    horizontal_scale: float = 1.0,
    vertical_scale: float = 1.0
) -> np.ndarray:
    """Apply aspect ratio distortion (squashing/stretching).

    Args:
        image: Input image
        horizontal_scale: Horizontal scaling factor (< 1.0 = squash, > 1.0 = stretch)
        vertical_scale: Vertical scaling factor

    Returns:
        Distorted image resized back to original canvas dimensions
    """
    h, w = image.shape[:2]
    new_w = int(w * horizontal_scale)
    new_h = int(h * vertical_scale)

    # Resize with new aspect ratio (actual distortion)
    distorted = cv2.resize(image, (new_w, new_h))

    # Add small margin padding to avoid edge artifacts during final resize
    margin = int(max(new_w, new_h) * 0.05)

    # Create canvas with padding
    canvas_w = new_w + 2 * margin
    canvas_h = new_h + 2 * margin
    canvas = np.full((canvas_h, canvas_w, 3) if len(image.shape) == 3 else (canvas_h, canvas_w),
                     255, dtype=np.uint8)

    # Center the distorted image on canvas
    offset_x = margin
    offset_y = margin
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = distorted

    # Crop whitespace and resize to original dimensions to fill the frame
    result = crop_to_content_and_resize(canvas, (h, w))
    return result


def process_distorted_image(
    image: np.ndarray,
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers_cfg: MarkerConfig,
) -> Tuple[bool, str, int, int, int, Optional[np.ndarray], Optional[np.ndarray]]:
    """Process a distorted image and return results.

    Returns:
        (success, error_message, anchors_detected, roll_bubbles, questions, processed_image, anchor_viz)
    """
    try:
        # Detect anchor markers
        anchor_points = detect_anchor_markers(image, geom, markers_cfg)

        # Create anchor visualization with corner zones overlay
        anchor_viz = image.copy()
        img_height, img_width = image.shape[:2]

        # Draw corner zone bands (where anchors are expected)
        # Match the actual detection logic in omr_processor.py
        corner_band_x = int(img_width * 0.25)
        corner_band_y_top = int(img_height * 0.35)     # 35% from top (accounts for 20% header)
        corner_band_y_bottom = int(img_height * 0.20)  # 20% from bottom (anchors near edge)

        # Create overlay for semi-transparent corner zones
        overlay = anchor_viz.copy()
        zone_color = (0, 165, 255)  # Orange in BGR

        # Top-left corner zone
        cv2.rectangle(overlay, (0, 0), (corner_band_x, corner_band_y_top), zone_color, -1)

        # Top-right corner zone
        cv2.rectangle(overlay, (img_width - corner_band_x, 0),
                     (img_width, corner_band_y_top), zone_color, -1)

        # Bottom-left corner zone
        cv2.rectangle(overlay, (0, img_height - corner_band_y_bottom),
                     (corner_band_x, img_height), zone_color, -1)

        # Bottom-right corner zone
        cv2.rectangle(overlay, (img_width - corner_band_x, img_height - corner_band_y_bottom),
                     (img_width, img_height), zone_color, -1)

        # Blend overlay with original (25% opacity)
        anchor_viz = cv2.addWeighted(overlay, 0.25, anchor_viz, 0.75, 0)

        # Draw detected anchors on top of zones
        if anchor_points is not None and len(anchor_points) > 0:
            for idx, (x, y) in enumerate(anchor_points):
                cv2.circle(anchor_viz, (x, y), 20, (0, 255, 0), 3)
                cv2.circle(anchor_viz, (x, y), 5, (0, 255, 0), -1)
                # Label corners
                labels = ["TL", "TR", "BL", "BR"]
                if idx < len(labels):
                    cv2.putText(anchor_viz, labels[idx], (x + 25, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if anchor_points is None:
            return False, "Failed to detect anchor markers", 0, 0, 0, None, anchor_viz

        anchors_detected = len(anchor_points)
        if anchors_detected != 4:
            return False, f"Only detected {anchors_detected}/4 anchors", anchors_detected, 0, 0, None, anchor_viz

        # Correct skew
        corrected = correct_skew(image, anchor_points, geom, markers_cfg)

        # Sample bubbles
        roll_groups, question_groups = sample_bubbles_from_coordinates(
            corrected, geom, layout, sheet, markers_cfg
        )

        roll_bubbles = sum(len(group) for group in roll_groups)
        questions = len(question_groups)

        # Overlay labels
        labeled = overlay_labels(corrected, roll_groups, question_groups, sheet)

        return True, "", anchors_detected, roll_bubbles, questions, labeled, anchor_viz

    except Exception as e:
        anchor_viz = image.copy()
        return False, str(e), 0, 0, 0, None, anchor_viz


def generate_distortion_tests(
    original_path: Path,
    output_dir: Path
) -> List[Tuple[str, str, np.ndarray]]:
    """Generate various distorted versions of the original image.

    Returns:
        List of (name, description, distorted_image) tuples
    """
    # Load original image
    original = cv2.imread(str(original_path))
    if original is None:
        raise ValueError(f"Failed to load image: {original_path}")

    tests = []

    # Original (control)
    tests.append(("original", "Original image (control)", original.copy()))

    # === Perspective Distortions ===
    tests.append((
        "perspective_tilt_x_2_5",
        "Perspective: 2.5° horizontal tilt (right closer)",
        apply_perspective_distortion(original, tilt_x=2.5)
    ))

    tests.append((
        "perspective_tilt_x_5",
        "Perspective: 5° horizontal tilt (right closer - moderate)",
        apply_perspective_distortion(original, tilt_x=5)
    ))

    tests.append((
        "perspective_tilt_y_2_5",
        "Perspective: 2.5° vertical tilt (bottom closer)",
        apply_perspective_distortion(original, tilt_y=2.5)
    ))

    tests.append((
        "perspective_tilt_y_5",
        "Perspective: 5° vertical tilt (bottom closer - moderate)",
        apply_perspective_distortion(original, tilt_y=5)
    ))

    tests.append((
        "perspective_both_tilts",
        "Perspective: Combined tilts (2.5° horizontal + 2.5° vertical)",
        apply_perspective_distortion(original, tilt_x=2.5, tilt_y=2.5)
    ))

    tests.append((
        "perspective_rotation_3",
        "Perspective: 3° rotation",
        apply_perspective_distortion(original, rotation=3)
    ))

    tests.append((
        "perspective_rotation_6",
        "Perspective: 6° rotation",
        apply_perspective_distortion(original, rotation=6)
    ))

    tests.append((
        "perspective_rotation_10",
        "Perspective: 10° rotation (challenging)",
        apply_perspective_distortion(original, rotation=10)
    ))

    # === Contrast Variations ===
    # These simulate real-world lighting issues that challenge fixed thresholds

    tests.append((
        "contrast_underexposed",
        "Contrast: Underexposed/dark photo (markers ~30, background ~120)",
        apply_contrast_variation(original, output_black=30, output_white=120)
    ))

    tests.append((
        "contrast_overexposed",
        "Contrast: Overexposed/washed out (markers ~190, background ~245)",
        apply_contrast_variation(original, output_black=190, output_white=245)
    ))

    tests.append((
        "contrast_low_range",
        "Contrast: Low contrast/compressed range (markers ~85, background ~165)",
        apply_contrast_variation(original, output_black=85, output_white=165)
    ))

    tests.append((
        "contrast_very_dark",
        "Contrast: Very dark/severe underexposure (markers ~10, background ~80)",
        apply_contrast_variation(original, output_black=10, output_white=80)
    ))

    tests.append((
        "contrast_shifted",
        "Contrast: Mid-gray shifted (markers ~100, background ~200)",
        apply_contrast_variation(original, output_black=100, output_white=200)
    ))

    # === Aspect Distortions ===
    tests.append((
        "aspect_squash_horizontal",
        "Aspect: Horizontal squash (85%)",
        apply_aspect_distortion(original, horizontal_scale=0.85)
    ))

    tests.append((
        "aspect_squash_vertical",
        "Aspect: Vertical squash (85%)",
        apply_aspect_distortion(original, vertical_scale=0.85)
    ))

    tests.append((
        "aspect_stretch_horizontal",
        "Aspect: Horizontal stretch (115%)",
        apply_aspect_distortion(original, horizontal_scale=1.15)
    ))

    # === Combined Distortions (Realistic) ===
    tests.append((
        "combined_realistic_1",
        "Combined: Slight rotation + underexposed",
        apply_contrast_variation(
            apply_perspective_distortion(original, rotation=7),
            output_black=40, output_white=130
        )
    ))

    tests.append((
        "combined_realistic_2",
        "Combined: Tilt + horizontal squash + low contrast",
        apply_contrast_variation(
            apply_aspect_distortion(
                apply_perspective_distortion(original, tilt_x=3, tilt_y=2.5),
                horizontal_scale=0.9
            ),
            output_black=90, output_white=170
        )
    ))

    return tests


def generate_html_report(
    results: List[DistortionResult],
    output_path: Path
) -> None:
    """Generate HTML report with visual comparison."""

    success_count = sum(1 for r in results if r.success)
    total_count = len(results)
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OMR Distortion Test Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-stats {{
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }}
        .stat {{
            flex: 1;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .test-result {{
            background: white;
            margin-bottom: 20px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .test-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .test-name {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }}
        .status {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }}
        .status.success {{
            background: #d4edda;
            color: #155724;
        }}
        .status.failure {{
            background: #f8d7da;
            color: #721c24;
        }}
        .test-description {{
            color: #666;
            margin-bottom: 15px;
        }}
        .test-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .detail-item {{
            display: flex;
            flex-direction: column;
        }}
        .detail-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .detail-value {{
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }}
        .error-message {{
            background: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }}
        .images {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .image-container {{
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .image-label {{
            margin-top: 10px;
            font-weight: bold;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>OMR Distortion Test Results</h1>

    <div class="summary">
        <h2>Summary</h2>
        <div class="summary-stats">
            <div class="stat">
                <div class="stat-value">{success_count}/{total_count}</div>
                <div class="stat-label">Tests Passed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{success_rate:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_count}</div>
                <div class="stat-label">Total Tests</div>
            </div>
        </div>
    </div>

    <h2>Test Results</h2>
"""

    for result in results:
        status_class = "success" if result.success else "failure"
        status_text = "PASSED" if result.success else "FAILED"

        html += f"""
    <div class="test-result">
        <div class="test-header">
            <div class="test-name">{result.name}</div>
            <div class="status {status_class}">{status_text}</div>
        </div>
        <div class="test-description">{result.description}</div>
"""

        if not result.success:
            html += f"""
        <div class="error-message">
            <strong>Error:</strong> {result.error_message}
        </div>
"""

        html += f"""
        <div class="test-details">
            <div class="detail-item">
                <span class="detail-label">Anchors Detected</span>
                <span class="detail-value">{result.anchors_detected}/4</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Roll Bubbles</span>
                <span class="detail-value">{result.roll_bubbles}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Questions</span>
                <span class="detail-value">{result.question_bubbles}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Processing Time</span>
                <span class="detail-value">{result.processing_time_ms:.1f} ms</span>
            </div>
        </div>

        <div class="images">
            <div class="image-container">
                <img src="{result.variation_path}" alt="Distorted Input">
                <div class="image-label">Distorted Input</div>
            </div>
            <div class="image-container">
                <img src="{result.anchor_viz_path}" alt="Anchor Detection">
                <div class="image-label">Anchor Detection</div>
            </div>
"""

        if result.success and result.processed_path:
            html += f"""
            <div class="image-container">
                <img src="{result.processed_path}" alt="Processed Output">
                <div class="image-label">Processed Output</div>
            </div>
"""

        html += """
        </div>
    </div>
"""

    html += """
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    """Main entry point."""
    print("="*60)
    print("OMR DISTORTION TESTING")
    print("="*60)

    # Configuration
    geom = PageGeometry()
    layout = BubbleLayout()
    sheet = SheetLayout()
    markers_cfg = MarkerConfig()

    # Paths
    input_image = Path("sheets") / "omr_sheet.png"
    output_dir = Path("distortion_tests")
    variations_dir = output_dir / "variations"
    anchors_dir = output_dir / "anchors"
    processed_dir = output_dir / "processed"

    # Create output directories
    variations_dir.mkdir(parents=True, exist_ok=True)
    anchors_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Check input file
    if not input_image.exists():
        print(f"Error: {input_image} not found")
        return

    print(f"\nInput image: {input_image}")
    print(f"Output directory: {output_dir}\n")

    # Generate distortion tests
    print("Generating distorted test images...")
    tests = generate_distortion_tests(input_image, output_dir)
    print(f"Generated {len(tests)} test variations\n")

    # Process each test
    results = []

    for idx, (name, description, distorted_image) in enumerate(tests, 1):
        print(f"[{idx}/{len(tests)}] Testing: {name}")
        print(f"  Description: {description}")

        # Save variation
        variation_path = variations_dir / f"{name}.png"
        cv2.imwrite(str(variation_path), distorted_image)

        # Process the distorted image
        start_time = time.perf_counter()
        success, error_msg, anchors, roll_bubbles, questions, processed, anchor_viz = \
            process_distorted_image(distorted_image, geom, layout, sheet, markers_cfg)
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Save anchor visualization
        anchor_viz_path = anchors_dir / f"anchors_{name}.png"
        if anchor_viz is not None:
            cv2.imwrite(str(anchor_viz_path), anchor_viz)
        anchor_viz_path_rel = f"anchors/anchors_{name}.png"

        # Save processed image if successful
        processed_path = ""
        if success and processed is not None:
            processed_path = processed_dir / f"processed_{name}.png"
            cv2.imwrite(str(processed_path), processed)
            processed_path_rel = f"processed/processed_{name}.png"
        else:
            processed_path_rel = ""

        # Store result
        result = DistortionResult(
            name=name,
            description=description,
            success=success,
            error_message=error_msg,
            anchors_detected=anchors,
            roll_bubbles=roll_bubbles,
            question_bubbles=questions,
            processing_time_ms=processing_time_ms,
            variation_path=f"variations/{name}.png",
            anchor_viz_path=anchor_viz_path_rel,
            processed_path=processed_path_rel
        )
        results.append(result)

        # Print result
        status = "[PASSED]" if success else "[FAILED]"
        print(f"  Result: {status}")
        if not success:
            print(f"  Error: {error_msg}")
        else:
            print(f"  Anchors: {anchors}/4, Roll bubbles: {roll_bubbles}, Questions: {questions}")
        print(f"  Time: {processing_time_ms:.1f} ms")
        print()

    # Generate HTML report
    report_path = output_dir / "report.html"
    print(f"Generating HTML report: {report_path}")
    generate_html_report(results, report_path)

    # Print summary
    success_count = sum(1 for r in results if r.success)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tests: {len(results)}")
    print(f"Passed: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print(f"Success rate: {success_count/len(results)*100:.1f}%")
    print(f"\nReport saved to: {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
