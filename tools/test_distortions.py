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
    # tilt_x: positive = viewing from right (left side appears farther/smaller)
    # tilt_y: positive = viewing from bottom (top appears farther/smaller)

    x_scale = tilt_x * 0.01  # Scale factor
    y_scale = tilt_y * 0.01

    if tilt_x != 0:
        # Horizontal tilt: left edge appears smaller (farther away)
        offset_x = int(w * x_scale)
        dst_corners[0][0] += offset_x  # top-left: move right
        dst_corners[2][0] += offset_x  # bottom-left: move right
        dst_corners[1][0] -= offset_x * 0.3  # top-right: slight adjustment
        dst_corners[3][0] -= offset_x * 0.3  # bottom-right: slight adjustment

    if tilt_y != 0:
        # Vertical tilt: top edge appears smaller (farther away)
        offset_y = int(h * y_scale)
        dst_corners[0][1] += offset_y  # top-left: move down
        dst_corners[1][1] += offset_y  # top-right: move down
        dst_corners[2][1] -= offset_y * 0.3  # bottom-left: slight adjustment
        dst_corners[3][1] -= offset_y * 0.3  # bottom-right: slight adjustment

    # Calculate and apply perspective transform
    matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
    distorted = cv2.warpPerspective(image, matrix, (w, h),
                                    borderValue=(255, 255, 255))

    return distorted


def apply_contrast_variation(
    image: np.ndarray,
    gamma: float = 1.0,
    brightness: int = 0
) -> np.ndarray:
    """Apply contrast and brightness variations.

    Args:
        image: Input image
        gamma: Gamma correction (< 1.0 = darker, > 1.0 = lighter)
        brightness: Brightness offset (-100 to 100)

    Returns:
        Modified image
    """
    # Build gamma correction lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    # Apply gamma correction
    result = cv2.LUT(image, table)

    # Apply brightness
    if brightness != 0:
        result = cv2.convertScaleAbs(result, alpha=1.0, beta=brightness)

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
        Distorted image
    """
    h, w = image.shape[:2]
    new_w = int(w * horizontal_scale)
    new_h = int(h * vertical_scale)

    # Resize with new aspect ratio
    distorted = cv2.resize(image, (new_w, new_h))

    # Resize back to original dimensions to maintain size
    result = cv2.resize(distorted, (w, h))

    return result


def process_distorted_image(
    image: np.ndarray,
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers_cfg: MarkerConfig,
) -> Tuple[bool, str, int, int, int, Optional[np.ndarray]]:
    """Process a distorted image and return results.

    Returns:
        (success, error_message, anchors_detected, roll_bubbles, questions, processed_image)
    """
    try:
        # Detect anchor markers
        anchor_points = detect_anchor_markers(image)
        if anchor_points is None:
            return False, "Failed to detect anchor markers", 0, 0, 0, None

        anchors_detected = len(anchor_points)
        if anchors_detected != 4:
            return False, f"Only detected {anchors_detected}/4 anchors", anchors_detected, 0, 0, None

        # Correct skew
        corrected = correct_skew(image, anchor_points, geom)

        # Sample bubbles
        roll_groups, question_groups = sample_bubbles_from_coordinates(
            corrected, geom, layout, sheet, markers_cfg
        )

        roll_bubbles = sum(len(group) for group in roll_groups)
        questions = len(question_groups)

        # Overlay labels
        labeled = overlay_labels(corrected, roll_groups, question_groups, sheet)

        return True, "", anchors_detected, roll_bubbles, questions, labeled

    except Exception as e:
        return False, str(e), 0, 0, 0, None


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
        "perspective_tilt_x_10",
        "Perspective: 10° horizontal tilt (right closer)",
        apply_perspective_distortion(original, tilt_x=10)
    ))

    tests.append((
        "perspective_tilt_x_20",
        "Perspective: 20° horizontal tilt (right closer - strong)",
        apply_perspective_distortion(original, tilt_x=20)
    ))

    tests.append((
        "perspective_tilt_y_10",
        "Perspective: 10° vertical tilt (bottom closer)",
        apply_perspective_distortion(original, tilt_y=10)
    ))

    tests.append((
        "perspective_tilt_y_20",
        "Perspective: 20° vertical tilt (bottom closer - strong)",
        apply_perspective_distortion(original, tilt_y=20)
    ))

    tests.append((
        "perspective_both_tilts",
        "Perspective: Combined tilts (10° horizontal + 10° vertical)",
        apply_perspective_distortion(original, tilt_x=10, tilt_y=10)
    ))

    tests.append((
        "perspective_rotation_5",
        "Perspective: 5° rotation",
        apply_perspective_distortion(original, rotation=5)
    ))

    tests.append((
        "perspective_rotation_15",
        "Perspective: 15° rotation",
        apply_perspective_distortion(original, rotation=15)
    ))

    tests.append((
        "perspective_rotation_30",
        "Perspective: 30° rotation (extreme)",
        apply_perspective_distortion(original, rotation=30)
    ))

    # === Contrast Variations ===
    tests.append((
        "contrast_low_dark",
        "Contrast: Low light (gamma 0.6)",
        apply_contrast_variation(original, gamma=0.6)
    ))

    tests.append((
        "contrast_low_washed",
        "Contrast: Washed out (gamma 1.5, brightness +20)",
        apply_contrast_variation(original, gamma=1.5, brightness=20)
    ))

    tests.append((
        "contrast_very_dark",
        "Contrast: Very dark (gamma 0.4, brightness -30)",
        apply_contrast_variation(original, gamma=0.4, brightness=-30)
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
        "Combined: Slight rotation + low contrast",
        apply_contrast_variation(
            apply_perspective_distortion(original, rotation=7),
            gamma=0.7
        )
    ))

    tests.append((
        "combined_realistic_2",
        "Combined: Tilt + horizontal squash + dark",
        apply_contrast_variation(
            apply_aspect_distortion(
                apply_perspective_distortion(original, tilt_x=8, tilt_y=5),
                horizontal_scale=0.9
            ),
            gamma=0.6,
            brightness=-15
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
            grid-template-columns: 1fr 1fr;
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
    processed_dir = output_dir / "processed"

    # Create output directories
    variations_dir.mkdir(parents=True, exist_ok=True)
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
        success, error_msg, anchors, roll_bubbles, questions, processed = \
            process_distorted_image(distorted_image, geom, layout, sheet, markers_cfg)
        processing_time_ms = (time.perf_counter() - start_time) * 1000

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
