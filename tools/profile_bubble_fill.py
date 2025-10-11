"""Profile the analyze_bubble_fill function to identify bottleneck."""
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2

from omr_config import PageGeometry, BubbleLayout, MarkerConfig, SheetLayout
from omr_processor import detect_anchor_markers, correct_skew, analyze_bubble_fill
from omr_layout import generate_all_bubble_coordinates


def profile_analyze_bubble_fill():
    """Profile the analyze_bubble_fill function."""
    # Load test image
    image_path = Path("sheets/omr_sheet.png")
    image = cv2.imread(str(image_path))

    # Setup
    geom = PageGeometry()
    layout = BubbleLayout()
    markers_cfg = MarkerConfig()

    # Detect and correct
    anchor_points = detect_anchor_markers(image)
    corrected = correct_skew(image, anchor_points, geom)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    print(f"Image size: {w}×{h} = {w*h:,} pixels")

    # Test with a single bubble position
    test_x, test_y = w // 2, h // 2
    test_radius = 18

    print(f"\nTesting analyze_bubble_fill with 100 calls...")
    print(f"Bubble position: ({test_x}, {test_y}), radius: {test_radius}")

    # Time 100 calls
    start = time.perf_counter()
    for i in range(100):
        is_filled, intensity = analyze_bubble_fill(
            gray, test_x, test_y, test_radius, layout.fill_threshold
        )
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / 100) * 1000
    print(f"\nResults:")
    print(f"  Total time for 100 calls: {elapsed:.3f} seconds")
    print(f"  Average per call: {avg_time_ms:.2f} ms")
    print(f"  Estimated time for 602 bubbles: {(avg_time_ms * 602) / 1000:.2f} seconds")

    # Calculate memory usage estimate
    pixels_per_array = w * h
    bytes_per_array = pixels_per_array * 8  # float64
    mb_per_call = (bytes_per_array * 3) / (1024 * 1024)  # 3 large arrays

    print(f"\nMemory allocation per call:")
    print(f"  Array size: {pixels_per_array:,} pixels")
    print(f"  Memory per call: ~{mb_per_call:.1f} MB")
    print(f"  Total for 602 bubbles: ~{mb_per_call * 602:.1f} MB")


if __name__ == "__main__":
    profile_analyze_bubble_fill()
