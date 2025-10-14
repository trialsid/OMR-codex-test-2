"""Profile the analyze_bubble_fill function with real bubble coordinates.

This script profiles the bubble fill analysis by:
- Loading a test OMR sheet image
- Generating all actual bubble coordinates using the procedural layout
- Timing analyze_bubble_fill for each real bubble position
- Reporting per-bubble statistics (min/avg/max) and aggregate totals
- Calculating accurate memory usage based on local region size

Usage:
    python tools/profile_bubble_fill.py

Output format:
    - Image dimensions and total bubble count
    - Per-bubble timing statistics (min/avg/max/total)
    - Memory usage estimates based on actual local region allocations
    - Overall throughput metrics

This ensures profiling reflects the actual production workload.
"""
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
    """Profile the analyze_bubble_fill function with real bubble coordinates."""
    # Load test image
    image_path = Path("sheets/omr_sheet.png")
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    # Setup
    geom = PageGeometry()
    layout = BubbleLayout()
    markers_cfg = MarkerConfig()
    sheet = SheetLayout()

    # Detect and correct
    anchor_points = detect_anchor_markers(image)
    if anchor_points is None:
        print("Error: Could not detect anchor markers")
        return

    corrected = correct_skew(image, anchor_points, geom)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    print(f"Image size: {w}×{h} = {w*h:,} pixels")

    # Generate real bubble coordinates
    roll_coords, question_coords = generate_all_bubble_coordinates(geom, layout, sheet)
    all_coords = roll_coords + question_coords
    total_bubbles = len(all_coords)

    print(f"\nProfiling {total_bubbles} bubbles ({len(roll_coords)} roll + {len(question_coords)} question)")

    # Transform coordinates to pixel space (same logic as in omr_processor.py)
    anchor_inset_x = geom.margin / 2.0 + markers_cfg.anchor_size / 2.0
    anchor_inset_y = geom.margin / 2.0 + markers_cfg.anchor_size / 2.0
    effective_width = geom.width - 2.0 * anchor_inset_x
    effective_height = geom.height - 2.0 * anchor_inset_y

    if effective_width <= 0 or effective_height <= 0:
        print("Error: Invalid effective dimensions")
        return

    scale_x = w / effective_width
    scale_y = h / effective_height

    # Time each bubble individually
    timings = []
    for coord in all_coords:
        # Transform PDF coordinates to pixel coordinates
        rel_x = (coord.x - anchor_inset_x) / effective_width
        rel_y = ((geom.height - anchor_inset_y) - coord.y) / effective_height
        rel_x = min(max(rel_x, 0.0), 1.0)
        rel_y = min(max(rel_y, 0.0), 1.0)

        pixel_x = int(round(rel_x * w))
        pixel_y = int(round(rel_y * h))
        pixel_x = min(max(pixel_x, 0), w - 1)
        pixel_y = min(max(pixel_y, 0), h - 1)
        pixel_radius = max(1, int(round(coord.radius * scale_x)))

        # Time this bubble
        start = time.perf_counter()
        is_filled, intensity = analyze_bubble_fill(
            gray, pixel_x, pixel_y, pixel_radius, layout.fill_threshold
        )
        elapsed = time.perf_counter() - start
        timings.append(elapsed * 1000)  # Convert to ms

    # Calculate statistics
    min_time = min(timings)
    max_time = max(timings)
    avg_time = sum(timings) / len(timings)
    total_time = sum(timings) / 1000  # Convert to seconds

    print(f"\nTiming results:")
    print(f"  Min per bubble:   {min_time:.3f} ms")
    print(f"  Avg per bubble:   {avg_time:.3f} ms")
    print(f"  Max per bubble:   {max_time:.3f} ms")
    print(f"  Total time:       {total_time:.3f} seconds")
    print(f"  Throughput:       {total_bubbles / total_time:.0f} bubbles/sec")

    # Calculate accurate memory usage based on local region size
    # analyze_bubble_fill extracts a local region of size (radius * 2.0 + 1)
    avg_radius = sum(coord.radius for coord in all_coords) / len(all_coords)
    pixel_radius = int(round(avg_radius * scale_x))
    region_size = int(pixel_radius * 2.0) + 1
    local_region_pixels = region_size * region_size

    # Memory per call: local_region (uint8) + 2 float64 arrays (distance grid and masks)
    bytes_local_region = local_region_pixels * 1  # uint8
    bytes_float_arrays = local_region_pixels * 8 * 2  # 2 float64 arrays
    total_bytes = bytes_local_region + bytes_float_arrays
    mb_per_call = total_bytes / (1024 * 1024)

    print(f"\nMemory allocation per call:")
    print(f"  Average region size: {region_size}×{region_size} = {local_region_pixels:,} pixels")
    print(f"  Memory per call:     ~{mb_per_call:.3f} MB")
    print(f"  Total for {total_bubbles} bubbles: ~{mb_per_call * total_bubbles:.1f} MB")


if __name__ == "__main__":
    profile_analyze_bubble_fill()
