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

import csv
import math
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from omr_config import PageGeometry, BubbleLayout, MarkerConfig, SheetLayout, AnchorDetectionZones
from omr_layout import (
    BubbleCoordinate,
    BubbleGroup,
    calculate_anchor_centers,
    generate_all_bubble_coordinates,
)


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


@dataclass
class BubbleDetection:
    """Pairing between a layout coordinate and a sampled bubble."""

    layout: BubbleCoordinate
    bubble: Bubble


@dataclass
class BubbleGroupSample:
    """Detected bubbles organized by their originating layout group."""

    group: BubbleGroup
    detections: List[BubbleDetection]


RowId = Tuple[str, int]


@dataclass
class GridCalibration:
    """Encapsulates Y-axis calibration derived from grid markers."""

    slope: float
    intercept: float
    residuals: Dict[RowId, float]
    coverage: float
    matched_rows: int

    def apply(self, row_id: RowId, y_value: float) -> float:
        """Apply calibration to a Y pixel coordinate for a specific row."""

        adjusted = self.slope * y_value + self.intercept
        return adjusted + self.residuals.get(row_id, 0.0)


def detect_anchor_markers(
    image: np.ndarray,
    geom: PageGeometry,
    markers_cfg: MarkerConfig,
    zones_cfg: Optional[AnchorDetectionZones] = None
) -> Optional[List[Tuple[int, int]]]:
    """Detect the four corner anchor markers with validation.

    Args:
        image: Input image
        geom: Page geometry configuration
        markers_cfg: Marker configuration with expected anchor size
        zones_cfg: Anchor detection zones configuration (optional, uses defaults if not provided)

    Returns:
        List of (x, y) coordinates for [top-left, top-right, bottom-left, bottom-right]
        or None if detection fails.
    """
    if zones_cfg is None:
        zones_cfg = AnchorDetectionZones()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Improve contrast to help in uneven lighting conditions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Gentle blur removes speckle noise before thresholding
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Use Otsu's automatic thresholding to handle varying lighting conditions
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Close small gaps inside markers - using 2 iterations for better robustness
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    img_height, img_width = gray.shape

    # Calculate expected anchor size in pixels
    # Assume image corresponds roughly to the page geometry
    scale = img_width / geom.width
    expected_anchor_size = markers_cfg.anchor_size * scale
    expected_area = expected_anchor_size ** 2

    expected_centers = calculate_anchor_centers(geom, markers_cfg)
    scale_x = img_width / geom.width if geom.width > 0 else 1.0
    scale_y = img_height / geom.height if geom.height > 0 else 1.0

    expected_points_px = [
        (
            int(round(expected_centers["top_left"][0] * scale_x)),
            int(round(img_height - expected_centers["top_left"][1] * scale_y)),
        ),
        (
            int(round(expected_centers["top_right"][0] * scale_x)),
            int(round(img_height - expected_centers["top_right"][1] * scale_y)),
        ),
        (
            int(round(expected_centers["bottom_left"][0] * scale_x)),
            int(round(img_height - expected_centers["bottom_left"][1] * scale_y)),
        ),
        (
            int(round(expected_centers["bottom_right"][0] * scale_x)),
            int(round(img_height - expected_centers["bottom_right"][1] * scale_y)),
        ),
    ]

    def collect_candidates(binary_img: np.ndarray, size_scale: Tuple[float, float], corner_expand: float) -> List[Tuple[int, int]]:
        min_area = expected_area * size_scale[0]
        max_area = expected_area * size_scale[1]

        corner_band_x, corner_band_y_top, corner_band_y_bottom = zones_cfg.get_zones(
            img_width, img_height, corner_expand
        )

        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        local_candidates: List[Tuple[int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            if not (0.6 < aspect_ratio < 1.4):
                continue

            center_x = x + w // 2
            center_y = y + h // 2

            in_left_band = center_x < corner_band_x
            in_right_band = center_x > (img_width - corner_band_x)
            in_top_band = center_y < corner_band_y_top
            in_bottom_band = center_y > (img_height - corner_band_y_bottom)

            if not ((in_left_band or in_right_band) and (in_top_band or in_bottom_band)):
                continue

            local_candidates.append((center_x, center_y))

        return local_candidates

    # First pass with stricter bounds
    candidates = collect_candidates(binary, (0.5, 2.0), 0.0)

    # Retry with adaptive thresholding and relaxed geometry if we missed markers
    if len(candidates) < 4:
        relaxed = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            5,
        )
        relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        relaxed_candidates = collect_candidates(relaxed, (0.3, 3.0), zones_cfg.relaxed_expansion)
        candidates.extend(relaxed_candidates)

    # Deduplicate candidate coordinates while preserving order
    unique_candidates: List[Tuple[int, int]] = []
    seen = set()
    for pt in candidates:
        if pt not in seen:
            unique_candidates.append(pt)
            seen.add(pt)

    if len(unique_candidates) < 4:
        print(f"Warning: Only found {len(unique_candidates)} anchor candidates (expected 4)")
        return None

    markers = _select_best_marker_set(unique_candidates, expected_points_px)
    if markers is None:
        print("Warning: Unable to select consistent anchor markers from candidates")
        return None

    # Validate geometric layout: check if markers form a reasonable rectangle
    if not _validate_rectangle_geometry(markers, img_width, img_height):
        print("Warning: Anchor markers do not form a valid rectangular layout")
        return None

    return markers


def detect_grid_markers(
    corrected: np.ndarray,
    geom: PageGeometry,
    markers_cfg: MarkerConfig,
) -> List[Tuple[int, int]]:
    """Detect rectangular grid markers along the left/right margins."""

    if not markers_cfg.grid_calibration_enabled:
        return []

    if corrected.ndim == 3:
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    else:
        gray = corrected

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    img_height, img_width = gray.shape

    scale_x = img_width / geom.width if geom.width > 0 else 1.0
    scale_y = img_height / geom.height if geom.height > 0 else 1.0

    expected_width = max(markers_cfg.grid_marker_size * 2.0 * scale_x, 1.0)
    expected_height = max(markers_cfg.grid_marker_size * scale_y, 1.0)
    expected_area = expected_width * expected_height

    area_tol = markers_cfg.grid_marker_area_tolerance
    min_area = expected_area * max(0.1, 1.0 - area_tol)
    max_area = expected_area * (1.0 + area_tol)

    expected_aspect = expected_width / expected_height if expected_height > 0 else 2.0
    aspect_tol = markers_cfg.grid_marker_aspect_tolerance
    min_aspect = expected_aspect * (1.0 - aspect_tol)
    max_aspect = expected_aspect * (1.0 + aspect_tol)

    margin_band = int(round(geom.margin * scale_x * 1.5 + expected_width))
    margin_band = max(margin_band, int(expected_width * 2))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers: List[Tuple[int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue

        aspect_ratio = w / h
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue

        center_x = x + w // 2
        center_y = y + h // 2

        if not (center_x < margin_band or center_x > (img_width - margin_band)):
            continue

        centers.append((center_x, center_y))

    centers.sort(key=lambda pt: pt[1])
    return centers


def _compute_grid_calibration(
    row_expectations: List[Tuple[RowId, float]],
    markers: List[Tuple[int, int]],
    layout: BubbleLayout,
    markers_cfg: MarkerConfig,
    scale_y: float,
) -> Optional[GridCalibration]:
    """Estimate per-row Y adjustments using detected grid markers."""

    if not markers_cfg.grid_calibration_enabled:
        return None

    if not row_expectations or not markers:
        return None

    vertical_gap_px = layout.vertical_gap * scale_y if layout.vertical_gap > 0 else 0.0
    base_distance = markers_cfg.grid_marker_size * scale_y
    tolerance = max(vertical_gap_px * markers_cfg.grid_marker_distance_limit, base_distance)
    tolerance = max(tolerance, 4.0)

    # Map each row to the closest detected marker within tolerance
    matches: Dict[RowId, Tuple[float, float, float]] = {}
    for _, marker_y in markers:
        best_row: Optional[Tuple[RowId, float]] = None
        best_dist = float("inf")
        for row_id, expected_y in row_expectations:
            dist = abs(marker_y - expected_y)
            if dist < best_dist:
                best_dist = dist
                best_row = (row_id, expected_y)

        if best_row is None or best_dist > tolerance:
            continue

        row_id, expected_y = best_row
        existing = matches.get(row_id)
        if existing is None or best_dist < existing[2]:
            matches[row_id] = (expected_y, float(marker_y), best_dist)

    if not matches:
        return None

    matched_rows = [
        (row_id, expected, observed)
        for row_id, (expected, observed, _dist) in matches.items()
    ]
    matched_rows.sort(key=lambda item: item[1])

    diffs = np.array([obs - exp for _, exp, obs in matched_rows], dtype=np.float64)
    if diffs.size == 0:
        return None

    median_diff = float(np.median(diffs))
    abs_deviation = np.abs(diffs - median_diff)
    mad = float(np.median(abs_deviation))

    if mad > 1e-6:
        normalized = 0.6745 * (diffs - median_diff) / mad
        mask = np.abs(normalized) <= markers_cfg.grid_marker_outlier_sigma
    else:
        mask = abs_deviation <= tolerance

    filtered_rows = [row for row, keep in zip(matched_rows, mask) if keep]

    if not filtered_rows:
        return None

    coverage = len(filtered_rows) / len(row_expectations)
    min_required = min(markers_cfg.grid_calibration_min_matches, len(row_expectations))

    if len(filtered_rows) < max(2, min_required) or coverage < markers_cfg.grid_calibration_min_fraction:
        return None

    expected_vals = np.array([exp for _, exp, _ in filtered_rows], dtype=np.float64)
    observed_vals = np.array([obs for _, _, obs in filtered_rows], dtype=np.float64)

    if expected_vals.size < 2:
        return None

    slope, intercept = np.polyfit(expected_vals, observed_vals, 1)
    slope = float(slope)
    intercept = float(intercept)

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None

    if abs(slope - 1.0) > markers_cfg.grid_marker_scale_tolerance:
        return None

    residuals: Dict[RowId, float] = {}
    for row_id, expected_y, observed_y in filtered_rows:
        predicted = slope * expected_y + intercept
        residuals[row_id] = float(observed_y - predicted)

    return GridCalibration(
        slope=slope,
        intercept=intercept,
        residuals=residuals,
        coverage=coverage,
        matched_rows=len(filtered_rows),
    )


def _select_best_marker_set(
    candidates: List[Tuple[int, int]],
    expected_points: List[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """Pick the four candidate points that best match the expected corners."""

    if len(candidates) < 4:
        return None

    # Keep only the candidates closest to any expected corner to limit combinations
    if len(candidates) > 8:
        def min_distance(pt: Tuple[int, int]) -> float:
            return min(
                math.hypot(pt[0] - exp[0], pt[1] - exp[1]) for exp in expected_points
            )

        candidates = sorted(candidates, key=min_distance)[:8]

    indices = list(range(len(candidates)))
    best_score = float("inf")
    best_assignment: Optional[List[Tuple[int, int]]] = None

    for combo in combinations(indices, 4):
        for perm in permutations(combo):
            score = 0.0
            for idx, expected in zip(perm, expected_points):
                candidate = candidates[idx]
                score += (candidate[0] - expected[0]) ** 2 + (candidate[1] - expected[1]) ** 2

            if score < best_score:
                best_score = score
                best_assignment = [candidates[idx] for idx in perm]

    return best_assignment


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


def correct_skew(
    image: np.ndarray,
    markers: List[Tuple[int, int]],
    geom: PageGeometry,
    markers_cfg: MarkerConfig,
) -> np.ndarray:
    """Apply perspective transformation to correct skew using shared anchor geometry."""

    tl, tr, bl, br = markers

    expected_centers = calculate_anchor_centers(geom, markers_cfg)
    exp_tl = expected_centers["top_left"]
    exp_tr = expected_centers["top_right"]
    exp_bl = expected_centers["bottom_left"]
    exp_br = expected_centers["bottom_right"]

    bubble_width_pts = max(exp_tr[0] - exp_tl[0], 1e-6)
    bubble_height_pts = max(exp_tl[1] - exp_bl[1], 1e-6)

    top_edge = np.linalg.norm(np.array(tr) - np.array(tl))
    bottom_edge = np.linalg.norm(np.array(br) - np.array(bl))
    left_edge = np.linalg.norm(np.array(bl) - np.array(tl))
    right_edge = np.linalg.norm(np.array(br) - np.array(tr))

    avg_width_pixels = (top_edge + bottom_edge) / 2.0
    avg_height_pixels = (left_edge + right_edge) / 2.0

    scale_x = avg_width_pixels / bubble_width_pts
    scale_y = avg_height_pixels / bubble_height_pts
    scale = max((scale_x + scale_y) / 2.0, 1e-6)

    # Compute output dimensions preserving the detected resolution
    output_width = max(50, int(round(geom.width * scale)))
    output_height = max(50, int(round(geom.height * scale)))

    scale_x_out = output_width / geom.width if geom.width > 0 else 1.0
    scale_y_out = output_height / geom.height if geom.height > 0 else 1.0

    src_points = np.float32(markers)
    dst_points = np.float32([
        [exp_tl[0] * scale_x_out, output_height - exp_tl[1] * scale_y_out],
        [exp_tr[0] * scale_x_out, output_height - exp_tr[1] * scale_y_out],
        [exp_bl[0] * scale_x_out, output_height - exp_bl[1] * scale_y_out],
        [exp_br[0] * scale_x_out, output_height - exp_br[1] * scale_y_out],
    ])

    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply transformation
    corrected = cv2.warpPerspective(image, matrix, (output_width, output_height))

    img_height, img_width = image.shape[:2]
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
    page_width: int,
    page_height: int,
    margin_x: int,
    margin_y: int,
    inner_gray: np.ndarray,
    layout: BubbleLayout,
    adaptive_threshold: float,
    row_id: RowId,
    calibration: Optional[GridCalibration],
) -> Optional[Bubble]:
    """Convert a PDF coordinate to pixel coordinates and sample fill state."""

    if geom.width <= 0 or geom.height <= 0:
        return None

    scale_x = page_width / geom.width
    scale_y = page_height / geom.height

    abs_x_float = coord.x * scale_x
    abs_y_float = page_height - coord.y * scale_y

    if calibration is not None:
        abs_y_float = calibration.apply(row_id, abs_y_float)

    abs_x = int(round(abs_x_float))
    abs_y = int(round(abs_y_float))
    abs_x = min(max(abs_x, 0), max(page_width - 1, 0))
    abs_y = min(max(abs_y, 0), max(page_height - 1, 0))

    inner_x = abs_x - margin_x
    inner_y = abs_y - margin_y

    if (
        inner_x < 0
        or inner_y < 0
        or inner_x >= inner_gray.shape[1]
        or inner_y >= inner_gray.shape[0]
    ):
        return None

    radius_scale = (scale_x + scale_y) / 2.0
    radius = max(3, int(round(coord.radius * radius_scale)))

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
) -> List[BubbleGroupSample]:
    """Generate bubble coordinates procedurally and sample their fill states."""

    if corrected.size == 0:
        return []

    # Generate bubble coordinates using shared logic
    layout_groups, _, _, _ = generate_all_bubble_coordinates(geom, layout, sheet, markers)

    height, width = corrected.shape[:2]
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if corrected.ndim == 3 else corrected

    scale_x = width / geom.width if geom.width > 0 else 1.0
    scale_y = height / geom.height if geom.height > 0 else 1.0

    row_expectations: List[Tuple[RowId, float]] = []
    seen_rows: Set[RowId] = set()
    for group in layout_groups:
        if not group.bubbles:
            continue
        row_id = (group.category, group.group_index)
        if row_id in seen_rows:
            continue
        expected_y = height - group.bubbles[0].y * scale_y
        row_expectations.append((row_id, expected_y))
        seen_rows.add(row_id)

    row_expectations.sort(key=lambda item: item[1])

    centers = calculate_anchor_centers(geom, markers)
    half_anchor = markers.anchor_size / 2.0
    half_anchor_x = int(round(half_anchor * scale_x))
    half_anchor_y = int(round(half_anchor * scale_y))

    left_center_px = int(round(centers["bottom_left"][0] * scale_x))
    right_center_px = int(round(centers["bottom_right"][0] * scale_x))
    top_center_px = int(round(height - centers["top_left"][1] * scale_y))
    bottom_center_px = int(round(height - centers["bottom_left"][1] * scale_y))

    left_crop = max(0, left_center_px - half_anchor_x)
    right_crop = min(width, right_center_px + half_anchor_x)
    top_crop = max(0, top_center_px - half_anchor_y)
    bottom_crop = min(height, bottom_center_px + half_anchor_y)

    if right_crop <= left_crop or bottom_crop <= top_crop:
        inner_gray = gray
        margin_x = 0
        margin_y = 0
    else:
        inner_gray = gray[top_crop:bottom_crop, left_crop:right_crop]
        margin_x = left_crop
        margin_y = top_crop

    if inner_gray.size == 0:
        inner_gray = gray
        margin_x = 0
        margin_y = 0

    # Calculate adaptive fill threshold based on inner region's contrast profile
    # Use inner_gray to exclude anchor markers which would skew percentiles
    adaptive_threshold = calculate_adaptive_fill_threshold(inner_gray, layout.fill_threshold)

    calibration: Optional[GridCalibration] = None
    if markers.grid_calibration_enabled:
        grid_markers = detect_grid_markers(corrected, geom, markers)
        calibration = _compute_grid_calibration(
            row_expectations,
            grid_markers,
            layout,
            markers,
            scale_y,
        )
        if calibration is not None:
            print(
                "Grid calibration matched "
                f"{calibration.matched_rows} rows "
                f"({calibration.coverage * 100:.1f}% coverage, scale {calibration.slope:.4f})"
            )

    group_samples: List[BubbleGroupSample] = []

    for group in layout_groups:
        detections: List[BubbleDetection] = []
        row_id = (group.category, group.group_index)
        for coord in group.bubbles:
            bubble = _coordinate_to_bubble(
                coord,
                geom,
                width,
                height,
                margin_x,
                margin_y,
                inner_gray,
                layout,
                adaptive_threshold,
                row_id,
                calibration,
            )
            if bubble is not None:
                detections.append(BubbleDetection(layout=coord, bubble=bubble))

        detections.sort(key=lambda item: item.layout.index)
        group_samples.append(BubbleGroupSample(group=group, detections=detections))

    return group_samples

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

    radius = max(radius, 3)

    ring_inner_radius = max(radius * 1.2, radius + 1)
    ring_outer_radius = max(ring_inner_radius + 2, radius * 1.8)

    # Extract local region around bubble (optimization: only process nearby pixels)
    # Region size: enough to include background ring
    region_radius = int(math.ceil(ring_outer_radius))
    x_min = max(0, x - region_radius)
    x_max = min(w, x + region_radius + 1)
    y_min = max(0, y - region_radius)
    y_max = min(h, y + region_radius + 1)

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
    background_mask = (
        (distance_from_center >= ring_inner_radius)
        & (distance_from_center <= ring_outer_radius)
    )

    if not np.any(background_mask):
        expanded_outer = ring_inner_radius + max(3, radius)
        background_mask = (
            (distance_from_center >= ring_inner_radius)
            & (distance_from_center <= expanded_outer)
        )

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


def overlay_labels(image: np.ndarray, group_samples: List[BubbleGroupSample]) -> np.ndarray:
    """Overlay group and option labels on detected bubbles."""

    output = image.copy()

    # First pass: Draw pink highlights for all filled bubbles
    pink_color = (255, 0, 255)  # Bright magenta/pink in BGR
    for sample in group_samples:
        for detection in sample.detections:
            bubble = detection.bubble
            if bubble.is_filled:
                cv2.circle(output, (bubble.x, bubble.y), bubble.radius + 2, pink_color, 2)

    # Second pass: Render textual labels based on group category
    for sample in group_samples:
        if not sample.detections:
            continue

        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1

        if sample.group.category == "roll":
            # Roll number: Draw digit label inside each bubble
            for detection in sample.detections:
                text = detection.layout.label
                font_scale = 0.4
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = detection.bubble.x - text_size[0] // 2
                text_y = detection.bubble.y + text_size[1] // 2
                cv2.putText(output, text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness)

        elif sample.group.category == "class":
            # Class: Draw "Class" label above first bubble and class numbers inside bubbles
            first_detection = sample.detections[0]
            label_text = "Class"
            font_scale = 0.35
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            text_x = first_detection.bubble.x - text_size[0] // 2
            text_y = first_detection.bubble.y - first_detection.bubble.radius - 5
            cv2.putText(output, label_text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

            # Draw class number inside each bubble
            for detection in sample.detections:
                class_text = detection.layout.label
                class_size = cv2.getTextSize(class_text, font, 0.4, thickness)[0]
                text_x = detection.bubble.x - class_size[0] // 2
                text_y = detection.bubble.y + class_size[1] // 2
                cv2.putText(
                    output,
                    class_text,
                    (text_x, text_y),
                    font,
                    0.4,
                    (0, 128, 0),
                    thickness,
                )

        elif sample.group.category == "class_section":
            # Class section: Draw "Sec" label above first bubble and section letters inside bubbles
            first_detection = sample.detections[0]
            label_text = "Sec"
            font_scale = 0.35
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            text_x = first_detection.bubble.x - text_size[0] // 2
            text_y = first_detection.bubble.y - first_detection.bubble.radius - 5
            cv2.putText(output, label_text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

            # Draw section letter (a/b/c/d) inside each bubble
            for detection in sample.detections:
                section_text = detection.layout.label
                section_size = cv2.getTextSize(section_text, font, 0.4, thickness)[0]
                text_x = detection.bubble.x - section_size[0] // 2
                text_y = detection.bubble.y + section_size[1] // 2
                cv2.putText(
                    output,
                    section_text,
                    (text_x, text_y),
                    font,
                    0.4,
                    (0, 128, 0),
                    thickness,
                )

        elif sample.group.category == "set":
            # Set: Draw "Set" label above first bubble and option letters inside bubbles
            first_detection = sample.detections[0]
            label_text = "Set"
            font_scale = 0.35
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            text_x = first_detection.bubble.x - text_size[0] // 2
            text_y = first_detection.bubble.y - first_detection.bubble.radius - 5
            cv2.putText(output, label_text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

            # Draw option label (A/B/C/D) inside each bubble
            for detection in sample.detections:
                option_text = detection.layout.label
                option_size = cv2.getTextSize(option_text, font, 0.4, thickness)[0]
                text_x = detection.bubble.x - option_size[0] // 2
                text_y = detection.bubble.y + option_size[1] // 2
                cv2.putText(
                    output,
                    option_text,
                    (text_x, text_y),
                    font,
                    0.4,
                    (0, 128, 0),
                    thickness,
                )

        elif sample.group.category == "question":
            # Question: Draw question label above and option labels inside bubbles
            first_detection = sample.detections[0]
            q_text = f"Q{sample.group.display_label}"
            font_scale = 0.35
            text_size = cv2.getTextSize(q_text, font, font_scale, thickness)[0]
            text_x = first_detection.bubble.x - text_size[0] // 2
            text_y = first_detection.bubble.y - first_detection.bubble.radius - 5
            cv2.putText(output, q_text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

            # Draw option label inside each bubble
            for detection in sample.detections:
                option_text = detection.layout.label
                option_size = cv2.getTextSize(option_text, font, 0.4, thickness)[0]
                text_x = detection.bubble.x - option_size[0] // 2
                text_y = detection.bubble.y + option_size[1] // 2
                cv2.putText(
                    output,
                    option_text,
                    (text_x, text_y),
                    font,
                    0.4,
                    (0, 128, 0),
                    thickness,
                )

    return output


def export_to_csv(
    filename: str,
    group_samples: List[GroupSample],
    csv_path: Path,
    sheet: SheetLayout,
) -> None:
    """Export detected bubble fills to CSV file.

    Args:
        filename: Name of the source image file
        group_samples: List of bubble group samples with detection results
        csv_path: Path where CSV file will be saved
        sheet: Sheet layout configuration for determining max questions
    """
    # Organize results by category
    results: Dict[str, str] = {}
    results["filename"] = filename

    # Initialize all fields to BLANK
    results["class"] = "BLANK"
    results["class_section"] = "BLANK"
    results["roll_number"] = "BLANK"
    results["set"] = "BLANK"

    for sample in group_samples:
        category = sample.group.category
        filled_labels = [
            detection.layout.label
            for detection in sample.detections
            if detection.bubble.is_filled
        ]

        if category == "class":
            # Single selection expected
            results["class"] = filled_labels[0] if filled_labels else "BLANK"

        elif category == "class_section":
            # Single selection expected
            results["class_section"] = filled_labels[0] if filled_labels else "BLANK"

        elif category == "roll":
            # Multiple columns - organize digits by column index
            if "roll_digits" not in results:
                results["roll_digits"] = {}
            for detection in sample.detections:
                if detection.bubble.is_filled:
                    col_idx = detection.layout.index
                    if col_idx not in results["roll_digits"]:
                        results["roll_digits"][col_idx] = []
                    results["roll_digits"][col_idx].append(detection.layout.label)

        elif category == "set":
            # Single selection expected
            results["set"] = filled_labels[0] if filled_labels else "BLANK"

        elif category == "question":
            # Question number from display_label
            q_num = sample.group.display_label
            # Multiple selections allowed - comma-separated
            if filled_labels:
                results[f"Q{q_num}"] = ",".join(filled_labels)
            else:
                results[f"Q{q_num}"] = "BLANK"

    # Concatenate roll number digits by column order
    if "roll_digits" in results:
        if results["roll_digits"]:
            # Sort by column index and concatenate the first filled digit from each column
            digit_parts = []
            for col_idx in sorted(results["roll_digits"].keys()):
                digits = results["roll_digits"][col_idx]
                # Use first filled digit if multiple (shouldn't happen but handle it)
                digit_parts.append(digits[0] if digits else "")
            results["roll_number"] = "".join(digit_parts) if digit_parts else "BLANK"
        else:
            results["roll_number"] = "BLANK"
        del results["roll_digits"]
    else:
        results["roll_number"] = "BLANK"

    # Determine max questions from sheet config
    max_q = sheet.max_questions if sheet.max_questions else 50

    # Build CSV header
    header = ["filename", "class", "class_section", "roll_number", "set"]
    header.extend([f"Q{i}" for i in range(1, max_q + 1)])

    # Fill missing question columns with BLANK
    for i in range(1, max_q + 1):
        key = f"Q{i}"
        if key not in results:
            results[key] = "BLANK"

    # Write CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(results)

    print(f"Saved CSV results to {csv_path}")


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
    corrected = correct_skew(image, anchor_points, geom, markers_cfg)

    # Sample bubbles at procedurally generated coordinates
    group_samples = sample_bubbles_from_coordinates(
        corrected,
        geom,
        layout,
        sheet,
        markers_cfg,
    )
    # Count bubbles by category
    category_counts = {}
    for sample in group_samples:
        category = sample.group.category
        if category not in category_counts:
            category_counts[category] = {"groups": 0, "bubbles": 0}
        category_counts[category]["groups"] += 1
        category_counts[category]["bubbles"] += len(sample.detections)

    # Build summary message
    summary_parts = []
    for category in ["class", "class_section", "roll", "set", "question"]:
        if category in category_counts:
            counts = category_counts[category]
            if category == "roll":
                summary_parts.append(f"{counts['bubbles']} roll bubbles")
            else:
                summary_parts.append(f"{counts['groups']} {category} groups")

    if summary_parts:
        print(f"Sampled {', '.join(summary_parts)}")

    # Validate that we detected some bubbles
    total_groups = sum(counts["groups"] for counts in category_counts.values())
    if total_groups == 0:
        print("No bubble samples evaluated")
        return False

    # Overlay labels
    labeled = overlay_labels(corrected, group_samples)

    # Save output image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), labeled)
    print(f"Saved processed image to {output_path}")

    # Export results to CSV
    csv_path = output_path.with_suffix('.csv')
    export_to_csv(input_path.name, group_samples, csv_path, sheet)

    return True


def main():
    """Main entry point."""
    import argparse
    import json
    from omr_config_loader import load_sheet_config

    parser = argparse.ArgumentParser(
        description="Process scanned OMR sheets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python omr_processor.py                             # Process with defaults
  python omr_processor.py --questions 50              # Process sheet with 50 questions
  python omr_processor.py --config midterm.json       # Use config file
  python omr_processor.py --config midterm.json --questions 60  # Config + override
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to JSON configuration file (must match generator config)"
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of questions (must match generator, overrides config)"
    )

    args = parser.parse_args()

    # Load configuration
    geom = PageGeometry()
    layout = BubbleLayout()
    markers_cfg = MarkerConfig()

    if args.config:
        # Load from config file
        try:
            sheet = load_sheet_config(args.config)
            print(f"Loaded configuration from: {args.config}")
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"Error loading config file: {e}")
            exit(1)

        # Override question ranges if specified on command line
        if args.questions is not None:
            from dataclasses import replace
            from omr_config import QuestionOptionRange
            # Create a single range with default 4 options per question
            new_ranges = [QuestionOptionRange(start=1, end=args.questions, options=4)]
            sheet = replace(sheet, question_option_ranges=new_ranges)
            print(f"Overriding question ranges to: 1-{args.questions} with 4 options each")
    else:
        # Create sheet layout with question range if specified
        if args.questions is not None:
            from omr_config import QuestionOptionRange
            question_ranges = [QuestionOptionRange(start=1, end=args.questions, options=4)]
            sheet = SheetLayout(question_option_ranges=question_ranges)
        else:
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
