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

import itertools
import math

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


@dataclass
class _AnchorCandidate:
    center: Tuple[float, float]
    area: float
    rect: Tuple[Tuple[float, float], Tuple[float, float], float]
    aspect_ratio: float
    contour: np.ndarray
    label: Optional[str]


def detect_anchor_markers(
    image: np.ndarray,
    geom: PageGeometry,
    markers: MarkerConfig,
) -> Optional[List[Tuple[int, int]]]:
    """Detect the four corner anchor markers.

    The detector looks for orientation-encoded fiducials printed at the page
    corners. It evaluates multiple contour combinations, scoring them by
    proximity to the image corners, geometric plausibility, and alignment with
    the vertical grid marker columns. Candidate sets that do not reproduce the
    expected grid spacing after rectification are rejected.
    """

    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    height, width = gray.shape
    diag = math.hypot(width, height)
    image_area = float(width * height)

    expected_anchor_w = (markers.anchor_size / geom.width) * width
    expected_anchor_h = (markers.anchor_size / geom.height) * height
    expected_anchor_area = max(expected_anchor_w * expected_anchor_h, image_area * 0.0005)

    expected_grid_w = (markers.grid_marker_size / geom.width) * width
    expected_grid_h = (markers.grid_marker_size / geom.height) * height
    expected_grid_area = max(expected_grid_w * expected_grid_h, image_area * 0.00005)

    anchor_candidates: List[_AnchorCandidate] = []
    grid_candidates: Dict[str, List[Tuple[float, float]]] = {"left": [], "right": []}

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _ = rect
        if rw <= 0 or rh <= 0:
            continue

        aspect_ratio = max(rw, rh) / max(min(rw, rh), 1e-6)

        if expected_anchor_area * 0.35 <= area <= expected_anchor_area * 3.5 and aspect_ratio <= 3.0:
            label = _decode_anchor_label(gray, rect)
            anchor_candidates.append(
                _AnchorCandidate(
                    center=(cx, cy),
                    area=area,
                    rect=rect,
                    aspect_ratio=aspect_ratio,
                    contour=contour,
                    label=label,
                )
            )
        elif expected_grid_area * 0.4 <= area <= expected_grid_area * 5.0 and aspect_ratio <= 4.0:
            if cx < width / 2:
                grid_candidates["left"].append((cx, cy))
            else:
                grid_candidates["right"].append((cx, cy))

    if len(anchor_candidates) < 4:
        return None

    # Keep top-N candidates closest to any corner to reduce combinatorics
    corner_targets = {
        "top_left": np.array([0.0, 0.0]),
        "top_right": np.array([float(width - 1), 0.0]),
        "bottom_left": np.array([0.0, float(height - 1)]),
        "bottom_right": np.array([float(width - 1), float(height - 1)]),
    }

    def _min_corner_distance(candidate: _AnchorCandidate) -> float:
        center = np.array(candidate.center)
        return min(np.linalg.norm(center - target) for target in corner_targets.values())

    anchor_candidates = sorted(anchor_candidates, key=_min_corner_distance)[:16]

    best_assignment: Optional[Dict[str, _AnchorCandidate]] = None
    best_score = float("inf")
    corner_order = ["top_left", "top_right", "bottom_left", "bottom_right"]

    for combo in itertools.combinations(anchor_candidates, 4):
        cost_matrix = []
        for corner_name in corner_order:
            target = corner_targets[corner_name]
            row = []
            for candidate in combo:
                center = np.array(candidate.center)
                distance = np.linalg.norm(center - target)
                distance_score = distance / diag
                size_ratio = candidate.area / expected_anchor_area if expected_anchor_area > 0 else 1.0
                size_penalty = abs(math.log(max(size_ratio, 1e-6))) * 0.2
                aspect_penalty = max(0.0, candidate.aspect_ratio - 1.2) * 0.15
                label_penalty = 0.0 if (candidate.label is None or candidate.label == corner_name) else 0.5
                row.append(distance_score + size_penalty + aspect_penalty + label_penalty)
            cost_matrix.append(row)

        for perm in itertools.permutations(range(4)):
            assignment = {corner_order[i]: combo[perm[i]] for i in range(4)}
            ordered_points = [
                np.array(assignment["top_left"].center),
                np.array(assignment["top_right"].center),
                np.array(assignment["bottom_left"].center),
                np.array(assignment["bottom_right"].center),
            ]

            if not _quadrilateral_is_valid(ordered_points, width, height):
                continue

            base_score = sum(cost_matrix[i][perm[i]] for i in range(4))
            shape_penalty = _quadrilateral_shape_penalty(ordered_points, geom)
            grid_penalty = _grid_alignment_penalty(assignment, grid_candidates, diag)
            total_score = base_score + shape_penalty + grid_penalty

            if total_score < best_score:
                best_score = total_score
                best_assignment = assignment

    if not best_assignment:
        return None

    ordered = [
        best_assignment["top_left"].center,
        best_assignment["top_right"].center,
        best_assignment["bottom_left"].center,
        best_assignment["bottom_right"].center,
    ]

    if not _validate_with_grid(gray, ordered, geom, markers):
        return None

    return [(int(round(x)), int(round(y))) for x, y in ordered]


def _compute_output_dimensions(geom: PageGeometry) -> Tuple[int, int]:
    aspect_ratio = geom.width / geom.height if geom.height else 1.0
    output_height = 1400
    output_width = max(1, int(round(output_height * aspect_ratio)))
    return output_width, output_height


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
    output_width, output_height = _compute_output_dimensions(geom)

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


def _decode_anchor_label(
    gray: np.ndarray,
    rect: Tuple[Tuple[float, float], Tuple[float, float], float],
) -> Optional[str]:
    (cx, cy), (rw, rh), angle = rect
    if rw < 2 or rh < 2:
        return None

    width, height = rw, rh
    rotation = angle
    if width < height:
        width, height = height, width
        rotation += 90.0

    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), rotation, 1.0)
    rotated = cv2.warpAffine(
        gray,
        rotation_matrix,
        (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    patch_size = (int(round(width)), int(round(height)))
    if patch_size[0] < 2 or patch_size[1] < 2:
        return None

    patch = cv2.getRectSubPix(rotated, patch_size, (cx, cy))
    if patch is None or patch.size == 0:
        return None

    norm_size = 64
    patch = cv2.resize(patch, (norm_size, norm_size))
    _, thresh = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    if num_labels < 3:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return None

    largest_idx = 1 + int(np.argmax(areas))
    outer_area = float(stats[largest_idx, cv2.CC_STAT_AREA])
    dot_candidates = [
        idx
        for idx in range(1, num_labels)
        if idx != largest_idx and 5 <= stats[idx, cv2.CC_STAT_AREA] <= outer_area * 0.35
    ]
    if not dot_candidates:
        return None

    dot_idx = max(dot_candidates, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])
    dot_cx, dot_cy = centroids[dot_idx]

    half = norm_size / 2.0
    horizontal = "left" if dot_cx < half else "right"
    vertical = "top" if dot_cy < half else "bottom"
    label = f"{vertical}_{horizontal}"
    if label not in {"top_left", "top_right", "bottom_left", "bottom_right"}:
        return None
    return label


def _quadrilateral_is_valid(
    points: List[np.ndarray], width: int, height: int
) -> bool:
    for x, y in points:
        if x < 0 or y < 0 or x >= width or y >= height:
            return False

    polygon = np.array(
        [points[0], points[1], points[3], points[2]], dtype=np.float32
    )  # TL, TR, BR, BL
    area = cv2.contourArea(polygon)
    if area < (width * height) * 0.05:
        return False

    for i in range(4):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % 4]
        p2 = polygon[(i + 2) % 4]
        cross = float(np.cross(p1 - p0, p2 - p1))
        if cross <= 0:
            return False

    return True


def _quadrilateral_shape_penalty(points: List[np.ndarray], geom: PageGeometry) -> float:
    top_width = np.linalg.norm(points[1] - points[0])
    bottom_width = np.linalg.norm(points[3] - points[2])
    left_height = np.linalg.norm(points[2] - points[0])
    right_height = np.linalg.norm(points[3] - points[1])

    avg_width = (top_width + bottom_width) / 2.0
    avg_height = (left_height + right_height) / 2.0

    if avg_width <= 0 or avg_height <= 0:
        return float("inf")

    expected_ratio = geom.width / geom.height if geom.height else avg_width / avg_height
    ratio = avg_width / avg_height
    ratio_penalty = abs(math.log(max(ratio / expected_ratio, 1e-6)))

    width_delta = abs(top_width - bottom_width) / max(avg_width, 1e-6)
    height_delta = abs(left_height - right_height) / max(avg_height, 1e-6)
    skew_penalty = (width_delta + height_delta) * 0.4

    return ratio_penalty + skew_penalty


def _grid_alignment_penalty(
    assignment: Dict[str, _AnchorCandidate],
    grid_candidates: Dict[str, List[Tuple[float, float]]],
    diag: float,
) -> float:
    penalty = 0.0
    for side, corners in ("left", ("top_left", "bottom_left")), ("right", ("top_right", "bottom_right")):
        samples = grid_candidates.get(side, [])
        if not samples:
            penalty += 0.4
            continue

        p1 = np.array(assignment[corners[0]].center)
        p2 = np.array(assignment[corners[1]].center)
        line_penalty = _average_distance_to_line(samples, p1, p2)
        penalty += (line_penalty / max(diag, 1e-6)) * 2.5

    return penalty


def _average_distance_to_line(
    samples: List[Tuple[float, float]], p1: np.ndarray, p2: np.ndarray
) -> float:
    line_vec = p2 - p1
    norm = math.hypot(line_vec[0], line_vec[1])
    if norm < 1e-6:
        return float("inf")

    total = 0.0
    for sx, sy in samples:
        diff_x = sx - p1[0]
        diff_y = sy - p1[1]
        distance = abs(line_vec[0] * diff_y - line_vec[1] * diff_x) / norm
        total += distance

    return total / max(len(samples), 1)


def _validate_with_grid(
    gray: np.ndarray,
    points: List[Tuple[float, float]],
    geom: PageGeometry,
    markers: MarkerConfig,
) -> bool:
    output_width, output_height = _compute_output_dimensions(geom)
    src = np.float32(points)
    dst = np.float32(
        [
            [0, 0],
            [output_width - 1, 0],
            [0, output_height - 1],
            [output_width - 1, output_height - 1],
        ]
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray, matrix, (output_width, output_height))

    return _grid_columns_validate_rectified(warped, geom, markers)


def _grid_columns_validate_rectified(
    corrected_gray: np.ndarray, geom: PageGeometry, markers: MarkerConfig
) -> bool:
    if corrected_gray.ndim == 3:
        corrected_gray = cv2.cvtColor(corrected_gray, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        corrected_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    height, width = binary.shape
    expected_spacing = (markers.grid_spacing / geom.height) * height if geom.height else 0

    expected_grid_area = (
        (markers.grid_marker_size / geom.width) * width
        * (markers.grid_marker_size / geom.height) * height
        if geom.width and geom.height
        else width * height * 0.00002
    )

    left_x = (geom.margin / 2 + markers.grid_marker_size / 2) / geom.width * width
    right_x = (
        geom.width - geom.margin / 2 - markers.grid_marker_size / 2
    ) / geom.width * width
    tolerance_x = max((markers.grid_marker_size / geom.width) * width * 1.5, 10)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left_points: List[Tuple[float, float]] = []
    right_points: List[Tuple[float, float]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        if area < expected_grid_area * 0.3 or area > expected_grid_area * 4.0:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        if abs(cx - left_x) <= tolerance_x:
            left_points.append((cx, cy))
        elif abs(cx - right_x) <= tolerance_x:
            right_points.append((cx, cy))

    return _grid_column_valid(left_points, expected_spacing, height) and _grid_column_valid(
        right_points, expected_spacing, height
    )


def _grid_column_valid(
    points: List[Tuple[float, float]], expected_spacing: float, image_height: int
) -> bool:
    if len(points) < 4 or expected_spacing <= 0:
        return False

    ordered = sorted(points, key=lambda p: p[1])
    diffs = [ordered[i + 1][1] - ordered[i][1] for i in range(len(ordered) - 1)]
    if not diffs:
        return False

    median_spacing = float(np.median(diffs))
    if median_spacing <= 0:
        return False

    ratio = median_spacing / expected_spacing
    if ratio < 0.65 or ratio > 1.35:
        return False

    coverage = ordered[-1][1] - ordered[0][1]
    if coverage < image_height * 0.4:
        return False

    return True



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

    # Analyze fill state
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

    # Process roll bubbles
    roll_groups: List[List[Bubble]] = [[] for _ in range(sheet.roll_rows)]
    for coord in roll_coords:
        bubble = _coordinate_to_bubble(
            coord, geom, markers, width, height, margin_x, margin_y, inner_gray, layout
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
            coord, geom, markers, width, height, margin_x, margin_y, inner_gray, layout
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

    # Detect anchor markers
    anchor_points = detect_anchor_markers(image, geom, markers_cfg)
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
