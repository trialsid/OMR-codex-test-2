import sys
import types

import pytest

np = pytest.importorskip("numpy")

try:  # pragma: no cover - executed only when OpenCV is available
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for test environment
    def _cvt_color(image, code):
        if image.ndim == 2:
            return image.copy()
        return image.mean(axis=2).astype(image.dtype)

    def _threshold(image, thresh, maxval, mode):
        data = image.astype(np.float32)
        if mode & 8:  # THRESH_OTSU flag value in OpenCV
            thresh = float(data.mean())
        if mode & 1:  # THRESH_BINARY_INV
            binary = np.where(data > thresh, 0, maxval)
        else:
            binary = np.where(data > thresh, maxval, 0)
        return thresh, binary.astype(np.uint8)

    def _count_non_zero(image):
        return int(np.count_nonzero(image))

    def _moments(image):
        mask = image > 0
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return {key: 0.0 for key in ("m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03")}
        xs = xs.astype(float)
        ys = ys.astype(float)
        ones = np.ones_like(xs)
        return {
            "m00": float(ones.sum()),
            "m10": float((xs * ones).sum()),
            "m01": float((ys * ones).sum()),
            "m20": float((xs ** 2).sum()),
            "m11": float((xs * ys).sum()),
            "m02": float((ys ** 2).sum()),
            "m30": float((xs ** 3).sum()),
            "m21": float(((xs ** 2) * ys).sum()),
            "m12": float((xs * (ys ** 2)).sum()),
            "m03": float((ys ** 3).sum()),
        }

    def _circle(image, center, radius, color, thickness=-1):
        x0, y0 = center
        h, w = image.shape[:2]
        y_grid, x_grid = np.ogrid[:h, :w]
        mask = (x_grid - x0) ** 2 + (y_grid - y0) ** 2 <= radius ** 2
        if image.ndim == 2:
            image[mask] = color if not isinstance(color, tuple) else color[0]
        else:
            fill = color if isinstance(color, tuple) else (color,) * image.shape[2]
            for idx in range(image.shape[2]):
                image[..., idx][mask] = fill[idx]
        return image

    def _gaussian_blur(image, ksize, sigma):  # pragma: no cover - deterministic stub
        return image

    def _put_text(image, text, org, font, font_scale, color, thickness=1):  # pragma: no cover - unused in tests
        return image

    def _get_text_size(text, font, font_scale, thickness):  # pragma: no cover - unused in tests
        height = int(10 * font_scale) or 1
        width = len(text) * height
        return (width, height), None

    fake_cv2 = types.SimpleNamespace(
        COLOR_BGR2GRAY=0,
        THRESH_BINARY_INV=1,
        THRESH_OTSU=8,
        FONT_HERSHEY_SIMPLEX=0,
        cvtColor=_cvt_color,
        threshold=_threshold,
        countNonZero=_count_non_zero,
        moments=_moments,
        circle=_circle,
        GaussianBlur=_gaussian_blur,
        putText=_put_text,
        getTextSize=_get_text_size,
    )

    sys.modules["cv2"] = fake_cv2
    import cv2  # type: ignore

from omr_config import BubbleLayout, MarkerConfig, PageGeometry, SheetLayout
from omr_layout import generate_all_bubble_coordinates
from omr_processor import sample_bubbles_from_coordinates


def _project_coordinate(coord, geom, markers, page_width, page_height):
    anchor_inset_x = geom.margin / 2.0 + markers.anchor_size / 2.0
    anchor_inset_y = geom.margin / 2.0 + markers.anchor_size / 2.0
    effective_width = geom.width - 2.0 * anchor_inset_x
    effective_height = geom.height - 2.0 * anchor_inset_y

    rel_x = (coord.x - anchor_inset_x) / effective_width
    rel_y = ((geom.height - anchor_inset_y) - coord.y) / effective_height
    rel_x = min(max(rel_x, 0.0), 1.0)
    rel_y = min(max(rel_y, 0.0), 1.0)

    abs_x = int(round(rel_x * page_width))
    abs_y = int(round(rel_y * page_height))
    return abs_x, abs_y


def test_group_offset_fallback_for_unrefined_bubbles():
    geom = PageGeometry(width=120, height=120, margin=10)
    layout = BubbleLayout(radius=5, vertical_gap=15, option_gap=6, column_padding=8, label_column_width=10)
    markers = MarkerConfig(anchor_size=18)
    sheet = SheetLayout(roll_columns=2, roll_rows=1, question_options=2)

    roll_coords, question_coords = generate_all_bubble_coordinates(geom, layout, sheet)

    width, height = 400, 400
    corrected = np.full((height, width, 3), 255, dtype=np.uint8)

    anchor_inset_x = geom.margin / 2.0 + markers.anchor_size / 2.0
    anchor_inset_y = geom.margin / 2.0 + markers.anchor_size / 2.0
    effective_width = geom.width - 2.0 * anchor_inset_x
    effective_height = geom.height - 2.0 * anchor_inset_y

    scale_x = width / effective_width

    roll_offset = (5, -4)
    question_offsets = {0: (3, -2), 1: (-2, 4)}

    roll_bases = {}
    for coord in roll_coords:
        base_x, base_y = _project_coordinate(coord, geom, markers, width, height)
        roll_bases[(coord.row, coord.column)] = (base_x, base_y)
        if coord.column == 0:
            radius = max(1, int(round(coord.radius * scale_x)))
            offset = roll_offset
            cv2.circle(corrected, (base_x + offset[0], base_y + offset[1]), radius, (0, 0, 0), -1)

    question_bases = {}
    for coord in question_coords:
        base_x, base_y = _project_coordinate(coord, geom, markers, width, height)
        question_bases[(coord.question, coord.option_index)] = (base_x, base_y)
        radius = max(1, int(round(coord.radius * scale_x)))
        offset = question_offsets.get(coord.question_column, (0, 0))

        # Draw only for selected bubbles to simulate refinement failures
        draw = True
        if coord.question == 3 and coord.option_index == 0:
            draw = False
        if coord.question == 5:
            draw = False
        if coord.question == 6 and coord.option_index == 1:
            draw = False

        if draw:
            cv2.circle(corrected, (base_x + offset[0], base_y + offset[1]), radius, (0, 0, 0), -1)

    roll_groups, question_groups = sample_bubbles_from_coordinates(
        corrected, geom, layout, sheet, markers
    )

    # Roll row should inherit the observed offset even when refinement fails
    assert len(roll_groups) == 1
    assert len(roll_groups[0]) == 2
    expected_roll_offsets = sorted(roll_bases.items(), key=lambda item: item[1][0])
    roll_groups[0].sort(key=lambda b: b.x)
    for bubble, ((_, _), base) in zip(roll_groups[0], expected_roll_offsets):
        dx = bubble.x - base[0]
        dy = bubble.y - base[1]
        assert abs(dx - roll_offset[0]) <= 1
        assert abs(dy - roll_offset[1]) <= 1

    # Verify question offsets by question id
    question_lookup = {
        (q_idx + 1, opt_idx): bubble
        for q_idx, group in enumerate(question_groups)
        for opt_idx, bubble in enumerate(group)
    }

    # Question 3 option 0 relies on question-level fallback
    base_q3_opt0 = question_bases[(3, 0)]
    bubble_q3_opt0 = question_lookup[(3, 0)]
    dx = bubble_q3_opt0.x - base_q3_opt0[0]
    dy = bubble_q3_opt0.y - base_q3_opt0[1]
    expected_offset = question_offsets[1]
    assert abs(dx - expected_offset[0]) <= 1
    assert abs(dy - expected_offset[1]) <= 1

    # Question 5 had no successful refinements; use column median
    for option_index in (0, 1):
        base = question_bases[(5, option_index)]
        bubble = question_lookup[(5, option_index)]
        dx = bubble.x - base[0]
        dy = bubble.y - base[1]
        expected_offset = question_offsets[1]
        assert abs(dx - expected_offset[0]) <= 1
        assert abs(dy - expected_offset[1]) <= 1
