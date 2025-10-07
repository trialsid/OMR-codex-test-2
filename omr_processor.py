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
from typing import List, Tuple, Optional

from omr_config import PageGeometry, BubbleLayout, MarkerConfig, SheetLayout


@dataclass
class Bubble:
    """Represents a detected bubble."""
    x: int
    y: int
    radius: int

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


def detect_bubbles(image: np.ndarray, layout: BubbleLayout) -> List[Bubble]:
    """Detect all bubbles in the corrected image.

    Args:
        image: Corrected grayscale or color image
        layout: Bubble layout configuration

    Returns:
        List of detected bubbles
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,
        param1=50,
        param2=30,
        minRadius=8,
        maxRadius=20
    )

    bubbles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            bubbles.append(Bubble(x=int(circle[0]), y=int(circle[1]), radius=int(circle[2])))

    # Sort bubbles top-to-bottom, left-to-right
    bubbles.sort()

    return bubbles


def group_bubbles_into_questions(
    bubbles: List[Bubble], sheet: SheetLayout, layout: BubbleLayout
) -> Tuple[List[List[Bubble]], List[List[Bubble]]]:
    """Group bubbles into roll number section and question sections.

    Args:
        bubbles: All detected bubbles sorted
        sheet: Sheet layout configuration
        layout: Bubble layout configuration (for spacing expectations)

    Returns:
        (roll_number_groups, question_groups) where each group is a list of bubbles
    """
    if not bubbles:
        return [], []

    # Group all bubbles by rows first (same y-coordinate within tolerance)
    bubbles_sorted = sorted(bubbles, key=lambda b: (b.y, b.x))
    rows = []
    current_row = []

    for bubble in bubbles_sorted:
        if not current_row or abs(bubble.y - current_row[0].y) < 15:
            current_row.append(bubble)
        else:
            if current_row:
                rows.append(sorted(current_row, key=lambda b: b.x))
            current_row = [bubble]
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b.x))

    # Process all rows
    roll_bubbles = []
    all_question_bubbles = []  # Collect all question bubbles with position info

    for i, row in enumerate(rows):
        if i < sheet.roll_rows:
            # First 10 rows: extract roll numbers (first 3) and questions (rest)
            if len(row) >= sheet.roll_columns:
                roll_bubbles.append(row[:sheet.roll_columns])
                # Remaining bubbles in this row are questions
                question_bubbles_in_row = row[sheet.roll_columns:]
            else:
                question_bubbles_in_row = row
        else:
            # Rows 10+: all bubbles are questions
            question_bubbles_in_row = row

        # Group question bubbles in this row into sets of 4
        for j in range(0, len(question_bubbles_in_row) - sheet.question_options + 1, sheet.question_options):
            group = question_bubbles_in_row[j:j + sheet.question_options]
            if len(group) == sheet.question_options:
                # Verify spacing is consistent (they're part of same question)
                gaps = [group[k+1].x - group[k].x for k in range(len(group)-1)]
                avg_gap = sum(gaps) / len(gaps) if gaps else 0
                # If gaps are relatively uniform, it's a valid question group
                if all(abs(gap - avg_gap) < 30 for gap in gaps):
                    # Store with average x position for column sorting
                    avg_x = sum(b.x for b in group) / len(group)
                    all_question_bubbles.append((avg_x, group[0].y, group))

    # Group questions into discrete columns by clustering x-coordinates
    if not all_question_bubbles:
        return roll_bubbles, []

    # Sort by x-coordinate to identify columns
    sorted_by_x = sorted(all_question_bubbles, key=lambda item: item[0])

    # Cluster into columns using the configured column width for tolerance
    column_width = layout.group_width(sheet.question_options)
    column_tolerance = column_width / 2
    columns = []
    current_column = [sorted_by_x[0]]

    for item in sorted_by_x[1:]:
        if item[0] - current_column[0][0] < column_tolerance:  # Same column
            current_column.append(item)
        else:  # New column
            columns.append(current_column)
            current_column = [item]
    columns.append(current_column)

    # Sort each column by y-coordinate (top to bottom), then concatenate
    question_groups = []
    for column in columns:
        # Sort by y within this column
        column_sorted = sorted(column, key=lambda item: item[1])
        question_groups.extend([item[2] for item in column_sorted])

    return roll_bubbles, question_groups


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

    # Label roll numbers (digits 0-9 for each of 3 columns)
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

    # Label questions (A, B, C, D)
    option_labels = ['A', 'B', 'C', 'D']
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


def process_omr_sheet(input_path: Path, output_path: Path,
                      geom: PageGeometry, layout: BubbleLayout, sheet: SheetLayout) -> bool:
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

    # Detect bubbles
    bubbles = detect_bubbles(corrected, layout)
    print(f"Detected {len(bubbles)} bubbles")

    if not bubbles:
        print("No bubbles detected")
        return False

    # Group bubbles
    roll_groups, question_groups = group_bubbles_into_questions(bubbles, sheet, layout)
    print(f"Found {len(roll_groups)} roll number rows and {len(question_groups)} questions")

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
        success = process_omr_sheet(image_file, output_file, geom, layout, sheet)
        print()


if __name__ == "__main__":
    main()
