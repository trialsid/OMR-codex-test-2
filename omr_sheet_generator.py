"""Generate a modern OMR sheet PDF with bubbles only.

This script produces an A4-sized OMR sheet that contains:
- Four anchor markers at the page corners.
- Grid markers along the vertical edges.
- A three-digit roll number section consisting solely of bubbles.
- Multi-column question bubbles (four options per question) filling the available space.

The resulting PDF is saved inside the ``sheets/`` directory.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable, List

from omr_config import PageGeometry, BubbleLayout, MarkerConfig, SheetLayout
from omr_layout import (
    generate_all_bubble_coordinates,
    BubbleGroup,
    BoxCoordinate,
    calculate_roll_label_position,
    calculate_questions_label_position,
)


class PDFContent:
    def __init__(self) -> None:
        self._commands: List[str] = []

    def add(self, command: str) -> None:
        self._commands.append(command)

    def draw_text(self, x: float, y: float, text: str, font: str = "F1", size: float = 12) -> None:
        escaped = _escape_pdf_text(text)
        self.add("BT")
        self.add(f"/{font} {_fmt(size)} Tf")
        self.add(f"{_fmt(x)} {_fmt(y)} Td")
        self.add(f"({escaped}) Tj")
        self.add("ET")

    def set_line_width(self, width: float) -> None:
        self.add(f"{_fmt(width)} w")

    def set_stroke_color(self, r: float, g: float, b: float) -> None:
        self.add(f"{_fmt(r)} {_fmt(g)} {_fmt(b)} RG")

    def set_fill_color(self, r: float, g: float, b: float) -> None:
        self.add(f"{_fmt(r)} {_fmt(g)} {_fmt(b)} rg")

    def stroke_circle(self, cx: float, cy: float, radius: float) -> None:
        kappa = 0.552284749831 * radius
        self.add(f"{_fmt(cx)} {_fmt(cy + radius)} m")
        self.add(
            f"{_fmt(cx + kappa)} {_fmt(cy + radius)} "
            f"{_fmt(cx + radius)} {_fmt(cy + kappa)} {_fmt(cx + radius)} {_fmt(cy)} c"
        )
        self.add(
            f"{_fmt(cx + radius)} {_fmt(cy - kappa)} {_fmt(cx + kappa)} {_fmt(cy - radius)} "
            f"{_fmt(cx)} {_fmt(cy - radius)} c"
        )
        self.add(
            f"{_fmt(cx - kappa)} {_fmt(cy - radius)} {_fmt(cx - radius)} {_fmt(cy - kappa)} "
            f"{_fmt(cx - radius)} {_fmt(cy)} c"
        )
        self.add(
            f"{_fmt(cx - radius)} {_fmt(cy + kappa)} {_fmt(cx - kappa)} {_fmt(cy + radius)} "
            f"{_fmt(cx)} {_fmt(cy + radius)} c"
        )
        self.add("h")
        self.add("S")

    def fill_rect(self, x: float, y: float, width: float, height: float) -> None:
        self.add(f"{_fmt(x)} {_fmt(y)} {_fmt(width)} {_fmt(height)} re")
        self.add("f")

    def stroke_rect(self, x: float, y: float, width: float, height: float) -> None:
        self.add(f"{_fmt(x)} {_fmt(y)} {_fmt(width)} {_fmt(height)} re")
        self.add("S")

    def render(self) -> str:
        return "\n".join(self._commands) + "\n"


def draw_anchor_markers(content: PDFContent, geom: PageGeometry, markers: MarkerConfig) -> None:
    inset = geom.margin / 2
    positions = [
        (inset, geom.height - inset - markers.anchor_size),
        (geom.width - inset - markers.anchor_size, geom.height - inset - markers.anchor_size),
        (inset, inset),
        (geom.width - inset - markers.anchor_size, inset),
    ]
    content.set_fill_color(0, 0, 0)
    for x, y in positions:
        content.fill_rect(x, y, markers.anchor_size, markers.anchor_size)


def draw_grid_markers(
    content: PDFContent,
    geom: PageGeometry,
    markers: MarkerConfig,
    bubble_groups: List[BubbleGroup],
) -> None:
    """Draw one grid marker per bubble row, with clearance from anchors."""
    # Calculate anchor boundaries (in PDF coordinates, y increases upward)
    vertical_inset = geom.margin / 2.0

    # Bottom anchor top edge
    bottom_anchor_top = vertical_inset + markers.anchor_size

    # Top anchor bottom edge
    top_anchor_bottom = geom.height - vertical_inset - markers.anchor_size

    # Collect all unique bubble row y-coordinates
    bubble_y_positions = set()
    for group in bubble_groups:
        for bubble in group.bubbles:
            bubble_y_positions.add(bubble.y)

    # Filter positions that are too close to anchors (5-point clearance)
    clearance = 5
    valid_positions = sorted([
        y for y in bubble_y_positions
        if bottom_anchor_top + clearance <= y <= top_anchor_bottom - clearance
    ])

    # Draw grid markers at valid bubble row positions
    x_offsets = (geom.margin / 2, geom.width - geom.margin / 2 - markers.grid_marker_size)

    content.set_fill_color(0, 0, 0)
    for y in valid_positions:
        for x in x_offsets:
            content.fill_rect(x, y, markers.grid_marker_size, markers.grid_marker_size)


def draw_roll_number_section(
    content: PDFContent,
    geom: PageGeometry,
    layout: BubbleLayout,
    groups: List[BubbleGroup],
    boxes: List[BoxCoordinate],
) -> None:
    """Draw roll number bubbles, labels, and write-in boxes at specified coordinates."""
    if not groups:
        return

    # Calculate label position using layout geometry
    label_x, label_y = calculate_roll_label_position(geom, layout)
    content.draw_text(label_x, label_y, "Roll Number")

    content.set_line_width(0.8)  # Thinner lines for subtlety
    content.set_stroke_color(0.5, 0.5, 0.5)  # Grey color

    for box in boxes:
        box_x = box.x - box.width / 2
        box_y = box.y - box.height / 2
        content.stroke_rect(box_x, box_y, box.width, box.height)

    content.set_line_width(1)
    content.set_stroke_color(0, 0, 0)

    # Constants for text sizing (Helvetica at size 12)
    digit_width = 6.5  # Approximate width of one digit in points
    gap_before_bubble = 5  # Minimum gap between label end and bubble edge

    label_end_x = geom.margin + layout.label_column_width - gap_before_bubble

    for group in groups:
        if not group.bubbles:
            continue

        digit_x = label_end_x - digit_width
        digit_y = group.bubbles[0].y - layout.radius / 2
        content.draw_text(digit_x, digit_y, group.display_label)

        for bubble in group.bubbles:
            content.stroke_circle(bubble.x, bubble.y, bubble.radius)


def draw_question_columns(
    content: PDFContent,
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers: MarkerConfig,
    groups: List[BubbleGroup],
    roll_bottom: float,
) -> None:
    """Draw question bubbles and labels at specified coordinates."""
    if not groups:
        return

    # Calculate "Questions" label position using layout geometry and roll section bottom
    label_x, label_y = calculate_questions_label_position(geom, layout, sheet, roll_bottom, markers)
    if label_x > 0 and label_y > 0:  # Valid position found
        content.draw_text(label_x, label_y, "Questions")

    content.set_line_width(1)
    content.set_stroke_color(0, 0, 0)

    # Find topmost bubble per question_column for header placement
    topmost_per_column = {}
    column_reference: dict[int, BubbleGroup] = {}
    for group in groups:
        if group.column_index is None:
            continue
        column_reference.setdefault(group.column_index, group)
        for bubble in group.bubbles:
            if group.column_index not in topmost_per_column or bubble.y > topmost_per_column[group.column_index]:
                topmost_per_column[group.column_index] = bubble.y

    # Draw option headers (A, B, C, D) above each column
    for col, top_y in topmost_per_column.items():
        group = column_reference.get(col)
        if not group:
            continue

        # Position headers above topmost bubble with comfortable spacing
        header_y = top_y + layout.vertical_gap * 0.7

        for bubble in group.bubbles:
            text_offset = 3.25  # Approximate half-width of a letter at size 10
            content.draw_text(bubble.x - text_offset, header_y, bubble.label, size=10)

    # First pass: find max question number per column
    max_question_per_column = {}
    for group in groups:
        if group.column_index is None:
            continue
        max_question_per_column[group.column_index] = max(
            max_question_per_column.get(group.column_index, 0),
            group.group_index,
        )

    # Calculate max digits needed per column
    max_digits_per_column = {}
    for col, max_q in max_question_per_column.items():
        max_digits_per_column[col] = len(str(max_q))

    # Constants for text sizing (Helvetica at size 12)
    digit_width = 6.5  # Approximate width of one digit in points
    gap_before_bubble = 5  # Minimum gap between label end and bubble edge

    # Group bubbles by question for label placement
    for group in groups:
        if not group.bubbles:
            continue

        column_origin = geom.margin + (group.column_index or 0) * layout.group_width(sheet.question_options)
        max_digits = max_digits_per_column.get(group.column_index, 1)

        label_end_x = column_origin + layout.label_column_width - gap_before_bubble
        current_text_width = len(group.display_label) * digit_width
        label_x = label_end_x - current_text_width
        label_y = group.bubbles[0].y - layout.radius / 2
        content.draw_text(label_x, label_y, group.display_label)

        for bubble in group.bubbles:
            content.stroke_circle(bubble.x, bubble.y, bubble.radius)


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    value = start
    while value <= stop + 1e-6:
        yield value
        value += step


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_pdf(width: float, height: float, content_stream: str, output_path: Path) -> None:
    buffer = BytesIO()
    buffer.write(b"%PDF-1.4\n")

    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 {_fmt(width)} {_fmt(height)}] >>",
        "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> /Contents 4 0 R >>",
        f"<< /Length {len(content_stream.encode('ascii'))} >>\nstream\n{content_stream}endstream",
    ]

    offsets: List[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(buffer.tell())
        buffer.write(f"{index} 0 obj\n".encode("ascii"))
        buffer.write(obj.encode("ascii"))
        buffer.write(b"\nendobj\n")

    xref_position = buffer.tell()
    total_objects = len(objects) + 1
    buffer.write(f"xref\n0 {total_objects}\n".encode("ascii"))
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets:
        buffer.write(f"{offset:010d} 00000 n \n".encode("ascii"))

    buffer.write(
        f"trailer\n<< /Size {total_objects} /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF".encode("ascii")
    )

    output_path.write_bytes(buffer.getvalue())


def generate_omr_sheet(output_path: Path) -> None:
    geom = PageGeometry()
    layout = BubbleLayout()
    markers = MarkerConfig()
    sheet = SheetLayout()

    ensure_output_directory(output_path.parent)
    content = PDFContent()

    # Generate bubble coordinates procedurally
    bubble_groups, roll_boxes, roll_bottom = generate_all_bubble_coordinates(geom, layout, sheet, markers)
    roll_groups = [group for group in bubble_groups if group.category == "roll"]
    question_groups = [group for group in bubble_groups if group.category == "question"]

    # Draw all components
    draw_anchor_markers(content, geom, markers)
    draw_grid_markers(content, geom, markers, bubble_groups)
    draw_roll_number_section(content, geom, layout, roll_groups, roll_boxes)
    draw_question_columns(content, geom, layout, sheet, markers, question_groups, roll_bottom)

    build_pdf(geom.width, geom.height, content.render(), output_path)


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _fmt(value: float) -> str:
    formatted = f"{value:.3f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


if __name__ == "__main__":
    target_path = Path("sheets") / "omr_sheet.pdf"
    generate_omr_sheet(target_path)
