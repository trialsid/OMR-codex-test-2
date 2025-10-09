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
    BubbleCoordinate,
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


def draw_grid_markers(content: PDFContent, geom: PageGeometry, markers: MarkerConfig) -> None:
    start_y = geom.margin + markers.grid_spacing
    end_y = geom.height - geom.margin - markers.grid_spacing
    y_positions = list(_frange(start_y, end_y, markers.grid_spacing))
    x_offsets = (geom.margin / 2, geom.width - geom.margin / 2 - markers.grid_marker_size)

    content.set_fill_color(0, 0, 0)
    for y in y_positions:
        for x in x_offsets:
            content.fill_rect(x, y, markers.grid_marker_size, markers.grid_marker_size)


def draw_roll_number_section(
    content: PDFContent,
    geom: PageGeometry,
    layout: BubbleLayout,
    bubbles: List[BubbleCoordinate],
) -> None:
    """Draw roll number bubbles and labels at specified coordinates."""
    if not bubbles:
        return

    # Calculate label position using layout geometry
    label_x, label_y = calculate_roll_label_position(geom, layout)
    content.draw_text(label_x, label_y, "Roll Number")

    content.set_line_width(1)
    content.set_stroke_color(0, 0, 0)

    # Constants for text sizing (Helvetica at size 12)
    digit_width = 6.5  # Approximate width of one digit in points
    gap_before_bubble = 5  # Minimum gap between label end and bubble edge

    # Group bubbles by row for digit label placement
    rows_seen = set()
    for bubble in bubbles:
        # Draw digit label (once per row, right-aligned)
        if bubble.row is not None and bubble.row not in rows_seen:
            # Right-align single digit (0-9)
            label_end_x = geom.margin + layout.label_column_width - gap_before_bubble
            digit_x = label_end_x - digit_width
            digit_y = bubble.y - layout.radius / 2
            content.draw_text(digit_x, digit_y, str(bubble.digit))
            rows_seen.add(bubble.row)

        # Draw bubble
        content.stroke_circle(bubble.x, bubble.y, bubble.radius)


def draw_question_columns(
    content: PDFContent,
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    bubbles: List[BubbleCoordinate],
) -> None:
    """Draw question bubbles and labels at specified coordinates."""
    if not bubbles:
        return

    # Calculate "Questions" label position using layout geometry
    label_x, label_y = calculate_questions_label_position(geom, layout, sheet)
    if label_x > 0 and label_y > 0:  # Valid position found
        content.draw_text(label_x, label_y, "Questions")

    content.set_line_width(1)
    content.set_stroke_color(0, 0, 0)

    # First pass: find max question number per column
    max_question_per_column = {}
    for bubble in bubbles:
        if bubble.question is not None and bubble.question_column is not None:
            col = bubble.question_column
            max_question_per_column[col] = max(
                max_question_per_column.get(col, 0),
                bubble.question
            )

    # Calculate max digits needed per column
    max_digits_per_column = {}
    for col, max_q in max_question_per_column.items():
        max_digits_per_column[col] = len(str(max_q))

    # Constants for text sizing (Helvetica at size 12)
    digit_width = 6.5  # Approximate width of one digit in points
    gap_before_bubble = 5  # Minimum gap between label end and bubble edge

    # Group bubbles by question for label placement
    questions_seen = set()
    for bubble in bubbles:
        # Draw question number label (once per question, on first option)
        if bubble.question is not None and bubble.option_index == 0:
            if bubble.question not in questions_seen:
                # Calculate label position based on column, right-aligned
                column_origin = geom.margin + (bubble.question_column or 0) * layout.group_width(sheet.question_options)

                # Determine max digits for this column
                max_digits = max_digits_per_column.get(bubble.question_column, 1)

                # Calculate right-aligned position
                # End of label area (before gap and bubbles)
                label_end_x = column_origin + layout.label_column_width - gap_before_bubble
                # Width of max-digit number in this column
                max_text_width = max_digits * digit_width
                # Start position for this specific label (right-aligned within max width)
                current_text_width = len(str(bubble.question)) * digit_width
                label_x = label_end_x - current_text_width

                label_y = bubble.y - layout.radius / 2
                content.draw_text(label_x, label_y, str(bubble.question))
                questions_seen.add(bubble.question)

        # Draw bubble
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
    roll_bubbles, question_bubbles = generate_all_bubble_coordinates(geom, layout, sheet)

    # Draw all components
    draw_anchor_markers(content, geom, markers)
    draw_grid_markers(content, geom, markers)
    draw_roll_number_section(content, geom, layout, roll_bubbles)
    draw_question_columns(content, geom, layout, sheet, question_bubbles)

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
