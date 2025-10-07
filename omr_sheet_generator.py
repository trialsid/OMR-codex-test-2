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


class PDFContent:
    def __init__(self) -> None:
        self._commands: List[str] = []

    def add(self, command: str) -> None:
        self._commands.append(command)

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


def draw_roll_number_section(content: PDFContent, geom: PageGeometry, layout: BubbleLayout, sheet: SheetLayout) -> tuple[float, float, float]:
    area_width = sheet.roll_columns * layout.diameter + (sheet.roll_columns - 1) * layout.option_gap + 2 * layout.column_padding
    x_start = geom.margin
    top_y = geom.height - geom.margin - layout.diameter

    content.set_line_width(1)
    content.set_stroke_color(0, 0, 0)
    for col in range(sheet.roll_columns):
        x = x_start + layout.column_padding + layout.radius + col * (layout.diameter + layout.option_gap)
        for row in range(sheet.roll_rows):
            y = top_y - row * layout.vertical_gap
            content.stroke_circle(x, y, layout.radius)

    # Calculate bottom of roll number section
    bottom_y = top_y - (sheet.roll_rows - 1) * layout.vertical_gap - layout.radius

    return top_y, area_width, bottom_y


def draw_question_columns(
    content: PDFContent,
    geom: PageGeometry,
    layout: BubbleLayout,
    top_y: float,
    x_start: float,
    sheet: SheetLayout,
    roll_bottom: float,
) -> None:
    options = sheet.question_options
    column_width = layout.group_width(options)
    available_width = geom.width - geom.margin - x_start
    columns = max(1, int(available_width // column_width))

    # Calculate total rows available from top to bottom
    available_height = top_y - geom.margin + layout.diameter
    total_rows = max(0, int(available_height // layout.vertical_gap))

    content.set_line_width(1)
    content.set_stroke_color(0, 0, 0)
    for col in range(columns):
        x_base = x_start + col * column_width + layout.column_padding / 2

        # First column: skip row 11 (gap), start questions at row 12
        # Other columns: start from row 1 (top)
        start_row = sheet.roll_rows + 1 if col == 0 else 0

        for row in range(start_row, total_rows):
            y = top_y - row * layout.vertical_gap
            if y - layout.radius <= geom.margin:
                break
            for opt in range(options):
                x = x_base + layout.radius + opt * (layout.diameter + layout.option_gap)
                content.stroke_circle(x, y, layout.radius)


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
        "<< /Type /Page /Parent 2 0 R /Resources << >> /Contents 4 0 R >>",
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

    draw_anchor_markers(content, geom, markers)
    draw_grid_markers(content, geom, markers)
    roll_top, roll_width, roll_bottom = draw_roll_number_section(content, geom, layout, sheet)
    question_x_start = geom.margin + roll_width
    draw_question_columns(content, geom, layout, roll_top, question_x_start, sheet, roll_bottom)

    build_pdf(geom.width, geom.height, content.render(), output_path)


def _fmt(value: float) -> str:
    formatted = f"{value:.3f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


if __name__ == "__main__":
    target_path = Path("sheets") / "omr_sheet.pdf"
    generate_omr_sheet(target_path)