"""Generate a modern OMR sheet PDF with bubbles only.

This script produces an A4-sized OMR sheet that contains:
- Four anchor markers at the page corners.
- Grid markers along the vertical edges.
- A three-digit roll number section consisting solely of bubbles.
- Multi-column question bubbles (four options per question) filling the available space.

The resulting PDF is saved inside the ``sheets/`` directory.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, cast

from omr_config import PageGeometry, BubbleLayout, MarkerConfig, SheetLayout


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
    sheet: SheetLayout,
    metadata: Dict[str, Any],
) -> tuple[float, float, float]:
    bubble_span = sheet.roll_columns * layout.diameter
    bubble_span += (sheet.roll_columns - 1) * layout.option_gap
    area_width = layout.label_column_width + layout.column_padding + bubble_span
    left_padding = layout.column_padding / 2
    x_start = geom.margin + layout.label_column_width + left_padding + layout.radius
    top_y = geom.height - geom.margin - layout.diameter

    label_x = geom.margin + layout.label_column_width + left_padding
    label_y = top_y + layout.radius / 2
    content.draw_text(label_x, label_y, "Roll Number")

    content.set_line_width(1)
    content.set_stroke_color(0, 0, 0)
    roll_entries: List[Dict[str, Any]] = cast(List[Dict[str, Any]], metadata.setdefault("roll_bubbles", []))

    for row in range(sheet.roll_rows):
        y = top_y - (row + 1) * layout.vertical_gap
        digit_x = geom.margin + layout.label_column_width - layout.radius
        digit_y = y - layout.radius / 2
        content.draw_text(digit_x, digit_y, str(row % 10))
        for col in range(sheet.roll_columns):
            x = x_start + col * (layout.diameter + layout.option_gap)
            content.stroke_circle(x, y, layout.radius)
            roll_entries.append(
                {
                    "row": row,
                    "column": col,
                    "digit": row % 10,
                    "center": {
                        "x": x / geom.width,
                        "y": y / geom.height,
                    },
                    "radius": layout.radius / geom.width,
                }
            )

    # Calculate bottom of roll number section
    bottom_y = top_y - sheet.roll_rows * layout.vertical_gap - layout.radius

    return top_y, area_width, bottom_y


def draw_question_columns(
    content: PDFContent,
    geom: PageGeometry,
    layout: BubbleLayout,
    top_y: float,
    x_start: float,
    sheet: SheetLayout,
    roll_bottom: float,
    metadata: Dict[str, Any],
) -> None:
    options = sheet.question_options
    column_width = layout.group_width(options)
    available_width = geom.width - geom.margin - x_start
    columns = max(1, int(available_width // column_width))
    question_entries: List[Dict[str, Any]] = cast(List[Dict[str, Any]], metadata.setdefault("question_bubbles", []))

    # Determine all candidate row centers, trimming anything that would spill past the margin
    row_centers: List[float] = []
    row_index = 1
    while True:
        y = top_y - row_index * layout.vertical_gap
        if y - layout.radius <= geom.margin:
            break
        row_centers.append(y)
        row_index += 1

    # For the first column we need to avoid overlapping the roll number section.
    # Compute the first row index whose bottom edge sits below the roll bubbles.
    first_column_start = next(
        (idx for idx, y in enumerate(row_centers) if y - layout.radius < roll_bottom),
        len(row_centers),
    )

    label_x = x_start + layout.label_column_width + layout.column_padding / 2
    label_row_index = first_column_start + 1
    if label_row_index < len(row_centers):
        label_y = row_centers[label_row_index]
        content.draw_text(label_x, label_y, "Questions")

    content.set_line_width(1)
    content.set_stroke_color(0, 0, 0)
    question_number = 1
    for col in range(columns):
        column_origin = x_start + col * column_width
        x_base = column_origin + layout.label_column_width + layout.column_padding / 2
        label_column_x = column_origin + layout.label_column_width - layout.radius

        if col == 0:
            start_row = first_column_start + 2
            if start_row >= len(row_centers):
                continue
        else:
            start_row = 0
            if not row_centers:
                continue

        for y in row_centers[start_row:]:
            label_y = y - layout.radius / 2
            content.draw_text(label_column_x, label_y, str(question_number))
            current_question = question_number
            for opt in range(options):
                x = x_base + layout.radius + opt * (layout.diameter + layout.option_gap)
                content.stroke_circle(x, y, layout.radius)
                question_entries.append(
                    {
                        "question": current_question,
                        "option_index": opt,
                        "column": col,
                        "center": {
                            "x": x / geom.width,
                            "y": y / geom.height,
                        },
                        "radius": layout.radius / geom.width,
                    }
                )
            question_number += 1


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

    metadata: Dict[str, Any] = {
        "page": {
            "width": geom.width,
            "height": geom.height,
            "margin": geom.margin,
        },
        "layout": {
            "question_options": sheet.question_options,
        },
    }

    draw_anchor_markers(content, geom, markers)
    draw_grid_markers(content, geom, markers)
    roll_top, _, roll_bottom = draw_roll_number_section(content, geom, layout, sheet, metadata)
    # Begin the first question column directly beneath the roll number section.
    question_x_start = geom.margin
    draw_question_columns(content, geom, layout, roll_top, question_x_start, sheet, roll_bottom, metadata)

    build_pdf(geom.width, geom.height, content.render(), output_path)

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))


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
