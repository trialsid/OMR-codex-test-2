"""Utility script to generate multiple OMR sheet PDF templates.

This module generates five different OMR sheet designs using the ReportLab
toolkit. Each sheet follows a contemporary OMR layout with clearly defined
candidate information areas, instructions, and answer bubbles that are ready
to be scanned. Running this script creates a ``sheets/`` directory in the
project root (if it does not already exist) and stores the five generated
PDFs inside it.

The script purposely relies on relative positioning so the layout adapts to
different portrait page sizes. While A4 is used as the default canvas, the
coordinate calculations are based on the page dimensions which keeps the
designs largely agnostic to the actual paper size used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


SHEETS_DIRECTORY = "sheets"


@dataclass(frozen=True)
class OMRVariant:
    """Configuration for an OMR sheet variant."""

    filename: str
    title: str
    subtitle: str
    question_count: int
    options: Sequence[str]
    columns: int
    instructions: Sequence[str]
    accent_color: colors.Color
    alternate_row_shading: bool = False
    section_labels: Sequence[str] | None = None


def draw_header(c: canvas.Canvas, variant: OMRVariant, width: float, height: float) -> float:
    """Draw the sheet header and return the vertical offset consumed."""

    header_height = height * 0.12
    c.setFillColor(variant.accent_color)
    c.rect(0, height - header_height, width, header_height, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", min(24, header_height * 0.45))
    c.drawString(width * 0.06, height - header_height * 0.55, variant.title)

    c.setFont("Helvetica", min(12, header_height * 0.22))
    c.drawString(width * 0.06, height - header_height * 0.75, variant.subtitle)

    return header_height


def draw_candidate_block(c: canvas.Canvas, width: float, top: float) -> float:
    """Draw the candidate information panel and return the consumed height."""

    block_height = 110
    left_margin = width * 0.06
    right_margin = width * 0.06
    field_width = (width - left_margin - right_margin) / 3
    field_height = block_height * 0.45

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, top - 24, "Candidate Information")

    fields = [
        "Candidate Name",
        "Roll Number",
        "Test / Subject Code",
        "Invigilator Signature",
        "Candidate Signature",
        "Test Date",
    ]

    c.setFont("Helvetica", 9)
    for idx, label in enumerate(fields):
        col = idx % 3
        row = idx // 3
        x = left_margin + col * field_width
        y = top - 40 - row * (field_height + 18)

        c.drawString(x, y, label)
        c.roundRect(x, y - field_height + 6, field_width - 10, field_height - 12, 6, stroke=1, fill=0)

    barcode_width = width * 0.22
    barcode_height = block_height * 0.9
    barcode_x = width - right_margin - barcode_width
    barcode_y = top - barcode_height - 10
    c.setDash(4, 4)
    c.roundRect(barcode_x, barcode_y, barcode_width, barcode_height, 10, stroke=1, fill=0)
    c.setDash()
    c.drawCentredString(barcode_x + barcode_width / 2, barcode_y + barcode_height / 2, "Affix Barcode / QR")

    return block_height


def draw_instructions(c: canvas.Canvas, width: float, top: float, instructions: Iterable[str]) -> float:
    """Draw instructions beneath the header."""

    consumed = 0
    left_margin = width * 0.06
    c.setFont("Helvetica", 9)

    for line in instructions:
        c.drawString(left_margin, top - consumed - 14, f"• {line}")
        consumed += 16

    return consumed + 6


def draw_question_grid(
    c: canvas.Canvas,
    variant: OMRVariant,
    width: float,
    available_height: float,
    bottom: float,
) -> None:
    """Draw the answer bubble grid for the variant."""

    left_margin = width * 0.06
    right_margin = width * 0.06
    columns = variant.columns

    column_width = (width - left_margin - right_margin) / columns
    questions_per_column = ceil(variant.question_count / columns)

    row_height = available_height / questions_per_column
    bubble_radius = min(row_height * 0.32, column_width / (len(variant.options) * 3.5))

    c.setFont("Helvetica", max(8, bubble_radius * 1.4))
    c.setStrokeColor(colors.black)

    for idx in range(variant.question_count):
        column = idx // questions_per_column
        row = idx % questions_per_column

        x_origin = left_margin + column * column_width
        y = bottom + available_height - (row + 0.5) * row_height

        if variant.alternate_row_shading and row % 2 == 0:
            c.setFillColor(colors.HexColor("#f3f4f6"))
            c.rect(x_origin, y - row_height / 2, column_width, row_height, fill=1, stroke=0)
            c.setFillColor(colors.black)

        question_number = idx + 1
        c.drawString(x_origin + 4, y - bubble_radius, f"{question_number:03}")

        total_option_width = column_width - 36
        option_spacing = total_option_width / len(variant.options)

        for opt_index, option in enumerate(variant.options):
            bubble_center_x = x_origin + 24 + opt_index * option_spacing
            c.circle(bubble_center_x, y, bubble_radius, stroke=1, fill=0)
            c.drawCentredString(bubble_center_x, y - bubble_radius - 6, option)


def draw_section_labels(
    c: canvas.Canvas,
    variant: OMRVariant,
    width: float,
    bottom: float,
    available_height: float,
) -> None:
    """Render section labels along the left margin if provided."""

    if not variant.section_labels:
        return

    left_margin = width * 0.06
    columns = variant.columns
    questions_per_column = ceil(variant.question_count / columns)
    row_height = available_height / questions_per_column

    c.setFont("Helvetica-Bold", 9)
    label_height = row_height * questions_per_column / len(variant.section_labels)

    for idx, label in enumerate(variant.section_labels):
        section_y = bottom + available_height - idx * label_height
        c.saveState()
        c.translate(left_margin - 14, section_y - label_height / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, label)
        c.restoreState()


def render_variant(variant: OMRVariant) -> None:
    """Generate a single OMR sheet PDF for the provided variant."""

    output_path = os.path.join(SHEETS_DIRECTORY, variant.filename)
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    top_offset = height
    top_offset -= draw_header(c, variant, width, height)

    top_offset -= draw_candidate_block(c, width, top_offset - 12) + 12
    instruction_offset = draw_instructions(c, width, top_offset, variant.instructions)
    top_offset -= instruction_offset

    bottom_margin = height * 0.06
    available_height = top_offset - bottom_margin

    draw_question_grid(c, variant, width, available_height, bottom_margin)
    draw_section_labels(c, variant, width, bottom_margin, available_height)

    c.showPage()
    c.save()


def generate_variants(variants: Sequence[OMRVariant]) -> None:
    """Render all variants into the sheets directory."""

    os.makedirs(SHEETS_DIRECTORY, exist_ok=True)

    for variant in variants:
        render_variant(variant)


VARIANTS: Sequence[OMRVariant] = [
    OMRVariant(
        filename="omr_classic_100.pdf",
        title="Unified OMR Answer Sheet",
        subtitle="Standard 100 Question Layout",
        question_count=100,
        options=("A", "B", "C", "D", "E"),
        columns=4,
        instructions=(
            "Use a 2B pencil to fill bubbles completely.",
            "Do not fold or staple the answer sheet.",
            "Ensure your roll number and booklet code are entered correctly.",
            "Multiple selections for a single question will be treated as incorrect.",
        ),
        accent_color=colors.HexColor("#004b8d"),
        alternate_row_shading=True,
    ),
    OMRVariant(
        filename="omr_compact_60.pdf",
        title="Compact OMR Sheet",
        subtitle="60 Question Rapid Assessment",
        question_count=60,
        options=("A", "B", "C", "D"),
        columns=3,
        instructions=(
            "Shade the bubble corresponding to your chosen option.",
            "Erase cleanly to change an answer.",
            "Return the sheet to the invigilator after completion.",
        ),
        accent_color=colors.HexColor("#006f43"),
        alternate_row_shading=False,
        section_labels=("Section A", "Section B", "Section C"),
    ),
    OMRVariant(
        filename="omr_engineering_120.pdf",
        title="Engineering Entrance OMR",
        subtitle="Four Section Diagnostic",
        question_count=120,
        options=("A", "B", "C", "D"),
        columns=4,
        instructions=(
            "Sections are color coded; attempt all questions.",
            "Mark responses firmly without straying outside bubbles.",
            "Report any printing errors to the invigilator immediately.",
        ),
        accent_color=colors.HexColor("#8b1f41"),
        alternate_row_shading=True,
        section_labels=("Physics", "Chemistry", "Mathematics", "Logical Reasoning"),
    ),
    OMRVariant(
        filename="omr_language_80.pdf",
        title="Language Proficiency OMR",
        subtitle="Listening & Reading Combination",
        question_count=80,
        options=("A", "B", "C", "D", "E"),
        columns=4,
        instructions=(
            "Listen carefully; mark answers only when prompted.",
            "For reading section, transfer answers within the allotted time.",
            "Use only blue or black ink for signatures.",
        ),
        accent_color=colors.HexColor("#0f4c75"),
        alternate_row_shading=False,
        section_labels=("Listening", "Reading", "Vocabulary", "Grammar"),
    ),
    OMRVariant(
        filename="omr_mock_test_50.pdf",
        title="Mock Test OMR",
        subtitle="Practice Series Template",
        question_count=50,
        options=("A", "B", "C", "D"),
        columns=2,
        instructions=(
            "Practice filling bubbles quickly and neatly.",
            "Scores are for self-evaluation; do not submit officially.",
            "Keep this sheet flat when scanning for self-grading.",
        ),
        accent_color=colors.HexColor("#f39c12"),
        alternate_row_shading=True,
    ),
]


def main() -> None:
    generate_variants(VARIANTS)
    print(f"Generated {len(VARIANTS)} OMR sheet templates in '{SHEETS_DIRECTORY}/'.")


if __name__ == "__main__":
    main()

