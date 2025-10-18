# OMR Sheet Processor

A Python toolkit for generating, validating, and grading optical mark recognition (OMR) answer sheets. The project bundles a shared layout engine, a PDF generator, an image processor, and supporting utilities so that the paper layout and bubble sampling logic always stay in sync.

<table align="center">
  <tr>
    <td><img src="sheets/omr_sheet.png" width="400" alt="Blank OMR Sheet"></td>
    <td><img src="processed/processed_omr_sheet.png" width="400" alt="Processed OMR Sheet"></td>
  </tr>
  <tr>
    <td align="center"><b>Blank OMR Sheet</b></td>
    <td align="center"><b>Processed Result</b></td>
  </tr>
</table>

## Highlights

- **Single geometry source of truth** – `omr_layout.py` computes anchors, bubble groups, roll number boxes, and question grids used by both the PDF generator and the processor.
- **Polished PDF sheets** – `omr_sheet_generator_new.py` uses [fpdf2](https://pyfpdf.github.io/fpdf2/) with bundled Noto Sans and Stinger fonts to render classroom-ready sheets with student instructions and school branding.
- **Robust processing pipeline** – `omr_processor.py` detects square anchor markers, applies perspective correction, samples every configured bubble with adaptive thresholding, and writes labeled overlays.
- **Helpful tooling** – PDF-to-image conversion, benchmarking scripts, and distortion stress tests simplify experimentation with different capture setups.

## Table of contents

1. [Quickstart](#quickstart)
2. [Installation](#installation)
3. [Command-line usage](#command-line-usage)
   - [Generate a sheet](#generate-a-sheet)
   - [Render PDFs to images](#render-pdfs-to-images)
   - [Process captured responses](#process-captured-responses)
4. [Configuration model](#configuration-model)
5. [Layout engine & data flow](#layout-engine--data-flow)
6. [Diagnostics & utilities](#diagnostics--utilities)
7. [Repository structure](#repository-structure)
8. [Troubleshooting](#troubleshooting)
9. [License](#license)

## Quickstart

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Generate the PDF sheet
python omr_sheet_generator_new.py --config sheet_config.json --questions 60 --output sheets/omr_sheet.pdf

# 2. (Optional) Render PDFs to PNG for dry runs
python pdf_to_image.py  # writes sheets/omr_sheet.png at 200 DPI

# 3. Process scans or photos dropped into sheets/
python omr_processor.py --config sheet_config.json --questions 60
```

Processed images appear in `processed/processed_<filename>.png`, annotated with bubble labels and filled-bubble highlights.

## Installation

### Requirements

- **Python 3.10+** – The codebase uses PEP 604 union syntax (`int | None`) introduced in Python 3.10.
- **No system dependencies** – OpenCV (`cv2`) and PyMuPDF ship as self-contained wheels; no additional binaries required.

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies** (from `requirements.txt`):
- `numpy` – Array operations and image processing math
- `opencv-python` – Anchor detection, perspective transforms, and bubble sampling
- `PyMuPDF` – PDF-to-image rendering (imported as `fitz`)
- `fpdf2==2.8.2` – PDF generation with TrueType font support

### Fonts

The generator requires TrueType fonts in the `fonts/` directory:
- `NotoSans-Regular.ttf`, `NotoSans-Bold.ttf`, `NotoSans-Italic.ttf` – Body text and labels
- `StingerFitTrial-Bold.ttf` – School name header

Keep the `fonts/` directory alongside the scripts or update paths in `omr_sheet_generator_new.py` lines 37-40.

## Command-line usage

### Generate a sheet

```bash
python omr_sheet_generator_new.py \
  --config sheet_config.json \
  --questions 60 \
  --output sheets/omr_sheet.pdf
```

**Arguments:**
- `--config` – Path to JSON configuration file (optional). See [Configuration model](#configuration-model) for details.
- `--questions` – Limit the number of question rows rendered (overrides `max_questions` in JSON).
- `--output` – Destination PDF path (default: `sheets/omr_sheet.pdf`). Parent directories created automatically.

**Note:** The generator validates that your requested layout fits on the page and logs warnings if sections overflow.

### Render PDFs to images

```bash
python pdf_to_image.py
```

Converts all `*.pdf` files in `sheets/` to sibling `.png` images at 200 DPI. Use this to:
- Verify alignment before printing
- Generate synthetic test images for the processor
- Share preview images without requiring a PDF viewer

### Process captured responses

```bash
python omr_processor.py \
  --config sheet_config.json \
  --questions 60
```

**Arguments:**
- `--config` – Path to JSON configuration file (must match the generator config used).
- `--questions` – Limit questions processed (must match generator setting).

**Processing workflow:**
1. Place scanned sheets (PNG/JPG/JPEG) in `sheets/`.
2. The processor detects four square anchors in the corners and validates their geometry.
3. Perspective correction normalizes the sheet while preserving input resolution (minimum ≈800×1000 px).
4. An adaptive fill threshold is computed from the inner zone between anchors (compensates for lighting variations).
5. Every bubble coordinate from `generate_all_bubble_coordinates()` is sampled and tagged (`class`, `roll`, `set`, or `question`).
6. Filled bubbles (darkness > threshold) are outlined in **magenta** and labeled.
7. Results saved to `processed/processed_<filename>.png` with console summaries.

**Error handling:** Processing continues even if individual files fail, making batch processing safe.

## Configuration model

### What can be configured

Configuration is split between **code** and **JSON**:

- **JSON files** control only `SheetLayout` fields (question counts, roll digits, etc.)
- **Python code** controls `PageGeometry`, `BubbleLayout`, and `MarkerConfig` (dimensions, spacing, thresholds)

This design keeps sheet content flexible while maintaining consistent physical layout.

### SheetLayout (JSON-configurable)

The `sheet_config.json` file controls these fields:

```json
{
  "class_options": 5,
  "roll_columns": 3,
  "set_options": 4,
  "question_options": 4,
  "max_questions": 50
}
```

**Fields:**
- `class_options` – Number of class/grade bubbles (e.g., 5 for grades 6-10)
- `roll_columns` – Number of digits in the roll number (e.g., 3 for roll numbers 000-999). Roll numbers always have 10 rows for digits 0-9.
- `set_options` – Number of test set/version bubbles (e.g., 4 for sets A-D)
- `question_options` – Number of answer choices per question (typically 4 for A/B/C/D)
- `max_questions` – Maximum questions to render (omit or set `null` to fill available space)

**Important:** Use the **same config file** for both generation and processing, or bubbles will be sampled at wrong coordinates.

### Other configuration classes (code-only)

These are defined in `omr_config.py` with sensible defaults:

**`PageGeometry`** – Paper dimensions and margins
- `width`, `height` – Page size in points (595.276 × 841.89 = A4)
- `margin` – Edge margin in points (default: 36pt ≈ 0.5")
- `header_ratio` – Header band height as fraction of page (default: 0.2 = 20%)

**`BubbleLayout`** – Bubble sizing and spacing
- `radius` – Bubble radius in points (default: 6.5pt)
- `vertical_gap` – Row spacing in points (default: 19pt)
- `option_gap` – Spacing between bubbles in a row (default: 9pt)
- `fill_threshold` – Base darkness ratio to consider bubble filled (default: 0.4, range 0.0-1.0)

**`MarkerConfig`** – Anchor and grid markers
- `anchor_size` – Square anchor side length (default: 20pt)
- `grid_marker_size` – Gutter tick size (default: 6pt)
- `grid_spacing` – Spacing between gutter ticks (default: 42pt)

To customize these, modify `omr_config.py` or instantiate custom objects in your scripts.

### Loading configuration in Python

```python
from omr_config_loader import load_sheet_config

# Load from JSON
layout = load_sheet_config("sheet_config.json")

# Access fields
print(f"Questions: {layout.max_questions}")
print(f"Options per question: {layout.question_options}")
```

### Saving configuration

```python
from omr_config import SheetLayout
from omr_config_loader import save_sheet_config

# Create custom layout
layout = SheetLayout(
    class_options=5,
    roll_columns=4,  # 4-digit roll numbers (always 10 rows for digits 0-9)
    set_options=4,
    question_options=5,  # A/B/C/D/E
    max_questions=80
)

# Save to JSON
save_sheet_config(layout, "custom_config.json")
```

## Layout engine & data flow

`omr_layout.py` is the shared geometry engine that ensures generator and processor stay synchronized.

### Key functions

**`build_section_specifications(sheet: SheetLayout)`**
Determines how roll numbers, class/set bubbles, and questions are partitioned across columns.

**`allocate_column_rows(specs, geometry, bubble_layout)`**
Fits sections into available vertical space and raises informative errors when layouts overflow.

**`calculate_anchor_positions(geometry, markers)`**
Computes the four corner anchor marker rectangles for PDF rendering.

**`calculate_anchor_centers(geometry, markers)`**
Returns anchor center points for image-based detection.

**`generate_all_bubble_coordinates(geometry, bubble_layout, sheet)`**
Yields every bubble's page coordinate as a `BubbleCoordinate` object, consumed by both:
- PDF renderer to draw circles
- Processor to sample pixel values

### Data flow

1. **Configuration** → `SheetLayout` + `BubbleLayout` + `PageGeometry`
2. **Layout engine** → Computes bubble coordinates, anchors, boxes
3. **Generator** → Draws PDF using coordinates
4. **Processor** → Samples image using the same coordinates
5. **Validation** → Coordinates match because both use the same source

This architecture eliminates coordinate drift and synchronization bugs.

## Diagnostics & utilities

Scripts in `tools/` extend the core workflow:

### `tools/test_distortions.py`

Synthesizes 19+ distortion scenarios and generates HTML reports:
- Perspective distortions (rotation, tilt, skew)
- Lighting variations (underexposed, overexposed, low contrast)
- Aspect ratio distortions (stretch, squash)
- Combined realistic scenarios

Outputs saved to `distortion_tests/` with side-by-side comparisons and pass/fail status.

### `tools/benchmark_omr.py`

Times each processing stage:
- `detect_anchor_markers` – Anchor detection time
- `correct_skew` – Perspective correction time
- Bubble analysis – Per-bubble sampling time

Optionally tracks memory usage via `psutil` (if installed).

### `tools/profile_bubble_fill.py`

Profiles `analyze_bubble_fill()` to expose:
- Per-bubble timing
- Memory allocation patterns
- Performance bottlenecks

Useful for optimizing high-volume processing.

### `tools/validate_bubbles.py`

Compares expected bubble coordinates against OpenCV Hough circle detections:
- Draws drift visualizations
- Highlights misaligned bubbles
- Helps tune scanning setups and detection parameters

**Usage:** Run from repository root with `python tools/<script>.py`. Each script adjusts `PYTHONPATH` automatically.

## Repository structure

```
.
├── fonts/                           # TrueType fonts for PDF generation
│   ├── NotoSans-Regular.ttf
│   ├── NotoSans-Bold.ttf
│   ├── NotoSans-Italic.ttf
│   └── StingerFitTrial-Bold.ttf
├── processed/                       # Processor outputs (safe to delete)
├── sheets/                          # Generated PDFs and input scans
├── tools/                           # Diagnostic and benchmarking utilities
│   ├── benchmark_omr.py
│   ├── profile_bubble_fill.py
│   ├── test_distortions.py
│   └── validate_bubbles.py
├── omr_config.py                    # Configuration dataclasses
├── omr_config_loader.py             # JSON load/save helpers
├── omr_layout.py                    # Shared geometry engine
├── omr_sheet_generator_new.py       # PDF generator (fpdf2)
├── omr_processor.py                 # Image processor (OpenCV)
├── pdf_to_image.py                  # PDF → PNG converter (PyMuPDF)
├── sheet_config.json                # Sample configuration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Troubleshooting

### Anchor detection fails

**Symptoms:** "Could not find all 4 anchor markers" or "No valid anchor found in zone"

**Solutions:**
- Ensure all four corner anchors are fully visible in the image
- Anchors must be inside these zones:
  - Horizontal: 25% from left/right edges
  - Top: 35% from top edge
  - Bottom: 20% from bottom edge
- Check that anchors are roughly square (aspect ratio validation ±15%)
- Increase anchor size in `MarkerConfig.anchor_size` if using low-resolution scans

### Incorrect bubble detection

**Symptoms:** Faint marks not detected, or dark areas falsely detected as marks

**Solutions:**
- Adjust `BubbleLayout.fill_threshold` in `omr_config.py`:
  - Lower values (0.3-0.35) → more sensitive, detects lighter marks
  - Higher values (0.45-0.5) → less sensitive, requires darker marks
- Ensure consistent pen/pencil type (ballpoint, gel, #2 pencil)
- Verify lighting is even during scanning/photography
- Check if adaptive threshold adjustment is working (processor logs threshold values)

### Layout mismatch errors

**Symptoms:** Bubbles detected at wrong positions, labels don't align

**Solutions:**
- Use the **exact same** JSON config and `--questions` value for both generator and processor
- Regenerate the PDF if you changed the config
- Verify `sheet_config.json` hasn't been modified between generation and processing

### Font errors

**Symptoms:** "Cannot load font" or "Font file not found"

**Solutions:**
- Confirm `fonts/` directory exists in the same location as `omr_sheet_generator_new.py`
- Check that all four font files are present (Noto Sans × 3, Stinger × 1)
- Update font paths in `omr_sheet_generator_new.py` lines 37-40 if you moved them

### PDF generation fails

**Symptoms:** "Layout does not fit", "Too many columns", or "Overflow" errors

**Solutions:**
- Reduce `max_questions` in config
- Decrease `question_options` if using 5+ choices
- Reduce `roll_columns` if layout is too wide
- Increase `BubbleLayout.radius` or gaps if bubbles overlap
- Check console warnings for specific dimension conflicts

### Low resolution images

**Symptoms:** Blurry bubbles, poor detection accuracy

**Solutions:**
- Scan at minimum 300 DPI (200 DPI acceptable for testing)
- Phone cameras: ensure good focus and adequate distance
- Processor enforces minimum ≈800×1000 px canvas
- Original resolution is preserved, only upscaled if too small

## License

MIT License – see `LICENSE` if included in your distribution.
