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
- **Grid-marker calibration** – Optional gutter tick detection refines bubble-row Y positions so warped scans stay aligned even after severe stretching.
- **Flexible question layouts** – Support for varying option counts across different question ranges (e.g., 4 options for questions 1-20, 5 options for 21-31, 3 options for 32-70) with automatic dynamic transition headers.
- **Dynamic column width optimization** – Intelligent space utilization that adjusts column widths based on actual question option counts, fitting more questions on the page when using fewer options.
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
python omr_sheet_generator_new.py --config sheet_config.json --output sheets/omr_sheet.pdf

# 2. (Optional) Render PDFs to PNG for dry runs
python pdf_to_image.py  # writes sheets/omr_sheet.png at 200 DPI

# 3. Process scans or photos dropped into sheets/
python omr_processor.py --config sheet_config.json
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
  --output sheets/omr_sheet.pdf
```

**Arguments:**
- `--config` – Path to JSON configuration file (optional). See [Configuration model](#configuration-model) for details.
- `--questions` – (Optional) Limit the number of question rows rendered. If not specified, uses `question_option_ranges` from config. This flag is primarily for backwards compatibility with older configs using `max_questions`.
- `--output` – Destination PDF path (default: `sheets/omr_sheet.pdf`). Parent directories created automatically.

**Note:** The generator validates that your requested layout fits on the page and logs warnings if sections overflow. With the dynamic column width optimization, the system automatically fits as many questions as possible based on the available space and option counts.

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
  --config sheet_config.json
```

**Arguments:**
- `--config` – Path to JSON configuration file (must match the generator config used).
- `--questions` – (Optional) Limit questions processed. If not specified, processes all questions defined in `question_option_ranges`. This must match the generator setting if you used `--questions` during generation.

**Processing workflow:**
1. Place scanned sheets (PNG/JPG/JPEG) in `sheets/`.
2. The processor detects four square anchors in the corners and validates their geometry.
3. Perspective correction normalizes the sheet while preserving input resolution (minimum ≈800×1000 px).
4. When enabled, gutter grid markers along both margins are detected to measure vertical drift and calibrate each bubble row.
5. An adaptive fill threshold is computed from the inner zone between anchors (compensates for lighting variations).
6. Every bubble coordinate from `generate_all_bubble_coordinates()` is sampled and tagged (`class`, `roll`, `set`, or `question`).
7. Filled bubbles (darkness > threshold) are outlined in **magenta** and labeled.
8. Results saved to `processed/processed_<filename>.png` with console summaries.

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
  "class_options": 2,
  "class_section_options": 3,
  "roll_columns": 4,
  "set_options": 4,
  "question_option_ranges": [
    {"start": 1, "end": 20, "options": 4},
    {"start": 21, "end": 31, "options": 5},
    {"start": 32, "end": 70, "options": 3}
  ]
}
```

**Fields:**
- `class_options` – Number of class/grade bubbles (e.g., 2 for grades 6-7, or 5 for grades 6-10)
- `class_section_options` – Number of class section/division bubbles (e.g., 3 for sections a, b, c). Set to 0 to skip this section entirely.
- `roll_columns` – Number of digits in the roll number (e.g., 4 for roll numbers 0000-9999). Roll numbers always have 10 rows for digits 0-9.
- `set_options` – Number of test set/version bubbles (e.g., 4 for sets A-D)
- `question_option_ranges` – Array of question ranges with varying option counts. Each range specifies:
  - `start` – First question number in the range (inclusive)
  - `end` – Last question number in the range (inclusive)
  - `options` – Number of answer choices for questions in this range (e.g., 3 for A/B/C, 4 for A/B/C/D, 5 for A/B/C/D/E)

**Dynamic column width optimization:** The layout engine automatically calculates column widths based on the actual number of options in each question range. Questions with fewer options (e.g., 3 choices) use narrower columns than questions with more options (e.g., 5 choices), allowing more questions to fit on the page.

**Legacy format:** For backwards compatibility, you can use `"question_options": 4` and `"max_questions": 50` instead of `question_option_ranges`. This creates a single range with uniform options for all questions.

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
- `grid_calibration_enabled` – Toggle gutter-based row calibration (default: enabled)
- `grid_calibration_min_fraction` – Minimum row coverage required before calibration applies
- `grid_calibration_min_matches` – Absolute minimum number of tick rows required for calibration
- `grid_marker_distance_limit` – Maximum normalized distance (in vertical-gap units) for matching ticks to rows
- `grid_marker_area_tolerance` / `grid_marker_aspect_tolerance` – Shape filters for tick contours after rectification
- `grid_marker_outlier_sigma` – MAD-based rejection threshold for spurious tick matches
- `grid_marker_scale_tolerance` – Max allowed deviation from anchor-derived scale when fitting calibration

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
from omr_config import SheetLayout, QuestionOptionRange
from omr_config_loader import save_sheet_config

# Create custom layout with varying question options
layout = SheetLayout(
    class_options=5,
    class_section_options=3,  # Class sections (a, b, c)
    roll_columns=4,  # 4-digit roll numbers (always 10 rows for digits 0-9)
    set_options=4,
    question_option_ranges=[
        QuestionOptionRange(start=1, end=50, options=4),    # Questions 1-50: A/B/C/D
        QuestionOptionRange(start=51, end=80, options=5),   # Questions 51-80: A/B/C/D/E
    ]
)

# Save to JSON
save_sheet_config(layout, "custom_config.json")

# For uniform options across all questions (legacy format):
layout_simple = SheetLayout(
    class_options=5,
    class_section_options=3,
    roll_columns=4,
    set_options=4,
    question_options=4,      # All questions have 4 options
    max_questions=60         # Maximum 60 questions
)
```

## Layout engine & data flow

`omr_layout.py` is the shared geometry engine that ensures generator and processor stay synchronized.

### Key functions

**`build_section_specifications(sheet: SheetLayout)`**
Determines how roll numbers, class/set bubbles, and questions are partitioned across columns.

**`allocate_column_rows(specs, geometry, bubble_layout)`**
Fits sections into available vertical space and raises informative errors when layouts overflow.

**`simulate_single_column(specs, row_centers, is_first_column, question_counter, ...)`**
Simulates the allocation of a single column to determine its width and question range based on actual option counts. Part of the iterative column allocation algorithm.

**`calculate_anchor_positions(geometry, markers)`**
Computes the four corner anchor marker rectangles for PDF rendering.

**`calculate_anchor_centers(geometry, markers)`**
Returns anchor center points for image-based detection.

**`generate_all_bubble_coordinates(geometry, bubble_layout, sheet)`**
Uses iterative column allocation to generate all bubble coordinates. The algorithm:
1. Simulates each column to determine actual width needed based on question options
2. Checks if the column fits in available horizontal space
3. Generates bubble coordinates only for columns that fit
4. Automatically handles transitions between different option counts with dynamic headers

Returns every bubble's page coordinate as a `BubbleCoordinate` object, consumed by both:
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
