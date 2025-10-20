"""Benchmark OMR sheet processing performance.

This script measures processing time for OMR sheets with multiple scenarios:
- Full pipeline (with image saving)
- Processing only (without image saving)
- Detection only (without label overlay)
"""
from __future__ import annotations

import sys
import time
import gc
import argparse
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, memory stats will be disabled")

from omr_config import PageGeometry, BubbleLayout, MarkerConfig, SheetLayout
from omr_config_loader import load_sheet_config
from omr_processor import (
    detect_anchor_markers,
    correct_skew,
    sample_bubbles_from_coordinates,
    overlay_labels,
)


@dataclass
class TimingBreakdown:
    """Timing breakdown for each processing step."""
    load_image: float = 0.0
    detect_anchors: float = 0.0
    correct_skew: float = 0.0
    sample_bubbles: float = 0.0
    overlay_labels: float = 0.0
    save_image: float = 0.0
    total: float = 0.0


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    iterations: int
    timings: List[TimingBreakdown]
    memory_peak_mb: float = 0.0
    memory_avg_mb: float = 0.0


def process_with_timing(
    input_path: Path,
    output_path: Optional[Path],
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers_cfg: MarkerConfig,
    save_image: bool = True,
    overlay: bool = True,
) -> TimingBreakdown:
    """Process OMR sheet with detailed timing measurements.

    Args:
        input_path: Path to input image
        output_path: Path to save processed image (if save_image=True)
        geom: Page geometry configuration
        layout: Bubble layout configuration
        sheet: Sheet layout configuration
        markers_cfg: Marker configuration
        save_image: Whether to save the output image
        overlay: Whether to overlay labels

    Returns:
        TimingBreakdown with per-step timings
    """
    timing = TimingBreakdown()
    total_start = time.perf_counter()

    # Load image
    t0 = time.perf_counter()
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")
    timing.load_image = time.perf_counter() - t0

    # Detect anchors
    t0 = time.perf_counter()
    anchor_points = detect_anchor_markers(image, geom, markers_cfg)
    if anchor_points is None or len(anchor_points) != 4:
        raise ValueError("Failed to detect anchor markers")
    timing.detect_anchors = time.perf_counter() - t0

    # Correct skew
    t0 = time.perf_counter()
    corrected = correct_skew(image, anchor_points, geom, markers_cfg)
    timing.correct_skew = time.perf_counter() - t0

    # Sample bubbles
    t0 = time.perf_counter()
    group_samples = sample_bubbles_from_coordinates(
        corrected, geom, layout, sheet, markers_cfg
    )
    timing.sample_bubbles = time.perf_counter() - t0

    # Overlay labels (if enabled)
    if overlay:
        t0 = time.perf_counter()
        labeled = overlay_labels(corrected, group_samples)
        timing.overlay_labels = time.perf_counter() - t0
    else:
        labeled = corrected

    # Save image (if enabled)
    if save_image and output_path is not None:
        t0 = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), labeled)
        timing.save_image = time.perf_counter() - t0

    timing.total = time.perf_counter() - total_start
    return timing


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark(
    input_path: Path,
    output_path: Optional[Path],
    geom: PageGeometry,
    layout: BubbleLayout,
    sheet: SheetLayout,
    markers_cfg: MarkerConfig,
    iterations: int,
    warmup_runs: int,
    save_image: bool,
    overlay: bool,
    name: str,
) -> BenchmarkResult:
    """Run benchmark with specified parameters.

    Args:
        input_path: Path to input image
        output_path: Path for output (used only in first run if save_image=True)
        geom, layout, sheet, markers_cfg: Configuration objects
        iterations: Number of iterations to run
        warmup_runs: Number of warm-up runs (not counted)
        save_image: Whether to save output images
        overlay: Whether to overlay labels
        name: Name of this benchmark run

    Returns:
        BenchmarkResult with timing statistics
    """
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    # Warm-up runs
    if warmup_runs > 0:
        print(f"Warm-up: {warmup_runs} runs...", end=" ", flush=True)
        for _ in range(warmup_runs):
            try:
                process_with_timing(
                    input_path, None, geom, layout, sheet, markers_cfg,
                    save_image=False, overlay=overlay
                )
            except Exception as e:
                print(f"\nWarm-up failed: {e}")
                return BenchmarkResult(name, 0, [], 0.0, 0.0)
        print("Done.")

    # Main benchmark runs
    print(f"\nRunning {iterations} iterations...")
    timings: List[TimingBreakdown] = []
    memory_readings: List[float] = []

    for i in range(iterations):
        # Progress indicator
        if (i + 1) % 10 == 0 or i == 0:
            progress = (i + 1) / iterations * 100
            print(f"  Progress: {i+1}/{iterations} ({progress:.1f}%)", flush=True)

        # Save image only on first iteration if enabled
        should_save = save_image and i == 0
        output = output_path if should_save else None

        try:
            # Force garbage collection before measurement
            gc.collect()

            # Measure memory before
            mem_before = get_memory_usage_mb()

            # Run processing
            timing = process_with_timing(
                input_path, output, geom, layout, sheet, markers_cfg,
                save_image=should_save, overlay=overlay
            )
            timings.append(timing)

            # Measure memory after
            mem_after = get_memory_usage_mb()
            memory_readings.append(mem_after)

        except Exception as e:
            print(f"\n  Error on iteration {i+1}: {e}")
            continue

    # Calculate memory statistics
    memory_peak = max(memory_readings) if memory_readings else 0.0
    memory_avg = np.mean(memory_readings) if memory_readings else 0.0

    print(f"  Completed: {len(timings)}/{iterations} successful runs\n")

    return BenchmarkResult(name, len(timings), timings, memory_peak, memory_avg)


def print_statistics(result: BenchmarkResult) -> None:
    """Print detailed statistics for a benchmark result."""
    if not result.timings:
        print("No timing data available.\n")
        return

    # Extract timing arrays
    total_times = [t.total * 1000 for t in result.timings]  # Convert to ms
    load_times = [t.load_image * 1000 for t in result.timings]
    anchor_times = [t.detect_anchors * 1000 for t in result.timings]
    skew_times = [t.correct_skew * 1000 for t in result.timings]
    sample_times = [t.sample_bubbles * 1000 for t in result.timings]
    overlay_times = [t.overlay_labels * 1000 for t in result.timings]
    save_times = [t.save_image * 1000 for t in result.timings]

    # Calculate statistics
    avg_total = np.mean(total_times)
    median_total = np.median(total_times)
    min_total = np.min(total_times)
    max_total = np.max(total_times)
    std_total = np.std(total_times)
    throughput = 1000.0 / avg_total if avg_total > 0 else 0.0

    print(f"Results for: {result.name}")
    print(f"  Total runs: {result.iterations}")
    print(f"  Average time: {avg_total:.2f} ms/sheet")
    print(f"  Median time: {median_total:.2f} ms/sheet")
    print(f"  Min time: {min_total:.2f} ms/sheet")
    print(f"  Max time: {max_total:.2f} ms/sheet")
    print(f"  Std deviation: {std_total:.2f} ms")
    print(f"  Throughput: {throughput:.2f} sheets/sec")

    if PSUTIL_AVAILABLE:
        print(f"\nMemory Usage:")
        print(f"  Peak memory: {result.memory_peak_mb:.1f} MB")
        print(f"  Average memory: {result.memory_avg_mb:.1f} MB")

    # Per-step breakdown
    print(f"\nPer-step breakdown (average):")
    avg_load = np.mean(load_times)
    avg_anchor = np.mean(anchor_times)
    avg_skew = np.mean(skew_times)
    avg_sample = np.mean(sample_times)
    avg_overlay = np.mean(overlay_times)
    avg_save = np.mean(save_times)

    steps = [
        ("Image loading", avg_load),
        ("Anchor detection", avg_anchor),
        ("Skew correction", avg_skew),
        ("Bubble sampling", avg_sample),
        ("Label overlay", avg_overlay),
        ("Image saving", avg_save),
    ]

    for step_name, step_time in steps:
        if step_time > 0:
            percentage = (step_time / avg_total) * 100 if avg_total > 0 else 0
            print(f"  - {step_name:20s}: {step_time:6.2f} ms ({percentage:5.1f}%)")

    print()


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark OMR sheet processing performance"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to sheet_config.json file (optional, uses defaults if not provided)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("sheets/omr_sheet.png")],
        help="Path(s) to input OMR sheet image(s) (default: sheets/omr_sheet.png)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_output"),
        help="Path to output directory for benchmark results (default: benchmark_output)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations to run (default: 100)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup runs (default: 5)"
    )

    args = parser.parse_args()

    print("="*60)
    print("OMR SHEET PROCESSING BENCHMARK")
    print("="*60)

    # Load sheet configuration
    if args.config:
        try:
            sheet = load_sheet_config(args.config)
            print(f"Loaded configuration from: {args.config}")
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    else:
        sheet = SheetLayout()
        print("Using default sheet configuration")

    # Configuration
    geom = PageGeometry()
    layout = BubbleLayout()
    markers_cfg = MarkerConfig()

    benchmark_dir = args.output

    # Check input files
    test_images = []
    for input_path in args.input:
        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping")
        else:
            test_images.append(input_path)

    if not test_images:
        print("Error: No valid input images found")
        return

    # Benchmark parameters
    ITERATIONS = args.iterations
    WARMUP_RUNS = args.warmup

    all_results: List[BenchmarkResult] = []

    # Test scenarios for each image
    for test_image in test_images:
        image_name = test_image.stem

        # Scenario 1: Full pipeline (with image saving)
        result = run_benchmark(
            input_path=test_image,
            output_path=benchmark_dir / f"full_{image_name}.png",
            geom=geom,
            layout=layout,
            sheet=sheet,
            markers_cfg=markers_cfg,
            iterations=ITERATIONS,
            warmup_runs=WARMUP_RUNS,
            save_image=True,
            overlay=True,
            name=f"{image_name} - Full Pipeline (with I/O)",
        )
        all_results.append(result)
        print_statistics(result)

        # Scenario 2: Processing only (no image saving)
        result = run_benchmark(
            input_path=test_image,
            output_path=None,
            geom=geom,
            layout=layout,
            sheet=sheet,
            markers_cfg=markers_cfg,
            iterations=ITERATIONS,
            warmup_runs=WARMUP_RUNS,
            save_image=False,
            overlay=True,
            name=f"{image_name} - Processing Only (no I/O)",
        )
        all_results.append(result)
        print_statistics(result)

        # Scenario 3: Detection only (no overlay, no saving)
        result = run_benchmark(
            input_path=test_image,
            output_path=None,
            geom=geom,
            layout=layout,
            sheet=sheet,
            markers_cfg=markers_cfg,
            iterations=ITERATIONS,
            warmup_runs=WARMUP_RUNS,
            save_image=False,
            overlay=False,
            name=f"{image_name} - Detection Only (no overlay/I/O)",
        )
        all_results.append(result)
        print_statistics(result)

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"\n{'Scenario':<50s} {'Avg Time':>10s}")
    print("-"*60)

    for result in all_results:
        if result.timings:
            avg_time = np.mean([t.total * 1000 for t in result.timings])
            print(f"{result.name:<50s} {avg_time:>9.2f} ms")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
