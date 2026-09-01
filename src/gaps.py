"""One-dimensional planting-gap detection and candidate filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CropConfig
from .rows import RowBand


@dataclass(frozen=True)
class GapCandidate:
    row_id: int
    start_x: int
    end_x: int
    y_center: int
    half_width: int
    confidence: float

    @property
    def length_working_px(self) -> float:
        return float(self.end_x - self.start_x + 1)


def find_false_runs(present: np.ndarray, min_gap_px: int) -> list[tuple[int, int]]:
    values = np.asarray(present, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_present in enumerate(values):
        if not is_present and start is None:
            start = index
        elif is_present and start is not None:
            if index - start >= min_gap_px:
                runs.append((start, index - 1))
            start = None
    if start is not None and len(values) - start >= min_gap_px:
        runs.append((start, len(values) - 1))
    return runs


def merge_close_runs(
    runs: list[tuple[int, int]], max_separation_px: int
) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end - 1 <= max_separation_px:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def suppress_short_islands(present: np.ndarray, max_island_px: int) -> np.ndarray:
    """Treat brief vegetation specks inside a hole as part of the hole."""
    values = np.asarray(present, dtype=bool).copy()
    if max_island_px <= 0 or values.size == 0:
        return values
    index = 0
    count = values.size
    while index < count:
        if not values[index]:
            index += 1
            continue
        end = index + 1
        while end < count and values[end]:
            end += 1
        if end - index <= max_island_px and index > 0 and end < count:
            values[index:end] = False
        index = end
    return values


def minimum_gap_working_px(
    config: CropConfig, meters_per_pixel: float | None, scale: float
) -> int:
    pixel_limit = float(config.min_gap_px)
    if meters_per_pixel is not None:
        meter_limit = config.min_gap_m / meters_per_pixel
        # Take the more sensitive bound so a pessimistic metre scale cannot
        # hide holes the pixel slider already considers long enough.
        original_pixels = min(pixel_limit, meter_limit)
    else:
        original_pixels = pixel_limit
    return max(2, int(round(original_pixels * scale)))


def adaptive_occupancy_threshold(occupancy: np.ndarray, config: CropConfig) -> float:
    """Raise the present/absent cut toward typical plant occupancy on this row.

    A fixed global cut misses real holes that still contain weeds, shadow, or
    adjacent-row bleed above `occupancy_threshold`.
    """
    if occupancy.size == 0:
        return float(config.occupancy_threshold)
    baseline = float(np.percentile(occupancy, 75))
    floor = max(0.05, config.occupancy_threshold * 0.45)
    if baseline <= config.occupancy_threshold:
        return float(np.clip(max(floor, baseline * 0.55), 0.05, 0.85))
    blended = config.occupancy_threshold + (
        baseline - config.occupancy_threshold
    ) * config.gap_occupancy_ratio
    return float(np.clip(max(floor, blended), 0.05, 0.85))


def detect_gaps(
    occupancy: np.ndarray,
    row: RowBand,
    config: CropConfig,
    meters_per_pixel: float | None,
    scale: float,
) -> list[GapCandidate]:
    occupancy = np.asarray(occupancy, dtype=float)
    threshold = adaptive_occupancy_threshold(occupancy, config)
    present = suppress_short_islands(
        occupancy >= threshold,
        max(0, int(round(config.max_gap_island_px * scale))),
    )
    minimum = minimum_gap_working_px(config, meters_per_pixel, scale)
    # Collect every absent run first. Filtering by length before merge dropped
    # holes that weeds or mask specks had split into sub-minimum fragments.
    runs = find_false_runs(present, 1)
    merge_distance = max(0, int(round(config.merge_gap_separation_px * scale)))
    runs = merge_close_runs(runs, merge_distance)
    runs = [(start, end) for start, end in runs if end - start + 1 >= minimum]

    margin = max(1, int(round(config.gap_border_margin_px * scale)))
    candidates: list[GapCandidate] = []
    profile_end = len(present) - 1

    for start, end in runs:
        clipped_start, clipped_end = start, end
        if clipped_start <= margin:
            clipped_start = margin + 1
        if clipped_end >= profile_end - margin:
            clipped_end = profile_end - margin - 1
        if clipped_end - clipped_start + 1 < minimum:
            continue

        flank_width = max(3, min(minimum // 2, 16))
        left = present[max(0, clipped_start - flank_width) : clipped_start]
        right = present[clipped_end + 1 : min(len(present), clipped_end + 1 + flank_width)]
        left_support = float(left.mean()) if left.size else 0.0
        right_support = float(right.mean()) if right.size else 0.0
        if config.require_gap_bracketing:
            stronger = max(left_support, right_support)
            weaker = min(left_support, right_support)
            at_profile_edge = start <= margin or end >= profile_end - margin
            if stronger < 0.28:
                continue
            if not at_profile_edge and weaker < 0.10 and stronger < 0.55:
                continue

        mean_occupancy = float(occupancy[clipped_start : clipped_end + 1].mean())
        deficit = np.clip(
            (threshold - mean_occupancy) / max(threshold, 1e-6),
            0.0,
            1.0,
        )
        length_score = np.clip(
            (clipped_end - clipped_start + 1) / max(minimum * 2.0, 1.0), 0.0, 1.0
        )
        row_score = np.clip(row.vegetation_fraction / 0.25, 0.0, 1.0)
        flank_score = max(left_support, right_support)
        confidence = float(
            np.clip(
                0.40 * deficit
                + 0.25 * length_score
                + 0.20 * flank_score
                + 0.15 * row_score,
                0.0,
                1.0,
            )
        )
        candidates.append(
            GapCandidate(
                row_id=row.row_id,
                start_x=row.x_start + clipped_start,
                end_x=row.x_start + clipped_end,
                y_center=row.y_center,
                half_width=row.half_width,
                confidence=confidence,
            )
        )
    return candidates
