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


def minimum_gap_working_px(
    config: CropConfig, meters_per_pixel: float | None, scale: float
) -> int:
    if meters_per_pixel is not None:
        original_pixels = config.min_gap_m / meters_per_pixel
    else:
        original_pixels = float(config.min_gap_px)
    return max(2, int(round(original_pixels * scale)))


def detect_gaps(
    occupancy: np.ndarray,
    row: RowBand,
    config: CropConfig,
    meters_per_pixel: float | None,
    scale: float,
) -> list[GapCandidate]:
    present = occupancy >= config.occupancy_threshold
    minimum = minimum_gap_working_px(config, meters_per_pixel, scale)
    runs = find_false_runs(present, minimum)
    merge_distance = max(0, int(round(config.merge_gap_separation_px * scale)))
    runs = merge_close_runs(runs, merge_distance)
    margin = max(1, int(round(config.gap_border_margin_px * scale)))
    candidates: list[GapCandidate] = []

    for start, end in runs:
        if start <= margin or end >= len(present) - margin - 1:
            continue
        flank_width = max(3, min(minimum // 2, 12))
        left = present[max(0, start - flank_width) : start]
        right = present[end + 1 : min(len(present), end + 1 + flank_width)]
        left_support = float(left.mean()) if left.size else 0.0
        right_support = float(right.mean()) if right.size else 0.0
        if config.require_gap_bracketing and min(left_support, right_support) < 0.35:
            continue

        mean_occupancy = float(occupancy[start : end + 1].mean())
        deficit = np.clip(
            (config.occupancy_threshold - mean_occupancy)
            / max(config.occupancy_threshold, 1e-6),
            0.0,
            1.0,
        )
        length_score = np.clip((end - start + 1) / max(minimum * 2.0, 1.0), 0.0, 1.0)
        row_score = np.clip(row.vegetation_fraction / 0.25, 0.0, 1.0)
        flank_score = min(left_support, right_support)
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
                start_x=row.x_start + start,
                end_x=row.x_start + end,
                y_center=row.y_center,
                half_width=row.half_width,
                confidence=confidence,
            )
        )
    return candidates

