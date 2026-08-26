"""Crop-row band detection and occupancy profiling."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .models import CropConfig
from .preprocessing import scaled_pixels


@dataclass(frozen=True)
class RowBand:
    row_id: int
    y_center: int
    half_width: int
    x_start: int
    x_end: int
    vegetation_fraction: float

    @property
    def length_working_px(self) -> float:
        return float(max(0, self.x_end - self.x_start + 1))


def resolve_row_spacing_px(
    config: CropConfig, meters_per_pixel: float | None, scale: float
) -> int:
    if meters_per_pixel is not None:
        value = config.expected_row_spacing_m / meters_per_pixel * scale
        return max(4, int(round(value * 0.60)))
    return scaled_pixels(config.min_row_spacing_px, scale, 4)


def detect_row_bands(
    rotated_mask: np.ndarray,
    config: CropConfig,
    scale: float,
    meters_per_pixel: float | None,
) -> tuple[list[RowBand], np.ndarray]:
    binary = rotated_mask > 0
    profile = binary.sum(axis=1).astype(float)
    smoothed = gaussian_filter1d(profile, sigma=max(1.0, 2.0 * scale))
    prominence = max(2.0, float(smoothed.max()) * config.row_prominence_ratio)
    peaks, _ = find_peaks(
        smoothed,
        distance=resolve_row_spacing_px(config, meters_per_pixel, scale),
        prominence=prominence,
    )
    half_width = scaled_pixels(config.row_band_half_width_px, scale, 2)
    rows: list[RowBand] = []
    for peak in sorted(int(value) for value in peaks):
        y1 = max(0, peak - half_width)
        y2 = min(binary.shape[0], peak + half_width + 1)
        band = binary[y1:y2]
        occupancy = band.mean(axis=0)
        support = occupancy >= max(0.04, config.occupancy_threshold * 0.35)
        indices = np.flatnonzero(support)
        if indices.size < 2:
            continue
        x_start, x_end = int(indices[0]), int(indices[-1])
        row_region = band[:, x_start : x_end + 1]
        vegetation_fraction = float(row_region.mean())
        if vegetation_fraction < config.min_row_vegetation_fraction:
            continue
        rows.append(
            RowBand(
                row_id=len(rows) + 1,
                y_center=peak,
                half_width=half_width,
                x_start=x_start,
                x_end=x_end,
                vegetation_fraction=vegetation_fraction,
            )
        )
    return rows, smoothed


def row_occupancy(
    rotated_mask: np.ndarray, row: RowBand, config: CropConfig, scale: float
) -> np.ndarray:
    y1 = max(0, row.y_center - row.half_width)
    y2 = min(rotated_mask.shape[0], row.y_center + row.half_width + 1)
    band = rotated_mask[y1:y2, row.x_start : row.x_end + 1] > 0
    occupancy = band.mean(axis=0).astype(float)
    sigma = max(0.5, config.occupancy_smoothing_sigma * scale)
    return gaussian_filter1d(occupancy, sigma=sigma)


def render_row_debug(rotated_mask: np.ndarray, rows: list[RowBand]) -> np.ndarray:
    canvas = cv2.cvtColor(rotated_mask, cv2.COLOR_GRAY2RGB)
    for row in rows:
        cv2.line(
            canvas,
            (row.x_start, row.y_center),
            (row.x_end, row.y_center),
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas

