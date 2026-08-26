"""Interpretable vegetation segmentation for RGB aerial imagery."""

from __future__ import annotations

import cv2
import numpy as np

from .models import CropConfig
from .preprocessing import scaled_pixels


def excess_green(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.int16)
    red, green, blue = values[:, :, 0], values[:, :, 1], values[:, :, 2]
    exg = 2 * green - red - blue
    return cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def vegetation_mask(
    rgb: np.ndarray, config: CropConfig
) -> tuple[np.ndarray, np.ndarray]:
    exg = excess_green(rgb)
    exg_mask = exg > config.exg_threshold
    if config.use_hsv:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv_mask = cv2.inRange(
            hsv,
            np.asarray(config.hsv_lower, dtype=np.uint8),
            np.asarray(config.hsv_upper, dtype=np.uint8),
        ) > 0
        # HSV prevents bright neutral soil from passing normalized ExG while ExG
        # keeps the method sensitive to vegetation with muted saturation.
        mask = exg_mask & hsv_mask
    else:
        mask = exg_mask
    return exg, mask.astype(np.uint8) * 255


def clean_mask(mask: np.ndarray, config: CropConfig, scale: float) -> np.ndarray:
    kernel_size = scaled_pixels(config.morphology_kernel_px, scale)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=config.morphology_iterations,
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=config.morphology_iterations,
    )

    minimum_area = max(1, int(round(config.min_component_area_px * scale * scale)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    if count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_area
    return (keep[labels].astype(np.uint8) * 255)
