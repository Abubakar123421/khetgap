"""Dominant crop-row orientation estimation and image rotation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import CropConfig
from .preprocessing import scaled_pixels


@dataclass(frozen=True)
class OrientationEstimate:
    angle_deg: float
    confidence: float
    method: str
    line_count: int


@dataclass(frozen=True)
class RotatedImage:
    image: np.ndarray
    forward_matrix: np.ndarray
    inverse_matrix: np.ndarray


def normalize_line_angle(angle_deg: float) -> float:
    while angle_deg >= 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0
    return float(angle_deg)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values, sorted_weights = values[order], weights[order]
    cutoff = sorted_weights.sum() * 0.5
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)])


def estimate_hough_angle(
    mask: np.ndarray, config: CropConfig, scale: float
) -> OrientationEstimate | None:
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(12, scaled_pixels(config.hough_threshold, scale)),
        minLineLength=scaled_pixels(config.hough_min_line_length_px, scale, 20),
        maxLineGap=scaled_pixels(config.hough_max_line_gap_px, scale, 4),
    )
    if lines is None:
        return None
    angles: list[float] = []
    lengths: list[float] = []
    # OpenCV 4 commonly returns N x 1 x 4 while OpenCV 5 may return N x 4.
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length <= 0:
            continue
        angles.append(normalize_line_angle(np.degrees(np.arctan2(dy, dx))))
        lengths.append(length)
    if len(angles) < config.min_orientation_lines:
        return None

    angle_values = np.asarray(angles)
    weights = np.asarray(lengths)
    initial = _weighted_median(angle_values, weights)
    residuals = np.abs(angle_values - initial)
    residuals = np.minimum(residuals, 180.0 - residuals)
    keep = residuals <= 12.0
    if int(keep.sum()) < config.min_orientation_lines:
        return None
    angle = _weighted_median(angle_values[keep], weights[keep])
    spread = float(np.average(residuals[keep], weights=weights[keep]))
    support = float(weights[keep].sum() / max(weights.sum(), 1.0))
    confidence = float(np.clip(support * (1.0 - spread / 20.0), 0.0, 1.0))
    return OrientationEstimate(angle, confidence, "hough", int(keep.sum()))


def _projection_score(mask: np.ndarray, angle_deg: float) -> float:
    height, width = mask.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0
    )
    profile = (rotated > 0).mean(axis=1).astype(np.float64)
    if profile.size < 3:
        return 0.0
    return float(np.var(profile) + 0.25 * np.mean(np.abs(np.diff(profile))))


def projection_search(mask: np.ndarray, config: CropConfig) -> OrientationEstimate:
    limit = config.orientation_search_limit_deg
    step = config.orientation_search_step_deg
    angles = np.arange(-limit, limit + step * 0.5, step)
    scores = np.asarray([_projection_score(mask, float(angle)) for angle in angles])
    best_index = int(np.argmax(scores))
    best = float(scores[best_index])
    baseline = float(np.median(scores))
    confidence = float(np.clip((best - baseline) / max(best, 1e-9), 0.0, 1.0))
    return OrientationEstimate(float(angles[best_index]), confidence, "projection", 0)


def estimate_row_angle(
    mask: np.ndarray, config: CropConfig, scale: float
) -> OrientationEstimate:
    hough = estimate_hough_angle(mask, config, scale)
    if hough is not None and hough.confidence >= 0.20:
        return hough
    return projection_search(mask, config)


def rotate_keep_bounds(image: np.ndarray, angle_deg: float) -> RotatedImage:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(np.ceil(height * sine + width * cosine))
    new_height = int(np.ceil(height * cosine + width * sine))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return RotatedImage(rotated, matrix, cv2.invertAffineTransform(matrix))
