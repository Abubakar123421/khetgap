"""Dominant crop-row orientation estimation and image rotation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .models import CropConfig
from .preprocessing import scaled_pixels

# Planting axis vs alley axis only. Not a lock to 0°/90° yaw.
_AXIS_FLIP_RATIO = 1.12
_XY_SWAP = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


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


def perpendicular_line_angle(angle_deg: float) -> float:
    return normalize_line_angle(angle_deg + 90.0)


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
    # Full line-angle circle so drone yaw is not clipped to ±45°.
    limit = min(90.0, max(float(config.orientation_search_limit_deg), 90.0))
    step = config.orientation_search_step_deg
    angles = np.arange(-limit, limit + step * 0.5, step)
    scores = np.asarray([_projection_score(mask, float(angle)) for angle in angles])
    best_index = int(np.argmax(scores))
    best = float(scores[best_index])
    baseline = float(np.median(scores))
    confidence = float(np.clip((best - baseline) / max(best, 1e-9), 0.0, 1.0))
    return OrientationEstimate(float(angles[best_index]), confidence, "projection", 0)


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


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    stacked = np.eye(3, dtype=np.float64)
    stacked[:2, :] = np.asarray(matrix, dtype=np.float64)
    return stacked


def vertical_deskew_angle(horizontal_deskew_deg: float) -> float:
    """Rotation that stands planting lines up, given the rotation that lays them flat."""
    return float(horizontal_deskew_deg + 90.0)


def transpose_to_detection(vertical: RotatedImage) -> RotatedImage:
    """Map vertical planting columns onto horizontal bands for 1D occupancy.

    Detection stays the existing row/gap code. Inverse maps those coordinates
    back through the swap into the vertical deskew, then into the working image.
    """
    image = np.ascontiguousarray(vertical.image.T)
    forward = (_XY_SWAP @ _homogeneous(vertical.forward_matrix))[:2]
    inverse = (_homogeneous(vertical.inverse_matrix) @ _XY_SWAP)[:2]
    return RotatedImage(image, forward, inverse)


def align_planting_vertical(
    mask: np.ndarray, horizontal_deskew_deg: float
) -> tuple[RotatedImage, RotatedImage]:
    """Deskew planting lines to vertical, then transpose for the occupancy detector."""
    vertical = rotate_keep_bounds(mask, vertical_deskew_angle(horizontal_deskew_deg))
    return vertical, transpose_to_detection(vertical)


def row_axis_score(
    mask: np.ndarray, angle_deg: float, config: CropConfig, scale: float
) -> float:
    """Score how well `angle_deg` lays planting lines horizontal.

    Used only to pick planting axis versus alley axis (θ vs θ+90), not to lock
    yaw to cardinal angles. After that pick, the pipeline stands the lines up.
    """
    rotated = rotate_keep_bounds(mask, angle_deg).image
    binary = rotated > 0
    profile = binary.sum(axis=1).astype(np.float64)
    peak = float(profile.max())
    if peak <= 0.0 or profile.size < 3:
        return 0.0
    smoothed = gaussian_filter1d(profile, sigma=max(1.0, 2.0 * scale))
    min_distance = scaled_pixels(config.min_row_spacing_px, scale, 4)
    prominence = max(2.0, float(smoothed.max()) * config.row_prominence_ratio)
    peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=prominence)
    if peaks.size < 2:
        return 0.0
    spacings = np.diff(peaks).astype(np.float64)
    mean_spacing = float(spacings.mean())
    spacing_cv = float(spacings.std() / max(mean_spacing, 1e-6))
    relative = smoothed / max(float(smoothed.max()), 1e-9)
    variance = float(np.var(relative))
    empty_fraction = float((relative < 0.20).mean())
    half = max(2, scaled_pixels(config.row_band_half_width_px, scale, 2))
    row_fill = []
    alley_fill = []
    for peak_y in peaks:
        y1 = max(0, int(peak_y) - half)
        y2 = min(binary.shape[0], int(peak_y) + half + 1)
        row_fill.append(float(binary[y1:y2].mean()))
    alley_half = max(1, int(round(2.0 * scale)))
    for left, right in zip(peaks[:-1], peaks[1:]):
        mid = int((int(left) + int(right)) // 2)
        y1 = max(0, mid - alley_half)
        y2 = min(binary.shape[0], mid + alley_half + 1)
        alley_fill.append(float(binary[y1:y2].mean()))
    fill = float(np.mean(row_fill))
    alley = float(np.mean(alley_fill)) if alley_fill else 1.0
    contrast = (fill - alley) / max(fill, 1e-6)
    expected = float(min_distance)
    spacing_fit = 1.0
    if mean_spacing < expected * 0.80:
        spacing_fit = mean_spacing / expected
    height = float(rotated.shape[0])
    if mean_spacing > height * 0.35:
        spacing_fit *= (height * 0.35) / mean_spacing
    regularity = 1.0 / (1.0 + 4.0 * spacing_cv)
    return float(
        peaks.size
        * variance
        * regularity
        * (0.25 + empty_fraction)
        * (0.25 + max(contrast, 0.0))
        * spacing_fit
    )


def disambiguate_row_axis(
    mask: np.ndarray,
    estimate: OrientationEstimate,
    config: CropConfig,
    scale: float,
) -> OrientationEstimate:
    angle = normalize_line_angle(estimate.angle_deg)
    perpendicular = perpendicular_line_angle(angle)
    score = row_axis_score(mask, angle, config, scale)
    perpendicular_score = row_axis_score(mask, perpendicular, config, scale)
    if perpendicular_score > score * _AXIS_FLIP_RATIO:
        confidence = float(
            np.clip(
                perpendicular_score / max(perpendicular_score + score, 1e-9),
                0.0,
                1.0,
            )
        )
        return OrientationEstimate(
            perpendicular,
            max(float(estimate.confidence), confidence),
            estimate.method,
            estimate.line_count,
        )
    return OrientationEstimate(angle, estimate.confidence, estimate.method, estimate.line_count)


def estimate_row_angle(
    mask: np.ndarray, config: CropConfig, scale: float
) -> OrientationEstimate:
    """Estimate any-yaw rotation that lays planting lines horizontal.

    Hough or a full ±90° projection search finds the continuous yaw. Axis
    disambiguation then chooses planting lines versus alleys. The pipeline
    adds 90° so working-space columns run vertically.
    """
    hough = estimate_hough_angle(mask, config, scale)
    if hough is not None and hough.confidence >= 0.20:
        candidate = hough
    else:
        candidate = projection_search(mask, config)
    return disambiguate_row_axis(mask, candidate, config, scale)

