from __future__ import annotations

import cv2
import numpy as np

from src.models import CropConfig
from src.orientation import (
    align_planting_vertical,
    estimate_hough_angle,
    estimate_row_angle,
    normalize_line_angle,
    perpendicular_line_angle,
    projection_search,
    rotate_keep_bounds,
    row_axis_score,
    vertical_deskew_angle,
)
from src.preprocessing import prepare_working_image
from src.segmentation import clean_mask, vegetation_mask


def _mask_from_rgb(image: np.ndarray) -> tuple[np.ndarray, float]:
    config = CropConfig()
    working = prepare_working_image(image, config.max_width)
    _, raw = vegetation_mask(working.image, config)
    return clean_mask(raw, config, working.scale, close_holes=True), working.scale


def test_normalize_line_angle() -> None:
    assert normalize_line_angle(100) == -80
    assert normalize_line_angle(-100) == 80
    assert perpendicular_line_angle(0.0) == -90.0
    assert perpendicular_line_angle(-90.0) == 0.0


def test_hough_estimates_parallel_rows() -> None:
    mask = np.zeros((300, 500), dtype=np.uint8)
    for y in range(40, 280, 40):
        cv2.line(mask, (20, y), (470, y + 80), 255, 5)
    result = estimate_hough_angle(mask, CropConfig(), 1.0)
    assert result is not None
    assert abs(result.angle_deg - 10.1) < 3.0
    deskewed = estimate_row_angle(mask, CropConfig(), 1.0)
    assert abs(deskewed.angle_deg - 10.1) < 3.0


def test_row_angle_stays_horizontal_for_row_crops(synthetic_field: np.ndarray) -> None:
    mask, scale = _mask_from_rgb(synthetic_field)
    result = estimate_row_angle(mask, CropConfig(), scale)
    assert abs(result.angle_deg) < 8.0
    assert row_axis_score(mask, 0.0, CropConfig(), scale) > row_axis_score(
        mask, -90.0, CropConfig(), scale
    ) * 1.12


def test_row_angle_follows_vertical_columns(vertical_column_field: np.ndarray) -> None:
    mask, scale = _mask_from_rgb(vertical_column_field)
    hough = estimate_hough_angle(mask, CropConfig(), scale)
    assert hough is not None
    # Discrete stools align as horizontal ranks, so Hough alone locks onto 0°.
    assert abs(hough.angle_deg) < 12.0
    result = estimate_row_angle(mask, CropConfig(), scale)
    assert abs(abs(result.angle_deg) - 90.0) < 12.0
    assert row_axis_score(mask, -90.0, CropConfig(), scale) > row_axis_score(
        mask, 0.0, CropConfig(), scale
    ) * 1.12


def test_vertical_deskew_stands_lines_up() -> None:
    mask = np.zeros((240, 400), dtype=np.uint8)
    for y in range(30, 210, 30):
        cv2.line(mask, (20, y), (380, y), 255, 5)
    vertical, detection = align_planting_vertical(mask, 0.0)
    binary = vertical.image > 0
    x_profile = binary.mean(axis=0)
    y_profile = binary.mean(axis=1)
    assert float(np.var(x_profile)) > float(np.var(y_profile))
    assert abs(vertical_deskew_angle(0.0) - 90.0) < 1e-6
    assert detection.image.shape == vertical.image.T.shape


def test_estimate_row_angle_tracks_40_deg_yaw() -> None:
    mask = np.zeros((320, 480), dtype=np.uint8)
    for y in range(40, 280, 36):
        cv2.line(mask, (15, y), (465, y), 255, 6)
    tilted = rotate_keep_bounds(mask, 40.0).image
    result = estimate_row_angle(tilted, CropConfig(), 1.0)
    assert abs(result.angle_deg + 40.0) < 8.0 or abs(result.angle_deg - 40.0) < 8.0
    fallback = projection_search(tilted, CropConfig())
    assert fallback.method == "projection"


def test_detection_inverse_returns_working_points() -> None:
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[20:60, 40:80] = 255
    vertical, detection = align_planting_vertical(mask, 0.0)
    points = np.array([[10.0, 15.0], [40.0, 50.0]], dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    working = homogeneous @ detection.inverse_matrix.T
    back = np.column_stack([working, np.ones(len(working))]) @ detection.forward_matrix.T
    assert np.allclose(back, points, atol=1.5)

