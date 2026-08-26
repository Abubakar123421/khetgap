from __future__ import annotations

import cv2
import numpy as np

from src.models import CropConfig
from src.orientation import estimate_hough_angle, normalize_line_angle


def test_normalize_line_angle() -> None:
    assert normalize_line_angle(100) == -80
    assert normalize_line_angle(-100) == 80


def test_hough_estimates_parallel_rows() -> None:
    mask = np.zeros((300, 500), dtype=np.uint8)
    for y in range(40, 280, 40):
        cv2.line(mask, (20, y), (470, y + 80), 255, 5)
    result = estimate_hough_angle(mask, CropConfig(), 1.0)
    assert result is not None
    assert abs(result.angle_deg - 10.1) < 3.0

