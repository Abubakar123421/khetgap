"""Input validation, ROI handling, and resize-coordinate bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class WorkingImage:
    image: np.ndarray
    scale: float
    roi_xyxy: tuple[int, int, int, int]
    original_shape: tuple[int, int]

    def working_to_original(self, points: np.ndarray) -> np.ndarray:
        mapped = np.asarray(points, dtype=np.float64).copy()
        mapped[:, 0] = mapped[:, 0] / self.scale + self.roi_xyxy[0]
        mapped[:, 1] = mapped[:, 1] / self.scale + self.roi_xyxy[1]
        return mapped


def validate_rgb_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape H x W x 3")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 RGB values")
    if image.shape[0] < 20 or image.shape[1] < 20:
        raise ValueError("image must be at least 20 x 20 pixels")
    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    return image


def prepare_working_image(
    image: np.ndarray,
    max_width: int,
    roi_xyxy: tuple[int, int, int, int] | None = None,
) -> WorkingImage:
    image = validate_rgb_image(image)
    height, width = image.shape[:2]
    if roi_xyxy is None:
        roi = (0, 0, width, height)
    else:
        x1, y1, x2, y2 = roi_xyxy
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 20 or y2 - y1 < 20:
            raise ValueError("ROI must overlap at least a 20 x 20 pixel image region")
        roi = (x1, y1, x2, y2)

    x1, y1, x2, y2 = roi
    cropped = image[y1:y2, x1:x2].copy()
    scale = min(1.0, max_width / float(cropped.shape[1]))
    if scale < 1.0:
        size = (
            max(1, int(round(cropped.shape[1] * scale))),
            max(1, int(round(cropped.shape[0] * scale))),
        )
        cropped = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
    return WorkingImage(cropped, scale, roi, (height, width))


def scaled_pixels(value: int | float, scale: float, minimum: int = 1) -> int:
    return max(minimum, int(round(float(value) * scale)))

