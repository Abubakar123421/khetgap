from __future__ import annotations

import cv2
import numpy as np
import pytest


CLEAN_FIELD_GAP_SPANS = ((155, 230), (385, 470), (650, 755))
SYNTHETIC_FIELD_GAP_SPANS = ((120, 190), (330, 410), (570, 660))


def expected_clean_field_holes() -> list[tuple[int, int, int]]:
    """Painted in-row hole rectangles for `scripts/generate_synthetic_data.clean_field`."""
    holes: list[tuple[int, int, int]] = []
    for row_index, y in enumerate(range(45, 421, 42)):
        for gap_index, (x1, x2) in enumerate(CLEAN_FIELD_GAP_SPANS):
            if (row_index + gap_index) % 3 != 2:
                holes.append((x1, x2, y))
    return holes


def expected_synthetic_field_holes() -> list[tuple[int, int, int]]:
    holes: list[tuple[int, int, int]] = []
    for row_index, y in enumerate(range(40, 341, 40)):
        x1, x2 = SYNTHETIC_FIELD_GAP_SPANS[row_index % len(SYNTHETIC_FIELD_GAP_SPANS)]
        holes.append((x1, x2, y))
    return holes


@pytest.fixture
def synthetic_field() -> np.ndarray:
    image = np.full((360, 760, 3), (132, 91, 54), dtype=np.uint8)
    for row_index, y in enumerate(range(40, 341, 40)):
        cv2.line(image, (20, y), (740, y), (38, 145, 54), 9, cv2.LINE_AA)
        x1, x2 = SYNTHETIC_FIELD_GAP_SPANS[row_index % len(SYNTHETIC_FIELD_GAP_SPANS)]
        cv2.rectangle(image, (x1, y - 7), (x2, y + 7), (132, 91, 54), -1)
    return image


@pytest.fixture
def clean_field_image() -> np.ndarray:
    image = np.full((440, 900, 3), (132, 91, 54), dtype=np.uint8)
    for row_index, y in enumerate(range(45, 421, 42)):
        cv2.line(image, (25, y), (875, y), (38, 145, 54), 9, cv2.LINE_AA)
        for gap_index, (x1, x2) in enumerate(CLEAN_FIELD_GAP_SPANS):
            if (row_index + gap_index) % 3 != 2:
                cv2.rectangle(image, (x1, y - 7), (x2, y + 7), (132, 91, 54), -1)
    return image


VERTICAL_COLUMN_XS = tuple(range(40, 601, 36))
VERTICAL_HOLE_BANDS = ((180, 280), (480, 590))


@pytest.fixture
def vertical_column_field() -> np.ndarray:
    """Planting columns run vertically; holes are missing plants along each column."""
    rng = np.random.default_rng(3)
    image = np.full((800, 640, 3), (132, 91, 54), dtype=np.uint8)
    for x in VERTICAL_COLUMN_XS:
        for y in range(40, 760, 28):
            if any(low <= y <= high for low, high in VERTICAL_HOLE_BANDS):
                continue
            radius_x = int(rng.integers(6, 12))
            radius_y = int(rng.integers(5, 10))
            jitter = int(rng.integers(-2, 3))
            cv2.ellipse(
                image,
                (x + jitter, y),
                (radius_x, radius_y),
                0,
                0,
                360,
                (38, 145, 54),
                -1,
            )
    return image
