from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_field() -> np.ndarray:
    image = np.full((360, 760, 3), (132, 91, 54), dtype=np.uint8)
    gaps = [(120, 190), (330, 410), (570, 660)]
    for row_index, y in enumerate(range(40, 341, 40)):
        cv2.line(image, (20, y), (740, y), (38, 145, 54), 9, cv2.LINE_AA)
        x1, x2 = gaps[row_index % len(gaps)]
        cv2.rectangle(image, (x1, y - 7), (x2, y + 7), (132, 91, 54), -1)
    return image

