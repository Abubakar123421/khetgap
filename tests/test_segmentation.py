from __future__ import annotations

import numpy as np

from src.models import CropConfig
from src.segmentation import clean_mask, vegetation_mask


def test_green_region_survives_segmentation() -> None:
    image = np.full((120, 180, 3), (140, 100, 65), dtype=np.uint8)
    image[35:85, 40:140] = (35, 150, 45)
    config = CropConfig()
    _, raw = vegetation_mask(image, config)
    cleaned = clean_mask(raw, config, 1.0)
    assert cleaned[60, 90] == 255
    assert cleaned[10, 10] == 0

