from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import prepare_working_image, validate_rgb_image


def test_roi_resize_and_coordinate_restore() -> None:
    image = np.zeros((1000, 2000, 3), dtype=np.uint8)
    working = prepare_working_image(image, 900, (100, 50, 1900, 950))
    assert working.image.shape[:2] == (450, 900)
    restored = working.working_to_original(np.array([[0.0, 0.0], [900.0, 450.0]]))
    assert np.allclose(restored[0], [100, 50])
    assert np.allclose(restored[1], [1900, 950])


def test_rejects_grayscale_and_non_uint8() -> None:
    with pytest.raises(ValueError, match="H x W x 3"):
        validate_rgb_image(np.zeros((100, 100), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        validate_rgb_image(np.zeros((100, 100, 3), dtype=np.float32))

