from __future__ import annotations

import numpy as np

from src.models import Gap
from src.orientation import RotatedImage
from src.overlay import draw_gap_overlay
from src.preprocessing import WorkingImage


def _identity_rotation(image: np.ndarray) -> RotatedImage:
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    return RotatedImage(image[:, :, 0], identity, identity)


def test_gap_overlay_contrasts_against_soil_and_canopy() -> None:
    height, width = 80, 160
    soil = (132, 91, 54)
    canopy = (38, 145, 54)
    image = np.full((height, width, 3), soil, dtype=np.uint8)
    image[:, :80] = canopy
    working = WorkingImage(image, 1.0, (0, 0, width, height), (height, width))
    gap = Gap(
        1,
        20,
        60,
        41.0,
        None,
        0.9,
        [(20, 20), (140, 20), (140, 60), (20, 60)],
        80.0,
    )
    overlay = draw_gap_overlay(
        image, [gap], [], _identity_rotation(image), working, False
    )
    soil_region = overlay[30:50, 100:130]
    canopy_region = overlay[30:50, 30:60]
    soil_delta = np.abs(soil_region.astype(int) - np.array(soil)).mean()
    canopy_delta = np.abs(canopy_region.astype(int) - np.array(canopy)).mean()
    assert soil_delta > 55
    assert canopy_delta > 55
    assert overlay.min() <= 30
    assert int(overlay[:, :, 0].max()) == 255
