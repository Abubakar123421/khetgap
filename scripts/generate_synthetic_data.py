"""Generate redistributable KhetGap demo and regression images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "synthetic"


def clean_field() -> np.ndarray:
    image = np.full((440, 900, 3), (132, 91, 54), dtype=np.uint8)
    gaps = [(155, 230), (385, 470), (650, 755)]
    for row_index, y in enumerate(range(45, 421, 42), start=0):
        cv2.line(image, (25, y), (875, y), (38, 145, 54), 9, cv2.LINE_AA)
        for gap_index, (x1, x2) in enumerate(gaps):
            if (row_index + gap_index) % 3 != 2:
                cv2.rectangle(image, (x1, y - 7), (x2, y + 7), (132, 91, 54), -1)
    return image


def hard_field() -> np.ndarray:
    rng = np.random.default_rng(2408)
    image = clean_field()
    for _ in range(450):
        x = int(rng.integers(0, image.shape[1]))
        y = int(rng.integers(0, image.shape[0]))
        radius = int(rng.integers(1, 4))
        cv2.circle(image, (x, y), radius, (55, 112, 45), -1)
    shadow = image.copy()
    cv2.rectangle(shadow, (250, 0), (420, image.shape[0]), (30, 35, 30), -1)
    image = cv2.addWeighted(shadow, 0.32, image, 0.68, 0)
    matrix = cv2.getRotationMatrix2D(
        (image.shape[1] / 2, image.shape[0] / 2), 14.0, 1.0
    )
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=(125, 88, 54),
    )


def negative_field() -> np.ndarray:
    image = np.full((440, 900, 3), (141, 104, 70), dtype=np.uint8)
    cv2.rectangle(image, (0, 260), (900, 440), (125, 93, 65), -1)
    return image


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    images = {
        "sugarcane_clean.png": clean_field(),
        "sugarcane_hard.png": hard_field(),
        "negative_no_rows.png": negative_field(),
    }
    for filename, array in images.items():
        Image.fromarray(array, mode="RGB").save(OUTPUT / filename)


if __name__ == "__main__":
    main()

