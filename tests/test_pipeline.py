from __future__ import annotations

import cv2
import numpy as np

from src import CropConfig, analyze_field


def _test_config() -> CropConfig:
    return CropConfig(
        use_hsv=True,
        min_row_spacing_px=25,
        min_gap_px=35,
        row_band_half_width_px=6,
        occupancy_threshold=0.16,
    )


def test_pipeline_detects_synthetic_gaps(synthetic_field: np.ndarray) -> None:
    result = analyze_field(
        synthetic_field,
        _test_config(),
        overrides={"row_angle_deg": 0.0},
    )
    assert result.status == "ok", result.errors
    assert result.metrics["rows_detected"] >= 7
    assert result.metrics["gaps_detected"] >= 3
    assert result.metrics["missing_length_m"] is None
    assert result.overlay_image.shape == synthetic_field.shape
    assert np.any(result.overlay_image != synthetic_field)


def test_pipeline_calibration_and_determinism(synthetic_field: np.ndarray) -> None:
    first = analyze_field(
        synthetic_field,
        _test_config(),
        meters_per_pixel=0.05,
        overrides={"row_angle_deg": 0.0},
    )
    second = analyze_field(
        synthetic_field,
        _test_config(),
        meters_per_pixel=0.05,
        overrides={"row_angle_deg": 0.0},
    )
    assert first.status == second.status == "ok"
    assert first.metrics == second.metrics
    assert first.metrics["missing_length_m"] is not None
    assert np.array_equal(first.overlay_image, second.overlay_image)


def test_pipeline_handles_rotated_rows(synthetic_field: np.ndarray) -> None:
    matrix = cv2.getRotationMatrix2D(
        (synthetic_field.shape[1] / 2, synthetic_field.shape[0] / 2), 13, 1.0
    )
    rotated = cv2.warpAffine(
        synthetic_field,
        matrix,
        (synthetic_field.shape[1], synthetic_field.shape[0]),
        borderValue=(132, 91, 54),
    )
    result = analyze_field(rotated, _test_config())
    assert result.status == "ok", result.errors
    assert result.metrics["orientation_method"] in {"hough", "projection"}
    assert result.metrics["rows_detected"] >= 5


def test_blank_and_invalid_images_return_controlled_statuses() -> None:
    blank = np.full((200, 300, 3), 120, dtype=np.uint8)
    blank_result = analyze_field(blank, CropConfig())
    assert blank_result.status == "no_vegetation"
    assert blank_result.errors[0]["code"] == "no_vegetation"

    invalid_result = analyze_field(np.zeros((20, 20), dtype=np.uint8), CropConfig())
    assert invalid_result.status == "invalid_input"
    bad_scale = analyze_field(blank, CropConfig(), meters_per_pixel=0)
    assert bad_scale.status == "invalid_input"

