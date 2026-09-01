from __future__ import annotations

import cv2
import numpy as np

from src import CropConfig, analyze_field
from src.orientation import rotate_keep_bounds
from src.overlay import _apply_affine
from tests.conftest import (
    VERTICAL_COLUMN_XS,
    VERTICAL_HOLE_BANDS,
    expected_clean_field_holes,
    expected_synthetic_field_holes,
)


def _gap_covers_hole(gap, x1: int, x2: int, y: int, y_tol: int = 16) -> bool:
    polygon = np.asarray(gap.polygon_original, dtype=float)
    _center_x, center_y = polygon.mean(axis=0)
    span_left = float(polygon[:, 0].min())
    span_right = float(polygon[:, 0].max())
    overlap = min(span_right, x2) - max(span_left, x1)
    return abs(center_y - y) <= y_tol and overlap >= 0.45 * (x2 - x1)


def _unmatched_holes(holes: list[tuple[int, int, int]], gaps) -> list[tuple[int, int, int]]:
    used: set[int] = set()
    missed: list[tuple[int, int, int]] = []
    for hole in holes:
        hit = None
        for index, gap in enumerate(gaps):
            if index in used:
                continue
            if _gap_covers_hole(gap, *hole):
                hit = index
                break
        if hit is None:
            missed.append(hole)
        else:
            used.add(hit)
    return missed


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
    assert result.metrics["gaps_detected"] >= 7
    assert result.metrics["missing_length_m"] is None
    assert result.overlay_image.shape == synthetic_field.shape
    assert np.any(result.overlay_image != synthetic_field)


def test_pipeline_detects_gaps_with_residual_vegetation(synthetic_field: np.ndarray) -> None:
    image = synthetic_field.copy()
    planted_gaps = [(120, 190), (330, 410), (570, 660)]
    for row_index, y in enumerate(range(40, 341, 40)):
        x1, x2 = planted_gaps[row_index % len(planted_gaps)]
        for x in range(x1 + 6, x2 - 6, 8):
            cv2.circle(image, (x, y), 2, (38, 145, 54), -1)
    result = analyze_field(
        image,
        _test_config(),
        overrides={"row_angle_deg": 0.0},
    )
    assert result.status == "ok", result.errors
    assert result.metrics["gaps_detected"] >= 6


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


def test_pipeline_detects_in_column_holes_not_alleys(
    vertical_column_field: np.ndarray,
) -> None:
    result = analyze_field(vertical_column_field, _test_config())
    assert result.status == "ok", result.errors
    assert abs(abs(float(result.metrics["row_angle_deg"])) - 90.0) < 15.0
    assert result.metrics["gaps_detected"] >= 8

    column_xs = np.asarray(VERTICAL_COLUMN_XS, dtype=float)
    alley_xs = (column_xs[:-1] + column_xs[1:]) / 2.0
    on_column = 0
    on_alley = 0
    hole_hits = 0
    for gap in result.gaps:
        polygon = np.asarray(gap.polygon_original, dtype=float)
        center_x, center_y = polygon.mean(axis=0)
        width = float(polygon[:, 0].max() - polygon[:, 0].min())
        height = float(polygon[:, 1].max() - polygon[:, 1].min())
        # In-column holes are tall along the planting line, not wide alley slabs.
        assert height > width
        if np.min(np.abs(column_xs - center_x)) <= np.min(np.abs(alley_xs - center_x)):
            on_column += 1
        else:
            on_alley += 1
        if any(low <= center_y <= high for low, high in VERTICAL_HOLE_BANDS):
            hole_hits += 1

    assert on_column > on_alley
    assert on_alley == 0
    assert on_column >= max(6, int(0.70 * len(result.gaps)))
    assert hole_hits >= max(4, int(0.50 * len(result.gaps)))
    n_columns = len(VERTICAL_COLUMN_XS)
    assert hole_hits >= n_columns
    assert "occupancy_mask" in result.debug_images


def test_pipeline_covers_known_synthetic_hole_rectangles(
    synthetic_field: np.ndarray,
) -> None:
    result = analyze_field(
        synthetic_field,
        _test_config(),
        overrides={"row_angle_deg": 0.0},
    )
    assert result.status == "ok", result.errors
    holes = expected_synthetic_field_holes()
    missed = _unmatched_holes(holes, result.gaps)
    assert missed == []
    assert len(result.gaps) == len(holes)


def test_pipeline_covers_known_clean_field_hole_rectangles(
    clean_field_image: np.ndarray,
) -> None:
    result = analyze_field(clean_field_image, CropConfig())
    assert result.status == "ok", result.errors
    holes = expected_clean_field_holes()
    missed = _unmatched_holes(holes, result.gaps)
    assert missed == []
    assert len(result.gaps) == len(holes)
    occupancy = result.debug_images["occupancy_mask"] > 0
    for gap in result.gaps:
        polygon = np.asarray(gap.polygon_original, dtype=np.int32)
        cover = np.zeros(occupancy.shape, dtype=np.uint8)
        cv2.fillPoly(cover, [polygon], 1)
        inside = occupancy[cover > 0]
        assert inside.size
        assert float(inside.mean()) < 0.35


def test_on_disk_synthetic_pngs_match_known_holes() -> None:
    from pathlib import Path

    from PIL import Image

    samples = Path("data/synthetic")
    clean_path = samples / "sugarcane_clean.png"
    hard_path = samples / "sugarcane_hard.png"
    negative_path = samples / "negative_no_rows.png"
    if clean_path.exists():
        clean = np.asarray(Image.open(clean_path).convert("RGB"))
        result = analyze_field(clean, CropConfig())
        assert result.status == "ok", result.errors
        missed = _unmatched_holes(expected_clean_field_holes(), result.gaps)
        assert missed == []
        assert len(result.gaps) == len(expected_clean_field_holes())
    if hard_path.exists():
        hard = np.asarray(Image.open(hard_path).convert("RGB"))
        result = analyze_field(hard, CropConfig())
        assert result.status == "ok", result.errors
        assert result.metrics["rows_detected"] == 9
        assert result.metrics["gaps_detected"] == 18
        assert abs(abs(float(result.metrics["row_angle_deg"])) - 14.0) < 3.0
    if negative_path.exists():
        negative = np.asarray(Image.open(negative_path).convert("RGB"))
        result = analyze_field(negative, CropConfig())
        assert result.status == "no_vegetation"


def test_blank_and_invalid_images_return_controlled_statuses() -> None:
    blank = np.full((200, 300, 3), 120, dtype=np.uint8)
    blank_result = analyze_field(blank, CropConfig())
    assert blank_result.status == "no_vegetation"
    assert blank_result.errors[0]["code"] == "no_vegetation"

    invalid_result = analyze_field(np.zeros((20, 20), dtype=np.uint8), CropConfig())
    assert invalid_result.status == "invalid_input"
    bad_scale = analyze_field(blank, CropConfig(), meters_per_pixel=0)
    assert bad_scale.status == "invalid_input"


def _hole_corners(x1: int, x2: int, y: int, half: int = 7) -> np.ndarray:
    return np.asarray(
        [[x1, y - half], [x2, y - half], [x2, y + half], [x1, y + half]],
        dtype=np.float64,
    )


def _aabb_hit_ratio(gap_poly, hole_corners: np.ndarray) -> float:
    gap = np.asarray(gap_poly, dtype=float)
    a1, a2 = gap.min(axis=0), gap.max(axis=0)
    b1, b2 = hole_corners.min(axis=0), hole_corners.max(axis=0)
    inter = np.maximum(0.0, np.minimum(a2, b2) - np.maximum(a1, b1))
    inter_area = float(inter[0] * inter[1])
    hole_area = max(float((b2[0] - b1[0]) * (b2[1] - b1[1])), 1.0)
    return inter_area / hole_area


def _unmatched_rotated_holes(holes: list[tuple[int, int, int]], gaps, matrix) -> int:
    missed = 0
    used: set[int] = set()
    for hole in holes:
        corners = _apply_affine(_hole_corners(*hole), matrix)
        hit = None
        for index, gap in enumerate(gaps):
            if index in used:
                continue
            if _aabb_hit_ratio(gap.polygon_original, corners) >= 0.20:
                hit = index
                break
        if hit is None:
            missed += 1
        else:
            used.add(hit)
    return missed


def test_pipeline_hits_along_line_holes_at_any_yaw(synthetic_field: np.ndarray) -> None:
    holes = expected_synthetic_field_holes()
    config = _test_config()
    for yaw in (0.0, 14.0, 40.0, 90.0):
        rotated = rotate_keep_bounds(synthetic_field, yaw)
        result = analyze_field(rotated.image, config)
        assert result.status == "ok", (yaw, result.errors)
        assert result.metrics["rows_detected"] >= 5, yaw
        missed = _unmatched_rotated_holes(holes, result.gaps, rotated.forward_matrix)
        assert missed == 0, (yaw, missed, result.metrics["gaps_detected"])
        assert "occupancy_mask" in result.debug_images
        vertical_mask = result.debug_images["rotated_mask"] > 0
        assert float(np.var(vertical_mask.mean(axis=0))) >= float(
            np.var(vertical_mask.mean(axis=1))
        ) * 0.85
