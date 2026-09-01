from __future__ import annotations

import numpy as np

from src.gaps import (
    adaptive_occupancy_threshold,
    detect_gaps,
    find_false_runs,
    merge_close_runs,
    minimum_gap_working_px,
    suppress_short_islands,
)
from src.models import CropConfig
from src.rows import RowBand, row_occupancy


def test_find_false_runs() -> None:
    present = np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    assert find_false_runs(present, 3) == [(2, 4), (8, 11)]


def test_merge_close_runs() -> None:
    assert merge_close_runs([(2, 5), (8, 11), (20, 24)], 2) == [(2, 11), (20, 24)]


def test_merge_after_collecting_short_fragments() -> None:
    present = np.ones(120, dtype=bool)
    present[20:80] = False
    present[40:44] = True
    present[58:61] = True
    runs = merge_close_runs(find_false_runs(present, 1), 8)
    long_enough = [(start, end) for start, end in runs if end - start + 1 >= 30]
    assert long_enough == [(20, 79)]
    assert find_false_runs(present, 30) != long_enough


def test_suppress_short_islands_keeps_real_plants() -> None:
    present = np.ones(80, dtype=bool)
    present[10:40] = False
    present[18:22] = True
    present[50:56] = True
    cleaned = suppress_short_islands(present, 6)
    assert not cleaned[18:22].any()
    assert cleaned[50:56].all()


def test_minimum_gap_uses_more_sensitive_bound() -> None:
    config = CropConfig(min_gap_px=18, min_gap_m=1.0)
    assert minimum_gap_working_px(config, 0.02, 1.0) == 18
    assert minimum_gap_working_px(config, None, 1.0) == 18


def test_detect_gaps_recovers_residual_vegetation() -> None:
    occupancy = np.full(400, 0.55, dtype=float)
    occupancy[60:130] = 0.22
    occupancy[80:84] = 0.62
    occupancy[220:290] = 0.20
    occupancy[240:243] = 0.58
    row = RowBand(1, 20, 4, 0, 399, 0.4)
    config = CropConfig(occupancy_threshold=0.18, min_gap_px=30, max_gap_island_px=6)
    candidates = detect_gaps(occupancy, row, config, None, 1.0)
    assert adaptive_occupancy_threshold(occupancy, config) > 0.18
    assert len(candidates) == 2
    lengths = [candidate.length_working_px for candidate in candidates]
    assert all(length >= 30 for length in lengths)


def test_detect_gaps_keeps_trimmed_edge_hole() -> None:
    occupancy = np.ones(200, dtype=float)
    occupancy[:70] = 0.0
    occupancy[140:] = 0.0
    row = RowBand(1, 20, 4, 0, 199, 0.5)
    config = CropConfig(min_gap_px=30, gap_border_margin_px=4, require_gap_bracketing=True)
    candidates = detect_gaps(occupancy, row, config, None, 1.0)
    assert len(candidates) >= 1
    assert all(candidate.start_x > 4 for candidate in candidates)
    assert all(candidate.end_x < 195 for candidate in candidates)


def test_row_occupancy_follows_mask_holes_not_rgb() -> None:
    mask = np.zeros((48, 240), dtype=np.uint8)
    mask[20:29, :] = 255
    mask[20:29, 70:140] = 0
    row = RowBand(1, 24, 6, 0, 239, 0.6)
    occupancy = row_occupancy(mask, row, CropConfig(min_gap_px=18), 1.0)
    assert occupancy[30] > 0.55
    assert occupancy[100] < 0.20
    assert occupancy[200] > 0.55
    candidates = detect_gaps(occupancy, row, CropConfig(min_gap_px=18), None, 1.0)
    assert len(candidates) == 1
    assert candidates[0].start_x <= 80
    assert candidates[0].end_x >= 130
