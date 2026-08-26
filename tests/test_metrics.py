from __future__ import annotations

from src.metrics import calculate_metrics
from src.models import CropConfig, Gap
from src.rows import RowBand


def _inputs() -> tuple[list[RowBand], list[Gap]]:
    rows = [RowBand(1, 20, 4, 0, 199, 0.2), RowBand(2, 60, 4, 0, 199, 0.2)]
    gaps = [Gap(1, 20, 59, 40.0, None, 0.8, [], 39.5)]
    return rows, gaps


def test_metrics_no_scale_has_no_meter_values() -> None:
    rows, gaps = _inputs()
    metrics = calculate_metrics(rows, gaps, 1.0, None, CropConfig())
    assert metrics["gap_percent"] == 10.0
    assert metrics["missing_length_m"] is None
    assert metrics["estimated_replant_units"] is None


def test_metrics_with_scale_converts_lengths() -> None:
    rows, gaps = _inputs()
    gaps[0].length_m = 2.0
    metrics = calculate_metrics(rows, gaps, 1.0, 0.05, CropConfig())
    assert metrics["missing_length_m"] == 2.0
    assert metrics["estimated_replant_units"] == 2.0

