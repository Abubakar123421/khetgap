"""Pixel/calibrated measurements and row-priority ranking."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import CropConfig, Gap
from .rows import RowBand


METRIC_KEYS = (
    "rows_detected",
    "gaps_detected",
    "total_row_length_px",
    "total_gap_length_px",
    "gap_percent",
    "largest_gap_px",
    "missing_length_m",
    "largest_gap_m",
    "estimated_replant_units",
    "priority_rows",
)


def empty_metrics() -> dict[str, Any]:
    return {
        "rows_detected": 0,
        "gaps_detected": 0,
        "total_row_length_px": 0.0,
        "total_gap_length_px": 0.0,
        "gap_percent": 0.0,
        "largest_gap_px": 0.0,
        "missing_length_m": None,
        "largest_gap_m": None,
        "estimated_replant_units": None,
        "priority_rows": [],
    }


def calculate_metrics(
    rows: list[RowBand],
    gaps: list[Gap],
    scale: float,
    meters_per_pixel: float | None,
    config: CropConfig,
) -> dict[str, Any]:
    row_lengths = {row.row_id: row.length_working_px / scale for row in rows}
    total_row = float(sum(row_lengths.values()))
    total_gap = float(sum(gap.length_px for gap in gaps))
    largest_gap = float(max((gap.length_px for gap in gaps), default=0.0))
    gap_percent = 100.0 * total_gap / total_row if total_row > 0 else 0.0

    row_gaps: dict[int, float] = defaultdict(float)
    for gap in gaps:
        row_gaps[gap.row_id] += gap.length_px
    priority = []
    for row_id, missing in row_gaps.items():
        valid = row_lengths.get(row_id, 0.0)
        priority.append(
            {
                "row_id": row_id,
                "missing_length_px": round(missing, 3),
                "gap_percent": round(100.0 * missing / valid, 3) if valid else 0.0,
                "gap_count": sum(gap.row_id == row_id for gap in gaps),
            }
        )
    priority.sort(key=lambda item: (-item["gap_percent"], item["row_id"]))

    missing_m = total_gap * meters_per_pixel if meters_per_pixel is not None else None
    largest_m = largest_gap * meters_per_pixel if meters_per_pixel is not None else None
    estimate = (
        missing_m * config.planting_units_per_meter * config.safety_factor
        if missing_m is not None
        else None
    )
    return {
        "rows_detected": len(rows),
        "gaps_detected": len(gaps),
        "total_row_length_px": round(total_row, 3),
        "total_gap_length_px": round(total_gap, 3),
        "gap_percent": round(gap_percent, 3),
        "largest_gap_px": round(largest_gap, 3),
        "missing_length_m": round(missing_m, 3) if missing_m is not None else None,
        "largest_gap_m": round(largest_m, 3) if largest_m is not None else None,
        "estimated_replant_units": round(estimate, 3) if estimate is not None else None,
        "priority_rows": priority,
    }

