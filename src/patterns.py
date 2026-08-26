"""Conservative, non-diagnostic spatial pattern hints."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .models import CropConfig, Gap


def analyze_gap_patterns(
    gaps: list[Gap], config: CropConfig, scale: float
) -> list[dict[str, Any]]:
    if not gaps:
        return []
    hints: list[dict[str, Any]] = []
    tolerance = max(3.0, config.pattern_x_tolerance_px * scale)

    # Gaps are sorted first so clustering remains linear even on dense fields.
    # Recomputing every cluster mean for every candidate made large orthomosaic
    # crops needlessly expensive.
    clusters: list[list[Gap]] = []
    running_centers: list[float] = []
    for gap in sorted(gaps, key=lambda item: item.center_x_px):
        if not clusters or gap.center_x_px - running_centers[-1] > tolerance:
            clusters.append([gap])
            running_centers.append(gap.center_x_px)
            continue
        cluster = clusters[-1]
        count = len(cluster)
        cluster.append(gap)
        running_centers[-1] = (running_centers[-1] * count + gap.center_x_px) / (count + 1)
    recurring = max(clusters, key=lambda cluster: len({gap.row_id for gap in cluster}))
    rows = sorted({gap.row_id for gap in recurring})
    adjacent_pairs = sum(b - a == 1 for a, b in zip(rows, rows[1:]))
    if len(rows) >= config.pattern_min_adjacent_rows and adjacent_pairs >= len(rows) - 2:
        hints.append(
            {
                "type": "repeated_position",
                "confidence": round(min(0.9, 0.5 + 0.08 * len(rows)), 2),
                "message": (
                    "Several neighboring rows contain gaps at a similar travel position. "
                    "This may indicate a recurring planting interruption."
                ),
            }
        )

    per_row: dict[int, float] = defaultdict(float)
    for gap in gaps:
        per_row[gap.row_id] += gap.length_px
    if len(per_row) >= 2:
        ordered = sorted(per_row.items(), key=lambda item: item[1], reverse=True)
        worst_row, worst_length = ordered[0]
        other_median = float(np.median([value for _, value in ordered[1:]]))
        if worst_length > max(other_median * 1.8, other_median + 10.0):
            hints.append(
                {
                    "type": "row_outlier",
                    "confidence": 0.65,
                    "message": (
                        f"Row {worst_row} has substantially more missing length than nearby rows. "
                        "This may be a row-specific planting issue, damage, or image occlusion."
                    ),
                }
            )
    return hints[:2]
