"""Map detections back to original pixels and render presentation overlays."""

from __future__ import annotations

import cv2
import numpy as np

from .gaps import GapCandidate
from .models import Gap
from .orientation import RotatedImage
from .preprocessing import WorkingImage
from .rows import RowBand


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return homogeneous @ matrix.T


def candidate_to_gap(
    candidate: GapCandidate,
    rotation: RotatedImage,
    working: WorkingImage,
    meters_per_pixel: float | None,
) -> Gap:
    y1 = candidate.y_center - candidate.half_width
    y2 = candidate.y_center + candidate.half_width
    polygon_rotated = np.asarray(
        [
            [candidate.start_x, y1],
            [candidate.end_x, y1],
            [candidate.end_x, y2],
            [candidate.start_x, y2],
        ],
        dtype=np.float64,
    )
    polygon_working = _apply_affine(polygon_rotated, rotation.inverse_matrix)
    polygon_original = working.working_to_original(polygon_working)
    height, width = working.original_shape
    polygon_original[:, 0] = np.clip(polygon_original[:, 0], 0, width - 1)
    polygon_original[:, 1] = np.clip(polygon_original[:, 1], 0, height - 1)
    polygon_int = [tuple(map(int, np.rint(point))) for point in polygon_original]
    length_px = candidate.length_working_px / working.scale
    return Gap(
        row_id=candidate.row_id,
        start_px=float(candidate.start_x),
        end_px=float(candidate.end_x),
        length_px=float(length_px),
        length_m=float(length_px * meters_per_pixel) if meters_per_pixel else None,
        confidence=candidate.confidence,
        polygon_original=polygon_int,
        center_x_px=(candidate.start_x + candidate.end_x) / 2.0,
    )


def _row_segment_original(
    row: RowBand, rotation: RotatedImage, working: WorkingImage
) -> np.ndarray:
    points = np.asarray(
        [[row.x_start, row.y_center], [row.x_end, row.y_center]], dtype=np.float64
    )
    points = _apply_affine(points, rotation.inverse_matrix)
    return working.working_to_original(points)


def draw_gap_overlay(
    original: np.ndarray,
    gaps: list[Gap],
    rows: list[RowBand],
    rotation: RotatedImage,
    working: WorkingImage,
    draw_row_guides: bool = False,
) -> np.ndarray:
    base = original.copy()
    layer = base.copy()
    if draw_row_guides:
        for row in rows:
            endpoints = np.rint(_row_segment_original(row, rotation, working)).astype(int)
            cv2.line(
                layer,
                tuple(endpoints[0]),
                tuple(endpoints[1]),
                (255, 220, 0),
                1,
                cv2.LINE_AA,
            )
    for gap in gaps:
        polygon = np.asarray(gap.polygon_original, dtype=np.int32)
        cv2.fillPoly(layer, [polygon], color=(255, 20, 20))
    alpha = 0.42 if gaps or draw_row_guides else 0.0
    overlay = cv2.addWeighted(layer, alpha, base, 1.0 - alpha, 0.0)
    for gap in gaps:
        polygon = np.asarray(gap.polygon_original, dtype=np.int32)
        cv2.polylines(overlay, [polygon], True, (255, 0, 0), 2, cv2.LINE_AA)
    return overlay
