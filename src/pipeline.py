"""Public orchestration entry point for KhetGap field analysis."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from .gaps import detect_gaps
from .metrics import calculate_metrics, empty_metrics
from .models import AnalysisOverrides, AnalysisResult, CropConfig
from .orientation import OrientationEstimate, align_planting_vertical, estimate_row_angle
from .overlay import candidate_to_gap, draw_gap_overlay
from .patterns import analyze_gap_patterns
from .preprocessing import prepare_working_image
from .rows import detect_row_bands, render_column_debug, row_occupancy
from .segmentation import clean_mask, vegetation_mask


def _safe_overlay(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
        if image.dtype == np.uint8:
            return image.copy()
    return np.zeros((1, 1, 3), dtype=np.uint8)


def _failure(
    status: str,
    overlay: np.ndarray,
    code: str,
    message: str,
    warnings: list[str] | None = None,
    debug_images: dict[str, np.ndarray] | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        status=status,  # type: ignore[arg-type]
        overlay_image=overlay,
        metrics=empty_metrics(),
        warnings=list(warnings or []),
        errors=[{"code": code, "message": message}],
        debug_images=dict(debug_images or {}),
    )


def _projection_debug(profile: np.ndarray, width: int = 640, height: int = 180) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if profile.size == 0 or float(profile.max()) <= 0:
        return canvas
    x_values = np.linspace(0, width - 1, profile.size)
    normalized = profile / float(profile.max())
    y_values = (height - 1) - normalized * (height - 12)
    points = np.column_stack([x_values, y_values]).astype(np.int32)
    cv2.polylines(canvas, [points], False, (30, 115, 50), 2, cv2.LINE_AA)
    return canvas


def _occupancy_debug(profiles: list[tuple[int, np.ndarray]], width: int = 640) -> np.ndarray:
    row_height = 42
    canvas = np.full((max(1, len(profiles)) * row_height, width, 3), 255, dtype=np.uint8)
    colors = [(30, 110, 220), (20, 150, 70), (180, 80, 40)]
    for output_row, (row_id, profile) in enumerate(profiles):
        y_offset = output_row * row_height
        cv2.putText(
            canvas,
            f"Row {row_id}",
            (5, y_offset + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        if profile.size:
            x = np.linspace(75, width - 1, profile.size)
            y = y_offset + row_height - 5 - np.clip(profile, 0, 1) * (row_height - 10)
            points = np.column_stack([x, y]).astype(np.int32)
            cv2.polylines(
                canvas,
                [points],
                False,
                colors[output_row % len(colors)],
                1,
                cv2.LINE_AA,
            )
    return canvas


def analyze_field(
    image: np.ndarray,
    config: CropConfig | Mapping[str, Any],
    meters_per_pixel: float | None = None,
    overrides: AnalysisOverrides | Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Analyze one RGB aerial image without any UI dependency.

    Expected field-analysis failures are returned as structured results. Values
    expressed in metres are disabled whenever calibration is absent.
    """

    original = _safe_overlay(image)
    try:
        crop_config = CropConfig.from_value(config)
        selected_overrides = AnalysisOverrides.from_value(overrides)
        if meters_per_pixel is not None:
            meters_per_pixel = float(meters_per_pixel)
            if not np.isfinite(meters_per_pixel) or meters_per_pixel <= 0:
                raise ValueError("meters_per_pixel must be finite and greater than zero")
        working = prepare_working_image(
            image,
            max_width=crop_config.max_width,
            roi_xyxy=selected_overrides.roi_xyxy,
        )
    except (TypeError, ValueError) as exc:
        return _failure("invalid_input", original, "invalid_input", str(exc))

    warnings: list[str] = []
    if meters_per_pixel is None:
        warnings.append(
            "No calibration was supplied; metre lengths and replanting estimates are disabled."
        )

    try:
        exg, raw_mask = vegetation_mask(working.image, crop_config)
        mask = clean_mask(raw_mask, crop_config, working.scale, close_holes=True)
        occupancy_mask = clean_mask(
            raw_mask, crop_config, working.scale, close_holes=False
        )
        debug_images: dict[str, np.ndarray] = {
            "excess_green": exg,
            "raw_mask": raw_mask,
            "mask": mask,
            "occupancy_mask": occupancy_mask,
        }
        vegetation_fraction = float((mask > 0).mean())
        if vegetation_fraction < 0.0005:
            return _failure(
                "no_vegetation",
                original,
                "no_vegetation",
                "No usable vegetation was found. Adjust the ROI or vegetation thresholds.",
                warnings,
                debug_images,
            )

        if selected_overrides.row_angle_deg is None:
            orientation = estimate_row_angle(mask, crop_config, working.scale)
        else:
            orientation = OrientationEstimate(
                float(selected_overrides.row_angle_deg), 1.0, "manual", 0
            )
        if orientation.confidence < 0.25:
            warnings.append(
                "Row orientation confidence is low; use a manual angle or a tighter ROI if alignment is poor."
            )
        vertical, detection = align_planting_vertical(mask, orientation.angle_deg)
        occupancy_vertical = cv2.warpAffine(
            occupancy_mask,
            vertical.forward_matrix,
            (vertical.image.shape[1], vertical.image.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        occupancy_detection = np.ascontiguousarray(occupancy_vertical.T)
        debug_images["rotated_mask"] = vertical.image

        rows, projection = detect_row_bands(
            detection.image,
            crop_config,
            working.scale,
            meters_per_pixel,
        )
        debug_images["row_projection"] = _projection_debug(projection)
        debug_images["row_bands"] = render_column_debug(vertical.image, rows)
        if not rows:
            return _failure(
                "no_rows",
                original,
                "no_rows",
                "Vegetation was found, but no reliable parallel crop rows were detected.",
                warnings,
                debug_images,
            )

        candidates = []
        occupancy_profiles: list[tuple[int, np.ndarray]] = []
        for row in rows:
            occupancy = row_occupancy(
                occupancy_detection, row, crop_config, working.scale
            )
            occupancy_profiles.append((row.row_id, occupancy))
            candidates.extend(
                detect_gaps(
                    occupancy,
                    row,
                    crop_config,
                    meters_per_pixel,
                    working.scale,
                )
            )
        debug_images["occupancy_profiles"] = _occupancy_debug(occupancy_profiles)

        gaps = [
            candidate_to_gap(candidate, detection, working, meters_per_pixel)
            for candidate in candidates
        ]
        gaps.sort(key=lambda gap: (gap.row_id, gap.start_px))
        metrics = calculate_metrics(
            rows, gaps, working.scale, meters_per_pixel, crop_config
        )
        hints = analyze_gap_patterns(gaps, crop_config, working.scale)
        overlay = draw_gap_overlay(
            image,
            gaps,
            rows,
            detection,
            working,
            crop_config.draw_row_guides,
        )
        metrics["row_angle_deg"] = round(orientation.angle_deg, 3)
        metrics["orientation_confidence"] = round(orientation.confidence, 3)
        metrics["orientation_method"] = orientation.method
        metrics["analysis_scale"] = round(working.scale, 6)
        metrics["vegetation_fraction"] = round(vegetation_fraction, 6)
        return AnalysisResult(
            status="ok",
            overlay_image=overlay,
            metrics=metrics,
            gaps=gaps,
            pattern_hints=hints,
            debug_images=debug_images,
            warnings=warnings,
        )
    except (cv2.error, FloatingPointError, OverflowError) as exc:
        return _failure(
            "invalid_input",
            original,
            "analysis_failed",
            "Analysis could not be completed for this image and configuration.",
            warnings,
        )

