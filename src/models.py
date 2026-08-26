"""Public data contract for the KhetGap analysis engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Mapping

import numpy as np

AnalysisStatus = Literal["ok", "invalid_input", "no_vegetation", "no_rows"]


@dataclass(frozen=True)
class CropConfig:
    """Validated crop and detector settings.

    Pixel thresholds refer to original-image pixels. The pipeline adjusts them
    when it downsizes an image for analysis.
    """

    name: str = "sugarcane_demo"
    max_width: int = 1600
    exg_threshold: int = 25
    use_hsv: bool = True
    hsv_lower: tuple[int, int, int] = (25, 35, 25)
    hsv_upper: tuple[int, int, int] = (100, 255, 255)
    morphology_kernel_px: int = 3
    morphology_iterations: int = 1
    min_component_area_px: int = 30
    hough_threshold: int = 35
    hough_min_line_length_px: int = 60
    hough_max_line_gap_px: int = 25
    min_orientation_lines: int = 3
    orientation_search_limit_deg: float = 45.0
    orientation_search_step_deg: float = 1.0
    expected_row_spacing_m: float = 1.2
    min_row_spacing_px: int = 24
    row_prominence_ratio: float = 0.12
    row_band_half_width_px: int = 5
    min_row_vegetation_fraction: float = 0.025
    occupancy_threshold: float = 0.18
    occupancy_smoothing_sigma: float = 2.0
    min_gap_m: float = 1.0
    min_gap_px: int = 30
    gap_border_margin_px: int = 4
    merge_gap_separation_px: int = 5
    require_gap_bracketing: bool = True
    planting_units_per_meter: float = 1.0
    safety_factor: float = 1.0
    pattern_x_tolerance_px: int = 35
    pattern_min_adjacent_rows: int = 3
    draw_row_guides: bool = False

    @classmethod
    def from_value(cls, value: CropConfig | Mapping[str, Any]) -> CropConfig:
        if isinstance(value, cls):
            return value.validated()
        if not isinstance(value, Mapping):
            raise TypeError("config must be a CropConfig or mapping")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown configuration fields: {', '.join(unknown)}")
        converted = dict(value)
        for key in ("hsv_lower", "hsv_upper"):
            if key in converted:
                converted[key] = tuple(converted[key])
        return cls(**converted).validated()

    def validated(self) -> CropConfig:
        if self.max_width < 200:
            raise ValueError("max_width must be at least 200")
        if not 0 <= self.exg_threshold <= 255:
            raise ValueError("exg_threshold must be between 0 and 255")
        if self.morphology_kernel_px < 1 or self.morphology_kernel_px % 2 == 0:
            raise ValueError("morphology_kernel_px must be a positive odd number")
        if self.row_band_half_width_px < 1 or self.min_row_spacing_px < 2:
            raise ValueError("row sizes must be positive")
        if not 0 < self.occupancy_threshold < 1:
            raise ValueError("occupancy_threshold must be between 0 and 1")
        if not 0 <= self.min_row_vegetation_fraction < 1:
            raise ValueError("min_row_vegetation_fraction must be in [0, 1)")
        if self.min_gap_px < 1 or self.min_gap_m <= 0:
            raise ValueError("minimum gap lengths must be positive")
        if self.planting_units_per_meter <= 0 or self.safety_factor < 1.0:
            raise ValueError("planting rate must be positive and safety_factor at least 1.0")
        if self.orientation_search_step_deg <= 0:
            raise ValueError("orientation_search_step_deg must be positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisOverrides:
    row_angle_deg: float | None = None
    roi_xyxy: tuple[int, int, int, int] | None = None

    @classmethod
    def from_value(
        cls, value: AnalysisOverrides | Mapping[str, Any] | None
    ) -> AnalysisOverrides:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value.validated()
        if not isinstance(value, Mapping):
            raise TypeError("overrides must be an AnalysisOverrides, mapping, or None")
        unknown = sorted(set(value) - {"row_angle_deg", "roi_xyxy"})
        if unknown:
            raise ValueError(f"unknown override fields: {', '.join(unknown)}")
        roi = value.get("roi_xyxy")
        return cls(
            row_angle_deg=value.get("row_angle_deg"),
            roi_xyxy=tuple(roi) if roi is not None else None,
        ).validated()

    def validated(self) -> AnalysisOverrides:
        if self.row_angle_deg is not None and not -90 <= self.row_angle_deg <= 90:
            raise ValueError("row_angle_deg must be between -90 and 90")
        if self.roi_xyxy is not None:
            if len(self.roi_xyxy) != 4 or any(
                not isinstance(value, (int, np.integer)) for value in self.roi_xyxy
            ):
                raise ValueError("roi_xyxy must contain four integers")
        return self


@dataclass
class Gap:
    row_id: int
    start_px: float
    end_px: float
    length_px: float
    length_m: float | None
    confidence: float
    polygon_original: list[tuple[int, int]]
    center_x_px: float


@dataclass
class AnalysisResult:
    status: AnalysisStatus
    overlay_image: np.ndarray
    metrics: dict[str, Any]
    gaps: list[Gap] = field(default_factory=list)
    pattern_hints: list[dict[str, Any]] = field(default_factory=list)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

