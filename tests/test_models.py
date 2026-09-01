from __future__ import annotations

import pytest

from src.models import AnalysisOverrides, CropConfig


def test_config_from_mapping_and_unknown_field() -> None:
    config = CropConfig.from_value({"exg_threshold": 31, "hsv_lower": [20, 30, 40]})
    assert config.exg_threshold == 31
    assert config.hsv_lower == (20, 30, 40)
    with pytest.raises(ValueError, match="unknown configuration"):
        CropConfig.from_value({"not_a_setting": 1})


def test_gap_detector_settings_validate() -> None:
    config = CropConfig.from_value({"gap_occupancy_ratio": 0.25, "max_gap_island_px": 4})
    assert config.gap_occupancy_ratio == 0.25
    with pytest.raises(ValueError, match="gap_occupancy_ratio"):
        CropConfig.from_value({"gap_occupancy_ratio": 1.5})


def test_override_validation() -> None:
    assert AnalysisOverrides.from_value({"row_angle_deg": 12}).row_angle_deg == 12
    with pytest.raises(ValueError, match="between -90 and 90"):
        AnalysisOverrides.from_value({"row_angle_deg": 120})

