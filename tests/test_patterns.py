from __future__ import annotations

from src.models import CropConfig, Gap
from src.patterns import analyze_gap_patterns


def test_repeated_position_hint_uses_cautious_wording() -> None:
    gaps = [
        Gap(row_id, 100, 150, 51, None, 0.8, [], 125 + row_id)
        for row_id in range(1, 5)
    ]
    hints = analyze_gap_patterns(gaps, CropConfig(), 1.0)
    assert hints[0]["type"] == "repeated_position"
    assert "may indicate" in hints[0]["message"].lower()
    assert len(hints) <= 2
