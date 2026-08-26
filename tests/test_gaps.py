from __future__ import annotations

import numpy as np

from src.gaps import find_false_runs, merge_close_runs


def test_find_false_runs() -> None:
    present = np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    assert find_false_runs(present, 3) == [(2, 4), (8, 11)]


def test_merge_close_runs() -> None:
    assert merge_close_runs([(2, 5), (8, 11), (20, 24)], 2) == [(2, 11), (20, 24)]

