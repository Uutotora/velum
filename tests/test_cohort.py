"""Unit tests for velum_core.cohort (batch/population aggregation)."""
import numpy as np

from velum_core.analysis import compute_measurements
from velum_core.cohort import (
    per_cell_long,
    per_image_summary,
    pooled_values,
    population_stats,
)


def _record(name, n_squares, side=10, coverage=1.0):
    mask = np.zeros((120, 120), dtype=np.int32)
    idx = 1
    r = c = 5
    for _ in range(n_squares):
        mask[r:r + side, c:c + side] = idx
        idx += 1
        c += side + 5
        if c > 120 - side:
            c = 5
            r += side + 5
    res = compute_measurements(mask)
    return (name, res, coverage)


def test_per_image_summary_counts():
    records = [_record("imgA", 3), _record("imgB", 5)]
    cols, rows = per_image_summary(records)
    assert cols[0] == "image" and "n_cells" in cols
    assert len(rows) == 2
    counts = {r[0]: r[1] for r in rows}
    assert counts["imgA"] == 3 and counts["imgB"] == 5


def test_per_cell_long_totals():
    records = [_record("imgA", 3), _record("imgB", 5)]
    header, rows = per_cell_long(records)
    assert header[0] == "image"
    assert len(rows) == 8                       # 3 + 5 cells, one row each
    assert {r[0] for r in rows} == {"imgA", "imgB"}


def test_population_stats_pools_all_cells():
    records = [_record("imgA", 3), _record("imgB", 5)]
    stats = population_stats(records)
    assert stats["n_images"] == 2
    assert stats["total_cells"] == 8
    assert stats["features"]["area"]["n"] == 8
    # all squares identical → std of area is ~0
    assert stats["features"]["area"]["std"] < 1e-6
    assert stats["features"]["area"]["mean"] > 0


def test_pooled_values_length():
    records = [_record("imgA", 3), _record("imgB", 5)]
    vals = pooled_values(records, "area")
    assert len(vals) == 8


def test_empty_records_are_safe():
    assert per_image_summary([]) == (
        ["image", "n_cells", "median_diam", "mean_area",
         "median_area", "median_circularity", "coverage_%"], [])
    header, rows = per_cell_long([])
    assert header == ["image"] and rows == []
    stats = population_stats([])
    assert stats["n_images"] == 0 and stats["total_cells"] == 0
