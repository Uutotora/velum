"""Unit tests for napari_app.analysis (per-cell morphometry).

These are regression guards: they pin the numeric contract of
compute_measurements so refactors of the measurement pipeline cannot
silently change what scientists read off the table.
"""
import math

import numpy as np
import pytest

from napari_app.analysis import (
    compute_measurements,
    rows_as_csv,
    summary_line,
)


def _square(canvas, label, r0, c0, side, value=None, img=None):
    """Paint a filled square of ``side`` px at (r0, c0) into ``canvas``."""
    canvas[r0:r0 + side, c0:c0 + side] = label
    if img is not None and value is not None:
        img[r0:r0 + side, c0:c0 + side] = value


def test_empty_mask_returns_no_cells():
    mask = np.zeros((32, 32), dtype=np.int32)
    res = compute_measurements(mask)
    assert res["n_cells"] == 0
    assert res["rows"] == []
    assert res["summary"] == {}
    # columns are always defined so the UI can render an empty table
    assert any(k == "area" for k, _l, _u in res["columns"])


def test_single_square_area_and_centroid_exact():
    mask = np.zeros((40, 40), dtype=np.int32)
    _square(mask, 1, r0=5, c0=5, side=20)  # rows/cols 5..24
    res = compute_measurements(mask)
    assert res["n_cells"] == 1
    cols = [k for k, _l, _u in res["columns"]]
    row = res["rows"][0]
    area = row[cols.index("area")]
    cx = row[cols.index("centroid_x")]
    cy = row[cols.index("centroid_y")]
    assert area == pytest.approx(400.0)          # 20*20, exact pixel count
    assert cx == pytest.approx(14.5)             # (5+24)/2
    assert cy == pytest.approx(14.5)


def test_circularity_bounded_and_square_less_than_circle():
    mask = np.zeros((40, 40), dtype=np.int32)
    _square(mask, 1, 5, 5, 20)
    res = compute_measurements(mask)
    cols = [k for k, _l, _u in res["columns"]]
    circ = res["rows"][0][cols.index("circularity")]
    # circularity is clamped to <= 1; a square is clearly not a perfect circle
    assert 0.0 < circ <= 1.0
    assert circ < 0.99


def test_two_cells_sorted_by_id():
    mask = np.zeros((60, 60), dtype=np.int32)
    _square(mask, 2, 5, 5, 10)
    _square(mask, 1, 40, 40, 10)
    res = compute_measurements(mask)
    assert res["n_cells"] == 2
    ids = [r[0] for r in res["rows"]]
    assert ids == [1, 2]                          # rows are id-sorted


def test_pixel_size_scales_area_and_length():
    mask = np.zeros((40, 40), dtype=np.int32)
    _square(mask, 1, 5, 5, 20)
    px = 0.5
    res = compute_measurements(mask, pixel_size_um=px)
    cols = [k for k, _l, _u in res["columns"]]
    units = {k: u for k, _l, u in res["columns"]}
    area = res["rows"][0][cols.index("area")]
    diam = res["rows"][0][cols.index("diameter")]
    assert area == pytest.approx(400.0 * px * px)          # 100 µm²
    assert units["area"] == "µm²"
    expected_diam_px = math.sqrt(4.0 * 400.0 / math.pi)
    assert diam == pytest.approx(expected_diam_px * px, rel=1e-3)
    assert units["diameter"] == "µm"


def test_intensity_columns_present_and_correct():
    mask = np.zeros((40, 40), dtype=np.int32)
    img = np.zeros((40, 40), dtype=np.float64)
    _square(mask, 1, 5, 5, 20, value=100.0, img=img)
    res = compute_measurements(mask, intensity_image=img)
    cols = [k for k, _l, _u in res["columns"]]
    assert "mean_intensity" in cols
    row = res["rows"][0]
    assert row[cols.index("mean_intensity")] == pytest.approx(100.0)
    assert row[cols.index("integrated_intensity")] == pytest.approx(100.0 * 400.0)


def test_misaligned_intensity_image_is_ignored_not_crash():
    mask = np.zeros((40, 40), dtype=np.int32)
    _square(mask, 1, 5, 5, 20)
    wrong = np.zeros((10, 10), dtype=np.float64)     # shape mismatch
    res = compute_measurements(mask, intensity_image=wrong)
    cols = [k for k, _l, _u in res["columns"]]
    assert "mean_intensity" not in cols              # intensity dropped, no crash
    assert res["n_cells"] == 1


def test_summary_line_and_csv_roundtrip():
    mask = np.zeros((60, 60), dtype=np.int32)
    _square(mask, 1, 5, 5, 10)
    _square(mask, 2, 30, 30, 10)
    res = compute_measurements(mask)
    line = summary_line(res)
    assert "median" in line.lower() or "Ø" in line
    csv = rows_as_csv(res)
    lines = csv.strip().splitlines()
    assert len(lines) == 1 + res["n_cells"]          # header + one row per cell
    assert "Area" in lines[0]


def test_summary_line_empty():
    res = compute_measurements(np.zeros((16, 16), dtype=np.int32))
    assert summary_line(res) == "No cells detected"
