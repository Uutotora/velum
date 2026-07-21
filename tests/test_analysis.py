"""Unit tests for velum_core.analysis (per-cell morphometry).

These are regression guards: they pin the numeric contract of
compute_measurements so refactors of the measurement pipeline cannot
silently change what scientists read off the table.
"""
import math

import numpy as np
import pytest

from velum_core.analysis import (
    compute_measurements,
    label_colormap_from_measurement,
    measurement_range,
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


# ── 3-D (z-stack) measurements ───────────────────────────────────────────────

def _cuboid(canvas, label, z0, r0, c0, dz, side, img=None, value=None):
    canvas[z0:z0 + dz, r0:r0 + side, c0:c0 + side] = label
    if img is not None and value is not None:
        img[z0:z0 + dz, r0:r0 + side, c0:c0 + side] = value


def test_empty_3d_mask_returns_no_cells():
    mask = np.zeros((4, 32, 32), dtype=np.int32)
    res = compute_measurements(mask)
    assert res["n_cells"] == 0
    assert res["rows"] == []
    assert res["summary"] == {}
    assert any(k == "volume" for k, _l, _u in res["columns"])


def test_single_cuboid_volume_and_centroid_exact():
    mask = np.zeros((10, 20, 20), dtype=np.int32)
    _cuboid(mask, 1, z0=2, r0=5, c0=5, dz=6, side=10)   # z 2..7, r/c 5..14
    res = compute_measurements(mask)
    assert res["n_cells"] == 1
    cols = [k for k, _l, _u in res["columns"]]
    row = res["rows"][0]
    assert row[cols.index("volume")] == pytest.approx(600.0)   # 6*10*10 voxels
    assert row[cols.index("centroid_z")] == pytest.approx(4.5)  # (2+7)/2
    assert row[cols.index("centroid_y")] == pytest.approx(9.5)  # (5+14)/2
    assert row[cols.index("centroid_x")] == pytest.approx(9.5)


def test_3d_equivalent_diameter_matches_sphere_formula():
    mask = np.zeros((10, 20, 20), dtype=np.int32)
    _cuboid(mask, 1, z0=2, r0=5, c0=5, dz=6, side=10)
    res = compute_measurements(mask)
    cols = [k for k, _l, _u in res["columns"]]
    row = res["rows"][0]
    volume = row[cols.index("volume")]
    expected = (6.0 * volume / math.pi) ** (1.0 / 3.0)
    assert row[cols.index("diameter")] == pytest.approx(expected, rel=1e-3)


def test_3d_pixel_size_scales_volume_and_length():
    mask = np.zeros((10, 20, 20), dtype=np.int32)
    _cuboid(mask, 1, z0=0, r0=0, c0=0, dz=4, side=10)   # 400 voxels
    res_px = compute_measurements(mask)
    res_um = compute_measurements(mask, pixel_size_um=0.5)
    cols = [k for k, _l, _u in res_um["columns"]]
    vol_px = res_px["rows"][0][cols.index("volume")]
    vol_um = res_um["rows"][0][cols.index("volume")]
    assert vol_um == pytest.approx(vol_px * 0.5 ** 3, rel=1e-6)
    vu = next(u for k, _l, u in res_um["columns"] if k == "volume")
    lu = next(u for k, _l, u in res_um["columns"] if k == "diameter")
    assert vu == "µm³"
    assert lu == "µm"
    vu_px = next(u for k, _l, u in res_px["columns"] if k == "volume")
    assert vu_px == "px³"


def test_3d_schema_excludes_2d_only_columns():
    mask = np.zeros((6, 16, 16), dtype=np.int32)
    _cuboid(mask, 1, 0, 0, 0, 4, 8)
    res = compute_measurements(mask)
    cols = {k for k, _l, _u in res["columns"]}
    for absent in ("area", "perimeter", "circularity", "eccentricity", "orientation"):
        assert absent not in cols
    for present in ("volume", "diameter", "major_axis", "minor_axis",
                   "solidity", "extent", "centroid_x", "centroid_y", "centroid_z"):
        assert present in cols


def test_3d_two_separate_cuboids_each_measured():
    mask = np.zeros((10, 30, 30), dtype=np.int32)
    _cuboid(mask, 1, 0, 0, 0, 4, 8)
    _cuboid(mask, 2, 5, 15, 15, 4, 8)
    res = compute_measurements(mask)
    assert res["n_cells"] == 2
    assert [r[0] for r in res["rows"]] == [1, 2]        # sorted by cell_id


def test_3d_solidity_and_extent_near_one_for_a_filled_cuboid():
    mask = np.zeros((10, 20, 20), dtype=np.int32)
    _cuboid(mask, 1, 2, 5, 5, 6, 10)
    res = compute_measurements(mask)
    cols = [k for k, _l, _u in res["columns"]]
    row = res["rows"][0]
    assert row[cols.index("solidity")] == pytest.approx(1.0, abs=1e-6)
    assert row[cols.index("extent")] == pytest.approx(1.0, abs=1e-6)


def test_3d_intensity_columns_with_grayscale_volume():
    mask = np.zeros((6, 16, 16), dtype=np.int32)
    img = np.zeros((6, 16, 16), dtype=np.float64)
    _cuboid(mask, 1, 1, 2, 2, 3, 6, img=img, value=50.0)
    res = compute_measurements(mask, intensity_image=img)
    cols = [k for k, _l, _u in res["columns"]]
    row = res["rows"][0]
    assert row[cols.index("mean_intensity")] == pytest.approx(50.0)
    assert row[cols.index("max_intensity")] == pytest.approx(50.0)
    assert row[cols.index("min_intensity")] == pytest.approx(50.0)


def test_3d_intensity_columns_with_color_volume():
    mask = np.zeros((6, 16, 16), dtype=np.int32)
    img = np.zeros((6, 16, 16, 3), dtype=np.float64)
    _cuboid(mask, 1, 1, 2, 2, 3, 6, img=img, value=90.0)
    res = compute_measurements(mask, intensity_image=img)
    cols = [k for k, _l, _u in res["columns"]]
    assert "mean_intensity" in cols
    row = res["rows"][0]
    assert row[cols.index("mean_intensity")] > 0.0      # luminosity-weighted, non-zero


def test_3d_intensity_misaligned_shape_ignored():
    mask = np.zeros((6, 16, 16), dtype=np.int32)
    _cuboid(mask, 1, 1, 2, 2, 3, 6)
    bad_img = np.zeros((6, 8, 8), dtype=np.float64)     # wrong H/W
    res = compute_measurements(mask, intensity_image=bad_img)
    cols = [k for k, _l, _u in res["columns"]]
    assert "mean_intensity" not in cols                 # ignored, no crash
    assert res["n_cells"] == 1


def test_summary_line_3d_result():
    mask = np.zeros((6, 16, 16), dtype=np.int32)
    _cuboid(mask, 1, 1, 2, 2, 3, 6)
    res = compute_measurements(mask)
    line = summary_line(res)
    assert "volume" in line.lower()
    assert "circularity" not in line.lower()


def test_rows_as_csv_3d_result():
    mask = np.zeros((6, 16, 16), dtype=np.int32)
    _cuboid(mask, 1, 1, 2, 2, 3, 6)
    res = compute_measurements(mask)
    csv = rows_as_csv(res)
    lines = csv.strip().splitlines()
    assert len(lines) == 1 + res["n_cells"]
    assert "Volume" in lines[0]


# ── label_colormap_from_measurement (colour cells by a measurement) ─────────

def _two_cell_result():
    mask = np.zeros((40, 40), dtype=np.int32)
    _square(mask, 1, 2, 2, 6)     # area 36, the smaller cell
    _square(mask, 2, 20, 20, 15)  # area 225, the bigger cell
    return compute_measurements(mask)


def test_label_colormap_orders_low_to_high_through_the_real_colormap():
    res = _two_cell_result()
    cmap_out = label_colormap_from_measurement(res, "area")
    assert set(cmap_out) == {1, 2}

    from matplotlib import colormaps
    viridis = colormaps["viridis"]
    assert cmap_out[1] == pytest.approx(tuple(viridis(0.0)))   # smallest -> low end
    assert cmap_out[2] == pytest.approx(tuple(viridis(1.0)))   # largest -> high end


def test_label_colormap_keys_are_cell_ids_not_row_index():
    mask = np.zeros((20, 20), dtype=np.int32)
    _square(mask, 5, 2, 2, 4)   # a non-consecutive label id
    res = compute_measurements(mask)
    cmap_out = label_colormap_from_measurement(res, "area")
    assert set(cmap_out) == {5}


def test_label_colormap_missing_key_returns_empty():
    res = _two_cell_result()
    assert label_colormap_from_measurement(res, "not_a_real_column") == {}


def test_label_colormap_cell_id_column_returns_empty():
    # Colouring "by instance id" is the default random-colour path already —
    # not a measurement, so it's explicitly excluded here.
    res = _two_cell_result()
    assert label_colormap_from_measurement(res, "cell_id") == {}


def test_label_colormap_empty_result_returns_empty():
    res = compute_measurements(np.zeros((10, 10), dtype=np.int32))
    assert label_colormap_from_measurement(res, "area") == {}


def test_label_colormap_no_spread_uses_middle_of_colormap():
    mask = np.zeros((20, 20), dtype=np.int32)
    _square(mask, 1, 1, 1, 4)   # two identically-sized cells -> no spread
    _square(mask, 2, 10, 10, 4)
    res = compute_measurements(mask)
    cmap_out = label_colormap_from_measurement(res, "area")

    from matplotlib import colormaps
    mid = tuple(colormaps["viridis"](0.5))
    assert cmap_out[1] == pytest.approx(mid)
    assert cmap_out[2] == pytest.approx(mid)


def test_label_colormap_respects_cmap_name():
    res = _two_cell_result()
    cmap_out = label_colormap_from_measurement(res, "area", cmap_name="plasma")
    from matplotlib import colormaps
    plasma = colormaps["plasma"]
    assert cmap_out[1] == pytest.approx(tuple(plasma(0.0)))
    assert cmap_out[2] == pytest.approx(tuple(plasma(1.0)))


# ── measurement_range (the legend's min/max for "colour cells by") ─────────

def test_measurement_range_returns_min_max_across_population():
    res = _two_cell_result()
    assert measurement_range(res, "area") == (36.0, 225.0)


def test_measurement_range_missing_key_returns_none():
    res = _two_cell_result()
    assert measurement_range(res, "not_a_real_column") is None


def test_measurement_range_cell_id_returns_none():
    res = _two_cell_result()
    assert measurement_range(res, "cell_id") is None


def test_measurement_range_empty_result_returns_none():
    res = compute_measurements(np.zeros((10, 10), dtype=np.int32))
    assert measurement_range(res, "area") is None
