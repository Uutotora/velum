"""Tests for the Cell Population Analytics view (studio/cell_analytics.py).

Two layers: the pure binning/metric functions run with no Qt at all, and a
handful of offscreen construction tests confirm the panel + dialog build and
switch metrics without crashing on the exact dict shape
``velum_core.analysis.compute_measurements`` returns.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# The analytics view is a Qt widget module (imports PyQt6 at top), so the whole
# test module skips in the light CI dependency-group that has no PyQt6 --
# matching test_workspace.py / test_overlays.py rather than dying at collection.
pytest.importorskip("PyQt6")
ca = pytest.importorskip("studio.cell_analytics")


def _result(n=120):
    """A measurement dict shaped exactly like compute_measurements' output."""
    import random
    random.seed(3)
    cols = [
        ("cell_id", "Cell", "id"),
        ("area", "Area", "µm²"),
        ("diameter", "Eq. diameter", "µm"),
        ("circularity", "Circularity", ""),
        ("aspect_ratio", "Aspect ratio", ""),
        ("centroid_x", "Centroid X", "µm"),
        ("centroid_y", "Centroid Y", "µm"),
    ]
    rows = []
    for i in range(n):
        a = max(20.0, random.gauss(180, 45))
        rows.append([i, a, (4 * a / 3.14159) ** 0.5,
                     min(1.0, random.gauss(0.8, 0.1)),
                     random.gauss(1.4, 0.3),
                     random.random() * 512, random.random() * 512])
    summary = {}
    for idx, (k, _l, _u) in enumerate(cols):
        if k in ("cell_id", "centroid_x", "centroid_y"):
            continue
        vals = sorted(r[idx] for r in rows)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        summary[k] = {"mean": mean, "median": vals[len(vals) // 2],
                      "std": var ** 0.5, "min": vals[0], "max": vals[-1]}
    return {"columns": cols, "rows": rows, "summary": summary,
            "n_cells": n, "pixel_size_um": 0.32}


# ── pure logic ───────────────────────────────────────────────────────────────
def test_plottable_metrics_excludes_id_and_centroids():
    keys = [k for k, _l, _u in ca.plottable_metrics(_result())]
    assert "area" in keys and "circularity" in keys
    assert "cell_id" not in keys
    assert "centroid_x" not in keys and "centroid_y" not in keys


def test_metric_values_reads_the_right_column():
    res = _result(10)
    vals = ca.metric_values(res, "area")
    assert len(vals) == 10
    # column 1 is area in the fixture
    assert vals == [r[1] for r in res["rows"]]


def test_metric_values_unknown_key_is_empty():
    assert ca.metric_values(_result(), "does_not_exist") == []


def test_histogram_counts_sum_to_population():
    edges, counts = ca.histogram([1, 2, 2, 3, 3, 3, 9], n_bins=6)
    assert sum(counts) == 7
    assert len(edges) == len(counts) + 1
    assert edges[0] == 1 and edges[-1] == 9


def test_histogram_empty_is_safe():
    edges, counts = ca.histogram([])
    assert counts == [0]
    assert len(edges) == 2


def test_histogram_all_identical_values_one_centered_bin():
    edges, counts = ca.histogram([5.0, 5.0, 5.0])
    assert counts == [3]
    assert edges[0] < 5.0 < edges[1]


def test_histogram_max_value_lands_in_last_bin():
    edges, counts = ca.histogram([0, 10], n_bins=5)
    # both extremes counted, none dropped by the right-edge boundary
    assert sum(counts) == 2
    assert counts[0] == 1 and counts[-1] == 1


def test_metric_stats_derives_cv_from_summary():
    res = _result()
    s = ca.metric_stats(res, "area")
    assert s["cv"] == pytest.approx(s["std"] / abs(s["mean"]) * 100.0)


def test_metric_stats_recomputes_when_summary_missing():
    res = _result(20)
    res["summary"] = {}  # force the raw-rows fallback path
    s = ca.metric_stats(res, "area")
    vals = ca.metric_values(res, "area")
    assert s["mean"] == pytest.approx(sum(vals) / len(vals))
    assert s["min"] == min(vals) and s["max"] == max(vals)


# ── headless construction ────────────────────────────────────────────────────
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402
from studio import theme  # noqa: E402


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def parent(app):
    w = QWidget()
    w.resize(1200, 900)
    w.show()
    return w


def test_panel_builds_and_defaults_to_a_featured_metric(app):
    panel = ca.CellAnalyticsPanel(_result(), theme.DARK, image_name="img.tif")
    assert panel._metric_key == "area"  # first featured key present
    assert panel.findChildren(ca.Histogram)


def test_panel_switch_metric_updates_selection_and_rerenders(app):
    panel = ca.CellAnalyticsPanel(_result(), theme.DARK)
    panel._select_metric("circularity")
    assert panel._metric_key == "circularity"
    assert panel._selectbox.text() == "Circularity"


def test_panel_export_and_close_callbacks_fire(app):
    calls = {"export": 0, "close": 0}
    panel = ca.CellAnalyticsPanel(
        _result(), theme.DARK,
        on_export=lambda: calls.__setitem__("export", calls["export"] + 1),
        on_close=lambda: calls.__setitem__("close", calls["close"] + 1))
    panel._export()
    panel._close()
    assert calls == {"export": 1, "close": 1}


def test_panel_handles_zero_cells_without_crashing(app):
    empty = {"columns": [("cell_id", "Cell", "id"), ("area", "Area", "µm²")],
             "rows": [], "summary": {}, "n_cells": 0, "pixel_size_um": 0.0}
    panel = ca.CellAnalyticsPanel(empty, theme.DARK)
    assert panel._metric_key in ("area", "")


def test_dialog_opens_and_disposes(app, parent):
    dlg = ca.CellAnalyticsDialog(parent, theme.DARK, _result(), image_name="img.tif")
    dlg.open()
    assert dlg.isVisible()
    dlg.hide()  # self-disposing hideEvent must not raise


def test_dialog_light_theme_builds(app, parent):
    dlg = ca.CellAnalyticsDialog(parent, theme.LIGHT, _result())
    dlg.open()
    assert dlg.findChildren(ca.CellAnalyticsPanel)
