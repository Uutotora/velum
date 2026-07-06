"""Wiring test: _add_filled_labels (fill + outline Labels-layer overlay).

napari's Labels layer can't show a translucent fill and a contour at once in
one layer (`contour` is a 0/N toggle, not additive) — _add_filled_labels adds
two stacked layers instead (a low-opacity filled one, a contour=1 one on
top), the standard way to get the "coloured fill + border, same colour" look
(QuPath's "Fill detections", CellProfiler's OverlayOutlines). This exercises
real napari.layers.Labels/Image construction: the MagicMock viewer's
add_labels/add_image are given a side_effect that builds real layer objects
and appends them to a real LayerList, so contour/opacity/get_color() are
observed for real, not just a recorded mock call. A real napari.Viewer()
segfaults in this sandbox's offscreen platform (see
test_predict_volume_wiring.py) — Labels/Image layer *construction* alone
doesn't need one.

Skipped in the lightweight CI image (no PyQt6/napari).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pw = pytest.importorskip("napari_app.widgets.predict_widget")

from PyQt6.QtWidgets import QApplication


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widget(app, monkeypatch):
    import napari.layers as nl
    from napari.components import LayerList

    viewer = MagicMock()
    layers = LayerList()
    viewer.layers = layers

    def fake_add_labels(data, name=None, opacity=1.0, **kwargs):
        layer = nl.Labels(data, name=name, opacity=opacity)
        layers.append(layer)
        return layer

    def fake_add_image(data, name=None, **kwargs):
        layer = nl.Image(data, name=name)
        layers.append(layer)
        return layer

    viewer.add_labels.side_effect = fake_add_labels
    viewer.add_image.side_effect = fake_add_image

    w = pw.PredictWidget(viewer)
    w.image_path.setText("/tmp/sample.png")
    # This file is about layer display, not logging or the measurements
    # window — both are module-level Qt singletons shared across every
    # PredictWidget instance in this process (get_log_window()/
    # get_measurements_window()), and repeatedly constructing/closing
    # PredictWidget (one per test) is a test-only pattern the real app
    # never does, which can tear down that shared singleton's underlying
    # Qt object between tests ("wrapped C/C++ object ... has been
    # deleted" — see test_predict_volume_wiring.py for the same issue
    # with the log window). Stubbed out here rather than touched in
    # production code.
    monkeypatch.setattr(w, "_append_log", lambda *a, **k: None)
    monkeypatch.setattr(w, "_recompute_measurements", lambda: None)
    yield w
    w.close()


def _mask_2_cells(h=30, w=30):
    mask = np.zeros((h, w), dtype=np.int32)
    mask[5:15, 5:15] = 1
    mask[18:25, 18:25] = 2
    return mask


# ── _show_results (2-D predict) ──────────────────────────────────────────────

def test_show_results_adds_fill_and_outline_layers(widget):
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    widget._show_results(img, _mask_2_cells())

    names = [l.name for l in widget.viewer.layers]
    assert names == ["sample_image", "sample_masks_fill", "sample_masks"]

    fill = widget.viewer.layers["sample_masks_fill"]
    outline = widget.viewer.layers["sample_masks"]
    assert fill.contour == 0                       # filled wash
    assert outline.contour == 1                    # crisp border, on top
    assert fill.opacity == pytest.approx(0.35)
    assert outline.opacity == pytest.approx(0.7)


def test_show_results_fill_and_outline_share_colour_per_cell(widget):
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    widget._show_results(img, _mask_2_cells())

    fill = widget.viewer.layers["sample_masks_fill"]
    outline = widget.viewer.layers["sample_masks"]
    assert tuple(fill.get_color(1)) == tuple(outline.get_color(1))
    assert tuple(fill.get_color(2)) == tuple(outline.get_color(2))
    assert tuple(outline.get_color(1)) != tuple(outline.get_color(2))  # still per-instance


def test_show_results_rerun_does_not_accumulate_layers(widget):
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    mask = _mask_2_cells()
    widget._show_results(img, mask)
    widget._show_results(img, mask)
    names = [l.name for l in widget.viewer.layers]
    assert names == ["sample_image", "sample_masks_fill", "sample_masks"]


def test_show_results_no_cells_adds_no_labels_layers(widget):
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    widget._show_results(img, np.zeros((30, 30), dtype=np.int32))
    names = [l.name for l in widget.viewer.layers]
    assert names == ["sample_image"]


# ── _show_volume_results (z-stack predict) ───────────────────────────────────

def test_show_volume_results_adds_fill_and_outline_layers(widget):
    img_vol = np.zeros((3, 30, 30, 3), dtype=np.uint8)
    mask_vol = np.zeros((3, 30, 30), dtype=np.int32)
    mask_vol[:, 5:15, 5:15] = 1

    widget._show_volume_results(img_vol, mask_vol)

    fill = widget.viewer.layers["sample_masks_fill"]
    outline = widget.viewer.layers["sample_masks"]
    assert fill.contour == 0
    assert outline.contour == 1
    assert tuple(fill.get_color(1)) == tuple(outline.get_color(1))


# ── _show_ground_truth ────────────────────────────────────────────────────────

def test_show_ground_truth_adds_fill_and_outline_uniform_green(widget, tmp_path):
    import cv2
    gt = np.zeros((20, 20), dtype=np.uint16)
    gt[2:8, 2:8] = 1
    gt[10:15, 10:15] = 5
    path = tmp_path / "gt.png"
    cv2.imwrite(str(path), gt)
    widget.gt_path.setText(str(path))

    widget._show_ground_truth()

    fill = widget.viewer.layers["sample_gt_fill"]
    outline = widget.viewer.layers["sample_gt"]
    assert fill.contour == 0
    assert outline.contour == 1
    green = (0.0, 1.0, 0.35, 1.0)
    assert tuple(fill.get_color(1)) == green
    assert tuple(fill.get_color(5)) == green            # uniform, not per-instance
    assert tuple(outline.get_color(1)) == green


def test_show_ground_truth_reload_does_not_accumulate_layers(widget, tmp_path):
    import cv2
    gt = np.zeros((20, 20), dtype=np.uint16)
    gt[2:8, 2:8] = 1
    path = tmp_path / "gt.png"
    cv2.imwrite(str(path), gt)
    widget.gt_path.setText(str(path))

    widget._show_ground_truth()
    widget._show_ground_truth()
    names = [l.name for l in widget.viewer.layers]
    assert names == ["sample_gt_fill", "sample_gt"]
