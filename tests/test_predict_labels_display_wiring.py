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
        extra = {k: v for k, v in kwargs.items() if k in ("scale", "units")}
        layer = nl.Labels(data, name=name, opacity=opacity, **extra)
        layers.append(layer)
        return layer

    def fake_add_image(data, name=None, **kwargs):
        extra = {k: v for k, v in kwargs.items() if k in ("scale", "units", "rgb")}
        layer = nl.Image(data, name=name, **extra)
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
    # Qt's isVisible() is composite (considers ancestors), so an unshown
    # top-level widget reports every child as invisible regardless of its
    # own setVisible() state — needed for the colour-by-measurement card's
    # visibility assertions below.
    w.show()
    app.processEvents()
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


# ── Colour cells by a measurement ─────────────────────────────────────────────
# Exercised via _show_volume_results rather than _show_results: the `widget`
# fixture stubs _recompute_measurements (the 2-D path's route to
# _refresh_color_by_options) to dodge the measurements-window singleton
# fragility documented above, but _show_volume_results calls
# _refresh_color_by_options directly and isn't affected either way.

def _mask_two_different_areas(h=30, w=30):
    mask = np.zeros((h, w), dtype=np.int32)
    mask[5:15, 5:15] = 1      # 10x10 = 100
    mask[18:25, 18:26] = 2    # 7x8 = 56
    return mask


def _volume_two_cells():
    mask = _mask_two_different_areas()
    img_vol = np.zeros((2, 30, 30, 3), dtype=np.uint8)
    mask_vol = np.stack([mask, mask], axis=0)
    return img_vol, mask_vol


def test_color_by_populates_dropdown_from_result_columns(widget, app):
    img_vol, mask_vol = _volume_two_cells()
    widget._show_volume_results(img_vol, mask_vol)
    app.processEvents()

    items = [widget.color_by.itemText(i) for i in range(widget.color_by.count())]
    keys = [widget.color_by.itemData(i) for i in range(widget.color_by.count())]
    assert items[0] == "Instance ID (default)"
    assert "Volume (px³)" in items          # the 3-D schema's counterpart to "area"
    assert "Centroid X (px)" in items
    assert "cell_id" not in keys            # never offered as a colour-by choice
    assert widget._color_card.isVisible() is True


def test_color_by_disabled_when_no_cells(widget, app):
    img_vol = np.zeros((2, 20, 20, 3), dtype=np.uint8)
    mask_vol = np.zeros((2, 20, 20), dtype=np.int32)
    widget._show_volume_results(img_vol, mask_vol)
    app.processEvents()
    # The "Display" card itself stays visible with no cells (the scale-bar
    # and view-in-3D toggles alongside it are still meaningful) — only
    # "Colour cells by" is disabled, since there's nothing to colour by.
    assert widget._color_card.isVisible() is True
    assert widget.color_by.isEnabled() is False


def test_color_by_volume_recolours_fill_and_outline_matching(widget, app):
    img_vol, mask_vol = _volume_two_cells()
    widget._show_volume_results(img_vol, mask_vol)

    idx = widget.color_by.findData("volume")
    assert idx >= 0
    widget.color_by.setCurrentIndex(idx)

    fill = widget.viewer.layers["sample_masks_fill"]
    outline = widget.viewer.layers["sample_masks"]
    assert tuple(fill.get_color(1)) == tuple(outline.get_color(1))
    assert tuple(fill.get_color(2)) == tuple(outline.get_color(2))
    assert tuple(fill.get_color(1)) != tuple(fill.get_color(2))   # different volumes


def test_color_by_instance_id_restores_default_colours(widget, app):
    img_vol, mask_vol = _volume_two_cells()
    widget._show_volume_results(img_vol, mask_vol)

    outline = widget.viewer.layers["sample_masks"]
    default_1, default_2 = tuple(outline.get_color(1)), tuple(outline.get_color(2))

    widget.color_by.setCurrentIndex(widget.color_by.findData("volume"))
    assert (tuple(outline.get_color(1)), tuple(outline.get_color(2))) != (default_1, default_2)

    widget.color_by.setCurrentIndex(widget.color_by.findData("instance_id"))
    assert tuple(outline.get_color(1)) == default_1
    assert tuple(outline.get_color(2)) == default_2


def test_apply_color_by_with_no_result_layers_is_a_noop(widget):
    # No prediction has been run yet -> no *_masks/*_masks_fill layers exist.
    widget._apply_color_by("area")   # must not raise


def test_color_by_shows_a_legend_with_the_metric_range(widget):
    img_vol, mask_vol = _volume_two_cells()
    widget._show_volume_results(img_vol, mask_vol)
    assert widget._color_legend.isVisible() is False   # default is instance_id -> no legend

    widget.color_by.setCurrentIndex(widget.color_by.findData("volume"))
    assert widget._color_legend.isVisible() is True
    assert widget._color_legend._lo.text() != ""
    assert widget._color_legend._hi.text() != ""

    widget.color_by.setCurrentIndex(widget.color_by.findData("instance_id"))
    assert widget._color_legend.isVisible() is False


# ── real-world µm calibration (scale bar + layer scale/units) ───────────────

def test_layer_scale_kwargs_empty_when_uncalibrated(widget):
    assert widget.pixel_size.value() == 0.0   # the documented "off" sentinel
    assert widget._layer_scale_kwargs(2) == {}
    assert widget._layer_scale_kwargs(3) == {}


def test_layer_scale_kwargs_2d(widget):
    widget.pixel_size.setValue(0.25)
    kw = widget._layer_scale_kwargs(2)
    assert kw["scale"] == (0.25, 0.25)
    assert kw["units"] == ("um", "um")


def test_layer_scale_kwargs_volume_leaves_leading_axis_unscaled(widget):
    widget.pixel_size.setValue(0.5)
    kw = widget._layer_scale_kwargs(3)
    assert kw["scale"] == (1.0, 0.5, 0.5)
    assert kw["units"] == ("pixel", "um", "um")


def test_is_rgb_like_true_even_for_a_small_image(widget):
    # napari's own auto-guess needs both non-channel axes > 30px; a small
    # image should still be recognised as RGB via the explicit shape check.
    small_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    assert widget._is_rgb_like(small_rgb, 2) is True


def test_is_rgb_like_false_for_grayscale():
    gray = np.zeros((100, 100), dtype=np.uint8)
    assert pw.PredictWidget._is_rgb_like(gray, 2) is False


def test_show_results_calibrated_image_and_labels_scale_stay_aligned(widget):
    """The image layer and every Labels layer for one result must share the
    exact same scale — a mismatch here would visually misalign the mask
    overlay from the image it was predicted on."""
    widget.pixel_size.setValue(0.25)
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    widget._show_results(img, _mask_2_cells())

    img_layer = widget.viewer.layers["sample_image"]
    fill_layer = widget.viewer.layers["sample_masks_fill"]
    outline_layer = widget.viewer.layers["sample_masks"]
    assert tuple(img_layer.scale) == tuple(fill_layer.scale) == tuple(outline_layer.scale)
    assert tuple(img_layer.scale) == (0.25, 0.25)


def test_scale_bar_toggle_sets_viewer_visibility(widget):
    widget.scale_bar_cb.setChecked(True)
    assert widget.viewer.scale_bar.visible is True
    widget.scale_bar_cb.setChecked(False)
    assert widget.viewer.scale_bar.visible is False


# ── view-in-3D toggle (volume results only) ─────────────────────────────────

def test_view3d_toggle_hidden_for_2d_shown_for_volume(widget):
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    widget._show_results(img, _mask_2_cells())
    assert widget.view3d_cb.isVisible() is False

    img_vol, mask_vol = _volume_two_cells()
    widget._show_volume_results(img_vol, mask_vol)
    assert widget.view3d_cb.isVisible() is True


def test_view3d_toggle_sets_ndisplay(widget):
    img_vol, mask_vol = _volume_two_cells()
    widget._show_volume_results(img_vol, mask_vol)
    widget.view3d_cb.setChecked(True)
    assert widget.viewer.dims.ndisplay == 3
    widget.view3d_cb.setChecked(False)
    assert widget.viewer.dims.ndisplay == 2


def test_a_new_2d_result_resets_view3d_off(widget):
    img_vol, mask_vol = _volume_two_cells()
    widget._show_volume_results(img_vol, mask_vol)
    widget.view3d_cb.setChecked(True)

    img = np.zeros((30, 30, 3), dtype=np.uint8)
    widget._show_results(img, _mask_2_cells())
    assert widget.view3d_cb.isChecked() is False
    assert widget.viewer.dims.ndisplay == 2


# ── linked selection: Measurements row -> highlighted cell in the viewer ───

def test_measurement_row_selected_highlights_both_layers(widget):
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    widget._show_results(img, _mask_2_cells())
    from napari_app import analysis
    widget._last_measure = analysis.compute_measurements(_mask_2_cells())

    widget._on_measurement_row_selected(2)
    fill = widget.viewer.layers["sample_masks_fill"]
    outline = widget.viewer.layers["sample_masks"]
    assert fill.show_selected_label is True and fill.selected_label == 2
    assert outline.show_selected_label is True and outline.selected_label == 2


def test_measurement_row_selected_minus_one_clears_highlight(widget):
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    widget._show_results(img, _mask_2_cells())
    widget._on_measurement_row_selected(1)
    widget._on_measurement_row_selected(-1)
    outline = widget.viewer.layers["sample_masks"]
    assert outline.show_selected_label is False


def test_measurement_row_selected_with_no_layers_is_a_noop(widget):
    widget._on_measurement_row_selected(1)   # nothing predicted yet -> must not raise


def test_open_measurements_wires_row_selected_exactly_once(widget, monkeypatch):
    """A throwaway MeasurementsWindow, monkeypatched in as
    get_measurements_window()'s return value, rather than the real
    process-wide singleton — connecting/emitting on the real one from a
    test turned out to corrupt it for whichever test runs next (the same
    "wrapped C/C++ object ... has been deleted" class of shared-Qt-singleton
    fragility the `widget` fixture above already works around for
    _recompute_measurements; _open_measurements reaches the same singleton
    by a path that fixture's stub doesn't cover).
    """
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    widget._show_results(img, _mask_2_cells())
    from napari_app import analysis
    widget._last_measure = analysis.compute_measurements(_mask_2_cells())

    from napari_app.widgets import measurements_window as mw_mod
    fresh = mw_mod.MeasurementsWindow()
    monkeypatch.setattr(mw_mod, "get_measurements_window", lambda: fresh)
    try:
        calls = []
        widget._on_measurement_row_selected = calls.append   # spy, replaces the bound method
        widget._open_measurements()
        widget._open_measurements()   # a second click must not double-connect
        fresh.row_selected.emit(2)
        assert calls == [2]   # not [2, 2] -- exactly one connection, not two
    finally:
        fresh.close()
