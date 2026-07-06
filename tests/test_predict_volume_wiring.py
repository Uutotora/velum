"""Wiring test: the z-stack/time-lapse toggle and SAM2 settings card in
PredictWidget.

Exercises real Qt widget construction, visibility, and state — using a
MagicMock in place of a real napari.Viewer. Constructing an actual
napari.Viewer segfaults under this sandbox's offscreen Qt platform (confirmed
separately: the crash happens inside napari.Viewer() itself, before
PredictWidget is ever touched), but PredictWidget's constructor only calls a
handful of Viewer methods (layers, add_image, add_labels, bind_key,
reset_view), all of which a MagicMock satisfies as harmless no-ops — so this
still exercises the real widget, real signals, and real Qt visibility
plumbing, just without a real napari canvas.

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
def widget(app):
    viewer = MagicMock()
    viewer.layers = []
    w = pw.PredictWidget(viewer)
    # Qt's isVisible() is composite (considers ancestors), so an unshown
    # top-level widget reports every child as invisible regardless of its own
    # setVisible() state — show() + processEvents() makes isVisible() checks
    # below meaningful instead of trivially False.
    w.show()
    app.processEvents()
    yield w
    w.close()


def _write_zstack_tiff(tmp_path):
    import tifffile
    path = tmp_path / "zstack.tif"
    tifffile.imwrite(path, np.zeros((4, 16, 16), dtype=np.uint16), metadata={"axes": "ZYX"})
    return path


def _write_plain_png(tmp_path):
    import cv2
    path = tmp_path / "flat.png"
    cv2.imwrite(str(path), np.zeros((16, 16, 3), dtype=np.uint8))
    return path


# ── z-stack checkbox visibility ──────────────────────────────────────────────

def test_zstack_checkbox_hidden_for_plain_image(widget, app, tmp_path):
    widget.image_path.setText(str(_write_plain_png(tmp_path)))
    app.processEvents()
    assert widget.zstack_cb.isVisible() is False


def test_zstack_checkbox_shown_for_multiplane_tiff(widget, app, tmp_path):
    widget.image_path.setText(str(_write_zstack_tiff(tmp_path)))
    app.processEvents()
    assert widget.zstack_cb.isVisible() is True


def test_zstack_checkbox_force_unchecked_when_switching_back_to_plain_image(widget, app, tmp_path):
    widget.image_path.setText(str(_write_zstack_tiff(tmp_path)))
    app.processEvents()
    widget.zstack_cb.setChecked(True)

    widget.image_path.setText(str(_write_plain_png(tmp_path)))
    app.processEvents()
    assert widget.zstack_cb.isChecked() is False
    assert widget.zstack_cb.isVisible() is False


def test_gather_params_zstack_true_only_when_checked_and_visible(widget, app, tmp_path):
    widget.image_path.setText(str(_write_zstack_tiff(tmp_path)))
    app.processEvents()
    widget.zstack_cb.setChecked(True)
    assert widget._gather_params()["zstack"] is True


def test_gather_params_zstack_false_when_hidden_even_if_checked(widget):
    # Defence in depth: _gather_params must not trust isChecked() alone once
    # the toggle is hidden, independent of _refresh_zstack_toggle's own
    # force-uncheck behaviour.
    widget.zstack_cb.setVisible(False)
    widget.zstack_cb.setChecked(True)
    assert widget._gather_params()["zstack"] is False


# ── engine switching: SAM2 settings card + hint text ─────────────────────────

def test_sam2_card_hidden_for_default_engine(widget):
    assert widget._current_engine() == "cellseg1"
    assert widget._sam2_card.isVisible() is False


def test_sam2_card_shown_when_engine_switched_to_sam2(widget, app):
    widget.engine.setCurrentIndex(widget.engine.findData("sam2"))
    app.processEvents()
    assert widget._sam2_card.isVisible() is True
    assert widget._cp_card.isVisible() is False
    assert widget._ckpt_card.isVisible() is False


def test_cp_card_shown_when_engine_switched_to_cellpose(widget, app):
    widget.engine.setCurrentIndex(widget.engine.findData("cellpose"))
    app.processEvents()
    assert widget._cp_card.isVisible() is True
    assert widget._sam2_card.isVisible() is False


def test_sam2_engine_hint_reflects_availability(widget, monkeypatch):
    import napari_app.engines_sam2 as es2

    monkeypatch.setattr(es2, "sam2_available", lambda: False)
    widget.engine.setCurrentIndex(widget.engine.findData("sam2"))
    assert "not installed" in widget._engine_hint.text()

    monkeypatch.setattr(es2, "sam2_available", lambda: True)
    widget._on_engine_changed()
    assert "z-stack" in widget._engine_hint.text()


def test_gather_params_includes_sam2_fields(widget):
    widget.sam2_model_type.setCurrentText("small")
    widget.sam2_checkpoint.setText("/x/y.pt")
    widget.sam2_config_text.setText("configs/custom.yaml")
    params = widget._gather_params()
    assert params["sam2_model_type"] == "small"
    assert params["sam2_checkpoint_text"] == "/x/y.pt"
    assert params["sam2_config_text"] == "configs/custom.yaml"


def test_run_prediction_dispatches_to_volume_path_when_zstack_checked(widget, app, tmp_path, monkeypatch):
    """_run_prediction must call run_volume_prediction_async (not
    run_prediction_async) when the z-stack toggle is on, and never reach the
    2-D on_result path."""
    path = _write_zstack_tiff(tmp_path)
    widget.image_path.setText(str(path))
    app.processEvents()
    widget.zstack_cb.setChecked(True)
    widget.engine.setCurrentIndex(widget.engine.findData("cellpose"))
    app.processEvents()

    calls = []
    monkeypatch.setattr(
        widget._controller, "run_volume_prediction_async",
        lambda config, **kw: calls.append(("volume", config.get("zstack"))))
    monkeypatch.setattr(
        widget._controller, "run_prediction_async",
        lambda config, **kw: calls.append(("2d", config.get("zstack"))))

    widget._run_prediction()
    assert calls == [("volume", True)]
