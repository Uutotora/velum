"""Wiring test: interactive box-prompt segmentation in AnnotateWidget.

SamPredictor already accepts a box prompt natively (see
napari_app.interactive.InteractiveSession) — this exercises the *widget*
side: drawing a rectangle on the box-prompt Shapes layer should extract its
bounding box, call the session with it, and paint the result into the
running Labels layer as a new object, exactly like a point click already
does. `_rect_to_box_xyxy` (the box-coordinate math) is also covered here
rather than in a separate file, since importing anything from
annotate_widget.py — even a plain function — needs PyQt6 importable first.

Uses a fake session (no torch/SamPredictor needed) and a MagicMock viewer
whose add_labels/add_image/add_points/add_shapes build real napari.layers
objects onto a real LayerList — the technique established in
test_predict_labels_display_wiring.py. `_on_ready` is called directly
(bypassing `_start()`, which needs a real SAM model) to set up
post-session-start state, mirroring how test_predict_volume_wiring.py calls
`_show_volume_results` directly instead of the full `_run_prediction` chain.

`_run_box_predict` (like the pre-existing `_run_predict`) hands off to a
background thread and delivers its result via a Qt signal, so tests that
exercise the full draw-a-box round trip poll briefly (`_wait_until`) rather
than asserting immediately after mutating the Shapes layer's data.

Skipped in the lightweight CI image (no PyQt6/napari).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("PyQt6")
aw = pytest.importorskip("napari_app.widgets.annotate_widget")

from PyQt6.QtWidgets import QApplication


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _FakeSession:
    """Stands in for InteractiveSession — no torch/SamPredictor needed."""

    def __init__(self, mask):
        self._mask = mask
        self.calls = []

    def predict(self, coords_xy, labels, mask_input=None, box=None):
        self.calls.append(dict(coords_xy=list(coords_xy), labels=list(labels),
                              mask_input=mask_input, box=box))
        return self._mask, np.zeros((1, 256, 256), dtype=np.float32), 0.95


@pytest.fixture
def widget(app):
    import napari.layers as nl
    from napari.components import LayerList

    viewer = MagicMock()
    layers = LayerList()
    viewer.layers = layers
    viewer.mouse_drag_callbacks = []

    viewer.add_labels.side_effect = (
        lambda data, name=None, opacity=1.0, **k:
            layers.append(nl.Labels(data, name=name, opacity=opacity)) or layers[-1])
    viewer.add_image.side_effect = (
        lambda data, name=None, **k: layers.append(nl.Image(data, name=name)) or layers[-1])
    viewer.add_points.side_effect = (
        lambda data, name=None, **k: layers.append(nl.Points(data, name=name)) or layers[-1])
    viewer.add_shapes.side_effect = (
        lambda data, name=None, **k:
            layers.append(nl.Shapes(data if len(data) else None, name=name)) or layers[-1])

    w = aw.AnnotateWidget(viewer, MagicMock())
    w.show()
    app.processEvents()
    yield w
    w.close()


def _start_session(widget, h=40, w=50):
    """Sets up the post-_start() state directly, bypassing the real SAM
    model load _start() itself does (needs torch + a checkpoint)."""
    widget._img_path = "/tmp/x.png"
    widget._pending_image = np.zeros((h, w, 3), dtype=np.uint8)
    widget._on_ready(True, "ready")


def _rect(y0, x0, y1, x1):
    """A napari Shapes rectangle in data coordinates: Nx2 (row, col)."""
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def _wait_until(cond, timeout=2.0):
    app_ = QApplication.instance()
    t0 = time.time()
    while time.time() - t0 < timeout:
        app_.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


# ── _rect_to_box_xyxy (pure box-coordinate math) ─────────────────────────────

def test_rect_to_box_xyxy_converts_row_col_to_xyxy():
    rect = _rect(5, 10, 20, 30)   # y0=5,x0=10, y1=20,x1=30
    assert aw._rect_to_box_xyxy(rect, h=40, w=50) == [10, 5, 30, 20]


def test_rect_to_box_xyxy_clips_to_image_bounds():
    rect = _rect(-10, -10, 20, 60)
    assert aw._rect_to_box_xyxy(rect, h=40, w=50) == [0, 0, 50, 20]


def test_rect_to_box_xyxy_degenerate_outside_image_returns_none():
    rect = _rect(-10, -10, -5, -5)
    assert aw._rect_to_box_xyxy(rect, h=40, w=50) is None


def test_rect_to_box_xyxy_zero_area_returns_none():
    rect = _rect(5, 5, 5, 5)      # a single point, not a real rectangle
    assert aw._rect_to_box_xyxy(rect, h=40, w=50) is None


# ── session start creates the box-prompt Shapes layer ───────────────────────

def test_start_creates_shapes_layer_pre_armed_for_rectangles(widget):
    _start_session(widget)
    assert widget._shapes_layer is not None
    assert widget._shapes_layer.mode == "add_rectangle"
    assert widget._shapes_layer.name == "x_box_prompt"


def test_labels_layer_stays_active_after_start(widget):
    _start_session(widget)
    assert widget.viewer.layers.selection.active is widget._labels_layer


# ── drawing a box segments a new object ─────────────────────────────────────

def test_drawing_a_box_calls_session_with_the_right_xyxy_box(widget):
    _start_session(widget)
    mask = np.zeros((40, 50), dtype=bool)
    mask[5:15, 5:20] = True
    widget._session = _FakeSession(mask)

    widget._shapes_layer.data = [_rect(5, 5, 15, 20)]
    assert _wait_until(lambda: widget._session.calls)

    call = widget._session.calls[0]
    assert call["box"] == [5, 5, 20, 15]        # XYXY: x0,y0,x1,y1
    assert call["coords_xy"] == [] and call["labels"] == []


def test_drawing_a_box_paints_a_new_object_and_clears_the_layer(widget):
    _start_session(widget)
    mask = np.zeros((40, 50), dtype=bool)
    mask[5:15, 5:20] = True
    widget._session = _FakeSession(mask)

    widget._shapes_layer.data = [_rect(5, 5, 15, 20)]
    assert _wait_until(lambda: widget._label_img.max() > 0)

    assert widget._active_id == 1
    assert np.array_equal(widget._label_img > 0, mask)
    assert np.array_equal(widget._labels_layer.data, widget._label_img)
    assert list(widget._shapes_layer.data) == []    # ready for the next box
    assert widget._last_low is not None             # a follow-up click can refine it


def test_second_box_starts_a_new_object(widget):
    _start_session(widget)
    mask = np.zeros((40, 50), dtype=bool)
    mask[5:15, 5:20] = True
    widget._session = _FakeSession(mask)

    widget._shapes_layer.data = [_rect(5, 5, 15, 20)]
    # _active_id is bumped synchronously as soon as the box is drawn, before
    # the async predict/paint round trip finishes — wait for _busy to clear
    # too, or the second box below can land while still "busy" and get
    # silently ignored by _on_box_drawn's own guard.
    assert _wait_until(lambda: widget._active_id == 1 and not widget._busy)
    widget._shapes_layer.data = [_rect(20, 25, 30, 40)]
    assert _wait_until(lambda: widget._active_id == 2 and not widget._busy)
    assert len(widget._session.calls) == 2


def test_degenerate_box_outside_image_is_ignored(widget):
    _start_session(widget)
    widget._session = _FakeSession(np.zeros((40, 50), dtype=bool))

    widget._shapes_layer.data = [_rect(-10, -10, -5, -5)]
    assert not _wait_until(lambda: bool(widget._session.calls), timeout=0.3)
    assert widget._active_id == 0


def test_clearing_the_shapes_layer_does_not_trigger_a_predict_call(widget, app):
    _start_session(widget)
    widget._session = _FakeSession(np.zeros((40, 50), dtype=bool))
    widget._shapes_layer.data = []   # nothing drawn yet, just an empty reset
    app.processEvents()
    assert widget._session.calls == []


# ── stop/clear cleanup ───────────────────────────────────────────────────────

def test_clear_all_also_clears_the_shapes_layer(widget):
    _start_session(widget)
    widget._shapes_layer.data = [_rect(1, 1, 5, 5)]
    widget._clear_all()
    assert list(widget._shapes_layer.data) == []


def test_stop_prevents_a_later_box_from_segmenting(widget):
    _start_session(widget)
    fake = widget._session = _FakeSession(np.zeros((40, 50), dtype=bool))
    widget._stop()   # sets widget._session back to None
    widget._shapes_layer.data = [_rect(1, 1, 5, 5)]
    assert not _wait_until(lambda: bool(fake.calls), timeout=0.3)


# ── point-click callback steps aside while the Shapes layer is active ──────

def test_point_click_callback_ignores_clicks_while_shapes_layer_active(widget):
    _start_session(widget)
    widget._session = _FakeSession(np.zeros((40, 50), dtype=bool))
    widget.viewer.layers.selection.active = widget._shapes_layer

    class FakeEvent:
        button = 1
        modifiers = ()
        position = (5.0, 5.0)
        type = "mouse_press"

    # `_cb` is a generator function (napari's mouse-callback convention);
    # the guard being tested runs before any `yield`, so a single next()
    # either raises StopIteration (guard fired, nothing else ran) or
    # actually starts prompting — either way _session.calls tells us which.
    gen = widget._cb(widget.viewer, FakeEvent())
    next(gen, None)
    assert widget._session.calls == []
