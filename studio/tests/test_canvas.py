"""Tests for the Segment workspace's own canvas (studio/canvas.py).

Offscreen Qt, no napari/torch. Mouse/wheel interaction is driven with real,
directly-constructed QMouseEvent/QWheelEvent objects passed straight to our
own Python event-handler overrides (never ``None`` into a *native* handler —
see the studio-subproject memory on why that specific pattern segfaults;
these handlers are plain Python we wrote, so a real event object is all a
normal call needs).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
canvas_mod = pytest.importorskip("studio.canvas")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication

from studio import theme
from studio.canvas import Canvas, _blend, _contour_mask, _interpolate, _render_image, _render_labels
from studio.layer_model import (
    ERASE, FILL, ImageLayer, LabelsLayer, LayerList, PAINT, PICK, POLYGON, PointsLayer,
)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _make_canvas(app, h=40, w=40, with_labels=True):
    layers = LayerList()
    layers.add(ImageLayer("Image", np.zeros((h, w, 3), dtype=np.uint8)))
    if with_labels:
        layers.add(LabelsLayer("Segmentation", np.zeros((h, w), dtype=np.int32)))
    statuses: list[str] = []
    picks: list[int] = []
    c = Canvas(theme.DARK, layers, on_status=statuses.append, on_label_picked=picks.append)
    c.resize(w, h)
    # Bypass home()'s fit-to-view maths so widget coords == image coords 1:1,
    # which keeps the interaction tests' expected pixels simple and exact.
    c._zoom = 1.0
    c._pan = QPointF(0.0, 0.0)
    c._fitted = True
    return c, layers, statuses, picks


def _press(widget, pos: QPoint, button=Qt.MouseButton.LeftButton):
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(pos), button, button,
                     Qt.KeyboardModifier.NoModifier)
    widget.mousePressEvent(ev)


def _move(widget, pos: QPoint, buttons=Qt.MouseButton.LeftButton):
    ev = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(pos), Qt.MouseButton.NoButton, buttons,
                     Qt.KeyboardModifier.NoModifier)
    widget.mouseMoveEvent(ev)


def _release(widget, pos: QPoint, button=Qt.MouseButton.LeftButton):
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(pos), button,
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    widget.mouseReleaseEvent(ev)


def _double_click(widget, pos: QPoint, button=Qt.MouseButton.LeftButton):
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonDblClick, QPointF(pos), button, button,
                     Qt.KeyboardModifier.NoModifier)
    widget.mouseDoubleClickEvent(ev)


# ── geometry ─────────────────────────────────────────────────────────────────
def test_base_shape_and_composite_size(app):
    c, layers, *_ = _make_canvas(app, h=30, w=50)
    assert c._base_shape() == (30, 50)
    image = c._composited_image()
    assert image is not None
    assert (image.width(), image.height()) == (50, 30)


def test_base_shape_none_without_layers(app):
    c, layers, *_ = _make_canvas(app, with_labels=False)
    layers.clear()
    assert c._base_shape() is None
    assert c._composited_image() is None


@pytest.mark.parametrize("transposed", [False, True])
def test_widget_image_roundtrip(app, transposed):
    c, *_ = _make_canvas(app)
    c.transposed = transposed
    c._zoom = 1.7
    c._pan = QPointF(3.0, -5.0)
    row, col = 12.0, 7.0
    pt = c.image_to_widget(row, col)
    col2, row2 = c.widget_to_image(pt)
    assert col2 == pytest.approx(col)
    assert row2 == pytest.approx(row)


def test_home_fits_and_centres_image(app):
    c, *_ = _make_canvas(app, h=40, w=40)
    c._fitted = False
    c.resize(400, 300)  # comfortably above Canvas's own setMinimumSize(200, 160)
    c.home()
    assert c._zoom > 0
    # the fitted image's centre should land near the widget's actual centre
    centre = c.image_to_widget(20, 20)
    assert centre.x() == pytest.approx(c.width() / 2, abs=2)
    assert centre.y() == pytest.approx(c.height() / 2, abs=2)


# ── mode + edit target ───────────────────────────────────────────────────────
def test_set_mode_mirrors_into_labels_layer(app):
    c, layers, *_ = _make_canvas(app)
    c.set_mode(PAINT)
    assert c.mode == PAINT
    assert layers[1].mode == PAINT


def test_edit_target_falls_back_to_first_labels_layer(app):
    c, layers, *_ = _make_canvas(app)
    layers.select(0)  # select the Image layer, not the Labels one
    target = c.edit_target()
    assert target is layers[1]


def test_edit_target_none_without_any_labels_layer(app):
    c, layers, *_ = _make_canvas(app, with_labels=False)
    assert c.edit_target() is None


# ── real mouse-driven edits ───────────────────────────────────────────────────
def test_paint_via_mouse_drag(app):
    c, layers, *_ = _make_canvas(app)
    c.set_mode(PAINT)
    _press(c, QPoint(20, 20))
    _move(c, QPoint(24, 20))
    _release(c, QPoint(24, 20))
    labels = layers[1]
    assert labels.data[20, 20] == 1  # default selected_label
    assert labels.data[20, 24] == 1  # interpolated stroke reached the end point
    assert labels.max_label == 1


def test_erase_via_mouse_click(app):
    c, layers, *_ = _make_canvas(app)
    layers[1].data[15:25, 15:25] = 5
    c.set_mode(ERASE)
    _press(c, QPoint(20, 20))
    _release(c, QPoint(20, 20))
    assert layers[1].data[20, 20] == 0


def test_fill_via_mouse_click(app):
    c, layers, *_ = _make_canvas(app)
    layers[1].data[10:20, 10:20] = 2
    layers[1].selected_label = 9
    c.set_mode(FILL)
    _press(c, QPoint(15, 15))
    assert (layers[1].data[10:20, 10:20] == 9).all()


def test_pick_via_mouse_click_invokes_callback(app):
    c, layers, _statuses, picks = _make_canvas(app)
    layers[1].data[5, 5] = 77
    c.set_mode(PICK)
    _press(c, QPoint(5, 5))
    assert picks == [77]
    assert layers[1].selected_label == 77


def test_polygon_mode_press_and_double_click_closes(app):
    c, layers, *_ = _make_canvas(app)
    layers[1].selected_label = 4
    c.set_mode(POLYGON)
    _press(c, QPoint(5, 5))
    _press(c, QPoint(5, 20))
    _press(c, QPoint(20, 20))
    _double_click(c, QPoint(20, 5))
    assert layers[1].data[12, 12] == 4  # inside the closed quad
    assert c._polygon_pts == []


def test_polygon_right_click_removes_last_vertex(app):
    c, layers, *_ = _make_canvas(app)
    c.set_mode(POLYGON)
    _press(c, QPoint(1, 1))
    _press(c, QPoint(2, 2))
    assert len(c._polygon_pts) == 2
    _press(c, QPoint(3, 3), button=Qt.MouseButton.RightButton)
    assert len(c._polygon_pts) == 1


def test_mouse_move_updates_hover_status(app):
    c, layers, statuses, _picks = _make_canvas(app)
    _move(c, QPoint(9, 9), buttons=Qt.MouseButton.NoButton)
    assert statuses  # at least one status string was emitted
    assert "9" in statuses[-1]


def test_no_edit_target_reports_status_hint(app):
    c, layers, statuses, _picks = _make_canvas(app, with_labels=False)
    c.set_mode(PAINT)
    _press(c, QPoint(5, 5))
    assert statuses
    assert "select a Labels layer" in statuses[-1]


# ── view actions ─────────────────────────────────────────────────────────────
def test_wheel_zooms_in_and_out(app):
    c, *_ = _make_canvas(app)
    z0 = c._zoom
    ev_in = QWheelEvent(QPointF(20, 20), QPointF(20, 20), QPoint(0, 0), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    c.wheelEvent(ev_in)
    assert c._zoom > z0
    z1 = c._zoom
    ev_out = QWheelEvent(QPointF(20, 20), QPointF(20, 20), QPoint(0, 0), QPoint(0, -120),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
    c.wheelEvent(ev_out)
    assert c._zoom < z1


def test_toggle_mip_is_a_noop_for_a_single_plane(app):
    c, *_ = _make_canvas(app)
    assert c.layers.n_planes == 1
    assert c.toggle_mip() is False
    assert c.mip is False


def test_toggle_mip_flips_for_a_volume(app):
    layers = LayerList()
    layers.add(ImageLayer("vol", np.zeros((3, 10, 10, 3), dtype=np.uint8)))
    c = Canvas(theme.DARK, layers)
    assert layers.n_planes == 3
    assert c.toggle_mip() is True
    assert c.mip is True


def test_roll_channel_cycles_visibility_across_image_layers(app):
    layers = LayerList()
    a = ImageLayer("DAPI", np.zeros((5, 5, 3), dtype=np.uint8))
    b = ImageLayer("Membrane", np.zeros((5, 5, 3), dtype=np.uint8))
    b.visible = False
    layers.add(a)
    layers.add(b)
    c = Canvas(theme.DARK, layers)
    c.roll_channel()
    assert a.visible is False
    assert b.visible is True
    c.roll_channel()
    assert a.visible is True
    assert b.visible is False


def test_step_z_moves_current_plane(app):
    layers = LayerList()
    layers.add(LabelsLayer("Segmentation", np.zeros((4, 5, 5), dtype=np.int32)))
    c = Canvas(theme.DARK, layers)
    c.step_z(2)
    assert layers.current_z == 2
    c.step_z(-1)
    assert layers.current_z == 1


def test_step_z_noop_in_mip_mode(app):
    layers = LayerList()
    layers.add(LabelsLayer("Segmentation", np.zeros((4, 5, 5), dtype=np.int32)))
    c = Canvas(theme.DARK, layers)
    c.mip = True
    c.step_z(2)
    assert layers.current_z == 0


# ── pure pixel-compositing functions (no Qt/widget needed) ──────────────────
def test_blend_translucent_opaque_additive():
    canvas = np.zeros((2, 2, 3), dtype=np.float32)
    rgb = np.full((2, 2, 3), 100.0, dtype=np.float32)
    alpha = np.full((2, 2), 0.5, dtype=np.float32)
    assert np.allclose(_blend(canvas, rgb, alpha, "translucent"), 50.0)
    assert np.allclose(_blend(canvas, rgb, alpha, "additive"), 50.0)
    opaque = _blend(canvas, rgb, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32), "opaque")
    assert opaque[0, 0, 0] == 0.0
    assert opaque[0, 1, 0] == 100.0


def test_contour_mask_marks_only_boundaries():
    plane = np.zeros((10, 10), dtype=np.int32)
    plane[3:7, 3:7] = 5
    mask = _contour_mask(plane, thickness=1)
    assert mask[3, 3]      # a boundary pixel of the block
    assert not mask[5, 5]  # the block's interior is not a boundary
    assert not mask[0, 0]  # background interior is not a boundary


def test_interpolate_fills_gaps_for_a_fast_drag():
    pts = _interpolate((0.0, 0.0), (10.0, 0.0), step=2.0)
    assert pts[-1] == (10.0, 0.0)
    assert len(pts) >= 4


def test_interpolate_from_none_returns_just_the_endpoint():
    assert _interpolate(None, (3.0, 4.0), step=1.0) == [(3.0, 4.0)]


def test_render_image_applies_contrast_stretch():
    layer = ImageLayer("g", np.array([[0, 128, 255]], dtype=np.uint8).reshape(1, 3, 1).repeat(3, axis=2))
    layer.contrast_limits = (0.0, 255.0)
    rgb, alpha = _render_image(layer, layer.data)
    assert rgb[0, 0].max() == pytest.approx(0.0, abs=1.0)
    assert rgb[0, 2].max() == pytest.approx(255.0, abs=1.0)
    assert (alpha == layer.opacity).all()


def test_render_labels_alpha_zero_at_background():
    layer = LabelsLayer("L", np.array([[0, 3], [3, 0]], dtype=np.int32))
    rgb, alpha = _render_labels(layer, layer.data)
    assert alpha[0, 0] == 0.0
    assert alpha[0, 1] == pytest.approx(layer.opacity)


def test_render_labels_show_selected_label_hides_others():
    data = np.array([[1, 2]], dtype=np.int32)
    layer = LabelsLayer("L", data)
    layer.selected_label = 1
    layer.show_selected_label = True
    _rgb, alpha = _render_labels(layer, data)
    assert alpha[0, 0] > 0
    assert alpha[0, 1] == 0
