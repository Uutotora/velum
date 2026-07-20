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
    ShapesLayer,
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


def test_points_layer_left_click_adds_a_point(app):
    layers = LayerList()
    layers.add(ImageLayer("Image", np.zeros((40, 40, 3), dtype=np.uint8)), select=False)
    points = PointsLayer("Prompts")
    layers.add(points)
    c = Canvas(theme.DARK, layers)
    c.resize(40, 40)
    c._zoom = 1.0
    c._pan = QPointF(0.0, 0.0)
    c._fitted = True
    c.set_mode(PAINT)  # pan_zoom always pans, regardless of selection — any
                       # other mode means "the active tool acts on the click"
    _press(c, QPoint(12, 20))
    assert points.points == [(20.0, 12.0)]


def test_points_layer_right_click_removes_nearest(app):
    layers = LayerList()
    points = PointsLayer("Prompts", points=[(5.0, 5.0), (30.0, 30.0)])
    layers.add(points)
    c = Canvas(theme.DARK, layers)
    c.resize(40, 40)
    c._zoom = 1.0
    c._pan = QPointF(0.0, 0.0)
    c._fitted = True
    c.set_mode(PAINT)
    _press(c, QPoint(6, 6), button=Qt.MouseButton.RightButton)
    assert points.points == [(30.0, 30.0)]


def test_shapes_layer_polygon_click_and_double_click_adds_a_shape(app):
    layers = LayerList()
    shapes = ShapesLayer("Corrections")
    layers.add(shapes)
    c = Canvas(theme.DARK, layers)
    c.resize(40, 40)
    c._zoom = 1.0
    c._pan = QPointF(0.0, 0.0)
    c._fitted = True
    c.set_mode(POLYGON)
    _press(c, QPoint(5, 5))
    _press(c, QPoint(5, 20))
    _press(c, QPoint(20, 20))
    _double_click(c, QPoint(20, 5))
    assert len(shapes.shapes) == 1
    assert shapes.shapes[0]["type"] == "polygon"
    assert c._polygon_pts == []


def test_shapes_layer_right_click_pops_last_vertex(app):
    layers = LayerList()
    layers.add(ShapesLayer("Corrections"))
    c = Canvas(theme.DARK, layers)
    c.set_mode(POLYGON)
    _press(c, QPoint(1, 1))
    _press(c, QPoint(2, 2))
    _press(c, QPoint(3, 3), button=Qt.MouseButton.RightButton)
    assert len(c._polygon_pts) == 1


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


def _image_overlaps_viewport(c, min_overlap=1.0):
    """The image rect (in widget coords) must still intersect the viewport —
    the pan-clamping invariant this whole test group checks."""
    shape = c._base_shape()
    h, w = shape
    scaled_w, scaled_h = w * c._zoom, h * c._zoom
    left, top = c._pan.x(), c._pan.y()
    right, bottom = left + scaled_w, top + scaled_h
    overlap_w = min(right, c.width()) - max(left, 0)
    overlap_h = min(bottom, c.height()) - max(top, 0)
    return overlap_w >= min_overlap and overlap_h >= min_overlap


def test_pan_cannot_escape_the_viewport_via_drag(app):
    c, *_ = _make_canvas(app, h=40, w=40)
    c.resize(200, 200)
    c.home()
    _press(c, QPoint(20, 20))
    # a wildly large drag, far past any reasonable pan distance
    _move(c, QPoint(20000, 20000))
    _release(c, QPoint(20000, 20000))
    assert _image_overlaps_viewport(c)


def test_pan_cannot_escape_the_viewport_via_repeated_wheel_zoom(app):
    c, *_ = _make_canvas(app, h=40, w=40)
    c.resize(200, 200)
    c.home()
    # zoom out repeatedly at a corner far from the image centre — the kind
    # of repeated extreme input that used to be able to walk the pan away
    # from the image indefinitely
    for _ in range(30):
        ev = QWheelEvent(QPointF(1, 1), QPointF(1, 1), QPoint(0, 0), QPoint(0, -120),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        c.wheelEvent(ev)
    assert _image_overlaps_viewport(c)
    for _ in range(30):
        ev = QWheelEvent(QPointF(199, 199), QPointF(199, 199), QPoint(0, 0), QPoint(0, 120),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        c.wheelEvent(ev)
    assert _image_overlaps_viewport(c)


def test_clamp_pan_handles_a_tiny_zoomed_out_image(app):
    """A heavily zoomed-out image (scaled size well under the margin) must
    not trigger an inverted clamp range."""
    c, *_ = _make_canvas(app, h=40, w=40)
    c.resize(400, 400)
    c._zoom = 0.05
    c._pan = QPointF(5000, -5000)
    c._clamp_pan()
    assert _image_overlaps_viewport(c, min_overlap=0.5)


def test_toggle_mip_works_even_on_a_single_plane(app):
    """Real napari's ndisplay toggle has no dimensionality guard at all —
    it works unconditionally, even on flat 2-D data (confirmed against the
    installed napari source). A single-plane image gets the pseudo-3D tilt
    (_draw_pseudo_3d) rather than a real MIP projection, but the *toggle
    itself* must never silently refuse, unlike an earlier version of this
    method."""
    c, *_ = _make_canvas(app)
    assert c.layers.n_planes == 1
    assert c.toggle_mip() is True
    assert c.mip is True
    assert c.toggle_mip() is False
    assert c.mip is False


def test_toggle_mip_flips_for_a_volume(app):
    layers = LayerList()
    layers.add(ImageLayer("vol", np.zeros((3, 10, 10, 3), dtype=np.uint8)))
    c = Canvas(theme.DARK, layers)
    assert layers.n_planes == 3
    assert c.toggle_mip() is True
    assert c.mip is True


def test_pseudo_3d_tilt_visibly_changes_a_flat_image(app):
    layers = LayerList()
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[10:30, 10:30] = [200, 60, 60]
    layers.add(ImageLayer("img", img))
    c = Canvas(theme.DARK, layers)
    c.resize(60, 60)
    c.home()
    before = c.grab().toImage()
    assert c.toggle_mip() is True
    c.update()
    app.processEvents()
    after = c.grab().toImage()
    assert before != after


def test_dragging_in_3d_mode_rotates_instead_of_panning(app):
    """Left-drag on a flat image with the "3D" toggle on orbits the tilt
    (real napari's 3-D view rotates on left-drag) rather than panning."""
    c, *_ = _make_canvas(app, h=40, w=40)
    c.mip = True
    start_pan = QPointF(c._pan)
    _press(c, QPoint(20, 20))
    assert c._rotating is True
    assert c._panning is False
    _move(c, QPoint(20, 50))
    assert c._rot_x == pytest.approx(0.7)   # 0.4 default + 30px * 0.01 sensitivity
    assert c._rot_y == pytest.approx(0.0)
    assert c._pan == start_pan               # a rotate-drag must never also pan
    _release(c, QPoint(20, 50))
    assert c._rotating is False


def test_horizontal_drag_in_3d_mode_changes_yaw(app):
    c, *_ = _make_canvas(app, h=40, w=40)
    c.mip = True
    _press(c, QPoint(20, 20))
    _move(c, QPoint(60, 20))
    assert c._rot_y == pytest.approx(0.4)    # 40px * 0.01 sensitivity
    assert c._rot_x == pytest.approx(0.4)    # unchanged from the default pitch


def test_rotation_drag_clamps_to_max_rot(app):
    c, *_ = _make_canvas(app, h=40, w=40)
    c.mip = True
    _press(c, QPoint(0, 0))
    _move(c, QPoint(0, 100_000))
    assert c._rot_x == pytest.approx(canvas_mod._MAX_ROT)
    _move(c, QPoint(0, -100_000))
    assert c._rot_x == pytest.approx(-canvas_mod._MAX_ROT)


def test_rotation_drag_actually_changes_rendered_pixels(app):
    layers = LayerList()
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[10:30, 10:30] = [200, 60, 60]
    layers.add(ImageLayer("img", img))
    c = Canvas(theme.DARK, layers)
    c.resize(60, 60)
    c.home()
    c.mip = True
    c.update()
    app.processEvents()
    before = c.grab().toImage()
    _press(c, QPoint(30, 30))
    _move(c, QPoint(30, 55))
    c.update()
    app.processEvents()
    after = c.grab().toImage()
    assert before != after


def test_home_resets_rotation_to_the_default_tilt(app):
    c, *_ = _make_canvas(app, h=40, w=40)
    c.mip = True
    c._rot_x = 1.2
    c._rot_y = -0.9
    c.home()
    assert c._rot_x == pytest.approx(0.4)
    assert c._rot_y == pytest.approx(0.0)


def test_middle_button_still_pans_in_3d_mode(app):
    c, *_ = _make_canvas(app, h=40, w=40)
    c.mip = True
    start_pan = QPointF(c._pan)
    start_rot = (c._rot_x, c._rot_y)
    _press(c, QPoint(10, 10), button=Qt.MouseButton.MiddleButton)
    assert c._panning is True
    assert c._rotating is False
    _move(c, QPoint(30, 30), buttons=Qt.MouseButton.MiddleButton)
    assert c._pan != start_pan
    assert (c._rot_x, c._rot_y) == start_rot


def test_drag_in_2d_mode_still_pans_not_rotates(app):
    c, *_ = _make_canvas(app, h=40, w=40)
    assert c.mip is False
    start_rot = (c._rot_x, c._rot_y)
    _press(c, QPoint(10, 10))
    assert c._panning is True
    assert c._rotating is False
    _move(c, QPoint(30, 30))
    assert (c._rot_x, c._rot_y) == start_rot


def test_roll_channel_cycles_visibility_across_image_layers(app):
    layers = LayerList()
    a = ImageLayer("DAPI", np.zeros((5, 5, 3), dtype=np.uint8))
    b = ImageLayer("Membrane", np.zeros((5, 5, 3), dtype=np.uint8))
    b.visible = False
    layers.add(a)
    layers.add(b)
    c = Canvas(theme.DARK, layers)
    assert c.roll_channel() is True
    assert a.visible is False
    assert b.visible is True
    assert c.roll_channel() is True
    assert a.visible is True
    assert b.visible is False


def test_roll_channel_reports_false_with_only_one_image_layer(app):
    layers = LayerList()
    layers.add(ImageLayer("DAPI", np.zeros((5, 5, 3), dtype=np.uint8)))
    c = Canvas(theme.DARK, layers)
    assert c.roll_channel() is False


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


# ── grid mode ────────────────────────────────────────────────────────────────
def test_grid_mode_falls_back_to_overlay_with_one_visible_layer(app):
    c, layers, *_ = _make_canvas(app, with_labels=False)
    c.grid = True
    c.update()
    app.processEvents()
    # must not raise, and must render something (the single-layer fallback path)
    assert c._composited_image() is not None


def test_grid_mode_responds_to_zoom(app):
    """Grid mode used to always auto-fit each tile from scratch, silently
    ignoring self._zoom — the mouse wheel did nothing visible while grid
    mode was on. Real napari's own grid mode shares one camera across every
    tile (confirmed against its source), so zoom must actually change what
    grid mode renders."""
    layers = LayerList()
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[10:30, 10:30] = [200, 60, 60]
    layers.add(ImageLayer("img", img))
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[10:30, 10:30] = 1
    layers.add(LabelsLayer("Segmentation", labels))
    c = Canvas(theme.DARK, layers)
    c.resize(120, 120)
    c.grid = True
    c.home()
    before = c.grab().toImage()
    c._zoom = 3.0
    c.update()
    app.processEvents()
    after = c.grab().toImage()
    assert before != after


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


def test_render_labels_default_combines_fill_and_outline():
    """The classic app's "fill + border, one colour" convention
    (predict_widget.py's _add_filled_labels: fill_opacity=0.35,
    outline opacity=0.7), reproduced as one per-pixel alpha mask instead of
    that module's two stacked layers."""
    data = np.zeros((10, 10), dtype=np.int32)
    data[2:8, 2:8] = 5
    layer = LabelsLayer("L", data)  # defaults: contour=1, fill_opacity=0.35, opacity=0.7
    _rgb, alpha = _render_labels(layer, data)
    assert alpha[4, 4] == pytest.approx(layer.fill_opacity)  # interior
    assert alpha[2, 4] == pytest.approx(layer.opacity)       # boundary row of the block
    assert alpha[0, 0] == 0.0                                # background


def test_render_labels_contour_zero_is_fill_only_everywhere():
    data = np.zeros((10, 10), dtype=np.int32)
    data[2:8, 2:8] = 5
    layer = LabelsLayer("L", data)
    layer.contour = 0
    _rgb, alpha = _render_labels(layer, data)
    assert alpha[4, 4] == pytest.approx(layer.fill_opacity)
    assert alpha[2, 4] == pytest.approx(layer.fill_opacity)  # no outline band now


def test_render_labels_show_selected_label_hides_others():
    data = np.array([[1, 2]], dtype=np.int32)
    layer = LabelsLayer("L", data)
    layer.selected_label = 1
    layer.show_selected_label = True
    _rgb, alpha = _render_labels(layer, data)
    assert alpha[0, 0] > 0
    assert alpha[0, 1] == 0


# ── undo / redo wiring (begin_edit at stroke start + canvas delegation) ────────
def test_paint_stroke_is_undoable_via_canvas(app):
    c, layers, *_ = _make_canvas(app)
    c.set_mode(PAINT)
    _press(c, QPoint(20, 20))
    _move(c, QPoint(24, 20))
    _release(c, QPoint(24, 20))
    assert layers[1].max_label == 1
    assert c.undo() is True
    assert layers[1].max_label == 0  # whole stroke reverted
    assert c.redo() is True
    assert layers[1].max_label == 1  # and re-applied


def test_fill_click_is_a_single_undo_step(app):
    c, layers, *_ = _make_canvas(app)
    layers[1].data[10:20, 10:20] = 2
    layers[1].selected_label = 9
    c.set_mode(FILL)
    _press(c, QPoint(15, 15))
    assert (layers[1].data[10:20, 10:20] == 9).all()
    assert c.undo() is True
    assert (layers[1].data[10:20, 10:20] == 2).all()  # back to the pre-fill label


def test_canvas_undo_redo_noop_without_labels_layer(app):
    c, layers, *_ = _make_canvas(app, with_labels=False)
    assert c.undo() is False
    assert c.redo() is False


def test_pick_does_not_create_an_undo_step(app):
    c, layers, *_ = _make_canvas(app)
    layers[1].data[20, 20] = 4
    c.set_mode(PICK)
    _press(c, QPoint(20, 20))
    assert not layers[1].can_undo  # picking a colour edits nothing
