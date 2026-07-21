"""Tests for studio/image_crop.py — the Steam-style region picker.

Pure crop maths (clamp_crop / apply_drag / crop_to_source) plus construction
smoke tests for the Qt canvas/dialog. Gated on PyQt6 (the module imports Qt at
top), so this skips in the light CI job.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
ic = pytest.importorskip("studio.image_crop")

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QWidget


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


# ── clamp_crop ────────────────────────────────────────────────────────────────
def test_clamp_crop_keeps_rect_in_bounds():
    assert ic.clamp_crop(-10, -10, 50, 50, 200, 200) == (0, 0, 50, 50)
    assert ic.clamp_crop(180, 180, 50, 50, 200, 200) == (150, 150, 50, 50)


def test_clamp_crop_enforces_min_size():
    x, y, w, h = ic.clamp_crop(10, 10, 5, 5, 200, 200, min_size=28)
    assert w == 28 and h == 28


def test_clamp_crop_caps_size_to_bounds():
    x, y, w, h = ic.clamp_crop(0, 0, 999, 999, 200, 150)
    assert (x, y, w, h) == (0, 0, 200, 150)


# ── apply_drag ────────────────────────────────────────────────────────────────
def test_apply_drag_move():
    assert ic.apply_drag((10, 10, 50, 50), "move", 5, 7, 200, 200) == (15, 17, 50, 50)


def test_apply_drag_move_is_clamped():
    x, y, w, h = ic.apply_drag((10, 10, 50, 50), "move", 1000, 1000, 200, 200)
    assert (x, y, w, h) == (150, 150, 50, 50)


def test_apply_drag_east_edge_grows_width_only():
    assert ic.apply_drag((10, 10, 50, 50), "e", 20, 0, 200, 200) == (10, 10, 70, 50)


def test_apply_drag_west_edge_moves_left_edge():
    x, y, w, h = ic.apply_drag((10, 10, 50, 50), "w", -10, 0, 200, 200)
    assert (x, y, w, h) == (0, 10, 60, 50)


def test_apply_drag_corner_resizes_both_axes():
    assert ic.apply_drag((10, 10, 50, 50), "se", 10, 20, 200, 200) == (10, 10, 60, 70)


def test_apply_drag_respects_min_size_when_shrinking():
    x, y, w, h = ic.apply_drag((10, 10, 50, 50), "e", -1000, 0, 200, 200, min_size=28)
    assert w == 28


# ── crop_to_source ────────────────────────────────────────────────────────────
def test_crop_to_source_scales_display_to_pixels():
    assert ic.crop_to_source(10, 10, 50, 50, 100, 100, 200, 200) == (20, 20, 100, 100)


def test_crop_to_source_clamps_to_source_bounds():
    x, y, w, h = ic.crop_to_source(90, 90, 50, 50, 100, 100, 100, 100)
    assert x + w <= 100 and y + h <= 100 and w >= 1 and h >= 1


# ── Qt smoke ──────────────────────────────────────────────────────────────────
def _test_image(app, w=200, h=120):
    pm = QPixmap(w, h)
    pm.fill()
    return pm


def test_cover_fit_returns_sized_pixmap(app):
    out = ic.cover_fit(_test_image(app), 220, 56)
    assert out.width() == 220 and out.height() == 56


def test_canvas_cropped_pixmap_is_a_subregion(app):
    from studio import theme
    canvas = ic._CropCanvas(_test_image(app, 400, 300), theme.DARK)
    cropped = canvas.cropped_pixmap()
    # default crop is ~84%×76% of the source, so strictly smaller than the whole
    assert 0 < cropped.width() < 400
    assert 0 < cropped.height() < 300


def test_crop_dialog_apply_hands_back_a_pixmap(app, tmp_path):
    from studio import theme
    src = tmp_path / "img.png"
    _test_image(app, 300, 200).save(str(src), "PNG")
    host = QWidget()
    got = {}
    dlg = ic.CropDialog(host, theme.DARK, str(src),
                        on_apply=lambda px: got.setdefault("px", px))
    dlg._apply()
    assert isinstance(got.get("px"), QPixmap)
    assert not got["px"].isNull()
