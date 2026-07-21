"""Tests for studio/covers.py — project cover art.

The pure helpers (``auto_color``/``resolve_color``) plus a smoke test that
``CoverView``/``cover_pixmap`` construct and render without a display. Gated on
PyQt6 (covers.py imports Qt at module top), so this skips in the light CI job.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
covers = pytest.importorskip("studio.covers")

from PyQt6.QtWidgets import QApplication


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


# ── auto_color / resolve_color (pure) ─────────────────────────────────────────
def test_auto_color_is_deterministic():
    assert covers.auto_color("fluorescence-nuclei-dapi") == covers.auto_color("fluorescence-nuclei-dapi")


def test_auto_color_is_from_the_palette():
    palette = {hex_ for _name, hex_ in covers.COVER_COLORS}
    assert covers.auto_color("any-project-id") in palette


def test_auto_color_differs_across_ids():
    ids = ["dapi", "he-tissue", "mitosis", "bbbc039", "organoid", "phantom"]
    colors = {covers.auto_color(i) for i in ids}
    assert len(colors) >= 4  # good spread, not all colliding


def test_resolve_color_uses_pinned_color_when_kind_is_color():
    assert covers.resolve_color("color", "#123456", "pid") == "#123456"


def test_resolve_color_falls_back_to_auto_for_auto_or_empty():
    assert covers.resolve_color("auto", "", "pid") == covers.auto_color("pid")
    assert covers.resolve_color("color", "", "pid") == covers.auto_color("pid")


# ── rendering smoke tests ─────────────────────────────────────────────────────
def test_cover_pixmap_auto_is_non_null(app):
    px = covers.cover_pixmap(120, 64, kind="auto", color="", image=None, project_id="pid")
    assert not px.isNull()


def test_cover_pixmap_color_is_non_null(app):
    px = covers.cover_pixmap(120, 64, kind="color", color="#6d87f1", image=None, project_id="pid")
    assert not px.isNull()


def test_cover_pixmap_image_falls_back_to_aurora_when_image_missing(app):
    # kind=image but no source pixmap → must still paint (the aurora), not crash
    px = covers.cover_pixmap(120, 64, kind="image", color="", image=None, project_id="pid")
    assert not px.isNull()


def test_cover_view_constructs_and_set_cover_updates(app):
    view = covers.CoverView(kind="auto", project_id="pid", min_size=(120, 64))
    view.resize(120, 64)
    view.set_cover(kind="color", color="#2bd4c0")
    assert view._kind == "color" and view._color == "#2bd4c0"
    view.set_cover(kind="image", image_path="/does/not/exist.png")
    assert view._kind == "image" and view._src is None  # missing file → no source
