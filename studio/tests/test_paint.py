"""Headless tests for studio/paint.py -- specifically NucleiView's paint cache.

Offscreen Qt, no napari/torch.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
paint = pytest.importorskip("studio.paint")

from PyQt6.QtWidgets import QApplication


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def counted(monkeypatch):
    """Wrap paint.paint_nuclei (the expensive procedural painter) to count
    real invocations, without changing what it actually draws."""
    calls = []
    orig = paint.paint_nuclei

    def counting(*a, **kw):
        calls.append(1)
        return orig(*a, **kw)

    monkeypatch.setattr(paint, "paint_nuclei", counting)
    return calls


def test_nuclei_view_caches_between_repaints_at_same_size(app, counted):
    """Repeated repaints at an unchanged size must not re-run the procedural
    field generator. This is the fix for visibly stuttering scroll in
    ProjectsScreen's card grid (docs/velum/BACKLOG.md's "Projects tab v2"
    entry): every visible card's NucleiView cover used to regenerate its
    whole gradient+polygon field from scratch on every scroll-triggered
    repaint, and there can be many simultaneously-visible cards. Confirmed
    (by temporarily reverting the cache) that this test fails with 3 calls,
    not 0, against the pre-cache implementation -- not just written and
    assumed correct.
    """
    view = paint.NucleiView(seed=1, big=False, radius=14, top_only=True, min_size=(0, 0))
    view.resize(200, 132)
    view.show()
    app.processEvents()
    assert len(counted) == 1  # the first paint still has to do real work

    counted.clear()
    view.repaint()
    view.repaint()
    view.repaint()
    assert counted == [], "a cached repaint at the same size must not redraw the field"


def test_nuclei_view_regenerates_after_resize(app, counted):
    """A real size change (a responsive grid card's column tracking the
    window) must still invalidate the cache -- the one thing a pre-baked
    nuclei_pixmap() alone can't do (see NucleiView's docstring), so the
    cache must not be "sticky" past an actual resize.
    """
    view = paint.NucleiView(seed=2, big=False, radius=14, top_only=True, min_size=(0, 0))
    view.resize(200, 132)
    view.show()
    app.processEvents()
    first_cache = view._cache

    counted.clear()
    view.resize(260, 132)
    view.repaint()
    assert len(counted) == 1
    assert view._cache is not first_cache
