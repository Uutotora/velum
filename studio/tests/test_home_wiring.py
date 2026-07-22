"""Headless tests for Home screen's clickable elements (studio/screens.py).

Offscreen Qt, no napari/torch. Documentation / Getting started guide now open
the in-app Guide screen (studio/guide_screen.py) rather than an external .md
file; GitHub is the one resource link still verified without really invoking
QDesktopServices — that would try to open a browser for real.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
screens = pytest.importorskip("studio.screens")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFrame, QLabel

from studio import theme
from studio.project import ProjectStore
from studio.project_controller import ProjectController, to_card


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(tmp_path):
    return ProjectController(ProjectStore(tmp_path))


@pytest.fixture
def empty_controller(tmp_path):
    return ProjectController(ProjectStore(tmp_path), seed_if_empty=False)


def _home(app, controller, on_navigate=None, on_open=None, on_new_project=None):
    return screens.HomeScreen(
        theme.DARK, controller,
        on_navigate or (lambda k: None),
        on_open or (lambda i: None),
        on_new_project or (lambda: None))


# ── quick-card click mechanism ────────────────────────────────────────────────
def test_quick_card_click_fires_callback(app, controller):
    home = _home(app, controller)
    seen = []
    card = home._quick_card("folder", "X", "y", "primary", lambda: seen.append(True))
    card.mouseReleaseEvent(None)
    assert seen == [True]


def test_new_project_cta_opens_dialog(app, controller):
    seen = []
    home = _home(app, controller, on_new_project=lambda: seen.append(True))
    home._new_project_cta.click()
    assert seen == [True]


def test_quick_grid_has_four_cards(app, controller):
    home = _home(app, controller)
    assert home._quick_grid.count() == 4


def test_new_project_and_import_images_cards_open_dialog(app, controller):
    seen = []
    home = _home(app, controller, on_new_project=lambda: seen.append(True))
    for i in (0, 1):  # "New Project", "Import Images"
        home._quick_grid.itemAt(i).widget().mouseReleaseEvent(None)
    assert seen == [True, True]


def test_train_a_model_card_navigates_to_train(app, controller):
    navigated = []
    home = _home(app, controller, on_navigate=navigated.append)
    home._quick_grid.itemAt(2).widget().mouseReleaseEvent(None)
    assert navigated == ["train"]


# ── "Datasets" quick card (replaced the old "Open Sample" card) ──────────────
def test_datasets_card_navigates_to_datasets(app, controller):
    navigated = []
    home = _home(app, controller, on_navigate=navigated.append)
    home._quick_grid.itemAt(3).widget().mouseReleaseEvent(None)  # "Datasets"
    assert navigated == ["datasets"]


# ── resource links ───────────────────────────────────────────────────────────
def test_ask_the_assistant_navigates(app, controller):
    navigated = []
    home = _home(app, controller, on_navigate=navigated.append)
    row = home._res_link("Ask the Assistant", "assistant", False, lambda: navigated.append("assistant"))
    row.mouseReleaseEvent(None)
    assert navigated == ["assistant"]


def _find_resource_row(home, link_text: str) -> QFrame:
    """The real ``_res_link`` row whose label reads ``link_text`` — not a
    re-fabricated stand-in — so this actually exercises HomeScreen's own
    ``_aside()`` wiring, the same technique as
    ``test_refresh_shows_a_project_created_after_construction`` below.

    Direct children only: ``QScrollArea`` is itself a ``QFrame`` and
    recursively contains every label on the page, so an unrestricted
    ``findChildren(QLabel)`` matches that outer container first, not the
    small row — and calling the *real* (non-monkeypatched)
    ``mouseReleaseEvent`` on an arbitrary Qt widget with ``None`` instead of
    a real ``QMouseEvent`` segfaults.
    """
    for frame in home.findChildren(QFrame):
        direct_labels = frame.findChildren(QLabel, options=Qt.FindChildOption.FindDirectChildrenOnly)
        if any(lb.text() == link_text for lb in direct_labels):
            return frame
    raise AssertionError(f"no resource row labelled {link_text!r}")


def test_documentation_link_opens_the_guide_screen(app, controller):
    navigated = []
    home = _home(app, controller, on_navigate=navigated.append)
    _find_resource_row(home, "Documentation").mouseReleaseEvent(None)
    assert navigated == ["guide"]


def test_getting_started_guide_link_deep_links_to_the_getting_started_article(app, controller):
    navigated = []
    home = _home(app, controller, on_navigate=navigated.append)
    _find_resource_row(home, "Getting started guide").mouseReleaseEvent(None)
    assert navigated == ["guide:getting-started"]


def test_github_url_is_none_or_a_real_https_url():
    url = screens._github_url()
    assert url is None or url.startswith("https://github.com/")


def test_open_github_calls_desktop_services_when_url_available(monkeypatch):
    monkeypatch.setattr(screens, "_github_url", lambda: "https://github.com/example/repo")
    calls = []
    monkeypatch.setattr(screens.QDesktopServices, "openUrl", lambda url: calls.append(url.toString()))
    screens._open_github()
    assert calls == ["https://github.com/example/repo"]


def test_open_github_noop_when_no_remote(monkeypatch):
    monkeypatch.setattr(screens, "_github_url", lambda: None)
    calls = []
    monkeypatch.setattr(screens.QDesktopServices, "openUrl", lambda url: calls.append(url))
    screens._open_github()
    assert calls == []


# ── refresh() picks up projects created elsewhere ─────────────────────────────
def test_refresh_shows_a_project_created_after_construction(app, empty_controller):
    from PyQt6.QtWidgets import QFrame, QLabel
    home = _home(app, empty_controller)
    assert len([f for f in home.findChildren(QFrame) if f.objectName() == "RRow"]) == 0

    empty_controller.store.create("Brand New Project")
    home.refresh()

    rows = [f for f in home.findChildren(QFrame) if f.objectName() == "RRow"]
    assert len(rows) == 1
    labels = [lb.text() for lb in rows[0].findChildren(QLabel)]
    assert any("Brand New Project" in text for text in labels)


# ── refresh() motion: scoped, not whole-screen ─────────────────────────────────
def test_refresh_fades_only_the_recent_section_not_the_whole_home_screen(app, controller):
    """Regression test: HomeScreen.refresh() used to rely on app.py fading
    the *entire* screen in via a QGraphicsOpacityEffect on every single
    visit to Home -- expensive (forces re-rasterising every quick-card's and
    every recent-row's own QGraphicsDropShadowEffect, from
    install_hover_lift/soft_shadow, on every frame of the fade -- up to ~10
    of them at once) and, since it replayed on every revisit to an
    already-built, mostly-unchanged screen rather than just the first time,
    reported directly as looking bad ("recent projects... каждый раз с
    ужасной анимацией" -- shows with a terrible animation every time).
    Fixed by dropping "home" from app.py's generic per-navigation fade_in()
    (see test_app_wiring.py's mirroring test) and instead fading only the
    part that actually changed, here: HomeScreen itself must never carry a
    graphics effect; only the freshly-rebuilt recent-projects section does --
    and only when the list actually changed (a project was created), which
    this test forces so the fade is exercised at all.
    """
    from PyQt6.QtWidgets import QGraphicsOpacityEffect
    home = _home(app, controller)
    controller.store.create("Forces A Recent-List Change")
    home.refresh()
    assert home.graphicsEffect() is None
    assert isinstance(home._recent_widget.graphicsEffect(), QGraphicsOpacityEffect)


def test_refresh_is_a_noop_when_the_recent_list_is_unchanged(app, controller):
    """The dominant complaint was that the recent-projects entrance animation
    replayed on *every* revisit to Home, even when nothing had changed. Now
    refresh() fingerprints the list (HomeScreen._recent_sig) and, on an
    identical revisit, neither rebuilds the section (same widget object) nor
    fades it (no graphics effect). Only a genuine change re-animates."""
    home = _home(app, controller)
    before = home._recent_widget
    home.refresh()  # nothing changed since construction
    assert home._recent_widget is before          # not rebuilt
    assert home._recent_widget.graphicsEffect() is None  # not faded


def test_refresh_defers_recent_row_hover_shadows_until_the_fade_settles(app, controller):
    """When refresh() does re-animate, the rebuilt rows are created *without*
    their hover-lift shadow effects so the fade never has to composite a
    QGraphicsDropShadowEffect per row on every frame; the shadows arrive only
    once the fade finishes, via _install_recent_hover(). Assert the two
    halves directly: right after the (still-running) fade the rows carry no
    effect, and _install_recent_hover() then gives them one."""
    home = _home(app, controller)
    controller.store.create("Another Fresh Project")
    home.refresh()
    rows = [f for f in home._recent_widget.findChildren(QFrame) if f.objectName() == "RRow"]
    assert rows and all(r.graphicsEffect() is None for r in rows)  # deferred
    home._install_recent_hover(home._recent_widget)
    assert all(r.graphicsEffect() is not None for r in rows)       # then installed


def test_refresh_plays_the_waving_hand_greeting(app, controller):
    """The "Welcome back" emoji waves once per Home visit -- refresh() is
    called on every navigation to Home, including the very first (see
    app.py's navigate()). Assert on the configured animation object right
    after triggering it, not on a settled end value: the shared
    QApplication's animation-driver timer can be silently wedged by an
    unrelated, earlier-alphabetical test module having started its own
    short-lived animations without pumping the event loop afterwards to let
    them finish (see test_components.py's SmoothScrollArea tests for the
    full story, and this file's own established pattern for the same
    reason)."""
    from PyQt6.QtCore import QVariantAnimation
    home = _home(app, controller)
    home.refresh()
    assert home._wave._anim.state() == QVariantAnimation.State.Running


# ── hover-lift wiring ────────────────────────────────────────────────────────
def test_quick_cards_and_recent_rows_get_hover_shadow(app, controller):
    home = _home(app, controller)
    card = home._quick_grid.itemAt(0).widget()
    assert card.graphicsEffect() is not None
    recent = controller.recent(limit=1)
    row = home._recent_row(to_card(recent[0]))
    assert row.graphicsEffect() is not None


def test_refresh_rebuilds_the_hero_for_the_new_most_recent_project(app, controller):
    """Regression: the KPI row + "pick up where you left off" hero used to be
    built once and never refreshed, so a changed cover / a newly-touched
    most-recent project only showed after an app restart. refresh() now rebuilds
    the top block, so the hero tracks the current most-recent project live."""
    home = _home(app, controller)
    newest = controller.recent(limit=1)[0]

    # make a *different* project the most recent (explicit far-future stamp so
    # it's unambiguously newest — _now_iso() is only second-resolution, which
    # can tie with the just-seeded projects in a fast test)
    others = [p for p in controller.list_projects() if p.id != newest.id]
    bumped = others[0]
    bumped.updated_at = "2099-01-01T00:00:00+00:00"
    controller.store.save(bumped, touch=False)

    home.refresh()
    top_labels = [lb.text() for lb in home._top_widget.findChildren(QLabel)]
    assert any(bumped.name in txt for txt in top_labels)


def test_refresh_reflects_a_cover_change_in_the_hero(app, controller):
    """A cover set on the most-recent project shows on the hero after refresh
    (the hero's CoverView is rebuilt from current state, not frozen)."""
    home = _home(app, controller)
    newest = controller.recent(limit=1)[0]
    controller.set_cover(newest.id, kind="color", color="#2bd4c0")
    home.refresh()
    # the hero's cover carries the new colour (find the CoverView by its state)
    from studio.covers import CoverView
    covers = [c for c in home._top_widget.findChildren(CoverView) if c._color == "#2bd4c0"]
    assert covers, "hero cover did not pick up the new colour after refresh"
