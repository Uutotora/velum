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


# ── "Open Sample" ────────────────────────────────────────────────────────────
def test_open_sample_opens_an_existing_project(app, controller):
    opened = []
    home = _home(app, controller, on_open=opened.append)
    home._open_sample()
    assert len(opened) == 1
    assert opened[0] in {p.id for p in controller.list_projects()}


def test_open_sample_falls_back_to_new_project_when_store_empty(app, empty_controller):
    seen = []
    home = _home(app, empty_controller, on_new_project=lambda: seen.append(True))
    home._open_sample()
    assert seen == [True]


def test_open_sample_card_wired_to_open_sample(app, controller):
    opened = []
    home = _home(app, controller, on_open=opened.append)
    home._quick_grid.itemAt(3).widget().mouseReleaseEvent(None)  # "Open Sample"
    assert len(opened) == 1


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


# ── hover-lift wiring ────────────────────────────────────────────────────────
def test_quick_cards_and_recent_rows_get_hover_shadow(app, controller):
    home = _home(app, controller)
    card = home._quick_grid.itemAt(0).widget()
    assert card.graphicsEffect() is not None
    recent = controller.recent(limit=1)
    row = home._recent_row(to_card(recent[0]))
    assert row.graphicsEffect() is not None
