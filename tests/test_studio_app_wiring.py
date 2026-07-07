"""Headless wiring/smoke tests for the Studio shell (napari_app/studio/*).

Constructs the sidebar, Home/Projects screens and the full StudioWindow under
``QT_QPA_PLATFORM=offscreen`` — no display, and crucially no napari (the shell
is designed so napari is only touched inside ``main()``/``build_workspace``).
Skipped in the lightweight CI job that has no PyQt6.

Verifies what's verifiable without a GUI: widgets construct, navigation flips
the active screen, and the Home/Projects views reflect the ProjectStore.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
# import the shell pieces (these must NOT pull in napari/torch)
components = pytest.importorskip("napari_app.studio.components")
screens = pytest.importorskip("napari_app.studio.screens")
app_mod = pytest.importorskip("napari_app.studio.app")

from PyQt6.QtWidgets import QApplication

from napari_app.studio import theme
from napari_app.studio.project import ProjectStore


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path):
    s = ProjectStore(tmp_path)
    s.create("Alpha", description="first", tags=["nuclei"])
    s.create("Beta", description="second", tags=["H&E"])
    return s


# ── components ───────────────────────────────────────────────────────────────
def test_sidebar_builds_and_emits_navigate(app):
    seen = []
    sb = components.Sidebar(app_mod._NAV, theme.DARK)
    sb.navigate.connect(seen.append)
    # click the Projects nav item
    sb._items["projects"].click()
    assert seen == ["projects"]
    # set_active only marks checkable nav items
    sb.set_active("home")
    assert sb._items["home"].isChecked()
    assert not sb._items["projects"].isChecked()


def test_chip_and_buttons_construct(app):
    for kind in ("default", "primary", "signal", "success", "muted"):
        assert components.Chip("x", theme.LIGHT, kind) is not None
    assert components.PrimaryButton("Go", theme.DARK, "plus").text() == "Go"
    assert components.GhostButton("Filter", theme.LIGHT, "filter").text() == "Filter"


# ── screens ──────────────────────────────────────────────────────────────────
def test_home_screen_lists_recent_projects(app, store):
    seen = {}
    home = screens.HomeScreen(store, theme.DARK,
                              on_new=lambda: seen.setdefault("new", True),
                              on_open=lambda pid: seen.setdefault("open", pid),
                              on_navigate=lambda k: seen.setdefault("nav", k))
    # two recent rows rendered
    assert home._recent_box.count() == 2


def test_projects_screen_grid_and_search(app, store):
    scr = screens.ProjectsScreen(store, theme.DARK,
                                 on_new=lambda: None, on_open=lambda pid: None)
    # 2 project cards + 1 ghost "new" card
    assert scr._grid.count() == 3
    # search narrows and drops the ghost card
    scr._on_search("alpha")
    kinds = [scr._grid.itemAt(i).widget() for i in range(scr._grid.count())]
    assert len(kinds) == 1
    assert isinstance(kinds[0], screens.ProjectCard)
    assert kinds[0].project.name == "Alpha"


def test_project_card_open_callback(app, store):
    opened = []
    p = store.list()[0]
    card = screens.ProjectCard(p, theme.LIGHT, on_open=opened.append)
    card._on_open(p.id)
    assert opened == [p.id]


# ── window ───────────────────────────────────────────────────────────────────
def test_studio_window_constructs_without_napari(app, store):
    win = app_mod.StudioWindow(store, theme_name="dark")
    assert win._stack.count() >= 2          # home + projects registered
    assert "home" in win._screens and "projects" in win._screens


def test_navigation_switches_active_screen(app, store):
    win = app_mod.StudioWindow(store, theme_name="dark")
    win.navigate("projects")
    assert win._stack.currentWidget() is win._screens["projects"]
    win.navigate("home")
    assert win._stack.currentWidget() is win._screens["home"]


def test_unregistered_screen_gets_placeholder(app, store):
    win = app_mod.StudioWindow(store, theme_name="dark")
    win.navigate("dashboard")               # not registered without main()
    assert "dashboard" in win._screens
    assert win._stack.currentWidget() is win._screens["dashboard"]


def test_new_project_flow_creates_and_opens(app, store):
    win = app_mod.StudioWindow(store, theme_name="dark")
    before = len(store.list())
    win._new_project()
    assert len(store.list()) == before + 1
    # opening navigates to workspace (placeholder here)
    assert win._stack.currentWidget() is win._screens["workspace"]


def test_toggle_theme_rebuilds_chrome(app, store):
    win = app_mod.StudioWindow(store, theme_name="dark")
    assert win._theme_name == "dark"
    win.toggle_theme()
    assert win._theme_name == "light"
    # home/projects still present after rebuild
    assert "home" in win._screens and "projects" in win._screens


def test_load_fonts_returns_a_family(app):
    fam = app_mod.load_fonts()
    assert isinstance(fam, str) and fam


# ── custom window chrome (frameless title bar) ───────────────────────────────
def test_window_is_frameless(app, store):
    from PyQt6.QtCore import Qt
    win = app_mod.StudioWindow(store, theme_name="dark")
    assert win.windowFlags() & Qt.WindowType.FramelessWindowHint


def test_titlebar_has_three_traffic_lights_and_grips(app, store):
    from napari_app.studio import window_chrome
    win = app_mod.StudioWindow(store, theme_name="dark")
    bars = win.findChildren(window_chrome.TitleBar)
    assert len(bars) == 1
    lights = bars[0].findChildren(window_chrome._TrafficButton)
    assert len(lights) == 3            # close / minimise / zoom
    assert len(win._grips) == 4        # four corner resize handles


def test_traffic_close_button_calls_window_close(app, store, monkeypatch):
    from napari_app.studio import window_chrome
    win = app_mod.StudioWindow(store, theme_name="dark")
    called = {"n": 0}
    monkeypatch.setattr(win, "close", lambda: called.__setitem__("n", called["n"] + 1))
    bar = win.findChildren(window_chrome.TitleBar)[0]
    close_btn = bar.findChildren(window_chrome._TrafficButton)[0]
    close_btn.click()
    assert called["n"] == 1


def test_titlebar_rebuilds_on_theme_toggle(app, store):
    from napari_app.studio import window_chrome
    win = app_mod.StudioWindow(store, theme_name="dark")
    win.toggle_theme()
    assert win._theme_name == "light"
    # still exactly one title bar after the rebuild
    assert len(win.findChildren(window_chrome.TitleBar)) == 1
