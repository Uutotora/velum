"""Headless wiring/smoke tests for the Studio shell.

Constructs the whole app (sidebar, all screens, overlays, frameless rounded
window) under ``QT_QPA_PLATFORM=offscreen`` with **no napari and no torch**.
Home/Projects are backed by a real ``ProjectController`` pointed at a
``tmp_path`` store — never the real ``data_store/projects`` — so these tests
stay hermetic. Skipped in the GUI-less CI job.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
components = pytest.importorskip("studio.components")
app_mod = pytest.importorskip("studio.app")
paint = pytest.importorskip("studio.paint")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from studio import theme
from studio.project import ProjectStore
from studio.project_controller import ProjectController


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(tmp_path):
    """A seeded (6 sample projects), tmp_path-backed controller."""
    return ProjectController(ProjectStore(tmp_path))


@pytest.fixture
def empty_controller(tmp_path):
    """A controller with no sample seeding, for empty-state assertions."""
    return ProjectController(ProjectStore(tmp_path), seed_if_empty=False)


# ── UI kit ───────────────────────────────────────────────────────────────────
def test_ui_atoms_construct(app):
    t = theme.DARK
    assert components.Chip("x", t, "primary") is not None
    assert components.PillButton("Go", t, "primary", "plus").text() == "Go"
    assert components.PillButton("Ghost", t, "ghost").text() == "Ghost"
    assert components.Badge("0.80", t) is not None
    assert components.SelectBox("512 px", t) is not None
    assert components.Stepper("32", t) is not None
    assert components.StatTile("25.5", "px", "MEDIAN", t) is not None


def test_toggle_flips_state(app):
    tg = components.Toggle(theme.DARK, on=False)
    assert not tg.is_on()
    tg.set_on(True)
    assert tg.is_on()


def test_segcontrol_selection_emits(app):
    seen = []
    seg = components.SegControl(["A", "B", "C"], theme.DARK, active=0)
    seg.changed.connect(seen.append)
    seg._select(2)
    assert seen == [2]
    assert seg._btns[2].isChecked() and not seg._btns[0].isChecked()


def test_accordion_toggles(app):
    acc = components.Accordion("Ground truth", theme.LIGHT, open_=False)
    assert not acc._body.isVisible()
    acc.toggle()
    assert acc._open


def test_sidebar_navigates(app):
    seen = []
    sb = components.Sidebar(app_mod._NAV, theme.DARK)
    sb.navigate.connect(seen.append)
    sb._items["workspace"].click()
    assert seen == ["workspace"]


# ── paint ────────────────────────────────────────────────────────────────────
def test_nuclei_pixmap_renders(app):
    px = paint.nuclei_pixmap(120, 90, seed=7)
    assert not px.isNull()
    assert paint.NucleiView(seed=7) is not None


# ── screens ──────────────────────────────────────────────────────────────────
def test_all_screens_construct(app, controller):
    from studio.screens import HomeScreen, ProjectsScreen
    from studio.workspace import WorkspaceScreen
    from studio.extra_screens import ModelsScreen, DashboardScreen
    t = theme.DARK
    assert HomeScreen(t, controller, lambda k: None, lambda i: None, lambda: None) is not None
    assert ProjectsScreen(t, controller, lambda k: None, lambda i: None) is not None
    assert WorkspaceScreen(t) is not None
    assert ModelsScreen(t) is not None
    assert DashboardScreen(t) is not None


def test_home_screen_lists_recent_real_projects(app, controller):
    from studio.screens import HomeScreen
    home = HomeScreen(theme.DARK, controller, lambda k: None, lambda i: None, lambda: None)
    assert home._recent_section().layout().count() == 1 + 4  # header + 4 rows


def test_home_screen_handles_empty_store(app, empty_controller):
    from studio.screens import HomeScreen
    home = HomeScreen(theme.DARK, empty_controller, lambda k: None, lambda i: None, lambda: None)
    assert home._recent_section() is not None


def test_projects_screen_grid_shows_seeded_projects_plus_ghost(app, controller):
    from studio.screens import ProjectsScreen
    scr = ProjectsScreen(theme.DARK, controller, lambda k: None, lambda i: None)
    assert scr._grid.count() == 6 + 1  # 6 sample projects + the "New Project" ghost


def test_projects_screen_search_narrows_grid(app, controller):
    from studio.screens import ProjectsScreen
    scr = ProjectsScreen(theme.DARK, controller, lambda k: None, lambda i: None)
    scr._on_search("mitosis")
    assert scr._grid.count() == 1 + 1  # one match + ghost
    scr._on_search("")
    assert scr._grid.count() == 6 + 1


def test_projects_screen_favorites_filter(app, controller):
    from studio.screens import ProjectsScreen
    scr = ProjectsScreen(theme.DARK, controller, lambda k: None, lambda i: None)
    scr._on_toggle_filter()
    n_favorites = len(controller.list_projects(favorites_only=True))
    assert scr._grid.count() == n_favorites + 1
    scr._on_toggle_filter()
    assert scr._grid.count() == 6 + 1


def test_projects_screen_open_callback_uses_project_id(app, controller):
    from studio.screens import ProjectsScreen
    opened = []
    scr = ProjectsScreen(theme.DARK, controller, lambda k: None, opened.append)
    first = controller.list_projects()[0]
    scr._open(first.id)
    assert opened == [first.id]


def test_projects_screen_star_toggles_favorite(app, controller):
    from studio.screens import ProjectsScreen
    scr = ProjectsScreen(theme.DARK, controller, lambda k: None, lambda i: None)
    target = controller.list_projects(favorites_only=True)[0]
    assert target.favorite
    scr._toggle_favorite(target.id)
    assert not controller.store.load(target.id).favorite


# ── window ───────────────────────────────────────────────────────────────────
def test_window_is_frameless_with_titlebar_and_grips(app, controller):
    from studio import window_chrome
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller)
    assert win.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert len(win.findChildren(window_chrome.TitleBar)) == 1
    assert len(win._grips) == 4


def test_window_constructs_without_napari_or_store(app, controller):
    # If this imported napari/torch, the light CI job would fail — it must not.
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller)
    assert win._stack.count() == len(app_mod._STACK_KEYS)


def test_navigation_switches_stack_screens(app, controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller)
    win.navigate("dashboard")
    assert win._stack.currentWidget() is win._screens["dashboard"]
    win.navigate("workspace")
    assert win._stack.currentWidget() is win._screens["workspace"]


def test_opening_a_project_sets_active_and_updates_workspace(app, controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller)
    project = controller.list_projects()[0]
    win._open_project(project.id)
    assert win._stack.currentWidget() is win._screens["workspace"]
    assert controller.get_active().id == project.id
    assert project.name in win._screens["workspace"]._crumb.text()


def test_active_project_survives_theme_toggle(app, controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller)
    project = controller.list_projects()[0]
    win._open_project(project.id)
    win.toggle_theme()
    assert project.name in win._screens["workspace"]._crumb.text()


# ── regression: a project created elsewhere must show up immediately ──────────
# (Home/Projects are built once and kept alive across navigation -- the stack
# just swaps pages -- so without an explicit refresh, a project created via
# the New Project dialog wouldn't appear until the whole app was restarted.)
def test_navigate_refreshes_home_and_projects_screens(app, empty_controller):
    from PyQt6.QtWidgets import QFrame
    win = app_mod.StudioWindow(theme_name="dark", project_controller=empty_controller)

    def rrows():
        return [f for f in win._screens["home"].findChildren(QFrame) if f.objectName() == "RRow"]

    def pcards():
        return [f for f in win._screens["projects"].findChildren(QFrame) if f.objectName() == "PCard"]

    assert rrows() == []
    assert pcards() == []  # only the ghost "New Project" card, which isn't a PCard

    empty_controller.store.create("Late Arrival")
    win.navigate("home")
    assert len(rrows()) == 1
    win.navigate("projects")
    assert len(pcards()) == 1


def test_creating_a_project_via_dialog_shows_up_immediately(app, empty_controller):
    """End-to-end regression for the exact bug reported: create -> navigate
    away and back -> the new project is there without restarting the app."""
    from PyQt6.QtWidgets import QFrame
    win = app_mod.StudioWindow(theme_name="dark", project_controller=empty_controller)

    dlg = win._new_project_dialog
    dlg.open()
    dlg._set_name("Freshly Created")
    dlg._go_next()
    dlg._go_next()
    dlg._go_next()  # creates + navigates to workspace

    win.navigate("home")
    home_rows = [f for f in win._screens["home"].findChildren(QFrame) if f.objectName() == "RRow"]
    assert len(home_rows) == 1

    win.navigate("projects")
    project_cards = [f for f in win._screens["projects"].findChildren(QFrame) if f.objectName() == "PCard"]
    assert len(project_cards) == 1


def test_assistant_and_logs_toggle_as_overlays(app, controller):
    # isHidden() is the explicit flag; isVisible() needs the top-level shown.
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller)
    assert win._assistant.isHidden()
    win.navigate("assistant")
    assert not win._assistant.isHidden()
    win.navigate("assistant")
    assert win._assistant.isHidden()
    win.navigate("logs")
    assert not win._logs.isHidden()


def test_command_palette_opens_and_escape_closes(app, controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller)
    win._toggle_palette()
    assert not win._palette.isHidden()
    win._close_overlays()
    assert win._palette.isHidden()


def test_theme_toggle_rebuilds(app, controller):
    from studio import window_chrome
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller)
    win.toggle_theme()
    assert win._theme_name == "light"
    assert win._stack.count() == len(app_mod._STACK_KEYS)
    assert len(win.findChildren(window_chrome.TitleBar)) == 1


def test_load_fonts_returns_family(app):
    assert isinstance(app_mod.load_fonts(), str) and app_mod.load_fonts()
