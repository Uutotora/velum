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
from studio import project_controller
from studio.project import ProjectStore
from studio.project_controller import ProjectController
from studio.train_controller import TrainController


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


@pytest.fixture
def train_controller(tmp_path):
    """A tmp_path-backed TrainController — never the real data_store."""
    return TrainController(storage_dir=tmp_path / "train_storage")


# ── UI kit ───────────────────────────────────────────────────────────────────
def test_ui_atoms_construct(app):
    t = theme.DARK
    assert components.Chip("x", t, "primary") is not None
    assert components.PillButton("Go", t, "primary", "plus").text() == "Go"
    assert components.PillButton("Ghost", t, "ghost").text() == "Ghost"
    assert components.Badge("0.80", t) is not None
    assert components.SelectBox("512 px", t) is not None
    assert components.Stepper(32, t) is not None
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


def test_accordion_caps_false_keeps_a_full_sentence_title_unshouted(app):
    from PyQt6.QtWidgets import QLabel
    acc = components.Accordion("Do I need a GPU?", theme.LIGHT, caps=False)
    titles = [lb.text() for lb in acc.findChildren(QLabel)]
    assert "Do I need a GPU?" in titles
    assert "DO I NEED A GPU?" not in titles


def test_sidebar_navigates(app):
    seen = []
    sb = components.Sidebar(app_mod._NAV, theme.DARK)
    sb.navigate.connect(seen.append)
    sb._items["workspace"].click()
    assert seen == ["workspace"]


def test_sidebar_guide_button_emits_open_guide(app):
    seen = []
    sb = components.Sidebar(app_mod._NAV, theme.DARK)
    sb.open_guide.connect(lambda: seen.append(True))
    guide_buttons = [w for w in sb.findChildren(components._NavItem) if w.key == "__guide__"]
    assert len(guide_buttons) == 1
    guide_buttons[0].click()
    assert seen == [True]


# ── paint ────────────────────────────────────────────────────────────────────
def test_nuclei_pixmap_renders(app):
    px = paint.nuclei_pixmap(120, 90, seed=7)
    assert not px.isNull()
    assert paint.NucleiView(seed=7) is not None


def test_nuclei_pixmap_top_only_radius_rounds_top_not_bottom(app):
    """Regression test: QSS border-radius doesn't clip a QLabel's own pixmap

    (only its background/border), so the rounding has to be baked into the
    pixmap itself. Assert the actual corner pixels: transparent where a
    ``top_only`` round-rect clip should have cut the corner off, opaque where
    the bottom should stay square.
    """
    px = paint.nuclei_pixmap(120, 90, seed=7, radius=14, top_only=True, dpr=1.0)
    img = px.toImage()
    assert img.pixelColor(0, 0).alpha() == 0  # top-left: clipped by the round corner
    assert img.pixelColor(119, 0).alpha() == 0  # top-right: same
    assert img.pixelColor(0, 89).alpha() > 0  # bottom-left: square, not clipped
    assert img.pixelColor(119, 89).alpha() > 0  # bottom-right: square, not clipped


def test_nuclei_pixmap_full_radius_rounds_every_corner(app):
    px = paint.nuclei_pixmap(120, 90, seed=7, radius=14, top_only=False, dpr=1.0)
    img = px.toImage()
    for x, y in [(0, 0), (119, 0), (0, 89), (119, 89)]:
        assert img.pixelColor(x, y).alpha() == 0, f"corner ({x},{y}) should be clipped"


def test_nucleiview_geometry_can_be_synced_to_a_raw_parent(app):
    """Regression test: a NucleiView set as a plain (non-layout) child via

    ``setParent()`` does not automatically track its parent's size — it's
    born with whatever geometry an unparented top-level widget happens to
    get, which has nothing to do with the eventual parent. This is exactly
    the project card's cover-art wiring; assert the pattern it relies on
    (an explicit resize callback) actually keeps the two in sync.
    """
    from PyQt6.QtWidgets import QFrame
    view = paint.NucleiView(seed=1, radius=14, top_only=True, min_size=(0, 0))
    wrap = QFrame()
    view.setParent(wrap)
    view.setGeometry(0, 0, wrap.width(), wrap.height())
    wrap.resizeEvent = lambda e: view.setGeometry(0, 0, wrap.width(), wrap.height())
    wrap.show()  # resizeEvent isn't reliably dispatched before a widget is shown
    for w, h in [(450, 132), (305, 132), (620, 132)]:
        wrap.resize(w, h)
        assert view.size().width() == w and view.size().height() == h


# ── screens ──────────────────────────────────────────────────────────────────
def test_all_screens_construct(app, controller, train_controller, tmp_path):
    from studio.screens import HomeScreen, ProjectsScreen
    from studio.workspace import WorkspaceScreen
    from studio.extra_screens import ModelsScreen, DashboardScreen
    from studio.segment_controller import SegmentController
    t = theme.DARK
    segment_controller = SegmentController(storage_dir=tmp_path / "segment_storage")
    assert HomeScreen(t, controller, lambda k: None, lambda i: None, lambda: None) is not None
    assert ProjectsScreen(t, controller, lambda k: None, lambda i: None, lambda: None) is not None
    assert WorkspaceScreen(t, segment_controller, controller, lambda title, sub: None) is not None
    assert ModelsScreen(t, train_controller, controller, lambda title, sub: None) is not None
    assert DashboardScreen(t, train_controller, controller, lambda title, sub: None) is not None


def test_home_screen_lists_recent_real_projects(app, controller):
    from studio.screens import HomeScreen
    home = HomeScreen(theme.DARK, controller, lambda k: None, lambda i: None, lambda: None)
    assert home._recent_section().layout().count() == 1 + 4  # header + 4 rows


def test_home_screen_handles_empty_store(app, empty_controller):
    from studio.screens import HomeScreen
    home = HomeScreen(theme.DARK, empty_controller, lambda k: None, lambda i: None, lambda: None)
    assert home._recent_section() is not None


def _projects_screen(controller, on_navigate=None, on_open=None, on_new_project=None):
    from studio.screens import ProjectsScreen
    return ProjectsScreen(
        theme.DARK, controller,
        on_navigate or (lambda k: None),
        on_open or (lambda i: None),
        on_new_project or (lambda: None))


def test_projects_screen_grid_shows_seeded_projects_plus_ghost(app, controller):
    scr = _projects_screen(controller)
    assert scr._grid.count() == 6 + 1  # 6 sample projects + the "New Project" ghost
    assert scr._list.count() == 6 + 1  # both views are always kept populated


def test_projects_screen_search_narrows_grid(app, controller):
    scr = _projects_screen(controller)
    scr._on_search("mitosis")
    assert scr._grid.count() == 1 + 1  # one match + ghost
    scr._on_search("")
    assert scr._grid.count() == 6 + 1


def test_projects_screen_search_with_no_matches_shows_empty_message(app, controller):
    scr = _projects_screen(controller)
    scr._on_search("nonexistent-xyz")
    assert scr._grid.count() == 0 + 1  # just the ghost
    assert not scr._empty_label.isHidden()
    assert "nonexistent-xyz" in scr._empty_label.text()


def test_projects_screen_favorites_scope_filter(app, controller):
    scr = _projects_screen(controller)
    scr._on_scope_changed(1)  # "Favorites"
    n_favorites = len(controller.list_projects(favorites_only=True))
    assert scr._grid.count() == n_favorites + 1
    assert scr._empty_label.isHidden()  # the seed set has favourites
    scr._on_scope_changed(0)  # "All"
    assert scr._grid.count() == 6 + 1


def test_projects_screen_shared_scope_is_always_empty(app, controller):
    scr = _projects_screen(controller)
    scr._on_scope_changed(2)  # "Shared"
    assert scr._grid.count() == 0 + 1  # just the ghost — no sharing backend yet
    assert not scr._empty_label.isHidden()
    assert "doesn't support shared projects" in scr._empty_label.text()


def test_projects_screen_card_star_renders_visibly_when_not_favorite(app, controller):
    """Regression test: the non-favourite star colour was an SVG rgba() string,

    which QSvgRenderer silently drops (no error, just an invisible icon) —
    the amber favourited star used a plain hex and was fine, masking this for
    every non-favourited card. Assert the rendered icon actually has visible
    (non-transparent) pixels, not just that construction doesn't raise.
    """
    scr = _projects_screen(controller)
    non_favorite = next(p for p in controller.list_projects() if not p.favorite)
    card = scr._card(project_controller.to_card(non_favorite))
    star = card.findChild(components.IconButton)
    assert star is not None
    image = star.icon().pixmap(star.iconSize()).toImage()
    assert any(
        image.pixelColor(x, y).alpha() > 0
        for x in range(image.width()) for y in range(image.height())
    ), "star icon rendered fully transparent — invalid SVG stroke colour"


def test_projects_screen_engine_filter(app, controller):
    scr = _projects_screen(controller)
    scr._on_engine_toggled("sam2", True)
    n = len(controller.list_projects(engines={"sam2"}))
    assert scr._grid.count() == n + 1
    scr._on_engine_toggled("cellseg1", True)
    n2 = len(controller.list_projects(engines={"sam2", "cellseg1"}))
    assert scr._grid.count() == n2 + 1
    scr._clear_engine_filter()
    assert scr._grid.count() == 6 + 1


def test_projects_screen_filter_menu_lists_every_engine(app, controller):
    scr = _projects_screen(controller)
    scr._open_filter_menu()
    assert [a.text() for a in scr._filter_menu.actions()] == [
        "CellSeg1 · LoRA", "Cellpose-SAM", "SAM 2"]
    scr._filter_menu.close()


def test_projects_screen_view_toggle_switches_grid_and_list(app, controller):
    scr = _projects_screen(controller)
    assert scr._view == "grid"
    assert not scr._grid_host.isHidden()
    assert scr._list_host.isHidden()
    scr._on_view_changed(1)
    assert scr._view == "list"
    assert scr._grid_host.isHidden()
    assert not scr._list_host.isHidden()


def test_projects_screen_toolbar_controls_share_one_height(app, controller):
    """Regression test: the search box, the All/Favorites/Shared segmented

    control, the Filter button and the grid/list toggle each computed their
    own height from padding + font metrics, and drifted apart under the
    real bundled Figtree font even though they matched in an offscreen dev
    run — exactly the kind of per-widget drift a shared, explicit height is
    supposed to rule out regardless of font/platform.
    """
    scr = _projects_screen(controller)
    H = scr._TOOLBAR_H
    assert scr._scope_seg.height() == H
    assert scr._filter_btn.height() == H
    assert scr._view_seg.height() == H


def test_projects_screen_engine_chip_has_the_right_dot_per_engine(app, controller):
    from PyQt6.QtWidgets import QFrame
    from studio.screens import _ENGINE_DOT
    p = next(pr for pr in controller.list_projects() if pr.engine == "sam2")
    card = _projects_screen(controller)._card(project_controller.to_card(p))
    chip = card.findChild(components.EngineChip)
    assert chip is not None
    dot = chip.findChildren(QFrame)[0]  # the dot is the chip's one QFrame child
    assert _ENGINE_DOT["sam2"] in dot.styleSheet()


def test_projects_screen_open_callback_uses_project_id(app, controller):
    opened = []
    scr = _projects_screen(controller, on_open=opened.append)
    first = controller.list_projects()[0]
    scr._open(first.id)
    assert opened == [first.id]


def test_projects_screen_star_toggles_favorite(app, controller):
    scr = _projects_screen(controller)
    target = controller.list_projects(favorites_only=True)[0]
    assert target.favorite
    scr._toggle_favorite(target.id)
    assert not controller.store.load(target.id).favorite


def test_projects_screen_list_row_opens_project(app, controller):
    scr = _projects_screen(controller)
    p = controller.list_projects()[0]
    row = scr._list_row(project_controller.to_card(p))
    opened = []
    scr._open = opened.append
    row.mouseReleaseEvent(None)
    assert opened == [p.id]


def test_projects_screen_new_project_cta_and_ghosts_open_dialog(app, controller):
    seen = []
    scr = _projects_screen(controller, on_new_project=lambda: seen.append(True))
    scr._header_widget.findChild(components.PillButton).click()
    scr._ghost().mouseReleaseEvent(None)
    scr._ghost_row().mouseReleaseEvent(None)
    assert seen == [True, True, True]


# ── window ───────────────────────────────────────────────────────────────────
def test_window_is_frameless_with_titlebar_and_grips(app, controller, train_controller):
    from studio import window_chrome
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller)
    assert win.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert len(win.findChildren(window_chrome.TitleBar)) == 1
    assert len(win._grips) == 4


def test_window_constructs_without_napari_or_store(app, controller, train_controller):
    # If this imported napari/torch, the light CI job would fail — it must not.
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller)
    assert win._stack.count() == len(app_mod._STACK_KEYS)


def test_navigation_switches_stack_screens(app, controller, train_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller)
    win.navigate("dashboard")
    assert win._stack.currentWidget() is win._screens["dashboard"]
    win.navigate("workspace")
    assert win._stack.currentWidget() is win._screens["workspace"]


def test_sidebars_guide_and_docs_row_opens_the_guide_screen(app, controller, train_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller)
    win._sidebar.open_guide.emit()
    assert win._stack.currentWidget() is win._screens["guide"]


def test_navigate_guide_colon_id_deep_links_into_a_specific_article(app, controller, train_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller)
    win.navigate("guide:engines")
    assert win._stack.currentWidget() is win._screens["guide"]
    assert win._screens["guide"]._current_id == "engines"


def test_opening_a_project_sets_active_and_updates_workspace(app, controller, train_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller)
    project = controller.list_projects()[0]
    win._open_project(project.id)
    assert win._stack.currentWidget() is win._screens["workspace"]
    assert controller.get_active().id == project.id
    assert project.name in win._screens["workspace"]._crumb_name.text()


def test_active_project_survives_theme_toggle(app, controller, train_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller)
    project = controller.list_projects()[0]
    win._open_project(project.id)
    win.toggle_theme()
    assert project.name in win._screens["workspace"]._crumb_name.text()


# ── regression: a project created elsewhere must show up immediately ──────────
# (Home/Projects are built once and kept alive across navigation -- the stack
# just swaps pages -- so without an explicit refresh, a project created via
# the New Project dialog wouldn't appear until the whole app was restarted.)
def test_navigate_refreshes_home_and_projects_screens(app, empty_controller, train_controller):
    from PyQt6.QtWidgets import QFrame
    win = app_mod.StudioWindow(theme_name="dark", project_controller=empty_controller,
                                train_controller=train_controller)

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


def test_creating_a_project_via_dialog_shows_up_immediately(app, empty_controller, train_controller):
    """End-to-end regression for the exact bug reported: create -> navigate
    away and back -> the new project is there without restarting the app."""
    from PyQt6.QtWidgets import QFrame
    win = app_mod.StudioWindow(theme_name="dark", project_controller=empty_controller,
                                train_controller=train_controller)

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


def test_assistant_and_logs_toggle_as_overlays(app, controller, train_controller):
    # isHidden() is the explicit flag; isVisible() needs the top-level shown.
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller)
    assert win._assistant.isHidden()
    win.navigate("assistant")
    assert not win._assistant.isHidden()
    win.navigate("assistant")
    assert win._assistant.isHidden()
    win.navigate("logs")
    assert not win._logs.isHidden()


def test_command_palette_opens_and_escape_closes(app, controller, train_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller)
    win._toggle_palette()
    assert not win._palette.isHidden()
    win._close_overlays()
    assert win._palette.isHidden()


def test_theme_toggle_rebuilds(app, controller, train_controller):
    from studio import window_chrome
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller)
    win.toggle_theme()
    assert win._theme_name == "light"
    assert win._stack.count() == len(app_mod._STACK_KEYS)
    assert len(win.findChildren(window_chrome.TitleBar)) == 1


def test_load_fonts_returns_family(app):
    assert isinstance(app_mod.load_fonts(), str) and app_mod.load_fonts()


# ── cross-tab: a Segment-tab run must show up on Dashboard, same session ──────
def test_predicting_in_workspace_shows_up_on_the_dashboard(
        app, empty_controller, train_controller, tmp_path, monkeypatch):
    """End-to-end proof, not just per-controller: run a real segmentation in
    the Segment tab, switch to Dashboard with no restart, and see it —
    exactly the "everything ... logs to the dashboard" path a real user
    session exercises, sharing one ProjectController instance the way
    StudioWindow actually wires the two screens together."""
    import time

    import cv2
    import numpy as np

    from studio.project import ProjectSettings
    from studio.segment_controller import SegmentController

    def _fake_predict_cached(config, image_rgb):
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint16)
        mask[h // 8: h // 2, w // 8: w // 2] = 1
        mask[h // 2: h - h // 8, w // 2: w - w // 8] = 2
        return mask

    monkeypatch.setattr("napari_app.inference_cache.predict_cached", _fake_predict_cached)
    monkeypatch.setattr("napari_app.engines.cellpose_available", lambda: False)

    storage = tmp_path / "storage"
    (storage / "sam_backbone").mkdir(parents=True)
    (storage / "sam_backbone" / "sam_vit_h_4b8939.pth").write_bytes(b"fake")
    (storage / "loras").mkdir(parents=True)
    lora = storage / "loras" / "nuclei-dapi-r8.pth"
    lora.write_bytes(b"fake-lora")

    img_path = tmp_path / "img_0.png"
    rng = np.random.default_rng(0)
    cv2.imwrite(str(img_path), (rng.random((48, 48, 3)) * 255).astype(np.uint8))

    project = empty_controller.store.create(
        "Integration Project", "x", image_paths=[str(img_path)],
        settings=ProjectSettings(engine="cellseg1", model_name=str(lora)))

    segment = SegmentController(storage_dir=storage)
    win = app_mod.StudioWindow(theme_name="dark", project_controller=empty_controller,
                                train_controller=train_controller, segment_controller=segment)
    win._open_project(project.id)

    ws = win._screens["workspace"]
    ws._start_predict()
    t0 = time.monotonic()
    while ws._predicting and time.monotonic() - t0 < 10:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(5):
        app.processEvents()
    assert ws._last_result is not None  # the run itself must have actually succeeded

    win.navigate("dashboard")
    assert win._stack.currentWidget() is win._screens["dashboard"]
    rows = win._screens["dashboard"]._dashboard.runs_table()
    row = next((r for r in rows if r.name == "Integration Project"), None)
    assert row is not None, f"project missing from dashboard runs_table(): {[r.name for r in rows]}"
    assert row.cells is not None
