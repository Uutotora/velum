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


@pytest.fixture
def assistant_controller(tmp_path):
    """A tmp_path-backed AssistantController — never the real data_store
    (which would leak a real API key/backend choice into test state)."""
    from studio.assistant_controller import AssistantController
    return AssistantController(storage_dir=tmp_path / "assistant_storage")


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


def test_accordion_default_fill_is_unchanged(app):
    # Every existing call site was designed/screenshotted against "inset" —
    # this must never move without every one of those being re-verified.
    acc = components.Accordion("Ground truth", theme.DARK)
    assert f"background:{theme.DARK['inset']}" in acc.styleSheet()


def test_accordion_custom_fill_overrides_the_default(app):
    acc = components.Accordion("Model", theme.DARK, fill="surface2")
    assert f"background:{theme.DARK['surface2']}" in acc.styleSheet()
    assert theme.DARK["inset"] not in acc.styleSheet()


def test_accordion_opening_fades_the_body_in_closing_does_not(app):
    from PyQt6.QtWidgets import QGraphicsOpacityEffect
    acc = components.Accordion("Ground truth", theme.LIGHT, open_=False)
    assert acc._body.graphicsEffect() is None
    acc.toggle()   # open
    assert acc._open is True
    assert isinstance(acc._body.graphicsEffect(), QGraphicsOpacityEffect)
    acc.toggle()   # close — no animation needed to disappear
    assert acc._open is False


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


def test_projects_screen_sort_box_defaults_to_last_modified(app, controller):
    scr = _projects_screen(controller)
    assert scr._sort == "modified"
    assert scr._sort_box.text() == "Last modified"


def test_projects_screen_sort_by_name_reorders_the_grid(app, controller):
    """The card at each grid position must follow the chosen sort order --
    checks the actual built grid, not just the controller call underneath
    it (that algorithm is already covered in test_project_controller.py).
    """
    scr = _projects_screen(controller)
    scr._on_sort_changed("Name (A–Z)")
    assert scr._sort == "name"

    expected_names = [p.name for p in controller.list_projects(sort="name")]
    shown_names = []
    for i in range(len(expected_names)):
        card = scr._grid.itemAtPosition(i // 3, i % 3).widget()
        # card's own layout: [0]=header row (engine chip/star/more), [1]=a
        # spacing item, [2]=the name label (_card()'s construction order) --
        # verified directly against a real card before trusting this here.
        name_label = card.layout().itemAt(2).widget()
        shown_names.append(name_label.text())
    assert shown_names == expected_names


def test_projects_screen_sort_box_options_match_the_controller(app, controller):
    scr = _projects_screen(controller)
    assert scr._sort_box._options == list(project_controller.ProjectController.SORT_OPTIONS.keys())


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
    assert scr._sort_box.height() == H
    assert scr._view_seg.height() == H


# ── kebab menu / settings flow ───────────────────────────────────────────────
def test_projects_screen_card_has_exactly_one_more_button(app, controller):
    scr = _projects_screen(controller)
    p = controller.list_projects()[0]
    card = scr._card(project_controller.to_card(p))
    more_buttons = [b for b in card.findChildren(components.IconButton) if b.toolTip() == "More"]
    assert len(more_buttons) == 1


def test_projects_screen_list_row_has_exactly_one_more_button(app, controller):
    scr = _projects_screen(controller)
    p = controller.list_projects()[0]
    row = scr._list_row(project_controller.to_card(p))
    more_buttons = [b for b in row.findChildren(components.IconButton) if b.toolTip() == "More"]
    assert len(more_buttons) == 1


def test_projects_screen_card_menu_lists_every_action(app, controller):
    """Deliberately short -- Open · Duplicate · Settings -- matching Label
    Studio's own minimal card overflow menu (Settings / Label) rather than
    listing rename/delete separately here; both live in Settings now."""
    from PyQt6.QtWidgets import QToolButton
    scr = _projects_screen(controller)
    p = controller.list_projects()[0]
    scr._open_card_menu(QToolButton(), p.id, p.name)
    labels = [a.text() for a in scr._card_menu.actions() if not a.isSeparator()]
    assert labels == ["Open", "Duplicate", "Settings"]
    scr._card_menu.close()


def test_projects_screen_duplicate_adds_a_project_and_toasts(app, controller):
    scr = _projects_screen(controller)
    p = controller.list_projects()[0]
    before = len(controller.list_projects())
    seen_toasts = []
    scr._toast = lambda *a, **kw: seen_toasts.append((a, kw))

    scr._duplicate(p.id)

    assert len(controller.list_projects()) == before + 1
    assert seen_toasts and seen_toasts[0][0][0] == "Project duplicated"


def test_projects_screen_open_settings_prefills_the_dialog(app, controller):
    from studio.project_dialogs import ProjectSettingsDialog
    scr = _projects_screen(controller)
    p = controller.list_projects()[0]

    scr._open_settings(p.id)

    dlg = next(w for w in scr.findChildren(ProjectSettingsDialog) if not w.isHidden())
    assert dlg._name_input.text() == p.name
    assert dlg._desc_input.text() == p.description


def test_projects_screen_save_settings_renames_and_updates_description(app, controller):
    scr = _projects_screen(controller)
    p = controller.list_projects()[0]

    scr._save_settings(p.id, "Renamed Project", "New description")

    reloaded = controller.store.load(p.id)
    assert reloaded.name == "Renamed Project"
    assert reloaded.description == "New description"


def test_projects_screen_settings_dialog_save_flows_through_to_the_store(app, controller):
    """End to end through the real dialog, not just _save_settings() -- the
    dialog's own Save button is the actual user-facing path."""
    from studio.project_dialogs import ProjectSettingsDialog
    scr = _projects_screen(controller)
    p = controller.list_projects()[0]

    scr._open_settings(p.id)
    dlg = next(w for w in scr.findChildren(ProjectSettingsDialog) if not w.isHidden())
    dlg._name_input.setText("Via Dialog")
    dlg._save_general()

    assert controller.store.load(p.id).name == "Via Dialog"


def test_projects_screen_delete_project_removes_it_and_toasts(app, controller):
    scr = _projects_screen(controller)
    p = controller.list_projects()[0]
    seen_toasts = []
    scr._toast = lambda *a, **kw: seen_toasts.append((a, kw))

    scr._delete_project(p.id, p.name)

    assert p.id not in [x.id for x in controller.list_projects()]
    assert not controller.store.exists(p.id)
    assert len(seen_toasts) == 1
    title, subtitle = seen_toasts[0][0]
    assert title == "Project deleted"
    assert p.name in subtitle


def test_projects_screen_delete_project_without_a_toast_callback_does_not_raise(app, controller):
    """on_toast is optional (None by default, e.g. in the bare
    _projects_screen() test helper) -- deleting must not assume it exists."""
    scr = _projects_screen(controller)
    p = controller.list_projects()[0]
    assert scr._toast is None
    scr._delete_project(p.id, p.name)  # must not raise
    assert p.id not in [x.id for x in controller.list_projects()]


def test_projects_screen_settings_delete_requires_its_own_confirm(app, controller):
    """Settings' "Delete Project" must not act immediately -- it opens its
    own nested ConfirmDialog first, same as ProjectSettingsDialog's own
    unit test, but exercised here through the screen's real wiring."""
    from studio.project_dialogs import ConfirmDialog, ProjectSettingsDialog
    scr = _projects_screen(controller)
    p = controller.list_projects()[0]

    scr._open_settings(p.id)
    settings_dlg = next(w for w in scr.findChildren(ProjectSettingsDialog) if not w.isHidden())
    settings_dlg._confirm_delete()

    assert controller.store.exists(p.id)  # not deleted yet
    nested = next(w for w in settings_dlg.findChildren(ConfirmDialog) if not w.isHidden())
    nested._confirm()
    assert not controller.store.exists(p.id)


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
def test_window_is_frameless_with_titlebar_and_grips(app, controller, train_controller, assistant_controller):
    from studio import window_chrome
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    assert win.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert len(win.findChildren(window_chrome.TitleBar)) == 1
    assert len(win._grips) == 4


def test_window_constructs_without_napari_or_store(app, controller, train_controller, assistant_controller):
    # If this imported napari/torch, the light CI job would fail — it must not.
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    assert win._stack.count() == len(app_mod._STACK_KEYS)


def test_navigation_switches_stack_screens(app, controller, train_controller, assistant_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    win.navigate("dashboard")
    assert win._stack.currentWidget() is win._screens["dashboard"]
    win.navigate("workspace")
    assert win._stack.currentWidget() is win._screens["workspace"]


def test_sidebars_guide_and_docs_row_opens_the_guide_screen(app, controller, train_controller, assistant_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    win._sidebar.open_guide.emit()
    assert win._stack.currentWidget() is win._screens["guide"]


def test_navigate_guide_colon_id_deep_links_into_a_specific_article(app, controller, train_controller, assistant_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    win.navigate("guide:engines")
    assert win._stack.currentWidget() is win._screens["guide"]
    assert win._screens["guide"]._current_id == "engines"


def test_opening_a_project_sets_active_and_updates_workspace(app, controller, train_controller, assistant_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    project = controller.list_projects()[0]
    win._open_project(project.id)
    assert win._stack.currentWidget() is win._screens["workspace"]
    assert controller.get_active().id == project.id
    assert project.name in win._screens["workspace"]._crumb_name.text()


def test_active_project_survives_theme_toggle(app, controller, train_controller, assistant_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    project = controller.list_projects()[0]
    win._open_project(project.id)
    win.toggle_theme()
    assert project.name in win._screens["workspace"]._crumb_name.text()


# ── regression: a project created elsewhere must show up immediately ──────────
# (Home/Projects are built once and kept alive across navigation -- the stack
# just swaps pages -- so without an explicit refresh, a project created via
# the New Project dialog wouldn't appear until the whole app was restarted.)
def test_navigate_refreshes_home_and_projects_screens(app, empty_controller, train_controller, assistant_controller):
    from PyQt6.QtWidgets import QFrame
    win = app_mod.StudioWindow(theme_name="dark", project_controller=empty_controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)

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


def test_navigate_home_skips_the_whole_screen_fade_other_screens_still_get_one(
        app, empty_controller, train_controller, assistant_controller):
    """Regression test, same root cause as screens.py's own
    test_refresh_fades_only_the_recent_section_not_the_whole_home_screen:
    navigate() used to fade_in() *every* non-workspace screen on *every*
    single visit, Home included -- expensive given how many
    QGraphicsDropShadowEffect-bearing cards Home carries, and, replaying on
    every revisit rather than just the first, reported directly as a
    "terrible animation" each time. "home" is now excluded from the generic
    fade alongside the pre-existing "workspace" exclusion; every other
    screen (e.g. "projects") must still get one -- this isn't fade_in
    breaking globally, just Home opting out in favour of its own scoped
    motion (see HomeScreen.refresh())."""
    from PyQt6.QtWidgets import QGraphicsOpacityEffect
    win = app_mod.StudioWindow(theme_name="dark", project_controller=empty_controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    win.navigate("home")
    assert win._screens["home"].graphicsEffect() is None
    win.navigate("projects")
    assert isinstance(win._screens["projects"].graphicsEffect(), QGraphicsOpacityEffect)


def test_new_project_dialog_scopes_to_the_stack_not_the_whole_window(
        app, empty_controller, train_controller, assistant_controller):
    """Regression test: NewProjectDialog used to be parented directly to
    the whole StudioWindow, so its scrim covered (and its panel centred
    against) the sidebar too, not just the content area -- reported
    directly against a screenshot ("окно надо центровать не по всему
    приложению а по центру области с эмодзи" -- centre the window not
    against the whole app but against the centre of the [content] area):
    the panel ended up noticeably left-of-centre relative to the actual
    page content behind it, with the sidebar fully undimmed the whole time
    despite the scrim technically covering it too. Fixed by parenting
    NewProjectDialog to win._stack (exactly the content area to the right
    of the sidebar) instead of the window -- matching how
    ConfirmDialog/ProjectSettingsDialog already behave, each parented to
    its own screen, itself always exactly the stack's bounds."""
    win = app_mod.StudioWindow(theme_name="dark", project_controller=empty_controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    assert win._new_project_dialog.parentWidget() is win._stack
    win.navigate("workspace")
    win._new_project_dialog.open()
    stack_top_left_in_window = win._stack.mapTo(win, win._stack.rect().topLeft())
    dialog_top_left_in_window = win._new_project_dialog.mapTo(win, win._new_project_dialog.rect().topLeft())
    assert dialog_top_left_in_window == stack_top_left_in_window
    assert win._new_project_dialog.width() == win._stack.width()
    assert win._new_project_dialog.width() < win.width(), (
        "dialog must be narrower than the full window -- it should exclude the sidebar")


def test_creating_a_project_via_dialog_shows_up_immediately(app, empty_controller, train_controller, assistant_controller):
    """End-to-end regression for the exact bug reported: create -> navigate
    away and back -> the new project is there without restarting the app."""
    from PyQt6.QtWidgets import QFrame
    win = app_mod.StudioWindow(theme_name="dark", project_controller=empty_controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)

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


def test_assistant_and_logs_toggle_as_overlays(app, controller, train_controller, assistant_controller):
    # isHidden() is the explicit flag; isVisible() needs the top-level shown.
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    assert win._assistant.isHidden()
    win.navigate("assistant")
    assert not win._assistant.isHidden()
    win.navigate("assistant")
    assert win._assistant.isHidden()
    win.navigate("logs")
    assert not win._logs.isHidden()


def test_opening_the_assistant_slides_in_from_the_right_not_an_instant_pop(
        app, controller, train_controller, assistant_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    win.navigate("assistant")
    anim = win._assistant._slide_anim
    assert anim.startValue().x() == anim.endValue().x() + win._assistant.WIDTH
    assert anim.startValue().y() == anim.endValue().y()   # right edge only, no vertical drift
    # QPropertyAnimation.start() applies t=0 (== startValue) synchronously —
    # proves this is a real, currently-mid-flight animation (starting
    # off-screen) rather than an instant teleport to the final position.
    assert win._assistant.geometry() == anim.startValue()


def test_opening_logs_slides_up_from_the_bottom(app, controller, train_controller, assistant_controller):
    from studio.overlays import LogsConsole
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    win.navigate("logs")
    anim = win._logs._slide_anim
    assert anim.startValue().y() == anim.endValue().y() + LogsConsole.HEIGHT
    assert anim.startValue().x() == anim.endValue().x()
    assert win._logs.geometry() == anim.startValue()


def test_command_palette_opens_and_escape_closes(app, controller, train_controller, assistant_controller):
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    win._toggle_palette()
    assert not win._palette.isHidden()
    win._close_overlays()
    assert win._palette.isHidden()


def test_ctrl_t_and_meta_t_shortcuts_toggle_the_assistant(app, controller, train_controller, assistant_controller):
    from PyQt6.QtGui import QShortcut, QKeySequence
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    for seq in ("Ctrl+T", "Meta+T"):
        matches = [s for s in win.findChildren(QShortcut) if s.key() == QKeySequence(seq)]
        assert len(matches) == 1, f"no registered shortcut for {seq!r}"

    ctrl_t = next(s for s in win.findChildren(QShortcut) if s.key() == QKeySequence("Ctrl+T"))
    assert win._assistant.isHidden()
    ctrl_t.activated.emit()
    assert not win._assistant.isHidden()
    ctrl_t.activated.emit()
    assert win._assistant.isHidden()


def test_ctrl_l_and_meta_l_shortcuts_toggle_logs(app, controller, train_controller, assistant_controller):
    from PyQt6.QtGui import QShortcut, QKeySequence
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    for seq in ("Ctrl+L", "Meta+L"):
        matches = [s for s in win.findChildren(QShortcut) if s.key() == QKeySequence(seq)]
        assert len(matches) == 1, f"no registered shortcut for {seq!r}"

    ctrl_l = next(s for s in win.findChildren(QShortcut) if s.key() == QKeySequence("Ctrl+L"))
    assert win._logs.isHidden()
    ctrl_l.activated.emit()
    assert not win._logs.isHidden()
    ctrl_l.activated.emit()
    assert win._logs.isHidden()


def test_theme_toggle_rebuilds(app, controller, train_controller, assistant_controller):
    from studio import window_chrome
    win = app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller)
    win.toggle_theme()
    assert win._theme_name == "light"
    assert win._stack.count() == len(app_mod._STACK_KEYS)
    assert len(win.findChildren(window_chrome.TitleBar)) == 1


def test_load_fonts_returns_family(app):
    assert isinstance(app_mod.load_fonts(), str) and app_mod.load_fonts()


def test_load_icon_loads_the_bundled_app_icon(app):
    """The Dock tile for the running app (macOS shows QApplication's
    windowIcon there for an unbundled process -- see main()'s own comment).
    Not a null/fallback QIcon, and exposes its real 1024x1024 source so Qt
    can generate crisp icons at whatever size the OS actually requests."""
    from PyQt6.QtCore import QSize
    icon = app_mod.load_icon()
    assert not icon.isNull()
    assert QSize(1024, 1024) in icon.availableSizes()


# ── cross-tab: a Segment-tab run must show up on Dashboard, same session ──────
def test_predicting_in_workspace_shows_up_on_the_dashboard(
        app, empty_controller, train_controller, assistant_controller, tmp_path, monkeypatch):
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

    monkeypatch.setattr("velum_core.inference_cache.predict_cached", _fake_predict_cached)
    monkeypatch.setattr("velum_core.engines.cellpose_available", lambda: False)

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
                                train_controller=train_controller, segment_controller=segment,
                                assistant_controller=assistant_controller)
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


# ── real logging (studio/log_bus.py) ────────────────────────────────────────
def test_studio_window_installs_the_log_handler_and_logs_reach_the_bus(
        app, controller, train_controller, assistant_controller):
    """StudioWindow.__init__ -- every construction path, real app or test --
    must wire the stdlib logging bridge, so an ordinary
    logging.getLogger(...).info(...) call anywhere in the process reaches
    the same LogBus the Logs console reads."""
    import logging
    from studio.log_bus import StudioLogHandler, get_log_bus

    app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                          train_controller=train_controller,
                          assistant_controller=assistant_controller)
    assert any(isinstance(h, StudioLogHandler) for h in logging.getLogger().handlers)
    logging.getLogger("studio.probe").info("probe line")
    assert any(r.message == "probe line" for r in get_log_bus().snapshot())


def test_uncaught_exception_hook_logs_a_critical_record():
    """_install_exception_hook must keep printing the traceback (unchanged)
    *and* put a real CRITICAL entry on the bus -- a crash shouldn't only be
    discoverable by whoever had a terminal open behind the app."""
    import logging
    import sys
    from studio.log_bus import LogBus, StudioLogHandler, install_handler

    bus = LogBus()
    app_logger = logging.getLogger("studio.app")
    install_handler(bus, logger=app_logger)
    added = next(h for h in app_logger.handlers if isinstance(h, StudioLogHandler) and h.bus is bus)
    original_hook = sys.excepthook
    app_mod._install_exception_hook()
    try:
        try:
            raise ValueError("boom-from-test")
        except ValueError:
            sys.excepthook(*sys.exc_info())
    finally:
        sys.excepthook = original_hook
        app_logger.removeHandler(added)
    assert any("boom-from-test" in r.message and r.level >= logging.CRITICAL
               for r in bus.snapshot())


# ── ⌘K command registry (studio/command_registry.py) ────────────────────────
def _win(controller, train_controller, assistant_controller, **kw):
    return app_mod.StudioWindow(theme_name="dark", project_controller=controller,
                                train_controller=train_controller,
                                assistant_controller=assistant_controller, **kw)


def test_build_commands_navigate_section_mirrors_nav_plus_guide(
        app, empty_controller, train_controller, assistant_controller):
    win = _win(empty_controller, train_controller, assistant_controller)
    nav_labels = {f"Go to {label}" for _key, _icon, label, _section in app_mod._NAV}
    nav_labels.add("Go to Guide & Docs")
    cmds = win._build_commands()
    got = {c.label for c in cmds if c.section == "Navigate"}
    assert got == nav_labels


def test_build_commands_navigate_hints_carry_the_real_shortcuts(
        app, empty_controller, train_controller, assistant_controller):
    win = _win(empty_controller, train_controller, assistant_controller)
    cmds = win._build_commands()
    by_label = {c.label: c.hint for c in cmds if c.section == "Navigate"}
    assert by_label["Go to Assistant"] == "⌘T"
    assert by_label["Go to Logs"] == "⌘L"
    assert by_label["Go to Home"] == ""


def test_build_commands_segment_actions_disabled_without_a_project(
        app, empty_controller, train_controller, assistant_controller):
    win = _win(empty_controller, train_controller, assistant_controller)
    cmds = win._build_commands()
    segment_ids = {"segment.run", "segment.batch", "segment.benchmark", "segment.save", "segment.export_csv"}
    by_id = {c.id: c for c in cmds if c.id in segment_ids}
    assert set(by_id) == segment_ids
    assert all(not c.enabled for c in by_id.values())
    # nothing to switch engine/preset *to* without a current project
    assert not [c for c in cmds if c.id.startswith("segment.engine.")]
    assert not [c for c in cmds if c.id.startswith("segment.preset.")]


def test_build_commands_segment_actions_enabled_with_a_project(
        app, empty_controller, train_controller, assistant_controller, tmp_path):
    from studio.project import ProjectSettings
    from studio.segment_controller import SegmentController

    storage = tmp_path / "storage"
    (storage / "loras").mkdir(parents=True)
    lora = storage / "loras" / "nuclei-dapi-r8.pth"
    lora.write_bytes(b"fake")
    project = empty_controller.store.create(
        "P", "x", image_paths=[], settings=ProjectSettings(engine="cellseg1", model_name=str(lora)))

    win = _win(empty_controller, train_controller, assistant_controller,
               segment_controller=SegmentController(storage_dir=storage))
    win._open_project(project.id)
    cmds = win._build_commands()
    segment_ids = {"segment.run", "segment.batch", "segment.benchmark", "segment.save", "segment.export_csv"}
    assert all(c.enabled for c in cmds if c.id in segment_ids)
    # cellseg1 is current -> only *other* available engines are offered
    engine_labels = {c.label for c in cmds if c.id.startswith("segment.engine.")}
    assert "Switch engine → CellSeg1 · LoRA" not in engine_labels
    assert all(label.startswith("Switch engine → ") for label in engine_labels)
    # "Balanced" is the default preset -> only Fast/Accurate are offered
    preset_labels = {c.label for c in cmds if c.id.startswith("segment.preset.")}
    assert preset_labels == {"Apply preset → Fast", "Apply preset → Accurate"}


def test_build_commands_train_start_stop_reflect_is_training(
        app, empty_controller, train_controller, assistant_controller, monkeypatch):
    win = _win(empty_controller, train_controller, assistant_controller)
    monkeypatch.setattr(train_controller, "is_training", lambda: False)
    cmds = {c.id: c for c in win._build_commands()}
    assert cmds["train.start"].enabled is True
    assert cmds["train.stop"].enabled is False

    monkeypatch.setattr(train_controller, "is_training", lambda: True)
    cmds = {c.id: c for c in win._build_commands()}
    assert cmds["train.start"].enabled is False
    assert cmds["train.stop"].enabled is True


def test_build_commands_assistant_backend_excludes_the_current_one(
        app, empty_controller, train_controller, assistant_controller):
    win = _win(empty_controller, train_controller, assistant_controller)
    assert assistant_controller.settings.backend == "offline"
    labels = {c.label for c in win._build_commands() if c.id.startswith("assistant.backend.")}
    assert labels == {"Switch Assistant backend → Ollama", "Switch Assistant backend → Custom API"}


def test_build_commands_appearance_names_the_other_theme(
        app, empty_controller, train_controller, assistant_controller):
    win = _win(empty_controller, train_controller, assistant_controller)
    cmds = {c.id: c for c in win._build_commands()}
    assert cmds["appearance.theme"].label == "Switch to Light theme"
    win._theme_name = "light"
    cmds = {c.id: c for c in win._build_commands()}
    assert cmds["appearance.theme"].label == "Switch to Dark theme"


def test_ctrl_k_opens_a_really_populated_palette(
        app, empty_controller, train_controller, assistant_controller):
    win = _win(empty_controller, train_controller, assistant_controller)
    win._toggle_palette()
    assert not win._palette.isHidden()
    assert len(win._palette._commands) > 20
    win._toggle_palette()
    assert win._palette.isHidden()


def _pump(app, timeout=0.5):
    # Lets a deferred QTimer.singleShot(0, ...) callback (CommandPalette._trigger's
    # established sipBadCatcherResult-safe deferral) actually fire -- paired
    # sleep+processEvents, the same convention test_workspace.py's/
    # test_assistant_panel.py's own _pump helpers already use for exactly
    # this kind of wait.
    import time
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)


def test_running_a_navigate_command_through_the_real_palette_switches_tabs(
        app, empty_controller, train_controller, assistant_controller):
    win = _win(empty_controller, train_controller, assistant_controller)
    win._toggle_palette()
    win._palette.input.setText("go to models")
    assert win._palette._visible[0].id == "nav.train"
    win._palette._activate_selected()
    _pump(app)
    assert win._stack.currentWidget() is win._screens["train"]
    assert win._palette.isHidden()


def test_running_the_theme_command_through_the_real_palette_toggles_theme(
        app, empty_controller, train_controller, assistant_controller):
    win = _win(empty_controller, train_controller, assistant_controller)
    win._toggle_palette()
    win._palette.input.setText("switch to light theme")
    assert win._palette._visible and win._palette._visible[0].id == "appearance.theme"
    win._palette._activate_selected()
    _pump(app)
    assert win._theme_name == "light"


def test_running_run_segmentation_through_the_real_palette_predicts_for_real(
        app, empty_controller, train_controller, assistant_controller, tmp_path, monkeypatch):
    """The flagship proof: type into the real ⌘K box, hit the equivalent of
    Enter, and a real prediction happens through the exact production call
    chain -- not a mocked handler standing in for the palette's own wiring."""
    import cv2
    import numpy as np

    from studio.project import ProjectSettings
    from studio.segment_controller import SegmentController

    def _fake_predict_cached(config, image_rgb):
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint16)
        mask[h // 8: h // 2, w // 8: w // 2] = 1
        return mask

    monkeypatch.setattr("velum_core.inference_cache.predict_cached", _fake_predict_cached)
    monkeypatch.setattr("velum_core.engines.cellpose_available", lambda: False)

    storage = tmp_path / "storage"
    (storage / "sam_backbone").mkdir(parents=True)
    (storage / "sam_backbone" / "sam_vit_h_4b8939.pth").write_bytes(b"fake")
    (storage / "loras").mkdir(parents=True)
    lora = storage / "loras" / "nuclei-dapi-r8.pth"
    lora.write_bytes(b"fake")
    img = tmp_path / "img.png"
    cv2.imwrite(str(img), (np.random.rand(48, 48, 3) * 255).astype(np.uint8))
    project = empty_controller.store.create(
        "P", "x", image_paths=[str(img)],
        settings=ProjectSettings(engine="cellseg1", model_name=str(lora)))

    win = _win(empty_controller, train_controller, assistant_controller,
               segment_controller=SegmentController(storage_dir=storage))
    win._open_project(project.id)

    win._toggle_palette()
    win._palette.input.setText("run segmentation")
    assert win._palette._visible[0].id == "segment.run"
    win._palette._activate_selected()

    ws = win._screens["workspace"]
    _pump(app)
    import time
    t0 = time.monotonic()
    while ws._predicting and time.monotonic() - t0 < 5:
        app.processEvents()
        time.sleep(0.01)
    _pump(app, timeout=0.5)
    assert ws._last_result is not None
    assert ws._last_result.get("n_cells", 0) >= 1
