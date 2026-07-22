"""Headless tests for the Guide & Docs screen (studio/guide_screen.py).

Offscreen Qt, no napari/torch. Mirrors the fixture/style conventions in
test_home_wiring.py: a tmp_path-backed ProjectController so nothing here
touches the real data_store.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
guide_screen = pytest.importorskip("studio.guide_screen")

from PyQt6.QtWidgets import QApplication

from studio import theme, guide_content
from studio.components import Accordion
from studio.project import ProjectStore
from studio.project_controller import ProjectController


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(tmp_path):
    return ProjectController(ProjectStore(tmp_path))


@pytest.fixture
def empty_controller(tmp_path):
    return ProjectController(ProjectStore(tmp_path), seed_if_empty=False)


@pytest.fixture
def styled_app(app):
    """The shared QApplication with the real app-wide stylesheet applied.

    ``studio.app.main()`` calls ``app.setStyleSheet(theme.build_qss(...))``
    at startup — the bare-QWidget-inherits-bg hazard (see the tests below)
    only cascades through that *app-wide* rule, so a test using the plain
    ``app`` fixture alone would pass whether or not the code is actually
    fixed. Reset afterward: ``app`` is a process-wide singleton shared with
    every other test module in this session, and alphabetically this file
    runs before test_home_wiring.py / test_motion.py / test_new_project_
    dialog.py — a stylesheet left set here would otherwise leak into them.
    """
    app.setStyleSheet(theme.build_qss(theme.DARK))
    yield app
    app.setStyleSheet("")


def _guide(app, controller, on_navigate=None, on_open=None, on_new_project=None):
    return guide_screen.GuideScreen(
        theme.DARK, controller,
        on_navigate or (lambda k: None),
        on_open or (lambda i: None),
        on_new_project or (lambda: None))


# ── construction / nav ───────────────────────────────────────────────────────
def test_one_nav_row_per_article(app, controller):
    g = _guide(app, controller)
    assert set(g._nav_rows.keys()) == {a.id for a in guide_content.ARTICLES}
    assert set(g._article_pages.keys()) == {a.id for a in guide_content.ARTICLES}


def test_default_article_is_shown_on_construction(app, controller):
    g = _guide(app, controller)
    assert g._current_id == guide_content.DEFAULT_ARTICLE_ID
    assert g._content_stack.currentWidget() is g._article_pages[guide_content.DEFAULT_ARTICLE_ID]


def test_close_button_navigates_home(app, controller):
    navigated = []
    g = _guide(app, controller, on_navigate=navigated.append)
    g._close_btn.click()
    assert navigated == ["home"]


def test_clicking_a_nav_row_opens_that_article(app, controller):
    g = _guide(app, controller)
    g._nav_rows["engines"].mouseReleaseEvent(None)
    assert g._current_id == "engines"
    assert g._content_stack.currentWidget() is g._article_pages["engines"]


def test_open_article_ignores_an_unknown_id(app, controller):
    g = _guide(app, controller)
    g.open_article("does-not-exist")
    assert g._current_id == guide_content.DEFAULT_ARTICLE_ID


# ── search ───────────────────────────────────────────────────────────────────
# isHidden() (this widget's own explicit flag), not isVisible() (also depends
# on a shown top-level ancestor, which nothing here has) — same convention
# test_projects_screen_view_toggle_switches_grid_and_list already relies on.
def test_search_hides_non_matching_rows(app, controller):
    g = _guide(app, controller)
    g._on_search("lora")
    assert not g._nav_rows["training"].isHidden()
    assert g._nav_rows["dashboard"].isHidden()


def test_search_cleared_shows_every_row_again(app, controller):
    g = _guide(app, controller)
    g._on_search("lora")
    g._on_search("")
    assert all(not row.isHidden() for row in g._nav_rows.values())


def test_search_hides_a_category_header_when_nothing_in_it_matches(app, controller):
    g = _guide(app, controller)
    g._on_search("lora")
    assert g._category_headers["Analysis"].isHidden()
    assert not g._category_headers["Training"].isHidden()


# ── Getting Started actions fire real callbacks ───────────────────────────────
def test_new_project_step_fires_the_real_new_project_callback(app, controller):
    seen = []
    g = _guide(app, controller, on_new_project=lambda: seen.append(True))
    g._run_action("new_project")
    assert seen == [True]


def test_plain_nav_key_action_calls_on_navigate(app, controller):
    navigated = []
    g = _guide(app, controller, on_navigate=navigated.append)
    g._run_action("workspace")
    assert navigated == ["workspace"]


def test_article_prefixed_action_opens_that_article_instead_of_navigating(app, controller):
    navigated = []
    g = _guide(app, controller, on_navigate=navigated.append)
    g._run_action("article:engines")
    assert g._current_id == "engines"
    assert navigated == []  # this is an in-guide jump, not a tab switch


def test_run_action_is_a_noop_for_none(app, controller):
    g = _guide(app, controller)
    g._run_action(None)  # must not raise


# ── block renderers ──────────────────────────────────────────────────────────
def test_shortcuts_block_has_one_row_per_shortcut(app):
    w = guide_screen._shortcuts_block(theme.DARK, guide_content.SHORTCUTS)
    assert w.layout().count() == len(guide_content.SHORTCUTS)


def test_bullets_block_has_one_row_per_item(app):
    items = ["first", "second", "third"]
    w = guide_screen._bullets(theme.DARK, items)
    assert w.layout().count() == len(items)


def test_table_block_has_a_header_and_one_row_per_data_row(app):
    headers = ["A", "B"]
    rows = [["1", "2"], ["3", "4"], ["5", "6"]]
    card = guide_screen._table_block(theme.DARK, headers, rows)
    # header row + hline, then (row + hline) per data row
    assert card.layout().count() == 2 + 2 * len(rows)


def test_faq_block_renders_one_accordion_per_item(app, controller):
    g = _guide(app, controller)
    faq_page = g._article_pages["faq"]
    assert len(faq_page.findChildren(Accordion)) == len(guide_content.FAQ)


def test_inline_bold_markup_becomes_html_bold():
    assert guide_screen._inline("a **bold** word") == "a <b>bold</b> word"
    assert guide_screen._inline("no markup here") == "no markup here"


# ── regression: bare QWidget wrappers must not paint an opaque bg patch ──────
# A plain QWidget() with no stylesheet of its own inherits the app-wide
# `QWidget { background: <bg> }` rule (theme.build_qss) and paints an *opaque*
# bg-coloured rectangle wherever it sits. Confirmed directly: it's invisible
# on the page canvas itself, but a real user's screenshot showed it as
# visibly banded/two-tone rows in the engine-comparison table and the
# keyboard-shortcuts list, where these wrappers sit *inside* an
# already-lighter surface2 card. _bare() (used for every plain grouping
# widget in this module) fixes it by setting background:transparent
# explicitly; these tests sample actual rendered pixels so a future re-
# introduction of a raw QWidget() in this file fails a test, not just a
# screenshot.
def test_bare_widget_is_transparent_not_opaque(styled_app):
    from PyQt6.QtWidgets import QLabel, QVBoxLayout

    t = theme.DARK
    card = guide_screen._callout(t, "x")  # any real #ObjectName-scoped card
    # Re-parent a _bare() widget directly onto the callout's own surface so
    # a bug would paint bg (near-black) instead of the callout's primary_weak
    # tint underneath it.
    probe = guide_screen._bare()
    QVBoxLayout(probe).addWidget(QLabel("x"))
    card.layout().addWidget(probe)
    card.resize(240, 160)
    card.show()
    styled_app.processEvents()
    img = card.grab().toImage()
    # Sample just inside probe's left edge, away from the label glyph.
    sample = img.pixelColor(4, card.height() - 8)
    assert sample.name() != t["bg"]


def test_table_block_row_fill_matches_the_card_not_the_page_bg(styled_app):
    from PyQt6.QtCore import Qt as _Qt
    from PyQt6.QtWidgets import QWidget

    t = theme.DARK
    card = guide_screen._table_block(
        t, ["Engine", "Best for", "Setup"],
        [["CellSeg1 · LoRA", "x", "y"], ["Cellpose-SAM", "x", "y"], ["SAM 2", "x", "y"]])
    card.resize(500, 220)
    card.show()
    styled_app.processEvents()
    img = card.grab().toImage()

    # _bare() returns a plain QWidget, not a QFrame -- search broadly.
    rows = [r for r in card.findChildren(QWidget, options=_Qt.FindChildOption.FindChildrenRecursively)
            if r.objectName() == "GuideTableRow"]
    assert len(rows) == 3
    for row in rows:
        # Row centre -- inside the card's own 16px left margin (immune to
        # the bug regardless of a fix, since nothing paints there) is *not*
        # a valid sample point; the row's own horizontal centre is
        # guaranteed to fall within the widget actually under test.
        pt = row.mapTo(card, row.rect().center())
        sample = img.pixelColor(pt.x(), pt.y())
        assert sample.name() == t["surface2"], (
            f"row at {pt} sampled {sample.name()!r}, expected the card's own "
            f"surface2 fill {t['surface2']!r} -- a bare QWidget() row wrapper "
            f"is painting the page bg over it again")


def test_shortcuts_block_keys_area_matches_row_fill_not_page_bg(styled_app):
    from PyQt6.QtCore import Qt as _Qt
    from PyQt6.QtWidgets import QFrame, QVBoxLayout, QWidget

    t = theme.DARK
    # Wrapped in a real, opaque bg-coloured parent -- exactly how this block
    # is actually embedded in the app (an unparented, "transparent" _bare()
    # top-level widget renders its OWN default window background when
    # grabbed standalone, which isn't the thing under test here).
    outer = QFrame()
    outer.setStyleSheet(f"background:{t['bg']};")
    QVBoxLayout(outer).setContentsMargins(0, 0, 0, 0)
    w = guide_screen._shortcuts_block(t, guide_content.SHORTCUTS)
    outer.layout().addWidget(w)
    outer.resize(500, 200)
    outer.show()
    styled_app.processEvents()
    img = outer.grab().toImage()

    # keys_wrap (a _bare() QWidget, not a QFrame) holds the key pills + a
    # trailing stretch. Its *row*'s own left padding (14px) is immune to the
    # bug regardless of a fix (nothing paints there) -- same trap as the
    # table-row test above. Sample near keys_wrap's own right edge instead,
    # past the pills, in the stretch space that's actually part of it.
    keys_wraps = [k for k in w.findChildren(QWidget, options=_Qt.FindChildOption.FindChildrenRecursively)
                  if k.objectName() == "GuideShortcutKeys"]
    assert len(keys_wraps) == len(guide_content.SHORTCUTS)
    for kw in keys_wraps:
        y = kw.mapTo(outer, kw.rect().center()).y()
        x = kw.mapTo(outer, kw.rect().topRight()).x() - 5
        sample = img.pixelColor(x, y)
        assert sample.name() == t["surface2"], (
            f"keys_wrap at ({x},{y}) sampled {sample.name()!r}, expected the "
            f"row's own surface2 fill {t['surface2']!r} -- a bare QWidget() "
            f"wrapper is painting the page bg over it again")
