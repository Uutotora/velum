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


def test_open_sample_step_opens_a_real_existing_project(app, controller):
    opened = []
    g = _guide(app, controller, on_open=opened.append)
    g._run_action("open_sample")
    assert len(opened) == 1
    assert opened[0] in {p.id for p in controller.list_projects()}


def test_open_sample_step_falls_back_to_new_project_when_store_is_empty(app, empty_controller):
    seen = []
    g = _guide(app, empty_controller, on_new_project=lambda: seen.append(True))
    g._run_action("open_sample")
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
