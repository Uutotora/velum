"""Headless wiring tests for studio/overlays.py's LogsConsole and
CommandPalette -- the real, live log stream (studio/log_bus.py) and the
real, live action registry (studio/command_registry.py), not static demo
content. Offscreen Qt only; no napari/torch.

Every test constructs its own private LogBus (never the real
log_bus.get_log_bus() singleton) for isolation, the same convention
test_project_controller.py/test_train_controller.py use tmp_path stores for.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
overlays = pytest.importorskip("studio.overlays")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from studio import theme
from studio.command_registry import Command
from studio.log_bus import LogBus


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def styled_app(app):
    """The shared QApplication with the real app-wide stylesheet applied --
    the bare-QWidget-inherits-bg hazard (see the CommandPalette test below)
    only cascades through that *app-wide* rule, so a test using the plain
    ``app`` fixture alone would pass whether or not the code is actually
    fixed. Reset afterward since ``app`` is a process-wide singleton shared
    with every other test module in this session (same convention as
    test_guide_screen.py's own ``styled_app`` fixture)."""
    app.setStyleSheet(theme.build_qss(theme.LIGHT))
    yield app
    app.setStyleSheet("")


@pytest.fixture
def parent(app):
    w = QWidget()
    w.resize(1200, 900)
    w.show()
    return w


@pytest.fixture
def bus():
    return LogBus()


def _console(parent, bus):
    c = overlays.LogsConsole(parent, theme.DARK, bus=bus)
    c.setParent(parent)
    c.resize(760, overlays.LogsConsole.HEIGHT)
    c.show()
    QApplication.processEvents()
    return c


# ── construction / backfill ──────────────────────────────────────────────────
def test_uses_the_injected_bus_not_the_global_singleton(parent, bus):
    from studio.log_bus import get_log_bus
    console = _console(parent, bus)
    assert console._bus is bus
    assert console._bus is not get_log_bus()


def test_backfills_from_bus_history_at_construction(parent, bus):
    bus.info("already here", source="segment")
    console = _console(parent, bus)
    assert "already here" in console._text.toPlainText()
    assert console._badge.text() == "1"


def test_starts_hidden(parent, bus):
    c = overlays.LogsConsole(parent, theme.DARK, bus=bus)
    assert c.isHidden()


# ── live updates ─────────────────────────────────────────────────────────────
def test_new_records_appear_live(parent, bus):
    console = _console(parent, bus)
    bus.info("fresh line", source="segment")
    assert "fresh line" in console._text.toPlainText()
    assert console._badge.text() == "1"


def test_badge_counts_errors_and_warnings(parent, bus):
    console = _console(parent, bus)
    bus.warning("careful")
    bus.error("boom")
    bus.error("boom again")
    assert console._badge.text() == "3 · 2 err · 1 warn"


def test_badge_omits_zero_counts(parent, bus):
    console = _console(parent, bus)
    bus.info("all quiet")
    assert console._badge.text() == "1"


# ── level filter ─────────────────────────────────────────────────────────────
def test_default_filter_hides_debug_but_shows_info(parent, bus):
    bus.debug("noisy breadcrumb")
    bus.info("normal line")
    console = _console(parent, bus)
    text = console._text.toPlainText()
    assert "normal line" in text
    assert "noisy breadcrumb" not in text


def test_level_filter_hides_lines_below_threshold(parent, bus):
    console = _console(parent, bus)
    bus.info("plain info")
    bus.error("a real problem")
    console._on_level_selected("Error")
    text = console._text.toPlainText()
    assert "a real problem" in text
    assert "plain info" not in text
    console._on_level_selected("All")
    text = console._text.toPlainText()
    assert "plain info" in text and "a real problem" in text


def test_level_filter_applies_to_new_live_records_too(parent, bus):
    console = _console(parent, bus)
    console._on_level_selected("Error")
    bus.info("should stay hidden")
    bus.error("should show up")
    text = console._text.toPlainText()
    assert "should show up" in text
    assert "should stay hidden" not in text


# ── text search ──────────────────────────────────────────────────────────────
def test_search_filters_by_message_and_source(parent, bus):
    console = _console(parent, bus)
    bus.info("loading model checkpoint", source="studio.segment")
    bus.info("baking tuned agent", source="studio.assistant")
    console._search.setText("checkpoint")
    text = console._text.toPlainText()
    assert "loading model checkpoint" in text
    assert "baking tuned agent" not in text
    console._search.setText("assistant")
    text = console._text.toPlainText()
    assert "baking tuned agent" in text
    assert "loading model checkpoint" not in text


def test_clearing_the_search_box_restores_everything(parent, bus):
    console = _console(parent, bus)
    bus.info("alpha")
    bus.info("beta")
    console._search.setText("alpha")
    assert "beta" not in console._text.toPlainText()
    console._search.setText("")
    text = console._text.toPlainText()
    assert "alpha" in text and "beta" in text


# ── autoscroll ───────────────────────────────────────────────────────────────
def test_autoscroll_on_by_default_snaps_to_bottom_on_new_lines(parent, bus):
    console = _console(parent, bus)
    assert console._autoscroll.is_on()
    for i in range(60):
        bus.info(f"line {i}")
    sb = console._text.verticalScrollBar()
    assert sb.maximum() > 0
    assert sb.value() == sb.maximum()


def test_autoscroll_off_does_not_move_the_scrollbar(parent, bus):
    console = _console(parent, bus)
    console._autoscroll.set_on(False)
    for i in range(60):
        bus.info(f"line {i}")
    sb = console._text.verticalScrollBar()
    assert sb.maximum() > 0
    sb.setValue(0)
    bus.info("one more after manually scrolling up")
    assert sb.value() == 0


def test_toggling_autoscroll_back_on_jumps_to_bottom_immediately(parent, bus):
    console = _console(parent, bus)
    for i in range(60):
        bus.info(f"line {i}")
    sb = console._text.verticalScrollBar()
    sb.setValue(0)
    console._on_autoscroll_toggled(True)
    assert sb.value() == sb.maximum()


# ── clear ────────────────────────────────────────────────────────────────────
def test_clear_empties_the_view_and_the_bus(parent, bus):
    console = _console(parent, bus)
    bus.info("one")
    bus.error("two")
    console._on_clear()
    assert console._text.toPlainText().strip() == ""
    assert console._badge.text() == "0"
    assert bus.snapshot() == []


# ── export ───────────────────────────────────────────────────────────────────
def test_export_writes_the_currently_filtered_lines(parent, bus, tmp_path, monkeypatch):
    console = _console(parent, bus)
    bus.info("excluded info line")
    bus.error("included error line")
    console._on_level_selected("Error")
    out = tmp_path / "out.txt"
    monkeypatch.setattr(overlays.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    console._export()
    content = out.read_text()
    assert "included error line" in content
    assert "excluded info line" not in content  # excluded by the active level filter


def test_export_cancelled_writes_nothing(parent, bus, tmp_path, monkeypatch):
    console = _console(parent, bus)
    bus.info("keep me")
    out = tmp_path / "should_not_exist.txt"
    monkeypatch.setattr(overlays.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    console._export()  # must not raise
    assert not out.exists()


# ── cross-thread / teardown safety ──────────────────────────────────────────
def test_safe_emit_record_guards_against_a_deleted_widget(parent, bus):
    from PyQt6 import sip
    console = _console(parent, bus)
    emit = console._safe_emit_record
    rec = bus.info("queued before teardown")
    sip.delete(console)
    emit(rec)  # must not raise


def test_destroyed_widget_unsubscribes_from_the_bus(parent, bus):
    from PyQt6 import sip
    console = _console(parent, bus)
    assert len(bus._subscribers) == 1
    sip.delete(console)
    assert len(bus._subscribers) == 0


def test_a_record_emitted_from_a_worker_thread_still_reaches_the_console(parent, bus):
    import threading
    console = _console(parent, bus)
    t = threading.Thread(target=lambda: bus.info("from a worker thread", source="train"))
    t.start()
    t.join()
    QApplication.processEvents()
    assert "from a worker thread" in console._text.toPlainText()


# ── place() geometry (unchanged contract test_app_wiring.py relies on) ──────
def test_place_anchors_to_the_bottom_spanning_the_remaining_width(parent, bus):
    from studio.components import Sidebar
    console = overlays.LogsConsole(parent, theme.DARK, bus=bus)
    console.setParent(parent)
    console.place()
    geom = console.geometry()
    assert geom.x() == Sidebar.WIDTH
    assert geom.height() == overlays.LogsConsole.HEIGHT
    assert geom.y() == parent.height() - overlays.LogsConsole.HEIGHT
    assert geom.width() == parent.width() - Sidebar.WIDTH


# ── CommandPalette -- a real, live action registry, not a static list ───────
def _pump(app, timeout=2):
    import time
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        app.processEvents()


def _commands(calls):
    return [
        Command(id="run", label="Run segmentation", section="Segment", icon="run", emoji="▶️",
                handler=lambda: calls.append("run")),
        Command(id="batch", label="Run batch prediction", section="Segment", icon="batch",
                handler=lambda: calls.append("batch")),
        Command(id="disabled", label="Stop training", section="Models & Train", icon="close",
                handler=lambda: calls.append("disabled"), enabled=False),
        Command(id="home", label="Go to Home", section="Navigate", icon="home",
                handler=lambda: calls.append("home")),
    ]


def _palette(parent, calls):
    p = overlays.CommandPalette(parent, theme.DARK, get_commands=lambda: _commands(calls))
    p.setParent(parent)
    return p


# ── Raycast-style redesign: no ESC chip, emoji rows, a dynamic footer ──────
def test_no_esc_chip_anywhere_in_the_input_row(parent, app):
    """Direct feedback: the ESC chip in the search row was unwanted --
    closing on Escape is a universal enough convention not to need a
    permanent on-screen reminder (Raycast itself doesn't show one)."""
    calls = []
    palette = _palette(parent, calls)
    input_row = palette.findChild(QWidget, "PaletteInputRow")
    labels = [w.text() for w in input_row.findChildren(QLabel)]
    assert "ESC" not in labels


def test_row_shows_the_command_emoji_when_present(parent, app):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    run_row = next(r for r in palette._rows if r.cmd.id == "run")
    assert run_row._icon.text() == "▶️"


def test_row_falls_back_to_the_icon_pixmap_without_an_emoji(parent, app):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    batch_row = next(r for r in palette._rows if r.cmd.id == "batch")
    assert batch_row._icon.text() == ""
    assert not batch_row._icon.pixmap().isNull()


def test_footer_shows_the_selected_commands_own_label_and_enter_hint(parent, app):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    assert palette._selected == 0
    assert palette._foot_action_lbl.text() == "Run segmentation"
    assert palette._foot_hint_lbl.text() == "⏎"

    palette._move_selection(1)
    assert palette._foot_action_lbl.text() == "Run batch prediction"


def test_footer_omits_the_enter_hint_for_a_disabled_selected_command(parent, app):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    disabled_idx = next(i for i, c in enumerate(palette._visible) if c.id == "disabled")
    palette._selected = disabled_idx
    palette._apply_selection_styles()
    assert palette._foot_action_lbl.text() == "Stop training"
    assert palette._foot_hint_lbl.text() == ""


def test_footer_clears_when_nothing_matches(parent, app):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    palette.input.setText("xyzxyz")
    assert palette._foot_action_lbl.text() == ""
    assert palette._foot_hint_lbl.text() == ""


def test_open_builds_rows_grouped_by_section_with_empty_query(parent, app):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    assert [c.label for c in palette._visible] == [
        "Run segmentation", "Run batch prediction", "Stop training", "Go to Home"]
    # a section header widget precedes each section's first row -- more
    # layout items than commands proves the headers are really there
    assert palette._results_layout.count() > len(palette._visible)


def test_typing_filters_to_a_flat_ranked_list_no_headers(parent, app):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    palette.input.setText("run")
    assert [c.label for c in palette._visible] == ["Run segmentation", "Run batch prediction"]
    # flat: exactly one layout item per visible row, no section headers, no
    # trailing stretch either -- the container sizes exactly to content
    # (see _BoundedScrollArea) rather than needing an internal stretch to
    # top-anchor a short list within an otherwise fixed-height box.
    assert palette._results_layout.count() == len(palette._visible)


def test_no_matching_query_shows_empty_state(parent, app):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    palette.input.setText("xyzxyz")
    assert palette._visible == []
    labels = [w.text() for w in palette._results_container.findChildren(QLabel)]
    assert any("No matching commands" in t for t in labels)


def test_arrow_keys_move_selection_and_wrap(parent, app):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    assert palette._selected == 0
    palette._move_selection(1)
    assert palette._selected == 1
    palette._move_selection(-1)
    assert palette._selected == 0
    palette._move_selection(-1)  # wraps to the last row
    assert palette._selected == len(palette._visible) - 1


def test_selection_resets_to_zero_when_the_query_changes(parent, app):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    palette._move_selection(1)
    assert palette._selected == 1
    palette.input.setText("run")
    assert palette._selected == 0


def test_enter_activates_the_selected_command(app, parent):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    palette.input.setText("run batch")
    assert palette._visible[0].id == "batch"
    palette._activate_selected()
    _pump(app)
    assert calls == ["batch"]


def test_clicking_a_row_activates_that_command_directly(app, parent):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtCore import QPointF
    row = palette._rows[1]   # "Run batch prediction" at empty-query position 1
    assert row.cmd.id == "batch"
    event = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(5, 5),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    row.mousePressEvent(event)
    _pump(app)
    assert calls == ["batch"]


def test_activating_a_disabled_command_does_nothing(app, parent):
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    disabled_idx = next(i for i, c in enumerate(palette._visible) if c.id == "disabled")
    palette._selected = disabled_idx
    palette._activate_selected()
    _pump(app)
    assert calls == []


def test_running_a_command_hides_the_palette_but_only_after_the_event_loop_turns(app, parent):
    """The hide+run is deferred (QTimer.singleShot(0, ...)) -- the same
    established fix as the documented sipBadCatcherResult hazard elsewhere
    in Studio, since a handler can itself rebuild the very screen the
    palette sits over. Confirm it's genuinely deferred, not accidentally
    synchronous."""
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    palette.input.setText("run segmentation")
    palette._activate_selected()
    assert not palette.isHidden()   # not yet -- still queued
    assert calls == []
    _pump(app)
    assert palette.isHidden()
    assert calls == ["run"]


def test_open_rebuilds_commands_fresh_and_resets_query_and_selection(app, parent):
    calls = []
    state = {"enabled": True}
    palette = overlays.CommandPalette(
        parent, theme.DARK,
        get_commands=lambda: [Command(id="x", label="X", section="S", enabled=state["enabled"])])
    palette.setParent(parent)
    palette.open()
    palette.input.setText("leftover query")
    palette._move_selection(1)
    state["enabled"] = False
    palette.open()   # reopening must re-fetch commands and reset state
    assert palette.input.text() == ""
    assert palette._selected == 0
    assert palette._visible[0].enabled is False


def test_results_area_shrinks_to_fit_a_short_list(parent, app):
    """Regression: the results area used to always reserve the full
    _MAX_RESULTS_HEIGHT box regardless of how few results were showing --
    a short list (a narrow search, or a fresh install with no project yet)
    rendered as a couple of rows sitting in a mostly-empty box, reading as
    broken/bloated. It must now size to its *actual* content instead."""
    calls = []
    palette = _palette(parent, calls)
    palette.open()
    palette.input.setText("run segmentation")  # narrows to exactly one row
    _pump(app)
    assert len(palette._visible) == 1
    height = palette._results_area.sizeHint().height()
    assert 0 < height < overlays.CommandPalette._MAX_RESULTS_HEIGHT


def test_results_area_caps_at_max_height_for_a_long_list(parent, app):
    many = [Command(id=f"c{i}", label=f"Command number {i}", section="Section")
            for i in range(60)]
    palette = overlays.CommandPalette(parent, theme.DARK, get_commands=lambda: many)
    palette.setParent(parent)
    palette.open()
    _pump(app)
    assert palette._results_area.sizeHint().height() == overlays.CommandPalette._MAX_RESULTS_HEIGHT


def test_panel_shrinks_and_grows_across_searches_not_stuck_at_one_size(parent, app):
    """The actual end-to-end regression: not just the results area's own
    sizeHint, but the *panel*'s real on-screen geometry must follow it --
    caught by an actual screenshot, not by any test passing, since a plain
    QFrame's default vertical size policy (Preferred) happily fills
    whatever space a layout offers regardless of alignment flags unless
    something else (a trailing stretch) absorbs the extra space instead."""
    from PyQt6.QtWidgets import QFrame

    calls = []
    palette = _palette(parent, calls)
    palette.open()
    _pump(app)
    full_height = palette._panel.height()

    palette.input.setText("batch")   # narrows _commands()'s 4 rows down to 1
    _pump(app)
    short_height = palette._panel.height()

    palette.input.setText("")
    _pump(app)
    restored_height = palette._panel.height()

    assert short_height < full_height
    assert restored_height == full_height


def test_input_and_footer_rows_match_the_panel_surface_not_the_page_bg(styled_app, parent):
    """Regression: inp_wrap/foot used to be plain QWidget()s, which inherit
    the app-wide QWidget{background:<bg>} rule and paint an opaque
    <bg>-coloured rectangle over their own children -- invisible against the
    near-identical dark tones of the dark theme but a glaring flat-grey
    patch in light theme (bg #f4f6f8 vs. this panel's own surface #ffffff),
    the same "bare QWidget() wrapper" family docs/velum/CHANGELOG.md's
    2026-07-09 entry already found and fixed elsewhere. CommandPalette was
    still 100% static content at the time and never got a real screenshot
    pass, so this instance went undiscovered until the palette actually
    rendered live content (caught by an actual light-theme screenshot, not
    by any test passing -- this test pins it after the fact)."""
    calls = []
    light_t = theme.LIGHT
    palette = overlays.CommandPalette(parent, light_t, get_commands=lambda: _commands(calls))
    palette.setParent(parent)
    palette.open()
    styled_app.processEvents()
    img = palette.grab().toImage()

    for name in ("PaletteInputRow", "PaletteFootRow"):
        row = palette.findChild(QWidget, name)
        assert row is not None, f"{name} not found"
        # Both rows' own layout has a 17px/10-15px margin before their first
        # child starts -- the row's *centre* falls on a child (the QLineEdit
        # spans most of the input row's width via a stretch factor), so it
        # isn't a valid sample point regardless of a fix, the same trap
        # test_guide_screen.py's own row-fill test already documents. A
        # corner inside that margin band is guaranteed to be painted only by
        # the row wrapper itself.
        pt = row.mapTo(palette, row.rect().topLeft() + QPoint(4, 4))
        sample = img.pixelColor(pt.x(), pt.y())
        assert sample.name() == light_t["surface"], (
            f"{name} margin-corner sampled {sample.name()!r}, expected the panel's "
            f"own surface fill {light_t['surface']!r} -- a bare QWidget() row "
            f"wrapper is painting the page bg over it again")


# ── Toast ────────────────────────────────────────────────────────────────────
def test_toast_announce_sets_text_and_shows(app, parent):
    toast = overlays.Toast(parent, theme.DARK)
    assert toast.isHidden()
    toast.announce("Project deleted", "“A” was permanently deleted.")
    assert toast.isVisible()
    assert toast._title.text() == "Project deleted"
    assert "permanently deleted" in toast._subtitle.text()


def test_toast_long_subtitle_stays_within_the_cards_own_height(app, parent):
    """Regression test: the subtitle QLabel used setMaximumWidth(280), not
    setFixedWidth -- with nothing else in the chain anchoring a width (the
    whole Toast frame's own size is itself computed via adjustSize()), a
    word-wrapping label's natural width for Qt's heightForWidth negotiation
    settled at an arbitrary, too-narrow value (measured against a real
    toast before the fix: 137px, not the intended 280px cap), undershooting
    the height the wrapped text actually needed and painting outside the
    card's own rounded background -- a real bug reported from the actual
    running app, not offscreen. A subtitle long enough to need 3 lines at
    137px width (comfortably fewer at a real 280px) reproduces it directly.
    """
    toast = overlays.Toast(parent, theme.DARK)
    toast.announce("Project deleted",
                   "“Тестовый проект copy” was permanently deleted.")
    assert toast._subtitle.width() == 280
    sub_bottom = toast._subtitle.geometry().y() + toast._subtitle.geometry().height()
    assert sub_bottom <= toast.height(), (
        f"subtitle bottom edge {sub_bottom} exceeds the toast's own height {toast.height()}")
