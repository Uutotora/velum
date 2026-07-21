"""Velum — overlay surfaces: Logs console, command palette, toast.

Created as children of the main window; the window shows/hides and positions
them. ``LogsConsole`` is a real, live view onto ``studio.log_bus`` — every
tab's actual operational log lines (segmentation runs, training, the
Assistant's backend/connection events, app startup/crashes), not a static
``demo`` transcript — with a level filter, text search, autoscroll, clear,
and export to a file. ``CommandPalette`` is a real, live action registry
(``studio.command_registry``) with fuzzy search and full keyboard
navigation — every tab's real actions, not a static 6-item demo list; the
registry itself (which commands exist, whether each is enabled right now)
is built by ``studio.app.StudioWindow._build_commands()``, passed in as the
``get_commands`` callback. The Assistant drawer (real chat, real
diagnostics, real model management) has grown into its own module,
``studio/assistant_panel.py`` — imported from there, not here; see its
docstring.
"""
from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Optional

from typing import Callable

from PyQt6.QtCore import Qt, QEvent, QSize, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QLineEdit,
    QTextEdit, QFileDialog, QScrollArea, QGraphicsOpacityEffect,
)

from studio import icons
from studio import theme
from studio.components import IconButton, Badge, SelectBox, Toggle, bare_widget, hline, label
from studio.command_registry import Command, search, group_by_section
from studio.log_bus import LogBus, LogRecord, get_log_bus, DEBUG, INFO, WARNING, ERROR, short_source


def _level_color(t: dict, rec: LogRecord) -> str:
    """The line's level colour -- status tokens for warn/error (an outcome),
    a plain/muted ink for debug/info (not "Primary hue = interactive only,"
    per DESIGN.md's rule 3), except a `on_log` success line (the existing
    `✓ ...` convention used throughout the reused ML core) reads as `success`
    even though it's technically INFO -- the console would otherwise render
    "247 cells found" in the same flat tone as routine progress chatter.
    """
    if rec.level >= ERROR:
        return t["danger"]
    if rec.level >= WARNING:
        return t["warning"]
    if rec.level >= INFO:
        return t["success"] if rec.message.startswith("✓") else t["text_subtle"]
    return t["text_muted"]


class LogsConsole(QFrame):
    """Bottom console: a real, live stream from the shared :class:`LogBus`.

    Backfills whatever the bus already holds at construction (so opening
    Logs after a background run finished still shows it), then stays live
    for as long as the widget exists. A ``QTextEdit`` rather than one
    ``QLabel`` per line (the original static version's approach) — the
    professional choice once the stream is unbounded instead of 7 fixed
    demo lines, and matches the classic app's own ``widgets/log_window.py``.
    """

    HEIGHT = 210
    _record_sig = pyqtSignal(object)

    _LEVEL_OPTIONS = ("All", "Debug", "Info", "Warn", "Error")
    _LEVEL_THRESHOLD = {"All": 0, "Debug": DEBUG, "Info": INFO, "Warn": WARNING, "Error": ERROR}

    def __init__(self, parent: QWidget, t: dict, bus: Optional[LogBus] = None):
        super().__init__(parent)
        self._t = t
        # Deliberately `is None`, not `bus or get_log_bus()` -- see
        # log_bus.install_handler's own comment: LogBus defines __len__, so
        # a freshly-constructed empty bus is falsy and a plain `or` would
        # silently discard an intentionally-injected (e.g. test) bus.
        self._bus = bus if bus is not None else get_log_bus()
        self._threshold = INFO
        self._records: list[LogRecord] = []
        self.setFixedHeight(self.HEIGHT)
        # Qualified selector: an unqualified background+border rule here
        # would cascade to every descendant that doesn't more specifically
        # override `border` (bare QWidget/QLabel have no such override) --
        # the exact rendering-bug family already found/fixed repeatedly
        # elsewhere in Studio (see AssistantDrawer's own comment).
        self.setObjectName("LogsConsole")
        self.setStyleSheet(
            f"QFrame#LogsConsole{{background:{t['surface']}; border-top:1px solid {t['border']};}}")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        head = QWidget()
        head.setStyleSheet(f"background:{t['inset']};")
        hr = QHBoxLayout(head)
        hr.setContentsMargins(14, 8, 12, 8)
        hr.setSpacing(8)
        hr.addWidget(label("LOGS", 11.5, t["text_subtle"], 600, 0.6))
        self._badge = Badge("0", t)
        hr.addWidget(self._badge)
        hr.addStretch(1)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter…")
        self._search.setFixedWidth(140)
        self._search.setStyleSheet(
            f"QLineEdit{{background:{t['surface']}; border:1px solid {t['border']};"
            f"border-radius:6px; padding:4px 8px; font-size:11.5px; min-height:0;}}")
        self._search.textChanged.connect(lambda _text: self._rerender())
        hr.addWidget(self._search)

        self._level = SelectBox("Info", t, options=list(self._LEVEL_OPTIONS),
                                 on_select=self._on_level_selected)
        # SelectBox has no stretch factor of its own -- everywhere else it's
        # used, its container either gives it a stretch factor or is the
        # sole child of a vertical layout (which stretches it to the full
        # container width regardless of sizeHint). Packed into a QHBoxLayout
        # next to other siblings with real stretch (the search box, the
        # earlier addStretch), Qt instead honours SelectBox's own sizeHint
        # literally -- and that sizeHint under-reports the width its value
        # label + chevron actually need, so "Debug"/"Error" collapsed to a
        # sliver (confirmed by inspecting _val's allocated geometry: width 0)
        # with only the chevron visibly left. A floor wide enough for the
        # longest option (measured: "Debug"/"Error" at 42px) plus its icon
        # and margins fixes this locally without touching the shared atom.
        self._level.setMinimumWidth(96)
        hr.addWidget(self._level)

        self._autoscroll = Toggle(t, on=True)
        self._autoscroll.toggled.connect(self._on_autoscroll_toggled)
        hr.addWidget(self._autoscroll)
        hr.addWidget(label("Auto", 10.5, t["text_muted"]))

        hr.addWidget(IconButton("trash", t, 27, "Clear", self._on_clear))
        hr.addWidget(IconButton("download", t, 27, "Save to file…", self._export))
        hr.addWidget(IconButton("close", t, 27, "Close", self.hide))
        v.addWidget(head)
        v.addWidget(hline(t))

        # Always a dark "scope" ground regardless of the app's light/dark
        # theme (same token the image viewport uses) -- a log console reads
        # as an instrument, not a page, in both themes; deliberate, not an
        # oversight (see theme.py's "the bench & the scope" concept).
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFrameShape(QFrame.Shape.NoFrame)
        self._text.setStyleSheet(
            f"QTextEdit{{background:{t['scope']}; border:none; padding:8px 14px;"
            f"color:#aeb9c7; font-family:{theme.MONO}; font-size:11.5px;}}")
        v.addWidget(self._text, 1)

        self._record_sig.connect(self._on_record)
        backlog, unsubscribe = self._bus.subscribe(self._safe_emit_record)
        self._unsubscribe = unsubscribe
        # Fires on real C++ destruction regardless of how it happens
        # (deleteLater during a theme toggle's overlay teardown, or a
        # test's sip.delete()) -- more robust than trying to catch every
        # teardown path by hand, and avoids leaking a subscriber closure
        # onto the bus for the rest of the process's life.
        self.destroyed.connect(unsubscribe)
        self._records = list(backlog)
        self._rerender()
        self._badge.setText(self._badge_text())
        self.hide()

    # ── filtering / rendering ────────────────────────────────────────────────
    def _matches(self, rec: LogRecord) -> bool:
        if rec.level < self._threshold:
            return False
        q = self._search.text().strip().lower()
        if not q:
            return True
        return q in rec.message.lower() or q in short_source(rec.source).lower()

    def _format_parts(self, rec: LogRecord) -> tuple[str, str, str]:
        ts = time.strftime("%H:%M:%S", time.localtime(rec.ts))
        return ts, rec.level_name, short_source(rec.source)

    def _line_html(self, rec: LogRecord) -> str:
        ts, lvl, src = self._format_parts(rec)
        color = _level_color(self._t, rec)
        msg = html.escape(rec.message).replace("\n", "<br>&nbsp;&nbsp;&nbsp;&nbsp;")
        lvl_pad = html.escape(lvl.ljust(8)).replace(" ", "&nbsp;")
        src_pad = html.escape(src.ljust(10)).replace(" ", "&nbsp;")
        return (
            f"<div><span style='color:#5b6472'>{ts}</span>&nbsp;&nbsp;"
            f"<span style='color:{color};font-weight:700'>{lvl_pad}</span>"
            f"<span style='color:#6c7480'>{src_pad}</span>"
            f"<span>{msg}</span></div>"
        )

    def _plain_line(self, rec: LogRecord) -> str:
        ts, lvl, src = self._format_parts(rec)
        return f"{ts}  {lvl:<8}{src:<10}{rec.message}"

    def _rerender(self) -> None:
        matching = [r for r in self._records if self._matches(r)]
        self._text.setHtml("".join(self._line_html(r) for r in matching))
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _badge_text(self) -> str:
        total = len(self._records)
        errors = sum(1 for r in self._records if r.level >= ERROR)
        warns = sum(1 for r in self._records if r.level == WARNING)
        parts = [str(total)]
        if errors:
            parts.append(f"{errors} err")
        if warns:
            parts.append(f"{warns} warn")
        return " · ".join(parts)

    # ── live updates (bus -> Qt main thread) ────────────────────────────────
    # A record can arrive from any thread (a predict/training worker, the
    # Assistant's urllib SSE thread) -- guarded the same way every other
    # cross-thread emit in Studio is (ModelsScreen._safe_emit_log, etc.): a
    # background callback can outlive this widget (torn down by a theme
    # toggle mid-run), and emitting a signal on a since-deleted QObject
    # raises RuntimeError.
    def _safe_emit_record(self, rec: LogRecord) -> None:
        try:
            self._record_sig.emit(rec)
        except RuntimeError:
            pass

    def _on_record(self, rec: LogRecord) -> None:
        self._records.append(rec)
        self._badge.setText(self._badge_text())
        if self._matches(rec):
            self._text.append(self._line_html(rec))
            if self._autoscroll.is_on():
                sb = self._text.verticalScrollBar()
                sb.setValue(sb.maximum())

    # ── toolbar actions ──────────────────────────────────────────────────────
    def _on_level_selected(self, choice: str) -> None:
        self._threshold = self._LEVEL_THRESHOLD[choice]
        self._rerender()

    def _on_autoscroll_toggled(self, on: bool) -> None:
        if on:
            sb = self._text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_clear(self) -> None:
        self._bus.clear()
        self._records = []
        self._text.clear()
        self._badge.setText(self._badge_text())

    def _export(self) -> None:
        default_name = f"cellseg1-studio-logs-{time.strftime('%Y%m%d-%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save logs", default_name, "Text files (*.txt);;All files (*)")
        if not path:
            return
        lines = [self._plain_line(r) for r in self._records if self._matches(r)]
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def place(self):
        p = self.parentWidget()
        if p:
            from studio.components import Sidebar
            x = Sidebar.WIDTH
            self.setGeometry(x, p.height() - self.HEIGHT, p.width() - x, self.HEIGHT)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        else:
            child = item.layout()
            if child is not None:
                _clear_layout(child)


class _PaletteRow(QFrame):
    """One command row, Raycast-style: a leading emoji (falling back to the
    existing line-icon set when a command has none) + label + an optional
    trailing hint, highlighted as a rounded "pill" (inset from the panel's
    edges by the results layout's own margins — see ``_build_panel``) when
    selected, rather than a flat edge-to-edge wash. ``set_selected``
    restyles cheaply in place (no rebuild) so arrow-key navigation stays
    instant and never disturbs the scroll position the way a full
    re-render would.
    """

    def __init__(self, t: dict, cmd: Command, on_activate: Callable[[Command], None]):
        super().__init__()
        self._t = t
        self.cmd = cmd
        self.setObjectName("PaletteRow")   # background-only rule -- see EngineChip's comment
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(11)
        self._icon = QLabel(cmd.emoji)
        self._icon.setFixedWidth(20)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon.setStyleSheet("font-size:15px; background:transparent;")
        lay.addWidget(self._icon)
        self._text = QLabel(cmd.label)
        self._text.setStyleSheet("background:transparent;")
        lay.addWidget(self._text)
        lay.addStretch(1)
        if cmd.hint:
            hint_lbl = QLabel(cmd.hint)
            hint_lbl.setStyleSheet(
                f"color:{t['text_muted']}; font-size:10.5px; font-family:{theme.MONO};"
                f"background:transparent;")
            lay.addWidget(hint_lbl)
        if cmd.enabled:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            # Direct instance-attribute override, not a connected signal --
            # the same convention Accordion's header row already uses for
            # exactly this "make a plain QFrame clickable" need.
            self.mousePressEvent = lambda e: on_activate(cmd)
        else:
            # A real dimming, not just a muted text colour -- covers the
            # emoji glyph too, which (unlike the line-art icon set) can't be
            # recoloured via a stylesheet `color:` property (emoji render
            # with their own fixed colours regardless of CSS).
            effect = QGraphicsOpacityEffect(self)
            effect.setOpacity(0.4)
            self.setGraphicsEffect(effect)
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        t = self._t
        if not self.cmd.enabled:
            bg, text_color, icon_color = "transparent", t["text_muted"], t["text_muted"]
        elif selected:
            bg, text_color, icon_color = t["primary_weak"], t["text"], t["primary"]
        else:
            bg, text_color, icon_color = "transparent", t["text_subtle"], t["text_muted"]
        hover = f"QFrame:hover{{background:{t['primary_weak']};}}" if self.cmd.enabled else ""
        self.setStyleSheet(f"QFrame#PaletteRow{{background:{bg}; border-radius:8px;}}" + hover)
        self._text.setStyleSheet(f"color:{text_color}; font-size:13px; background:transparent;")
        if not self.cmd.emoji:
            self._icon.setPixmap(icons.pixmap(self.cmd.icon, icon_color, 16))


class _BoundedScrollArea(QScrollArea):
    """A QScrollArea whose ``sizeHint`` tracks its content widget's *actual*
    natural height, capped at ``max_height``, instead of the fixed
    "reserve the full max regardless of content" footprint a bare
    ``QScrollArea`` (with only ``setMaximumHeight``) has by default.

    This lets the *parent layout* size (and resize, across re-renders) it
    correctly on its own — the robust, idiomatic fix. An earlier version
    tried to force this after the fact with ``setFixedHeight()`` +
    ``self._panel.adjustSize()`` on the container; that fought the layout
    system unpredictably (confirmed empirically: the panel's geometry
    reverted to its old, larger size on a later event-loop pass the manual
    resize never anticipated). Overriding ``sizeHint`` instead means there
    is no snapshotted value to go stale — anything that asks (the parent
    layout, on any re-flow) always gets a fresh answer computed from the
    content widget's own current sizeHint.
    """

    def __init__(self, max_height: int):
        super().__init__()
        self._max_height = max_height

    def sizeHint(self) -> QSize:
        w = self.widget()
        content_height = w.sizeHint().height() if w is not None else 0
        return QSize(super().sizeHint().width(), min(content_height, self._max_height))


class CommandPalette(QWidget):
    """Centered ⌘K command palette over a scrim — a real, live action
    registry (``studio.command_registry``) with fuzzy search and full
    keyboard navigation, not a static demo list. ``get_commands`` is called
    fresh every time the palette opens, so availability (an active project,
    what's currently running, the current theme/backend) is always current.
    """

    _MAX_RESULTS_HEIGHT = 420

    def __init__(self, parent: QWidget, t: dict, get_commands: Callable[[], list[Command]]):
        super().__init__(parent)
        self._t = t
        self._get_commands = get_commands
        self._commands: list[Command] = []
        self._visible: list[Command] = []     # flattened, in on-screen order
        self._rows: list[_PaletteRow] = []     # parallel to _visible
        self._selected = 0
        self.setStyleSheet(f"background:{theme.SCRIM};")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 96, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self._panel = self._build_panel()
        outer.addWidget(self._panel)
        # A real stretch item, not just the AlignTop flag above: a QFrame's
        # default vertical size policy is Preferred, which happily grows to
        # fill whatever extra space a layout has -- alignment flags only
        # decide how a *shorter-than-available* layout is positioned, they
        # don't stop a Preferred-policy sole child from being stretched to
        # fill in the first place. Without this, the panel silently filled
        # the whole scrim height regardless of how few rows it held
        # (confirmed empirically: identical panel.sizeHint() either way;
        # only the stretch changes the actual on-screen geometry) -- a short
        # search result rendered as a couple of rows sitting in a
        # mostly-empty white box reaching almost to the window's bottom.
        outer.addStretch(1)
        self.input.textChanged.connect(self._on_query_changed)
        self.input.installEventFilter(self)
        self.hide()

    def _build_panel(self) -> QFrame:
        t = self._t
        panel = QFrame()
        panel.setFixedWidth(560)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setObjectName("PalettePanel")   # qualified -- see AssistantDrawer's comment
        panel.setStyleSheet(
            f"QFrame#PalettePanel{{background:{t['surface']}; border:1px solid {t['border_strong']};"
            f" border-radius:14px;}}")
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # bare_widget(), not a plain QWidget() -- an unstyled QWidget here
        # would inherit the app-wide QWidget{background:<bg>} rule and paint
        # an opaque <bg>-coloured rectangle over its own children, invisible
        # against the near-identical dark tones of the dark theme but a
        # glaring flat-grey patch in light theme (bg #f4f6f8 vs. this
        # panel's own surface #ffffff) -- the exact "bare QWidget() wrapper"
        # bug family docs/velum/CHANGELOG.md's 2026-07-09 entry already found
        # and fixed elsewhere (guide_screen.py's table/shortcut rows);
        # CommandPalette was still 100% static content at the time and never
        # got a real screenshot pass, so this instance went undiscovered
        # until the palette actually rendered live content here.
        # Raycast-style search row: a plain magnifying glass, no visible
        # field boundary (the row IS the field), no ESC chip -- closing on
        # Escape is a universal-enough convention not to need a permanent
        # on-screen reminder, and Raycast itself doesn't show one either.
        inp_wrap = bare_widget()
        inp_wrap.setObjectName("PaletteInputRow")
        ir = QHBoxLayout(inp_wrap)
        ir.setContentsMargins(16, 13, 16, 13)
        ir.setSpacing(11)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("search", t["text_muted"], 17))
        ir.addWidget(ic)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Search actions, projects, engines…")
        self.input.setStyleSheet("QLineEdit{border:none; background:transparent; font-size:15px;}")
        ir.addWidget(self.input, 1)
        v.addWidget(inp_wrap)
        v.addWidget(hline(t))

        # A QScrollArea (bounded height) rather than the flat list the
        # static skeleton got away with -- a real registry spanning every
        # tab is dozens of commands, not 6 fixed demo rows. Both the scroll
        # area and its content widget are explicitly re-pinned to this
        # panel's own `surface` token: theme.build_qss's app-wide
        # "QScrollArea, QScrollArea > QWidget > QWidget { background: bg }"
        # rule would otherwise paint them the *page* background instead,
        # a visible seam against this panel's `surface` (the same family of
        # token mismatch as this file's own "always-dark scope" comment on
        # LogsConsole, just the opposite direction).
        self._results_container = bare_widget()
        self._results_container.setStyleSheet(f"background:{t['surface']};")
        self._results_layout = QVBoxLayout(self._results_container)
        # Horizontal inset (not 0) -- Raycast's rows sit inset from the
        # panel's own edges, so a selected row's rounded "pill" highlight
        # (_PaletteRow.set_selected) reads as a distinct rounded shape
        # rather than a hard-edged rectangle flush with the panel border.
        # A little vertical breathing room between rows too, now that each
        # has its own rounded corners.
        self._results_layout.setContentsMargins(8, 8, 8, 8)
        self._results_layout.setSpacing(2)
        self._results_area = _BoundedScrollArea(self._MAX_RESULTS_HEIGHT)
        self._results_area.setWidgetResizable(True)
        self._results_area.setFrameShape(QFrame.Shape.NoFrame)
        self._results_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._results_area.setStyleSheet(f"QScrollArea{{background:{t['surface']}; border:none;}}")
        self._results_area.setWidget(self._results_container)
        v.addWidget(self._results_area)

        # Raycast-style footer: app branding on the left, the *currently
        # selected* command's own label + a "⏎" hint on the right (updated
        # live by _update_footer_action) -- tells you what Enter actually
        # does, rather than a generic "↑↓ navigate / ⏎ run / esc close"
        # legend that never changes and states the merely-mechanical.
        foot = bare_widget()   # see inp_wrap's comment above -- same bug, same fix
        foot.setObjectName("PaletteFootRow")
        fr = QHBoxLayout(foot)
        fr.setContentsMargins(16, 9, 14, 9)
        fr.setSpacing(8)
        brand_ic = QLabel()
        brand_ic.setPixmap(icons.pixmap("spark", t["text_muted"], 13))
        fr.addWidget(brand_ic)
        fr.addWidget(label("Velum", 11, t["text_muted"], 600))
        fr.addStretch(1)
        self._foot_action_lbl = label("", 11, t["text_subtle"], 600)
        fr.addWidget(self._foot_action_lbl)
        self._foot_hint_lbl = QLabel()
        self._foot_hint_lbl.setStyleSheet(
            f"color:{t['text_muted']}; font-family:{theme.MONO}; font-size:11px; background:transparent;")
        fr.addWidget(self._foot_hint_lbl)
        v.addWidget(hline(t))
        v.addWidget(foot)
        return panel

    # ── search / render ──────────────────────────────────────────────────────
    def _on_query_changed(self, _text: str) -> None:
        self._rerender()

    def _rerender(self) -> None:
        _clear_layout(self._results_layout)
        self._rows = []
        self._visible = []
        query = self.input.text()
        if query.strip():
            for cmd in search(self._commands, query):
                self._visible.append(cmd)
                self._add_row(cmd)
        else:
            for section, cmds in group_by_section(self._commands):
                header = label(section.upper(), 10.5, self._t["text_muted"], 600, 0.6)
                # 10px, not the old 17px: _results_layout's own 8px inset
                # (_build_panel) + this margin should land the header text
                # roughly under each row's own text (10px row margin + 8px
                # container inset), not indented further than it.
                header.setContentsMargins(10, 12, 10, 5)
                self._results_layout.addWidget(header)
                for cmd in cmds:
                    self._visible.append(cmd)
                    self._add_row(cmd)
        if not self._visible:
            empty = label("No matching commands", 12.5, self._t["text_muted"])
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setContentsMargins(0, 30, 0, 30)
            self._results_layout.addWidget(empty)
        self._selected = 0
        self._apply_selection_styles()
        # Deferred: right after adding brand-new widgets to the layout, Qt
        # hasn't settled them yet -- self._results_container.sizeHint() and
        # ensureWidgetVisible() both read back stale/zero geometry if called
        # synchronously here (confirmed empirically: sizeHint() measured
        # (0, 12) -- just this container's own margins, none of the new
        # rows -- immediately after _clear_layout + re-add, correcting
        # itself to the real value exactly one event-loop tick later).
        # Rows already on screen (the _move_selection path below, which
        # never adds anything) don't have this problem and stay synchronous.
        QTimer.singleShot(0, self._safe_finish_layout)

    def _add_row(self, cmd: Command) -> None:
        row = _PaletteRow(self._t, cmd, self._trigger)
        self._results_layout.addWidget(row)
        self._rows.append(row)

    def _safe_finish_layout(self) -> None:
        try:
            self._finish_layout()
        except RuntimeError:
            pass

    def _finish_layout(self) -> None:
        """Tell Qt's layout system the results area's ``sizeHint`` has
        changed, so it re-flows the panel to fit — shrinking for a short
        list, growing (up to ``_BoundedScrollArea``'s cap, where it starts
        scrolling instead) for a long one, exactly the "grows with content,
        caps, then scrolls" behaviour every real command palette (Spotlight,
        Raycast, VS Code's Quick Open) has.

        ``updateGeometry()`` — not a manual ``setFixedHeight()`` +
        ``self._panel.adjustSize()`` — is the correct, idiomatic way to ask:
        a widget managed by a layout (``self._results_area`` is, via
        ``_build_panel``'s own ``v.addWidget(...)``) has its geometry
        *authoritatively* owned by that layout, so forcing it manually
        fights the layout system unpredictably — confirmed empirically: the
        panel's geometry reverted to its old, larger size on a later
        event-loop pass the manual resize never anticipated.
        ``_BoundedScrollArea.sizeHint()`` is what actually reports the new
        (correct, capped) height; this just triggers the re-flow that reads
        it.
        """
        self._results_layout.activate()
        self._results_area.updateGeometry()
        self._panel.updateGeometry()
        self._scroll_to_selected()

    def _apply_selection_styles(self) -> None:
        for i, row in enumerate(self._rows):
            row.set_selected(i == self._selected)
        self._update_footer_action()

    def _update_footer_action(self) -> None:
        if 0 <= self._selected < len(self._visible):
            cmd = self._visible[self._selected]
            self._foot_action_lbl.setText(cmd.label)
            self._foot_hint_lbl.setText("⏎" if cmd.enabled else "")
        else:
            self._foot_action_lbl.setText("")
            self._foot_hint_lbl.setText("")

    def _scroll_to_selected(self) -> None:
        if 0 <= self._selected < len(self._rows):
            self._results_area.ensureWidgetVisible(self._rows[self._selected])

    # ── keyboard navigation ──────────────────────────────────────────────────
    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Down:
                self._move_selection(1)
                return True
            if key == Qt.Key.Key_Up:
                self._move_selection(-1)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._activate_selected()
                return True
        return super().eventFilter(obj, event)

    def _move_selection(self, delta: int) -> None:
        if not self._rows:
            return
        self._selected = (self._selected + delta) % len(self._rows)
        self._apply_selection_styles()
        self._scroll_to_selected()

    def _activate_selected(self) -> None:
        if 0 <= self._selected < len(self._visible):
            self._trigger(self._visible[self._selected])

    # ── running a command ────────────────────────────────────────────────────
    def _trigger(self, cmd: Command) -> None:
        if not cmd.enabled:
            return
        # Deferred to the next event-loop tick -- the same established fix
        # as the documented sipBadCatcherResult hazard elsewhere in Studio
        # (see workspace.py's 2026-07-10 fix): closing the palette can
        # itself be part of what the handler triggers (navigating tabs
        # rebuilds a screen, switching engines rebuilds the Segment pane),
        # so this must not run synchronously from inside the very click/key
        # dispatch that's still on the call stack.
        QTimer.singleShot(0, lambda: self._run(cmd))

    def _run(self, cmd: Command) -> None:
        self.hide()
        cmd.handler()

    # ── open / close ─────────────────────────────────────────────────────────
    def place(self):
        p = self.parentWidget()
        if p:
            self.setGeometry(0, 0, p.width(), p.height())

    def open(self):
        self._commands = list(self._get_commands())
        self.input.blockSignals(True)
        self.input.clear()
        self.input.blockSignals(False)
        self._rerender()
        self.place()
        self.show()
        self.raise_()
        self.input.setFocus()

    def mousePressEvent(self, e):
        # click on the scrim (outside the panel) closes
        child = self.childAt(e.position().toPoint())
        if child is None:
            self.hide()
        super().mousePressEvent(e)


class Toast(QFrame):
    """Bottom-right success toast. Static by default; ``announce()`` for real use."""

    def __init__(self, parent: QWidget, t: dict):
        super().__init__(parent)
        self._t = t
        self.setObjectName("Toast")   # qualified -- see AssistantDrawer's comment
        self.setStyleSheet(
            f"QFrame#Toast{{background:{t['surface']}; border:1px solid {t['border']};"
            f"border-left:3px solid {t['success']}; border-radius:11px;}}")
        row = QHBoxLayout(self)
        row.setContentsMargins(15, 12, 15, 12)
        row.setSpacing(12)
        ic = QLabel()
        ic.setFixedSize(30, 30)
        ic.setPixmap(icons.pixmap("check", t["success"], 16))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic.setStyleSheet(f"background:{t['success_weak']}; border-radius:8px;")
        row.addWidget(ic, alignment=Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(1)
        self._title = label("Segmentation complete", 13, t["text"], 600)
        col.addWidget(self._title)
        self._subtitle = label("247 cells · F1 0.94 vs ground truth · 3.2 s", 11.5, t["text_muted"])
        self._subtitle.setWordWrap(True)
        # setFixedWidth, not setMaximumWidth: a word-wrapping QLabel with no
        # anchor anywhere in this chain (the whole Toast frame's own size is
        # itself auto-computed via adjustSize()) has an ambiguous natural
        # width for Qt's heightForWidth negotiation to resolve -- confirmed
        # by measuring a real toast with 3-line-wrapping text: the label
        # settled at 137px wide (not the intended 280px cap), so the height
        # computed for *that* width undershot what the text actually needed,
        # and the wrapped tail painted outside the card's own rounded
        # background. A fixed width removes the ambiguity outright.
        self._subtitle.setFixedWidth(280)
        col.addWidget(self._subtitle)
        row.addLayout(col)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
        self.hide()

    def place(self):
        p = self.parentWidget()
        if p:
            self.adjustSize()
            self.move(p.width() - self.width() - 22, p.height() - self.height() - 22)

    def announce(self, title: str, subtitle: str, duration_ms: int = 3200) -> None:
        """Show a real, timed confirmation with the given text."""
        self._title.setText(title)
        self._subtitle.setText(subtitle)
        self.place()
        self.show()
        self.raise_()
        self._hide_timer.start(duration_ms)
