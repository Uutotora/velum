"""Velum — the Assistant drawer: a real chat, not a static mockup.

Own UI, reuse the logic: everything here is Studio's own Qt (its own chat
view, its own icons/tokens — no import of ``velum_core.widgets.chat`` or
``assistant_widget``), talking to ``AssistantController``
(``studio/assistant_controller.py``) for the actual diagnostics/chat/model
logic, and to a ``workspace`` object (a ``WorkspaceScreen``, or anything
satisfying the same three-method contract — see
``workspace.py``'s "Assistant integration" section) for cross-tab context and
actions. Everything the controller can reach runs locally; nothing leaves the
machine unless the user has configured a remote Custom-API endpoint
themselves.

Layout: a clean header (title · Diagnose · Settings · Close — no decorative
badge) · a one-line provider status strip that jumps to Settings · the chat
(the hero surface, filling whatever room is left) · the input row. All
provider/model *configuration* moved out to ``studio/settings_screen.py``;
this drawer only reads whichever provider is active.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout,
    QLineEdit, QToolButton, QScrollArea,
)

from studio import icons
from studio import theme
from studio.components import (
    IconButton, PillButton, bare_widget, hline, label,
)
from studio.assistant_controller import AssistantController

_log = logging.getLogger("studio.assistant")

BUBBLE_MAXW = 300


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


def _field(caption: str, control: QWidget, t: dict) -> QWidget:
    """A caption label stacked above a full-width control — FieldRow puts
    the label *beside* the control, which doesn't suit a 336px-wide text
    field inside this drawer."""
    w = bare_widget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    cap = QLabel(caption)
    cap.setStyleSheet(f"color:{t['text_muted']}; font-size:10.5px; font-weight:600; background:transparent;")
    lay.addWidget(cap)
    lay.addWidget(control)
    return w


# ── chat surface (own port of the chat idiom — studio tokens, not velum_core's) ──

class _TypingDots(QLabel):
    """A tiny animated "assistant is typing" indicator."""

    def __init__(self, t: dict):
        super().__init__("●   ·   ·")
        self.setStyleSheet(f"color:{t['signal']}; font-size:11px; background:transparent;")
        self._i = 0
        self._timer = QTimer(self)
        self._timer.setInterval(340)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        seq = ("●   ·   ·", "·   ●   ·", "·   ·   ●")
        self.setText(seq[self._i % 3])
        self._i += 1

    def stop(self) -> None:
        self._timer.stop()
        self.hide()
        self.deleteLater()


class ChatView(QScrollArea):
    """Message bubbles, an assistant avatar, a streaming typing indicator,
    an empty state. Presentation only — the small API ``AssistantDrawer``
    drives:

        add_user(text)              user bubble (right)
        add_assistant_start()       open a streaming assistant bubble (left)
        append_token(text)          stream text into the open bubble
        assistant_done()            finalise the streaming bubble
        add_assistant_full(text)    a complete assistant bubble (no streaming)
        system_note(text)           a centred, dim status line
        add_widget(w)                drop an arbitrary widget into the flow
    """

    def __init__(self, t: dict):
        super().__init__()
        self._t = t
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        # No explicit background here: theme.build_qss's
        # "QScrollArea, QScrollArea > QWidget > QWidget" rule already paints
        # every scroll area's content widget in t['bg'] app-wide — matching
        # that instead of inventing a one-off style keeps this consistent
        # with every other scrollable panel in Studio.
        self._body = QWidget()
        self._v = QVBoxLayout(self._body)
        self._v.setContentsMargins(12, 14, 12, 14)
        self._v.setSpacing(12)
        self._v.addStretch()
        self.setWidget(self._body)

        self._cur: Optional[QLabel] = None
        self._cur_text = ""
        self._typing: Optional[_TypingDots] = None
        self._empty: Optional[QWidget] = None
        self._build_empty()

    # ── empty state ──────────────────────────────────────────────────────────
    def _build_empty(self) -> None:
        t = self._t
        self._empty = QWidget()
        el = QVBoxLayout(self._empty)
        el.setContentsMargins(0, 40, 0, 0)
        el.setSpacing(9)
        el.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("spark", t["signal"], 30))
        ic.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        ttl = label("Ask the assistant", 13, t["text"], 600)
        ttl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        sub = QLabel("It sees your image and mask, and can tune the pipeline.\n"
                    "e.g. “why are my cells over-merged?”")
        sub.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{t['text_muted']}; font-size:11px; background:transparent;")
        el.addWidget(ic)
        el.addWidget(ttl)
        el.addWidget(sub)
        self._insert(self._empty)

    def _hide_empty(self) -> None:
        if self._empty is not None:
            self._empty.hide()

    # ── flow helpers ─────────────────────────────────────────────────────────
    def _insert(self, w) -> None:
        self._v.insertWidget(self._v.count() - 1, w)  # before the trailing stretch
        # Every bubble/card/note in the flow goes through here exactly once
        # (append_token() only mutates an already-inserted bubble's text, it
        # never re-inserts) — one fade-in per new message, not per streamed
        # token, and the single choke point new message *kinds* need to gain
        # the same treatment automatically.
        from studio.motion import fade_in
        fade_in(w, 200)

    def _scroll(self) -> None:
        QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()))

    def _bubble(self, text: str, bg: str, fg: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl.setMaximumWidth(BUBBLE_MAXW)
        lbl.setStyleSheet(
            f"background:{bg}; color:{fg}; border-radius:12px; padding:9px 12px; font-size:12.5px;")
        return lbl

    def _row(self, inner, align: str) -> QWidget:
        w = bare_widget()
        r = QHBoxLayout(w)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(8)
        if align == "right":
            r.addStretch()
            r.addWidget(inner)
        else:
            r.addWidget(inner)
            r.addStretch()
        self._insert(w)
        self._scroll()
        return w

    def _assistant_container(self) -> QVBoxLayout:
        t = self._t
        w = bare_widget()
        r = QHBoxLayout(w)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(9)
        av = QLabel()
        av.setFixedSize(26, 26)
        av.setAlignment(Qt.AlignmentFlag.AlignCenter)
        av.setPixmap(icons.pixmap("spark", t["signal"], 15))
        av.setStyleSheet(f"background:{t['signal_weak']}; border-radius:13px;")
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(5)
        r.addWidget(av, alignment=Qt.AlignmentFlag.AlignTop)
        r.addLayout(col)
        r.addStretch()
        self._insert(w)
        self._scroll()
        return col

    # ── public API ───────────────────────────────────────────────────────────
    def add_user(self, text: str) -> None:
        t = self._t
        self._hide_empty()
        self._row(self._bubble(text, t["primary"], "#ffffff"), "right")

    def add_assistant_start(self) -> None:
        t = self._t
        self._hide_empty()
        col = self._assistant_container()
        self._cur = self._bubble("", t["surface2"], t["text"])
        self._cur.hide()  # hidden until the first token arrives
        self._cur_text = ""
        col.addWidget(self._cur)
        self._typing = _TypingDots(t)
        col.addWidget(self._typing, alignment=Qt.AlignmentFlag.AlignLeft)
        self._scroll()

    def append_token(self, text: str) -> None:
        if self._cur is None:
            return
        self._cur_text += text
        self._cur.setText(self._cur_text)
        if self._cur_text.strip():
            self._cur.show()
        self._scroll()

    def assistant_done(self) -> None:
        if self._typing is not None:
            self._typing.stop()
            self._typing = None
        if self._cur is not None and not self._cur_text.strip():
            self._cur.setText("(no response)")
            self._cur.show()
        self._cur = None
        self._scroll()

    def add_assistant_full(self, text: str) -> None:
        t = self._t
        self._hide_empty()
        col = self._assistant_container()
        col.addWidget(self._bubble(text, t["surface2"], t["text"]))
        self._scroll()

    def system_note(self, text: str) -> None:
        t = self._t
        self._hide_empty()
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color:{t['text_muted']}; font-size:10.5px; background:transparent;")
        self._insert(lbl)
        self._scroll()

    def add_widget(self, w: QWidget) -> None:
        self._hide_empty()
        self._insert(w)
        self._scroll()


class ChangeCard(QFrame):
    """A recommended parameter change with Apply / Apply & re-run buttons —
    Studio's own version of the same idea the classic app's Assistant uses,
    built from Studio's own atoms/tokens."""

    def __init__(self, t: dict, title: str, detail: str, changes: dict, color: str,
                action: str, on_apply: Callable[[dict], None], on_apply_rerun: Callable[[dict], None]):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified by objectName, NOT a bare "QFrame{...}" *type* selector —
        # QLabel is itself a QFrame subclass (paints border/background/
        # radius natively), so an unscoped QFrame{...} rule set here also
        # matches this card's own title/detail QLabels, giving each one its
        # own small bordered box around just its own text. Exactly the
        # rendering-bug family docs/velum/CHANGELOG.md's 2026-07-08 "Guide &
        # Docs" entry already root-caused and named ("even a bare type
        # selector like QFrame{…} cascades") — reproduced here despite that
        # lesson being on record, caught by an actual offscreen screenshot,
        # not by any test.
        self.setObjectName("ChangeCard")
        self.setStyleSheet(
            f"QFrame#ChangeCard{{ background:{t['surface']}; border:1px solid {t['border']};"
            f" border-radius:10px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(13, 11, 13, 11)
        lay.setSpacing(6)

        trow = QHBoxLayout()
        trow.setContentsMargins(0, 2, 0, 0)
        trow.setSpacing(9)
        # A fixed-size dot added with AlignTop (not a nested widget+layout
        # with its own addStretch() to "push it up") -- that nested-stretch
        # version is what velum_core's own ChangeCard uses, and it turns out
        # to have a real bug specific to living inside a QScrollArea's
        # dynamically-inserted content: Qt propagates the inner stretch's
        # "wants to expand" all the way up through the wrapper widget's
        # auto-assigned size policy, and the whole card balloons to 3-4x its
        # sizeHint once actually laid out inside ChatView (confirmed by
        # instrumenting card.geometry() vs. sizeHint() before this fix, and
        # again after removing just the inner addStretch()). AlignTop gives
        # the identical "dot pinned to the title's cap-height, even if the
        # title wraps to two lines" visual without that failure mode.
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{color}; border-radius:4px;")
        trow.addWidget(dot, alignment=Qt.AlignmentFlag.AlignTop)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color:{t['text']}; font-size:12px; font-weight:600; background:transparent;")
        title_lbl.setWordWrap(True)
        trow.addWidget(title_lbl, 1)
        lay.addLayout(trow)

        if detail:
            d = QLabel(detail)
            d.setStyleSheet(f"color:{t['text_subtle']}; font-size:11px; background:transparent;")
            d.setWordWrap(True)
            lay.addWidget(d)

        if changes:
            chg = "  ·  ".join(f"{k} → {v}" for k, v in changes.items())
            c = QLabel(chg)
            c.setStyleSheet(
                f"color:{color}; font-size:10px; font-family:{theme.MONO};"
                f"background:transparent; padding-top:2px;")
            c.setWordWrap(True)
            lay.addWidget(c)

            btns = QHBoxLayout()
            btns.setSpacing(7)
            btns.setContentsMargins(0, 3, 0, 0)
            apply_btn = PillButton("Apply", t, "ghost", small=True)
            apply_btn.clicked.connect(lambda: on_apply(changes))
            btns.addWidget(apply_btn)
            rerun_btn = PillButton((action or "Apply & re-run").replace("&", "&&"), t, "primary", small=True)
            rerun_btn.clicked.connect(lambda: on_apply_rerun(changes))
            btns.addWidget(rerun_btn)
            btns.addStretch()
            lay.addLayout(btns)


_SEVERITY_TOKEN = {"good": "success", "info": "primary", "warn": "warning"}

class AssistantDrawer(QFrame):
    """Right-side chat drawer — a pure chat surface backed by
    ``AssistantController``, reading/acting on the active Segment session
    through ``workspace`` (see ``workspace.py``'s "Assistant integration"
    methods).

    All provider/model configuration lives in the Settings screen
    (``studio/settings_screen.py``) now, not here: this drawer shows only a
    thin "which provider · which model" status strip with a gear that jumps
    to Settings, then the chat itself. ``on_open_settings`` is the callback
    (wired in ``app.py``) that navigates there.
    """

    WIDTH = 360

    _token_sig = pyqtSignal(str)
    _done_sig = pyqtSignal(str)
    _error_sig = pyqtSignal(str)

    def __init__(self, parent: QWidget, t: dict, controller: Optional[AssistantController] = None,
                 workspace=None, on_open_settings: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self._t = t
        self._controller = controller or AssistantController()
        self._workspace = workspace
        self._on_open_settings = on_open_settings or (lambda: None)
        self._history: list[dict] = []
        self._last_diag: Optional[dict] = None
        self._last_send_was_live = False

        self.setFixedWidth(self.WIDTH)
        # Qualified selector (see the git-blame lesson that was here before):
        # an unqualified border-left leaks onto inset children.
        self.setObjectName("AssistantDrawer")
        self.setStyleSheet(
            f"QFrame#AssistantDrawer{{background:{t['surface']}; border-left:1px solid {t['border']};}}")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._header())
        v.addWidget(self._status_strip())
        v.addWidget(hline(t))

        self._chat = ChatView(t)
        v.addWidget(self._chat, 1)
        v.addWidget(self._input_row())
        self.hide()

        self._token_sig.connect(self._on_token)
        self._done_sig.connect(self._on_done)
        self._error_sig.connect(self._on_error)

    # ── chrome ───────────────────────────────────────────────────────────────
    def _header(self) -> QWidget:
        t = self._t
        h = bare_widget()
        row = QHBoxLayout(h)
        row.setContentsMargins(16, 13, 12, 13)
        row.setSpacing(10)
        # No leading spark badge — a clean wordmark-style title reads more
        # like a finished product panel than a decorated widget.
        title = label("Assistant", 14.5, t["text"], 600)
        row.addWidget(title)
        row.addStretch(1)
        row.addWidget(IconButton("diagnose", t, 27,
                                 "Diagnose the current result — offline, no model needed",
                                 self._run_diagnose))
        row.addWidget(IconButton("settings", t, 27, "Assistant settings", self._on_open_settings))
        row.addWidget(IconButton("close", t, 27, "Close", self.hide))
        wrap = bare_widget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(0)
        wl.addWidget(h)
        wl.addWidget(hline(t))
        return wrap

    def _status_strip(self) -> QWidget:
        """A single quiet line: which provider (and model) is answering, a
        readiness dot, and a shortcut into Settings — replacing the old
        collapsible model-config accordion entirely."""
        t = self._t
        self._status_wrap = bare_widget()
        self._status_wrap.setCursor(Qt.CursorShape.PointingHandCursor)
        row = QHBoxLayout(self._status_wrap)
        row.setContentsMargins(16, 9, 12, 9)
        row.setSpacing(9)
        self._status_dot = QFrame()
        self._status_dot.setFixedSize(8, 8)
        self._status_dot.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.addWidget(self._status_dot, alignment=Qt.AlignmentFlag.AlignVCenter)
        self._status_lbl = label("", 11.5, t["text_muted"], 600)
        row.addWidget(self._status_lbl, 1)
        chev = QLabel()
        chev.setPixmap(icons.pixmap("chevron", t["text_muted"], 12))
        row.addWidget(chev, alignment=Qt.AlignmentFlag.AlignVCenter)
        self._status_wrap.mouseReleaseEvent = lambda e: self._on_open_settings()
        self._refresh_status_strip()
        return self._status_wrap

    def _refresh_status_strip(self) -> None:
        """Re-read the active provider from the controller (called on show, so
        a change made in Settings is reflected when the drawer reopens)."""
        t = self._t
        spec = self._controller.active_provider()
        if spec.kind == "offline":
            text, dot = spec.label, t["text_muted"]
        else:
            model = self._controller.resolved_model() or "no model selected"
            text = f"{spec.label} · {model}"
            dot = t["success"] if self._controller.backend_ready() else t["warning"]
        self._status_lbl.setText(text)
        self._status_dot.setStyleSheet(f"background:{dot}; border-radius:4px;")

    def _input_row(self) -> QWidget:
        t = self._t
        w = bare_widget()
        row = QHBoxLayout(w)
        row.setContentsMargins(12, 12, 12, 12)
        row.setSpacing(8)
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask about this image, settings or results…")
        self._input.returnPressed.connect(self._send)
        row.addWidget(self._input, 1)
        self._send_btn = QToolButton()
        self._send_btn.setFixedSize(34, 34)
        self._send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_btn.setIcon(icons.icon("send", "#fff", 16))
        self._send_btn.setStyleSheet(
            f"QToolButton{{background:{t['primary']}; border:none; border-radius:9px;}}"
            f"QToolButton:disabled{{background:{t['surface2']};}}")
        self._send_btn.clicked.connect(self._send)
        row.addWidget(self._send_btn)
        wrap = bare_widget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(0)
        wl.addWidget(hline(t))
        wl.addWidget(w)
        return wrap

    def place(self) -> None:
        p = self.parentWidget()
        if p:
            self.setGeometry(p.width() - self.WIDTH, 42, self.WIDTH, p.height() - 42)

    def showEvent(self, e):
        super().showEvent(e)
        # Reflect any provider/model change made in Settings since last shown.
        if hasattr(self, "_status_lbl"):
            self._refresh_status_strip()

    # ── Command palette integration ─────────────────────────────────────────
    def run_diagnose(self) -> None:
        self._run_diagnose()

    def open_settings(self) -> None:
        self._on_open_settings()

    # ── diagnostics ──────────────────────────────────────────────────────────
    def _run_diagnose(self) -> None:
        image, mask, params = self._workspace.assistant_context()
        diag = self._controller.diagnose(image, mask, params)
        self._chat.system_note("Diagnosis of the current result")
        findings = diag.get("findings", [])
        if not findings:
            self._chat.add_assistant_full("No problems detected — the segmentation looks healthy.")
            return
        t = self._t
        for f in findings:
            color = t[_SEVERITY_TOKEN.get(f.severity, "primary")]
            self._chat.add_widget(ChangeCard(
                t, f.title, f.detail, f.changes, color, f.action, self._apply, self._apply_rerun))

    def _apply(self, changes: dict) -> None:
        applied = self._workspace.apply_assistant_changes(changes)
        if applied is None:
            self._chat.system_note("Open a project in Segment first to apply changes.")
        elif applied:
            self._chat.system_note(f"Applied: {', '.join(applied)}")
        else:
            self._chat.system_note("Nothing to apply.")

    def _apply_rerun(self, changes: dict) -> None:
        applied = self._workspace.apply_assistant_changes(changes)
        if applied is None:
            self._chat.system_note("Open a project in Segment first to apply changes.")
            return
        if applied:
            self._chat.system_note(f"Applied: {', '.join(applied)} — re-running…")
        self._workspace.rerun_predict()

    # ── chat ─────────────────────────────────────────────────────────────────
    def _send(self) -> None:
        text = self._input.text().strip()
        if not text or self._controller.chat_busy():
            return
        self._input.clear()
        self._chat.add_user(text)

        image, mask, params = self._workspace.assistant_context()
        diag = self._controller.diagnose(image, mask, params)
        self._last_diag = diag

        live = self._controller.backend_ready()
        self._last_send_was_live = live
        prior_history = self._history[-8:]
        if live:
            self._chat.add_assistant_start()
            self._send_btn.setEnabled(False)

        self._history.append({"role": "user", "content": text})
        self._controller.send_async(
            prior_history, text, diag, params,
            on_token=self._safe_emit_token, on_done=self._safe_emit_done,
            on_error=self._safe_emit_error)

    def _offer_suggestions(self, changes: dict) -> None:
        if not changes:
            return
        t = self._t
        self._chat.add_widget(ChangeCard(
            t, "Assistant suggests", "", changes, t["primary"],
            "Apply & re-run", self._apply, self._apply_rerun))

    # ── signal plumbing (background thread -> Qt main thread) ────────────────
    def _safe_emit_token(self, text: str) -> None:
        try:
            self._token_sig.emit(text)
        except RuntimeError:
            pass

    def _safe_emit_done(self, text: str) -> None:
        try:
            self._done_sig.emit(text)
        except RuntimeError:
            pass

    def _safe_emit_error(self, msg: str) -> None:
        try:
            self._error_sig.emit(msg)
        except RuntimeError:
            pass

    def _on_token(self, text: str) -> None:
        self._chat.append_token(text)

    def _on_done(self, full_text: str) -> None:
        self._history.append({"role": "assistant", "content": full_text})
        if self._last_send_was_live:
            self._chat.assistant_done()
            self._send_btn.setEnabled(True)
            changes = self._controller.parse_suggestions(full_text)
        else:
            self._chat.add_assistant_full(full_text)
            changes = self._controller.merge_changes(self._last_diag) if self._last_diag else {}
        self._offer_suggestions(changes)

    def _on_error(self, msg: str) -> None:
        _log.warning("chat error: %s", msg)
        self._chat.append_token(f"\n[error contacting model: {msg}]")
        self._chat.assistant_done()
        self._send_btn.setEnabled(True)
