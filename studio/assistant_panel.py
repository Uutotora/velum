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

Layout: header · a collapsed-by-default "Model" settings accordion (backend
picker + per-backend fields + live status) · the chat (the hero surface,
filling whatever room is left) · the input row (Diagnose · text · Send).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QPropertyAnimation, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QFrame, QGraphicsOpacityEffect, QLabel, QHBoxLayout, QVBoxLayout,
    QLineEdit, QToolButton, QScrollArea, QProgressBar,
)

from studio import icons
from studio import theme
from studio.components import (
    Accordion, IconButton, PillButton, SegControl, SelectBox, bare_widget, hline, label,
)
from studio.assistant_controller import AssistantController, BACKENDS, BACKEND_LABELS

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

# state -> token: "unknown" (not checked yet) / "checking" (in flight,
# pulses) / "ok" / "bad" (both solid).
_STATUS_DOT_TOKEN = {"unknown": "text_muted", "checking": "primary", "ok": "success", "bad": "text_muted"}


class _StatusDot(QLabel):
    """A small live-status dot: solid once a check resolves, gently
    *breathing* (a looping opacity pulse) while one is in flight — reads as
    "checking right now", not just a static "please wait" label sitting
    next to an inert circle."""

    def __init__(self, t: dict, state: str = "unknown", size: int = 8):
        super().__init__()
        self._size = size
        self.setFixedSize(size, size)
        self._pulse: Optional[QPropertyAnimation] = None
        self.set_state(t, state)

    def set_state(self, t: dict, state: str) -> None:
        # Guards the *whole* body, not just the animation setup: this can be
        # called from a background-thread status check's completion
        # callback, which can land after the drawer (and this dot) was torn
        # down mid-check (e.g. a theme toggle) -- same deleted-widget hazard
        # motion.py's helpers guard against, narrowly, so a genuine bug
        # elsewhere still surfaces instead of being silently swallowed.
        try:
            color = t[_STATUS_DOT_TOKEN.get(state, "text_muted")]
            self.setStyleSheet(f"background:{color}; border-radius:{self._size // 2}px;")
            if self._pulse is not None:
                self._pulse.stop()
                self._pulse = None
                self.setGraphicsEffect(None)
            if state == "checking":
                eff = QGraphicsOpacityEffect(self)
                self.setGraphicsEffect(eff)
                anim = QPropertyAnimation(eff, b"opacity", self)
                anim.setDuration(900)
                anim.setKeyValueAt(0.0, 0.35)
                anim.setKeyValueAt(0.5, 1.0)
                anim.setKeyValueAt(1.0, 0.35)
                anim.setLoopCount(-1)
                anim.start()
                self._pulse = anim
        except RuntimeError:
            pass


class AssistantDrawer(QFrame):
    """Right-side chat drawer — a real chat backed by ``AssistantController``,
    reading/acting on the active Segment session through ``workspace`` (see
    ``workspace.py``'s "Assistant integration" methods)."""

    WIDTH = 360

    _token_sig = pyqtSignal(str)
    _done_sig = pyqtSignal(str)
    _error_sig = pyqtSignal(str)
    _status_sig = pyqtSignal(object, object, object, object)   # backend_at_call, ok, msg, models
    _pull_progress_sig = pyqtSignal(str, float)
    _pull_done_sig = pyqtSignal(str, bool)
    _create_status_sig = pyqtSignal(str)
    _create_done_sig = pyqtSignal(bool)

    def __init__(self, parent: QWidget, t: dict, controller: Optional[AssistantController] = None,
                workspace=None):
        super().__init__(parent)
        self._t = t
        self._controller = controller or AssistantController()
        self._workspace = workspace
        self._history: list[dict] = []
        self._last_diag: Optional[dict] = None
        self._last_send_was_live = False
        self._found_models: list[str] = []
        self._status_ok: Optional[bool] = None
        self._status_msg = ""
        self._op_status_text = ""

        self.setFixedWidth(self.WIDTH)
        # Qualified selector, not a bare "background:...;border-left:...;"
        # rule: an *unqualified* setStyleSheet() on a container cascades to
        # every descendant that doesn't more specifically override the same
        # property. QWidget/QLabel have an app-wide type-selector for
        # `background` (theme.build_qss), so that property is always safely
        # overridden lower down -- but nothing overrides plain `border` for
        # a bare QWidget, so the drawer's own border-left was leaking onto
        # ChatView's inset empty-state widget as a stray vertical line at
        # its own left edge (confirmed by pixel-sampling: exactly the
        # border colour, exactly the empty-state's own x-offset and height,
        # gone the moment this selector was scoped) -- the same rendering-
        # bug family docs/velum/CHANGELOG.md's 2026-07-08 entries already
        # named twice, just leaking `border` instead of `background`.
        self.setObjectName("AssistantDrawer")
        self.setStyleSheet(
            f"QFrame#AssistantDrawer{{background:{t['surface']}; border-left:1px solid {t['border']};}}")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._header())

        self._model_acc = Accordion(
            f"Model · {BACKEND_LABELS[self._controller.settings.backend]}",
            t, lead="settings", open_=False, fill="surface2")
        self._backend_seg = SegControl(
            [BACKEND_LABELS[b] for b in BACKENDS], t,
            active=BACKENDS.index(self._controller.settings.backend))
        self._backend_seg.changed.connect(self._on_backend_changed)
        self._model_acc.add(self._backend_seg)
        self._model_body_wrap = bare_widget()
        self._model_body_lay = QVBoxLayout(self._model_body_wrap)
        self._model_body_lay.setContentsMargins(0, 10, 0, 0)
        self._model_body_lay.setSpacing(10)
        self._model_acc.add(self._model_body_wrap)
        acc_wrap = bare_widget()
        acc_wrap_lay = QVBoxLayout(acc_wrap)
        acc_wrap_lay.setContentsMargins(12, 10, 12, 10)
        acc_wrap_lay.addWidget(self._model_acc)
        v.addWidget(acc_wrap)
        v.addWidget(hline(t))

        self._chat = ChatView(t)
        v.addWidget(self._chat, 1)

        v.addWidget(self._input_row())
        self.hide()

        self._token_sig.connect(self._on_token)
        self._done_sig.connect(self._on_done)
        self._error_sig.connect(self._on_error)
        self._status_sig.connect(self._on_status_result)
        self._pull_progress_sig.connect(self._on_pull_progress)
        self._pull_done_sig.connect(self._on_pull_done)
        self._create_status_sig.connect(self._on_create_status)
        self._create_done_sig.connect(self._on_create_done)

        self._rebuild_model_body()
        if self._controller.settings.backend != "offline":
            self._refresh_status()

    # ── chrome (header / input row) ─────────────────────────────────────────
    def _header(self) -> QWidget:
        t = self._t
        h = bare_widget()
        row = QHBoxLayout(h)
        row.setContentsMargins(15, 13, 12, 13)
        row.setSpacing(10)
        spark = QLabel()
        spark.setFixedSize(23, 23)
        spark.setPixmap(icons.pixmap("spark", "#fff", 13))
        spark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spark.setStyleSheet(
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            f"stop:0 {t['primary']}, stop:1 {t['signal']}); border-radius:7px;")
        row.addWidget(spark)
        row.addWidget(label("Assistant", 14, t["text"], 600))
        row.addStretch(1)
        row.addWidget(IconButton("close", t, 27, "Close", self.hide))
        wrap = bare_widget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(0)
        wl.addWidget(h)
        wl.addWidget(hline(t))
        return wrap

    def _input_row(self) -> QWidget:
        t = self._t
        w = bare_widget()
        row = QHBoxLayout(w)
        row.setContentsMargins(12, 12, 12, 12)
        row.setSpacing(8)
        row.addWidget(IconButton(
            "diagnose", t, 34, "Diagnose the current result — offline, no model needed",
            self._run_diagnose))
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

    # ── Command palette integration ─────────────────────────────────────────
    # Thin public aliases — the ⌘K palette's "Diagnose" / "Switch backend →
    # X" entries, same convention as WorkspaceScreen's/ModelsScreen's own
    # command-palette aliases. switch_backend reuses _backend_seg._select()
    # exactly as this module's own tests already do
    # (test_switching_backend_persists_and_updates_title), rather than a
    # second copy of _on_backend_changed's effect.
    def run_diagnose(self) -> None:
        self._run_diagnose()

    def switch_backend(self, idx: int) -> None:
        self._backend_seg._select(idx)

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

    # ── signal plumbing (background-thread -> Qt main thread) ──────────────
    # Every emit is guarded: a background thread's completion callback can
    # outlive this widget (e.g. torn down by a theme toggle mid-chat) —
    # the same documented hazard + fix already used throughout Studio
    # (ModelsScreen._safe_emit_log, etc.).
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

    def _safe_emit_status(self, backend_at_call: str, ok: bool, msg: str, models: list) -> None:
        try:
            self._status_sig.emit(backend_at_call, ok, msg, models)
        except RuntimeError:
            pass

    def _safe_emit_pull_progress(self, status: str, frac: float) -> None:
        try:
            self._pull_progress_sig.emit(status, frac)
        except RuntimeError:
            pass

    def _safe_emit_pull_done(self, name: str, ok: bool) -> None:
        try:
            self._pull_done_sig.emit(name, ok)
        except RuntimeError:
            pass

    def _safe_emit_create_status(self, status: str) -> None:
        try:
            self._create_status_sig.emit(status)
        except RuntimeError:
            pass

    def _safe_emit_create_done(self, ok: bool) -> None:
        try:
            self._create_done_sig.emit(ok)
        except RuntimeError:
            pass

    # ── chat signal handlers ────────────────────────────────────────────────
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

    # ── model settings: backend switch ──────────────────────────────────────
    def _on_backend_changed(self, idx: int) -> None:
        backend = BACKENDS[idx]
        _log.info("backend switched to %s", BACKEND_LABELS[backend])
        self._controller.settings.backend = backend
        self._controller.save_settings()
        self._found_models = []
        self._status_ok = None
        self._status_msg = ""
        self._model_acc.set_title(f"Model · {BACKEND_LABELS[backend]}")
        self._rebuild_model_body()
        if backend != "offline":
            self._refresh_status()

    def _rebuild_model_body(self) -> None:
        _clear_layout(self._model_body_lay)
        backend = self._controller.settings.backend
        if backend == "offline":
            self._build_offline_body()
        elif backend == "ollama":
            self._build_ollama_body()
        else:
            self._build_custom_body()

    def _status_line_text(self) -> str:
        return "Checking…" if self._status_ok is None else self._status_msg

    def _status_color(self) -> str:
        t = self._t
        if self._status_ok is None:
            return t["text_muted"]
        return t["success"] if self._status_ok else t["text_muted"]

    def _status_state(self) -> str:
        if self._status_ok is None:
            return "checking"
        return "ok" if self._status_ok else "bad"

    def _refresh_status(self) -> None:
        backend_at_call = self._controller.settings.backend
        self._status_ok = None
        self._status_msg = "Checking…"
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(self._status_line_text())
            self._status_lbl.setStyleSheet(
                f"color:{self._status_color()}; font-size:11px; background:transparent;")
        if hasattr(self, "_status_dot"):
            self._status_dot.set_state(self._t, "checking")
        self._controller.refresh_status_async(
            on_result=lambda ok, msg, models: self._safe_emit_status(backend_at_call, ok, msg, models))

    def _on_status_result(self, backend_at_call: str, ok: bool, msg: str, models: list) -> None:
        if backend_at_call != self._controller.settings.backend:
            return  # stale — the user switched backends while this was in flight
        # DEBUG, not INFO -- this fires automatically on every backend
        # switch/accordion open, not just a deliberate user action; the
        # Logs console's default level filter hides it unless asked for.
        _log.debug("status check (%s): ok=%s %s", backend_at_call, ok, msg)
        self._status_ok = ok
        self._status_msg = msg
        self._found_models = list(models or [])
        self._rebuild_model_body()

    # ── offline body ─────────────────────────────────────────────────────────
    def _build_offline_body(self) -> None:
        t = self._t
        desc = QLabel(
            "Built-in diagnostics only — deterministic, always available, nothing "
            "leaves your machine. No model or network needed.")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{t['text_muted']}; font-size:11px; background:transparent;")
        self._model_body_lay.addWidget(desc)

    # ── Ollama body ──────────────────────────────────────────────────────────
    def _build_ollama_body(self) -> None:
        t = self._t
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self._status_dot = _StatusDot(t, self._status_state())
        status_row.addWidget(self._status_dot, alignment=Qt.AlignmentFlag.AlignVCenter)
        self._status_lbl = label(self._status_line_text(), 11, self._status_color())
        self._status_lbl.setWordWrap(True)
        status_row.addWidget(self._status_lbl, 1)
        status_row.addWidget(IconButton("refresh", t, 28, "Re-check Ollama", self._refresh_status))
        self._model_body_lay.addLayout(status_row)

        models = self._found_models
        if models:
            current = self._controller.settings.ollama_model or models[0]
            select = SelectBox(current, t, options=models, on_select=self._pick_ollama_model)
            self._model_body_lay.addWidget(select)
            bake_row = QHBoxLayout()
            bake_row.setContentsMargins(0, 4, 0, 0)
            bake_btn = PillButton("Tune for CellSeg1", t, "ghost", small=True)
            bake_btn.setEnabled(not self._controller.model_op_busy())
            bake_btn.setToolTip(
                "Bake a task-specialised agent (cellseg1-assistant) from the selected "
                "model: pins the domain persona and a low temperature for precise advice.")
            bake_btn.clicked.connect(self._create_agent)
            bake_row.addWidget(bake_btn)
            bake_row.addStretch(1)
            self._model_body_lay.addLayout(bake_row)
        elif self._status_ok is not None:
            none_lbl = label("No local models yet — download one below.", 10.5, t["text_muted"])
            none_lbl.setWordWrap(True)
            self._model_body_lay.addWidget(none_lbl)

        catalog = Accordion("Download a model", t, lead="download", open_=False)
        for m in self._controller.recommended_models:
            row = QHBoxLayout()
            row.setSpacing(6)
            info = QLabel(f"{m['name']}  ·  {m['size']}\n{m['note']}")
            info.setStyleSheet(f"color:{t['text_muted']}; font-size:10px; background:transparent;")
            info.setWordWrap(True)
            row.addWidget(info, 1)
            dl = PillButton("Get", t, "ghost", small=True)
            dl.setEnabled(not self._controller.model_op_busy())
            dl.clicked.connect(lambda _c=False, name=m["name"]: self._download_model(name))
            row.addWidget(dl)
            catalog.add_layout(row)
        hint = QLabel(
            "Requires Ollama running locally (ollama.com). Models download once and "
            "stay on disk — nothing is sent to the cloud.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{t['text_muted']}; font-size:10px; background:transparent;")
        catalog.add(hint)
        self._model_body_lay.addWidget(catalog)

        self._op_progress = QProgressBar()
        self._op_progress.setRange(0, 100)
        self._op_progress.setFixedHeight(4)
        self._op_progress.setVisible(self._controller.model_op_busy())
        self._model_body_lay.addWidget(self._op_progress)
        self._op_status_lbl = label(self._op_status_text, 10, t["text_muted"])
        self._op_status_lbl.setWordWrap(True)
        self._model_body_lay.addWidget(self._op_status_lbl)

    def _pick_ollama_model(self, choice: str) -> None:
        self._controller.settings.ollama_model = choice
        self._controller.save_settings()

    def _download_model(self, name: str) -> None:
        if self._controller.model_op_busy():
            return
        self._op_status_text = f"Downloading {name}…"
        thread = self._controller.pull_ollama_model_async(
            name, on_progress=self._safe_emit_pull_progress, on_done=self._safe_emit_pull_done)
        if thread is not None:
            self._rebuild_model_body()

    def _on_pull_progress(self, status: str, frac: float) -> None:
        self._op_status_text = status
        if hasattr(self, "_op_progress"):
            self._op_progress.setValue(int(frac * 100))
        if hasattr(self, "_op_status_lbl"):
            self._op_status_lbl.setText(status)

    def _on_pull_done(self, name: str, ok: bool) -> None:
        if ok:
            _log.info("model pulled: %s", name)
        else:
            _log.warning("model pull failed: %s", name)
        self._op_status_text = f"✓ {name} ready" if ok else f"✗ failed to download {name}"
        self._rebuild_model_body()
        self._refresh_status()

    def _create_agent(self) -> None:
        if self._controller.model_op_busy():
            return
        base = self._controller.settings.ollama_model
        if not base or base.startswith(self._controller.agent_model_name):
            self._op_status_text = "Pick a base model (not the agent itself) to tune."
            if hasattr(self, "_op_status_lbl"):
                self._op_status_lbl.setText(self._op_status_text)
            return
        self._op_status_text = f"Baking {self._controller.agent_model_name} from {base}…"
        thread = self._controller.create_agent_async(
            base, on_status=self._safe_emit_create_status, on_done=self._safe_emit_create_done)
        if thread is not None:
            self._rebuild_model_body()

    def _on_create_status(self, status: str) -> None:
        self._op_status_text = status
        if hasattr(self, "_op_status_lbl"):
            self._op_status_lbl.setText(status)

    def _on_create_done(self, ok: bool) -> None:
        if ok:
            _log.info("tuned agent cellseg1-assistant ready")
        else:
            _log.warning("failed to create the tuned agent")
        self._op_status_text = ("✓ cellseg1-assistant ready — pick it above" if ok
                                else "✗ could not create the tuned agent")
        self._rebuild_model_body()
        self._refresh_status()

    # ── Custom API body ──────────────────────────────────────────────────────
    def _build_custom_body(self) -> None:
        t = self._t
        s = self._controller.settings

        url_edit = QLineEdit(s.custom_base_url)
        url_edit.setPlaceholderText("http://localhost:1234/v1")
        url_edit.textChanged.connect(lambda v: setattr(s, "custom_base_url", v))
        url_edit.editingFinished.connect(self._controller.save_settings)
        self._model_body_lay.addWidget(_field("Base URL", url_edit, t))

        key_edit = QLineEdit(s.custom_api_key)
        key_edit.setPlaceholderText("optional — sent as a Bearer token")
        key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_edit.textChanged.connect(lambda v: setattr(s, "custom_api_key", v))
        key_edit.editingFinished.connect(self._controller.save_settings)
        self._model_body_lay.addWidget(_field("API key", key_edit, t))

        model_edit = QLineEdit(s.custom_model)
        model_edit.setPlaceholderText("model id")
        model_edit.textChanged.connect(lambda v: setattr(s, "custom_model", v))
        model_edit.editingFinished.connect(self._controller.save_settings)
        self._model_body_lay.addWidget(_field("Model", model_edit, t))

        test_row = QHBoxLayout()
        test_row.setContentsMargins(0, 2, 0, 0)
        test_row.setSpacing(8)
        test_btn = PillButton("Test connection", t, "ghost", small=True)
        test_btn.clicked.connect(self._refresh_status)
        test_row.addWidget(test_btn)
        self._status_dot = _StatusDot(t, self._status_state())
        test_row.addWidget(self._status_dot, alignment=Qt.AlignmentFlag.AlignVCenter)
        self._status_lbl = label(self._status_line_text(), 10.5, self._status_color())
        self._status_lbl.setWordWrap(True)
        test_row.addWidget(self._status_lbl, 1)
        self._model_body_lay.addLayout(test_row)

        if self._found_models:
            n = len(self._found_models)
            pick = SelectBox(f"{n} model{'s' if n != 1 else ''} found — pick one", t,
                             options=self._found_models, on_select=self._pick_custom_model)
            self._model_body_lay.addWidget(pick)

        hint = QLabel(
            "Works with any OpenAI-compatible server: OpenAI, LM Studio, vLLM, "
            "llama.cpp, OpenRouter and others. Leave the key blank for a local "
            "server that doesn't need one.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{t['text_muted']}; font-size:10px; background:transparent;")
        self._model_body_lay.addWidget(hint)

    def _pick_custom_model(self, choice: str) -> None:
        self._controller.settings.custom_model = choice
        self._controller.save_settings()
        self._rebuild_model_body()
