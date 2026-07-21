"""Velum — the Settings screen (application-level preferences).

A real main-stack screen (same scaffold as ``extra_screens.py``'s
ModelsScreen: ``page_header`` + a section switch + a scrolling body) that
gathers every machine-level setting in one place — modelled on Zed's
Settings page. Three sections:

  * **AI Assistant** — the provider picker. All Assistant configuration
    lives here now (it used to be crammed into the chat drawer): a card per
    provider from ``assistant_controller.PROVIDERS`` (Offline · Ollama ·
    OpenAI · OpenRouter · Groq · LM Studio · Custom), each with step-by-step
    setup copy, an API-key/URL/model form, a live "Test connection" status,
    and — for Ollama — a local-model catalog + the "Tune for CellSeg1"
    agent-bake flow. The chat drawer (``assistant_panel.py``) is now a pure
    chat surface that just reads whichever provider is active here.
  * **Compute** — the detected inference device (``studio.hardware``).
  * **About** — version + links.

Qt UI only; all logic is the Qt-free ``AssistantController`` (unit-tested in
``tests/test_assistant_controller.py``). Network/model work runs on the
controller's background threads and is marshalled back through ``pyqtSignal``
with the same deleted-widget-safe ``_safe_emit_*`` guard used across Studio.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QPropertyAnimation, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QLineEdit, QProgressBar,
    QScrollArea,
)

from studio import icons
from studio import theme
from studio.components import (
    Accordion, Chip, IconButton, PillButton, SegControl, SelectBox, bare_widget,
    hline, label, soft_shadow,
)
from studio.screens import page_header, scroll
from studio.assistant_controller import AssistantController, PROVIDERS, provider

_log = logging.getLogger("studio.settings")


# A distinct hue per provider so the cards read apart at a glance (matches the
# Segment engine-dot convention in extra_screens.py).
def _hue(t: dict, provider_id: str) -> str:
    return {
        "offline": t["text_muted"], "ollama": t["signal"], "openai": t["success"],
        "openrouter": t["primary"], "groq": t["warning"], "lmstudio": t["signal"],
        "custom": t["text_subtle"],
    }.get(provider_id, t["primary"])


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
    """A caption stacked above a full-width control."""
    w = bare_widget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(5)
    cap = QLabel(caption)
    cap.setStyleSheet(f"color:{t['text_muted']}; font-size:10.5px; font-weight:600;"
                      f" letter-spacing:0.4px; background:transparent;")
    lay.addWidget(cap)
    lay.addWidget(control)
    return w


class _StatusDot(QLabel):
    """A small live-status dot: solid once a check resolves, gently breathing
    while one is in flight (mirrors the drawer's own dot)."""

    _TOKEN = {"unknown": "text_muted", "checking": "primary", "ok": "success", "bad": "text_muted"}

    def __init__(self, t: dict, state: str = "unknown", size: int = 9):
        super().__init__()
        self._size = size
        self.setFixedSize(size, size)
        self._pulse: Optional[QPropertyAnimation] = None
        self.set_state(t, state)

    def set_state(self, t: dict, state: str) -> None:
        try:
            from PyQt6.QtWidgets import QGraphicsOpacityEffect
            color = t[self._TOKEN.get(state, "text_muted")]
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


class SettingsScreen(QWidget):
    """Application settings — AI providers, compute, about."""

    _status_sig = pyqtSignal(str, object, object, object)   # provider_id, ok, msg, models
    _pull_progress_sig = pyqtSignal(str, float)
    _pull_done_sig = pyqtSignal(str, bool)
    _create_status_sig = pyqtSignal(str)
    _create_done_sig = pyqtSignal(bool)

    def __init__(self, t: dict, assistant_controller: AssistantController,
                 on_toast: Optional[Callable[[str, str], None]] = None):
        super().__init__()
        self._t = t
        self._assistant = assistant_controller
        self._toast = on_toast or (lambda *a: None)
        self._section = 0                                   # 0 AI · 1 Compute · 2 About
        # Which provider card is expanded (only one at a time). Default: the
        # active provider, so a returning user lands on what they're using.
        self._open_provider: Optional[str] = self._assistant.settings.active
        # provider_id -> (ok: Optional[bool], msg, models) — None ok == checking.
        self._status: dict[str, tuple] = {}
        self._op_status_text = ""

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)

        self._status_sig.connect(self._on_status_result)
        self._pull_progress_sig.connect(self._on_pull_progress)
        self._pull_done_sig.connect(self._on_pull_done)
        self._create_status_sig.connect(self._on_create_status)
        self._create_done_sig.connect(self._on_create_done)

        self.refresh()

    # ── (re)build ────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        scroll_pos = self._scroll_pos()
        _clear_layout(self._outer)
        t = self._t
        self._outer.addWidget(page_header(
            "Settings", "Assistant providers · compute · about", t))
        self._outer.addWidget(self._section_bar())
        if self._section == 1:
            body = self._compute_body()
        elif self._section == 2:
            body = self._about_body()
        else:
            body = self._ai_body()
        sa = scroll(body)
        self._outer.addWidget(sa)
        if scroll_pos is not None:
            QTimer.singleShot(0, lambda v=scroll_pos: sa.verticalScrollBar().setValue(v))

    def _scroll_pos(self) -> Optional[int]:
        for i in range(self._outer.count()):
            w = self._outer.itemAt(i).widget()
            if isinstance(w, QScrollArea):
                return w.verticalScrollBar().value()
        return None

    def _section_bar(self) -> QWidget:
        t = self._t
        wrap = bare_widget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(34, 2, 34, 14)
        seg = SegControl(["AI Assistant", "Compute", "About"], t, active=self._section)
        seg.setFixedWidth(360)
        seg.changed.connect(self._set_section)
        row.addWidget(seg)
        row.addStretch(1)
        return wrap

    def _set_section(self, idx: int) -> None:
        self._section = idx
        self.refresh()

    # ── AI Assistant section ─────────────────────────────────────────────────
    def _ai_body(self) -> QWidget:
        t = self._t
        body = bare_widget()
        v = QVBoxLayout(body)
        v.setContentsMargins(34, 4, 34, 40)
        v.setSpacing(14)

        head = QHBoxLayout()
        head.setSpacing(10)
        col = QVBoxLayout()
        col.setSpacing(3)
        col.addWidget(label("AI Assistant", 16, t["text"], 600))
        sub = label("Choose where the Assistant's answers come from. Nothing leaves this "
                    "machine unless you connect a cloud provider below.", 12.5, t["text_muted"])
        sub.setWordWrap(True)
        col.addWidget(sub)
        head.addLayout(col, 1)
        active_lbl = label(f"Active · {provider(self._assistant.settings.active).label}",
                           12, t["signal"], 600)
        head.addWidget(active_lbl, alignment=Qt.AlignmentFlag.AlignBottom)
        v.addLayout(head)

        for spec in PROVIDERS:
            v.addWidget(self._provider_card(spec))
        v.addStretch(1)
        return body

    def _provider_card(self, spec) -> QFrame:
        t = self._t
        is_active = self._assistant.settings.active == spec.id
        is_open = self._open_provider == spec.id
        card = QFrame()
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setObjectName("ProviderCard")
        border = t["primary_line"] if is_active else t["border"]
        card.setStyleSheet(
            f"QFrame#ProviderCard{{background:{t['surface']}; border:1px solid {border};"
            f" border-radius:14px;}}")
        if is_active:
            soft_shadow(card, 16, 22, 3)
        cv = QVBoxLayout(card)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self._provider_header(spec, is_active, is_open))
        if is_open:
            cv.addWidget(self._provider_body(spec))
        return card

    def _provider_header(self, spec, is_active: bool, is_open: bool) -> QWidget:
        t = self._t
        h = bare_widget()
        h.setCursor(Qt.CursorShape.PointingHandCursor)
        row = QHBoxLayout(h)
        row.setContentsMargins(16, 14, 15, 14)
        row.setSpacing(11)
        dot = QFrame()
        dot.setFixedSize(10, 10)
        dot.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dot.setStyleSheet(f"background:{_hue(t, spec.id)}; border-radius:5px;")
        row.addWidget(dot, alignment=Qt.AlignmentFlag.AlignVCenter)

        col = QVBoxLayout()
        col.setSpacing(2)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_row.addWidget(label(spec.label, 14, t["text"], 600))
        if spec.local:
            title_row.addWidget(Chip("Local", t, "success"))
        elif spec.needs_key:
            title_row.addWidget(Chip("API key", t, "muted"))
        title_row.addStretch(1)
        col.addLayout(title_row)
        tag = label(spec.tagline, 11.5, t["text_muted"])
        tag.setWordWrap(True)
        col.addWidget(tag)
        row.addLayout(col, 1)

        # Per-provider readiness tick — a quiet "this one is configured".
        if self._assistant.provider_ready(spec.id) and not is_active:
            ready = QLabel()
            ready.setPixmap(icons.pixmap("check", t["success"], 14))
            ready.setToolTip("Configured and ready")
            row.addWidget(ready, alignment=Qt.AlignmentFlag.AlignVCenter)

        if is_active:
            row.addWidget(Chip("Active", t, "signal"), alignment=Qt.AlignmentFlag.AlignVCenter)
        else:
            use = PillButton("Use", t, "ghost", small=True)
            use.clicked.connect(lambda _=False, pid=spec.id: self._use_provider(pid))
            row.addWidget(use, alignment=Qt.AlignmentFlag.AlignVCenter)

        chev = QLabel()
        chev.setPixmap(icons.pixmap("chevron_down" if is_open else "chevron", t["text_muted"], 14))
        row.addWidget(chev, alignment=Qt.AlignmentFlag.AlignVCenter)

        h.mouseReleaseEvent = lambda e, pid=spec.id: self._toggle_provider(pid)
        return h

    def _toggle_provider(self, provider_id: str) -> None:
        self._open_provider = None if self._open_provider == provider_id else provider_id
        self.refresh()

    def _use_provider(self, provider_id: str) -> None:
        self._assistant.set_active(provider_id)
        self._open_provider = provider_id
        _log.info("assistant provider set to %s", provider(provider_id).label)
        self._toast("Assistant provider", f"Now using {provider(provider_id).label}")
        self.refresh()

    # ── one expanded provider's body ─────────────────────────────────────────
    def _provider_body(self, spec) -> QWidget:
        t = self._t
        w = bare_widget()
        v = QVBoxLayout(w)
        v.setContentsMargins(37, 0, 16, 16)
        v.setSpacing(12)
        v.addWidget(hline(t))
        v.addSpacing(2)

        if spec.steps:
            v.addWidget(self._steps_block(spec))

        if spec.kind == "openai":
            self._build_openai_fields(spec, v)
        elif spec.kind == "ollama":
            self._build_ollama_body(spec, v)
        # offline: steps say it all.

        if spec.docs_url:
            link = QLabel(f"<a href='{spec.docs_url}' style='color:{t['primary']};"
                          f"text-decoration:none;'>{spec.docs_url.replace('https://', '')} ↗</a>")
            link.setOpenExternalLinks(True)
            link.setStyleSheet("background:transparent; font-size:11.5px;")
            v.addWidget(link)
        return w

    def _steps_block(self, spec) -> QWidget:
        t = self._t
        panel = QFrame()
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setObjectName("StepsPanel")
        panel.setStyleSheet(
            f"QFrame#StepsPanel{{background:{t['inset']}; border:1px solid {t['border']};"
            f" border-radius:10px;}}")
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(14, 12, 14, 12)
        pv.setSpacing(9)
        for i, step in enumerate(spec.steps, 1):
            r = QHBoxLayout()
            r.setSpacing(10)
            num = QLabel(str(i))
            num.setFixedSize(18, 18)
            num.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num.setStyleSheet(
                f"background:{t['primary_weak']}; color:{t['primary']}; border-radius:9px;"
                f" font-size:10.5px; font-weight:700;")
            r.addWidget(num, alignment=Qt.AlignmentFlag.AlignTop)
            txt = label(step, 12, t["text_subtle"])
            txt.setWordWrap(True)
            r.addWidget(txt, 1)
            pv.addLayout(r)
        return panel

    # ── OpenAI-compatible provider fields ────────────────────────────────────
    def _build_openai_fields(self, spec, v: QVBoxLayout) -> None:
        t = self._t
        s = self._assistant.settings

        if spec.editable_url:
            url_edit = QLineEdit(s.custom_base_url)
            url_edit.setPlaceholderText("http://localhost:1234/v1")
            url_edit.textChanged.connect(lambda val: setattr(s, "custom_base_url", val))
            url_edit.editingFinished.connect(self._assistant.save_settings)
            v.addWidget(_field("Base URL", url_edit, t))

        if spec.needs_key or spec.editable_url:
            key_edit = QLineEdit(self._assistant.key_for(spec.id))
            key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            key_edit.setPlaceholderText(spec.key_hint or ("optional — Bearer token" if spec.editable_url else "API key"))
            key_edit.textChanged.connect(lambda val, pid=spec.id: self._assistant.set_key(pid, val))
            key_edit.editingFinished.connect(self._assistant.save_settings)
            v.addWidget(_field("API key", key_edit, t))

        # Model: a free-text id, plus a picker of any discovered models.
        model_edit = QLineEdit(self._assistant.settings.models.get(spec.id, ""))
        model_edit.setPlaceholderText(spec.default_model or "model id")
        model_edit.textChanged.connect(lambda val, pid=spec.id: self._assistant.set_model(pid, val))
        model_edit.editingFinished.connect(self._assistant.save_settings)
        v.addWidget(_field("Model", model_edit, t))

        found = self._status.get(spec.id, (None, "", []))[2] or []
        if found:
            n = len(found)
            pick = SelectBox(f"{n} model{'s' if n != 1 else ''} available — pick one", t,
                             options=list(found),
                             on_select=lambda choice, pid=spec.id, e=model_edit: self._pick_model(pid, choice, e))
            v.addWidget(_field("Discovered models", pick, t))
        elif spec.model_hints:
            hint = label("Examples: " + " · ".join(spec.model_hints), 10.5, t["text_muted"])
            hint.setWordWrap(True)
            v.addWidget(hint)

        v.addLayout(self._test_row(spec))

    def _pick_model(self, provider_id: str, choice: str, edit: QLineEdit) -> None:
        self._assistant.set_model(provider_id, choice)
        self._assistant.save_settings()
        edit.setText(choice)

    def _test_row(self, spec) -> QHBoxLayout:
        t = self._t
        ok, msg, _models = self._status.get(spec.id, (None, "Not checked yet", []))
        row = QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 0)
        row.setSpacing(9)
        test_btn = PillButton("Test connection", t, "ghost", small=True)
        test_btn.clicked.connect(lambda _=False, pid=spec.id: self._check(pid))
        row.addWidget(test_btn)
        state = "checking" if (spec.id in self._status and ok is None) else \
                ("ok" if ok else ("bad" if ok is False else "unknown"))
        row.addWidget(_StatusDot(t, state), alignment=Qt.AlignmentFlag.AlignVCenter)
        color = t["success"] if ok else (t["text_muted"])
        lbl = label(msg if spec.id in self._status else "Not checked yet", 11, color)
        lbl.setWordWrap(True)
        row.addWidget(lbl, 1)
        return row

    # ── Ollama body (status · model picker · catalog · tune) ─────────────────
    def _build_ollama_body(self, spec, v: QVBoxLayout) -> None:
        t = self._t
        ok, msg, models = self._status.get(spec.id, (None, "", []))

        status_row = QHBoxLayout()
        status_row.setSpacing(9)
        state = "checking" if (spec.id in self._status and ok is None) else \
                ("ok" if ok else ("bad" if ok is False else "unknown"))
        status_row.addWidget(_StatusDot(t, state), alignment=Qt.AlignmentFlag.AlignVCenter)
        color = t["success"] if ok else t["text_muted"]
        status_lbl = label(msg if spec.id in self._status else "Checking Ollama…", 11.5, color)
        status_lbl.setWordWrap(True)
        status_row.addWidget(status_lbl, 1)
        status_row.addWidget(IconButton("refresh", t, 28, "Re-check Ollama",
                                        lambda pid=spec.id: self._check(pid)))
        v.addLayout(status_row)
        # Kick a check the first time this card is opened.
        if spec.id not in self._status:
            self._check(spec.id)

        if models:
            current = self._assistant.settings.ollama_model or models[0]
            select = SelectBox(current, t, options=list(models), on_select=self._pick_ollama_model)
            v.addWidget(_field("Model", select, t))
            tune_row = QHBoxLayout()
            tune_row.setContentsMargins(0, 2, 0, 0)
            tune = PillButton("Tune for CellSeg1", t, "ghost", small=True)
            tune.setEnabled(not self._assistant.model_op_busy())
            tune.setToolTip("Bake a task-specialised agent (cellseg1-assistant) from the selected "
                            "model: pins the domain persona and a low temperature for precise advice.")
            tune.clicked.connect(self._create_agent)
            tune_row.addWidget(tune)
            tune_row.addStretch(1)
            v.addLayout(tune_row)
        elif ok is not None:
            v.addWidget(label("No local models yet — download one below.", 11, t["text_muted"]))

        catalog = Accordion("Download a model", t, lead="download", open_=False, fill="inset")
        for m in self._assistant.recommended_models:
            r = QHBoxLayout()
            r.setSpacing(8)
            info = label(f"{m['name']}  ·  {m['size']}\n{m['note']}", 10.5, t["text_muted"])
            info.setWordWrap(True)
            r.addWidget(info, 1)
            dl = PillButton("Get", t, "ghost", small=True)
            dl.setEnabled(not self._assistant.model_op_busy())
            dl.clicked.connect(lambda _=False, name=m["name"]: self._download_model(name))
            r.addWidget(dl)
            catalog.add_layout(r)
        v.addWidget(catalog)

        self._op_progress = QProgressBar()
        self._op_progress.setRange(0, 100)
        self._op_progress.setFixedHeight(4)
        self._op_progress.setVisible(self._assistant.model_op_busy())
        v.addWidget(self._op_progress)
        self._op_status_lbl = label(self._op_status_text, 10.5, t["text_muted"])
        self._op_status_lbl.setWordWrap(True)
        v.addWidget(self._op_status_lbl)

    def _pick_ollama_model(self, choice: str) -> None:
        self._assistant.settings.ollama_model = choice
        self._assistant.save_settings()

    # ── async: status check ──────────────────────────────────────────────────
    def _check(self, provider_id: str) -> None:
        self._status[provider_id] = (None, "Checking…", [])
        self._assistant.check_provider_async(
            provider_id,
            on_result=lambda ok, msg, models: self._safe_emit_status(provider_id, ok, msg, models))

    def _on_status_result(self, provider_id: str, ok, msg, models) -> None:
        self._status[provider_id] = (ok, msg, list(models or []))
        if self._section == 0:
            self.refresh()

    # ── async: Ollama pull / tune ────────────────────────────────────────────
    def _download_model(self, name: str) -> None:
        if self._assistant.model_op_busy():
            return
        self._op_status_text = f"Downloading {name}…"
        thread = self._assistant.pull_ollama_model_async(
            name, on_progress=self._safe_emit_pull_progress, on_done=self._safe_emit_pull_done)
        if thread is not None:
            self.refresh()

    def _on_pull_progress(self, status: str, frac: float) -> None:
        self._op_status_text = status
        if hasattr(self, "_op_progress"):
            self._op_progress.setValue(int(frac * 100))
        if hasattr(self, "_op_status_lbl"):
            self._op_status_lbl.setText(status)

    def _on_pull_done(self, name: str, ok: bool) -> None:
        _log.info("model pull %s: %s", "ok" if ok else "failed", name)
        self._op_status_text = f"✓ {name} ready" if ok else f"✗ failed to download {name}"
        self._status.pop("ollama", None)   # force a re-list
        self.refresh()

    def _create_agent(self) -> None:
        if self._assistant.model_op_busy():
            return
        base = self._assistant.settings.ollama_model
        if not base or base.startswith(self._assistant.agent_model_name):
            self._op_status_text = "Pick a base model (not the agent itself) to tune."
            if hasattr(self, "_op_status_lbl"):
                self._op_status_lbl.setText(self._op_status_text)
            return
        self._op_status_text = f"Baking {self._assistant.agent_model_name} from {base}…"
        thread = self._assistant.create_agent_async(
            base, on_status=self._safe_emit_create_status, on_done=self._safe_emit_create_done)
        if thread is not None:
            self.refresh()

    def _on_create_status(self, status: str) -> None:
        self._op_status_text = status
        if hasattr(self, "_op_status_lbl"):
            self._op_status_lbl.setText(status)

    def _on_create_done(self, ok: bool) -> None:
        _log.info("tune agent: %s", "ok" if ok else "failed")
        self._op_status_text = ("✓ cellseg1-assistant ready — pick it above" if ok
                                else "✗ could not create the tuned agent")
        self._status.pop("ollama", None)
        self.refresh()

    # ── Compute section ──────────────────────────────────────────────────────
    def _compute_body(self) -> QWidget:
        t = self._t
        body = bare_widget()
        v = QVBoxLayout(body)
        v.setContentsMargins(34, 4, 34, 40)
        v.setSpacing(14)
        v.addWidget(label("Compute", 16, t["text"], 600))
        try:
            from studio import hardware
            info = hardware.detect()
            kind, dev_label, os_name = info.kind, info.label, info.os_name
        except Exception:
            kind, dev_label, os_name = "cpu", "Unknown", ""

        card = QFrame()
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setObjectName("ComputeCard")
        card.setStyleSheet(f"QFrame#ComputeCard{{background:{t['surface']};"
                           f" border:1px solid {t['border']}; border-radius:14px;}}")
        soft_shadow(card, 14, 20, 3)
        cv = QHBoxLayout(card)
        cv.setContentsMargins(20, 18, 20, 18)
        cv.setSpacing(15)
        badge = QLabel()
        badge.setFixedSize(44, 44)
        accent = {"cuda": t["success"], "mps": t["primary"], "cpu": t["text_muted"]}.get(kind, t["primary"])
        badge.setPixmap(icons.pixmap("cube3d", accent, 22))
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(f"background:{t['primary_weak']}; border-radius:11px;")
        cv.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(3)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(label(dev_label, 15, t["text"], 600))
        badge_kind = {"cuda": "success", "mps": "primary", "cpu": "muted"}.get(kind, "primary")
        row.addWidget(Chip(kind.upper(), t, badge_kind))
        row.addStretch(1)
        col.addLayout(row)
        sub = label(
            "Detected inference device — used for segmentation and one-shot training. "
            "CUDA (NVIDIA) and MPS (Apple Silicon) accelerate; CPU is the honest fallback."
            + (f"  ·  {os_name}" if os_name else ""), 12, t["text_muted"])
        sub.setWordWrap(True)
        col.addWidget(sub)
        cv.addLayout(col, 1)
        v.addWidget(card)
        v.addStretch(1)
        return body

    # ── About section ────────────────────────────────────────────────────────
    def _about_body(self) -> QWidget:
        t = self._t
        body = bare_widget()
        v = QVBoxLayout(body)
        v.setContentsMargins(34, 4, 34, 40)
        v.setSpacing(14)
        v.addWidget(label("About", 16, t["text"], 600))

        card = QFrame()
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setObjectName("AboutCard")
        card.setStyleSheet(f"QFrame#AboutCard{{background:{t['surface']};"
                           f" border:1px solid {t['border']}; border-radius:14px;}}")
        soft_shadow(card, 14, 20, 3)
        cv = QVBoxLayout(card)
        cv.setContentsMargins(22, 20, 22, 20)
        cv.setSpacing(6)
        cv.addWidget(QLabel(
            f"<span style='font-size:24px;font-weight:600;letter-spacing:-0.4px;color:{t['text']}'>"
            f"Velum<span style='color:{t['primary']}'>.</span></span>"))
        ver = label("Version 0.1  ·  Cell instance segmentation studio", 12.5, t["text_muted"])
        cv.addWidget(ver)
        desc = label("One-shot fine-tuning (SAM + LoRA), zero-shot generalists (Cellpose-SAM), "
                     "and z-stack/time-lapse (SAM 2) — with morphometry, cohort stats and a "
                     "local-first assistant. Built for microscopists, not ML engineers.",
                     12.5, t["text_subtle"])
        desc.setWordWrap(True)
        cv.addWidget(desc)
        cv.addSpacing(8)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        gh = PillButton("GitHub", t, "ghost", "settings")
        gh.clicked.connect(self._open_github)
        btn_row.addWidget(gh)
        btn_row.addStretch(1)
        cv.addLayout(btn_row)
        v.addWidget(card)
        v.addStretch(1)
        return body

    def _open_github(self) -> None:
        from studio import screens
        screens._open_github()

    # ── command-palette / external aliases ───────────────────────────────────
    def open_section(self, idx: int) -> None:
        self._set_section(idx)

    def focus_provider(self, provider_id: str) -> None:
        """Open the AI section with ``provider_id`` expanded (used when the
        chat drawer's gear jumps here)."""
        self._section = 0
        self._open_provider = provider_id
        self.refresh()

    # ── deleted-widget-safe cross-thread emit (see studio/motion.py) ─────────
    def _safe_emit_status(self, pid, ok, msg, models) -> None:
        try:
            self._status_sig.emit(pid, ok, msg, models)
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
