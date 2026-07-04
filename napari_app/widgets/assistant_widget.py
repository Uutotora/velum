"""
Assistant tab — the user-facing side of the local segmentation agent.

Everything here runs on the machine. Three layers:
  1. A deterministic diagnostic engine that turns the current mask/image into
     one-click parameter fixes.
  2. Local-model management: pick, download (Ollama pull) and bake a
     task-specialised agent (`cellseg1-assistant`) from any base model.
  3. A grounded chat whose `SUGGEST:` lines become one-click Apply buttons, so
     the language model can drive the pipeline, not just describe it.
No network calls leave localhost.
"""
from __future__ import annotations

import threading

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QTextEdit, QLineEdit, QComboBox, QFrame, QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal

from napari_app.theme import (
    BG, FG, BORDER, TEXT, DIM, LABEL, ACCENT, SUCCESS, CONSOLE,
    WIDGET_SS, BTN_PRIMARY, BTN_SECONDARY,
)
from napari_app.widgets.common import SectionCard, CollapsibleCard

WARN = "#d6a54a"   # amber, reserved for actionable warnings

_SEV_COLOR = {"good": SUCCESS, "info": ACCENT, "warn": WARN}
_SEV_ICON = {"good": "✓", "info": "•", "warn": "⚠"}


class ChangeCard(QFrame):
    """A recommended parameter change with Apply / Apply-and-re-run buttons."""

    def __init__(self, title, detail, changes, color, action, on_apply, on_apply_rerun):
        super().__init__()
        self.setObjectName("Finding")
        self.setStyleSheet(
            f"QFrame#Finding {{ background:{FG}; border:1px solid {BORDER};"
            f" border-left:5px solid {color}; border-radius:7px; }}")
        L = QVBoxLayout(); L.setContentsMargins(12, 10, 12, 10); L.setSpacing(4)

        t = QLabel(title)
        t.setStyleSheet(f"color:{TEXT}; font-size:12px; font-weight:600; background:transparent;")
        t.setWordWrap(True)
        L.addWidget(t)

        if detail:
            d = QLabel(detail)
            d.setStyleSheet(f"color:{LABEL}; font-size:11px; background:transparent;")
            d.setWordWrap(True)
            L.addWidget(d)

        if changes:
            chg = "  ·  ".join(f"{k} → {v}" for k, v in changes.items())
            c = QLabel(chg)
            c.setStyleSheet(
                f"color:{color}; font-size:10px; font-family:'Menlo','SF Mono',monospace;"
                f"background:transparent; padding-top:2px;")
            c.setWordWrap(True)
            L.addWidget(c)

            btns = QHBoxLayout(); btns.setSpacing(6)
            a = QPushButton("Apply"); a.setFixedHeight(28); a.setStyleSheet(BTN_SECONDARY)
            a.clicked.connect(lambda: on_apply(changes))
            btns.addWidget(a)
            r = QPushButton(action or "Apply & re-run")
            r.setFixedHeight(28); r.setStyleSheet(BTN_PRIMARY)
            r.clicked.connect(lambda: on_apply_rerun(changes))
            btns.addWidget(r); btns.addStretch()
            L.addLayout(btns)
        self.setLayout(L)


class AssistantWidget(QWidget):
    _token_signal   = pyqtSignal(str)
    _chat_done      = pyqtSignal()
    _pull_progress  = pyqtSignal(str, float)
    _pull_done      = pyqtSignal(str, bool)
    _create_status  = pyqtSignal(str)
    _create_done    = pyqtSignal(bool)

    def __init__(self, viewer, predict_widget):
        super().__init__()
        self.viewer = viewer
        self.predict = predict_widget
        self._history: list[dict] = []
        self._chat_thread = None
        self._reply_buf: list[str] = []
        self._worker = None

        self.setStyleSheet(WIDGET_SS)
        outer = QVBoxLayout(); outer.setSpacing(0); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        L = QVBoxLayout(); L.setSpacing(0); L.setContentsMargins(14, 8, 14, 16)

        # ── Diagnostics card ───────────────────────────────────────────────────
        diag_card = SectionCard("Diagnostics  ·  no model needed")
        intro = QLabel(
            "Works fully offline — no AI model required. It analyses your image "
            "and mask directly and proposes concrete, one-click tuning fixes. "
            "(The chat below is separate and optional.)")
        intro.setStyleSheet(f"color:{LABEL}; font-size:11px; background:transparent;")
        intro.setWordWrap(True)
        diag_card.addWidget(intro)
        diag_btn = QPushButton("🔍  Diagnose current result")
        diag_btn.setFixedHeight(36); diag_btn.setStyleSheet(BTN_PRIMARY)
        diag_btn.clicked.connect(self._run_diagnose)
        diag_card.addWidget(diag_btn)
        L.addWidget(diag_card)

        self._findings_box = QVBoxLayout()
        self._findings_box.setSpacing(8); self._findings_box.setContentsMargins(0, 8, 0, 0)
        L.addLayout(self._findings_box)

        # ── Local model card ───────────────────────────────────────────────────
        model_card = SectionCard("Local model")
        status_row = QHBoxLayout(); status_row.setSpacing(8)
        self._status_lbl = QLabel("checking…")
        self._status_lbl.setStyleSheet(f"color:{DIM}; font-size:11px; background:transparent;")
        status_row.addWidget(self._status_lbl)
        status_row.addStretch()
        refresh = QPushButton("↻")
        refresh.setFixedSize(28, 26); refresh.setStyleSheet(BTN_SECONDARY)
        refresh.setToolTip("Re-scan for local models")
        refresh.clicked.connect(self._refresh_models)
        status_row.addWidget(refresh)
        model_card.addLayout(status_row)

        mrow = QHBoxLayout(); mrow.setSpacing(6)
        self._model_combo = QComboBox()
        mrow.addWidget(self._model_combo, stretch=1)
        self._tune_btn = QPushButton("Tune for CellSeg1")
        self._tune_btn.setFixedHeight(30); self._tune_btn.setStyleSheet(BTN_SECONDARY)
        self._tune_btn.setToolTip(
            "Bake a task-specialised agent (cellseg1-assistant) from the selected "
            "model: pins the domain persona and a low temperature for precise advice.")
        self._tune_btn.clicked.connect(self._create_agent)
        mrow.addWidget(self._tune_btn)
        model_card.addLayout(mrow)

        # Downloadable catalogue
        cat = CollapsibleCard("Download a model", collapsed=True)
        from napari_app import advisor
        for m in advisor.RECOMMENDED_MODELS:
            row = QHBoxLayout(); row.setSpacing(6)
            info = QLabel(f"{m['name']}  ·  {m['size']}\n{m['note']}")
            info.setStyleSheet(f"color:{LABEL}; font-size:10px; background:transparent;")
            info.setWordWrap(True)
            row.addWidget(info, stretch=1)
            dl = QPushButton("Download")
            dl.setFixedHeight(28); dl.setStyleSheet(BTN_SECONDARY)
            dl.clicked.connect(lambda _c=False, name=m["name"]: self._download_model(name))
            row.addWidget(dl)
            cat.addLayout(row)
        hint = QLabel(
            "Requires Ollama running locally (ollama.com). Models download once "
            "and stay on disk. Nothing is sent to the cloud.")
        hint.setStyleSheet(f"color:{DIM}; font-size:10px; background:transparent;")
        hint.setWordWrap(True)
        cat.addWidget(hint)
        model_card.addWidget(cat)

        self._model_progress = QProgressBar()
        self._model_progress.setRange(0, 100); self._model_progress.setFixedHeight(3)
        self._model_progress.setVisible(False)
        model_card.addWidget(self._model_progress)
        self._model_status = QLabel("")
        self._model_status.setStyleSheet(
            f"color:{DIM}; font-size:10px; background:transparent;"
            f"font-family:'Menlo','SF Mono',monospace;")
        self._model_status.setWordWrap(True)
        model_card.addWidget(self._model_status)
        L.addWidget(model_card)

        # ── Chat card ──────────────────────────────────────────────────────────
        chat_card = SectionCard("Ask the assistant  ·  optional local LLM")
        chat_intro = QLabel(
            "Natural-language Q&A. Needs a local model (above). Without one, "
            "questions are answered by the offline diagnostic engine.")
        chat_intro.setStyleSheet(f"color:{DIM}; font-size:10px; background:transparent;")
        chat_intro.setWordWrap(True)
        chat_card.addWidget(chat_intro)
        self._chat_view = QTextEdit()
        self._chat_view.setReadOnly(True); self._chat_view.setFixedHeight(220)
        self._chat_view.setStyleSheet(
            f"background:{CONSOLE}; color:{TEXT}; border:1px solid {BORDER};"
            f"border-radius:6px; font-size:12px; padding:8px;")
        chat_card.addWidget(self._chat_view)

        in_row = QHBoxLayout(); in_row.setSpacing(6)
        self._input = QLineEdit()
        self._input.setPlaceholderText("e.g. why are my cells over-merged?")
        self._input.returnPressed.connect(self._send)
        in_row.addWidget(self._input)
        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedHeight(30); self._send_btn.setStyleSheet(BTN_PRIMARY)
        self._send_btn.clicked.connect(self._send)
        in_row.addWidget(self._send_btn)
        chat_card.addLayout(in_row)

        self._chat_actions = QVBoxLayout()
        self._chat_actions.setSpacing(6); self._chat_actions.setContentsMargins(0, 6, 0, 0)
        chat_card.addLayout(self._chat_actions)
        L.addWidget(chat_card)

        L.addStretch()
        inner.setLayout(L)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.setLayout(outer)
        self.setMinimumWidth(360)

        self._token_signal.connect(self._append_token)
        self._chat_done.connect(self._on_chat_done)
        self._pull_progress.connect(self._on_pull_progress)
        self._pull_done.connect(self._on_pull_done)
        self._create_status.connect(lambda s: self._model_status.setText(s))
        self._create_done.connect(self._on_create_done)

        self._refresh_models()

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def _clear_findings(self):
        while self._findings_box.count():
            w = self._findings_box.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _run_diagnose(self):
        from napari_app import advisor
        img, mask = self.predict.last_context()
        params = self.predict.current_params()
        diag = advisor.diagnose(img, mask, params)
        self._clear_findings()
        for f in diag["findings"]:
            self._findings_box.addWidget(ChangeCard(
                f"{_SEV_ICON.get(f.severity,'•')}  {f.title}", f.detail, f.changes,
                _SEV_COLOR.get(f.severity, ACCENT), f.action,
                self._apply, self._apply_rerun))

    def _apply(self, changes):
        applied = self.predict.apply_params(changes)
        if applied:
            self._system_note(f"Applied: {', '.join(applied)}")

    def _apply_rerun(self, changes):
        applied = self.predict.apply_params(changes)
        if applied:
            self._system_note(f"Applied: {', '.join(applied)} — re-running…")
        self.predict.rerun()

    # ── Model management ───────────────────────────────────────────────────────

    def _refresh_models(self):
        from napari_app import advisor
        models = advisor.ollama_models()
        self._model_combo.clear()
        if models:
            # Surface the tuned agent first if it exists.
            ordered = ([m for m in models if m.startswith(advisor.AGENT_MODEL_NAME)]
                       + [m for m in models if not m.startswith(advisor.AGENT_MODEL_NAME)])
            self._model_combo.addItems(ordered)
            self._model_combo.setEnabled(True)
            self._tune_btn.setEnabled(True)
            has_agent = any(m.startswith(advisor.AGENT_MODEL_NAME) for m in models)
            self._status_lbl.setText(
                "● tuned agent ready" if has_agent else "● local model connected")
            self._status_lbl.setStyleSheet(
                f"color:{SUCCESS}; font-size:11px; background:transparent;")
        else:
            self._model_combo.addItem("(no local model)")
            self._model_combo.setEnabled(False)
            self._tune_btn.setEnabled(False)
            self._status_lbl.setText("○ offline — using built-in engine")
            self._status_lbl.setStyleSheet(f"color:{DIM}; font-size:11px; background:transparent;")

    def _busy_models(self, busy: bool):
        self._tune_btn.setEnabled(not busy and self._model_combo.isEnabled())
        self._model_progress.setVisible(busy)

    def _download_model(self, name: str):
        if self._worker and self._worker.is_alive():
            self._model_status.setText("A model operation is already running."); return
        from napari_app import advisor
        if not advisor.ollama_available():
            self._model_status.setText(
                "Ollama is not running. Install it from ollama.com and start it.")
            return
        self._busy_models(True)
        self._model_progress.setValue(0)
        self._model_status.setText(f"Downloading {name}…")

        def run():
            ok = advisor.ollama_pull(
                name, lambda s, f: self._pull_progress.emit(s, f))
            self._pull_done.emit(name, ok)

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _on_pull_progress(self, status: str, frac: float):
        self._model_progress.setValue(int(frac * 100))
        self._model_status.setText(status)

    def _on_pull_done(self, name: str, ok: bool):
        self._busy_models(False)
        self._model_status.setText(f"✓ {name} ready" if ok else f"✗ failed to download {name}")
        self._refresh_models()

    def _create_agent(self):
        if self._worker and self._worker.is_alive():
            return
        base = self._model_combo.currentText()
        from napari_app import advisor
        if base.startswith(advisor.AGENT_MODEL_NAME):
            self._model_status.setText("Pick a base model (not the agent itself) to re-bake.")
            return
        self._busy_models(True)
        self._model_progress.setRange(0, 0)  # indeterminate
        self._model_status.setText(f"Baking cellseg1-assistant from {base}…")

        def run():
            ok = advisor.ollama_create_agent(base, lambda s: self._create_status.emit(s))
            self._create_done.emit(ok)

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _on_create_done(self, ok: bool):
        self._model_progress.setRange(0, 100)
        self._busy_models(False)
        self._model_status.setText(
            "✓ cellseg1-assistant ready — selected below" if ok
            else "✗ could not create tuned agent (see log)")
        self._refresh_models()

    # ── Chat ───────────────────────────────────────────────────────────────────

    def _system_note(self, text: str):
        self._chat_view.append(f"<span style='color:{DIM};'>— {text}</span>")

    def _append_token(self, text: str):
        self._reply_buf.append(text)
        cursor = self._chat_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self._chat_view.setTextCursor(cursor)
        sb = self._chat_view.verticalScrollBar(); sb.setValue(sb.maximum())

    def _clear_chat_actions(self):
        while self._chat_actions.count():
            w = self._chat_actions.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _send(self):
        text = self._input.text().strip()
        if not text or (self._chat_thread and self._chat_thread.is_alive()):
            return
        self._input.clear()
        self._clear_chat_actions()
        self._chat_view.append(f"<b style='color:{ACCENT};'>You:</b> {text}")

        from napari_app import advisor
        img, mask = self.predict.last_context()
        params = self.predict.current_params()
        diag = advisor.diagnose(img, mask, params)

        if not self._model_combo.isEnabled():
            self._chat_view.append(f"<b style='color:{SUCCESS};'>Assistant (offline):</b>")
            cur = self._chat_view.textCursor(); cur.movePosition(cur.MoveOperation.End)
            cur.insertText("\n" + advisor.findings_to_text(diag) + "\n")
            self._chat_view.setTextCursor(cur)
            self._offer_suggestions(_merge_changes(diag))
            return

        model = self._model_combo.currentText()
        is_agent = model.startswith(advisor.AGENT_MODEL_NAME)
        if is_agent:
            # Persona is baked into the model; feed only the live context.
            system = None
            live = advisor.build_live_message(diag, params)
            self._history.append({"role": "user", "content": f"{live}\n\nQuestion: {text}"})
            messages = self._history[-8:]
        else:
            system = advisor.build_context_prompt(diag, params)
            self._history.append({"role": "user", "content": text})
            messages = [{"role": "system", "content": system}] + self._history[-8:]

        self._chat_view.append(f"<b style='color:{SUCCESS};'>Assistant:</b> ")
        self._reply_buf = []
        self._send_btn.setEnabled(False)

        def run():
            try:
                full = advisor.ollama_chat(model, messages, self._token_signal.emit)
                self._history.append({"role": "assistant", "content": full})
            except Exception as e:
                self._token_signal.emit(f"\n[error contacting local model: {e}]")
            finally:
                self._chat_done.emit()

        self._chat_thread = threading.Thread(target=run, daemon=True)
        self._chat_thread.start()

    def _on_chat_done(self):
        self._send_btn.setEnabled(True)
        self._chat_view.append("")
        from napari_app import advisor
        changes = advisor.parse_suggestions("".join(self._reply_buf))
        self._offer_suggestions(changes)

    def _offer_suggestions(self, changes: dict):
        self._clear_chat_actions()
        if not changes:
            return
        self._chat_actions.addWidget(ChangeCard(
            "Assistant suggests", "", changes, ACCENT, "Apply & re-run",
            self._apply, self._apply_rerun))


def _merge_changes(diag: dict) -> dict:
    out: dict = {}
    for f in diag["findings"]:
        out.update(f.changes)
    return out
