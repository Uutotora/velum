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
    BG, FG, BORDER, BORDER_STRONG, TEXT, DIM, LABEL, ACCENT, ACCENT_SOFT,
    SUCCESS, CONSOLE, INPUT, MONO, WARNING,
    WIDGET_SS, BTN_PRIMARY, BTN_SECONDARY,
)
from napari_app.widgets.common import SectionCard, CollapsibleCard
from napari_app.widgets.controls import Combo
from napari_app.widgets.chat import ChatView
from napari_app import icons

WARN = WARNING   # amber, reserved for actionable warnings

_SEV_COLOR = {"good": SUCCESS, "info": ACCENT, "warn": WARN}
_SEV_ICON = {"good": "✓", "info": "•", "warn": "⚠"}


class ChangeCard(QFrame):
    """A recommended parameter change with Apply / Apply-and-re-run buttons."""

    def __init__(self, title, detail, changes, color, action, on_apply, on_apply_rerun):
        super().__init__()
        self.setObjectName("Finding")
        self.setStyleSheet(
            f"QFrame#Finding {{ background:{INPUT}; border:1px solid {BORDER};"
            f" border-radius:8px; }}")
        L = QVBoxLayout(); L.setContentsMargins(13, 11, 13, 11); L.setSpacing(6)

        # Title row: a small severity dot + the title (no heavy left stripe).
        trow = QHBoxLayout(); trow.setContentsMargins(0, 0, 0, 0); trow.setSpacing(9)
        dot_wrap = QWidget(); dot_wrap.setFixedWidth(8)
        dwl = QVBoxLayout(dot_wrap); dwl.setContentsMargins(0, 4, 0, 0); dwl.setSpacing(0)
        dot = QLabel(); dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{color}; border-radius:4px;")
        dwl.addWidget(dot); dwl.addStretch()
        t = QLabel(title)
        t.setStyleSheet(f"color:{TEXT}; font-size:12px; font-weight:600; background:transparent;")
        t.setWordWrap(True)
        trow.addWidget(dot_wrap)
        trow.addWidget(t, stretch=1)
        L.addLayout(trow)

        if detail:
            d = QLabel(detail)
            d.setStyleSheet(f"color:{LABEL}; font-size:11px; background:transparent;")
            d.setWordWrap(True)
            L.addWidget(d)

        if changes:
            chg = "  ·  ".join(f"{k} → {v}" for k, v in changes.items())
            c = QLabel(chg)
            c.setStyleSheet(
                f"color:{color}; font-size:10px; font-family:{MONO};"
                f"background:transparent; padding-top:2px;")
            c.setWordWrap(True)
            L.addWidget(c)

            btns = QHBoxLayout(); btns.setSpacing(7); btns.setContentsMargins(0, 3, 0, 0)
            a = QPushButton("Apply"); a.setFixedHeight(34); a.setStyleSheet(BTN_SECONDARY)
            a.clicked.connect(lambda: on_apply(changes))
            btns.addWidget(a)
            # Escape '&' so Qt renders it literally (not as a mnemonic).
            r = QPushButton((action or "Apply & re-run").replace("&", "&&"))
            r.setFixedHeight(34); r.setStyleSheet(BTN_PRIMARY)
            r.clicked.connect(lambda: on_apply_rerun(changes))
            btns.addWidget(r); btns.addStretch()
            L.addLayout(btns)
        self.setLayout(L)


class AutoTuneStepCard(QFrame):
    """One round of the auto-tune trajectory: its score/changes plus a
    one-click way to restore its exact parameters (the loop's "undo" — jump
    back to any recorded step, not only the one it ended on)."""

    def __init__(self, step_no, changes, score, delta, n_cells, on_restore):
        super().__init__()
        self.setObjectName("TuneStep")
        self.setStyleSheet(
            f"QFrame#TuneStep {{ background:{INPUT}; border:1px solid {BORDER};"
            f" border-radius:8px; }}")
        L = QVBoxLayout(); L.setContentsMargins(13, 11, 13, 11); L.setSpacing(6)

        if delta is None:
            score_txt = f"score {score:.3f}"
        else:
            sign = "+" if delta >= 0 else ""
            score_txt = f"score {score:.3f} ({sign}{delta:.3f})"
        title = QLabel(f"Step {step_no}   ·   {score_txt}   ·   {n_cells} cells")
        title.setStyleSheet(f"color:{TEXT}; font-size:12px; font-weight:600; background:transparent;")
        title.setWordWrap(True)
        L.addWidget(title)

        chg = ("  ·  ".join(f"{k} → {v}" for k, v in changes.items())
               if changes else "(baseline parameters)")
        c = QLabel(chg)
        c.setStyleSheet(
            f"color:{ACCENT}; font-size:10px; font-family:{MONO};"
            f"background:transparent; padding-top:2px;")
        c.setWordWrap(True)
        L.addWidget(c)

        btns = QHBoxLayout(); btns.setSpacing(7); btns.setContentsMargins(0, 3, 0, 0)
        b = QPushButton("Use these params"); b.setFixedHeight(34); b.setStyleSheet(BTN_SECONDARY)
        b.clicked.connect(on_restore)
        btns.addWidget(b); btns.addStretch()
        L.addLayout(btns)
        self.setLayout(L)


class AssistantWidget(QWidget):
    _token_signal   = pyqtSignal(str)
    _chat_done      = pyqtSignal()
    _pull_progress  = pyqtSignal(str, float)
    _pull_done      = pyqtSignal(str, bool)
    _create_status  = pyqtSignal(str)
    _create_done    = pyqtSignal(bool)
    _autotune_step   = pyqtSignal(object)
    _autotune_finish = pyqtSignal()

    def __init__(self, viewer, predict_widget):
        super().__init__()
        self.viewer = viewer
        self.predict = predict_widget
        self._history: list[dict] = []
        self._chat_thread = None
        self._reply_buf: list[str] = []
        self._worker = None
        self._autotune_active = False
        self._autotune_prev_score = None
        self._autotune_best = None

        self.setStyleSheet(WIDGET_SS)
        outer = QVBoxLayout(self); outer.setSpacing(0); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        L = QVBoxLayout(inner); L.setSpacing(0); L.setContentsMargins(14, 8, 14, 16)

        # ── Chat (the hero surface, on top) ────────────────────────────────────
        chat_card = SectionCard("Assistant", icon="assistant")
        self._chat = ChatView()
        self._chat.setMinimumHeight(360)
        chat_card.addWidget(self._chat)

        in_row = QHBoxLayout(); in_row.setSpacing(7); in_row.setContentsMargins(0, 9, 0, 0)
        self._diag_btn = QPushButton()
        self._diag_btn.setFixedSize(36, 36); self._diag_btn.setStyleSheet(BTN_SECONDARY)
        self._diag_btn.setIcon(icons.icon("diagnose", LABEL, 16))
        self._diag_btn.setToolTip("Diagnose the current result — offline, no model needed")
        self._diag_btn.clicked.connect(self._run_diagnose)
        in_row.addWidget(self._diag_btn)
        self._autotune_btn = QPushButton()
        self._autotune_btn.setFixedSize(36, 36); self._autotune_btn.setStyleSheet(BTN_SECONDARY)
        self._autotune_btn.setIcon(icons.icon("target", LABEL, 16))
        self._autotune_btn.setToolTip(
            "Auto-tune: predict, score against ground truth, adjust, repeat "
            "until the score plateaus")
        self._autotune_btn.clicked.connect(self._toggle_autotune)
        in_row.addWidget(self._autotune_btn)
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask about your segmentation…")
        self._input.returnPressed.connect(self._send)
        in_row.addWidget(self._input, stretch=1)
        self._send_btn = QPushButton()
        self._send_btn.setFixedSize(36, 36); self._send_btn.setStyleSheet(BTN_PRIMARY)
        self._send_btn.setIcon(icons.icon("send", "#ffffff", 15))
        self._send_btn.setToolTip("Send")
        self._send_btn.clicked.connect(self._send)
        in_row.addWidget(self._send_btn)
        chat_card.addLayout(in_row)
        L.addWidget(chat_card)

        # ── Local model (optional — collapsed settings) ────────────────────────
        model_card = CollapsibleCard("Local model  ·  optional", collapsed=True, icon="spark")
        status_row = QHBoxLayout(); status_row.setSpacing(8)
        self._status_lbl = QLabel("checking…")
        self._status_lbl.setStyleSheet(f"color:{DIM}; font-size:11px; background:transparent;")
        status_row.addWidget(self._status_lbl)
        status_row.addStretch()
        refresh = QPushButton()
        refresh.setIcon(icons.icon("refresh", LABEL, 15))
        refresh.setFixedSize(30, 30); refresh.setStyleSheet(BTN_SECONDARY)
        refresh.setToolTip("Re-scan for local models")
        refresh.clicked.connect(self._refresh_models)
        status_row.addWidget(refresh)
        model_card.addLayout(status_row)

        mrow = QHBoxLayout(); mrow.setSpacing(6)
        self._model_combo = Combo()
        mrow.addWidget(self._model_combo, stretch=1)
        self._tune_btn = QPushButton("Tune for CellSeg1")
        self._tune_btn.setFixedHeight(32); self._tune_btn.setStyleSheet(BTN_SECONDARY)
        self._tune_btn.setToolTip(
            "Bake a task-specialised agent (cellseg1-assistant) from the selected "
            "model: pins the domain persona and a low temperature for precise advice.")
        self._tune_btn.clicked.connect(self._create_agent)
        mrow.addWidget(self._tune_btn)
        model_card.addLayout(mrow)

        # Downloadable catalogue
        cat = CollapsibleCard("Download a model", collapsed=True, icon="download")
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
            f"font-family:{MONO};")
        self._model_status.setWordWrap(True)
        model_card.addWidget(self._model_status)
        L.addWidget(model_card)

        L.addStretch()
        foot = QLabel("CellSeg1 · everything runs locally · nothing leaves your machine")
        foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        foot.setStyleSheet(f"color:{DIM}; font-size:10.5px; background:transparent; padding-top:16px;")
        L.addWidget(foot)

        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.setMinimumWidth(260)

        self._token_signal.connect(self._append_token)
        self._chat_done.connect(self._on_chat_done)
        self._pull_progress.connect(self._on_pull_progress)
        self._pull_done.connect(self._on_pull_done)
        self._create_status.connect(lambda s: self._model_status.setText(s))
        self._create_done.connect(self._on_create_done)
        self._autotune_step.connect(self._on_autotune_step)
        self._autotune_finish.connect(self._on_autotune_finished)

        self._refresh_models()

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def _run_diagnose(self):
        from napari_app import advisor
        img, mask = self.predict.last_context()
        params = self.predict.current_params()
        diag = advisor.diagnose(img, mask, params)
        # Diagnosis is delivered straight into the chat — one conversation surface.
        self._chat.system_note("Diagnosis of the current result")
        findings = diag.get("findings", [])
        if not findings:
            self._chat.add_assistant_full(
                "No problems detected — the segmentation looks healthy.")
            return
        for f in findings:
            self._chat.add_widget(ChangeCard(
                f.title, f.detail, f.changes,
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

    # ── Auto-tune (agentic predict -> score -> adjust loop) ─────────────────────

    def _toggle_autotune(self):
        if self._autotune_active:
            self.predict.stop_auto_tune()
            return
        err = self.predict.start_auto_tune(
            on_step=lambda step: self._autotune_step.emit(step),
            on_finish=lambda: self._autotune_finish.emit())
        if err:
            self._chat.system_note(err)
            return
        self._autotune_active = True
        self._autotune_prev_score = None
        self._autotune_best = None
        self._autotune_btn.setStyleSheet(
            f"QPushButton {{ background:{WARN}; border:1px solid {WARN}; border-radius:8px; }}")
        self._autotune_btn.setToolTip("Auto-tuning… click to stop")
        self._chat.system_note("Auto-tuning against ground truth…")

    def _on_autotune_step(self, step):
        prev = self._autotune_prev_score
        delta = (step.score - prev) if prev is not None else None
        self._autotune_prev_score = step.score
        if self._autotune_best is None or step.score > self._autotune_best.score:
            self._autotune_best = step
        self._chat.add_widget(AutoTuneStepCard(
            step.step, step.changes, step.score, delta, step.n_cells,
            lambda p=step.params: self._restore_autotune_step(p)))

    def _on_autotune_finished(self):
        self._autotune_active = False
        self._autotune_btn.setStyleSheet(BTN_SECONDARY)
        self._autotune_btn.setToolTip(
            "Auto-tune: predict, score against ground truth, adjust, repeat "
            "until the score plateaus")
        best = self._autotune_best
        if best is not None:
            self._chat.system_note(
                f"Auto-tune finished — best score {best.score:.3f} at step "
                f"{best.step} ({best.n_cells} cells). Use its card above to keep it.")
        else:
            self._chat.system_note("Auto-tune finished — no steps completed.")

    def _restore_autotune_step(self, params):
        self.predict.restore_tuning_step(params)
        self._system_note("Restored that step's parameters and re-ran.")

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
        self._chat.system_note(text)

    def _append_token(self, text: str):
        self._reply_buf.append(text)
        self._chat.append_token(text)

    def _send(self):
        text = self._input.text().strip()
        if not text or (self._chat_thread and self._chat_thread.is_alive()):
            return
        self._input.clear()
        self._chat.add_user(text)

        from napari_app import advisor
        img, mask = self.predict.last_context()
        params = self.predict.current_params()
        diag = advisor.diagnose(img, mask, params)

        if not self._model_combo.isEnabled():
            self._chat.add_assistant_full(advisor.findings_to_text(diag))
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

        self._chat.add_assistant_start()
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
        self._chat.assistant_done()
        from napari_app import advisor
        changes = advisor.parse_suggestions("".join(self._reply_buf))
        self._offer_suggestions(changes)

    def _offer_suggestions(self, changes: dict):
        if not changes:
            return
        self._chat.add_widget(ChangeCard(
            "Assistant suggests", "", changes, ACCENT, "Apply & re-run",
            self._apply, self._apply_rerun))


def _merge_changes(diag: dict) -> dict:
    out: dict = {}
    for f in diag["findings"]:
        out.update(f.changes)
    return out
