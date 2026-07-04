"""
Assistant tab — the user-facing side of the local segmentation agent.

Everything here runs on the machine: a deterministic diagnostic engine that
turns the current mask/image into one-click parameter fixes, plus an optional
chat backed by a locally-served Ollama model when one is running. No network
calls leave localhost.
"""
from __future__ import annotations

import threading

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QTextEdit, QLineEdit, QComboBox, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal

from napari_app.theme import (
    BG, FG, BORDER, TEXT, DIM, LABEL, ACCENT, SUCCESS, CONSOLE,
    WIDGET_SS, BTN_PRIMARY, BTN_SECONDARY,
)
from napari_app.widgets.common import SectionCard

WARN = "#d6a54a"   # amber, reserved for actionable warnings

_SEV_COLOR = {"good": SUCCESS, "info": ACCENT, "warn": WARN}
_SEV_ICON = {"good": "✓", "info": "•", "warn": "⚠"}


class FindingCard(QFrame):
    """A single diagnosis with optional apply / apply-and-rerun buttons."""

    def __init__(self, finding, on_apply, on_apply_rerun):
        super().__init__()
        color = _SEV_COLOR.get(finding.severity, ACCENT)
        self.setObjectName("Finding")
        self.setStyleSheet(
            f"QFrame#Finding {{ background:{FG}; border:1px solid {BORDER};"
            f" border-left:5px solid {color}; border-radius:7px; }}")
        L = QVBoxLayout(); L.setContentsMargins(12, 10, 12, 10); L.setSpacing(4)

        title = QLabel(f"{_SEV_ICON.get(finding.severity,'•')}  {finding.title}")
        title.setStyleSheet(
            f"color:{TEXT}; font-size:12px; font-weight:600; background:transparent;")
        title.setWordWrap(True)
        L.addWidget(title)

        detail = QLabel(finding.detail)
        detail.setStyleSheet(f"color:{LABEL}; font-size:11px; background:transparent;")
        detail.setWordWrap(True)
        L.addWidget(detail)

        if finding.changes:
            chg = "  ·  ".join(f"{k} → {v}" for k, v in finding.changes.items())
            chg_lbl = QLabel(chg)
            chg_lbl.setStyleSheet(
                f"color:{color}; font-size:10px; font-family:'Menlo','SF Mono',monospace;"
                f"background:transparent; padding-top:2px;")
            chg_lbl.setWordWrap(True)
            L.addWidget(chg_lbl)

            btns = QHBoxLayout(); btns.setSpacing(6)
            apply_btn = QPushButton("Apply")
            apply_btn.setFixedHeight(28)
            apply_btn.setStyleSheet(BTN_SECONDARY)
            apply_btn.clicked.connect(lambda: on_apply(finding))
            btns.addWidget(apply_btn)
            rerun_btn = QPushButton(finding.action or "Apply & re-run")
            rerun_btn.setFixedHeight(28)
            rerun_btn.setStyleSheet(BTN_PRIMARY)
            rerun_btn.clicked.connect(lambda: on_apply_rerun(finding))
            btns.addWidget(rerun_btn)
            btns.addStretch()
            L.addLayout(btns)

        self.setLayout(L)


class AssistantWidget(QWidget):
    _token_signal = pyqtSignal(str)
    _chat_done    = pyqtSignal()

    def __init__(self, viewer, predict_widget):
        super().__init__()
        self.viewer = viewer
        self.predict = predict_widget
        self._history: list[dict] = []
        self._chat_thread = None

        self.setStyleSheet(WIDGET_SS)
        outer = QVBoxLayout(); outer.setSpacing(0); outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        L = QVBoxLayout(); L.setSpacing(0); L.setContentsMargins(14, 8, 14, 16)

        # ── Diagnostics card ───────────────────────────────────────────────────
        diag_card = SectionCard("Diagnostics")
        intro = QLabel(
            "Inspect the latest prediction and get concrete, one-click tuning "
            "fixes. Runs entirely on your machine.")
        intro.setStyleSheet(f"color:{LABEL}; font-size:11px; background:transparent;")
        intro.setWordWrap(True)
        diag_card.addWidget(intro)

        diag_btn = QPushButton("🔍  Diagnose current result")
        diag_btn.setFixedHeight(36)
        diag_btn.setStyleSheet(BTN_PRIMARY)
        diag_btn.clicked.connect(self._run_diagnose)
        diag_card.addWidget(diag_btn)
        L.addWidget(diag_card)

        self._findings_box = QVBoxLayout()
        self._findings_box.setSpacing(8)
        self._findings_box.setContentsMargins(0, 8, 0, 0)
        L.addLayout(self._findings_box)

        # ── Chat card ──────────────────────────────────────────────────────────
        chat_card = SectionCard("Ask the assistant")

        status_row = QHBoxLayout(); status_row.setSpacing(8)
        self._status_lbl = QLabel("checking for local model…")
        self._status_lbl.setStyleSheet(
            f"color:{DIM}; font-size:11px; background:transparent;")
        status_row.addWidget(self._status_lbl)
        status_row.addStretch()
        self._model_combo = QComboBox()
        self._model_combo.setFixedWidth(150)
        status_row.addWidget(self._model_combo)
        chat_card.addLayout(status_row)

        self._chat_view = QTextEdit()
        self._chat_view.setReadOnly(True)
        self._chat_view.setFixedHeight(220)
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
        self._send_btn.setFixedHeight(30)
        self._send_btn.setStyleSheet(BTN_PRIMARY)
        self._send_btn.clicked.connect(self._send)
        in_row.addWidget(self._send_btn)
        chat_card.addLayout(in_row)

        hint = QLabel(
            "No local model? Install Ollama and run e.g. `ollama pull llama3.2` — "
            "the assistant will pick it up automatically. Without it, questions "
            "are answered by the built-in diagnostic engine.")
        hint.setStyleSheet(f"color:{DIM}; font-size:10px; background:transparent;")
        hint.setWordWrap(True)
        chat_card.addWidget(hint)
        L.addWidget(chat_card)

        L.addStretch()
        inner.setLayout(L)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.setLayout(outer)
        self.setMinimumWidth(360)

        self._token_signal.connect(self._append_token)
        self._chat_done.connect(self._on_chat_done)

        self._refresh_models()

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def _clear_findings(self):
        while self._findings_box.count():
            item = self._findings_box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _run_diagnose(self):
        from napari_app import advisor
        img, mask = self.predict.last_context()
        params = self.predict.current_params()
        diag = advisor.diagnose(img, mask, params)
        self._last_diag = diag
        self._clear_findings()
        for f in diag["findings"]:
            self._findings_box.addWidget(
                FindingCard(f, self._apply_finding, self._apply_and_rerun))

    def _apply_finding(self, finding):
        applied = self.predict.apply_params(finding.changes)
        if applied:
            self._system_note(f"Applied: {', '.join(applied)}")

    def _apply_and_rerun(self, finding):
        applied = self.predict.apply_params(finding.changes)
        if applied:
            self._system_note(f"Applied: {', '.join(applied)} — re-running prediction…")
        self.predict.rerun()

    # ── Chat ───────────────────────────────────────────────────────────────────

    def _refresh_models(self):
        from napari_app import advisor
        models = advisor.ollama_models()
        self._model_combo.clear()
        if models:
            self._model_combo.addItems(models)
            self._model_combo.setEnabled(True)
            self._status_lbl.setText("● local model connected")
            self._status_lbl.setStyleSheet(
                f"color:{SUCCESS}; font-size:11px; background:transparent;")
        else:
            self._model_combo.addItem("(no local model)")
            self._model_combo.setEnabled(False)
            self._status_lbl.setText("○ offline — using built-in engine")
            self._status_lbl.setStyleSheet(
                f"color:{DIM}; font-size:11px; background:transparent;")

    def _system_note(self, text: str):
        self._chat_view.append(f"<span style='color:{DIM};'>— {text}</span>")

    def _append_user(self, text: str):
        self._chat_view.append(f"<b style='color:{ACCENT};'>You:</b> {text}")

    def _append_token(self, text: str):
        cursor = self._chat_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self._chat_view.setTextCursor(cursor)
        sb = self._chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _send(self):
        text = self._input.text().strip()
        if not text or self._chat_thread and self._chat_thread.is_alive():
            return
        self._input.clear()
        self._append_user(text)

        from napari_app import advisor
        img, mask = self.predict.last_context()
        params = self.predict.current_params()
        diag = advisor.diagnose(img, mask, params)

        if not self._model_combo.isEnabled():
            # Offline fallback: answer from the diagnostic engine.
            self._chat_view.append(
                f"<b style='color:{SUCCESS};'>Assistant (offline):</b>")
            cursor = self._chat_view.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText("\n" + advisor.findings_to_text(diag) + "\n")
            self._chat_view.setTextCursor(cursor)
            sb = self._chat_view.verticalScrollBar()
            sb.setValue(sb.maximum())
            return

        model = self._model_combo.currentText()
        system = advisor.build_context_prompt(diag, params)
        self._history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": system}] + self._history[-8:]

        self._chat_view.append(f"<b style='color:{SUCCESS};'>Assistant:</b> ")
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
