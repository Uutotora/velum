"""
Assistant tab — the user-facing side of the local segmentation agent.

Everything here runs on the machine. Four layers:
  1. A deterministic diagnostic engine that turns the current mask/image into
     one-click parameter fixes.
  2. Local-model management: pick, download (Ollama pull) and bake a
     task-specialised agent (`cellseg1-assistant`) from any base model.
  3. A grounded chat whose `SUGGEST:` lines become one-click Apply buttons, so
     the language model can drive the pipeline, not just describe it.
  4. Auto-tune: a real predict -> score-against-ground-truth -> adjust loop,
     either rule-based (the same diagnostic engine as #1, looped) or handed
     to a connected local model each round (a real tool-calling agent) —
     with a live score chart, a sortable trajectory table, CSV export and a
     parameter-importance readout, the same instruments AutoML dashboards
     (Optuna, Weights & Biases) and this app's own engine-benchmark table
     already use.
No network calls leave localhost.
"""
from __future__ import annotations

import threading
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QTextEdit, QLineEdit, QComboBox, QFrame, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QSpinBox, QDoubleSpinBox, QAbstractSpinBox, QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from napari_app.theme import (
    BG, FG, BORDER, BORDER_STRONG, TEXT, DIM, LABEL, ACCENT, ACCENT_SOFT,
    SUCCESS, DANGER, CONSOLE, INPUT, MONO, WARNING,
    WIDGET_SS, BTN_PRIMARY, BTN_SECONDARY,
)
from napari_app.widgets.common import SectionCard, CollapsibleCard, param_row as _param_row
from napari_app.widgets.controls import Combo
from napari_app.widgets.chat import ChatView
from napari_app import icons

WARN = WARNING   # amber, reserved for actionable warnings/active states
_DLG = QFileDialog.Option.DontUseNativeDialog

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


class TuningChart(QWidget):
    """Score-vs-round trajectory chart for the auto-tune loop — pyqtgraph if
    present, else matplotlib: the exact fallback pattern
    ``train_widget.LossChart``/``measurements_window.Histogram`` already use
    elsewhere in this app, applied to an Optuna/W&B-style optimization-
    history plot. The best-so-far round is marked."""

    def __init__(self):
        super().__init__()
        self._use_pg = False
        try:
            import pyqtgraph as pg
            self._pg = pg
            self._plot = pg.PlotWidget(background=CONSOLE)
            self._plot.setFixedHeight(120)
            self._plot.showGrid(x=True, y=True, alpha=0.15)
            for axis in ("bottom", "left"):
                self._plot.getAxis(axis).setTextPen(DIM)
                self._plot.getAxis(axis).setPen(BORDER)
            self._plot.setLabel("bottom", "round", color=DIM)
            self._plot.setLabel("left", "score", color=DIM)
            self._curve = self._plot.plot(
                pen=pg.mkPen(ACCENT, width=1.8), symbol="o",
                symbolSize=5, symbolBrush=ACCENT, symbolPen=None)
            self._best = pg.ScatterPlotItem(
                size=11, brush=pg.mkBrush(SUCCESS), pen=pg.mkPen(BG, width=1.4))
            self._plot.addItem(self._best)
            L = QVBoxLayout(); L.setContentsMargins(0, 0, 0, 0); L.addWidget(self._plot)
            self.setLayout(L)
            self._use_pg = True
        except Exception:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self._fig = Figure(figsize=(3.6, 1.3), dpi=90)
            self._fig.patch.set_facecolor(BG)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setFixedHeight(120)
            L = QVBoxLayout(); L.setContentsMargins(0, 0, 0, 0); L.addWidget(self._canvas)
            self.setLayout(L)
        self.setVisible(False)

    def clear(self):
        if self._use_pg:
            self._curve.setData([], [])
            self._best.setData([], [])
        else:
            self._ax.cla()
        self.setVisible(False)

    def update_trajectory(self, steps):
        if not steps:
            return
        xs = [s.step for s in steps]
        ys = [s.score for s in steps]
        best_i = max(range(len(steps)), key=lambda i: steps[i].score)
        if self._use_pg:
            self._curve.setData(xs, ys)
            self._best.setData([xs[best_i]], [ys[best_i]])
        else:
            self._ax.cla()
            self._ax.set_facecolor(CONSOLE)
            self._ax.plot(xs, ys, color=ACCENT, lw=1.6, marker="o", ms=4)
            self._ax.scatter([xs[best_i]], [ys[best_i]], color=SUCCESS, s=44, zorder=5)
            self._ax.tick_params(colors=DIM, labelsize=8)
            self._ax.set_xlabel("round", color=DIM, fontsize=8)
            self._ax.set_ylabel("score", color=DIM, fontsize=8)
            for sp in self._ax.spines.values():
                sp.set_edgecolor(BORDER)
            self._fig.tight_layout(pad=0.5)
            self._canvas.draw_idle()
        self.setVisible(True)


class AssistantWidget(QWidget):
    _token_signal        = pyqtSignal(str)
    _chat_done           = pyqtSignal()
    _pull_progress       = pyqtSignal(str, float)
    _pull_done           = pyqtSignal(str, bool)
    _create_status       = pyqtSignal(str)
    _create_done         = pyqtSignal(bool)
    _autotune_step        = pyqtSignal(object)
    _autotune_round_start = pyqtSignal(int, int)
    _autotune_finish      = pyqtSignal(str, str)

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
        self._tune_rows: list = []

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

        # ── Auto-tune (agentic predict -> score -> adjust loop) ────────────────
        tune_card = SectionCard("Auto-tune", icon="target")
        tune_desc = QLabel(
            "Runs predict → score against ground truth → adjust, on repeat, "
            "until the score plateaus — the Diagnose + Apply & re-run + "
            "Evaluate cycle above, automated.")
        tune_desc.setWordWrap(True)
        tune_desc.setStyleSheet(f"color:{DIM}; font-size:10.5px; background:transparent;")
        tune_card.addWidget(tune_desc)

        self._strategy_combo = Combo()
        self._strategy_combo.addItem("Rule-based (fast, deterministic)")
        self._strategy_combo.addItem("Local model (reasons about each round)")
        tune_card.addLayout(_param_row(
            "Strategy", self._strategy_combo,
            "Rule-based reuses the same suggestions Diagnose already makes, "
            "looped automatically. Local model asks your connected Ollama "
            "model what to try next each round instead — a real tool-"
            "calling agent loop — falling back to rule-based if the model "
            "errors or is disconnected.", label_width=90))

        self._max_steps_sb = QSpinBox()
        self._max_steps_sb.setRange(1, 30); self._max_steps_sb.setValue(8)
        self._max_steps_sb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        tune_card.addLayout(_param_row(
            "Max rounds", self._max_steps_sb,
            "Hard cap on rounds, regardless of whether the score is still "
            "improving.", label_width=90))

        self._patience_sb = QSpinBox()
        self._patience_sb.setRange(1, 10); self._patience_sb.setValue(2)
        self._patience_sb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        tune_card.addLayout(_param_row(
            "Patience", self._patience_sb,
            "Stop after this many consecutive rounds with no real "
            "improvement (a plateau).", label_width=90))

        self._min_delta_sb = QDoubleSpinBox()
        self._min_delta_sb.setDecimals(3); self._min_delta_sb.setRange(0.0, 0.5)
        self._min_delta_sb.setSingleStep(0.005); self._min_delta_sb.setValue(0.005)
        self._min_delta_sb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        tune_card.addLayout(_param_row(
            "Min improvement", self._min_delta_sb,
            "A round only counts as progress if the score rises by more "
            "than this.", label_width=90))

        run_row = QHBoxLayout(); run_row.setSpacing(7); run_row.setContentsMargins(0, 6, 0, 3)
        self._autotune_btn = QPushButton("Start auto-tune")
        self._autotune_btn.setFixedHeight(34); self._autotune_btn.setStyleSheet(BTN_PRIMARY)
        self._autotune_btn.clicked.connect(self._toggle_autotune)
        run_row.addWidget(self._autotune_btn, stretch=1)
        tune_card.addLayout(run_row)

        self._autotune_status = QLabel("Predict, set a ground-truth mask, then start.")
        self._autotune_status.setWordWrap(True)
        self._autotune_status.setStyleSheet(
            f"color:{DIM}; font-size:10.5px; background:transparent; padding-top:2px;")
        tune_card.addWidget(self._autotune_status)

        self._tune_chart = TuningChart()
        tune_card.addWidget(self._tune_chart)

        self._tune_table = QTableWidget(0, 5)
        self._tune_table.setHorizontalHeaderLabels(["Round", "Score", "Δ", "Cells", "Reason"])
        self._tune_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._tune_table.verticalHeader().setVisible(False)
        self._tune_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tune_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tune_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tune_table.setMaximumHeight(160)
        self._tune_table.itemSelectionChanged.connect(self._on_tune_row_selected)
        self._tune_table.setVisible(False)
        tune_card.addWidget(self._tune_table)

        tbl_btn_row = QHBoxLayout(); tbl_btn_row.setSpacing(7)
        self._use_round_btn = QPushButton("Use selected round")
        self._use_round_btn.setFixedHeight(30); self._use_round_btn.setStyleSheet(BTN_SECONDARY)
        self._use_round_btn.setEnabled(False)
        self._use_round_btn.clicked.connect(self._use_selected_tune_row)
        tbl_btn_row.addWidget(self._use_round_btn)
        self._export_tune_btn = QPushButton("Export CSV")
        self._export_tune_btn.setFixedHeight(30); self._export_tune_btn.setStyleSheet(BTN_SECONDARY)
        self._export_tune_btn.setEnabled(False)
        self._export_tune_btn.clicked.connect(self._export_tune_csv)
        tbl_btn_row.addWidget(self._export_tune_btn)
        tbl_btn_row.addStretch()
        tune_card.addLayout(tbl_btn_row)

        self._tune_importance = QLabel("")
        self._tune_importance.setWordWrap(True)
        self._tune_importance.setStyleSheet(
            f"color:{LABEL}; font-size:10px; font-family:{MONO};"
            f"background:transparent; padding-top:4px;")
        self._tune_importance.setVisible(False)
        tune_card.addWidget(self._tune_importance)
        L.addWidget(tune_card)

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
        self._tune_agent_btn = QPushButton("Tune for CellSeg1")
        self._tune_agent_btn.setFixedHeight(32); self._tune_agent_btn.setStyleSheet(BTN_SECONDARY)
        self._tune_agent_btn.setToolTip(
            "Bake a task-specialised agent (cellseg1-assistant) from the selected "
            "model: pins the domain persona and a low temperature for precise advice.")
        self._tune_agent_btn.clicked.connect(self._create_agent)
        mrow.addWidget(self._tune_agent_btn)
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
        self._autotune_round_start.connect(self._on_autotune_round_start)
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

    def _selected_strategy(self) -> str:
        return "llm" if self._strategy_combo.currentIndex() == 1 else "advisor"

    def _selected_model(self) -> str | None:
        if not self._model_combo.isEnabled():
            return None
        return self._model_combo.currentText()

    def _toggle_autotune(self):
        if self._autotune_active:
            self.predict.stop_auto_tune()
            self._autotune_status.setText("Stopping…")
            return

        strategy = self._selected_strategy()
        model = self._selected_model()
        if strategy == "llm" and not model:
            self._autotune_status.setText(
                "No local model connected — pick one in “Local model” below, "
                "or switch strategy to Rule-based.")
            return

        max_steps = self._max_steps_sb.value()
        err = self.predict.start_auto_tune(
            on_step=lambda step: self._autotune_step.emit(step),
            on_round_start=lambda s, t: self._autotune_round_start.emit(s, t),
            on_finish=lambda reason, detail: self._autotune_finish.emit(reason, detail),
            strategy=strategy, model=model, max_steps=max_steps,
            patience=self._patience_sb.value(), min_delta=self._min_delta_sb.value())
        if err:
            self._autotune_status.setText(err)
            return

        self._autotune_active = True
        self._autotune_prev_score = None
        self._autotune_best = None
        self._tune_rows = []
        self._tune_table.setRowCount(0)
        self._tune_table.setVisible(False)
        self._tune_chart.clear()
        self._tune_importance.setVisible(False)
        self._use_round_btn.setEnabled(False)
        self._export_tune_btn.setEnabled(False)
        self._autotune_btn.setText("Stop auto-tune")
        self._autotune_btn.setStyleSheet(
            f"QPushButton {{ background:{WARN}; border:1px solid {WARN};"
            f" border-radius:8px; color:#ffffff; font-weight:600; }}")
        strat_label = "local model" if strategy == "llm" else "rule-based"
        self._autotune_status.setText(f"Round 1/{max_steps} — starting ({strat_label})…")
        self._chat.system_note(
            f"Auto-tuning started — strategy: {strat_label}, up to {max_steps} rounds.")

    def _on_autotune_round_start(self, step: int, total: int):
        self._autotune_status.setText(f"Round {step + 1}/{total} — predicting & scoring…")

    def _on_autotune_step(self, step):
        prev = self._autotune_prev_score
        delta = (step.score - prev) if prev is not None else None
        self._autotune_prev_score = step.score
        if self._autotune_best is None or step.score > self._autotune_best.score:
            self._autotune_best = step
        self._tune_rows.append(step)

        r = self._tune_table.rowCount()
        self._tune_table.insertRow(r)
        self._fill_tune_row(r, step, delta)
        self._tune_table.setVisible(True)
        self._tune_chart.update_trajectory(self._tune_rows)
        self._export_tune_btn.setEnabled(True)

        delta_txt = "" if delta is None else f" ({'+' if delta >= 0 else ''}{delta:.3f})"
        chg_txt = ", ".join(f"{k}→{v}" for k, v in step.changes.items()) or "baseline"
        reason_txt = f" — {step.reason}" if step.reason else ""
        self._chat.system_note(
            f"Round {step.step}: score {step.score:.3f}{delta_txt} · "
            f"{step.n_cells} cells · {chg_txt}{reason_txt}")

    def _fill_tune_row(self, row: int, step, delta):
        def _item(text: str) -> QTableWidgetItem:
            it = QTableWidgetItem(text)
            it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            return it

        self._tune_table.setItem(row, 0, _item(str(step.step)))
        score_item = _item(f"{step.score:.3f}")
        col = SUCCESS if step.score >= 0.8 else ("#d6a54a" if step.score >= 0.5 else DANGER)
        score_item.setForeground(QColor(col))
        self._tune_table.setItem(row, 1, score_item)
        delta_txt = "—" if delta is None else f"{'+' if delta >= 0 else ''}{delta:.3f}"
        self._tune_table.setItem(row, 2, _item(delta_txt))
        self._tune_table.setItem(row, 3, _item(str(step.n_cells)))
        reason_item = QTableWidgetItem(step.reason or ("baseline" if step.step == 0 else ""))
        reason_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self._tune_table.setItem(row, 4, reason_item)

    def _on_tune_row_selected(self):
        self._use_round_btn.setEnabled(bool(self._tune_table.selectedItems()))

    def _use_selected_tune_row(self):
        rows = self._tune_table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if idx >= len(self._tune_rows):
            return
        step = self._tune_rows[idx]
        self.predict.restore_tuning_step(step.params)
        self._system_note(f"Restored round {step.step}'s parameters and re-ran.")

    def _export_tune_csv(self):
        if not self._tune_rows:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export auto-tune trajectory", "autotune_trajectory.csv",
            "CSV files (*.csv)", options=_DLG)
        if not path:
            return
        from napari_app.core import tuning_loop
        try:
            tuning_loop.write_trajectory_csv(path, self._tune_rows)
            self._autotune_status.setText(f"Exported {len(self._tune_rows)} rounds to {Path(path).name}")
        except Exception as e:
            self._autotune_status.setText(f"Export failed: {e}")

    def _on_autotune_finished(self, stop_reason: str, stop_detail: str):
        self._autotune_active = False
        self._autotune_btn.setText("Start auto-tune")
        self._autotune_btn.setStyleSheet(BTN_PRIMARY)
        from napari_app.core import tuning_loop
        reason_txt = tuning_loop.describe_stop_reason(stop_reason)
        if stop_detail:
            reason_txt = f"{reason_txt} ({stop_detail})"
        best = self._autotune_best
        if best is None:
            self._autotune_status.setText(f"Stopped: {reason_txt}. No rounds completed.")
            self._chat.system_note(f"Auto-tune stopped ({reason_txt}) before any round finished.")
            return

        self._autotune_status.setText(
            f"Stopped: {reason_txt}. Best: round {best.step}, score {best.score:.3f} "
            f"({best.n_cells} cells).")
        self._chat.system_note(
            f"Auto-tune finished ({reason_txt}) — best round {best.step}, "
            f"score {best.score:.3f}. Select it in the table above and click "
            "“Use selected round” to keep it.")

        importance = tuning_loop.parameter_importance(self._tune_rows)
        if importance:
            lines = []
            for k, corr in importance[:5]:
                strength = "strong" if abs(corr) >= 0.6 else ("moderate" if abs(corr) >= 0.3 else "weak")
                direction = "higher → better" if corr > 0 else "lower → better"
                lines.append(f"{k}: {strength} ({direction}, r={corr:+.2f})")
            self._tune_importance.setText("What mattered most:\n" + "\n".join(lines))
            self._tune_importance.setVisible(True)

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
            self._tune_agent_btn.setEnabled(True)
            has_agent = any(m.startswith(advisor.AGENT_MODEL_NAME) for m in models)
            self._status_lbl.setText(
                "● tuned agent ready" if has_agent else "● local model connected")
            self._status_lbl.setStyleSheet(
                f"color:{SUCCESS}; font-size:11px; background:transparent;")
        else:
            self._model_combo.addItem("(no local model)")
            self._model_combo.setEnabled(False)
            self._tune_agent_btn.setEnabled(False)
            self._status_lbl.setText("○ offline — using built-in engine")
            self._status_lbl.setStyleSheet(f"color:{DIM}; font-size:11px; background:transparent;")

    def _busy_models(self, busy: bool):
        self._tune_agent_btn.setEnabled(not busy and self._model_combo.isEnabled())
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
