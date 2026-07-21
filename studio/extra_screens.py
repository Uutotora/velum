"""Velum — Models & Train and Dashboard screens.

Models & Train is real (bound to ``studio.train_controller.TrainController`` —
one-shot LoRA training on a background thread, a real trained-models list and
recent-runs history from disk). Dashboard is real (bound to
``studio.dashboard_controller.DashboardController``). See ``docs/velum/
BACKLOG.md`` / ``CHANGELOG.md`` for how each was wired.
"""
from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QPolygonF, QPainterPath
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QScrollArea,
    QFileDialog,
)

from studio import icons
from studio import theme
from studio.components import (
    Chip, Badge, PillButton, SelectBox, GroupLabel, hline, soft_shadow, label,
)
from studio.screens import page_header, scroll
from studio import train_controller as tc
from studio.train_controller import TrainController, TrainedModel
from studio.project_controller import ProjectController
from studio.dashboard_controller import DashboardController
from studio.components import bare_widget
from studio.log_bus import get_log_bus, emit_prefixed

_DLG = QFileDialog.Option.DontUseNativeDialog


# ── Models & Train ───────────────────────────────────────────────────────────
class ModelsScreen(QWidget):
    """Real one-shot LoRA training + model management, bound to a
    ``TrainController`` (``studio/train_controller.py``). Reuses the exact
    training pipeline the classic app's Train tab already uses
    (``velum_core.train_model``), imported lazily inside the controller
    — this module itself never imports torch.
    """

    _log_signal = pyqtSignal(str)
    _finish_signal = pyqtSignal()

    def __init__(self, t: dict, train_controller: TrainController,
                 project_controller: ProjectController,
                 on_toast: Callable[[str, str], None]):
        super().__init__()
        self._t = t
        self._train = train_controller
        self._projects = project_controller
        self._toast = on_toast

        self._image_path: Optional[Path] = None
        self._mask_path: Optional[Path] = None
        backbones = self._train.available_backbones()
        self._vit_name: Optional[str] = backbones[0][0] if backbones else None
        self._backbone_path: Optional[Path] = None
        self._lora_rank = tc.DEFAULT_RANK
        self._epochs = tc.DEFAULT_EPOCHS

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(1000)
        self._live_timer.timeout.connect(self._on_live_tick)
        self._log_signal.connect(self._on_log)
        self._finish_signal.connect(self._on_training_finished)

        self.refresh()

    # ── (re)build ────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Rebuild from the controllers' current state — disk (trained
        models, run history) and any live training progress. Called on tab
        navigation (``app.py``'s ``navigate()``) and after every action here.

        Preserves scroll position across the rebuild: the live-progress
        timer calls this every second while training, and without this a
        user scrolled down to watch the trained-models list would get
        yanked back to the top each tick.
        """
        scroll_pos = self._scroll_pos()
        while self._outer.count():
            item = self._outer.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        t = self._t
        models = self._train.list_trained_models()
        n = len(models)
        import_btn = PillButton("Import model", t, "ghost", "download")
        import_btn.clicked.connect(self._import_model)
        self._outer.addWidget(page_header(
            "Models & Train",
            f"{n} trained adapter{'s' if n != 1 else ''} · one-shot LoRA fine-tuning",
            t, import_btn))

        body = bare_widget()
        row = QHBoxLayout(body)
        row.setContentsMargins(34, 4, 34, 40)
        row.setSpacing(24)
        row.addLayout(self._left(models), 1)
        row.addLayout(self._aside(), 0)
        scroll_area = scroll(body)
        self._outer.addWidget(scroll_area)
        if scroll_pos is not None:
            QTimer.singleShot(0, lambda v=scroll_pos: scroll_area.verticalScrollBar().setValue(v))

        if self._train.is_training():
            self._live_timer.start()

    def _scroll_pos(self) -> Optional[int]:
        for i in range(self._outer.count()):
            w = self._outer.itemAt(i).widget()
            if isinstance(w, QScrollArea):
                return w.verticalScrollBar().value()
        return None

    def _on_live_tick(self) -> None:
        if not self._train.is_training():
            self._live_timer.stop()
            return
        self.refresh()

    # ── left: train form + trained models ───────────────────────────────────
    def _left(self, models: list[TrainedModel]) -> QVBoxLayout:
        t = self._t
        col = QVBoxLayout()
        col.setSpacing(24)
        col.addWidget(self._train_card())

        col.addWidget(label("Trained models", 15, t["text"], 600))
        if models:
            for m in models:
                col.addWidget(self._model_row(m))
        else:
            cap = label("No trained models yet — train one above, or import an existing checkpoint.",
                        12.5, t["text_muted"])
            cap.setWordWrap(True)
            col.addWidget(cap)
        col.addStretch(1)
        return col

    def _image_field_text(self) -> str:
        if self._image_path is None:
            return "Choose image…"
        if self._mask_path is None:
            return self._image_path.name
        n = tc.count_cells_in_mask(self._mask_path)
        return f"{self._image_path.name} · {n} cells" if n is not None else self._image_path.name

    def _backbone_field_text(self) -> str:
        if self._backbone_path is not None:
            return self._backbone_path.name
        if self._vit_name is not None:
            return tc.BACKBONE_LABELS[self._vit_name]
        return "Not found"

    def _has_backbone(self) -> bool:
        return self._backbone_path is not None or self._vit_name is not None

    def _status_text(self) -> str:
        if self._image_path is not None and self._mask_path is None:
            return ("No mask found for this image — annotate it in the Segment tab "
                    "first, then pick it here.")
        if not self._has_backbone():
            return ("No SAM backbone auto-detected — click SAM backbone to browse for a "
                    "checkpoint, or run scripts/setup.sh to download one.")
        return ""

    def _can_start(self) -> bool:
        return (self._image_path is not None and self._mask_path is not None
                and self._has_backbone() and not self._train.is_training())

    def _train_card(self) -> QFrame:
        t = self._t
        card = QFrame()
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified -- see components.EngineChip's comment. label() never
        # sets its own `border`, so every label() call below this card was
        # individually double-boxed by this unqualified rule.
        card.setObjectName("TrainCard")
        card.setStyleSheet(f"QFrame#TrainCard{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;}}")
        soft_shadow(card, 14, 22, 3)
        cv = QVBoxLayout(card)
        cv.setContentsMargins(20, 20, 20, 20)
        cv.setSpacing(4)
        cv.addWidget(label("Train a new model", 16, t["text"], 600))
        cap = label("Fine-tune SAM with LoRA from a single annotated image — no ML setup, minutes on this device.",
                    13, t["text_muted"])
        cap.setWordWrap(True)
        cv.addWidget(cap)
        cv.addSpacing(14)

        form = QGridLayout()
        form.setSpacing(14)
        image_box = SelectBox(self._image_field_text(), t, lead_icon="image", on_click=self._pick_image)
        backbones = self._train.available_backbones()
        if backbones:
            options = [lbl for _, lbl in backbones] + ["Browse…"]
            backbone_box = SelectBox(self._backbone_field_text(), t,
                                      options=options, on_select=self._on_backbone_menu_choice)
        else:
            backbone_box = SelectBox(self._backbone_field_text(), t, on_click=self._pick_backbone_file)
        rank_box = SelectBox(self._lora_rank, t, options=tc.RANK_OPTIONS, on_select=self._set_rank)
        epochs_box = SelectBox(self._epochs, t, options=tc.EPOCH_OPTIONS, on_select=self._set_epochs)
        fields = [("Annotated image", image_box), ("SAM backbone", backbone_box),
                  ("LoRA rank", rank_box), ("Epochs", epochs_box)]
        for i, (fname, control) in enumerate(fields):
            fc = QVBoxLayout()
            fc.setSpacing(7)
            fc.addWidget(GroupLabel(fname, t))
            fc.addWidget(control)
            form.addLayout(fc, i // 2, i % 2)
        cv.addLayout(form)

        status = self._status_text()
        if status:
            cv.addSpacing(4)
            st_lbl = label(status, 11.5, t["warning"])
            st_lbl.setWordWrap(True)
            cv.addWidget(st_lbl)

        cv.addSpacing(16)
        if self._train.is_training():
            btn = PillButton("Stop training", t, "danger")
            btn.clicked.connect(self._stop_training)
        else:
            btn = PillButton("Start training", t, "primary", "run")
            btn.setEnabled(self._can_start())
            btn.clicked.connect(self._start_training)
        cv.addWidget(btn, alignment=Qt.AlignmentFlag.AlignLeft)
        return card

    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an annotated image", "",
            "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp *.npy);;All files (*)", options=_DLG)
        if not path:
            return
        self._image_path = Path(path)
        self._mask_path = self._train.find_mask_for_image(self._image_path)
        self.refresh()

    def _on_backbone_menu_choice(self, choice: str) -> None:
        if choice == "Browse…":
            self._pick_backbone_file()
            return
        inverse = {v: k for k, v in tc.BACKBONE_LABELS.items()}
        self._vit_name = inverse.get(choice, self._vit_name)
        self._backbone_path = None
        self.refresh()

    def _pick_backbone_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a SAM backbone checkpoint", "",
            "PyTorch (*.pth);;All files (*)", options=_DLG)
        if not path:
            return
        self._backbone_path = Path(path)
        self._vit_name = tc.guess_vit_name(self._backbone_path)
        self.refresh()

    def _set_rank(self, value: str) -> None:
        self._lora_rank = value
        self.refresh()

    def _set_epochs(self, value: str) -> None:
        self._epochs = value
        self.refresh()

    def _start_training(self) -> None:
        try:
            config = self._train.build_config(
                image_path=self._image_path, mask_path=self._mask_path,
                vit_name=self._vit_name, backbone_path=self._backbone_path,
                lora_rank=int(self._lora_rank), epochs=int(self._epochs))
        except ValueError as e:
            self._toast("Can't start training", str(e))
            return
        self._train.start_training(config, on_log=self._safe_emit_log, on_finish=self._safe_emit_finish)
        self._toast("Training started", Path(config["result_pth_path"]).stem)
        self.refresh()

    def _stop_training(self) -> None:
        self._train.stop_training()
        self.refresh()

    # ── Command palette integration ─────────────────────────────────────────
    # Thin aliases over already-self-guarding private methods — the ⌘K
    # palette's equivalent of clicking Start/Stop/Import on this tab,
    # callable regardless of which tab is actually visible.
    def start_training(self) -> None:
        self._start_training()

    def stop_training(self) -> None:
        self._stop_training()

    def import_model(self) -> None:
        self._import_model()

    def _safe_emit_log(self, msg: str) -> None:
        # The training thread outlives this screen across a theme toggle
        # (which tears the old screen down and builds a fresh one) — guard
        # the cross-thread signal emit the same way motion.py guards a
        # stale hover callback touching a deleted widget.
        try:
            self._log_signal.emit(msg)
        except RuntimeError:
            pass

    def _safe_emit_finish(self) -> None:
        try:
            self._finish_signal.emit()
        except RuntimeError:
            pass

    def _on_log(self, msg: str) -> None:
        # train_model's real progress/error stream -- previously only
        # skimmed for a [ERROR] toast, now also reaches the Logs console.
        emit_prefixed(get_log_bus(), msg, source="studio.train")
        if msg.startswith("[ERROR]"):
            self._toast("Training failed", msg[len("[ERROR] "):])

    def _on_training_finished(self) -> None:
        models = self._train.list_trained_models()
        if models:
            self._toast("Training complete", models[0].name)
        self.refresh()

    def _import_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import a trained checkpoint", "", "PyTorch (*.pth);;All files (*)", options=_DLG)
        if not path:
            return
        try:
            dst = self._train.import_model(path)
        except ValueError as e:
            self._toast("Import failed", str(e))
            return
        self._toast("Model imported", dst.stem)
        self.refresh()

    def _model_row(self, model: TrainedModel) -> QFrame:
        t = self._t
        row = QFrame()
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified -- see components.EngineChip's comment. The badge icon
        # and the F1 QLabel below both set only `background`/`color`, never
        # `border`, so an unscoped QFrame{...} rule here leaked this row's
        # border onto each individually. Shared objectName across every
        # row (one per trained model) -- QSS matches by name, not
        # uniqueness, same as e.g. screens.py's "PCard"/"RRow".
        row.setObjectName("ModelRow")
        row.setStyleSheet(
            f"QFrame#ModelRow{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:10px;}}"
            f"QFrame#ModelRow:hover{{border-color:{t['border_strong']};}}")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(13)
        badge = QLabel()
        badge.setFixedSize(38, 38)
        badge.setPixmap(icons.pixmap("models", t["primary"], 18))
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(f"background:{t['primary_weak']}; border-radius:9px;")
        lay.addWidget(badge)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(label(model.name, 13.5, t["text"], 600))
        col.addWidget(label(model.meta, 11.5, t["text_muted"]))
        lay.addLayout(col, 1)
        f1col = QVBoxLayout()
        f1col.setSpacing(1)
        v = QLabel(model.f1 if model.f1 is not None else "—")
        f1_color = t["success"] if model.f1 is not None else t["text_muted"]
        v.setStyleSheet(f"color:{f1_color}; font-family:{theme.MONO}; font-size:14px; font-weight:600;")
        v.setAlignment(Qt.AlignmentFlag.AlignRight)
        f1col.addWidget(v)
        f1col.addWidget(label("F1", 10, t["text_muted"], 600, 0.5), alignment=Qt.AlignmentFlag.AlignRight)
        lay.addLayout(f1col)
        row.mouseReleaseEvent = lambda e, m=model: self._select_model(m)
        return row

    def _select_model(self, model: TrainedModel) -> None:
        project = self._projects.get_active()
        if project is None:
            self._toast("No active project", "Open or create a project first, then select a model.")
            return
        self._train.select_model_for_project(model, project)
        self._projects.store.save(project)
        self._toast("Model selected", f"“{model.name}” → {project.name}")

    # ── aside: recent runs ───────────────────────────────────────────────────
    def _aside(self) -> QVBoxLayout:
        t = self._t
        col = QVBoxLayout()
        col.setSpacing(16)
        card = QFrame()
        card.setFixedWidth(320)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setObjectName("RecentRunsCard")   # qualified -- see _train_card's comment
        card.setStyleSheet(f"QFrame#RecentRunsCard{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;}}")
        soft_shadow(card, 14, 20, 3)
        cv = QVBoxLayout(card)
        cv.setContentsMargins(16, 16, 16, 16)
        cv.setSpacing(10)
        cv.addWidget(label("Recent training runs", 13.5, t["text"], 600))

        runs = list(self._train.list_recent_runs())
        live = self._train.current_run()
        if live is not None:
            runs.insert(0, live)
        if not runs:
            cap = label("No runs yet.", 12, t["text_muted"])
            cv.addWidget(cap)
        for run in runs:
            r = QHBoxLayout()
            r.setSpacing(10)
            dot = QLabel()
            dot.setFixedSize(8, 8)
            dcol = t["signal"] if run.state == "run" else t["success"]
            dot.setStyleSheet(f"background:{dcol}; border-radius:4px;")
            r.addWidget(dot)
            c = QVBoxLayout()
            c.setSpacing(1)
            c.addWidget(label(run.name, 12.5, t["text"], 600))
            c.addWidget(label(run.meta, 11, t["text_muted"]))
            r.addLayout(c, 1)
            cv.addLayout(r)
        col.addWidget(card)

        tip = QFrame()
        tip.setFixedWidth(320)
        tip.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        tip.setObjectName("TrainTip")   # qualified -- see _train_card's comment
        tip.setStyleSheet(f"QFrame#TrainTip{{background:{t['primary_weak']}; border:1px solid {t['primary_line']}; border-radius:14px;}}")
        tv = QVBoxLayout(tip)
        tv.setContentsMargins(16, 16, 16, 16)
        tv.setSpacing(6)
        tv.addWidget(label("✦ One-shot fine-tuning", 13, t["primary"], 600))
        p = label("CellSeg1 specialises SAM to your assay from a single annotated field — the rest of the cohort inherits it.",
                  12.5, t["text_subtle"])
        p.setWordWrap(True)
        tv.addWidget(p)
        col.addWidget(tip)
        col.addStretch(1)
        return col


# ── Dashboard ────────────────────────────────────────────────────────────────
class _LineChart(QWidget):
    def __init__(self, data, color: str, t: dict):
        super().__init__()
        self._data = data
        self._color = color
        self._t = t
        self.setMinimumHeight(120)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h, pad = self.width(), self.height(), 8
        p.setPen(QPen(QColor(self._t["border"]), 1))
        for g in range(4):
            y = pad + (h - 2 * pad) * g / 3
            p.drawLine(0, int(y), w, int(y))
        data = self._data
        mn, mx = min(data), max(data)
        rng = (mx - mn) or 1

        def X(i):
            return pad + (w - 2 * pad) * i / (len(data) - 1)

        def Y(v):
            return pad + (h - 2 * pad) * (1 - (v - mn) / rng)
        path = QPainterPath()
        for i, v in enumerate(data):
            pt = QPointF(X(i), Y(v))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        col = QColor(self._color)
        fill = QColor(self._color)
        fill.setAlpha(34)
        area = QPainterPath(path)
        area.lineTo(X(len(data) - 1), h - pad)
        area.lineTo(X(0), h - pad)
        area.closeSubpath()
        p.fillPath(area, fill)
        p.setPen(QPen(col, 2))
        p.drawPath(path)
        p.setBrush(col)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(X(len(data) - 1), Y(data[-1])), 3, 3)
        p.end()


class _BarChart(QWidget):
    def __init__(self, data, color: str, t: dict):
        super().__init__()
        self._data = data
        self._color = color
        self._t = t
        self.setMinimumHeight(120)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        bw = w / len(self._data)
        base = QColor(self._color)
        faded = QColor(self._color)
        faded.setAlpha(102)
        for i, v in enumerate(self._data):
            bh = max(4, v * (h - 10))
            p.setBrush(base if i == len(self._data) - 1 else faded)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(i * bw + 4, h - bh, bw - 8, bh), 3, 3)
        p.end()


class DashboardScreen(QWidget):
    """Real experiment tracking, bound to a ``DashboardController``
    (``studio/dashboard_controller.py``) — the training-loss chart, the
    F1-across-runs chart, and the runs table all read real on-disk data
    (training history, per-checkpoint sidecars, benchmarked project stats).
    "Open in Aim" launches the real Aim dashboard server in the system
    browser (see the controller module docstring for why Studio's own charts
    don't query Aim's storage directly).
    """

    def __init__(self, t: dict, train_controller: TrainController,
                 project_controller: ProjectController,
                 on_toast: Callable[[str, str], None]):
        super().__init__()
        self._t = t
        self._dashboard = DashboardController(train_controller, project_controller)
        self._toast = on_toast
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild from the controller's current state — called on tab
        navigation (``app.py``'s ``navigate()``), so switching here after
        training or benchmarking elsewhere always shows fresh data."""
        while self._outer.count():
            item = self._outer.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        t = self._t
        open_btn = PillButton("Open in Aim", t, "ghost", "settings")
        open_btn.clicked.connect(self._open_in_aim)
        self._outer.addWidget(page_header("Dashboard", "Experiment tracking · embedded Aim", t, open_btn))

        body = bare_widget()
        v = QVBoxLayout(body)
        v.setContentsMargins(34, 4, 34, 40)
        v.setSpacing(16)
        charts = QHBoxLayout()
        charts.setSpacing(16)
        losses, loss_caption = self._dashboard.loss_curve()
        charts.addWidget(self._chart_card(
            "Training loss", loss_caption or "No training runs yet",
            self._chart_or_empty(_LineChart, losses, t["primary"], "No runs yet")))
        bars = self._dashboard.f1_bars()
        charts.addWidget(self._chart_card(
            "F1 across runs", "held-out validation" if bars else "No benchmarked runs yet",
            self._chart_or_empty(_BarChart, bars, t["signal"], "No runs yet")))
        v.addLayout(charts)
        v.addWidget(self._runs_table())
        v.addStretch(1)
        self._outer.addWidget(scroll(body))

    def _chart_or_empty(self, cls, data: list[float], color: str, empty_msg: str) -> QWidget:
        if not data:
            w = QWidget()
            w.setMinimumHeight(120)
            lay = QVBoxLayout(w)
            lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl = label(empty_msg, 12, self._t["text_muted"])
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(lbl)
            return w
        return cls(data, color, self._t)

    def _open_in_aim(self) -> None:
        try:
            url = self._dashboard.open_in_aim()
        except RuntimeError as e:
            self._toast("Aim isn't installed", str(e))
            return
        webbrowser.open(url)
        self._toast("Opening Aim", url)

    def open_in_aim(self) -> None:
        """Public alias for the ⌘K palette — same convention as
        ModelsScreen's start_training()/stop_training()/import_model()."""
        self._open_in_aim()

    def _chart_card(self, title: str, cap: str, chart: QWidget) -> QFrame:
        t = self._t
        card = QFrame()
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified -- see ModelsScreen._train_card's comment. Shared
        # objectName across both callers (loss chart + F1 chart).
        card.setObjectName("ChartCard")
        card.setStyleSheet(f"QFrame#ChartCard{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;}}")
        soft_shadow(card, 14, 20, 3)
        v = QVBoxLayout(card)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(2)
        v.addWidget(label(title, 13.5, t["text"], 600))
        v.addWidget(label(cap, 11.5, t["text_muted"]))
        v.addSpacing(10)
        v.addWidget(chart)
        return card

    def _runs_table(self) -> QFrame:
        t = self._t
        card = QFrame()
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setObjectName("RunsTableCard")   # qualified -- see _train_card's comment
        card.setStyleSheet(f"QFrame#RunsTableCard{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;}}")
        soft_shadow(card, 14, 20, 3)
        v = QVBoxLayout(card)
        v.setContentsMargins(18, 16, 18, 8)
        v.setSpacing(2)
        v.addWidget(label("Runs", 13.5, t["text"], 600))
        runs = self._dashboard.runs_table()
        v.addWidget(label(f"{len(runs)} tracked run{'s' if len(runs) != 1 else ''}", 11.5, t["text_muted"]))
        v.addSpacing(8)
        header = ["Run", "Engine", "F1", "Cells", "Duration", "When"]
        hrow = QHBoxLayout()
        for i, hcol in enumerate(header):
            l = label(hcol.upper(), 10, t["text_muted"], 600, 0.5)
            hrow.addWidget(l, 2 if i == 0 else 1)
        v.addLayout(hrow)
        v.addWidget(hline(t))
        if not runs:
            cap = label("No runs yet — train a model or benchmark a project to see them here.",
                        12, t["text_muted"])
            cap.setWordWrap(True)
            v.addWidget(cap)
        for run in runs:
            r = QHBoxLayout()
            cells_map = [(run.name, "mono"), (run.engine_label, ""),
                         (run.f1 or "—", "ok" if run.ok else "mono"),
                         (run.cells or "—", "mono"), (run.duration or "—", "mono"), (run.when, "")]
            for i, (val, style) in enumerate(cells_map):
                if style == "mono":
                    col = t["text_subtle"]
                    fam = f"font-family:{theme.MONO};"
                elif style == "ok":
                    col = t["success"]
                    fam = f"font-family:{theme.MONO}; font-weight:600;"
                else:
                    col = t["text_subtle"]
                    fam = ""
                l = QLabel(val)
                l.setStyleSheet(f"color:{col}; font-size:12.5px; {fam}")
                r.addWidget(l, 2 if i == 0 else 1)
            rowwrap = bare_widget()
            rowwrap.setLayout(r)
            r.setContentsMargins(0, 8, 0, 8)
            v.addWidget(rowwrap)
            v.addWidget(hline(t))
        return card
