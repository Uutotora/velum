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
from PyQt6.QtGui import (QPainter, QColor, QPen, QPolygonF, QPainterPath,
                         QLinearGradient, QFont)
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QScrollArea,
    QFileDialog,
)

from studio import icons
from studio import theme
from studio.components import (
    Chip, Badge, PillButton, SelectBox, GroupLabel, SegControl, Accordion,
    StatTile, hline, soft_shadow, label,
)
from studio.screens import page_header, scroll
from studio import train_controller as tc
from studio.train_controller import TrainController, TrainedModel
from studio.project_controller import ProjectController
from studio.dashboard_controller import DashboardController
from studio.components import bare_widget
from studio.log_bus import get_log_bus, emit_prefixed
from studio.model_library_controller import ModelLibraryController
from studio.model_library_screen import ModelCatalogView

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
                 on_toast: Callable[[str, str], None],
                 library_controller: Optional[ModelLibraryController] = None):
        super().__init__()
        self._t = t
        self._train = train_controller
        self._projects = project_controller
        self._toast = on_toast
        # The Model Library catalog used to be a separate top-level tab; it now
        # lives here as the "Library" section, so everything about models —
        # download, train, your models, engines — is in one place.
        self._library = library_controller or ModelLibraryController()

        self._image_path: Optional[Path] = None
        self._mask_path: Optional[Path] = None
        backbones = self._train.available_backbones()
        self._vit_name: Optional[str] = backbones[0][0] if backbones else None
        self._backbone_path: Optional[Path] = None
        self._lora_rank = tc.DEFAULT_RANK
        self._epochs = tc.DEFAULT_EPOCHS
        # Which section is showing: 0 Train · 1 Library · 2 My models ·
        # 3 Engines. Train is the default so every pre-existing test (which
        # calls refresh() without touching this) still renders the form.
        self._section = 0

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
        self._outer.addWidget(page_header(
            "Models & Train",
            f"{n} trained adapter{'s' if n != 1 else ''} · one-shot LoRA fine-tuning",
            t, self._header_action()))
        self._outer.addWidget(self._section_bar())

        if self._section == 1:
            body = self._library_body()
        elif self._section == 2:
            body = self._models_body(models)
        elif self._section == 3:
            body = self._engines_body()
        else:
            body = self._train_body(models)
        scroll_area = scroll(body)
        self._outer.addWidget(scroll_area)
        if scroll_pos is not None:
            QTimer.singleShot(0, lambda v=scroll_pos: scroll_area.verticalScrollBar().setValue(v))

        if self._train.is_training():
            self._live_timer.start()

    # ── section navigation ──────────────────────────────────────────────────
    def _section_bar(self) -> QWidget:
        """The Train · Library · My models · Engines segmented switch, in its
        own row under the page header."""
        t = self._t
        wrap = bare_widget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(34, 2, 34, 14)
        seg = SegControl(["Train", "Library", "My models", "Engines"], t, active=self._section)
        seg.setFixedWidth(420)
        seg.changed.connect(self._set_section)
        row.addWidget(seg)
        row.addStretch(1)
        return wrap

    def _set_section(self, idx: int) -> None:
        self._section = idx
        self.refresh()

    def open_library(self) -> None:
        """Public entry (⌘K) — show the Library section."""
        self._set_section(1)

    def _header_action(self) -> Optional[QWidget]:
        """The header's right-hand action, chosen per section — Import model on
        the Library and My models sections, nothing on Train (its primary
        action is Start training, in the card) or Engines (which has its own
        Register button in its intro card)."""
        t = self._t
        if self._section in (1, 2):
            btn = PillButton("Import model", t, "ghost", "download")
            btn.clicked.connect(self._import_model)
            return btn
        return None

    # ── Library section body (the model catalog) ─────────────────────────────
    def _library_body(self) -> QWidget:
        return ModelCatalogView(self._t, self._library, self._toast)

    # ── Train section body ───────────────────────────────────────────────────
    def _train_body(self, models: list[TrainedModel]) -> QWidget:
        body = bare_widget()
        row = QHBoxLayout(body)
        row.setContentsMargins(34, 4, 34, 40)
        row.setSpacing(24)
        row.addLayout(self._left(models), 1)
        row.addLayout(self._aside(), 0)
        return body

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

    # ── left: live progress (while training) + guided train form ─────────────
    def _left(self, models: list[TrainedModel]) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(20)
        if self._train.is_training():
            col.addWidget(self._live_progress_card())
        col.addWidget(self._train_card())
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
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_row.addWidget(label("Train a new model", 16, t["text"], 600))
        title_row.addWidget(Chip("One-shot LoRA", t, "primary"), alignment=Qt.AlignmentFlag.AlignVCenter)
        title_row.addStretch(1)
        cv.addLayout(title_row)
        cap = label("Fine-tune SAM to your assay from a single annotated image — no ML setup, minutes on this device.",
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
        fields = [("Annotated image", image_box,
                   "PNG / TIFF / JPG (grayscale or RGB). Needs a label mask "
                   "named <name>_mask.png beside it — one integer id per cell."),
                  ("SAM backbone", backbone_box,
                   "The base SAM weights LoRA fine-tunes. Auto-detected in "
                   "sam_backbone/, or browse for a .pth."),
                  ("LoRA rank", rank_box, tc.rank_help(self._lora_rank)),
                  ("Epochs", epochs_box, tc.epoch_help(self._epochs))]
        for i, (fname, control, hint) in enumerate(fields):
            fc = QVBoxLayout()
            fc.setSpacing(6)
            fc.addWidget(GroupLabel(fname, t))
            fc.addWidget(control)
            if hint:
                hl = label(hint, 11, t["text_muted"])
                hl.setWordWrap(True)
                fc.addWidget(hl)
            fc.addStretch(1)
            form.addLayout(fc, i // 2, i % 2)
        cv.addLayout(form)

        cv.addSpacing(14)
        cv.addWidget(self._checklist())

        status = self._status_text()
        if status:
            cv.addSpacing(8)
            st_lbl = label(status, 11.5, t["warning"])
            st_lbl.setWordWrap(True)
            cv.addWidget(st_lbl)

        cv.addSpacing(16)
        if self._train.is_training():
            btn = PillButton("Stop training", t, "danger")
            btn.clicked.connect(self._stop_training)
            cv.addWidget(btn, alignment=Qt.AlignmentFlag.AlignLeft)
        else:
            bottom = QHBoxLayout()
            bottom.setSpacing(12)
            btn = PillButton("Start training", t, "primary", "run")
            btn.setEnabled(self._can_start())
            btn.clicked.connect(self._start_training)
            bottom.addWidget(btn)
            dev = label(f"Runs on {self._train.detected_device_label()}", 11.5, t["text_muted"])
            bottom.addWidget(dev)
            bottom.addStretch(1)
            cv.addLayout(bottom)
        return card

    # ── guided readiness checklist ───────────────────────────────────────────
    def _checklist(self) -> QFrame:
        """A small "assistant" strip: the three things that must be true before
        training can start, each ticking green as it's satisfied — so the user
        always knows what the next step is instead of a greyed-out button."""
        t = self._t
        has_image = self._image_path is not None
        has_mask = self._mask_path is not None
        n = tc.count_cells_in_mask(self._mask_path) if has_mask else None
        if not has_image:
            img_state, img_text = "todo", "Pick an annotated image to fine-tune on"
        elif not has_mask:
            img_state, img_text = "warn", f"{self._image_path.name} — no mask found yet"
        else:
            cells = f" · {n} cells" if n is not None else ""
            img_state, img_text = "done", f"{self._image_path.name}{cells}"

        if self._has_backbone():
            bb_state, bb_text = "done", f"SAM backbone · {self._backbone_field_text()}"
        else:
            bb_state, bb_text = "todo", "Choose a SAM backbone checkpoint"

        if self._can_start():
            rd_state = "done"
            rd_text = f"Ready — rank {self._lora_rank}, {self._epochs} epochs on {self._train.detected_device_label()}"
        else:
            rd_state, rd_text = "todo", "Ready to train once the steps above are met"

        panel = QFrame()
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setObjectName("TrainChecklist")
        panel.setStyleSheet(f"QFrame#TrainChecklist{{background:{t['inset']}; border:1px solid {t['border']}; border-radius:10px;}}")
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(14, 12, 14, 12)
        pv.setSpacing(9)
        pv.addWidget(self._check_row(img_state, img_text))
        pv.addWidget(self._check_row(bb_state, bb_text))
        pv.addWidget(self._check_row(rd_state, rd_text))
        return panel

    def _check_row(self, state: str, text: str) -> QWidget:
        t = self._t
        color = {"done": t["success"], "warn": t["warning"], "todo": t["text_muted"]}[state]
        icon_name = {"done": "check", "warn": "diagnose", "todo": "target"}[state]
        w = bare_widget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(9)
        ic = QLabel()
        ic.setPixmap(icons.pixmap(icon_name, color, 14))
        ic.setFixedWidth(16)
        row.addWidget(ic)
        txt_color = t["text"] if state == "done" else t["text_subtle"] if state == "warn" else t["text_muted"]
        lb = label(text, 12, txt_color, 600 if state == "done" else 500)
        lb.setWordWrap(True)
        row.addWidget(lb, 1)
        return w

    # ── live training progress ───────────────────────────────────────────────
    def _live_progress_card(self) -> QFrame:
        """The at-a-glance "watching over the run" panel shown while training —
        epoch progress bar, current loss, and a live loss sparkline. Reads the
        same drained progress the aside's run row does."""
        t = self._t
        run = self._train.current_run()
        history = self._train.current_loss_history()
        total = self._train.active_epoch_max() or 0
        epoch = history[-1]["epoch"] if history else 0
        loss = history[-1]["loss"] if history else None
        frac = (epoch / total) if total else 0.0

        card = QFrame()
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setObjectName("LiveTrainCard")
        card.setStyleSheet(f"QFrame#LiveTrainCard{{background:{t['surface']}; border:1px solid {t['primary_line']}; border-radius:14px;}}")
        soft_shadow(card, 16, 26, 3)
        cv = QVBoxLayout(card)
        cv.setContentsMargins(20, 18, 20, 18)
        cv.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{t['signal']}; border-radius:4px;")
        head.addWidget(dot)
        head.addWidget(label("Training in progress", 14, t["text"], 600))
        head.addStretch(1)
        head.addWidget(Chip(f"epoch {epoch}/{total}" if total else "starting…", t, "signal"))
        cv.addLayout(head)
        name = self._train._active_name or (run.name if run else "run")
        cv.addWidget(label(name, 11.5, t["text_muted"]))

        cv.addWidget(_ProgressBar(frac, t))

        stats = QHBoxLayout()
        stats.setSpacing(10)
        stats.addWidget(StatTile(f"{epoch}", f"/ {total}" if total else "", "EPOCH", t))
        stats.addWidget(StatTile(f"{loss:.3f}" if loss is not None else "—", "", "LOSS", t))
        stats.addWidget(StatTile(f"{int(frac*100)}", "%", "COMPLETE", t))
        cv.addLayout(stats)

        if len(history) >= 2:
            cv.addSpacing(2)
            cv.addWidget(label("Loss", 10.5, t["text_muted"], 600, 0.4))
            chart = _LineChart([h["loss"] for h in history], t["primary"], t)
            chart.setMinimumHeight(90)
            cv.addWidget(chart)
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

    # ── My models section ────────────────────────────────────────────────────
    def _models_body(self, models: list[TrainedModel]) -> QWidget:
        t = self._t
        body = bare_widget()
        v = QVBoxLayout(body)
        v.setContentsMargins(34, 4, 34, 40)
        v.setSpacing(14)

        head = QHBoxLayout()
        head.addWidget(label("Your trained models", 15, t["text"], 600))
        head.addStretch(1)
        head.addWidget(label(
            "Click a model to use it in the active project.", 12, t["text_muted"]))
        v.addLayout(head)

        if models:
            for m in models:
                v.addWidget(self._model_row(m))
        else:
            v.addWidget(self._models_empty_state())
        v.addStretch(1)
        return body

    def _models_empty_state(self) -> QFrame:
        t = self._t
        card = QFrame()
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setObjectName("ModelsEmpty")
        card.setStyleSheet(f"QFrame#ModelsEmpty{{background:{t['surface']}; border:1px dashed {t['border_strong']}; border-radius:14px;}}")
        # Centre a fixed-width inner column: a word-wrapped QLabel added to a
        # box layout *with an alignment flag* doesn't get its heightForWidth
        # honoured (the caption collapsed to one line and the buttons drew over
        # it — the "кривой текст" bug). Giving the column a fixed width lets the
        # label wrap and reserve its real height without any alignment flag.
        outer = QHBoxLayout(card)
        outer.setContentsMargins(28, 34, 28, 34)
        outer.addStretch(1)
        inner = bare_widget()
        inner.setFixedWidth(460)
        cv = QVBoxLayout(inner)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(9)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("models", t["text_muted"], 30))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cv.addWidget(ic)
        h = label("No trained models yet", 15, t["text"], 600)
        h.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cv.addWidget(h)
        cap = label("Fine-tune one from a single annotated image, or import an existing "
                    "checkpoint — both land here for you to reuse across projects.",
                    12.5, t["text_muted"])
        cap.setWordWrap(True)
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cv.addWidget(cap)
        cv.addSpacing(6)
        cta_row = QHBoxLayout()
        cta_row.setSpacing(10)
        cta_row.addStretch(1)
        train_btn = PillButton("Train a model", t, "primary", "run")
        train_btn.clicked.connect(lambda: self._set_section(0))
        import_btn = PillButton("Import checkpoint…", t, "ghost", "download")
        import_btn.clicked.connect(self._import_model)
        cta_row.addWidget(train_btn)
        cta_row.addWidget(import_btn)
        cta_row.addStretch(1)
        cv.addLayout(cta_row)
        outer.addWidget(inner)
        outer.addStretch(1)
        return card

    # ── Engines section ──────────────────────────────────────────────────────
    def _engines_body(self) -> QWidget:
        t = self._t
        body = bare_widget()
        v = QVBoxLayout(body)
        v.setContentsMargins(34, 4, 34, 40)
        v.setSpacing(16)

        v.addWidget(self._engines_intro())

        engines = self._train.list_engines()
        grid = QGridLayout()
        grid.setSpacing(14)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for i, info in enumerate(engines):
            grid.addWidget(self._engine_card(info), i // 2, i % 2)
        v.addLayout(grid)

        v.addWidget(self._engine_help())
        v.addStretch(1)
        return body

    def _engines_intro(self) -> QFrame:
        t = self._t
        card = QFrame()
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setObjectName("EnginesIntro")
        card.setStyleSheet(f"QFrame#EnginesIntro{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;}}")
        soft_shadow(card, 14, 20, 3)
        row = QHBoxLayout(card)
        row.setContentsMargins(20, 16, 20, 16)
        row.setSpacing(14)
        badge = QLabel()
        badge.setFixedSize(40, 40)
        badge.setPixmap(icons.pixmap("cube3d", t["primary"], 20))
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(f"background:{t['primary_weak']}; border-radius:10px;")
        row.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(3)
        col.addWidget(label("Segmentation engines", 15, t["text"], 600))
        p = label("Every project runs on a segmentation engine. Velum ships three, and "
                  "your own drop in the same way — a Python plugin that registers itself. "
                  "Load one with “Register engine…” and it appears here and in the Segment "
                  "tab’s engine picker.", 12.5, t["text_subtle"])
        p.setWordWrap(True)
        col.addWidget(p)
        row.addLayout(col, 1)
        register = PillButton("Register engine…", t, "primary", "plus")
        register.clicked.connect(self._register_engine)
        row.addWidget(register, alignment=Qt.AlignmentFlag.AlignVCenter)
        return card

    def _engine_dot(self, key: str) -> str:
        return {"cellseg1": self._t["primary"], "cellpose": self._t["signal"],
                "sam2": self._t["success"]}.get(key, self._t["warning"])

    def _engine_card(self, info) -> QFrame:
        t = self._t
        card = QFrame()
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setObjectName("EngineCard")
        card.setStyleSheet(
            f"QFrame#EngineCard{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:12px;}}"
            f"QFrame#EngineCard:hover{{border-color:{t['border_strong']};}}")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(16, 14, 16, 14)
        cv.setSpacing(9)
        top = QHBoxLayout()
        top.setSpacing(9)
        dot = QFrame()
        dot.setFixedSize(9, 9)
        dot.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dot.setStyleSheet(f"background:{self._engine_dot(info.key)}; border-radius:4px;")
        top.addWidget(dot, alignment=Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(label(info.label, 13.5, t["text"], 600), 1)
        if info.custom:
            top.addWidget(Chip("Custom", t, "warning"))
        top.addWidget(Chip("Ready", t, "success") if info.available
                      else Chip("Not installed", t, "muted"))
        cv.addLayout(top)

        sub = info.status or (
            "Ready to run." if info.available
            else "Optional dependency not installed — install it to enable this engine.")
        sl = label(sub, 11.5, t["text_muted"] if info.available else t["text_subtle"])
        sl.setWordWrap(True)
        cv.addWidget(sl)

        if info.custom:
            foot = QHBoxLayout()
            foot.addStretch(1)
            remove = PillButton("Remove", t, "ghost", "trash")
            remove.clicked.connect(lambda _=False, k=info.key: self._remove_engine(k))
            foot.addWidget(remove)
            cv.addLayout(foot)
        return card

    def _engine_help(self) -> Accordion:
        t = self._t
        acc = Accordion("How do I add my own engine?", t, lead="guide",
                        caps=False, fill="surface2")
        body = label(
            "An engine turns an image into a label mask. Velum ships three — "
            "CellSeg1 (SAM + LoRA, one-shot), Cellpose-SAM (zero-shot), and "
            "SAM 2 (z-stacks / video) — and your own plug in the same way.\n\n"
            "A plugin is one Python (.py) file. When imported it calls "
            "velum_core.engine_registry.register() with an EngineSpec: a stable "
            "key, a display label, a predict(image, config) → int-label-mask "
            "function (each cell a distinct id, 0 = background), and an optional "
            "available() probe. “Register engine…” imports the file, registers "
            "whatever it declares, and remembers it for the next launch — it "
            "then shows here and in the Segment tab’s engine picker. Nothing is "
            "compiled or packaged; import any library you like inside predict().",
            12.5, t["text_subtle"])
        body.setWordWrap(True)
        acc.add(body)
        snippet = QLabel(
            "# my_engine.py  →  Register engine…\n"
            "import numpy as np\n"
            "from velum_core.engine_registry import EngineSpec, register\n\n"
            "def predict(image, config):\n"
            "    # image: HxWx3 uint8 → return an int label mask (HxW),\n"
            "    # one id per cell, 0 = background\n"
            "    return np.zeros(image.shape[:2], dtype=np.int32)\n\n"
            "register(EngineSpec(\n"
            "    key=\"my_engine\", label=\"My Engine\",\n"
            "    predict=predict, available=lambda: True))")
        snippet.setStyleSheet(
            f"color:{t['text_subtle']}; background:{t['inset']}; border:1px solid {t['border']};"
            f"border-radius:8px; padding:12px 14px; font-family:{theme.MONO}; font-size:11.5px;")
        acc.add(snippet)
        return acc

    def _register_engine(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Register an engine plugin", "", "Python (*.py);;All files (*)", options=_DLG)
        if not path:
            return
        try:
            keys = self._train.add_custom_engine(path)
        except ValueError as e:
            self._toast("Couldn't register engine", str(e))
            return
        self._toast("Engine registered", ", ".join(keys))
        self.refresh()

    def _remove_engine(self, key: str) -> None:
        self._train.remove_custom_engine(key)
        self._toast("Engine removed", key)
        self.refresh()


# ── small widgets ─────────────────────────────────────────────────────────────
class _ProgressBar(QWidget):
    """A slim rounded progress bar (0..1) — the live training epoch fill."""

    def __init__(self, frac: float, t: dict):
        super().__init__()
        self._frac = max(0.0, min(1.0, frac))
        self._t = t
        self.setFixedHeight(8)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self._t["inset"]))
        p.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)
        fw = max(h, w * self._frac)
        p.setBrush(QColor(self._t["primary"]))
        p.drawRoundedRect(QRectF(0, 0, fw, h), h / 2, h / 2)
        p.end()


# ── Dashboard ────────────────────────────────────────────────────────────────
def _smooth_path(pts: list[QPointF]) -> QPainterPath:
    """Catmull-Rom → cubic-Bézier smoothing through ``pts`` — the soft,
    analytics-grade curve (vs. the old jagged polyline) the loss chart draws."""
    path = QPainterPath()
    if not pts:
        return path
    path.moveTo(pts[0])
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else p2
        c1 = QPointF(p1.x() + (p2.x() - p0.x()) / 6.0, p1.y() + (p2.y() - p0.y()) / 6.0)
        c2 = QPointF(p2.x() - (p3.x() - p1.x()) / 6.0, p2.y() - (p3.y() - p1.y()) / 6.0)
        path.cubicTo(c1, c2, p2)
    return path


def _round_top_rect(r: QRectF, radius: float) -> QPainterPath:
    """A rectangle with only its top corners rounded — a column that sits
    flush on the baseline but reads as a soft, modern bar up top."""
    rad = max(0.0, min(radius, r.width() / 2, r.height()))
    path = QPainterPath()
    path.moveTo(r.left(), r.bottom())
    path.lineTo(r.left(), r.top() + rad)
    path.quadTo(r.left(), r.top(), r.left() + rad, r.top())
    path.lineTo(r.right() - rad, r.top())
    path.quadTo(r.right(), r.top(), r.right(), r.top() + rad)
    path.lineTo(r.right(), r.bottom())
    path.closeSubpath()
    return path


class _LineChart(QWidget):
    """A gradient-filled, smoothed line chart with y-axis value labels and a
    glowing latest-value marker — the training-loss curve."""

    def __init__(self, data, color: str, t: dict):
        super().__init__()
        self._data = data
        self._color = color
        self._t = t
        self.setMinimumHeight(132)

    def paintEvent(self, e):
        data = self._data
        if not data:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        t = self._t
        W, H = self.width(), self.height()
        lpad, rpad, tpad, bpad = 42, 12, 12, 14
        plot_w = max(1.0, W - lpad - rpad)
        plot_h = max(1.0, H - tpad - bpad)
        mn, mx = min(data), max(data)
        if mx - mn < 1e-9:
            pad_v = abs(mx) * 0.1 or 1.0
            mn, mx = mn - pad_v, mx + pad_v
        rng = mx - mn

        def X(i):
            return lpad + plot_w * (i / (len(data) - 1) if len(data) > 1 else 0.5)

        def Y(v):
            return tpad + plot_h * (1 - (v - mn) / rng)

        def fmt(v):
            return f"{v:.2f}" if abs(mx) < 10 else f"{v:.1f}"

        # dashed gridlines + right-aligned y-axis value labels
        grid_pen = QPen(QColor(t["border"]), 1)
        grid_pen.setDashPattern([2, 4])
        lab_font = QFont(); lab_font.setPointSizeF(7.6)
        for k in range(4):
            frac = k / 3
            y = tpad + plot_h * frac
            p.setPen(grid_pen)
            p.drawLine(int(lpad), int(y), int(W - rpad), int(y))
            p.setPen(QColor(t["text_muted"]))
            p.setFont(lab_font)
            p.drawText(QRectF(0, y - 8, lpad - 8, 16),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                       fmt(mx - rng * frac))

        pts = [QPointF(X(i), Y(v)) for i, v in enumerate(data)]
        line = _smooth_path(pts)
        col = QColor(self._color)

        area = QPainterPath(line)
        area.lineTo(pts[-1].x(), tpad + plot_h)
        area.lineTo(pts[0].x(), tpad + plot_h)
        area.closeSubpath()
        grad = QLinearGradient(0, tpad, 0, tpad + plot_h)
        top = QColor(col); top.setAlpha(72)
        bot = QColor(col); bot.setAlpha(0)
        grad.setColorAt(0.0, top)
        grad.setColorAt(1.0, bot)
        p.fillPath(area, grad)

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(col, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.drawPath(line)

        end = pts[-1]
        p.setPen(Qt.PenStyle.NoPen)
        glow = QColor(col); glow.setAlpha(64)
        p.setBrush(glow); p.drawEllipse(end, 7.5, 7.5)
        p.setBrush(col); p.drawEllipse(end, 3.6, 3.6)
        p.setBrush(QColor(t["surface"])); p.drawEllipse(end, 1.5, 1.5)
        p.end()


class _BarChart(QWidget):
    """F1-across-runs columns: width-capped, gradient-filled, rounded-top bars
    with the value printed above each — a single run reads as one labelled
    column, never a panel-filling solid block."""

    def __init__(self, data, color: str, t: dict):
        super().__init__()
        self._data = data
        self._color = color
        self._t = t
        self.setMinimumHeight(132)

    def paintEvent(self, e):
        data = self._data
        if not data:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        t = self._t
        W, H = self.width(), self.height()
        lpad, rpad, tpad, bpad = 34, 14, 20, 6
        plot_w = max(1.0, W - lpad - rpad)
        plot_h = max(1.0, H - tpad - bpad)
        ymax = 1.0  # F1 lives in [0, 1] — an honest absolute axis

        # dashed gridlines + y labels at 0.0 / 0.5 / 1.0
        grid_pen = QPen(QColor(t["border"]), 1)
        grid_pen.setDashPattern([2, 4])
        lab_font = QFont(); lab_font.setPointSizeF(7.6)
        for gv in (0.0, 0.5, 1.0):
            y = tpad + plot_h * (1 - gv / ymax)
            p.setPen(grid_pen)
            p.drawLine(int(lpad), int(y), int(W - rpad), int(y))
            p.setPen(QColor(t["text_muted"]))
            p.setFont(lab_font)
            p.drawText(QRectF(0, y - 8, lpad - 8, 16),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                       f"{gv:.1f}")

        n = len(data)
        slot = plot_w / n
        barw = min(46.0, slot * 0.62)
        base = QColor(self._color)
        val_font = QFont(); val_font.setPointSizeF(8.4); val_font.setBold(True)
        for i, v in enumerate(data):
            is_last = i == n - 1
            cx = lpad + slot * (i + 0.5)
            bh = plot_h * (max(0.0, min(v, 1.0)) / ymax)
            y = tpad + plot_h - bh
            rect = QRectF(cx - barw / 2, y, barw, bh)
            grad = QLinearGradient(0, y, 0, tpad + plot_h)
            top = QColor(base); top.setAlpha(255 if is_last else 150)
            bot = QColor(base); bot.setAlpha(150 if is_last else 70)
            grad.setColorAt(0.0, top)
            grad.setColorAt(1.0, bot)
            p.fillPath(_round_top_rect(rect, 4.0), grad)
            p.setFont(val_font)
            p.setPen(QColor(t["text"] if is_last else t["text_subtle"]))
            p.drawText(QRectF(cx - 30, y - 17, 60, 14),
                       int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom),
                       f"{v:.2f}")

        p.setPen(QPen(QColor(t["border_strong"]), 1))
        p.drawLine(int(lpad), int(tpad + plot_h), int(W - rpad), int(tpad + plot_h))
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
        v.addWidget(self._kpi_row())
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
        v.addWidget(self._engine_comparison_card())
        v.addWidget(self._runs_table())
        v.addStretch(1)
        self._outer.addWidget(scroll(body))

    # ── KPI tiles ────────────────────────────────────────────────────────────
    def _kpi_row(self) -> QWidget:
        t = self._t
        s = self._dashboard.dash_summary()
        best = f"{s.best_f1:.2f}" if s.best_f1 is not None else "—"
        avg = f"{s.avg_f1:.2f}" if s.avg_f1 is not None else "—"
        from studio import project_controller as _pc
        tiles = [
            ("run", "primary", str(s.n_runs), "Tracked runs", "training + segmentation"),
            ("spark", "success", best, "Best F1", "highest benchmarked"),
            ("chart", "signal", avg, "Avg F1", f"over {s.n_benchmarked} benchmarked" if s.n_benchmarked else "none benchmarked yet"),
            ("target", "warning", _pc.format_count(s.n_cells), "Cells segmented", "across all projects"),
        ]
        w = bare_widget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)
        for icon_name, kind, value, cap, sub in tiles:
            row.addWidget(self._kpi_tile(icon_name, kind, value, cap, sub))
        return w

    def _kpi_tile(self, icon_name: str, kind: str, value: str, cap: str, sub: str) -> QFrame:
        t = self._t
        colm = {"primary": t["primary"], "signal": t["signal"], "warning": t["warning"], "success": t["success"]}
        weakm = {"primary": t["primary_weak"], "signal": t["signal_weak"], "warning": t["warning_weak"], "success": t["success_weak"]}
        card = QFrame()
        card.setObjectName("DashKpi")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(f"#DashKpi{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;}}")
        soft_shadow(card, 14, 20, 3)
        cv = QVBoxLayout(card)
        cv.setContentsMargins(15, 14, 15, 14)
        cv.setSpacing(3)
        badge = QLabel()
        badge.setFixedSize(30, 30)
        badge.setPixmap(icons.pixmap(icon_name, colm[kind], 16))
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(f"background:{weakm[kind]}; border-radius:8px;")
        cv.addWidget(badge)
        cv.addSpacing(4)
        val = QLabel(value)
        val.setStyleSheet(f"color:{t['text']}; font-family:{theme.MONO}; font-size:23px; font-weight:600; letter-spacing:-0.5px;")
        cv.addWidget(val)
        cv.addWidget(label(cap.upper(), 10.5, t["text_muted"], 600, 0.3))
        sub_lbl = label(sub, 11, t["text_muted"])
        sub_lbl.setWordWrap(True)
        cv.addWidget(sub_lbl)
        return card

    # ── engine comparison ────────────────────────────────────────────────────
    def _engine_comparison_card(self) -> QFrame:
        t = self._t
        card = QFrame()
        card.setObjectName("EngineCmpCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(f"QFrame#EngineCmpCard{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;}}")
        soft_shadow(card, 14, 20, 3)
        v = QVBoxLayout(card)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(2)
        v.addWidget(label("Engine comparison", 13.5, t["text"], 600))
        v.addWidget(label("average benchmarked F1 by engine, across your projects", 11.5, t["text_muted"]))
        v.addSpacing(10)
        rows = self._dashboard.engine_comparison()
        if not rows:
            v.addWidget(label("No projects yet — create one and pick an engine to compare.",
                              12, t["text_muted"]))
            return card
        dotcol = {"cellseg1": t["primary"], "cellpose": t["signal"], "sam2": t["success"]}
        for r in rows:
            line = QHBoxLayout()
            line.setSpacing(12)
            name_w = QWidget()
            name_w.setFixedWidth(150)
            nl = QHBoxLayout(name_w)
            nl.setContentsMargins(0, 0, 0, 0)
            nl.setSpacing(8)
            dot = QFrame()
            dot.setFixedSize(8, 8)
            dot.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            dot.setStyleSheet(f"background:{dotcol.get(r.engine_key, t['warning'])}; border-radius:4px;")
            nl.addWidget(dot)
            nl.addWidget(label(r.engine_label, 12.5, t["text"], 600), 1)
            line.addWidget(name_w)

            frac = (r.avg_f1 or 0.0)
            bar = _ProgressBar(frac, t)
            bar.setFixedHeight(8)
            line.addWidget(bar, 1)

            val = "F1 " + (f"{r.avg_f1:.2f}" if r.avg_f1 is not None else "—")
            meta = f"{val} · {r.n_projects} project{'s' if r.n_projects != 1 else ''}"
            ml = label(meta, 12, t["text_subtle"])
            ml.setStyleSheet(f"color:{t['text_subtle']}; font-family:{theme.MONO}; font-size:12px; background:transparent;")
            ml.setFixedWidth(150)
            ml.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            line.addWidget(ml)
            rowwrap = bare_widget()
            rowwrap.setLayout(line)
            line.setContentsMargins(0, 7, 0, 7)
            v.addWidget(rowwrap)
        return card

    def _chart_or_empty(self, cls, data: list[float], color: str, empty_msg: str) -> QWidget:
        if not data:
            t = self._t
            w = QWidget()
            w.setMinimumHeight(132)
            lay = QVBoxLayout(w)
            lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.setSpacing(8)
            icon = QLabel()
            icon.setFixedSize(34, 34)
            icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon.setPixmap(icons.pixmap("chart", t["text_muted"], 18))
            icon.setStyleSheet(f"background:{t['inset']}; border-radius:10px;")
            lay.addWidget(icon, 0, Qt.AlignmentFlag.AlignHCenter)
            lbl = label(empty_msg, 12, t["text_muted"])
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(lbl, 0, Qt.AlignmentFlag.AlignHCenter)
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
