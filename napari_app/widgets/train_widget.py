import os
import threading
from pathlib import Path

import numpy as np
import queue

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QFileDialog, QScrollArea,
    QTextEdit, QSizePolicy, QFrame, QAbstractSpinBox, QButtonGroup,
)
from napari_app.widgets.log_window import get_log_window
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from napari_app.core.train_state_manager import TrainingStateManager
from project_root import STORAGE_DIR
from napari_app.theme import (
    WIDGET_SS, BTN_PRIMARY, BTN_DANGER, BTN_SECONDARY, BTN_PRESET, BTN_BROWSE,
    BG, FG, BORDER, TEXT, ACCENT, DIM, LABEL, CONSOLE, MONO,
)
from napari_app.widgets.common import (
    section_header, divider as _divider, param_row as _param_row,
    CollapsibleSection, SectionCard, CollapsibleCard,
)
from napari_app import icons

TRAIN_IMAGE_DIR  = STORAGE_DIR / "train_images"
TRAIN_MASK_DIR   = STORAGE_DIR / "train_masks"
LORA_OUT_DIR     = STORAGE_DIR / "loras"
SAM_BACKBONE_DIR = STORAGE_DIR / "sam_backbone"

STATE_MANAGER = TrainingStateManager(str(STORAGE_DIR))
_DLG = QFileDialog.Option.DontUseNativeDialog

PRESETS = {
    "Fast · MPS":   {"epochs": 150, "batch_size": 1, "grad_accum": 32, "lr": 3e-3, "lora_rank": 4, "resize": "512",  "_sub": "150e · r4 · 512"},
    "Balanced":     {"epochs": 300, "batch_size": 1, "grad_accum": 32, "lr": 3e-3, "lora_rank": 4, "resize": "512",  "_sub": "300e · r4 · 512"},
    "Best quality": {"epochs": 500, "batch_size": 1, "grad_accum": 32, "lr": 1e-3, "lora_rank": 8, "resize": "1024", "_sub": "500e · r8 · 1024"},
}


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {LABEL}; font-size: 11px; font-weight: 500; padding: 3px 0 1px 0;")
    return lbl


def _pick_dir(parent, le, start=None):
    p = QFileDialog.getExistingDirectory(
        parent, "Select folder", start or le.text() or str(Path.home()), _DLG)
    if p:
        le.setText(p)
        le.setToolTip(p)


def _pick_file(parent, le, caption, ext="All (*)", start=None):
    p, _ = QFileDialog.getOpenFileName(
        parent, caption, start or str(Path(le.text()).parent if le.text() else Path.home()),
        ext, options=_DLG)
    if p:
        le.setText(p)
        le.setToolTip(p)


def _pick_save(parent, le):
    p, _ = QFileDialog.getSaveFileName(
        parent, "Save checkpoint", str(LORA_OUT_DIR), "PyTorch (*.pth)", options=_DLG)
    if p:
        le.setText(p if p.endswith(".pth") else p + ".pth")


def _browse(parent, callback):
    b = QPushButton("⋯")
    b.setFixedSize(30, 30)
    b.setStyleSheet(BTN_BROWSE)
    b.clicked.connect(callback)
    return b


def _folder_row(parent, le, start):
    row = QHBoxLayout(); row.setSpacing(6)
    row.addWidget(le)
    row.addWidget(_browse(parent, lambda: _pick_dir(parent, le, start)))
    return row


def _file_row(parent, le, caption, ext, start=None):
    row = QHBoxLayout(); row.setSpacing(6)
    row.addWidget(le)
    row.addWidget(_browse(parent, lambda: _pick_file(parent, le, caption, ext, start)))
    return row


# ── Loss chart ────────────────────────────────────────────────────────────────

class LossChart(QWidget):
    def __init__(self):
        super().__init__()
        self._use_pg = False
        try:
            import pyqtgraph as pg
            self._pg = pg
            self._plot = pg.PlotWidget(background=CONSOLE)
            self._plot.setFixedHeight(96)
            self._plot.showGrid(x=False, y=True, alpha=0.15)
            for axis in ("bottom", "left"):
                self._plot.getAxis(axis).setTextPen(DIM)
                self._plot.getAxis(axis).setPen(BORDER)
            self._curve = self._plot.plot(pen=pg.mkPen(ACCENT, width=1.6))
            L = QVBoxLayout(); L.setContentsMargins(0, 0, 0, 0); L.addWidget(self._plot)
            self.setLayout(L)
            self._use_pg = True
        except Exception:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(3, 1.1), dpi=90)
            self.fig.patch.set_facecolor(BG)
            self.ax = self.fig.add_subplot(111)
            self._style_mpl()
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.canvas.setFixedHeight(96)
            L = QVBoxLayout(); L.setContentsMargins(0, 0, 0, 0); L.addWidget(self.canvas)
            self.setLayout(L)
        self.setVisible(False)

    def _style_mpl(self):
        ax = self.ax
        ax.set_facecolor(CONSOLE)
        ax.tick_params(colors=DIM, labelsize=8)
        ax.set_xlabel("Epoch", color=DIM, fontsize=8)
        ax.set_ylabel("Loss",  color=DIM, fontsize=8)
        for s in ax.spines.values():
            s.set_edgecolor(BORDER)
        self.fig.tight_layout(pad=0.6)

    def update(self, history, epoch_max):
        if not history: return
        epochs = [d["epoch"] for d in history]
        losses = [d["loss"]  for d in history]
        if self._use_pg:
            self._curve.setData(epochs, losses)
        else:
            self.ax.cla(); self._style_mpl()
            self.ax.plot(epochs, losses, color=ACCENT, lw=1.5)
            self.ax.fill_between(epochs, losses, alpha=0.12, color=ACCENT)
            if epoch_max:
                self.ax.set_xlim(1, epoch_max)
            self.ax.set_title(
                f"loss {losses[-1]:.5f}   best {min(losses):.5f}",
                color=DIM, fontsize=8, pad=2)
            self.fig.tight_layout(pad=0.6)
            self.canvas.draw_idle()
        self.setVisible(True)


# ── Main widget ───────────────────────────────────────────────────────────────

class TrainWidget(QWidget):
    _log_signal    = pyqtSignal(str)
    _finish_signal = pyqtSignal()

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._train_thread  = None
        self._stop_event    = threading.Event()
        self._progress_queue: queue.Queue = queue.Queue()
        self._loss_history: list = []

        self.setStyleSheet(WIDGET_SS)

        outer = QVBoxLayout(); outer.setSpacing(0); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        L = QVBoxLayout(); L.setSpacing(0); L.setContentsMargins(14, 8, 14, 16)

        # ── Presets card ──────────────────────────────────────────────────────
        presets_card = SectionCard("Presets", icon="spark")

        row_pre = QHBoxLayout(); row_pre.setSpacing(6)
        self._preset_group = QButtonGroup(self); self._preset_group.setExclusive(True)
        self._preset_btns = {}
        for name, vals in PRESETS.items():
            b = QPushButton(f"{name}\n{vals.get('_sub', '')}")
            b.setStyleSheet(BTN_PRESET)
            b.setFixedHeight(46)
            b.setCheckable(True)
            b.clicked.connect(lambda _, v=vals: self._apply_preset(v))
            self._preset_group.addButton(b)
            self._preset_btns[name] = b
            row_pre.addWidget(b)
        self._preset_btns["Balanced"].setChecked(True)
        presets_card.addLayout(row_pre)

        self._eff_lbl = QLabel()
        self._eff_lbl.setStyleSheet(
            f"color: {DIM}; font-size: 10px; font-family: {MONO}; padding: 2px 0;")
        presets_card.addWidget(self._eff_lbl)
        L.addWidget(presets_card)

        # ── Training data card ────────────────────────────────────────────────
        TRAIN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        TRAIN_MASK_DIR.mkdir(parents=True, exist_ok=True)
        LORA_OUT_DIR.mkdir(parents=True, exist_ok=True)

        data_card = SectionCard("Training data", icon="folder")

        data_card.addWidget(_field_label("Images folder"))
        self.image_dir = QLineEdit(str(TRAIN_IMAGE_DIR))
        data_card.addLayout(_folder_row(self, self.image_dir, str(TRAIN_IMAGE_DIR)))

        data_card.addWidget(_field_label("Masks folder"))
        self.mask_dir = QLineEdit(str(TRAIN_MASK_DIR))
        data_card.addLayout(_folder_row(self, self.mask_dir, str(TRAIN_MASK_DIR)))

        use_layers_btn = QPushButton("  Use active napari layers as training data")
        use_layers_btn.setFixedHeight(32)
        use_layers_btn.setStyleSheet(BTN_SECONDARY)
        use_layers_btn.setIcon(icons.icon("download", LABEL, 14))
        use_layers_btn.setToolTip(
            "Exports the Image + Labels layers currently open in napari\n"
            "into the training folders above. Use napari's label brush\n"
            "to annotate cells first, then click this button.")
        use_layers_btn.clicked.connect(self._use_napari_layers)
        data_card.addWidget(use_layers_btn)

        self._layer_status_lbl = QLabel("")
        self._layer_status_lbl.setStyleSheet(
            f"color: {LABEL}; font-size: 10px; padding: 1px 0;")
        data_card.addWidget(self._layer_status_lbl)

        data_card.addWidget(_field_label("Output checkpoint"))
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("auto-named  lora_vit_h_r4_s512_<timestamp>.pth")
        r_out = QHBoxLayout(); r_out.setSpacing(6)
        r_out.addWidget(self.output_path)
        r_out.addWidget(_browse(self, lambda: _pick_save(self, self.output_path)))
        data_card.addLayout(r_out)
        L.addWidget(data_card)

        # ── Model settings (collapsible card) ─────────────────────────────────
        _model_card = CollapsibleCard("Model settings", collapsed=True, icon="settings")

        self.vit_name = QComboBox()
        self.vit_name.addItems(["vit_h", "vit_l", "vit_b"])
        self.vit_name.currentTextChanged.connect(self._on_vit_changed)
        _model_card.addLayout(_param_row("SAM type", self.vit_name,
            "vit_h = best quality (~2.5 GB), vit_b = fastest (~375 MB)", label_width=92))

        self.sam_path = QLineEdit()
        self.sam_path.setPlaceholderText("auto-detected")
        _model_card.addLayout(_file_row(self, self.sam_path, "SAM backbone", "PyTorch (*.pth)",
            str(SAM_BACKBONE_DIR)))
        self._on_vit_changed("vit_h")

        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(1, 64); self.lora_rank.setValue(4)
        self.lora_rank.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.lora_rank.valueChanged.connect(self._update_eff)
        _model_card.addLayout(_param_row("LoRA rank", self.lora_rank,
            "Adapter size. Higher = more parameters = better accuracy, more memory.", label_width=92))
        L.addWidget(_model_card)

        # ── Training parameters card ──────────────────────────────────────────
        # One aligned column: every label shares a fixed width so the fields
        # line up perfectly (no more mixed full-width / two-up rows).
        params_card = SectionCard("Training parameters", icon="settings")
        _LW = 92

        self.resize_size = QComboBox()
        for v in ["256", "512", "768", "1024"]:
            self.resize_size.addItem(v)
        self.resize_size.setCurrentText("512")
        params_card.addLayout(_param_row("Resize", self.resize_size,
            "Match prediction resize for best results.", label_width=_LW))

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5000); self.epochs.setValue(300)
        self.epochs.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        params_card.addLayout(_param_row("Epochs", self.epochs,
            "Total number of training epochs.", label_width=_LW))

        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(5); self.lr.setRange(1e-6, 1.0)
        self.lr.setSingleStep(1e-4); self.lr.setValue(3e-3)
        self.lr.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        params_card.addLayout(_param_row("Learning rate", self.lr,
            "Base learning rate (OneCycle schedule).", label_width=_LW))

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 16); self.batch_size.setValue(1)
        self.batch_size.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.batch_size.valueChanged.connect(self._update_eff)
        params_card.addLayout(_param_row("Batch", self.batch_size,
            "Images per step.", label_width=_LW))

        self.grad_accum = QSpinBox()
        self.grad_accum.setRange(1, 128); self.grad_accum.setValue(32)
        self.grad_accum.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.grad_accum.valueChanged.connect(self._update_eff)
        params_card.addLayout(_param_row("Grad accum", self.grad_accum,
            "Gradient accumulation. Effective batch = batch × accum.", label_width=_LW))

        self.device = QComboBox(); self._populate_devices()
        params_card.addLayout(_param_row("Device", self.device, label_width=_LW))
        L.addWidget(params_card)

        # ── Run — no card, button is the visual anchor ─────────────────────────
        L.addSpacing(14)

        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self.start_btn = QPushButton("  Start Training")
        self.start_btn.setFixedHeight(44)
        self.start_btn.setStyleSheet(BTN_PRIMARY)
        self.start_btn.setIcon(icons.icon("run", "#ffffff", 16))
        self.start_btn.setToolTip("Ctrl+T")
        self.start_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedHeight(44)
        self.stop_btn.setFixedWidth(72)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(BTN_SECONDARY)
        self.stop_btn.setToolTip("Esc")
        self.stop_btn.clicked.connect(self._stop_training)
        btn_row.addWidget(self.stop_btn)
        L.addLayout(btn_row)

        L.addSpacing(6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(3); self.progress_bar.setRange(0, 100)
        L.addWidget(self.progress_bar)

        from napari_app.theme import MONO as _MONO
        status_row = QHBoxLayout(); status_row.setContentsMargins(0, 4, 0, 0)
        self.epoch_lbl = QLabel("")
        self.epoch_lbl.setStyleSheet(
            f"color: {LABEL}; font-size: 11px; font-family: {_MONO};")
        self.loss_lbl = QLabel("")
        self.loss_lbl.setStyleSheet(
            f"color: {LABEL}; font-size: 11px; font-family: {_MONO};")
        status_row.addWidget(self.epoch_lbl)
        status_row.addStretch()
        status_row.addWidget(self.loss_lbl)
        L.addLayout(status_row)

        self.loss_chart = LossChart()
        L.addWidget(self.loss_chart)

        # ── Training history card ─────────────────────────────────────────────
        hist_card = SectionCard("Training history", icon="log")
        self.history_box = QTextEdit()
        self.history_box.setReadOnly(True)
        self.history_box.setFixedHeight(96)
        hist_card.addWidget(self.history_box)
        L.addWidget(hist_card)

        L.addStretch()
        inner.setLayout(L)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # ── Log footer ────────────────────────────────────────────────────────
        outer.addWidget(_divider())
        _footer = QHBoxLayout()
        _footer.setContentsMargins(16, 4, 16, 6)
        _log_btn = QPushButton("  Log")
        _log_btn.setIcon(icons.icon("log", DIM, 13))
        _log_btn.setStyleSheet(
            f"color: {DIM}; background: transparent; border: none; font-size: 11px;")
        _log_btn.setToolTip("Open the floating log window")
        _log_btn.clicked.connect(lambda: get_log_window().show_and_raise())
        _footer.addStretch()
        _footer.addWidget(_log_btn)
        outer.addLayout(_footer)

        self.setLayout(outer)
        self.setMinimumWidth(260)

        self._log_signal.connect(self._append_log)
        self._finish_signal.connect(self._on_finish)
        self._timer = QTimer(); self._timer.setInterval(1000)
        self._timer.timeout.connect(self._poll_progress)

        viewer.bind_key('Control-t', lambda v: self._start_training() if self.start_btn.isEnabled() else None)
        viewer.bind_key('Escape',    lambda v: self._stop_training()  if self.stop_btn.isEnabled()  else None)

        self._update_eff()
        self._refresh_history()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _append_log(self, text: str):
        lw = get_log_window()
        lw.append(text)
        if "[ERROR]" in text:
            lw.show_and_raise()
        elif not lw.isVisible():
            lw.show()

    def _populate_devices(self):
        import torch
        self.device.addItem("cpu")
        if torch.backends.mps.is_available():
            self.device.addItem("mps"); self.device.setCurrentText("mps")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.device.addItem(str(i))

    def _on_vit_changed(self, vit):
        names = {"vit_h": "sam_vit_h_4b8939.pth",
                 "vit_l": "sam_vit_l_0b3195.pth",
                 "vit_b": "sam_vit_b_01ec64.pth"}
        c = SAM_BACKBONE_DIR / names.get(vit, "")
        if c.exists():
            self.sam_path.setText(str(c))
        else:
            self.sam_path.clear()
            self.sam_path.setPlaceholderText(f"Not found: {c.name}")

    def _apply_preset(self, v):
        self.epochs.setValue(v["epochs"])
        self.batch_size.setValue(v["batch_size"])
        self.grad_accum.setValue(v["grad_accum"])
        self.lr.setValue(v["lr"])
        self.lora_rank.setValue(v["lora_rank"])
        self.resize_size.setCurrentText(v["resize"])
        self._update_eff()

    def _update_eff(self):
        eff = self.batch_size.value() * self.grad_accum.value()
        self._eff_lbl.setText(
            f"Effective batch size:  {self.batch_size.value()} × {self.grad_accum.value()} = {eff}")

    def _resolve_sam(self):
        p = self.sam_path.text().strip()
        if p and Path(p).exists():
            return p
        vit = self.vit_name.currentText()
        names = {"vit_h": "sam_vit_h_4b8939.pth",
                 "vit_l": "sam_vit_l_0b3195.pth",
                 "vit_b": "sam_vit_b_01ec64.pth"}
        c = SAM_BACKBONE_DIR / names[vit]
        if c.exists():
            return str(c)
        raise ValueError(f"SAM backbone not found: {names[vit]}")

    def _resolve_output(self):
        p = self.output_path.text().strip()
        if p:
            return p if p.endswith(".pth") else p + ".pth"
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        v  = self.vit_name.currentText()
        r  = self.lora_rank.value()
        s  = self.resize_size.currentText()
        return str(LORA_OUT_DIR / f"lora_{v}_r{r}_s{s}_{ts}.pth")

    def _build_config(self):
        sam  = self._resolve_sam()
        idir = self.image_dir.text().strip()
        mdir = self.mask_dir.text().strip()
        out  = self._resolve_output()
        for p, name in [(idir, "Image folder"), (mdir, "Mask folder")]:
            if not p or not Path(p).exists():
                raise ValueError(f"{name} not found: {p}")
        imgs = sorted(f for f in Path(idir).iterdir()
                      if f.suffix.lower() in (".png",".jpg",".jpeg",".tif",".tiff",".bmp",".npy"))
        if not imgs:
            raise ValueError(f"No images in {idir}")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        rs = int(self.resize_size.currentText())
        return {
            "deterministic": True, "seed": 0,
            "allow_tf32_on_cudnn": True, "allow_tf32_on_matmul": True,
            "vit_name": self.vit_name.currentText(), "model_path": sam,
            "train_image_dir": idir, "train_mask_dir": mdir,
            "result_pth_path": out,
            "resize_size": [rs, rs], "patch_size": rs // 2, "sam_image_size": rs,
            "train_id": list(range(len(imgs))), "duplicate_data": 32,
            "epoch_max": self.epochs.value(), "batch_size": self.batch_size.value(),
            "gradient_accumulation_step": self.grad_accum.value(),
            "base_lr": self.lr.value(), "onecycle_lr_pct_start": 0.3, "num_workers": 0,
            "image_encoder_lora_rank": self.lora_rank.value(),
            "mask_decoder_lora_rank":  self.lora_rank.value(),
            "freeze_image_encoder": True, "freeze_prompt_encoder": True,
            "freeze_mask_decoder_transformer": True, "freeze_upscaling_cnn": True,
            "freeze_output_hypernetworks_mlps": True,
            "freeze_mask_decoder_mask_tokens": True, "freeze_mask_decoder_iou": True,
            "lora_dropout": 0.1,
            "pos_rate": 1.0, "neg_rate": 0.5, "max_point_num": 30,
            "edge_distance": 20, "neg_area_ratio_threshold": 5,
            "neg_area_threshold": 1000, "min_cell_area": 100,
            "foreground_sample_area_ratio": 0.2, "background_sample_area_ratio": 0.2,
            "foreground_equal_prob": True, "background_equal_prob": True,
            "data_augmentation": True, "bright_limit": 0.1, "contrast_limit": 0.1,
            "bright_prob": 0.5, "flip_prob": 0.75, "rotate_prob": 0.8,
            "scale_limit": [-0.5, 0.5], "crop_prob": 0.5,
            "crop_scale": [0.3, 1.0], "crop_ratio": [0.75, 1.3333],
            "ce_loss_weight": 1.0, "punish_background_point": False,
            "track_gpu_memory": False, "selected_device": self.device.currentText(),
        }

    # ── Napari layers export ──────────────────────────────────────────────────

    def _use_napari_layers(self):
        import cv2

        image_layer = None
        labels_layer = None
        try:
            import napari.layers as nl
            for layer in self.viewer.layers:
                if isinstance(layer, nl.Labels) and labels_layer is None:
                    labels_layer = layer
                elif isinstance(layer, nl.Image) and image_layer is None:
                    image_layer = layer
        except Exception as e:
            self._layer_status_lbl.setText(f"[ERROR] {e}"); return

        if image_layer is None or labels_layer is None:
            missing = [n for n, l in [("Image", image_layer), ("Labels", labels_layer)] if l is None]
            self._layer_status_lbl.setText(f"Missing: {', '.join(missing)}"); return

        img  = np.asarray(image_layer.data, dtype=np.float32)
        mask = np.asarray(labels_layer.data).astype(np.int32)

        if img.max() > 1.0:
            img = (img / img.max() * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        stem      = Path(image_layer.name).stem or "napari_image"
        img_path  = Path(self.image_dir.text()) / f"{stem}.png"
        mask_path = Path(self.mask_dir.text())  / f"{stem}.png"
        Path(self.image_dir.text()).mkdir(parents=True, exist_ok=True)
        Path(self.mask_dir.text()).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(img_path),  cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), np.clip(mask, 0, 65535).astype(np.uint16))

        n_cells = int(mask.max())
        self._layer_status_lbl.setText(f"✓ Saved {img_path.name}  ({n_cells} cells)")

    def _refresh_history(self):
        h = STATE_MANAGER.load_history()
        if not h:
            self.history_box.setPlainText("No training runs yet."); return
        lines = []
        for r in h[:6]:
            ts  = r.get("started_at","")[:16].replace("T", " ")
            fl  = r.get("final_loss")
            ep  = r.get("epochs_run", 0)
            epm = r.get("epoch_max","")
            ck  = Path(r.get("checkpoint","")).name
            st  = "✓" if r.get("status") == "completed" else "✗"
            ls  = f"{fl:.5f}" if fl is not None else "—"
            lines.append(f"{st} {ts}  {ep}/{epm} ep  {ls}  {ck}")
        self.history_box.setPlainText("\n".join(lines))

    # ── Mask validation ───────────────────────────────────────────────────────

    def _validate_masks(self, mask_dir: str, min_cell_area: int) -> tuple[str, bool]:
        import cv2
        masks = [m for m in sorted(Path(mask_dir).glob("*"))
                 if m.suffix.lower() in (".png",".tif",".tiff",".npy",".bmp")]
        if not masks:
            return "[ERROR] No mask files found", True
        total_cells = 0; small_cells = 0
        for mp in masks:
            m = np.load(str(mp)).astype(np.int32) if mp.suffix.lower() == ".npy" \
                else cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if m is None: continue
            n = int(m.max()); total_cells += n
            counts = np.bincount(m.ravel())
            for i in range(1, n + 1):
                if i < len(counts) and counts[i] < min_cell_area:
                    small_cells += 1
        parts = [f"✓ {total_cells} cells across {len(masks)} mask(s)"]
        if small_cells:
            parts.append(f"⚠ {small_cells} cells < min_area={min_cell_area} (filtered)")
        if total_cells < 20:
            parts.append("⚠ Fewer than 20 cells — training may be unstable")
        return "  ".join(parts), False

    # ── Training ──────────────────────────────────────────────────────────────

    def _start_training(self):
        try:
            config = self._build_config()
        except ValueError as e:
            self._append_log(f"[ERROR] {e}"); return

        validation_msg, is_fatal = self._validate_masks(
            config["train_mask_dir"], config["min_cell_area"])
        self._append_log(validation_msg)
        if is_fatal:
            return

        STATE_MANAGER.clear_training_state()
        STATE_MANAGER.clear_stop_flag()
        STATE_MANAGER.clear_loss_history()
        self._stop_event.clear()
        while not self._progress_queue.empty():
            self._progress_queue.get_nowait()
        self._loss_history = []

        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0); self.loss_chart.setVisible(False)
        self.epoch_lbl.setText("—"); self.loss_lbl.setText("—")
        n_imgs = len(list(Path(config["train_image_dir"]).iterdir()))
        self._append_log(
            f"▶ {config['epoch_max']} epochs · {n_imgs} image(s) · "
            f"rank {config['image_encoder_lora_rank']} · {config['selected_device']}")

        pq, se = self._progress_queue, self._stop_event

        def run():
            from napari_app.core.train_model import train_model
            try:
                train_model(config, STATE_MANAGER, progress_queue=pq, stop_event=se)
                self._log_signal.emit("✓ Training complete")
            except Exception as e:
                import traceback
                self._log_signal.emit(f"[ERROR] {e}\n{traceback.format_exc()}")
            finally:
                self._finish_signal.emit()

        self._train_thread = threading.Thread(target=run, daemon=True)
        self._train_thread.start()
        self._timer.start()

    def _stop_training(self):
        self._stop_event.set()
        STATE_MANAGER.set_stop_flag()
        self._append_log("■ Stop requested")
        self.stop_btn.setEnabled(False)

    def _poll_progress(self):
        total = self.epochs.value()
        last  = None
        while not self._progress_queue.empty():
            item = self._progress_queue.get_nowait()
            self._loss_history.append({"epoch": item["epoch"], "loss": item["loss"]})
            last = item
        if last is not None:
            self.progress_bar.setValue(last["pct"])
            self.epoch_lbl.setText(f"Epoch {last['epoch']} / {total}")
            self.loss_lbl.setText(f"loss  {last['loss']:.6f}")
            self.loss_chart.update(self._loss_history, total)
        if self._train_thread and not self._train_thread.is_alive():
            self._timer.stop()

    def _on_finish(self):
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self._timer.stop(); self._refresh_history()
