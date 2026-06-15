import os
import threading
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QFileDialog, QScrollArea, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

from gui.pages.utils.train_state_manager import TrainingStateManager
from project_root import STORAGE_DIR

TRAIN_IMAGE_DIR = STORAGE_DIR / "train_images"
TRAIN_MASK_DIR  = STORAGE_DIR / "train_masks"
LORA_OUT_DIR    = STORAGE_DIR / "loras"
SAM_BACKBONE_DIR = STORAGE_DIR / "sam_backbone"

STATE_MANAGER = TrainingStateManager(str(STORAGE_DIR))

_DLG_OPT = QFileDialog.Option.DontUseNativeDialog


def _pick_dir(parent, line_edit, start=None):
    start = start or (str(Path(line_edit.text())) if line_edit.text() else str(Path.home()))
    path = QFileDialog.getExistingDirectory(
        parent, "Select folder", start, QFileDialog.Option.DontUseNativeDialog)
    if path:
        line_edit.setText(path)


def _pick_file(parent, line_edit, caption="Select file", ext="All (*)", start=None):
    start = start or (str(Path(line_edit.text()).parent) if line_edit.text() else str(Path.home()))
    path, _ = QFileDialog.getOpenFileName(parent, caption, start, ext, options=_DLG_OPT)
    if path:
        line_edit.setText(path)


def _pick_save(parent, line_edit, caption="Save checkpoint as", start=None):
    start = start or str(LORA_OUT_DIR)
    path, _ = QFileDialog.getSaveFileName(parent, caption, start, "PyTorch (*.pth)", options=_DLG_OPT)
    if path:
        if not path.endswith(".pth"):
            path += ".pth"
        line_edit.setText(path)


def _divider():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #333;")
    return line


class SectionLabel(QLabel):
    def __init__(self, text):
        super().__init__(text)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        self.setFont(font)
        self.setStyleSheet("color: #90CAF9; margin-top: 8px;")


class TrainWidget(QWidget):
    _log_signal = pyqtSignal(str)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._train_thread = None

        outer = QVBoxLayout()
        outer.setSpacing(6)
        outer.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        L = QVBoxLayout()
        L.setSpacing(6)
        L.setContentsMargins(0, 0, 0, 0)

        # ── Data folders ─────────────────────────────────────────────────────
        L.addWidget(SectionLabel("Training data"))

        TRAIN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        TRAIN_MASK_DIR.mkdir(parents=True, exist_ok=True)

        L.addWidget(QLabel("Image folder"))
        self.image_dir = QLineEdit(str(TRAIN_IMAGE_DIR))
        r1 = QHBoxLayout()
        r1.addWidget(self.image_dir)
        b1 = QPushButton("…"); b1.setFixedWidth(30)
        b1.clicked.connect(lambda: _pick_dir(self, self.image_dir, str(TRAIN_IMAGE_DIR)))
        r1.addWidget(b1)
        L.addLayout(r1)

        L.addWidget(QLabel("Mask folder"))
        self.mask_dir = QLineEdit(str(TRAIN_MASK_DIR))
        r2 = QHBoxLayout()
        r2.addWidget(self.mask_dir)
        b2 = QPushButton("…"); b2.setFixedWidth(30)
        b2.clicked.connect(lambda: _pick_dir(self, self.mask_dir, str(TRAIN_MASK_DIR)))
        r2.addWidget(b2)
        L.addLayout(r2)

        L.addWidget(QLabel("Output checkpoint"))
        LORA_OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("auto-named in streamlit_storage/loras/")
        r3 = QHBoxLayout()
        r3.addWidget(self.output_path)
        b3 = QPushButton("…"); b3.setFixedWidth(30)
        b3.clicked.connect(lambda: _pick_save(self, self.output_path))
        r3.addWidget(b3)
        L.addLayout(r3)

        L.addWidget(_divider())

        # ── Model ────────────────────────────────────────────────────────────
        L.addWidget(SectionLabel("Model"))

        row_v = QHBoxLayout()
        row_v.addWidget(QLabel("SAM type"))
        self.vit_name = QComboBox()
        self.vit_name.addItems(["vit_h", "vit_l", "vit_b"])
        self.vit_name.currentTextChanged.connect(self._on_vit_changed)
        row_v.addWidget(self.vit_name)
        L.addLayout(row_v)

        L.addWidget(QLabel("SAM backbone (auto-detected)"))
        self.sam_path = QLineEdit()
        self.sam_path.setPlaceholderText("auto")
        self.sam_path.setToolTip("Leave blank to auto-detect; or pick manually")
        r_sam = QHBoxLayout()
        r_sam.addWidget(self.sam_path)
        b_sam = QPushButton("…"); b_sam.setFixedWidth(30)
        b_sam.clicked.connect(lambda: _pick_file(
            self, self.sam_path, "Select SAM backbone", "PyTorch (*.pth)",
            start=str(SAM_BACKBONE_DIR)))
        r_sam.addWidget(b_sam)
        L.addLayout(r_sam)
        self._on_vit_changed("vit_h")  # pre-fill

        row_rank = QHBoxLayout()
        row_rank.addWidget(QLabel("LoRA rank"))
        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(1, 64)
        self.lora_rank.setValue(4)
        row_rank.addWidget(self.lora_rank)
        L.addLayout(row_rank)

        L.addWidget(_divider())

        # ── Training ─────────────────────────────────────────────────────────
        L.addWidget(SectionLabel("Training"))

        for label, attr, lo, hi, val in [
            ("Epochs", "epochs", 1, 2000, 300),
            ("Batch size", "batch_size", 1, 16, 1),
            ("Grad accumulation", "grad_accum", 1, 128, 32),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setValue(val)
            setattr(self, attr, w)
            row.addWidget(w)
            L.addLayout(row)

        row_lr = QHBoxLayout()
        row_lr.addWidget(QLabel("Learning rate"))
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(5)
        self.lr.setRange(1e-6, 1.0)
        self.lr.setSingleStep(1e-4)
        self.lr.setValue(3e-3)
        row_lr.addWidget(self.lr)
        L.addLayout(row_lr)

        row_dev = QHBoxLayout()
        row_dev.addWidget(QLabel("Device"))
        self.device = QComboBox()
        self._populate_devices()
        row_dev.addWidget(self.device)
        L.addLayout(row_dev)

        L.addWidget(_divider())

        # ── Progress ─────────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        L.addWidget(self.progress_bar)

        self.epoch_label = QLabel("Epoch: —")
        self.loss_label = QLabel("Loss: —")
        L.addWidget(self.epoch_label)
        L.addWidget(self.loss_label)

        # ── Buttons ──────────────────────────────────────────────────────────
        self.start_btn = QPushButton("▶  Start Training")
        self.start_btn.setFixedHeight(36)
        self.start_btn.setStyleSheet(
            "background-color: #1565C0; color: white; font-weight: bold; border-radius: 4px;")
        self.start_btn.clicked.connect(self._start_training)
        L.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "background-color: #B71C1C; color: white; font-weight: bold; border-radius: 4px;")
        self.stop_btn.clicked.connect(self._stop_training)
        L.addWidget(self.stop_btn)

        # ── Log ──────────────────────────────────────────────────────────────
        L.addWidget(SectionLabel("Log"))
        from PyQt6.QtWidgets import QTextEdit
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(110)
        self.log_box.setStyleSheet(
            "background: #1A1A2E; color: #A0A0C0; font-family: Menlo, Monaco, Courier; font-size: 11px;")
        L.addWidget(self.log_box)

        L.addStretch()
        inner.setLayout(L)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.setLayout(outer)
        self.setMinimumWidth(290)

        self._log_signal.connect(self._append_log)

        self._timer = QTimer()
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._poll_progress)

    def _populate_devices(self):
        import torch
        self.device.addItem("cpu")
        if torch.backends.mps.is_available():
            self.device.addItem("mps")
            self.device.setCurrentText("mps")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.device.addItem(str(i))

    def _on_vit_changed(self, vit):
        names = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }
        candidate = SAM_BACKBONE_DIR / names.get(vit, "")
        if candidate.exists():
            self.sam_path.setText(str(candidate))
        else:
            self.sam_path.clear()
            self.sam_path.setPlaceholderText(f"Not found: {candidate.name} — specify manually")

    def _resolve_sam_path(self):
        p = self.sam_path.text().strip()
        if p and Path(p).exists():
            return p
        raise ValueError(
            f"SAM backbone not found. Download it and place in {SAM_BACKBONE_DIR}/ "
            "or specify the path manually.")

    def _resolve_output_path(self):
        p = self.output_path.text().strip()
        if p:
            if not p.endswith(".pth"):
                p += ".pth"
            return p
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        vit = self.vit_name.currentText()
        rank = self.lora_rank.value()
        return str(LORA_OUT_DIR / f"lora_{vit}_r{rank}_{ts}.pth")

    def _build_config(self):
        sam_path = self._resolve_sam_path()
        image_dir = self.image_dir.text().strip()
        mask_dir = self.mask_dir.text().strip()
        output = self._resolve_output_path()

        if not image_dir or not Path(image_dir).exists():
            raise ValueError(f"Image folder not found: {image_dir}")
        if not mask_dir or not Path(mask_dir).exists():
            raise ValueError(f"Mask folder not found: {mask_dir}")

        image_files = [f for f in Path(image_dir).iterdir()
                       if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy")]
        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        Path(output).parent.mkdir(parents=True, exist_ok=True)

        return {
            "deterministic": True,
            "seed": 0,
            "allow_tf32_on_cudnn": True,
            "allow_tf32_on_matmul": True,
            "vit_name": self.vit_name.currentText(),
            "model_path": sam_path,
            "train_image_dir": image_dir,
            "train_mask_dir": mask_dir,
            "result_pth_path": output,
            "resize_size": [512, 512],
            "patch_size": 256,
            "sam_image_size": 512,
            "train_id": list(range(len(image_files))),
            "duplicate_data": 32,
            "epoch_max": self.epochs.value(),
            "batch_size": self.batch_size.value(),
            "gradient_accumulation_step": self.grad_accum.value(),
            "base_lr": self.lr.value(),
            "onecycle_lr_pct_start": 0.3,
            "num_workers": 0,
            "image_encoder_lora_rank": self.lora_rank.value(),
            "mask_decoder_lora_rank": self.lora_rank.value(),
            "freeze_image_encoder": True,
            "freeze_prompt_encoder": True,
            "freeze_mask_decoder_transformer": True,
            "freeze_upscaling_cnn": True,
            "freeze_output_hypernetworks_mlps": True,
            "freeze_mask_decoder_mask_tokens": True,
            "freeze_mask_decoder_iou": True,
            "lora_dropout": 0.1,
            "pos_rate": 1.0,
            "neg_rate": 0.5,
            "max_point_num": 30,
            "edge_distance": 20,
            "neg_area_ratio_threshold": 5,
            "neg_area_threshold": 1000,
            "min_cell_area": 100,
            "foreground_sample_area_ratio": 0.2,
            "background_sample_area_ratio": 0.2,
            "foreground_equal_prob": True,
            "background_equal_prob": True,
            "data_augmentation": True,
            "bright_limit": 0.1,
            "contrast_limit": 0.1,
            "bright_prob": 0.5,
            "flip_prob": 0.75,
            "rotate_prob": 0.8,
            "scale_limit": [-0.5, 0.5],
            "crop_prob": 0.5,
            "crop_scale": [0.3, 1.0],
            "crop_ratio": [0.75, 1.3333],
            "ce_loss_weight": 1.0,
            "punish_background_point": False,
            "track_gpu_memory": False,
            "selected_device": self.device.currentText(),
        }

    def _start_training(self):
        try:
            config = self._build_config()
        except ValueError as e:
            self._append_log(f"[ERROR] {e}")
            return

        STATE_MANAGER.clear_training_state()
        STATE_MANAGER.clear_stop_flag()
        STATE_MANAGER.clear_loss_history()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.epoch_label.setText("Epoch: 0")
        self.loss_label.setText("Loss: —")
        self._append_log("Starting training…")

        def run():
            from gui.pages.utils.train_model import train_model
            try:
                train_model(config, STATE_MANAGER)
                self._log_signal.emit("Training finished.")
            except Exception as e:
                self._log_signal.emit(f"[ERROR] {e}")
            finally:
                self._finish_training()

        self._train_thread = threading.Thread(target=run, daemon=True)
        self._train_thread.start()
        self._timer.start()

    def _stop_training(self):
        STATE_MANAGER.set_stop_flag()
        self._append_log("Stop requested…")
        self.stop_btn.setEnabled(False)

    def _poll_progress(self):
        progress = STATE_MANAGER.load_progress()
        pct = progress.get("progress", 0)
        epoch = progress.get("current_epoch", 0)
        self.progress_bar.setValue(pct)
        self.epoch_label.setText(f"Epoch: {epoch} / {self.epochs.value()}")

        history = STATE_MANAGER.load_loss_history()
        if history:
            last_loss = history[-1]["loss"]
            self.loss_label.setText(f"Loss: {last_loss:.6f}")

        if self._train_thread and not self._train_thread.is_alive():
            self._timer.stop()

    def _finish_training(self):
        from PyQt6.QtCore import QMetaObject, Q_ARG
        QMetaObject.invokeMethod(self, "_on_finish", Qt.ConnectionType.QueuedConnection)

    def _on_finish(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._timer.stop()

    def _append_log(self, text):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )
