import os
import threading
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QFileDialog, QGroupBox, QScrollArea,
    QSizePolicy, QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

from gui.pages.utils.train_state_manager import TrainingStateManager

STORAGE_DIR = Path(__file__).parents[2] / "streamlit_storage"
STATE_MANAGER = TrainingStateManager(str(STORAGE_DIR))


def _pick_dir(parent, line_edit):
    path = QFileDialog.getExistingDirectory(parent, "Select folder")
    if path:
        line_edit.setText(path)


def _pick_file(parent, line_edit, caption="Select file", ext="All (*)"):
    path, _ = QFileDialog.getOpenFileName(parent, caption, "", ext)
    if path:
        line_edit.setText(path)


def _pick_save(parent, line_edit, caption="Save checkpoint as"):
    path, _ = QFileDialog.getSaveFileName(parent, caption, "", "PyTorch (*.pth)")
    if path:
        if not path.endswith(".pth"):
            path += ".pth"
        line_edit.setText(path)


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

        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout()
        inner_layout.setSpacing(6)
        inner_layout.setContentsMargins(0, 0, 0, 0)

        # --- Paths ---
        inner_layout.addWidget(SectionLabel("Paths"))
        inner_layout.addWidget(QLabel("SAM backbone (.pth)"))
        self.sam_path = QLineEdit()
        self.sam_path.setPlaceholderText("streamlit_storage/sam_backbone/sam_vit_h_4b8939.pth")
        default_sam = str(STORAGE_DIR / "sam_backbone" / "sam_vit_h_4b8939.pth")
        if Path(default_sam).exists():
            self.sam_path.setText(default_sam)
        r = QHBoxLayout()
        r.addWidget(self.sam_path)
        btn = QPushButton("…")
        btn.setFixedWidth(30)
        btn.clicked.connect(lambda: _pick_file(self, self.sam_path, "Select SAM checkpoint", "PyTorch (*.pth)"))
        r.addWidget(btn)
        inner_layout.addLayout(r)

        inner_layout.addWidget(QLabel("Image folder"))
        self.image_dir = QLineEdit()
        r2 = QHBoxLayout()
        r2.addWidget(self.image_dir)
        btn2 = QPushButton("…")
        btn2.setFixedWidth(30)
        btn2.clicked.connect(lambda: _pick_dir(self, self.image_dir))
        r2.addWidget(btn2)
        inner_layout.addLayout(r2)

        inner_layout.addWidget(QLabel("Mask folder"))
        self.mask_dir = QLineEdit()
        r3 = QHBoxLayout()
        r3.addWidget(self.mask_dir)
        btn3 = QPushButton("…")
        btn3.setFixedWidth(30)
        btn3.clicked.connect(lambda: _pick_dir(self, self.mask_dir))
        r3.addWidget(btn3)
        inner_layout.addLayout(r3)

        inner_layout.addWidget(QLabel("Output checkpoint"))
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("checkpoints/my_model.pth")
        r4 = QHBoxLayout()
        r4.addWidget(self.output_path)
        btn4 = QPushButton("…")
        btn4.setFixedWidth(30)
        btn4.clicked.connect(lambda: _pick_save(self, self.output_path))
        r4.addWidget(btn4)
        inner_layout.addLayout(r4)

        # --- Model ---
        inner_layout.addWidget(SectionLabel("Model"))
        row_m = QHBoxLayout()
        row_m.addWidget(QLabel("SAM type"))
        self.vit_name = QComboBox()
        self.vit_name.addItems(["vit_h", "vit_l", "vit_b"])
        row_m.addWidget(self.vit_name)
        inner_layout.addLayout(row_m)

        row_lr = QHBoxLayout()
        row_lr.addWidget(QLabel("LoRA rank"))
        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(1, 64)
        self.lora_rank.setValue(4)
        row_lr.addWidget(self.lora_rank)
        inner_layout.addLayout(row_lr)

        # --- Training ---
        inner_layout.addWidget(SectionLabel("Training"))
        row_ep = QHBoxLayout()
        row_ep.addWidget(QLabel("Epochs"))
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 2000)
        self.epochs.setValue(300)
        row_ep.addWidget(self.epochs)
        inner_layout.addLayout(row_ep)

        row_bs = QHBoxLayout()
        row_bs.addWidget(QLabel("Batch size"))
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 16)
        self.batch_size.setValue(1)
        row_bs.addWidget(self.batch_size)
        inner_layout.addLayout(row_bs)

        row_ga = QHBoxLayout()
        row_ga.addWidget(QLabel("Grad accumulation"))
        self.grad_accum = QSpinBox()
        self.grad_accum.setRange(1, 128)
        self.grad_accum.setValue(32)
        row_ga.addWidget(self.grad_accum)
        inner_layout.addLayout(row_ga)

        row_lr2 = QHBoxLayout()
        row_lr2.addWidget(QLabel("Learning rate"))
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(5)
        self.lr.setRange(1e-6, 1.0)
        self.lr.setSingleStep(1e-4)
        self.lr.setValue(3e-3)
        row_lr2.addWidget(self.lr)
        inner_layout.addLayout(row_lr2)

        # --- Device ---
        inner_layout.addWidget(SectionLabel("Device"))
        row_dev = QHBoxLayout()
        row_dev.addWidget(QLabel("Device"))
        self.device = QComboBox()
        self._populate_devices()
        row_dev.addWidget(self.device)
        inner_layout.addLayout(row_dev)

        # --- Progress ---
        inner_layout.addWidget(SectionLabel("Progress"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        inner_layout.addWidget(self.progress_bar)

        self.epoch_label = QLabel("Epoch: —")
        self.loss_label = QLabel("Loss: —")
        inner_layout.addWidget(self.epoch_label)
        inner_layout.addWidget(self.loss_label)

        # --- Buttons ---
        self.start_btn = QPushButton("▶  Start Training")
        self.start_btn.setFixedHeight(36)
        self.start_btn.setStyleSheet(
            "background-color: #1565C0; color: white; font-weight: bold; border-radius: 4px;"
        )
        self.start_btn.clicked.connect(self._start_training)
        inner_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "background-color: #B71C1C; color: white; font-weight: bold; border-radius: 4px;"
        )
        self.stop_btn.clicked.connect(self._stop_training)
        inner_layout.addWidget(self.stop_btn)

        # --- Log ---
        inner_layout.addWidget(SectionLabel("Log"))
        from PyQt6.QtWidgets import QTextEdit
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(120)
        self.log_box.setStyleSheet("background: #1A1A2E; color: #A0A0A0; font-family: Menlo, Monaco, Courier; font-size: 11px;")
        inner_layout.addWidget(self.log_box)

        inner_layout.addStretch()
        inner.setLayout(inner_layout)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self.setMinimumWidth(280)

        self._log_signal.connect(self._append_log)

        # Poll progress every second while training
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

    def _build_config(self):
        sam_path = self.sam_path.text().strip()
        image_dir = self.image_dir.text().strip()
        mask_dir = self.mask_dir.text().strip()
        output = self.output_path.text().strip()

        if not sam_path or not Path(sam_path).exists():
            raise ValueError(f"SAM backbone not found: {sam_path}")
        if not image_dir or not Path(image_dir).exists():
            raise ValueError(f"Image folder not found: {image_dir}")
        if not mask_dir or not Path(mask_dir).exists():
            raise ValueError(f"Mask folder not found: {mask_dir}")
        if not output:
            raise ValueError("Specify output checkpoint path")

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
            "train_id": [0],
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
