import threading
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QScrollArea, QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from gui.pages.utils.predict_state_manager import PredictionStateManager

STORAGE_DIR = Path(__file__).parents[2] / "streamlit_storage"
STATE_MANAGER = PredictionStateManager(str(STORAGE_DIR))


def _pick_file(parent, line_edit, caption, ext="All (*)"):
    path, _ = QFileDialog.getOpenFileName(parent, caption, "", ext)
    if path:
        line_edit.setText(path)


def _pick_dir(parent, line_edit):
    path = QFileDialog.getExistingDirectory(parent, "Select folder")
    if path:
        line_edit.setText(path)


class SectionLabel(QLabel):
    def __init__(self, text):
        super().__init__(text)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        self.setFont(font)
        self.setStyleSheet("color: #A5D6A7; margin-top: 8px;")


class PredictWidget(QWidget):
    _log_signal = pyqtSignal(str)
    _done_signal = pyqtSignal(object, object)  # image array, masks list

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._pred_thread = None

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
        default_sam = str(STORAGE_DIR / "sam_backbone" / "sam_vit_h_4b8939.pth")
        if Path(default_sam).exists():
            self.sam_path.setText(default_sam)
        r = QHBoxLayout()
        r.addWidget(self.sam_path)
        b = QPushButton("…"); b.setFixedWidth(30)
        b.clicked.connect(lambda: _pick_file(self, self.sam_path, "SAM backbone", "PyTorch (*.pth)"))
        r.addWidget(b)
        inner_layout.addLayout(r)

        inner_layout.addWidget(QLabel("LoRA checkpoint (.pth)"))
        self.lora_path = QLineEdit()
        r2 = QHBoxLayout()
        r2.addWidget(self.lora_path)
        b2 = QPushButton("…"); b2.setFixedWidth(30)
        b2.clicked.connect(lambda: _pick_file(self, self.lora_path, "LoRA checkpoint", "PyTorch (*.pth)"))
        r2.addWidget(b2)
        inner_layout.addLayout(r2)

        inner_layout.addWidget(QLabel("Input image"))
        self.image_path = QLineEdit()
        r3 = QHBoxLayout()
        r3.addWidget(self.image_path)
        b3 = QPushButton("…"); b3.setFixedWidth(30)
        b3.clicked.connect(lambda: _pick_file(self, self.image_path, "Input image",
                                              "Images (*.png *.tif *.tiff *.jpg *.bmp)"))
        r3.addWidget(b3)
        inner_layout.addLayout(r3)

        # --- Model ---
        inner_layout.addWidget(SectionLabel("Model"))
        row_v = QHBoxLayout()
        row_v.addWidget(QLabel("SAM type"))
        self.vit_name = QComboBox()
        self.vit_name.addItems(["vit_h", "vit_l", "vit_b"])
        row_v.addWidget(self.vit_name)
        inner_layout.addLayout(row_v)

        row_lr = QHBoxLayout()
        row_lr.addWidget(QLabel("LoRA rank"))
        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(1, 64)
        self.lora_rank.setValue(4)
        row_lr.addWidget(self.lora_rank)
        inner_layout.addLayout(row_lr)

        row_dev = QHBoxLayout()
        row_dev.addWidget(QLabel("Device"))
        self.device = QComboBox()
        self._populate_devices()
        row_dev.addWidget(self.device)
        inner_layout.addLayout(row_dev)

        # --- Prediction params ---
        inner_layout.addWidget(SectionLabel("Parameters"))
        for label, attr, lo, hi, val, dec, step in [
            ("Points/side", "points_per_side", 4, 128, 32, 0, 4),
            ("IoU threshold", "pred_iou_thresh", 0.0, 1.0, 0.8, 2, 0.05),
            ("Stability score", "stability_score_thresh", 0.0, 1.0, 0.6, 2, 0.05),
            ("Box NMS thresh", "box_nms_thresh", 0.0, 1.0, 0.05, 3, 0.01),
            ("Min mask area", "min_mask_area", 0, 10000, 20, 0, 10),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            if dec == 0:
                w = QSpinBox()
                w.setRange(int(lo), int(hi))
                w.setValue(int(val))
                w.setSingleStep(int(step))
            else:
                w = QDoubleSpinBox()
                w.setDecimals(dec)
                w.setRange(lo, hi)
                w.setValue(val)
                w.setSingleStep(step)
            setattr(self, attr, w)
            row.addWidget(w)
            inner_layout.addLayout(row)

        # --- Progress ---
        inner_layout.addWidget(SectionLabel("Progress"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate by default
        self.progress_bar.setVisible(False)
        inner_layout.addWidget(self.progress_bar)

        # --- Buttons ---
        self.run_btn = QPushButton("▶  Run Prediction")
        self.run_btn.setFixedHeight(36)
        self.run_btn.setStyleSheet(
            "background-color: #1B5E20; color: white; font-weight: bold; border-radius: 4px;"
        )
        self.run_btn.clicked.connect(self._run_prediction)
        inner_layout.addWidget(self.run_btn)

        # --- Log ---
        inner_layout.addWidget(SectionLabel("Log"))
        from PyQt6.QtWidgets import QTextEdit
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(120)
        self.log_box.setStyleSheet("background: #1A2E1A; color: #A0A0A0; font-family: Menlo, Monaco, Courier; font-size: 11px;")
        inner_layout.addWidget(self.log_box)

        inner_layout.addStretch()
        inner.setLayout(inner_layout)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self.setMinimumWidth(280)

        self._log_signal.connect(self._append_log)
        self._done_signal.connect(self._show_results)

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
        lora_path = self.lora_path.text().strip()
        image_path = self.image_path.text().strip()

        if not sam_path or not Path(sam_path).exists():
            raise ValueError(f"SAM backbone not found: {sam_path}")
        if not lora_path or not Path(lora_path).exists():
            raise ValueError(f"LoRA checkpoint not found: {lora_path}")
        if not image_path or not Path(image_path).exists():
            raise ValueError(f"Image not found: {image_path}")

        return {
            "vit_name": self.vit_name.currentText(),
            "model_path": sam_path,
            "lora_path": lora_path,
            "image_path": image_path,
            "image_encoder_lora_rank": self.lora_rank.value(),
            "mask_decoder_lora_rank": self.lora_rank.value(),
            "sam_image_size": 1024,
            "resize_size": [512, 512],
            "points_per_side": self.points_per_side.value(),
            "points_per_batch": 64,
            "pred_iou_thresh": self.pred_iou_thresh.value(),
            "stability_score_thresh": self.stability_score_thresh.value(),
            "stability_score_offset": 0.8,
            "box_nms_thresh": self.box_nms_thresh.value(),
            "crop_nms_thresh": 0.05,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": self.min_mask_area.value(),
            "max_mask_region_area_ratio": 0.1,
            "selected_device": self.device.currentText(),
            "deterministic": True,
            "seed": 0,
            "allow_tf32_on_cudnn": True,
            "allow_tf32_on_matmul": True,
        }

    def _run_prediction(self):
        try:
            config = self._build_config()
        except ValueError as e:
            self._append_log(f"[ERROR] {e}")
            return

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self._append_log(f"Running prediction on {Path(config['image_path']).name} …")

        def run():
            try:
                image_arr, label_mask = _predict(config)
                n_cells = int(label_mask.max()) if label_mask is not None else 0
                self._done_signal.emit(image_arr, label_mask)
                self._log_signal.emit(f"Done — {n_cells} cells found.")
            except Exception as e:
                self._log_signal.emit(f"[ERROR] {e}")
            finally:
                from PyQt6.QtCore import QMetaObject
                QMetaObject.invokeMethod(self, "_on_done", Qt.ConnectionType.QueuedConnection)

        self._pred_thread = threading.Thread(target=run, daemon=True)
        self._pred_thread.start()

    def _on_done(self):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _show_results(self, image_arr, label_mask):
        name = Path(self.image_path.text()).stem

        # Remove old layers with same name
        for layer in list(self.viewer.layers):
            if layer.name.startswith(name):
                self.viewer.layers.remove(layer)

        self.viewer.add_image(image_arr, name=f"{name}_image")

        if label_mask is not None and label_mask.max() > 0:
            self.viewer.add_labels(label_mask.astype(np.int32), name=f"{name}_masks")
            self.viewer.reset_view()

    def _append_log(self, text):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )


def _predict(config):
    import os
    dev = config.get("selected_device", "cpu")
    if dev == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif dev == "mps":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = dev

    import cv2
    import numpy as np
    from data.utils import resize_image
    from predict import load_model_from_config, predict_images

    # predict.py uses result_pth_path for the LoRA checkpoint
    predict_config = dict(config)
    predict_config["result_pth_path"] = config["lora_path"]

    image_path = config["image_path"]
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = resize_image(img_rgb, predict_config["resize_size"])
    pred_masks_int = predict_images(predict_config, [img_resized])
    # Convert integer label mask back to list-of-dict format for napari labels
    label_mask = pred_masks_int[0]
    return img_rgb, label_mask
