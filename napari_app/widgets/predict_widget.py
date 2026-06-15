import threading
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QScrollArea, QProgressBar, QTextEdit,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from gui.pages.utils.predict_state_manager import PredictionStateManager
from project_root import STORAGE_DIR

LORA_DIR = STORAGE_DIR / "loras"
BUILTIN_LORA_DIR = Path(__file__).parents[2] / "checkpoints"
TEST_IMAGE_DIR = STORAGE_DIR / "test_images"

LORA_META = {
    "cellpose_specialized_12.pth":      ("General cells",         0.917),
    "cellseg_blood_117.pth":            ("Blood cells",           0.941),
    "deepbacs_rod_brightfield_9.pth":   ("E. coli brightfield",  0.860),
    "deepbacs_rod_fluorescence_75.pth": ("B. subtilis fluor.",    0.821),
    "dsb2018_stardist_435.pth":         ("Cell nuclei",           0.872),
}

STATE_MANAGER = PredictionStateManager(str(STORAGE_DIR))


def _pick_file(parent, line_edit, caption, ext="All (*)"):
    start = str(Path(line_edit.text()).parent) if line_edit.text() else str(Path.home())
    path, _ = QFileDialog.getOpenFileName(
        parent, caption, start, ext,
        options=QFileDialog.Option.DontUseNativeDialog,
    )
    if path:
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
        self.setStyleSheet("color: #A5D6A7; margin-top: 6px;")


class PredictWidget(QWidget):
    _log_signal = pyqtSignal(str)
    _done_signal = pyqtSignal(object, object)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._pred_thread = None

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

        # ── LoRA checkpoint ──────────────────────────────────────────────────
        L.addWidget(SectionLabel("LoRA checkpoint"))
        self.lora_combo = QComboBox()
        self.lora_combo.setToolTip("Select a pre-trained or custom LoRA checkpoint")
        self._populate_lora_combo()
        L.addWidget(self.lora_combo)

        L.addWidget(QLabel("or load custom .pth:"))
        row_lora = QHBoxLayout()
        self.lora_custom = QLineEdit()
        self.lora_custom.setPlaceholderText("Leave empty to use selection above")
        row_lora.addWidget(self.lora_custom)
        b = QPushButton("…"); b.setFixedWidth(30)
        b.clicked.connect(lambda: _pick_file(
            self, self.lora_custom, "Select LoRA checkpoint", "PyTorch (*.pth)"))
        row_lora.addWidget(b)
        L.addLayout(row_lora)

        L.addWidget(_divider())

        # ── Input image ──────────────────────────────────────────────────────
        L.addWidget(SectionLabel("Input image"))
        row_img = QHBoxLayout()
        self.image_path = QLineEdit()
        default_img = _find_test_image()
        if default_img:
            self.image_path.setText(str(default_img))
        row_img.addWidget(self.image_path)
        b2 = QPushButton("…"); b2.setFixedWidth(30)
        b2.clicked.connect(lambda: _pick_file(
            self, self.image_path, "Select image",
            "Images (*.png *.tif *.tiff *.jpg *.bmp *.npy)"))
        row_img.addWidget(b2)
        L.addLayout(row_img)

        L.addWidget(_divider())

        # ── SAM / model settings ─────────────────────────────────────────────
        L.addWidget(SectionLabel("Model"))
        row_v = QHBoxLayout()
        row_v.addWidget(QLabel("SAM type"))
        self.vit_name = QComboBox()
        self.vit_name.addItems(["vit_h", "vit_l", "vit_b"])
        row_v.addWidget(self.vit_name)
        L.addLayout(row_v)

        row_sam = QHBoxLayout()
        row_sam.addWidget(QLabel("SAM backbone"))
        self.sam_path = QLineEdit()
        self.sam_path.setPlaceholderText("auto")
        self.sam_path.setToolTip("Leave blank to auto-detect from streamlit_storage/sam_backbone/")
        row_sam.addWidget(self.sam_path)
        b3 = QPushButton("…"); b3.setFixedWidth(30)
        b3.clicked.connect(lambda: _pick_file(
            self, self.sam_path, "Select SAM backbone", "PyTorch (*.pth)"))
        row_sam.addWidget(b3)
        L.addLayout(row_sam)

        row_lr = QHBoxLayout()
        row_lr.addWidget(QLabel("LoRA rank"))
        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(1, 64)
        self.lora_rank.setValue(4)
        row_lr.addWidget(self.lora_rank)
        L.addLayout(row_lr)

        row_dev = QHBoxLayout()
        row_dev.addWidget(QLabel("Device"))
        self.device = QComboBox()
        self._populate_devices()
        row_dev.addWidget(self.device)
        L.addLayout(row_dev)

        L.addWidget(_divider())

        # ── Inference parameters ─────────────────────────────────────────────
        L.addWidget(SectionLabel("Inference parameters"))
        params = [
            ("Points/side", "points_per_side", 4, 128, 32, 0, 4),
            ("IoU threshold", "pred_iou_thresh", 0.0, 1.0, 0.8, 2, 0.05),
            ("Stability score", "stability_score_thresh", 0.0, 1.0, 0.6, 2, 0.05),
            ("Box NMS thresh", "box_nms_thresh", 0.0, 1.0, 0.05, 3, 0.01),
            ("Min mask area", "min_mask_area", 0, 10000, 20, 0, 10),
        ]
        for label, attr, lo, hi, val, dec, step in params:
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
            L.addLayout(row)

        L.addWidget(_divider())

        # ── Run button + progress ────────────────────────────────────────────
        self.run_btn = QPushButton("▶  Run Prediction")
        self.run_btn.setFixedHeight(36)
        self.run_btn.setStyleSheet(
            "background-color: #1B5E20; color: white; font-weight: bold; border-radius: 4px;")
        self.run_btn.clicked.connect(self._run_prediction)
        L.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        L.addWidget(self.progress_bar)

        # ── Log ──────────────────────────────────────────────────────────────
        L.addWidget(SectionLabel("Log"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(110)
        self.log_box.setStyleSheet(
            "background: #1A2E1A; color: #A0C0A0; font-family: Menlo, Monaco, Courier; font-size: 11px;")
        L.addWidget(self.log_box)

        L.addStretch()
        inner.setLayout(L)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.setLayout(outer)
        self.setMinimumWidth(290)

        self._log_signal.connect(self._append_log)
        self._done_signal.connect(self._show_results)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _populate_lora_combo(self):
        self.lora_combo.clear()
        self._lora_paths = {}

        # Built-in pre-trained checkpoints
        for f in sorted(BUILTIN_LORA_DIR.glob("*.pth")):
            desc, mAP = LORA_META.get(f.name, ("Custom", 0.0))
            label = f"{desc}  ·  mAP {mAP:.3f}  [{f.name}]" if mAP else f.name
            self.lora_combo.addItem(label)
            self._lora_paths[label] = str(f)

        # User-trained models in storage
        for f in sorted(LORA_DIR.glob("*.pth")):
            if f.name not in LORA_META:
                label = f"[trained] {f.name}"
                self.lora_combo.addItem(label)
                self._lora_paths[label] = str(f)

    def _populate_devices(self):
        import torch
        self.device.addItem("cpu")
        if torch.backends.mps.is_available():
            self.device.addItem("mps")
            self.device.setCurrentText("mps")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.device.addItem(str(i))

    def _resolve_lora_path(self):
        custom = self.lora_custom.text().strip()
        if custom:
            return custom
        label = self.lora_combo.currentText()
        return self._lora_paths.get(label, "")

    def _resolve_sam_path(self):
        manual = self.sam_path.text().strip()
        if manual and Path(manual).exists():
            return manual
        vit = self.vit_name.currentText()
        names = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }
        candidate = STORAGE_DIR / "sam_backbone" / names[vit]
        if candidate.exists():
            return str(candidate)
        raise ValueError(f"SAM backbone not found for {vit}. Download it or specify the path manually.")

    def _build_config(self):
        lora_path = self._resolve_lora_path()
        image_path = self.image_path.text().strip()
        sam_path = self._resolve_sam_path()

        if not lora_path or not Path(lora_path).exists():
            raise ValueError(f"LoRA checkpoint not found: {lora_path}")
        if not image_path or not Path(image_path).exists():
            raise ValueError(f"Image not found: {image_path}")

        return {
            "vit_name": self.vit_name.currentText(),
            "model_path": sam_path,
            "result_pth_path": lora_path,
            "image_path": image_path,
            "image_encoder_lora_rank": self.lora_rank.value(),
            "mask_decoder_lora_rank": self.lora_rank.value(),
            "sam_image_size": 512,
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

    # ── Actions ──────────────────────────────────────────────────────────────

    def _run_prediction(self):
        try:
            config = self._build_config()
        except ValueError as e:
            self._append_log(f"[ERROR] {e}")
            return

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self._append_log(f"Running on {Path(config['image_path']).name} …")

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
        # Refresh lora list in case a new model was saved
        self._populate_lora_combo()

    def _show_results(self, image_arr, label_mask):
        name = Path(self.image_path.text()).stem
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
            self.log_box.verticalScrollBar().maximum())


# ── Prediction logic (runs in thread) ────────────────────────────────────────

def _predict(config):
    import os
    dev = config.get("selected_device", "cpu")
    if dev in ("cpu", "mps"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if dev == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    if dev not in ("cpu", "mps"):
        os.environ["CUDA_VISIBLE_DEVICES"] = dev

    import cv2
    from data.utils import resize_image
    from predict import predict_images

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

    img_resized = resize_image(img_rgb, config["resize_size"])
    label_mask = predict_images(config, [img_resized])[0]
    return img_rgb, label_mask


def _find_test_image():
    for d in [TEST_IMAGE_DIR]:
        for ext in ("*.png", "*.tif", "*.tiff", "*.jpg"):
            hits = list(d.glob(ext))
            if hits:
                return hits[0]
    return None
