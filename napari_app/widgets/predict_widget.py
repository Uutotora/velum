import threading
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QScrollArea, QProgressBar, QTextEdit,
    QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from gui.pages.utils.predict_state_manager import PredictionStateManager
from project_root import STORAGE_DIR
from napari_app.theme import (
    WIDGET_SS, BTN_SUCCESS, BTN_SECONDARY, BTN_BROWSE,
    BG, FG, BORDER, TEXT, ACCENT, DIM, CONSOLE,
)
from napari_app.widgets.common import CollapsibleSection, divider as _divider, param_row as _param_row

LORA_DIR        = STORAGE_DIR / "loras"
BUILTIN_LORA_DIR = Path(__file__).parents[2] / "checkpoints"
TEST_IMAGE_DIR  = STORAGE_DIR / "test_images"

LORA_META = {
    "cellpose_specialized_12.pth":      ("General cells",         0.917),
    "cellseg_blood_117.pth":            ("Blood cells",           0.941),
    "deepbacs_rod_brightfield_9.pth":   ("E. coli brightfield",  0.860),
    "deepbacs_rod_fluorescence_75.pth": ("B. subtilis fluor.",    0.821),
    "dsb2018_stardist_435.pth":         ("Cell nuclei",           0.872),
}

STATE_MANAGER = PredictionStateManager(str(STORAGE_DIR))
_DLG = QFileDialog.Option.DontUseNativeDialog


# ── Reusable UI helpers ───────────────────────────────────────────────────────

def _browse_btn(parent, callback):
    b = QPushButton("⋯")
    b.setFixedSize(28, 28)
    b.setStyleSheet(BTN_BROWSE)
    b.clicked.connect(callback)
    return b


def _pick_file(parent, line_edit, caption, ext="All (*)"):
    start = str(Path(line_edit.text()).parent) if line_edit.text() else str(Path.home())
    p, _ = QFileDialog.getOpenFileName(parent, caption, start, ext, options=_DLG)
    if p:
        line_edit.setText(p)


def _file_row(parent, line_edit, caption, ext="All (*)"):
    row = QHBoxLayout()
    row.setSpacing(4)
    row.addWidget(line_edit)
    row.addWidget(_browse_btn(parent, lambda: _pick_file(parent, line_edit, caption, ext)))
    return row




class PredictWidget(QWidget):
    _log_signal    = pyqtSignal(str)
    _done_signal   = pyqtSignal(object, object)
    _finish_signal = pyqtSignal()

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._pred_thread   = None
        self._last_mask     = None
        self._last_img_path = None
        self._lora_paths    = {}

        self.setStyleSheet(WIDGET_SS)

        outer = QVBoxLayout()
        outer.setSpacing(0)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        L = QVBoxLayout()
        L.setSpacing(8)
        L.setContentsMargins(12, 12, 12, 12)

        # ── Quick run ─────────────────────────────────────────────────────────
        qr = CollapsibleSection("Quick run")

        qr.addWidget(QLabel("Checkpoint"))
        self.lora_combo = QComboBox()
        self._populate_lora_combo()
        self.lora_combo.currentIndexChanged.connect(self._on_checkpoint_changed)
        qr.addWidget(self.lora_combo)

        qr.addWidget(QLabel("Image"))
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Select microscopy image…")
        img = _find_test_image()
        if img:
            self.image_path.setText(str(img))
        qr.addLayout(_file_row(self, self.image_path, "Select image",
            "Images (*.png *.tif *.tiff *.jpg *.bmp *.npy)"))

        self.run_btn = QPushButton("▶   Run Prediction")
        self.run_btn.setFixedHeight(38)
        self.run_btn.setStyleSheet(BTN_SUCCESS)
        self.run_btn.clicked.connect(self._run_prediction)
        qr.addWidget(self.run_btn)

        self.active_btn = QPushButton("▶   Predict on active layer")
        self.active_btn.setStyleSheet(BTN_SECONDARY)
        self.active_btn.setToolTip(
            "Run prediction on whatever Image layer is currently\n"
            "selected in the napari viewer — no file picker needed."
        )
        self.active_btn.clicked.connect(self._predict_active_layer)
        qr.addWidget(self.active_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(4)
        qr.addWidget(self.progress_bar)

        L.addWidget(qr)
        L.addWidget(_divider())

        # ── Results ───────────────────────────────────────────────────────────
        self.results_section = CollapsibleSection("Results")
        self.results_section.setVisible(False)

        self._stats_lbl = QLabel()
        self._stats_lbl.setStyleSheet(
            f"color:{TEXT}; background:{FG}; border-radius:5px; padding:8px;"
            f"font-size:12px; line-height:160%;")
        self._stats_lbl.setWordWrap(True)
        self.results_section.addWidget(self._stats_lbl)

        save_btn = QPushButton("💾  Save masks as PNG/TIFF")
        save_btn.setStyleSheet(BTN_SECONDARY)
        save_btn.clicked.connect(self._save_masks)
        self.results_section.addWidget(save_btn)

        L.addWidget(self.results_section)
        L.addWidget(_divider())

        # ── Ground truth ──────────────────────────────────────────────────────
        gt_sec = CollapsibleSection("Ground truth overlay")
        gt_sec._on_toggle(False)  # collapsed by default

        gt_sec.addWidget(QLabel("GT mask file"))
        self.gt_path = QLineEdit()
        self.gt_path.setPlaceholderText("Optional: compare with ground truth")
        gt_sec.addLayout(_file_row(self, self.gt_path, "Select ground truth",
            "Images (*.png *.tif *.tiff *.npy)"))
        gt_btn = QPushButton("Show GT layer")
        gt_btn.setStyleSheet(BTN_SECONDARY)
        gt_btn.clicked.connect(self._show_ground_truth)
        gt_sec.addWidget(gt_btn)

        L.addWidget(gt_sec)
        L.addWidget(_divider())

        # ── Model settings ────────────────────────────────────────────────────
        mdl_sec = CollapsibleSection("Model settings")
        mdl_sec._on_toggle(False)

        mdl_sec.addLayout(_param_row("Custom .pth", _make_custom_lora_row(self)))

        row_vit = QHBoxLayout(); row_vit.setSpacing(8)
        row_vit.addWidget(QLabel("SAM type", styleSheet=f"color:{DIM}; min-width:115px;"))
        self.vit_name = QComboBox()
        self.vit_name.addItems(["vit_h", "vit_l", "vit_b"])
        self.vit_name.currentTextChanged.connect(self._on_vit_changed)
        row_vit.addWidget(self.vit_name)
        mdl_sec.addLayout(row_vit)

        self.sam_path = QLineEdit()
        self.sam_path.setPlaceholderText("auto-detected")
        self._on_vit_changed("vit_h")
        mdl_sec.addLayout(_file_row(self, self.sam_path, "Select SAM backbone", "PyTorch (*.pth)"))

        self.lora_rank = QSpinBox(); self.lora_rank.setRange(1, 64); self.lora_rank.setValue(4)
        mdl_sec.addLayout(_param_row("LoRA rank", self.lora_rank,
            "Must match the rank used during training (default 4)"))

        self.device = QComboBox(); self._populate_devices()
        mdl_sec.addLayout(_param_row("Device", self.device))

        L.addWidget(mdl_sec)
        L.addWidget(_divider())

        # ── Inference parameters ──────────────────────────────────────────────
        inf_sec = CollapsibleSection("Inference parameters")
        inf_sec._on_toggle(False)

        self.resize_size = QComboBox()
        for v in ["256", "512", "768", "1024"]:
            self.resize_size.addItem(v)
        self.resize_size.setCurrentText("512")
        inf_sec.addLayout(_param_row("Resize size", self.resize_size,
            "SAM inference resolution. Higher = better accuracy, slower. Try 1024 for small particles."))

        params = [
            ("Points/side",     "points_per_side",        4,   128,  32,  0, 4,
             "Grid density for mask proposals. Higher = more candidates (slower)."),
            ("IoU threshold",   "pred_iou_thresh",         0.0, 1.0, 0.8,  2, 0.05,
             "Min predicted IoU to keep a mask. Raise to reduce false positives."),
            ("Stability score", "stability_score_thresh",  0.0, 1.0, 0.6,  2, 0.05,
             "Mask stability threshold. Raise to keep only confident masks."),
            ("Box NMS thresh",  "box_nms_thresh",          0.0, 1.0, 0.05, 3, 0.01,
             "Non-max suppression overlap. Lower = separates touching objects better."),
            ("Min mask area",   "min_mask_area",           0,   10000, 20, 0, 10,
             "Discard masks smaller than this (pixels)."),
        ]
        for label, attr, lo, hi, val, dec, step, tip in params:
            if dec == 0:
                w = QSpinBox(); w.setRange(int(lo), int(hi)); w.setValue(int(val)); w.setSingleStep(int(step))
            else:
                w = QDoubleSpinBox(); w.setDecimals(dec); w.setRange(lo, hi); w.setValue(val); w.setSingleStep(step)
            setattr(self, attr, w)
            inf_sec.addLayout(_param_row(label, w, tip))

        L.addWidget(inf_sec)
        L.addWidget(_divider())

        # ── Log ───────────────────────────────────────────────────────────────
        log_sec = CollapsibleSection("Log")
        log_sec._on_toggle(False)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(100)
        log_sec.addWidget(self.log_box)
        L.addWidget(log_sec)

        L.addStretch()
        inner.setLayout(L)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.setLayout(outer)
        self.setMinimumWidth(320)

        self._log_signal.connect(self._append_log)
        self._done_signal.connect(self._show_results)
        self._finish_signal.connect(self._on_done)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _on_checkpoint_changed(self, _index: int):
        from napari_app.inference_cache import invalidate_model
        invalidate_model()
        self._autofill_from_sidecar()

    def _autofill_from_sidecar(self):
        """Read .json sidecar next to checkpoint and populate SAM type / resize / rank."""
        lora_path = self._resolve_lora()
        if not lora_path:
            return
        sidecar = Path(lora_path).with_suffix(".json")
        if not sidecar.exists():
            return
        try:
            import json
            with open(sidecar) as f:
                meta = json.load(f)
        except Exception:
            return

        vit = meta.get("vit_name")
        if vit and vit in ("vit_h", "vit_l", "vit_b"):
            self.vit_name.setCurrentText(vit)

        rs = meta.get("sam_image_size") or (meta.get("resize_size") or [None])[0]
        if rs and str(rs) in ("256", "512", "768", "1024"):
            self.resize_size.setCurrentText(str(rs))

        rank = meta.get("image_encoder_lora_rank")
        if rank and isinstance(rank, int):
            self.lora_rank.setValue(rank)

        fl = meta.get("final_loss")
        epochs = meta.get("epochs_run", "?")
        epoch_max = meta.get("epoch_max", "?")
        saved_at = (meta.get("saved_at", "") or "")[:16].replace("T", " ")
        parts = []
        if saved_at:
            parts.append(saved_at)
        if fl is not None:
            parts.append(f"loss {fl:.5f}")
        parts.append(f"{epochs}/{epoch_max} ep")
        self._append_log("auto: " + "  ·  ".join(parts))

    def _populate_lora_combo(self):
        self.lora_combo.clear()
        self._lora_paths = {}
        for f in sorted(BUILTIN_LORA_DIR.glob("*.pth")):
            desc, mAP = LORA_META.get(f.name, ("Custom", 0.0))
            lbl = f"{desc}  ·  mAP {mAP:.3f}" if mAP else f.name
            self.lora_combo.addItem(lbl)
            self._lora_paths[lbl] = str(f)
        for f in sorted(LORA_DIR.glob("*.pth")):
            if f.name not in LORA_META:
                lbl = f"[trained]  {f.stem}"
                self.lora_combo.addItem(lbl)
                self._lora_paths[lbl] = str(f)

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
        c = STORAGE_DIR / "sam_backbone" / names.get(vit, "")
        if c.exists():
            self.sam_path.setText(str(c))
        else:
            self.sam_path.clear(); self.sam_path.setPlaceholderText(f"Not found: {c.name}")

    def _resolve_lora(self):
        if hasattr(self, 'lora_custom') and self.lora_custom.text().strip():
            return self.lora_custom.text().strip()
        return self._lora_paths.get(self.lora_combo.currentText(), "")

    def _resolve_sam(self):
        p = self.sam_path.text().strip()
        if p and Path(p).exists():
            return p
        vit = self.vit_name.currentText()
        names = {"vit_h": "sam_vit_h_4b8939.pth",
                 "vit_l": "sam_vit_l_0b3195.pth",
                 "vit_b": "sam_vit_b_01ec64.pth"}
        c = STORAGE_DIR / "sam_backbone" / names[vit]
        if c.exists():
            return str(c)
        raise ValueError(f"SAM backbone not found. Place {names[vit]} in {STORAGE_DIR/'sam_backbone'}/")

    def _build_config(self):
        lora = self._resolve_lora()
        img  = self.image_path.text().strip()
        sam  = self._resolve_sam()
        if not lora or not Path(lora).exists():
            raise ValueError(f"LoRA checkpoint not found: {lora}")
        if not img or not Path(img).exists():
            raise ValueError(f"Image not found: {img}")
        rs = int(self.resize_size.currentText())
        return {
            "vit_name": self.vit_name.currentText(),
            "model_path": sam, "result_pth_path": lora, "image_path": img,
            "image_encoder_lora_rank": self.lora_rank.value(),
            "mask_decoder_lora_rank":  self.lora_rank.value(),
            "freeze_image_encoder": True, "freeze_prompt_encoder": True,
            "freeze_mask_decoder_transformer": True, "freeze_upscaling_cnn": True,
            "freeze_output_hypernetworks_mlps": True,
            "freeze_mask_decoder_mask_tokens": True, "freeze_mask_decoder_iou": True,
            "lora_dropout": 0.1,
            "sam_image_size": rs, "resize_size": [rs, rs],
            "points_per_side": self.points_per_side.value(),
            "points_per_batch": 64,
            "pred_iou_thresh": self.pred_iou_thresh.value(),
            "stability_score_thresh": self.stability_score_thresh.value(),
            "stability_score_offset": 0.8,
            "box_nms_thresh": self.box_nms_thresh.value(),
            "crop_nms_thresh": 0.05, "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": self.min_mask_area.value(),
            "max_mask_region_area_ratio": 0.1,
            "selected_device": self.device.currentText(),
            "deterministic": True, "seed": 0,
            "allow_tf32_on_cudnn": True, "allow_tf32_on_matmul": True,
        }

    # ── Actions ──────────────────────────────────────────────────────────────

    def _predict_active_layer(self):
        """Run prediction on the active Image layer in the napari viewer."""
        import napari.layers as nl
        import tempfile, cv2, numpy as np

        image_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, nl.Image):
                image_layer = layer
                break

        if image_layer is None:
            self._append_log("[ERROR] No Image layer found in viewer"); return

        img = np.asarray(image_layer.data, dtype=np.float32)
        if img.max() > 1.0:
            img = (img / img.max() * 255).clip(0, 255).astype(np.uint8)
        else:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Write to a temp PNG so the rest of the pipeline can read image_path
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.image_path.setText(tmp.name)
        self._run_prediction()

    def _run_prediction(self):
        try:
            config = self._build_config()
        except ValueError as e:
            self._append_log(f"[ERROR] {e}"); return

        self.run_btn.setEnabled(False)
        self.active_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.results_section.setVisible(False)

        from napari_app.inference_cache import cache_status
        self._append_log(f"▶ {Path(config['image_path']).name}  [{cache_status()}]")

        def run():
            try:
                img_arr, mask = _predict_cached(config)
                self._done_signal.emit(img_arr, mask)
                from napari_app.inference_cache import cache_status as cs
                self._log_signal.emit(f"✓ {int(mask.max())} cells  [{cs()}]")
            except Exception as e:
                import traceback
                self._log_signal.emit(f"[ERROR] {e}\n{traceback.format_exc()}")
            finally:
                self._finish_signal.emit()

        threading.Thread(target=run, daemon=True).start()

    def _on_done(self):
        self.run_btn.setEnabled(True)
        self.active_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._populate_lora_combo()

    def _show_results(self, img_arr, mask):
        name = Path(self.image_path.text()).stem
        self._last_mask = mask
        self._last_img_path = self.image_path.text()

        for lyr in list(self.viewer.layers):
            if lyr.name.startswith(name) and "_gt" not in lyr.name:
                self.viewer.layers.remove(lyr)

        self.viewer.add_image(img_arr, name=f"{name}_image")
        if mask is not None and mask.max() > 0:
            lyr = self.viewer.add_labels(mask.astype(np.int32), name=f"{name}_masks", opacity=0.7)
            lyr.contour = 2
            self._show_stats(mask, img_arr.shape[:2])
        self.viewer.reset_view()

    def _show_stats(self, mask, shape):
        n = int(mask.max())
        counts = np.bincount(mask.ravel())       # O(N) single pass
        areas = counts[1:n + 1].tolist() if n > 0 else []
        avg_a = int(np.mean(areas)) if areas else 0
        med_a = int(np.median(areas)) if areas else 0
        cov   = counts[1:].sum() / (shape[0] * shape[1]) * 100
        self._stats_lbl.setText(
            f"<b style='font-size:18px;color:{TEXT};'>{n}</b>"
            f"<span style='color:{DIM};'> cells detected</span><br>"
            f"<span style='color:{DIM};'>Avg {avg_a} px²  ·  Median {med_a} px²"
            f"  ·  {cov:.1f}% coverage</span>"
        )
        self.results_section.setVisible(True)

    def _show_ground_truth(self):
        p = self.gt_path.text().strip()
        if not p or not Path(p).exists():
            self._append_log("[ERROR] GT file not found"); return
        try:
            import cv2
            gt = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if gt is None:
                gt = np.load(p)
            name = Path(self.image_path.text()).stem
            lname = f"{name}_gt"
            for lyr in list(self.viewer.layers):
                if lyr.name == lname:
                    self.viewer.layers.remove(lyr)
            l = self.viewer.add_labels(gt.astype(np.int32), name=lname, opacity=0.5)
            l.contour = 2
            self._append_log(f"✓ GT loaded — {int(gt.max())} cells")
        except Exception as e:
            self._append_log(f"[ERROR] {e}")

    def _save_masks(self):
        if self._last_mask is None:
            return
        stem = Path(self._last_img_path).stem if self._last_img_path else "mask"
        default = str(STORAGE_DIR / "predict_masks" / f"{stem}_mask.png")
        p, _ = QFileDialog.getSaveFileName(
            self, "Save mask", default, "PNG (*.png);;TIFF (*.tif)", options=_DLG)
        if not p:
            return
        import cv2
        cv2.imwrite(p, self._last_mask.astype(np.uint16))
        self._append_log(f"✓ Saved {Path(p).name}")

    def _append_log(self, text):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())


def _make_custom_lora_row(parent):
    """Returns a widget containing the custom lora path field — assigned to parent."""
    parent.lora_custom = QLineEdit()
    parent.lora_custom.setPlaceholderText("Leave blank to use dropdown above")
    container = QWidget()
    row = QHBoxLayout(); row.setContentsMargins(0,0,0,0); row.setSpacing(4)
    row.addWidget(parent.lora_custom)
    row.addWidget(_browse_btn(parent,
        lambda: _pick_file(parent, parent.lora_custom, "Select LoRA checkpoint", "PyTorch (*.pth)")))
    container.setLayout(row)
    return container


# ── Prediction core (with model + embedding cache) ────────────────────────────

def _predict_cached(config):
    """Load image, run prediction via inference_cache, scale mask back."""
    import cv2
    from data.utils import resize_image
    from napari_app.inference_cache import predict_cached

    img = cv2.imread(config["image_path"], cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read: {config['image_path']}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = img.shape[:2]
    resized = resize_image(img, config["resize_size"])
    small = predict_cached(config, resized)

    if small.shape != (orig_h, orig_w):
        mask = cv2.resize(small.astype(np.float32), (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST).astype(small.dtype)
    else:
        mask = small
    return img, mask


def _find_test_image():
    for ext in ("*.png", "*.tif", "*.tiff", "*.jpg"):
        hits = list(TEST_IMAGE_DIR.glob(ext))
        if hits:
            return hits[0]
    return None
