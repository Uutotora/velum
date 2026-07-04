import queue as _queue
import threading
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QScrollArea, QProgressBar, QFrame,
    QAbstractSpinBox, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
)
from napari_app.widgets.log_window import get_log_window
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor

from napari_app.core.predict_state_manager import PredictionStateManager
from project_root import STORAGE_DIR
from napari_app.theme import (
    WIDGET_SS, BTN_PRIMARY, BTN_SUCCESS, BTN_SECONDARY, BTN_BROWSE,
    BG, FG, BORDER, TEXT, ACCENT, DIM, LABEL, CONSOLE,
)
from napari_app.widgets.common import (
    section_header, divider as _divider, param_row as _param_row,
    CollapsibleSection, SectionCard, CollapsibleCard,
)

LORA_DIR         = STORAGE_DIR / "loras"
BUILTIN_LORA_DIR = Path(__file__).parents[2] / "checkpoints"
TEST_IMAGE_DIR   = STORAGE_DIR / "test_images"

LORA_META = {
    "cellpose_specialized_12.pth":      ("General cells",        0.917),
    "cellseg_blood_117.pth":            ("Blood cells",          0.941),
    "deepbacs_rod_brightfield_9.pth":   ("E. coli brightfield",  0.860),
    "deepbacs_rod_fluorescence_75.pth": ("B. subtilis fluor.",   0.821),
    "dsb2018_stardist_435.pth":         ("Cell nuclei",          0.872),
}

STATE_MANAGER = PredictionStateManager(str(STORAGE_DIR))
_DLG = QFileDialog.Option.DontUseNativeDialog


# ── Reusable UI helpers ────────────────────────────────────────────────────────

def _browse_btn(parent, callback):
    b = QPushButton("⋯")
    b.setFixedSize(30, 30)
    b.setStyleSheet(BTN_BROWSE)
    b.clicked.connect(callback)
    return b


def _pick_file(parent, line_edit, caption, ext="All (*)"):
    start = str(Path(line_edit.text()).parent) if line_edit.text() else str(Path.home())
    p, _ = QFileDialog.getOpenFileName(parent, caption, start, ext, options=_DLG)
    if p:
        line_edit.setText(p)
        line_edit.setToolTip(p)


def _file_row(parent, line_edit, caption, ext="All (*)"):
    row = QHBoxLayout(); row.setSpacing(6)
    row.addWidget(line_edit)
    row.addWidget(_browse_btn(parent, lambda: _pick_file(parent, line_edit, caption, ext)))
    return row


def _dir_row(parent, line_edit, caption):
    def pick():
        p = QFileDialog.getExistingDirectory(
            parent, caption, line_edit.text() or str(Path.home()), _DLG)
        if p:
            line_edit.setText(p)
            line_edit.setToolTip(p)
    row = QHBoxLayout(); row.setSpacing(6)
    row.addWidget(line_edit)
    row.addWidget(_browse_btn(parent, pick))
    return row


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {LABEL}; font-size: 11px; font-weight: 500; padding: 3px 0 1px 0;")
    return lbl


# ── Main widget ────────────────────────────────────────────────────────────────

class PredictWidget(QWidget):
    _log_signal            = pyqtSignal(str)
    _done_signal           = pyqtSignal(object, object)
    _finish_signal         = pyqtSignal()
    _batch_progress_signal = pyqtSignal(int, int)
    _tile_progress_signal  = pyqtSignal(int, int)
    _batch_finish_signal   = pyqtSignal()
    _refine_finish_signal  = pyqtSignal(str)
    _sample_finish_signal  = pyqtSignal()
    _cohort_ready_signal   = pyqtSignal()
    _benchmark_row_signal  = pyqtSignal(str)
    _benchmark_done_signal = pyqtSignal()
    _dataset_progress_signal = pyqtSignal(str)
    _dataset_done_signal   = pyqtSignal(object)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._pred_thread   = None
        self._last_mask     = None
        self._last_img_rgb  = None
        self._last_measure  = None
        self._last_img_path = None
        self._was_autofilled = False
        self._lora_paths    = {}
        self._batch_stop    = threading.Event()
        self._refine_timer  = QTimer()
        self._refine_timer.setInterval(3000)
        self._refine_lh: list = []

        self.setAcceptDrops(True)
        self.setStyleSheet(WIDGET_SS)

        outer = QVBoxLayout()
        outer.setSpacing(0)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        L = QVBoxLayout()
        L.setSpacing(0)
        L.setContentsMargins(14, 8, 14, 16)

        # ── Engine ────────────────────────────────────────────────────────────
        engine_card = SectionCard("Engine")
        self.engine = QComboBox()
        self.engine.addItem("CellSeg1 · LoRA (one-shot, fine-tuned)", "cellseg1")
        self.engine.addItem("Cellpose-SAM (zero-shot, generalist)",   "cellpose")
        self.engine.setToolTip(
            "CellSeg1: SAM + LoRA — best when you've fine-tuned on your own data.\n"
            "Cellpose-SAM: a 2025 generalist foundation model — strong accuracy "
            "out-of-the-box with no checkpoint or training.")
        engine_card.addWidget(self.engine)
        self._engine_hint = QLabel("")
        self._engine_hint.setStyleSheet(f"color: {LABEL}; font-size: 11px; padding-top: 3px;")
        self._engine_hint.setWordWrap(True)
        engine_card.addWidget(self._engine_hint)
        L.addWidget(engine_card)

        # ── Checkpoint ────────────────────────────────────────────────────────
        self._ckpt_card = SectionCard("Checkpoint")
        ckpt_card = self._ckpt_card
        self.lora_combo = QComboBox()
        self._populate_lora_combo()
        self.lora_combo.currentIndexChanged.connect(self._on_checkpoint_changed)
        ckpt_card.addWidget(self.lora_combo)
        self._meta_lbl = QLabel("")
        self._meta_lbl.setStyleSheet(
            f"color: {LABEL}; font-size: 11px; padding: 3px 1px 0 1px;")
        self._meta_lbl.setWordWrap(True)
        ckpt_card.addWidget(self._meta_lbl)
        L.addWidget(ckpt_card)

        # ── Image ─────────────────────────────────────────────────────────────
        img_card = SectionCard("Image")
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Path to microscopy image  (or drop here)")
        img = _find_test_image()
        if img:
            self.image_path.setText(str(img))
        img_card.addLayout(_file_row(self, self.image_path, "Select image",
            "Images (*.png *.tif *.tiff *.jpg *.bmp *.npy)"))
        # Switcher first — the everyday action is *selecting* a loaded image.
        sw_row = QHBoxLayout(); sw_row.setSpacing(6)
        sw_lbl = QLabel("Sample")
        sw_lbl.setStyleSheet(f"color: {LABEL}; font-size: 11px; font-weight: 500;")
        sw_lbl.setFixedWidth(52)
        sw_row.addWidget(sw_lbl)
        self.sample_combo = QComboBox()
        self.sample_combo.setToolTip("Switch between images already in your test-images folder")
        self.sample_combo.activated.connect(self._on_sample_selected)
        sw_row.addWidget(self.sample_combo, stretch=1)
        img_card.addLayout(sw_row)

        # A single "Get images" menu button — download once, then use the switcher.
        from PyQt6.QtWidgets import QMenu
        self._sample_btn = QPushButton("⬇  Get images  ▾")
        self._sample_btn.setFixedHeight(30)
        self._sample_btn.setStyleSheet(BTN_SECONDARY)
        _menu = QMenu(self._sample_btn)
        _menu.setStyleSheet(
            f"QMenu {{ background:{FG}; color:{TEXT}; border:1px solid {BORDER}; }}"
            f"QMenu::item:selected {{ background:{ACCENT}; color:#fff; }}")
        _menu.addAction("Quick samples (offline, instant)", self._load_samples)
        _menu.addAction("Download BBBC039 nuclei dataset (real, with ground truth)",
                        self._download_dataset)
        self._sample_btn.setMenu(_menu)
        img_card.addWidget(self._sample_btn)

        self._dataset_lbl = QLabel("")
        self._dataset_lbl.setStyleSheet(
            f"color: {LABEL}; font-size: 10px; font-family:'Menlo','SF Mono',monospace;")
        self._dataset_lbl.setWordWrap(True)
        img_card.addWidget(self._dataset_lbl)
        L.addWidget(img_card)

        # ── Run (prominent, outside cards — the main action) ──────────────────
        L.addSpacing(20)

        self.run_btn = QPushButton("▶   Run Prediction")
        self.run_btn.setFixedHeight(44)
        self.run_btn.setStyleSheet(BTN_PRIMARY)
        self.run_btn.setToolTip("Ctrl+R")
        self.run_btn.clicked.connect(self._run_prediction)
        L.addWidget(self.run_btn)

        L.addSpacing(6)

        self.active_btn = QPushButton("▶   Predict on active napari layer")
        self.active_btn.setFixedHeight(32)
        self.active_btn.setStyleSheet(BTN_SECONDARY)
        self.active_btn.setToolTip("Ctrl+Shift+R — runs on the Image layer selected in viewer")
        self.active_btn.clicked.connect(self._predict_active_layer)
        L.addWidget(self.active_btn)

        # Quality selector — a friendly front-end over resize + sampling density.
        q_row = QHBoxLayout(); q_row.setSpacing(8)
        q_lbl = QLabel("Quality")
        q_lbl.setStyleSheet(f"color: {LABEL}; font-size: 11px; font-weight: 500;")
        q_lbl.setFixedWidth(70)
        q_row.addWidget(q_lbl)
        self.quality = QComboBox()
        self.quality.setToolTip(
            "Fast = low resolution & sparse sampling; Accurate = 1024px & dense "
            "sampling (slower). Sets resize and inference parameters for you.")
        q_row.addWidget(self.quality)
        L.addLayout(q_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setContentsMargins(0, 4, 0, 0)
        L.addWidget(self.progress_bar)

        # ── Results (hidden until first prediction) ────────────────────────────
        from napari_app.theme import SUCCESS
        self._results_card = SectionCard("Results", accent_color=SUCCESS)
        self._results_card.setVisible(False)

        count_row = QHBoxLayout(); count_row.setSpacing(8); count_row.setContentsMargins(0, 2, 0, 4)
        self._cell_count_lbl = QLabel("—")
        self._cell_count_lbl.setStyleSheet(
            f"color: {SUCCESS}; font-size: 34px; font-weight: 800;"
            f"letter-spacing: -1px; background: transparent;")
        _count_cap = QLabel("cells\ndetected")
        _count_cap.setStyleSheet(
            f"color: {DIM}; font-size: 11px; font-weight: 600;"
            f"letter-spacing: 0.5px; background: transparent;")
        count_row.addWidget(self._cell_count_lbl)
        count_row.addWidget(_count_cap)
        count_row.addStretch()
        self._results_card.addLayout(count_row)

        # Stat chips — three compact tiles reading left-to-right by importance.
        chips = QHBoxLayout(); chips.setSpacing(7); chips.setContentsMargins(0, 2, 0, 2)
        self._chip_diam, _f1 = _make_stat_chip("MEDIAN Ø")
        self._chip_area, _f2 = _make_stat_chip("MEAN AREA")
        self._chip_cov,  _f3 = _make_stat_chip("COVERAGE")
        for f in (_f1, _f2, _f3):
            chips.addWidget(f)
        self._results_card.addLayout(chips)
        # kept for code paths that still write a one-line summary
        self._stats_lbl = QLabel()
        self._stats_lbl.setVisible(False)

        _cal_lbl = QLabel("PIXEL CALIBRATION")
        _cal_lbl.setStyleSheet(
            f"color: {LABEL}; font-size: 10px; font-weight: 700; letter-spacing: 1px;"
            f"padding: 6px 0 2px 0;")
        self._results_card.addWidget(_cal_lbl)
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setDecimals(4)
        self.pixel_size.setRange(0.0, 1000.0)
        self.pixel_size.setValue(0.0)
        self.pixel_size.setSingleStep(0.05)
        self.pixel_size.setSuffix("  µm / pixel")
        self.pixel_size.setSpecialValueText("off  —  measure in pixels")
        self.pixel_size.setMinimumHeight(34)
        self.pixel_size.setToolTip(
            "Microns per pixel from your microscope. Set it to report areas in "
            "µm² and sizes in µm. Leave at 0 to measure in pixels.")
        self.pixel_size.valueChanged.connect(self._on_pixel_size_changed)
        self._results_card.addWidget(self.pixel_size)
        _cal_hint = QLabel(
            "Enter your microscope's µm-per-pixel to get real-world units. "
            "0 = pixels.")
        _cal_hint.setStyleSheet(f"color: {DIM}; font-size: 10px; padding-bottom: 2px;")
        _cal_hint.setWordWrap(True)
        self._results_card.addWidget(_cal_hint)

        _line = QFrame(); _line.setFrameShape(QFrame.Shape.HLine)
        _line.setStyleSheet(f"background: {BORDER}; border: none;"); _line.setFixedHeight(1)
        self._results_card.addWidget(_line)

        res_btns = QHBoxLayout(); res_btns.setSpacing(6)
        self._save_btn = QPushButton("Save masks")
        self._save_btn.setStyleSheet(BTN_SECONDARY)
        self._save_btn.clicked.connect(self._save_masks)
        self._csv_btn = QPushButton("Export CSV")
        self._csv_btn.setStyleSheet(BTN_SECONDARY)
        self._csv_btn.setToolTip("Saves cell_id, area, centroid for each detected cell")
        self._csv_btn.clicked.connect(self._export_csv)
        self._refine_btn = QPushButton("Refine…")
        self._refine_btn.setStyleSheet(BTN_SECONDARY)
        self._refine_btn.setToolTip(
            "Edit the Labels layer in napari to correct cells,\n"
            "then click to run a 50-epoch fine-tune from the current checkpoint.")
        self._refine_btn.clicked.connect(self._run_refine)
        res_btns.addWidget(self._save_btn)
        res_btns.addWidget(self._csv_btn)
        res_btns.addWidget(self._refine_btn)
        self._results_card.addLayout(res_btns)

        self._measure_btn = QPushButton("📊  Open measurements table")
        self._measure_btn.setFixedHeight(32)
        self._measure_btn.setStyleSheet(BTN_SECONDARY)
        self._measure_btn.setToolTip(
            "Per-cell morphometry: area, diameter, circularity, elongation,\n"
            "convexity and intensity — with distribution histograms.")
        self._measure_btn.clicked.connect(self._open_measurements)
        self._results_card.addWidget(self._measure_btn)
        L.addWidget(self._results_card)

        # ── Ground truth & evaluation (collapsed — validation tool) ────────────
        _gt_card = CollapsibleCard("Ground truth & evaluation", collapsed=True)
        _gt_card.addWidget(_field_label("GT mask file"))
        self.gt_path = QLineEdit()
        self.gt_path.setPlaceholderText("Auto-detected from a *_gt / masks sidecar, or pick one")
        self.gt_path.textChanged.connect(lambda _t: self._on_gt_path_changed())
        _gt_card.addLayout(_file_row(self, self.gt_path, "Select ground truth mask",
            "Images (*.png *.tif *.tiff *.npy)"))
        self._gt_status = QLabel("")
        self._gt_status.setStyleSheet(f"color: {LABEL}; font-size: 10px; padding: 1px 0;")
        self._gt_status.setWordWrap(True)
        _gt_card.addWidget(self._gt_status)

        gt_row = QHBoxLayout(); gt_row.setSpacing(6)
        gt_btn = QPushButton("Show GT layer")
        gt_btn.setFixedHeight(32); gt_btn.setStyleSheet(BTN_SECONDARY)
        gt_btn.clicked.connect(self._show_ground_truth)
        gt_row.addWidget(gt_btn)
        self._eval_btn = QPushButton("Evaluate vs GT")
        self._eval_btn.setFixedHeight(32); self._eval_btn.setStyleSheet(BTN_SUCCESS)
        self._eval_btn.setToolTip(
            "Instance-level accuracy of the prediction against ground truth: "
            "F1 and Average Precision (AP) at IoU 0.5 / 0.75 / 0.9.")
        self._eval_btn.clicked.connect(self._evaluate_gt)
        gt_row.addWidget(self._eval_btn)
        _gt_card.addLayout(gt_row)

        self._eval_summary = QLabel("")
        self._eval_summary.setStyleSheet(
            f"color: {LABEL}; font-size: 10px; padding-top: 4px;")
        self._eval_summary.setWordWrap(True)
        _gt_card.addWidget(self._eval_summary)

        self._eval_table = QTableWidget(0, 2)
        self._eval_table.setHorizontalHeaderLabels(["Metric", "Score"])
        self._eval_table.verticalHeader().setVisible(False)
        self._eval_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._eval_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._eval_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._eval_table.setFixedHeight(150)
        self._eval_table.setStyleSheet(_EVAL_TABLE_SS)
        _hh = self._eval_table.horizontalHeader()
        _hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        _hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._eval_table.setVisible(False)
        _gt_card.addWidget(self._eval_table)

        self._eval_hint = QLabel(
            "F1 / AP: 1.0 = perfect match, higher is better. AP@IoU0.9 is the "
            "strictest (needs near-exact overlap), so it is normally the lowest.")
        self._eval_hint.setStyleSheet(f"color: {DIM}; font-size: 10px; padding-top: 2px;")
        self._eval_hint.setWordWrap(True)
        self._eval_hint.setVisible(False)
        _gt_card.addWidget(self._eval_hint)
        L.addWidget(_gt_card)

        # ── Batch prediction (collapsed — for processing multiple images) ──────
        _batch_card = CollapsibleCard("Batch prediction", collapsed=True)
        _batch_card.addWidget(_field_label("Input folder"))
        self.batch_in = QLineEdit()
        self.batch_in.setPlaceholderText("Folder with images to process")
        _batch_card.addLayout(_dir_row(self, self.batch_in, "Select input folder"))
        _batch_card.addWidget(_field_label("Output folder"))
        self.batch_out = QLineEdit(str(STORAGE_DIR / "predict_masks"))
        _batch_card.addLayout(_dir_row(self, self.batch_out, "Select output folder"))
        _br = QHBoxLayout(); _br.setSpacing(8)
        self.batch_btn = QPushButton("Run Batch")
        self.batch_btn.setFixedHeight(36)
        self.batch_btn.setStyleSheet(BTN_SUCCESS)
        self.batch_btn.clicked.connect(self._run_batch)
        _br.addWidget(self.batch_btn)
        self.batch_stop_btn = QPushButton("Stop")
        self.batch_stop_btn.setFixedHeight(36)
        self.batch_stop_btn.setFixedWidth(70)
        self.batch_stop_btn.setEnabled(False)
        self.batch_stop_btn.setStyleSheet(BTN_SECONDARY)
        self.batch_stop_btn.clicked.connect(self._stop_batch)
        _br.addWidget(self.batch_stop_btn)
        _batch_card.addLayout(_br)
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setFixedHeight(3)
        self.batch_progress.setVisible(False)
        _batch_card.addWidget(self.batch_progress)
        self.batch_lbl = QLabel("")
        self.batch_lbl.setStyleSheet(f"color: {LABEL}; font-size: 11px; padding-top: 2px;")
        _batch_card.addWidget(self.batch_lbl)
        L.addWidget(_batch_card)

        # ── Benchmark engines vs GT (collapsed — validation over a folder) ─────
        _bench_card = CollapsibleCard("Benchmark engines vs GT", collapsed=True)
        _bench_card.addWidget(_field_label("Images folder"))
        self.bench_img = QLineEdit()
        self.bench_img.setPlaceholderText("Folder of images with ground-truth masks")
        _bench_card.addLayout(_dir_row(self, self.bench_img, "Select images folder"))
        _bench_card.addWidget(_field_label("Ground-truth folder"))
        self.bench_gt = QLineEdit()
        self.bench_gt.setPlaceholderText("Folder of label masks (matched by file name)")
        _bench_card.addLayout(_dir_row(self, self.bench_gt, "Select GT folder"))

        self.bench_cellseg1 = QCheckBox("CellSeg1 · LoRA (current checkpoint)")
        self.bench_cellseg1.setChecked(True)
        self.bench_cellseg1.setStyleSheet(f"color: {LABEL}; font-size: 11px;")
        self.bench_cellpose = QCheckBox("Cellpose-SAM (zero-shot)")
        self.bench_cellpose.setChecked(True)
        self.bench_cellpose.setStyleSheet(f"color: {LABEL}; font-size: 11px;")
        _bench_card.addWidget(self.bench_cellseg1)
        _bench_card.addWidget(self.bench_cellpose)

        self.bench_btn = QPushButton("Run benchmark")
        self.bench_btn.setFixedHeight(34); self.bench_btn.setStyleSheet(BTN_SUCCESS)
        self.bench_btn.clicked.connect(self._run_benchmark)
        _bench_card.addWidget(self.bench_btn)
        self.bench_progress = QLabel("")
        self.bench_progress.setStyleSheet(
            f"color: {LABEL}; font-size: 10px; font-family:'Menlo','SF Mono',monospace;")
        self.bench_progress.setWordWrap(True)
        _bench_card.addWidget(self.bench_progress)

        self.bench_table = QTableWidget(0, 0)
        self.bench_table.verticalHeader().setVisible(False)
        self.bench_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.bench_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.bench_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.bench_table.setFixedHeight(120)
        self.bench_table.setStyleSheet(_EVAL_TABLE_SS)
        self.bench_table.setVisible(False)
        _bench_card.addWidget(self.bench_table)
        L.addWidget(_bench_card)

        # ── Model settings (collapsed — set once, rarely changed) ──────────────
        _model_card = CollapsibleCard("Model settings", collapsed=True)
        _model_card.addWidget(_field_label("Custom checkpoint (.pth)"))
        _model_card.addLayout(_make_custom_lora_row(self))
        self.vit_name = QComboBox()
        self.vit_name.addItems(["vit_h", "vit_l", "vit_b"])
        self.vit_name.currentTextChanged.connect(self._on_vit_changed)
        _model_card.addLayout(_param_row("SAM type", self.vit_name,
            "vit_h = highest quality, vit_b = fastest"))
        self.sam_path = QLineEdit()
        self.sam_path.setPlaceholderText("auto-detected from SAM type above")
        self._on_vit_changed("vit_h")
        _model_card.addLayout(_file_row(self, self.sam_path, "Select SAM backbone", "PyTorch (*.pth)"))
        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(1, 64); self.lora_rank.setValue(4)
        self.lora_rank.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        _model_card.addLayout(_param_row("LoRA rank", self.lora_rank,
            "Must match the rank used during training (default 4)"))
        self.device = QComboBox(); self._populate_devices()
        _model_card.addLayout(_param_row("Device", self.device))
        self._model_card = _model_card
        L.addWidget(_model_card)

        # ── Inference parameters (collapsed — tune when needed) ────────────────
        _inf_card = CollapsibleCard("Inference parameters", collapsed=True)
        self.resize_size = QComboBox()
        for v in ["256", "512", "768", "1024"]:
            self.resize_size.addItem(v)
        self.resize_size.setCurrentText("512")
        _inf_card.addLayout(_param_row("Resize", self.resize_size,
            "SAM inference resolution. Higher = better accuracy, slower. Try 1024 for small cells."))
        params = [
            ("Points/side",     "points_per_side",       4,   128, 32,  0, 4,
             "Grid density for mask proposals. Higher = more candidates, slower."),
            ("IoU threshold",   "pred_iou_thresh",       0.0, 1.0, 0.80, 2, 0.05,
             "Minimum predicted IoU to keep a mask. Raise to reduce false positives."),
            ("Stability score", "stability_score_thresh", 0.0, 1.0, 0.60, 2, 0.05,
             "Mask stability threshold. Raise to keep only confident masks."),
            ("Box NMS thresh",  "box_nms_thresh",        0.0, 1.0, 0.05, 3, 0.01,
             "Non-max suppression overlap. Lower separates touching cells better."),
            ("Min mask area",   "min_mask_area",         0, 10000, 20,  0, 10,
             "Discard masks smaller than this (pixels)."),
        ]
        for label, attr, lo, hi, val, dec, step, tip in params:
            if dec == 0:
                w = QSpinBox(); w.setRange(int(lo), int(hi)); w.setValue(int(val)); w.setSingleStep(int(step))
            else:
                w = QDoubleSpinBox(); w.setDecimals(dec); w.setRange(lo, hi); w.setValue(val); w.setSingleStep(step)
            w.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            setattr(self, attr, w)
            _inf_card.addLayout(_param_row(label, w, tip))

        self.clahe = QCheckBox("Enhance contrast (CLAHE) before segmenting")
        self.clahe.setToolTip(
            "Adaptive histogram equalisation. Helps recover faint or low-contrast "
            "cells; leave off for already well-exposed images.")
        self.clahe.setStyleSheet(f"color: {LABEL}; font-size: 11px;")
        _inf_card.addWidget(self.clahe)

        self.tiled = QCheckBox("Large image: tile at native resolution")
        self.tiled.setToolTip(
            "For whole-slide / high-content images. Instead of shrinking the "
            "whole image to the inference size (which loses small cells), run "
            "the engine on overlapping tiles at full resolution and stitch the "
            "cells back together. Off for ordinary-sized images.")
        self.tiled.setStyleSheet(f"color: {LABEL}; font-size: 11px;")
        _inf_card.addWidget(self.tiled)

        self._inf_card = _inf_card
        L.addWidget(_inf_card)

        # ── Cellpose-SAM settings (shown only for the Cellpose engine) ─────────
        self._cp_card = SectionCard("Cellpose-SAM settings")
        self.cp_diameter = QSpinBox()
        self.cp_diameter.setRange(0, 500); self.cp_diameter.setValue(0)
        self.cp_diameter.setSpecialValueText("auto")
        self.cp_diameter.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._cp_card.addLayout(_param_row("Cell diameter", self.cp_diameter,
            "Expected cell diameter in pixels. 0 = let the model estimate it."))
        self.cp_flow = QDoubleSpinBox()
        self.cp_flow.setRange(0.0, 3.0); self.cp_flow.setDecimals(2)
        self.cp_flow.setSingleStep(0.1); self.cp_flow.setValue(0.4)
        self.cp_flow.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._cp_card.addLayout(_param_row("Flow threshold", self.cp_flow,
            "Max allowed flow error per mask. Lower = fewer, cleaner cells."))
        self.cp_cellprob = QDoubleSpinBox()
        self.cp_cellprob.setRange(-6.0, 6.0); self.cp_cellprob.setDecimals(2)
        self.cp_cellprob.setSingleStep(0.5); self.cp_cellprob.setValue(0.0)
        self.cp_cellprob.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._cp_card.addLayout(_param_row("Cell prob. thresh", self.cp_cellprob,
            "Lower to detect more (dimmer) cells; raise to keep only confident ones."))
        self._cp_card.setVisible(False)
        L.addWidget(self._cp_card)

        L.addStretch()
        inner.setLayout(L)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # ── Log footer ────────────────────────────────────────────────────────
        outer.addWidget(_divider())
        _footer = QHBoxLayout()
        _footer.setContentsMargins(16, 4, 16, 6)
        _log_btn = QPushButton("Log ↗")
        _log_btn.setStyleSheet(
            f"color: {DIM}; background: transparent; border: none; font-size: 11px;")
        _log_btn.setToolTip("Open the floating log window")
        _log_btn.clicked.connect(lambda: get_log_window().show_and_raise())
        _footer.addStretch()
        _footer.addWidget(_log_btn)
        outer.addLayout(_footer)

        self.setLayout(outer)
        self.setMinimumWidth(360)

        self._log_signal.connect(self._append_log)
        self._done_signal.connect(self._show_results)
        self._finish_signal.connect(self._on_done)
        self._batch_progress_signal.connect(self._on_batch_progress)
        self._tile_progress_signal.connect(self._on_tile_progress)
        self._batch_finish_signal.connect(self._on_batch_done)
        self._refine_finish_signal.connect(self._on_refine_done)
        self._sample_finish_signal.connect(self._on_samples_done)
        self._cohort_ready_signal.connect(self._on_cohort_ready)
        self._benchmark_row_signal.connect(self._on_benchmark_row)
        self._benchmark_done_signal.connect(self._on_benchmark_done)
        self._dataset_progress_signal.connect(lambda s: self._dataset_lbl.setText(s))
        self._dataset_done_signal.connect(self._on_dataset_done)

        viewer.bind_key('Control-r',       lambda v: self._run_prediction())
        viewer.bind_key('Control-Shift-r', lambda v: self._predict_active_layer())

        # Populate the quality selector now that every parameter widget exists.
        self.quality.addItems(["Fast", "Balanced", "Accurate", "Custom"])
        self.quality.setCurrentText("Balanced")
        self.quality.currentTextChanged.connect(self._apply_quality)

        self.engine.currentIndexChanged.connect(self._on_engine_changed)
        self._on_engine_changed()

        self.image_path.textChanged.connect(lambda _t: self._autofill_gt())
        self._autofill_gt()
        self._on_gt_path_changed()
        self._populate_samples()

        self._autofill_from_sidecar()

    # ── Engine switching ──────────────────────────────────────────────────────

    def _current_engine(self) -> str:
        return self.engine.currentData() or "cellseg1"

    def _on_engine_changed(self, _idx=None):
        is_cp = self._current_engine() == "cellpose"
        # Checkpoint / SAM-specific controls are irrelevant for Cellpose.
        self._ckpt_card.setVisible(not is_cp)
        self._inf_card.setVisible(not is_cp)
        if hasattr(self, "_model_card"):
            self._model_card.setVisible(not is_cp)
        self.quality.setEnabled(not is_cp)
        self._cp_card.setVisible(is_cp)
        if is_cp:
            from napari_app.engines import cellpose_available
            if cellpose_available():
                self._engine_hint.setText(
                    "Zero-shot generalist — no checkpoint needed. First run downloads "
                    "the model weights (~a few hundred MB).")
            else:
                self._engine_hint.setText(
                    "⚠ Cellpose is not installed. Run:  pip install cellpose")
        else:
            self._engine_hint.setText(
                "SAM + LoRA. Pick a checkpoint below; fine-tune your own in the Train tab.")

    # ── Quality presets & samples ─────────────────────────────────────────────

    _QUALITY = {
        "Fast":     {"resize": "512",  "points_per_side": 16, "pred_iou_thresh": 0.80,
                     "stability_score_thresh": 0.62, "box_nms_thresh": 0.07},
        "Balanced": {"resize": "512",  "points_per_side": 32, "pred_iou_thresh": 0.80,
                     "stability_score_thresh": 0.60, "box_nms_thresh": 0.05},
        "Accurate": {"resize": "1024", "points_per_side": 48, "pred_iou_thresh": 0.78,
                     "stability_score_thresh": 0.58, "box_nms_thresh": 0.04},
    }

    def _apply_quality(self, name: str):
        preset = self._QUALITY.get(name)
        if not preset:
            return
        self.resize_size.setCurrentText(preset["resize"])
        self.points_per_side.setValue(preset["points_per_side"])
        self.pred_iou_thresh.setValue(preset["pred_iou_thresh"])
        self.stability_score_thresh.setValue(preset["stability_score_thresh"])
        self.box_nms_thresh.setValue(preset["box_nms_thresh"])

    def _load_samples(self):
        self._sample_btn.setEnabled(False)
        self._sample_btn.setText("Fetching samples…")

        def run():
            try:
                from napari_app.sample_data import fetch_samples
                paths = fetch_samples(TEST_IMAGE_DIR)
                self._done_signal_sample(paths)
            except Exception as e:
                self._log_signal.emit(f"[ERROR] sample data: {e}")
                self._done_signal_sample([])

        threading.Thread(target=run, daemon=True).start()

    def _done_signal_sample(self, paths):
        # Marshal back to the UI thread via a queued signal.
        self._sample_paths = paths
        self._sample_finish_signal.emit()

    def _download_dataset(self):
        if getattr(self, "_dataset_thread", None) and self._dataset_thread.is_alive():
            return
        self._dataset_lbl.setText("Starting BBBC039 download (~45 MB, one time)…")

        def run():
            try:
                from napari_app.sample_data import download_bbbc039
                paths = download_bbbc039(
                    TEST_IMAGE_DIR, limit=20,
                    progress=lambda s: self._dataset_progress_signal.emit(s))
                self._dataset_done_signal.emit(paths)
            except Exception as e:
                self._dataset_progress_signal.emit(f"[ERROR] {e}")
                self._dataset_done_signal.emit([])

        self._dataset_thread = threading.Thread(target=run, daemon=True)
        self._dataset_thread.start()

    def _on_dataset_done(self, paths):
        if not paths:
            self._append_log("[WARN] BBBC039 download produced no images")
            return
        self._dataset_lbl.setText(
            f"✓ {len(paths)} BBBC039 images + ground truth ready. Use the Sample list.")
        self._populate_samples()
        self.image_path.setText(paths[0])
        self.image_path.setToolTip(paths[0])
        self._append_log(f"✓ BBBC039: {len(paths)} images with GT → test_images/BBBC039/")

    def _on_samples_done(self):
        self._sample_btn.setEnabled(True)
        self._sample_btn.setText("⬇  Load sample microscopy images")
        paths = getattr(self, "_sample_paths", [])
        if not paths:
            self._append_log("[WARN] No samples were written")
            return
        self._populate_samples()
        self.image_path.setText(paths[0])
        self.image_path.setToolTip(paths[0])
        names = ", ".join(Path(p).name for p in paths)
        self._append_log(f"✓ Saved {len(paths)} sample(s): {names}")

    def _populate_samples(self):
        """List images in the test-images folder in the switcher combo."""
        if not hasattr(self, "sample_combo"):
            return
        exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy"}
        files = sorted(f for f in TEST_IMAGE_DIR.rglob("*")
                       if f.suffix.lower() in exts and "_gt" not in f.stem
                       and not f.stem.endswith(("_mask", "_masks", "_label", "_labels")))
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        if not files:
            self.sample_combo.addItem("— no samples yet —", "")
            self.sample_combo.setEnabled(False)
        else:
            self.sample_combo.setEnabled(True)
            cur = self.image_path.text().strip()
            for f in files:
                self.sample_combo.addItem(f.name, str(f))
            # reflect the current image if it's one of them
            idx = self.sample_combo.findData(cur)
            if idx >= 0:
                self.sample_combo.setCurrentIndex(idx)
        self.sample_combo.blockSignals(False)

    def _on_sample_selected(self, _idx: int):
        path = self.sample_combo.currentData()
        if path:
            self.image_path.setText(path)
            self.image_path.setToolTip(path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _append_log(self, text: str):
        lw = get_log_window()
        lw.append(text)
        if "[ERROR]" in text:
            lw.show_and_raise()
        elif not lw.isVisible():
            lw.show()

    def _on_checkpoint_changed(self, _index: int):
        from napari_app.inference_cache import invalidate_model
        invalidate_model()
        self._autofill_from_sidecar()

    def _autofill_from_sidecar(self):
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

        fl     = meta.get("final_loss")
        epochs = meta.get("epochs_run", "?")
        emax   = meta.get("epoch_max", "?")
        saved  = (meta.get("saved_at", "") or "")[:16].replace("T", " ")
        parts  = []
        if saved:
            parts.append(saved)
        if fl is not None:
            parts.append(f"loss {fl:.5f}")
        parts.append(f"{epochs}/{emax} ep")
        self._meta_lbl.setText("  ·  ".join(parts))

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
            self.sam_path.clear()
            self.sam_path.setPlaceholderText(f"Not found: {c.name}")

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
        raise ValueError(f"SAM backbone not found. Place {names[vit]} in {STORAGE_DIR / 'sam_backbone'}/")

    def _build_config(self):
        img = self.image_path.text().strip()
        if not img or not Path(img).exists():
            raise ValueError(f"Image not found: {img}")
        rs = int(self.resize_size.currentText())

        # Cellpose-SAM needs no LoRA/SAM checkpoint — a much shorter config.
        if self._current_engine() == "cellpose":
            from napari_app.engines import cellpose_available
            if not cellpose_available():
                raise ValueError("Cellpose is not installed — run: pip install cellpose")
            return {
                "engine": "cellpose", "image_path": img,
                "resize_size": [rs, rs],
                "cp_diameter": self.cp_diameter.value(),
                "cp_flow_threshold": self.cp_flow.value(),
                "cp_cellprob_threshold": self.cp_cellprob.value(),
                "selected_device": self.device.currentText(),
                "clahe": self.clahe.isChecked(),
                "tiled": self.tiled.isChecked(),
                "tile_size": rs, "tile_overlap": 0,
                # kept so downstream (refine, caching keys) stays valid
                "vit_name": self.vit_name.currentText(),
                "image_encoder_lora_rank": self.lora_rank.value(),
                "sam_image_size": rs, "result_pth_path": "",
            }

        return self._sam_config()

    def _sam_config(self) -> dict:
        """Full SAM + LoRA config. Used by the CellSeg1 engine and always by the
        interactive Annotate session (which needs SAM regardless of the engine
        selector). Requires an image, a LoRA checkpoint and a SAM backbone."""
        img = self.image_path.text().strip()
        if not img or not Path(img).exists():
            raise ValueError(f"Image not found: {img}")
        lora = self._resolve_lora()
        sam  = self._resolve_sam()
        if not lora or not Path(lora).exists():
            raise ValueError(f"LoRA checkpoint not found: {lora}")
        rs = int(self.resize_size.currentText())
        return {
            "engine": "cellseg1",
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
            "points_per_side":          self.points_per_side.value(),
            "points_per_batch":         64,
            "pred_iou_thresh":          self.pred_iou_thresh.value(),
            "stability_score_thresh":   self.stability_score_thresh.value(),
            "stability_score_offset":   0.8,
            "box_nms_thresh":           self.box_nms_thresh.value(),
            "crop_nms_thresh": 0.05, "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area":     self.min_mask_area.value(),
            "max_mask_region_area_ratio": 0.1,
            "selected_device": self.device.currentText(),
            "deterministic": True, "seed": 0,
            "allow_tf32_on_cudnn": True, "allow_tf32_on_matmul": True,
            "clahe": self.clahe.isChecked(),
            "tiled": self.tiled.isChecked(),
            "tile_size": rs, "tile_overlap": 0,
        }

    # ── Parameter access for the Assistant agent ──────────────────────────────

    def current_params(self) -> dict:
        """Live inference parameters, keyed to match advisor recommendations."""
        return {
            "points_per_side": self.points_per_side.value(),
            "pred_iou_thresh": self.pred_iou_thresh.value(),
            "stability_score_thresh": self.stability_score_thresh.value(),
            "box_nms_thresh": self.box_nms_thresh.value(),
            "min_mask_area": self.min_mask_area.value(),
            "resize_size": int(self.resize_size.currentText()),
            "lora_rank": self.lora_rank.value(),
        }

    def apply_params(self, changes: dict) -> list[str]:
        """Apply advisor changes to the controls. Returns human-readable diffs."""
        applied = []
        for key, val in changes.items():
            if key == "resize_size":
                self.resize_size.setCurrentText(str(int(val)))
            elif key == "clahe":
                self.clahe.setChecked(bool(val))
            elif key in ("points_per_side", "min_mask_area", "lora_rank"):
                getattr(self, key).setValue(int(val))
            elif hasattr(self, key) and hasattr(getattr(self, key), "setValue"):
                getattr(self, key).setValue(float(val))
            else:
                continue
            applied.append(f"{key} → {val}")
        return applied

    def rerun(self):
        """Re-run prediction on the current image (used by the Assistant)."""
        self._run_prediction()

    def last_context(self):
        """(image_rgb, mask) of the most recent prediction, or (None, None)."""
        return self._last_img_rgb, self._last_mask

    # ── Actions ───────────────────────────────────────────────────────────────

    def _predict_active_layer(self):
        import napari.layers as nl
        import tempfile, cv2

        image_layer = None
        for layer in self.viewer.layers:
            if isinstance(layer, nl.Image):
                image_layer = layer; break
        if image_layer is None:
            self._append_log("[ERROR] No Image layer found in viewer"); return

        img = np.asarray(image_layer.data, dtype=np.float32)
        img = (img / img.max() * 255).clip(0, 255).astype(np.uint8) if img.max() > 1.0 \
              else (img * 255).clip(0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

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

        is_cp = config.get("engine") == "cellpose"
        if is_cp:
            self._append_log(f"▶ {Path(config['image_path']).name}  [Cellpose-SAM · {config['selected_device']}]")
        else:
            from napari_app.inference_cache import cache_status
            self._append_log(f"▶ {Path(config['image_path']).name}  [{cache_status()}]")

        def run():
            try:
                img_arr, mask = _predict_cached(
                    config, on_tile=self._tile_progress_signal.emit)
                self._done_signal.emit(img_arr, mask)
                if is_cp:
                    self._log_signal.emit(f"✓ {int(mask.max())} cells  [Cellpose-SAM]")
                else:
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
        # Reset to the indeterminate spinner so the next (non-tiled) run looks
        # unchanged; a tiled run flips it back to determinate via _on_tile_progress.
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("")
        self.progress_bar.setTextVisible(False)
        self._populate_lora_combo()

    def _show_results(self, img_arr, mask):
        name = Path(self.image_path.text()).stem
        self._last_mask     = mask
        self._last_img_rgb  = img_arr
        self._last_img_path = self.image_path.text()

        for lyr in list(self.viewer.layers):
            if lyr.name.startswith(name) and "_gt" not in lyr.name:
                self.viewer.layers.remove(lyr)

        self.viewer.add_image(img_arr, name=f"{name}_image")
        if mask is not None and mask.max() > 0:
            lyr = self.viewer.add_labels(mask.astype(np.int32), name=f"{name}_masks", opacity=0.7)
            lyr.contour = 1
        self._recompute_measurements()
        self.viewer.reset_view()

    def _on_pixel_size_changed(self, _value=None):
        if self._last_mask is not None:
            self._recompute_measurements()

    def _recompute_measurements(self):
        """Measure the current mask and refresh the results panel + window."""
        mask = self._last_mask
        if mask is None or int(mask.max()) == 0:
            self._results_card.setVisible(mask is not None)
            if mask is not None:
                self._cell_count_lbl.setText("0")
                for chip in (self._chip_diam, self._chip_area, self._chip_cov):
                    chip.setText("—")
                self._last_measure = None
            return
        from napari_app import analysis
        try:
            result = analysis.compute_measurements(
                mask, intensity_image=self._last_img_rgb,
                pixel_size_um=self.pixel_size.value())
        except Exception as e:
            self._append_log(f"[WARN] measurement failed: {e}")
            result = None
        self._last_measure = result

        n = int(mask.max())
        self._cell_count_lbl.setText(str(n))
        if result:
            cov = float((mask > 0).sum()) / mask.size * 100.0
            s = result["summary"]
            au = next((u for k, _l, u in result["columns"] if k == "area"), "px²")
            lu = next((u for k, _l, u in result["columns"] if k == "diameter"), "px")
            self._chip_diam.setText(f"{s.get('diameter', {}).get('median', 0):.1f} {lu}")
            self._chip_area.setText(f"{s.get('area', {}).get('mean', 0):.0f} {au}")
            self._chip_cov.setText(f"{cov:.1f}%")
        self._results_card.setVisible(True)

        from napari_app.widgets.measurements_window import get_measurements_window
        mw = get_measurements_window()
        if mw.isVisible() and result:
            mw.set_result(result, Path(self._last_img_path).name if self._last_img_path else "")

    def _open_measurements(self):
        if self._last_measure is None:
            self._append_log("[WARN] Run a prediction with detected cells first"); return
        from napari_app.widgets.measurements_window import get_measurements_window
        mw = get_measurements_window()
        mw.set_result(self._last_measure,
                      Path(self._last_img_path).name if self._last_img_path else "")
        mw.show_and_raise()

    # ── Ground truth: auto-detect + evaluate ──────────────────────────────────

    def _gt_sidecar(self, image_path: str):
        """Find a ground-truth mask that ships next to ``image_path``, if any."""
        if not image_path:
            return None
        p = Path(image_path)
        stem, exts = p.stem, (".png", ".tif", ".tiff", ".npy")
        candidates = []
        for suffix in ("_gt", "_masks", "_mask", "_label", "_labels"):
            for e in exts:
                candidates.append(p.with_name(stem + suffix + e))
        # sibling masks/ folder with a same-named file
        for folder in ("masks", "gt", "labels"):
            for e in exts:
                candidates.append(p.parent / folder / (stem + e))
                candidates.append(p.parent.parent / folder / (stem + e))
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def _autofill_gt(self):
        """Keep the GT field in sync with the current image.

        Sets it to a detected sidecar mask, or clears it when the new image has
        none — so switching images never leaves a stale ground truth behind.
        """
        found = self._gt_sidecar(self.image_path.text().strip())
        current = self.gt_path.text().strip()
        if found:
            if current != found:
                self.gt_path.setText(found)
        elif current and self._was_autofilled:
            # only clear values we set ourselves, not a manual pick
            self.gt_path.clear()
        self._was_autofilled = bool(found)

    def _on_gt_path_changed(self):
        p = self.gt_path.text().strip()
        if p and Path(p).exists():
            self._gt_status.setText(f"✓ ground truth: {Path(p).name}")
            self._eval_btn.setEnabled(True)
        else:
            self._gt_status.setText("No ground truth set." if not p else "GT file not found.")
            self._eval_btn.setEnabled(bool(p) and Path(p).exists())

    def _load_gt_mask(self):
        import cv2
        p = self.gt_path.text().strip()
        if not p or not Path(p).exists():
            raise ValueError("GT file not found")
        gt = np.load(p) if p.lower().endswith(".npy") else cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if gt is None:
            raise ValueError(f"Cannot read GT: {p}")
        return np.ascontiguousarray(gt).astype(np.int32)

    def _evaluate_gt(self):
        if self._last_mask is None:
            self._append_log("[ERROR] Run a prediction first"); return
        try:
            gt = self._load_gt_mask()
        except ValueError as e:
            self._append_log(f"[ERROR] {e}"); return

        pred = self._last_mask.astype(np.int32)
        if gt.shape != pred.shape:
            import cv2
            gt = cv2.resize(gt.astype(np.float32), (pred.shape[1], pred.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
        try:
            from metrics import average_precision
            # Call one threshold at a time — average_precision's internal guard
            # is not vectorised across multiple thresholds for a single image.
            ap, f1, tp0, fp0, fn0 = {}, 0.0, 0, 0, 0
            for th in (0.5, 0.75, 0.9):
                a, tp, fp, fn = average_precision(gt, pred, threshold=[th])
                ap[th] = float(np.atleast_1d(a)[0])
                tp = float(np.atleast_1d(tp)[0]); fp = float(np.atleast_1d(fp)[0])
                fn = float(np.atleast_1d(fn)[0])
                if th == 0.5:
                    denom = 2 * tp + fp + fn
                    f1 = (2 * tp / denom) if denom else 0.0
                    tp0, fp0, fn0 = int(tp), int(fp), int(fn)

            rows = [
                ("F1 @ IoU 0.5",  f1),
                ("AP @ IoU 0.5",  ap[0.5]),
                ("AP @ IoU 0.75", ap[0.75]),
                ("AP @ IoU 0.9",  ap[0.9]),
            ]
            self._fill_eval_table(rows)
            self._eval_summary.setText(
                f"GT {int(gt.max())} cells vs prediction {int(pred.max())} cells   ·   "
                f"at IoU 0.5:  TP {tp0} · FP {fp0} · FN {fn0}")
            self._append_log(
                f"✓ Eval — F1@0.5 {f1:.3f}, AP@0.5 {ap[0.5]:.3f}, "
                f"AP@0.75 {ap[0.75]:.3f}, AP@0.9 {ap[0.9]:.3f}")
        except Exception as e:
            import traceback
            self._append_log(f"[ERROR] eval: {e}\n{traceback.format_exc()}")

    def _fill_eval_table(self, rows):
        from napari_app.theme import SUCCESS, DANGER
        self._eval_table.setRowCount(len(rows))
        for r, (name, val) in enumerate(rows):
            m = QTableWidgetItem(name)
            m.setFlags(Qt.ItemFlag.ItemIsEnabled)
            v = QTableWidgetItem(f"{val:.3f}")
            v.setFlags(Qt.ItemFlag.ItemIsEnabled)
            v.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # colour the score: green ≥0.8, amber ≥0.5, red below
            col = SUCCESS if val >= 0.8 else ("#d6a54a" if val >= 0.5 else DANGER)
            v.setForeground(QColor(col))
            self._eval_table.setItem(r, 0, m)
            self._eval_table.setItem(r, 1, v)
        self._eval_table.setVisible(True)
        self._eval_hint.setVisible(True)

    def _show_ground_truth(self):
        p = self.gt_path.text().strip()
        if not p or not Path(p).exists():
            self._append_log("[ERROR] GT file not found"); return
        try:
            import cv2
            gt = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if gt is None:
                gt = np.load(p)
            gt = gt.astype(np.int32)
            name  = Path(self.image_path.text()).stem
            lname = f"{name}_gt"
            for lyr in list(self.viewer.layers):
                if lyr.name == lname:
                    self.viewer.layers.remove(lyr)
            l = self.viewer.add_labels(gt, name=lname, opacity=0.9)
            l.contour = 1
            _color_labels_solid(l, gt, (0.0, 1.0, 0.35, 1.0))  # uniform green
            self._append_log(f"✓ GT loaded — {int(gt.max())} cells (green outline)")
        except Exception as e:
            self._append_log(f"[ERROR] {e}")

    def _save_masks(self):
        if self._last_mask is None: return
        stem    = Path(self._last_img_path).stem if self._last_img_path else "mask"
        default = str(STORAGE_DIR / "predict_masks" / f"{stem}_mask.png")
        p, _ = QFileDialog.getSaveFileName(
            self, "Save mask", default, "PNG (*.png);;TIFF (*.tif)", options=_DLG)
        if not p: return
        import cv2
        cv2.imwrite(p, self._last_mask.astype(np.uint16))
        self._write_manifest(p)
        self._append_log(f"✓ Saved {Path(p).name}  (+ provenance .json)")

    def _write_manifest(self, mask_path: str):
        """Write a reproducibility manifest next to a saved mask.

        Records engine, parameters, versions and an input hash so a result can
        be traced back to exactly how it was produced — a basic requirement for
        publishable analysis.
        """
        import json, hashlib, platform
        from datetime import datetime
        try:
            engine = self._current_engine()
            manifest = {
                "app": "CellSeg1 napari",
                "created": datetime.now().isoformat(timespec="seconds"),
                "engine": engine,
                "image": self._last_img_path,
                "n_cells": int(self._last_mask.max()) if self._last_mask is not None else 0,
                "pixel_size_um": self.pixel_size.value(),
                "clahe": self.clahe.isChecked(),
                "resize": int(self.resize_size.currentText()),
                "device": self.device.currentText(),
            }
            if engine == "cellpose":
                manifest["cellpose"] = {
                    "diameter": self.cp_diameter.value(),
                    "flow_threshold": self.cp_flow.value(),
                    "cellprob_threshold": self.cp_cellprob.value(),
                }
            else:
                manifest["checkpoint"] = self._resolve_lora()
                manifest["params"] = self.current_params()
            try:
                import torch
                manifest["versions"] = {"python": platform.python_version(),
                                        "torch": torch.__version__}
            except Exception:
                manifest["versions"] = {"python": platform.python_version()}
            if self._last_img_path and Path(self._last_img_path).exists():
                h = hashlib.sha256(Path(self._last_img_path).read_bytes()).hexdigest()[:16]
                manifest["image_sha256_16"] = h
            with open(Path(mask_path).with_suffix(".json"), "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            self._append_log(f"[WARN] manifest not written: {e}")

    def _run_batch(self):
        in_dir  = Path(self.batch_in.text().strip())
        out_dir = Path(self.batch_out.text().strip())
        if not in_dir.is_dir():
            self._append_log("[ERROR] Input folder not found"); return
        exts   = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
        images = sorted(f for f in in_dir.iterdir() if f.suffix.lower() in exts)
        if not images:
            self._append_log("[ERROR] No images found in input folder"); return
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            config = self._build_config()
        except ValueError as e:
            self._append_log(f"[ERROR] {e}"); return

        self._batch_stop.clear()
        self.batch_btn.setEnabled(False)
        self.batch_stop_btn.setEnabled(True)
        self.batch_progress.setVisible(True)
        self.batch_progress.setValue(0)
        self.batch_lbl.setText(f"0 / {len(images)}")
        self._append_log(f"▶ Batch: {len(images)} images → {out_dir.name}/")

        import cv2 as _cv2
        px = self.pixel_size.value()
        self._cohort_records = []
        self._cohort_out = out_dir

        def run():
            from napari_app import analysis
            n = len(images)
            done = 0
            records = []
            for img_path in images:
                if self._batch_stop.is_set():
                    self._log_signal.emit(f"■ Stopped at {done}/{n}"); break
                self._log_signal.emit(f"[{done + 1}/{n}] {img_path.name}")
                try:
                    cfg = {**config, "image_path": str(img_path)}
                    img_arr, mask = _predict_cached(cfg)
                    _cv2.imwrite(str(out_dir / f"{img_path.stem}_mask.png"),
                                 mask.astype(np.uint16))
                    result = analysis.compute_measurements(
                        mask, intensity_image=img_arr, pixel_size_um=px)
                    cov = float((mask > 0).sum()) / mask.size * 100.0
                    records.append((img_path.name, result, cov))
                except Exception as e:
                    self._log_signal.emit(f"  [ERROR] {e}")
                done += 1
                self._batch_progress_signal.emit(done, n)
            else:
                # completed without a break
                try:
                    from napari_app import cohort
                    cell_csv, summ_csv = cohort.write_cohort_csvs(out_dir, records)
                    pop = cohort.population_stats(records)
                    self._cohort_records = records
                    self._log_signal.emit(
                        f"✓ Batch done — {n} masks + cohort CSVs in {out_dir.name}/  "
                        f"({pop['total_cells']} cells across {pop['n_images']} images)")
                except Exception as e:
                    self._log_signal.emit(f"  [WARN] cohort analysis failed: {e}")
                self._cohort_ready_signal.emit()
            self._batch_finish_signal.emit()

        threading.Thread(target=run, daemon=True).start()

    def _on_cohort_ready(self):
        records = getattr(self, "_cohort_records", [])
        if not records:
            return
        from napari_app.widgets.cohort_window import get_cohort_window
        w = get_cohort_window()
        w.set_records(records, str(getattr(self, "_cohort_out", "")))
        w.show_and_raise()

    # ── Benchmark engines vs ground truth ─────────────────────────────────────

    _ENGINE_LABELS = {"cellseg1": "CellSeg1 (LoRA)", "cellpose": "Cellpose-SAM"}

    def _run_benchmark(self):
        img_dir = Path(self.bench_img.text().strip())
        gt_dir  = Path(self.bench_gt.text().strip())
        if not img_dir.is_dir():
            self._append_log("[ERROR] Benchmark: images folder not found"); return
        if not gt_dir.is_dir():
            self._append_log("[ERROR] Benchmark: GT folder not found"); return
        engines = []
        if self.bench_cellseg1.isChecked(): engines.append("cellseg1")
        if self.bench_cellpose.isChecked(): engines.append("cellpose")
        if not engines:
            self._append_log("[ERROR] Benchmark: select at least one engine"); return

        rs = int(self.resize_size.currentText())
        bases = {}
        try:
            if "cellseg1" in engines:
                bases["cellseg1"] = self._sam_config()
            if "cellpose" in engines:
                from napari_app.engines import cellpose_available
                if not cellpose_available():
                    raise ValueError("Cellpose is not installed")
                bases["cellpose"] = {
                    "engine": "cellpose", "resize_size": [rs, rs], "image_path": "",
                    "cp_diameter": self.cp_diameter.value(),
                    "cp_flow_threshold": self.cp_flow.value(),
                    "cp_cellprob_threshold": self.cp_cellprob.value(),
                    "selected_device": self.device.currentText(),
                    "clahe": self.clahe.isChecked(),
                }
        except ValueError as e:
            self._append_log(f"[ERROR] Benchmark: {e}"); return

        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
        images = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in exts)

        def find_gt(stem):
            for e in (".png", ".tif", ".tiff", ".npy"):
                for suf in ("", "_masks", "_mask", "_gt", "_label", "_labels"):
                    c = gt_dir / f"{stem}{suf}{e}"
                    if c.exists():
                        return c
            return None

        pairs = [(im, find_gt(im.stem)) for im in images]
        pairs = [(i, g) for i, g in pairs if g is not None]
        if not pairs:
            self._append_log("[ERROR] Benchmark: no images with matching GT masks"); return

        self.bench_btn.setEnabled(False)
        self.bench_table.setVisible(False)
        total = len(pairs) * len(engines)
        self._append_log(
            f"▶ Benchmark: {len(engines)} engine(s) × {len(pairs)} images = {total} runs")

        def run():
            from napari_app import benchmark
            import cv2
            per_engine = {e: [] for e in engines}
            done = 0
            for eng in engines:
                for img_path, gt_path in pairs:
                    try:
                        cfg = {**bases[eng], "image_path": str(img_path)}
                        _, pred = _predict_cached(cfg)
                        if str(gt_path).lower().endswith(".npy"):
                            gt = np.load(str(gt_path))
                        else:
                            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
                        gt = np.ascontiguousarray(gt).astype(np.int32)
                        if gt.shape != pred.shape:
                            gt = cv2.resize(gt.astype(np.float32),
                                            (pred.shape[1], pred.shape[0]),
                                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
                        per_engine[eng].append(benchmark.evaluate(gt, pred))
                    except Exception as ex:
                        self._log_signal.emit(f"  [ERROR] {eng} {img_path.name}: {ex}")
                    done += 1
                    self._benchmark_row_signal.emit(f"{done} / {total}  ({self._ENGINE_LABELS[eng]})")
            summaries = {self._ENGINE_LABELS[e]: benchmark.summarize(per_engine[e])
                         for e in engines}
            cols, rows = benchmark.results_table(summaries)
            try:
                benchmark.write_csv(str(img_dir / "benchmark.csv"), cols, rows)
            except Exception:
                pass
            self._bench_cols, self._bench_rows = cols, rows
            self._benchmark_done_signal.emit()

        threading.Thread(target=run, daemon=True).start()

    def _on_benchmark_row(self, text: str):
        self.bench_progress.setText(text)

    def _on_benchmark_done(self):
        self.bench_btn.setEnabled(True)
        cols = getattr(self, "_bench_cols", [])
        rows = getattr(self, "_bench_rows", [])
        if not rows:
            self.bench_progress.setText("no results"); return
        # best mAP wins
        map_idx = cols.index("mAP") if "mAP" in cols else len(cols) - 1
        best_row = max(range(len(rows)), key=lambda r: rows[r][map_idx])
        self.bench_table.setColumnCount(len(cols))
        self.bench_table.setRowCount(len(rows))
        self.bench_table.setHorizontalHeaderLabels(cols)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if r == best_row and c == map_idx:
                    from napari_app.theme import SUCCESS
                    item.setForeground(QColor(SUCCESS))
                self.bench_table.setItem(r, c, item)
        self.bench_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.bench_table.setVisible(True)
        best_name = rows[best_row][0]
        self.bench_progress.setText(f"✓ done — best mAP: {best_name}")
        self._append_log(f"✓ Benchmark done — winner (mAP): {best_name}. CSV saved.")

    def _stop_batch(self):
        self._batch_stop.set()
        self.batch_stop_btn.setEnabled(False)
        self.batch_lbl.setText("stopping…")

    def _on_batch_progress(self, done: int, total: int):
        self.batch_progress.setValue(int(done / total * 100))
        self.batch_lbl.setText(f"Image {done} / {total}")

    def _on_tile_progress(self, done: int, total: int):
        # Turn the indeterminate spinner into a determinate "tile 7/48" bar so a
        # long whole-slide run reads as progressing, not hung.
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self.progress_bar.setFormat(f"tile {done}/{total}")
        self.progress_bar.setTextVisible(True)

    def _on_batch_done(self):
        self.batch_btn.setEnabled(True)
        self.batch_stop_btn.setEnabled(False)
        self.batch_progress.setVisible(False)
        self.batch_lbl.setText("")

    # ── Drag-and-drop ─────────────────────────────────────────────────────────

    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile() and \
                Path(urls[0].toLocalFile()).suffix.lower() in self._IMG_EXTS:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = event.mimeData().urls()[0].toLocalFile()
        self.image_path.setText(path)
        self._append_log(f"Dropped: {Path(path).name}")

    # ── Refine ────────────────────────────────────────────────────────────────

    def _run_refine(self):
        import napari.layers as nl, tempfile, shutil, cv2
        from napari_app.core.train_state_manager import TrainingStateManager

        labels_layer = next((l for l in self.viewer.layers if isinstance(l, nl.Labels)), None)
        if labels_layer is None:
            self._append_log("[ERROR] No Labels layer — edit the prediction in napari first"); return
        if self._last_img_path is None:
            self._append_log("[ERROR] Run prediction first"); return

        lora_path = self._resolve_lora()
        if not lora_path or not Path(lora_path).exists():
            self._append_log("[ERROR] No checkpoint selected"); return
        try:
            base_config = self._build_config()
        except ValueError as e:
            self._append_log(f"[ERROR] {e}"); return

        tmp     = tempfile.mkdtemp(prefix="cellseg_refine_")
        tmp_img = Path(tmp) / "images"; tmp_img.mkdir()
        tmp_msk = Path(tmp) / "masks";  tmp_msk.mkdir()

        img = cv2.imread(self._last_img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self._append_log(f"[ERROR] Cannot read {self._last_img_path}"); return
        cv2.imwrite(str(tmp_img / "img.png"), img)
        mask = np.asarray(labels_layer.data, dtype=np.int32)
        cv2.imwrite(str(tmp_msk / "img.png"), mask.clip(0, 65535).astype(np.uint16))

        refined_path = str(LORA_DIR / f"{Path(lora_path).stem}_refined.pth")
        rs = int(self.resize_size.currentText())
        config = {
            **base_config,
            "train_image_dir": str(tmp_img),
            "train_mask_dir":  str(tmp_msk),
            "result_pth_path": refined_path,
            "finetune_from":   lora_path,
            "epoch_max":       50,
            "train_id":        [0],
            "duplicate_data":  64,
            "patch_size":      rs // 2,
        }

        sm = TrainingStateManager(str(STORAGE_DIR / "refine_state"))
        pq = _queue.Queue()
        self._refine_lh = []
        self._refine_timer.stop()
        self._refine_timer = QTimer()
        self._refine_timer.setInterval(3000)

        def _poll():
            while not pq.empty():
                self._refine_lh.append(pq.get_nowait())
            if self._refine_lh:
                last = self._refine_lh[-1]
                self._log_signal.emit(
                    f"  refine ep {last['epoch']}/50  loss {last['loss']:.5f}")

        self._refine_timer.timeout.connect(_poll)
        self._refine_timer.start()

        n_cells = int(mask.max())
        self._append_log(f"▶ Refine: 50 ep, {n_cells} cells, from {Path(lora_path).name}")

        def run():
            from napari_app.core.train_model import train_model
            try:
                train_model(config, sm, progress_queue=pq)
                self._log_signal.emit(f"✓ Refined → {Path(refined_path).name}")
                self._refine_finish_signal.emit(refined_path)
            except Exception as e:
                import traceback
                self._log_signal.emit(f"[ERROR] refine: {e}\n{traceback.format_exc()}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        threading.Thread(target=run, daemon=True).start()

    def _on_refine_done(self, refined_path: str):
        self._refine_timer.stop()
        self._populate_lora_combo()
        stem = Path(refined_path).stem
        for i in range(self.lora_combo.count()):
            if stem in self.lora_combo.itemText(i):
                self.lora_combo.setCurrentIndex(i); break

    def _export_csv(self):
        if self._last_mask is None:
            return
        if self._last_measure is None:
            self._recompute_measurements()
        result = self._last_measure
        if not result or result["n_cells"] == 0:
            self._append_log("[WARN] No cells to export"); return

        from napari_app import analysis
        stem    = Path(self._last_img_path).stem if self._last_img_path else "mask"
        (STORAGE_DIR / "predict_masks").mkdir(parents=True, exist_ok=True)
        default = str(STORAGE_DIR / "predict_masks" / f"{stem}_measurements.csv")
        p, _ = QFileDialog.getSaveFileName(
            self, "Export measurements CSV", default, "CSV (*.csv)", options=_DLG)
        if not p:
            return
        with open(p, "w", newline="") as f:
            f.write(analysis.rows_as_csv(result))
        self._append_log(f"✓ Exported {result['n_cells']} cells × "
                         f"{len(result['columns'])} features → {Path(p).name}")


# ── Custom LoRA path ──────────────────────────────────────────────────────────

_EVAL_TABLE_SS = f"""
QTableWidget {{
    background: {CONSOLE}; color: {TEXT};
    gridline-color: {BORDER};
    border: 1px solid {BORDER}; border-radius: 6px;
    font-family: 'Menlo','SF Mono',monospace; font-size: 11px;
}}
QHeaderView::section {{
    background: {FG}; color: {LABEL};
    padding: 4px 8px; border: none; border-bottom: 1px solid {BORDER};
    font-weight: 600; font-size: 10px;
}}
QTableWidget::item {{ padding: 3px 8px; }}
"""


def _color_labels_solid(layer, mask, rgba):
    """Paint every non-zero label in a Labels layer a single solid colour.

    Used to render ground truth as one distinct (green) outline instead of a
    rainbow of per-cell colours that would blend with the prediction layer.
    """
    try:
        from napari.utils.colormaps import DirectLabelColormap
        ids = np.unique(mask)
        cmap = {int(i): rgba for i in ids if i != 0}
        cmap[None] = (0.0, 0.0, 0.0, 0.0)
        cmap[0] = (0.0, 0.0, 0.0, 0.0)
        layer.colormap = DirectLabelColormap(color_dict=cmap)
    except Exception:
        pass  # older napari — fall back to default multicolour labels


def _make_stat_chip(caption: str):
    """A compact stat tile: big value over a small uppercase caption.

    Returns (value_label, frame) — keep the label to update, add the frame.
    """
    frame = QFrame()
    frame.setObjectName("StatChip")
    frame.setStyleSheet(
        f"QFrame#StatChip {{ background: {CONSOLE}; border: 1px solid {BORDER};"
        f" border-radius: 8px; }}")
    v = QVBoxLayout(frame)
    v.setContentsMargins(11, 8, 11, 8); v.setSpacing(1)
    val = QLabel("—")
    val.setStyleSheet(
        f"color: {TEXT}; font-size: 15px; font-weight: 700;"
        f"font-family: 'Menlo','SF Mono',monospace; background: transparent;")
    cap = QLabel(caption)
    cap.setStyleSheet(
        f"color: {DIM}; font-size: 9px; font-weight: 600; letter-spacing: 0.8px;"
        f"background: transparent;")
    v.addWidget(val)
    v.addWidget(cap)
    return val, frame


def _make_custom_lora_row(parent) -> QHBoxLayout:
    parent.lora_custom = QLineEdit()
    parent.lora_custom.setPlaceholderText("Leave blank to use dropdown above")
    row = QHBoxLayout(); row.setSpacing(6)
    row.addWidget(parent.lora_custom)
    row.addWidget(_browse_btn(parent,
        lambda: _pick_file(parent, parent.lora_custom, "Select LoRA checkpoint", "PyTorch (*.pth)")))
    return row


# ── Prediction core ───────────────────────────────────────────────────────────

def _predict_cached(config, on_tile=None):
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

    # Large-image path: tile at native resolution instead of shrinking the
    # whole image (which loses small cells). Opt-in via the "Large image" box.
    from napari_app.tiling import should_tile
    if config.get("tiled") and should_tile(img.shape, tile=int(config.get("tile_size") or 1024)):
        return img, _predict_tiled(config, img, on_tile=on_tile)

    orig_h, orig_w = img.shape[:2]
    resized = resize_image(img, config["resize_size"])
    if config.get("clahe"):
        resized = _apply_clahe(resized)

    if config.get("engine") == "cellpose":
        from napari_app.engines import predict_cellpose
        small = predict_cellpose(
            resized,
            diameter=config.get("cp_diameter", 0),
            flow_threshold=config.get("cp_flow_threshold", 0.4),
            cellprob_threshold=config.get("cp_cellprob_threshold", 0.0),
            device=config.get("selected_device", "cpu"),
        )
    else:
        small = predict_cached(config, resized)

    if small.shape != (orig_h, orig_w):
        mask = cv2.resize(small.astype(np.float32), (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST).astype(small.dtype)
    else:
        mask = small
    return img, mask


def _predict_tiled(config, img, on_tile=None):
    """Segment a large RGB image tile-by-tile at native resolution and stitch.

    Reuses the exact per-image engine calls of the normal path, applied to each
    overlapping tile; cells crossing a seam are merged by the stitcher. Returns
    a full-resolution instance mask the same H×W as ``img``. ``on_tile(done,
    total)`` is forwarded to the tiler for per-tile progress reporting.
    """
    from napari_app.tiling import recommend_overlap, tiled_predict

    tile = int(config.get("tile_size") or 1024)
    overlap = int(config.get("tile_overlap") or 0)
    if overlap <= 0:
        overlap = recommend_overlap(float(config.get("cp_diameter") or 0), tile)

    if config.get("engine") == "cellpose":
        from napari_app.engines import predict_cellpose

        def _fn(t):
            if config.get("clahe"):
                t = _apply_clahe(t)
            return predict_cellpose(
                t,
                diameter=config.get("cp_diameter", 0),
                flow_threshold=config.get("cp_flow_threshold", 0.4),
                cellprob_threshold=config.get("cp_cellprob_threshold", 0.0),
                device=config.get("selected_device", "cpu"),
            )
    else:
        from napari_app.inference_cache import predict_cached

        def _fn(t):
            if config.get("clahe"):
                t = _apply_clahe(t)
            return predict_cached(config, t)

    min_area = int(config.get("min_mask_area") or config.get("min_mask_region_area") or 0)
    return tiled_predict(img, _fn, tile=tile, overlap=overlap, min_area=min_area,
                         on_tile=on_tile)


def _apply_clahe(rgb: np.ndarray) -> np.ndarray:
    """Adaptive histogram equalisation on the luminance channel (uint8 RGB)."""
    import cv2
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def _find_test_image():
    for ext in ("*.png", "*.tif", "*.tiff", "*.jpg"):
        hits = list(TEST_IMAGE_DIR.glob(ext))
        if hits:
            return hits[0]
    return None
