import queue as _queue
import threading
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QScrollArea, QProgressBar, QFrame,
    QAbstractSpinBox, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QListWidget, QListWidgetItem,
)
from napari_app.widgets.log_window import get_log_window
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor

from napari_app.core.predict_state_manager import PredictionStateManager
from napari_app.core.predict_controller import (
    PredictController, _to_display_uint8, _read_for_predict, _predict_cached,
    _predict_tiled, _apply_clahe,
)
from napari_app import engine_registry
from project_root import STORAGE_DIR
from napari_app.theme import (
    WIDGET_SS, BTN_PRIMARY, BTN_SUCCESS, BTN_SECONDARY, BTN_BROWSE,
    BG, FG, BORDER, BORDER_STRONG, TEXT, ACCENT, ACCENT_SOFT, DIM, LABEL,
    CONSOLE, INPUT, MONO, SUCCESS, CARD_HEADER,
)
from napari_app import icons, motion
from napari_app.widgets.common import (
    section_header, divider as _divider, param_row as _param_row,
    CollapsibleSection, SectionCard, CollapsibleCard,
)
from napari_app.widgets.controls import Combo

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
    _volume_done_signal    = pyqtSignal(object, object)
    _finish_signal         = pyqtSignal()
    _batch_progress_signal = pyqtSignal(int, int)
    _tile_progress_signal  = pyqtSignal(int, int)
    _slice_progress_signal = pyqtSignal(int, int)
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
        self._last_channel_stack = None
        self._last_measure  = None
        self._last_img_path = None
        self._mw_row_selected_wired = False
        self._was_autofilled = False
        self._lora_paths    = {}
        self._controller    = PredictController()
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
        # Populated from the engine registry (napari_app/engine_registry.py) so
        # a newly registered engine shows up here without touching this widget.
        engine_card = SectionCard("Engine", icon="target")
        self.engine = Combo()
        for _spec in engine_registry.all_engines():
            self.engine.addItem(_spec.label, _spec.key)
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
        self._ckpt_card = SectionCard("Checkpoint", icon="layers")
        ckpt_card = self._ckpt_card
        self.lora_combo = Combo()
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
        img_card = SectionCard("Image", icon="image")
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Path to microscopy image  (or drop here)")
        img = _find_test_image()
        if img:
            self.image_path.setText(str(img))
        img_card.addLayout(_file_row(self, self.image_path, "Select image",
            "Images (*.png *.tif *.tiff *.jpg *.bmp *.npy "
            "*.ome.tif *.ome.tiff *.nd2 *.czi *.lif)"))
        # Switcher first — the everyday action is *selecting* a loaded image.
        sw_row = QHBoxLayout(); sw_row.setSpacing(6)
        sw_lbl = QLabel("Sample")
        sw_lbl.setStyleSheet(f"color: {LABEL}; font-size: 11px; font-weight: 500;")
        sw_lbl.setFixedWidth(52)
        sw_row.addWidget(sw_lbl)
        self.sample_combo = Combo()
        self.sample_combo.setToolTip(
            "Switch between images already in your test-images folder. "
            "\"has GT\" means a ground-truth mask ships alongside it, for the "
            "GT / Evaluate workflow further down.")
        self.sample_combo.activated.connect(self._on_sample_selected)
        sw_row.addWidget(self.sample_combo, stretch=1)
        img_card.addLayout(sw_row)

        # A single "Get images" menu button — download once, then use the switcher.
        from PyQt6.QtWidgets import QMenu
        self._sample_btn = QPushButton("  Get images")
        self._sample_btn.setFixedHeight(32)
        self._sample_btn.setStyleSheet(BTN_SECONDARY)
        self._sample_btn.setIcon(icons.icon("download", LABEL, 14))
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
            f"color: {LABEL}; font-size: 10px; font-family:{MONO};")
        self._dataset_lbl.setWordWrap(True)
        img_card.addWidget(self._dataset_lbl)

        # ── Channel picker (multi-channel microscopy only) ─────────────────────
        # Hidden for ordinary RGB/grayscale images so the default path is
        # untouched; revealed only when a genuine >3-channel / multi-page stack
        # is selected. Checked channels drive segmentation (mapped to R/G/B).
        self._chan_lbl = QLabel("Segmentation channels")
        self._chan_lbl.setStyleSheet(
            f"color: {LABEL}; font-size: 11px; font-weight: 500;")
        img_card.addWidget(self._chan_lbl)
        self.channel_list = QListWidget()
        self.channel_list.setToolTip(
            "Tick the channel(s) to segment on. One → grayscale; two → red+green; "
            "three → RGB. Each channel is percentile-normalised independently.")
        self.channel_list.setMaximumHeight(96)
        self.channel_list.setStyleSheet(
            f"QListWidget {{ background:{FG}; color:{TEXT}; border:1px solid {BORDER}; "
            f"font-size: 11px; }}")
        img_card.addWidget(self.channel_list)
        self._chan_lbl.setVisible(False)
        self.channel_list.setVisible(False)
        self._channel_names: list[str] = []

        # ── Z-stack / time-lapse toggle (multi-plane TIFF only) ────────────────
        # Hidden for an ordinary single-plane image; revealed only when the
        # loaded file actually has more than one Z/T plane. Segments each
        # plane independently and links instances across planes by overlap
        # (see napari_app.volume_stitch) instead of collapsing to one plane.
        # QCheckBox text never wraps (unlike QLabel), so it must stay short —
        # a longer label here widens the whole Predict panel and forces
        # horizontal scrolling; the full explanation lives in the tooltip.
        self.zstack_cb = QCheckBox("Segment as z-stack / time-lapse")
        self.zstack_cb.setToolTip(
            "For multi-plane TIFF/OME-TIFF (confocal z-stacks, time-lapse). "
            "Segments each plane independently and links instances across "
            "planes by overlap, producing one n-D label volume instead of "
            "unrelated 2-D masks per plane. Works with any engine; SAM2 is "
            "the flagship choice (trained for video/volumetric consistency).")
        self.zstack_cb.setStyleSheet(f"color: {LABEL}; font-size: 11px;")
        self.zstack_cb.setVisible(False)
        img_card.addWidget(self.zstack_cb)

        L.addWidget(img_card)

        # ── Run (prominent, outside cards — the main action) ──────────────────
        L.addSpacing(14)

        self.run_btn = QPushButton("  Run Prediction")
        self.run_btn.setFixedHeight(44)
        self.run_btn.setStyleSheet(BTN_PRIMARY)
        self.run_btn.setIcon(icons.icon("run", "#ffffff", 16))
        self.run_btn.setToolTip("Ctrl+R")
        self.run_btn.clicked.connect(self._run_prediction)
        L.addWidget(self.run_btn)

        L.addSpacing(8)

        self.active_btn = QPushButton("  Predict on active napari layer")
        self.active_btn.setFixedHeight(34)
        self.active_btn.setStyleSheet(BTN_SECONDARY)
        self.active_btn.setIcon(icons.icon("run", LABEL, 14))
        self.active_btn.setToolTip("Ctrl+Shift+R — runs on the Image layer selected in viewer")
        self.active_btn.clicked.connect(self._predict_active_layer)
        L.addWidget(self.active_btn)

        L.addSpacing(16)

        # Quality selector — a friendly front-end over resize + sampling density.
        q_row = QHBoxLayout(); q_row.setSpacing(8); q_row.setContentsMargins(0, 0, 0, 4)
        q_lbl = QLabel("Quality")
        q_lbl.setStyleSheet(f"color: {LABEL}; font-size: 11px; font-weight: 500;")
        q_lbl.setFixedWidth(70)
        q_row.addWidget(q_lbl)
        self.quality = Combo()
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
        self._results_card = SectionCard("Results", accent_color=SUCCESS, icon="check")
        self._results_card.setVisible(False)

        count_row = QHBoxLayout(); count_row.setSpacing(8); count_row.setContentsMargins(0, 2, 0, 4)
        self._cell_count_lbl = QLabel("—")
        self._cell_count_lbl.setStyleSheet(
            f"color: {SUCCESS}; font-size: 42px; font-weight: 800;"
            f"letter-spacing: -1.5px; background: transparent;")
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
        self.pixel_size.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.pixel_size.setMinimumHeight(34)
        self.pixel_size.setToolTip(
            "Microns per pixel from your microscope. Set it to report areas in "
            "µm² and sizes in µm. Leave at 0 to measure in pixels.")
        self.pixel_size.valueChanged.connect(self._on_pixel_size_changed)
        # µm/pixel starts as auto-filled-from-metadata; a manual edit makes the
        # field "owned" by the user so opening another file won't clobber it.
        self._pixel_size_is_auto = True
        self._setting_pixel_auto = False
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

        # Four result actions on a 2×2 grid (Measurements now sits inline).
        res_btns = QGridLayout(); res_btns.setSpacing(7); res_btns.setContentsMargins(0, 2, 0, 0)
        self._save_btn = QPushButton("Save masks")
        self._save_btn.setStyleSheet(BTN_SECONDARY)
        self._save_btn.setIcon(icons.icon("save", LABEL))
        self._save_btn.clicked.connect(self._save_masks)
        self._csv_btn = QPushButton("Export CSV")
        self._csv_btn.setStyleSheet(BTN_SECONDARY)
        self._csv_btn.setIcon(icons.icon("csv", LABEL))
        self._csv_btn.setToolTip("Saves cell_id, area, centroid for each detected cell")
        self._csv_btn.clicked.connect(self._export_csv)
        self._refine_btn = QPushButton("Refine…")
        self._refine_btn.setStyleSheet(BTN_SECONDARY)
        self._refine_btn.setIcon(icons.icon("refine", LABEL))
        self._refine_btn.setToolTip(
            "Edit the Labels layer in napari to correct cells,\n"
            "then click to run a 50-epoch fine-tune from the current checkpoint.")
        self._refine_btn.clicked.connect(self._run_refine)
        self._measure_btn = QPushButton("Measurements")
        self._measure_btn.setStyleSheet(BTN_SECONDARY)
        self._measure_btn.setIcon(icons.icon("measure", LABEL))
        self._measure_btn.setToolTip(
            "Per-cell morphometry: area, diameter, circularity, elongation,\n"
            "convexity and intensity — with distribution histograms.")
        self._measure_btn.clicked.connect(self._open_measurements)
        for _b in (self._save_btn, self._csv_btn, self._refine_btn, self._measure_btn):
            _b.setFixedHeight(32)
        res_btns.addWidget(self._save_btn,    0, 0)
        res_btns.addWidget(self._csv_btn,     0, 1)
        res_btns.addWidget(self._refine_btn,  1, 0)
        res_btns.addWidget(self._measure_btn, 1, 1)
        self._results_card.addLayout(res_btns)
        L.addWidget(self._results_card)

        # ── Display: colour cells by a measurement (hidden until a result with
        # at least one cell exists) — kept as its own card rather than nested
        # in Results, since Results' hero chips are 2-D-only and hidden for a
        # volume result, but colouring by measurement is just as meaningful
        # there (both share the same _last_measure/_masks(_fill) layers). ────
        self._color_card = SectionCard("Display", icon="settings")
        self.color_by = Combo()
        self.color_by.addItem("Instance ID (default)", "instance_id")
        self.color_by.currentIndexChanged.connect(self._on_color_by_changed)
        self.color_by.setEnabled(False)   # nothing to colour by until a result exists
        self._color_card.addLayout(_param_row("Colour cells by", self.color_by,
            "Paint every cell's fill + outline by one of its measured values "
            "instead of a random per-instance colour — a heatmap over the "
            "population (e.g. spot the biggest or roundest cells at a "
            "glance). Uses matplotlib's viridis colormap by default."))
        self._color_legend = _ColorLegend()
        self._color_legend.setVisible(False)
        self._color_card.addWidget(self._color_legend)

        self.view3d_cb = QCheckBox("View in 3D (rotate the volume)")
        self.view3d_cb.setStyleSheet(f"color:{LABEL}; font-size:11.5px; background:transparent;")
        self.view3d_cb.toggled.connect(self._on_view3d_toggled)
        self.view3d_cb.setVisible(False)   # only meaningful for a z-stack/time-lapse result
        self._color_card.addWidget(self.view3d_cb)
        # Card stays visible even with no result yet — "Colour cells by" is
        # the only part disabled below when there's nothing to colour by.
        L.addWidget(self._color_card)

        # Scale bar + a one-line info caption are always on, no toggle —
        # every screenshot/figure should be self-contained by default rather
        # than depend on the user remembering to switch something on. See
        # _refresh_info_overlay for what the caption shows.
        try:
            self.viewer.scale_bar.visible = True
        except Exception:
            pass   # older napari without a scale_bar overlay — no-op

        # ── Ground truth & evaluation (collapsed — validation tool) ────────────
        _gt_card = CollapsibleCard("Ground truth & evaluation", collapsed=True, icon="check")
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
        _batch_card = CollapsibleCard("Batch prediction", collapsed=True, icon="batch")
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
        _bench_card = CollapsibleCard("Benchmark engines vs GT", collapsed=True, icon="chart")
        _bench_card.addWidget(_field_label("Images folder"))
        self.bench_img = QLineEdit()
        self.bench_img.setPlaceholderText("Folder of images with ground-truth masks")
        _bench_card.addLayout(_dir_row(self, self.bench_img, "Select images folder"))
        _bench_card.addWidget(_field_label("Ground-truth folder"))
        self.bench_gt = QLineEdit()
        self.bench_gt.setPlaceholderText("Folder of label masks (matched by file name)")
        _bench_card.addLayout(_dir_row(self, self.bench_gt, "Select GT folder"))

        self._bench_checks: dict[str, QCheckBox] = {}
        for _spec in engine_registry.all_engines():
            cb = QCheckBox(_spec.bench_label)
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {LABEL}; font-size: 11px;")
            _bench_card.addWidget(cb)
            self._bench_checks[_spec.key] = cb

        self.bench_btn = QPushButton("Run benchmark")
        self.bench_btn.setFixedHeight(34); self.bench_btn.setStyleSheet(BTN_SUCCESS)
        self.bench_btn.clicked.connect(self._run_benchmark)
        _bench_card.addWidget(self.bench_btn)
        self.bench_progress = QLabel("")
        self.bench_progress.setStyleSheet(
            f"color: {LABEL}; font-size: 10px; font-family:{MONO};")
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
        _model_card = CollapsibleCard("Model settings", collapsed=True, icon="settings")
        _model_card.addWidget(_field_label("Custom checkpoint (.pth)"))
        _model_card.addLayout(_make_custom_lora_row(self))
        self.vit_name = Combo()
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
        self.device = Combo(); self._populate_devices()
        _model_card.addLayout(_param_row("Device", self.device))

        self.half_precision = QCheckBox("Half precision (fp16 autocast)")
        self.half_precision.setToolTip(
            "Run mask generation under CUDA autocast(fp16) for faster "
            "inference. CUDA only — has no effect on CPU/MPS.")
        self.half_precision.setStyleSheet(f"color: {LABEL}; font-size: 11px;")
        _model_card.addWidget(self.half_precision)

        self.compile_decoder = QCheckBox("Compile mask decoder (experimental)")
        self.compile_decoder.setToolTip(
            "torch.compile the mask decoder for faster repeated inference. "
            "CUDA only. The first prediction after enabling this (or after "
            "loading a new checkpoint) is slower while it compiles; falls "
            "back to eager silently if compilation fails.")
        self.compile_decoder.setStyleSheet(f"color: {LABEL}; font-size: 11px;")
        _model_card.addWidget(self.compile_decoder)

        self._model_card = _model_card
        L.addWidget(_model_card)

        # ── Inference parameters (collapsed — tune when needed) ────────────────
        _inf_card = CollapsibleCard("Inference parameters", collapsed=True, icon="settings")
        self.resize_size = Combo()
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
        self._cp_card = SectionCard("Cellpose-SAM settings", icon="settings")
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

        # ── SAM2 settings (shown only for the SAM2 engine) ─────────────────────
        self._sam2_card = SectionCard("SAM 2 settings", icon="settings")
        self.sam2_model_type = Combo()
        self.sam2_model_type.addItems(["large", "base_plus", "small", "tiny"])
        self._sam2_card.addLayout(_param_row("Model size", self.sam2_model_type,
            "Larger = better accuracy, slower. Must match the checkpoint below."))
        self._sam2_card.addWidget(_field_label("Checkpoint (.pt)"))
        self.sam2_checkpoint = QLineEdit()
        self.sam2_checkpoint.setPlaceholderText(
            "auto-detected from data_store/sam2_checkpoints/")
        self._sam2_card.addLayout(_file_row(self, self.sam2_checkpoint,
            "Select SAM2 checkpoint", "PyTorch (*.pt *.pth)"))
        self._sam2_card.addWidget(_field_label("Config override (advanced)"))
        self.sam2_config_text = QLineEdit()
        self.sam2_config_text.setPlaceholderText("auto (Hydra config name)")
        self.sam2_config_text.setToolTip(
            "Package-relative Hydra config name, e.g. "
            "configs/sam2.1/sam2.1_hiera_l.yaml. Leave blank to use the "
            "default for the model size above.")
        self._sam2_card.addWidget(self.sam2_config_text)
        # A Combo's width is set by its widest item (see the engine combo's
        # own comment in engines_sam2.py) — kept short for the same reason;
        # the tooltip below carries the full explanation.
        self.sam2_tracking_mode = Combo()
        self.sam2_tracking_mode.addItem("Independent + stitch (default)", "automatic")
        self.sam2_tracking_mode.addItem("Propagate (experimental)", "propagate")
        self.sam2_tracking_mode.setToolTip(
            "Only applies to \"Segment as z-stack\" above.\n"
            "Independent + stitch: segments each plane on its own, links "
            "instances by overlap between neighbouring planes.\n"
            "Propagate: seeds objects on the first plane, then tracks each "
            "one across every other plane with SAM2's memory-bank video "
            "model — stronger consistency, but experimental and unverified "
            "in this build.")
        self._sam2_card.addLayout(_param_row("Tracking mode", self.sam2_tracking_mode))
        self._sam2_card.setVisible(False)
        L.addWidget(self._sam2_card)

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
        self._done_signal.connect(self._show_results)
        self._volume_done_signal.connect(self._show_volume_results)
        self._finish_signal.connect(self._on_done)
        self._batch_progress_signal.connect(self._on_batch_progress)
        self._tile_progress_signal.connect(self._on_tile_progress)
        self._slice_progress_signal.connect(self._on_slice_progress)
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
        self.image_path.textChanged.connect(lambda _t: self._refresh_channel_picker())
        self.image_path.textChanged.connect(lambda _t: self._autofill_pixel_size())
        self.image_path.textChanged.connect(lambda _t: self._refresh_zstack_toggle())
        self._autofill_gt()
        self._on_gt_path_changed()
        self._populate_samples()
        self._refresh_zstack_toggle()

        self._autofill_from_sidecar()

    # ── Engine switching ──────────────────────────────────────────────────────

    def _current_engine(self) -> str:
        return self.engine.currentData() or "cellseg1"

    def _on_engine_changed(self, _idx=None):
        # Which settings cards/hint text apply is bespoke per engine (unlike the
        # combo list itself, this isn't registry-driven) — a new engine needs
        # its own branch/settings card here.
        engine = self._current_engine()
        is_cp = engine == "cellpose"
        is_sam2 = engine == "sam2"
        is_sam_lora = not is_cp and not is_sam2   # cellseg1 (SAM + LoRA)
        # Checkpoint / LoRA-specific controls are irrelevant for Cellpose and SAM2.
        self._ckpt_card.setVisible(is_sam_lora)
        self._inf_card.setVisible(is_sam_lora)
        if hasattr(self, "_model_card"):
            self._model_card.setVisible(is_sam_lora)
        self.quality.setEnabled(is_sam_lora)
        self._cp_card.setVisible(is_cp)
        self._sam2_card.setVisible(is_sam2)
        if is_cp:
            from napari_app.engines import cellpose_available
            if cellpose_available():
                self._engine_hint.setText(
                    "Zero-shot generalist — no checkpoint needed. First run downloads "
                    "the model weights (~a few hundred MB).")
            else:
                self._engine_hint.setText(
                    "⚠ Cellpose is not installed. Run:  pip install cellpose")
        elif is_sam2:
            from napari_app.engines_sam2 import sam2_available
            if sam2_available():
                self._engine_hint.setText(
                    "Zero-shot, with native z-stack/video support — tick "
                    "\"Segment as z-stack\" above for multi-plane files.")
            else:
                self._engine_hint.setText(
                    "⚠ SAM2 is not installed. Run:  pip install sam2  "
                    "(see github.com/facebookresearch/sam2)")
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
        self._sample_btn.setText("  Load sample microscopy images")
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
        exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy",
                ".nd2", ".czi", ".lif"}
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
                # Most bundled samples ship without a ground-truth mask (only
                # the synthetic phantom, plus any downloaded BBBC039 image,
                # have one) — flag it here so "GT didn't autofill" isn't a
                # surprise for the others.
                label = f"{f.name}   ·   has GT" if self._gt_sidecar(str(f)) else f.name
                self.sample_combo.addItem(label, str(f))
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
        lora_custom_text = self.lora_custom.text() if hasattr(self, "lora_custom") else ""
        return PredictController.resolve_lora(
            lora_custom_text, self.lora_combo.currentText(), self._lora_paths)

    def _resolve_sam(self):
        return PredictController.resolve_sam(
            self.sam_path.text(), self.vit_name.currentText(), STORAGE_DIR)

    def _refresh_channel_picker(self):
        """Show/populate the channel picker when a multi-channel stack is loaded.

        Probes the selected file's channel count cheaply (metadata only). The
        picker is revealed only for a genuine multi-channel image (>3 channels,
        or a labelled channel axis whose size is not the ordinary 1/3); normal
        RGB/grayscale images keep it hidden and thus flow through the legacy
        read path unchanged.
        """
        path = self.image_path.text().strip()
        self._channel_names = []
        self.channel_list.clear()
        show = False
        if path and Path(path).exists():
            try:
                from napari_app.channels import probe_channels
                n, names = probe_channels(path)
                if n > 3 or (1 < n < 3):   # not a normal RGB(3)/gray(1) image
                    self._channel_names = names
                    default = set(range(min(n, 3)))
                    for i, name in enumerate(names):
                        item = QListWidgetItem(f"{i}: {name}")
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        item.setCheckState(Qt.CheckState.Checked if i in default
                                           else Qt.CheckState.Unchecked)
                        self.channel_list.addItem(item)
                    show = True
            except Exception:
                show = False
        self._chan_lbl.setVisible(show)
        self.channel_list.setVisible(show)

    def _refresh_zstack_toggle(self):
        """Show the z-stack checkbox only for a file that genuinely has more
        than one Z/T plane; an ordinary single-plane image keeps it hidden
        (and force-unchecked, so _gather_params reports zstack=False) so the
        default 2-D path is completely unaffected."""
        path = self.image_path.text().strip()
        show = False
        if path and Path(path).exists():
            try:
                from napari_app.channels import has_z_stack
                show = has_z_stack(path)
            except Exception:
                show = False
        self.zstack_cb.setVisible(show)
        if not show:
            self.zstack_cb.setChecked(False)

    def _selected_channels(self) -> list[int] | None:
        """Checked segmentation channels, or ``None`` when the picker is inactive.

        ``None`` means "ordinary image" and keeps the default read path. When
        the picker is shown but nothing is ticked we fall back to the first
        channel so a run never fails on an empty selection.
        """
        if not self.channel_list.isVisible() or not self._channel_names:
            return None
        picked = [i for i in range(self.channel_list.count())
                  if self.channel_list.item(i).checkState() == Qt.CheckState.Checked]
        return picked or [0]

    def _gather_params(self) -> dict:
        """Snapshot every control PredictController.build_config/sam_config
        need, as plain values — the only bridge between Qt state and the
        Qt-free controller."""
        return {
            "engine": self._current_engine(),
            "image_path": self.image_path.text().strip(),
            "resize_size": int(self.resize_size.currentText()),
            "vit_name": self.vit_name.currentText(),
            "sam_path_text": self.sam_path.text(),
            "storage_dir": STORAGE_DIR,
            "lora_custom_text": self.lora_custom.text() if hasattr(self, "lora_custom") else "",
            "lora_combo_text": self.lora_combo.currentText(),
            "lora_paths": self._lora_paths,
            "lora_rank": self.lora_rank.value(),
            "device": self.device.currentText(),
            "half_precision": self.half_precision.isChecked(),
            "compile_decoder": self.compile_decoder.isChecked(),
            "points_per_side": self.points_per_side.value(),
            "pred_iou_thresh": self.pred_iou_thresh.value(),
            "stability_score_thresh": self.stability_score_thresh.value(),
            "box_nms_thresh": self.box_nms_thresh.value(),
            "min_mask_area": self.min_mask_area.value(),
            "clahe": self.clahe.isChecked(),
            "tiled": self.tiled.isChecked(),
            "cp_diameter": self.cp_diameter.value(),
            "cp_flow_threshold": self.cp_flow.value(),
            "cp_cellprob_threshold": self.cp_cellprob.value(),
            "channels": self._selected_channels(),
            "zstack": self.zstack_cb.isChecked() if self.zstack_cb.isVisible() else False,
            "stitch_iou": 0.25,
            "sam2_model_type": self.sam2_model_type.currentText(),
            "sam2_checkpoint_text": self.sam2_checkpoint.text(),
            "sam2_config_text": self.sam2_config_text.text(),
            "sam2_tracking_mode": self.sam2_tracking_mode.currentData() or "automatic",
        }

    def _build_config(self):
        return self._controller.build_config(self._gather_params())

    def _sam_config(self) -> dict:
        """Full SAM + LoRA config. Used by the CellSeg1 engine and always by
        the interactive Annotate session and Refine (both need SAM+LoRA
        regardless of the engine selector — Refine fine-tunes a LoRA
        checkpoint via cellseg1_train.py, which has no Cellpose equivalent).
        Requires an image, a LoRA checkpoint and a SAM backbone."""
        return self._controller.sam_config(self._gather_params())

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

    # ── Agentic tuning loop (the Assistant's "Auto-tune") ─────────────────────

    def has_ground_truth(self) -> bool:
        p = self.gt_path.text().strip()
        return bool(p) and Path(p).exists()

    def start_auto_tune(self, on_step, on_finish) -> str | None:
        """Start the tuning loop against the current image + loaded ground
        truth. Returns an error string (and starts nothing) if a
        precondition isn't met, else ``None`` once the loop has started in
        the background. ``on_step``/``on_finish`` are forwarded to
        :meth:`PredictController.run_tuning_loop_async` verbatim — they fire
        from a background thread exactly like every other async controller
        callback, so the caller (the Assistant widget) is responsible for
        making them thread-safe (a Qt signal), same as it already does for
        ``rerun``'s own callbacks.
        """
        if self._last_mask is None:
            return "Run a prediction first."
        if not self.has_ground_truth():
            return "Set a ground-truth mask first (Ground truth card) to auto-tune against."
        try:
            gt = self._load_gt_mask()
        except ValueError as e:
            return str(e)
        if gt.shape != self._last_mask.shape:
            import cv2
            gt = cv2.resize(gt.astype(np.float32),
                            (self._last_mask.shape[1], self._last_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
        params = self._gather_params()
        self._controller.run_tuning_loop_async(
            params, gt, on_step=on_step, on_log=self._append_log, on_finish=on_finish)
        return None

    def stop_auto_tune(self):
        self._controller.stop_tuning()

    def restore_tuning_step(self, params: dict):
        """Apply a previous tuning step's full parameter snapshot and
        re-run — the loop's "undo": jump back to any recorded step, not only
        the one it happened to end on."""
        self.apply_params(params)
        self.rerun()

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
        elif config.get("engine") == "sam2":
            from napari_app.engines_sam2 import cache_status as sam2_cache_status
            self._append_log(f"▶ {Path(config['image_path']).name}  [{sam2_cache_status()}]")
        else:
            from napari_app.inference_cache import cache_status
            self._append_log(f"▶ {Path(config['image_path']).name}  [{cache_status()}]")

        if config.get("zstack"):
            # z-stack/time-lapse: a scoped-down sibling path (see
            # _show_volume_results) — engine-agnostic orchestration lives in
            # PredictController.run_volume_prediction_async.
            def on_volume_result(img_vol, mask_vol, stack):
                self._volume_done_signal.emit(img_vol, mask_vol)

            self._controller.run_volume_prediction_async(
                config, on_slice=self._slice_progress_signal.emit, on_result=on_volume_result,
                on_log=self._log_signal.emit, on_finish=self._finish_signal.emit)
            return

        def on_result(img_arr, mask, stack):
            # Stash the raw channel stack (None on the ordinary path) so the
            # measurement pass can report per-channel intensity. Set before
            # emitting so the queued main-thread handler sees it.
            self._last_channel_stack = stack
            self._done_signal.emit(img_arr, mask)

        self._controller.run_prediction_async(
            config, on_tile=self._tile_progress_signal.emit, on_result=on_result,
            on_log=self._log_signal.emit, on_finish=self._finish_signal.emit)

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

    def _layer_scale_kwargs(self, ndim: int) -> dict:
        """``scale=``/``units=`` kwargs for add_image/add_labels, so napari's
        scale bar (and anything else that reads world coordinates) reflects
        real µm instead of raw pixels.

        Empty when the µm/pixel field is at its "off" sentinel (0 — see
        ``self.pixel_size``'s ``setSpecialValueText``), so uncalibrated
        images keep napari's plain 1-unit-per-pixel default exactly as
        before. ``ndim`` is the number of *spatial* axes — a Labels mask's
        own ndim (2 for a plane, 3 for a z-stack/time-lapse); a leading z/t
        axis has no pixel-size calibration of its own so it gets scale 1 /
        unit "pixel", while the last two (always the imaging plane) get the
        real value. The same tuple length is correct for the matching
        add_image call too *once that call also passes an explicit ``rgb=``*
        (see ``_is_rgb_like``) — napari's Image layer only strips the
        trailing colour channel axis when it's told the data is RGB, and
        its own size-based auto-guess (both non-channel dims > 30px) isn't
        reliable enough to lean on for an arbitrary user image.
        """
        px = self.pixel_size.value()
        if px <= 0:
            return {}
        lead = ndim - 2
        return dict(scale=(1.0,) * lead + (px, px),
                   units=("pixel",) * lead + ("um", "um"))

    @staticmethod
    def _is_rgb_like(arr: np.ndarray, spatial_ndim: int) -> bool:
        """Whether ``arr`` has a trailing colour-channel axis on top of
        ``spatial_ndim`` spatial axes (matching mask.ndim), so ``add_image``
        can be told ``rgb=`` explicitly instead of relying on napari's own
        size-based guess (>30px on the non-channel axes) — a fine heuristic
        for a typical microscopy image, but wrong for a small crop/thumbnail,
        which would then get an extra unwanted axis in `scale`/`units`.
        """
        return arr.ndim == spatial_ndim + 1 and arr.shape[-1] in (3, 4)

    def _add_filled_labels(self, mask: np.ndarray, name: str, *,
                           outline_opacity: float = 0.7, fill_opacity: float = 0.35,
                           solid_rgba: tuple | None = None, scale_kwargs: dict | None = None):
        """Add a mask as two stacked Labels layers instead of outline-only:
        a low-opacity filled wash underneath a crisp outline on top, so a
        cell reads clearly even where its border happens to blend into the
        underlying image — the same "fill + border, one colour" look tools
        like QuPath ("Fill detections") and CellProfiler's OverlayOutlines
        already default to.

        napari's own Labels layer can't blend fill and contour in a single
        layer — `contour` is a 0/N *toggle* (filled *or* outline-only), not
        additive — so two layers sharing the same label data is the
        standard way to get both at once; the outline layer is added last
        so it renders on top of the fill.

        ``solid_rgba``, if given, paints every label the same colour on
        both layers instead of napari's default random-per-label colours
        (used for the ground-truth overlay, so it reads as one consistent
        colour rather than a rainbow that would blend with the prediction
        layer). Returns the outline layer (the one on top, contour=1).
        """
        sk = scale_kwargs or {}
        fill = self.viewer.add_labels(mask, name=f"{name}_fill", opacity=fill_opacity, **sk)
        outline = self.viewer.add_labels(mask, name=name, opacity=outline_opacity, **sk)
        outline.contour = 1
        # Remembered so "Colour cells by" can restore napari's default
        # per-instance colours later without having to reconstruct them.
        fill.metadata["default_colormap"] = fill.colormap
        outline.metadata["default_colormap"] = outline.colormap
        if solid_rgba is not None:
            _color_labels_solid(fill, mask, solid_rgba)
            _color_labels_solid(outline, mask, solid_rgba)
        return outline

    def _refresh_color_by_options(self):
        """(Re)populate "Colour cells by" from the current result's numeric
        columns, and enable/disable it. Called after any measurement — 2-D
        (_recompute_measurements) or 3-D (_show_volume_results) — so it
        always reflects whichever schema the current result actually used
        (analysis._SCHEMA vs. _SCHEMA_3D have different columns).

        Only the combo itself is disabled when there's nothing to colour by
        — the "Display" card it lives in stays visible regardless, since
        the scale-bar toggle alongside it is meaningful even before any
        prediction has cells to measure.
        """
        result = self._last_measure
        current = self.color_by.currentData() or "instance_id"
        self.color_by.blockSignals(True)
        self.color_by.clear()
        self.color_by.addItem("Instance ID (default)", "instance_id")
        if result:
            for key, label, unit in result["columns"][1:]:   # [0] is always cell_id
                self.color_by.addItem(f"{label} ({unit})" if unit else label, key)
        idx = self.color_by.findData(current)
        self.color_by.setCurrentIndex(idx if idx >= 0 else 0)
        self.color_by.blockSignals(False)

        self.color_by.setEnabled(bool(result) and int(result.get("n_cells", 0)) > 0)
        self._apply_color_by(self.color_by.currentData() or "instance_id")

    def _on_color_by_changed(self, _idx=None):
        self._apply_color_by(self.color_by.currentData() or "instance_id")

    def _on_view3d_toggled(self, checked: bool):
        try:
            self.viewer.dims.ndisplay = 3 if checked else 2
        except Exception:
            pass

    def _refresh_info_overlay(self):
        """Keep a one-line caption burned onto the canvas itself (napari's
        built-in text_overlay), in sync with the current result.

        This is deliberately not just a mirror of the side panel's stat
        chips: its whole point is to survive a screenshot/exported figure
        that doesn't include the panel, the way a scale bar does — cell
        count, the headline size stat, calibration status (so a figure
        never silently implies real units when none were set), and which
        measurement the colours mean, if "Colour cells by" is active.
        """
        try:
            ov = self.viewer.text_overlay
        except Exception:
            return   # older napari without a text_overlay — no-op

        result = self._last_measure
        if not result or not int(result.get("n_cells", 0)):
            ov.visible = False
            return

        from napari_app import analysis
        n = result["n_cells"]
        s = result["summary"]
        size_key = "volume" if "volume" in s else "diameter"
        size_label = "volume" if size_key == "volume" else "Ø"
        size_unit = next((u for k, _l, u in result["columns"] if k == size_key), "")
        size_val = s.get(size_key, {}).get("median" if size_key == "diameter" else "mean", 0)
        calib = f"{self.pixel_size.value():g} µm/px" if self.pixel_size.value() > 0 else "uncalibrated (px)"

        lines = [f"{n} cell{'s' if n != 1 else ''}  ·  {size_label} {size_val:.1f} {size_unit}  ·  {calib}"]

        metric_key = self.color_by.currentData()
        if metric_key and metric_key != "instance_id":
            rng = analysis.measurement_range(result, metric_key)
            if rng is not None:
                lines.append(f"Coloured by {self.color_by.currentText()}: {rng[0]:.3g} – {rng[1]:.3g}")

        ov.text = "\n".join(lines)
        ov.position = "top_left"   # scale bar's own default is bottom_right — keep them apart
        ov.visible = True

    def _apply_color_by(self, metric_key: str):
        """Re-colour the current result's fill+outline layers by a measured
        value, or restore napari's default per-instance colours for
        "instance_id". A no-op if there's no current result layer (e.g.
        right after clearing, before the layers exist) — never raises, since
        this can fire from a signal at almost any point in the widget's
        lifecycle.

        Also the single point that keeps the canvas info caption in sync —
        called on every new result (via _refresh_color_by_options) *and*
        every manual "Colour cells by" change (via _on_color_by_changed), so
        rather than duplicate that trigger, the caption refresh just piggy-
        backs here instead of needing its own call site in both places.
        """
        self._refresh_info_overlay()
        name = Path(self.image_path.text()).stem
        try:
            fill = self.viewer.layers[f"{name}_masks_fill"]
            outline = self.viewer.layers[f"{name}_masks"]
        except (KeyError, TypeError):
            # KeyError: no such layer (nothing predicted yet, or cleared).
            # TypeError: self.viewer.layers doesn't support name-keyed lookup
            # at all (only ever seen in tests using a plain list stand-in for
            # a real napari LayerList) — same "nothing to colour" outcome.
            return

        if metric_key == "instance_id" or not self._last_measure:
            fill.colormap = fill.metadata.get("default_colormap", fill.colormap)
            outline.colormap = outline.metadata.get("default_colormap", outline.colormap)
            self._color_legend.setVisible(False)
            return

        from napari_app import analysis
        id_to_rgba = analysis.label_colormap_from_measurement(self._last_measure, metric_key)
        if not id_to_rgba:
            self._color_legend.setVisible(False)
            return
        rng = analysis.measurement_range(self._last_measure, metric_key)
        if rng is not None:
            self._color_legend.set_range(*rng)
            self._color_legend.setVisible(True)
        _color_labels_by_map(fill, fill.data, id_to_rgba)
        _color_labels_by_map(outline, outline.data, id_to_rgba)

    def _show_results(self, img_arr, mask):
        name = Path(self.image_path.text()).stem
        self._last_mask     = mask
        self._last_img_rgb  = img_arr
        self._last_img_path = self.image_path.text()

        for lyr in list(self.viewer.layers):
            if lyr.name.startswith(name) and "_gt" not in lyr.name:
                self.viewer.layers.remove(lyr)

        self.view3d_cb.setChecked(False)   # a plain 2-D result has no volume to rotate
        self.view3d_cb.setVisible(False)
        sk = self._layer_scale_kwargs(2)
        self.viewer.add_image(img_arr, name=f"{name}_image",
                              rgb=self._is_rgb_like(img_arr, 2), **sk)
        if mask is not None and mask.max() > 0:
            self._add_filled_labels(mask.astype(np.int32), f"{name}_masks", scale_kwargs=sk)
        self._recompute_measurements()
        # Count the headline number up for a live, product feel.
        if mask is not None and int(mask.max()) > 0:
            motion.count_up(self._cell_count_lbl, int(mask.max()))
        self.viewer.reset_view()

    def _show_volume_results(self, img_vol, mask_vol):
        """Add a z-stack/time-lapse prediction's results as n-D layers.

        A scoped-down sibling of _show_results for volumes: napari's Image/
        Labels layers are n-D natively, so add_image/add_labels need no
        change. Measurements *are* computed (analysis.compute_measurements
        dispatches on mask.ndim to a 3-D-specific schema — volume instead of
        area, no perimeter/circularity/eccentricity/orientation, which have
        no 3-D equivalent), populating _last_measure so "Open measurements"
        and "Export measurements" work exactly like the 2-D path. The
        compact hero-chip row (_chip_diam/_chip_area/_chip_cov) stays hidden
        for this result, though — those chips' captions ("Area", a 2-D word)
        are hardcoded Qt labels, not schema-driven, so relabelling them for a
        volume result is a UI change of its own rather than a size a 3-D
        mask should silently trigger; the full, correctly-labelled table is
        one click away in the Measurements window instead. The "Colour cells
        by" card (_refresh_color_by_options) isn't schema-specific though, so
        it's shown here too, alongside a "View in 3D" toggle (viewer.dims.
        ndisplay) that only makes sense for a real volume. GT overlay is
        still 2-D-only.
        """
        name = Path(self.image_path.text()).stem
        self._last_mask     = None
        self._last_img_rgb  = None
        self._last_img_path = self.image_path.text()

        for lyr in list(self.viewer.layers):
            if lyr.name.startswith(name) and "_gt" not in lyr.name:
                self.viewer.layers.remove(lyr)

        self.view3d_cb.setVisible(True)   # leaves the checked-state as the user set it
        sk = self._layer_scale_kwargs(mask_vol.ndim)
        self.viewer.add_image(img_vol, name=f"{name}_image",
                              rgb=self._is_rgb_like(img_vol, mask_vol.ndim), **sk)
        n_cells = int(mask_vol.max()) if mask_vol is not None and mask_vol.size else 0
        if n_cells > 0:
            self._add_filled_labels(mask_vol.astype(np.int32), f"{name}_masks", scale_kwargs=sk)

        from napari_app import analysis
        try:
            result = (analysis.compute_measurements(mask_vol, intensity_image=img_vol,
                                                     pixel_size_um=self.pixel_size.value())
                      if n_cells > 0 else None)
        except Exception as e:
            self._append_log(f"[WARN] 3-D measurement failed: {e}")
            result = None
        self._last_measure = result
        self._results_card.setVisible(False)   # hero chips are 2-D-worded; see docstring

        if result:
            self._append_log(f"[INFO] 3-D result: {analysis.summary_line(result)}  "
                             f"(open Measurements for the full table)")
        else:
            self._append_log(f"[INFO] 3-D result: {n_cells} cells across {mask_vol.shape[0]} planes.")
        if n_cells > 0:
            motion.count_up(self._cell_count_lbl, n_cells)
        self._refresh_color_by_options()
        self.viewer.reset_view()

    def _autofill_pixel_size(self):
        """Fill the µm/pixel field from the selected file's metadata.

        OME-TIFF / ND2 / CZI / LIF carry a physical pixel size; when they do and
        the user hasn't manually set the field, we pre-fill it so measurements
        come out in real units without hunting for the scope's calibration.
        A file without calibration (plain PNG/TIFF) leaves the field untouched,
        and a value the user typed themselves is never overwritten.
        """
        if not getattr(self, "_pixel_size_is_auto", True):
            return
        path = self.image_path.text().strip()
        if not path or not Path(path).exists():
            return
        try:
            from napari_app.channels import read_pixel_size_um
            um = read_pixel_size_um(path)
        except Exception:
            um = None
        if not um or um <= 0:
            return
        self._setting_pixel_auto = True
        try:
            self.pixel_size.setValue(float(um))
        finally:
            self._setting_pixel_auto = False

    def _on_pixel_size_changed(self, _value=None):
        # A change we didn't originate means the user edited the field; stop
        # auto-filling it from metadata on subsequent file selections.
        if not getattr(self, "_setting_pixel_auto", False):
            self._pixel_size_is_auto = False
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
            self._refresh_color_by_options()
            return
        from napari_app import analysis
        stack = getattr(self, "_last_channel_stack", None)
        ci = stack.data if stack is not None else None
        cn = stack.names if stack is not None else None
        try:
            result = analysis.compute_measurements(
                mask, intensity_image=self._last_img_rgb,
                pixel_size_um=self.pixel_size.value(),
                channel_intensities=ci, channel_names=cn)
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
        self._refresh_color_by_options()

    def _open_measurements(self):
        if self._last_measure is None:
            self._append_log("[WARN] Run a prediction with detected cells first"); return
        from napari_app.widgets.measurements_window import get_measurements_window
        mw = get_measurements_window()
        if not self._mw_row_selected_wired:
            # get_measurements_window() is a module-level singleton reused
            # across every "Open measurements" click — connect exactly once
            # per PredictWidget instance rather than re-connecting (and thus
            # firing _on_measurement_row_selected multiple times per click)
            # on each call.
            mw.row_selected.connect(self._on_measurement_row_selected)
            self._mw_row_selected_wired = True
        mw.set_result(self._last_measure,
                      Path(self._last_img_path).name if self._last_img_path else "")
        mw.show_and_raise()

    def _on_measurement_row_selected(self, cell_id: int):
        """Highlight the clicked Measurements-table row's cell in the
        viewer — QuPath's own "select in table -> selects on the image"
        behaviour, so a value that stands out in the table can be found on
        the image without hunting for it by eye. ``cell_id < 0`` means the
        table selection was cleared, so the highlight lifts and every cell
        is shown normally again.
        """
        name = Path(self.image_path.text()).stem
        try:
            fill = self.viewer.layers[f"{name}_masks_fill"]
            outline = self.viewer.layers[f"{name}_masks"]
        except (KeyError, TypeError):
            return
        show = cell_id >= 0
        for layer in (fill, outline):
            layer.show_selected_label = show
            if show:
                layer.selected_label = int(cell_id)

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
                if lyr.name in (lname, f"{lname}_fill"):
                    self.viewer.layers.remove(lyr)
            self._add_filled_labels(gt, lname, outline_opacity=0.9,
                                    solid_rgba=(0.0, 1.0, 0.35, 1.0),  # uniform green
                                    scale_kwargs=self._layer_scale_kwargs(gt.ndim))
            self._append_log(f"✓ GT loaded — {int(gt.max())} cells (green)")
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

        self.batch_btn.setEnabled(False)
        self.batch_stop_btn.setEnabled(True)
        self.batch_progress.setVisible(True)
        self.batch_progress.setValue(0)
        self.batch_lbl.setText(f"0 / {len(images)}")
        self._append_log(f"▶ Batch: {len(images)} images → {out_dir.name}/")

        px = self.pixel_size.value()
        self._cohort_records = []
        self._cohort_out = out_dir

        def on_cohort_ready(records, cohort_out_dir):
            self._cohort_records = records
            self._cohort_out = cohort_out_dir
            self._cohort_ready_signal.emit()

        self._controller.run_batch_async(
            config, images, out_dir, px,
            on_log=self._log_signal.emit,
            on_progress=self._batch_progress_signal.emit,
            on_cohort_ready=on_cohort_ready,
            on_finish=self._batch_finish_signal.emit)

    def _on_cohort_ready(self):
        records = getattr(self, "_cohort_records", [])
        if not records:
            return
        from napari_app.widgets.cohort_window import get_cohort_window
        w = get_cohort_window()
        w.set_records(records, str(getattr(self, "_cohort_out", "")))
        w.show_and_raise()

    # ── Benchmark engines vs ground truth ─────────────────────────────────────

    def _run_benchmark(self):
        img_dir = Path(self.bench_img.text().strip())
        gt_dir  = Path(self.bench_gt.text().strip())
        if not img_dir.is_dir():
            self._append_log("[ERROR] Benchmark: images folder not found"); return
        if not gt_dir.is_dir():
            self._append_log("[ERROR] Benchmark: GT folder not found"); return
        engines = [key for key, cb in self._bench_checks.items() if cb.isChecked()]
        if not engines:
            self._append_log("[ERROR] Benchmark: select at least one engine"); return

        rs = int(self.resize_size.currentText())
        bases = {}
        try:
            if "cellseg1" in engines:
                bases["cellseg1"] = self._sam_config()
            if "cellpose" in engines:
                if not engine_registry.get("cellpose").available():
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

        def on_done(cols, rows):
            self._bench_cols, self._bench_rows = cols, rows
            self._benchmark_done_signal.emit()

        self._controller.run_benchmark_async(
            engines, bases, pairs, img_dir,
            on_row=self._benchmark_row_signal.emit,
            on_log=self._log_signal.emit,
            on_done=on_done)

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
        self._controller.stop_batch()
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

    def _on_slice_progress(self, done: int, total: int):
        # Same determinate-bar treatment as _on_tile_progress, worded for the
        # z-stack/time-lapse path (one plane at a time instead of one tile).
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self.progress_bar.setFormat(f"plane {done}/{total}")
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
            # Always the full SAM+LoRA config, like Annotate — Refine always
            # fine-tunes the selected LoRA checkpoint via cellseg1_train.py,
            # regardless of which engine the Predict tab's selector shows.
            base_config = self._sam_config()
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
    border: 1px solid {BORDER}; border-radius: 8px;
    font-family: {MONO}; font-size: 11px;
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
}}
QHeaderView::section {{
    background: {CARD_HEADER}; color: {LABEL};
    padding: 5px 9px; border: none; border-bottom: 1px solid {BORDER};
    font-weight: 700; font-size: 10px; letter-spacing: 0.5px;
}}
QTableWidget::item {{ padding: 3px 9px; }}
QTableCornerButton::section {{ background: {CARD_HEADER}; border: none; }}
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


def _color_labels_by_map(layer, mask, id_to_rgba: dict, default_rgba=(0.5, 0.5, 0.5, 1.0)):
    """Paint each non-zero label its own colour from ``id_to_rgba`` — the
    "colour by measurement" counterpart to ``_color_labels_solid``'s "one
    uniform colour". A label present in ``mask`` but missing from
    ``id_to_rgba`` (shouldn't normally happen — every label in the mask has a
    measurement row) falls back to ``default_rgba`` rather than raising.
    """
    try:
        from napari.utils.colormaps import DirectLabelColormap
        ids = np.unique(mask)
        cmap = {int(i): tuple(id_to_rgba.get(int(i), default_rgba)) for i in ids if i != 0}
        cmap[None] = (0.0, 0.0, 0.0, 0.0)
        cmap[0] = (0.0, 0.0, 0.0, 0.0)
        layer.colormap = DirectLabelColormap(color_dict=cmap)
    except Exception:
        pass  # older napari — fall back to whatever colouring was already set


class _ColorLegend(QWidget):
    """A small gradient swatch + min/max value labels: the key for "Colour
    cells by", so a heatmap has a way to read a value off it instead of
    just floating colours (mirrors QuPath's measurement-map legend).
    """

    def __init__(self):
        super().__init__()
        row = QHBoxLayout(); row.setContentsMargins(0, 2, 0, 2); row.setSpacing(6)
        self._lo = QLabel("")
        self._lo.setStyleSheet(f"color:{DIM}; font-size:10px; background:transparent;")
        self._swatch = QLabel()
        self._swatch.setFixedHeight(10)
        self._swatch.setMinimumWidth(50)
        self._hi = QLabel("")
        self._hi.setStyleSheet(f"color:{DIM}; font-size:10px; background:transparent;")
        row.addWidget(self._lo)
        row.addWidget(self._swatch, stretch=1)
        row.addWidget(self._hi)
        self.setLayout(row)

    def set_range(self, lo: float, hi: float, cmap_name: str = "viridis"):
        from matplotlib import colormaps
        cmap = colormaps[cmap_name]
        stops = ", ".join(
            f"stop:{t:.2f} rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},255)"
            for t, (r, g, b, _a) in ((t, cmap(t)) for t in (0.0, 0.25, 0.5, 0.75, 1.0)))
        self._swatch.setStyleSheet(
            "border-radius:3px;"
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops});")
        self._lo.setText(f"{lo:.3g}")
        self._hi.setText(f"{hi:.3g}")


def _make_stat_chip(caption: str):
    """A compact stat tile: big value over a small uppercase caption.

    Returns (value_label, frame) — keep the label to update, add the frame.
    """
    frame = QFrame()
    frame.setObjectName("StatChip")
    frame.setStyleSheet(
        f"QFrame#StatChip {{ background: {INPUT}; border: 1px solid {BORDER};"
        f" border-radius: 8px; }}")
    v = QVBoxLayout(frame)
    v.setContentsMargins(11, 9, 11, 9); v.setSpacing(3)
    val = QLabel("—")
    val.setStyleSheet(
        f"color: {TEXT}; font-size: 16px; font-weight: 700;"
        f"font-family: {MONO}; background: transparent;")
    cap = QLabel(caption)
    cap.setStyleSheet(
        f"color: {DIM}; font-size: 9px; font-weight: 700; letter-spacing: 1px;"
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
# _to_display_uint8 / _read_for_predict / _predict_cached / _predict_tiled /
# _apply_clahe now live in napari_app.core.predict_controller (Qt-free, unit
# tested) and are imported at the top of this file; kept importable from here
# too since existing wiring tests reference them as `predict_widget.<name>`.

def _find_test_image():
    for ext in ("*.png", "*.tif", "*.tiff", "*.jpg"):
        hits = list(TEST_IMAGE_DIR.glob(ext))
        if hits:
            return hits[0]
    return None
