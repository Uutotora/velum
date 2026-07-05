"""
Annotate tab — interactive click-to-segment powered by SAM point prompts.

Workflow:
  • Start a session (computes the image embedding once).
  • Left-click a cell           → segments that cell as a new label.
  • Shift + left-click          → add a positive point to refine the last cell.
  • Ctrl/⌘ + left-click         → add a negative point to carve the last cell.
The result is a fully editable napari Labels layer, ready to correct with the
brush and feed straight into the Train or Refine flows.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal

from napari_app.theme import (
    BG, FG, BORDER, BORDER_STRONG, TEXT, DIM, LABEL, ACCENT, ACCENT_SOFT,
    ACCENT_LINE, SUCCESS, DANGER, CARD_HEADER, MONO,
    WIDGET_SS, BTN_PRIMARY, BTN_SECONDARY, BTN_DANGER,
)
from napari_app.widgets.common import SectionCard
from napari_app.widgets.log_window import get_log_window
from napari_app import icons
from PyQt6.QtWidgets import QSizePolicy


def _kbd(text: str) -> QLabel:
    k = QLabel(text)
    k.setStyleSheet(
        f"background:{CARD_HEADER}; border:1px solid {BORDER_STRONG};"
        f"border-bottom-width:2px; border-radius:5px; padding:1px 7px;"
        f"color:{TEXT}; font-family:{MONO}; font-size:11px;")
    k.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return k


def _legend_row(dot_color: str, keys: list[str], desc: str) -> QWidget:
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(7)
    dot = QLabel()
    dot.setFixedSize(9, 9)
    dot.setStyleSheet(f"background:{dot_color}; border-radius:4px;")
    row.addWidget(dot)
    for i, k in enumerate(keys):
        row.addWidget(_kbd(k))
        if i < len(keys) - 1:
            plus = QLabel("+")
            plus.setStyleSheet(f"color:{DIM}; font-size:11px; background:transparent;")
            row.addWidget(plus)
    d = QLabel(desc)
    d.setStyleSheet(f"color:{LABEL}; font-size:12px; background:transparent;")
    row.addWidget(d)
    row.addStretch()
    return w


def _step(n: int, html: str) -> QWidget:
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 3, 0, 3)
    row.setSpacing(11)
    badge = QLabel(str(n))
    badge.setFixedSize(22, 22)
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setStyleSheet(
        f"background:{ACCENT_SOFT}; color:{ACCENT}; border:1px solid {ACCENT_LINE};"
        f"border-radius:11px; font-size:12px; font-weight:700;")
    txt = QLabel(html)
    txt.setTextFormat(Qt.TextFormat.RichText)
    txt.setStyleSheet(f"color:{LABEL}; font-size:12px; background:transparent;")
    txt.setWordWrap(True)
    row.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)
    row.addWidget(txt, stretch=1)
    return w


class AnnotateWidget(QWidget):
    _ready_signal  = pyqtSignal(bool, str)      # (ok, message)
    _result_signal = pyqtSignal(object, int, float)  # (mask_bool, label_id, score)

    def __init__(self, viewer, predict_widget):
        super().__init__()
        self.viewer = viewer
        self.predict = predict_widget

        self._session = None
        self._label_img = None
        self._labels_layer = None
        self._image_layer = None
        self._pts_layer = None
        self._cb = None
        self._busy = False

        # current-object prompt state
        self._active_id = 0
        self._points: list = []
        self._plabels: list = []
        self._last_low = None
        self._prompt_coords: list = []
        self._prompt_face: list = []

        self.setStyleSheet(WIDGET_SS)
        outer = QVBoxLayout(self); outer.setSpacing(0); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        L = QVBoxLayout(inner); L.setSpacing(0); L.setContentsMargins(14, 8, 14, 16)

        # ── Session card ───────────────────────────────────────────────────────
        sess_card = SectionCard("Interactive session", icon="annotate")
        intro = QLabel(
            "Click one point on a cell and SAM outlines the whole cell for you — "
            "you never draw the outline by hand. Each click adds one cell. "
            "(Uses the checkpoint + image from the Predict tab.)")
        intro.setStyleSheet(f"color:{LABEL}; font-size:12px; background:transparent;")
        intro.setWordWrap(True)
        sess_card.addWidget(intro)

        self._start_btn = QPushButton("  Start interactive session")
        self._start_btn.setFixedHeight(40)
        self._start_btn.setStyleSheet(BTN_PRIMARY)
        self._start_btn.setIcon(icons.icon("run", "#ffffff", 15))
        self._start_btn.clicked.connect(self._start)
        sess_card.addWidget(self._start_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(4)
        sess_card.addWidget(self._progress)

        self._status = QLabel("Not started.")
        self._status.setStyleSheet(
            f"color:{DIM}; font-size:11px; background:transparent; font-family:{MONO};")
        self._status.setWordWrap(True)
        sess_card.addWidget(self._status)
        L.addWidget(sess_card)

        # ── How it works ───────────────────────────────────────────────────────
        how_card = SectionCard("How it works", icon="guide")
        how_card.addWidget(_step(1, "<b>Left-click</b> a cell → segments it as a new label."))
        how_card.addWidget(_step(2, "<b>Shift-click</b> → grows the last cell."))
        how_card.addWidget(_step(3, "<b>Ctrl / ⌘-click</b> → carves the last cell."))
        L.addWidget(how_card)

        # ── Controls card ──────────────────────────────────────────────────────
        ctl_card = SectionCard("Controls", icon="settings")
        ctl_card.addWidget(_legend_row(SUCCESS, ["click"], "segment a new cell"))
        ctl_card.addWidget(_legend_row(SUCCESS, ["⇧", "click"], "add to the last cell"))
        ctl_card.addWidget(_legend_row(DANGER,  ["⌘", "click"], "remove from the last cell"))
        ctl_card.addWidget(_legend_row(DIM,     ["drag"], "pan (never segments)"))
        manual = QLabel(
            "Prefer to draw by hand? Select the *_annotate_masks layer and use "
            "napari's paintbrush (top-left tools).")
        manual.setStyleSheet(f"color:{DIM}; font-size:10.5px; background:transparent; padding-top:4px;")
        manual.setWordWrap(True)
        ctl_card.addWidget(manual)

        row = QHBoxLayout(); row.setSpacing(7)
        self._undo_btn = QPushButton("  Undo last")
        self._undo_btn.setFixedHeight(32)
        self._undo_btn.setStyleSheet(BTN_SECONDARY)
        self._undo_btn.setIcon(icons.icon("undo", LABEL, 14))
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._undo_last)
        row.addWidget(self._undo_btn)

        self._clear_btn = QPushButton("Clear all")
        self._clear_btn.setFixedHeight(32)
        self._clear_btn.setStyleSheet(BTN_SECONDARY)
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self._clear_all)
        row.addWidget(self._clear_btn)
        ctl_card.addLayout(row)

        self._stop_btn = QPushButton("End session")
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.setStyleSheet(BTN_DANGER)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        ctl_card.addWidget(self._stop_btn)

        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet(
            f"color:{TEXT}; font-size:13px; font-weight:600; background:transparent; padding-top:2px;")
        ctl_card.addWidget(self._count_lbl)
        L.addWidget(ctl_card)

        L.addStretch()
        foot = QLabel("CellSeg1 · one-shot cell segmentation")
        foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        foot.setStyleSheet(f"color:{DIM}; font-size:10.5px; background:transparent; padding-top:18px;")
        L.addWidget(foot)

        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.setMinimumWidth(260)

        self._ready_signal.connect(self._on_ready)
        self._result_signal.connect(self._on_result)

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def _log(self, text: str):
        lw = get_log_window(); lw.append(text)
        if not lw.isVisible():
            lw.show()

    def _acquire_image(self, config):
        """Full-resolution RGB for the current image (reuse Predict's if fresh)."""
        img_path = config["image_path"]
        cached, _mask = self.predict.last_context()
        if cached is not None and self.predict._last_img_path == img_path:
            return np.asarray(cached), img_path
        import cv2
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, img_path

    def _start(self):
        # Interactive prompting always uses the SAM + LoRA path, regardless of
        # the engine chosen in Predict — so build the full SAM config directly.
        try:
            config = self.predict._sam_config()
        except ValueError as e:
            self._status.setText(f"[ERROR] {e}  (needs a LoRA checkpoint + SAM backbone)")
            self._log(f"[ERROR] annotate: {e}")
            return
        try:
            image_rgb, img_path = self._acquire_image(config)
        except ValueError as e:
            self._status.setText(f"[ERROR] {e}"); return

        self._start_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._status.setText("Computing image embedding… (first click will be instant)")
        self._img_path = img_path

        from napari_app.interactive import InteractiveSession

        def run():
            try:
                sess = InteractiveSession(config, image_rgb)
                self._session = sess
                self._pending_image = image_rgb
                self._ready_signal.emit(True, f"{Path(img_path).name}  ·  ready")
            except Exception as e:
                import traceback
                self._ready_signal.emit(False, f"{e}\n{traceback.format_exc()}")

        threading.Thread(target=run, daemon=True).start()

    def _on_ready(self, ok: bool, message: str):
        self._progress.setVisible(False)
        if not ok:
            self._start_btn.setEnabled(True)
            self._status.setText(f"[ERROR] {message.splitlines()[0]}")
            self._log(f"[ERROR] interactive: {message}")
            return

        img = self._pending_image
        h, w = img.shape[:2]
        self._label_img = np.zeros((h, w), dtype=np.int32)
        self._active_id = 0
        self._reset_prompt()

        stem = Path(self._img_path).stem
        self._image_layer = self.viewer.add_image(img, name=f"{stem}_annotate")
        self._labels_layer = self.viewer.add_labels(
            self._label_img.copy(), name=f"{stem}_annotate_masks", opacity=0.6)
        self._labels_layer.contour = 1
        try:
            self._pts_layer = self.viewer.add_points(
                np.empty((0, 2)), name=f"{stem}_clicks", size=12, border_width=0)
        except Exception:
            self._pts_layer = None
        # keep the Labels layer active so the click callback (and manual brush) work
        try:
            self.viewer.layers.selection.active = self._labels_layer
        except Exception:
            pass
        self.viewer.reset_view()

        self._cb = self._make_callback()
        self.viewer.mouse_drag_callbacks.append(self._cb)

        self._status.setText(f"● live  ·  {message}")
        self._status.setStyleSheet(
            f"color:{SUCCESS}; font-size:11px; background:transparent;"
            f"font-family:{MONO};")
        for b in (self._stop_btn, self._clear_btn, self._undo_btn):
            b.setEnabled(True)
        self._update_count()

    def _stop(self):
        self._detach_callback()
        self._session = None
        self._start_btn.setEnabled(True)
        for b in (self._stop_btn, self._clear_btn, self._undo_btn):
            b.setEnabled(False)
        self._status.setText("Session ended. Labels layer kept for editing/training.")
        self._status.setStyleSheet(
            f"color:{DIM}; font-size:11px; background:transparent;"
            f"font-family:{MONO};")

    def _detach_callback(self):
        if self._cb is not None and self._cb in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.remove(self._cb)
        self._cb = None

    # ── Prompting ──────────────────────────────────────────────────────────────

    def _reset_prompt(self):
        self._points = []
        self._plabels = []
        self._last_low = None
        self._prompt_coords = []
        self._prompt_face = []
        self._refresh_points()

    def _refresh_points(self):
        if self._pts_layer is None:
            return
        try:
            if self._prompt_coords:
                self._pts_layer.data = np.asarray(self._prompt_coords, dtype=float)
                self._pts_layer.face_color = np.asarray(self._prompt_face, dtype=float)
            else:
                self._pts_layer.data = np.empty((0, 2))
        except Exception:
            pass

    def _make_callback(self):
        # Generator callback: only act on a genuine click, never on a drag/pan.
        def callback(viewer, event):
            if event.button != 1 or self._session is None:
                return
            mods = set(event.modifiers or ())
            if "Alt" in mods:
                return  # leave Alt free for panning / other tools
            start = event.position
            yield
            moved = 0.0
            while event.type == "mouse_move":
                p = event.position
                moved += abs(p[0] - start[0]) + abs(p[1] - start[1])
                yield
            if moved > 8 or self._busy:
                return  # it was a drag (pan) or we're busy — ignore

            try:
                dy, dx = self._image_layer.world_to_data(start)[:2]
            except Exception:
                return
            x, y = int(round(dx)), int(round(dy))
            h, w = self._label_img.shape
            if not (0 <= x < w and 0 <= y < h):
                return

            grow = "Shift" in mods
            carve = bool({"Control", "Meta"} & mods)
            if (grow or carve) and self._active_id > 0 and self._points:
                self._points.append((x, y))
                self._plabels.append(0 if carve else 1)
            else:
                self._active_id = int(self._label_img.max()) + 1
                self._points = [(x, y)]
                self._plabels = [1]
                self._last_low = None
                self._prompt_coords = []
                self._prompt_face = []

            self._prompt_coords.append([y, x])
            self._prompt_face.append([0.85, 0.15, 0.15, 1.0] if carve
                                     else [0.15, 0.95, 0.4, 1.0])
            self._refresh_points()
            self._run_predict()
        return callback

    def _run_predict(self):
        self._busy = True
        pts = list(self._points)
        lbls = list(self._plabels)
        low = self._last_low
        active = self._active_id

        def run():
            try:
                mask, low_res, score = self._session.predict(pts, lbls, mask_input=low)
                self._last_low = low_res
                self._result_signal.emit(mask, active, score)
            except Exception as e:
                import traceback
                self._log(f"[ERROR] segment: {e}\n{traceback.format_exc()}")
                self._result_signal.emit(None, active, 0.0)

        threading.Thread(target=run, daemon=True).start()

    def _on_result(self, mask, label_id: int, score: float):
        self._busy = False
        if mask is None:
            return
        # Overwrite this object's previous pixels, then paint the new mask.
        self._label_img[self._label_img == label_id] = 0
        self._label_img[mask] = label_id
        self._labels_layer.data = self._label_img
        self._labels_layer.refresh()
        self._labels_layer.selected_label = label_id
        self._update_count()
        self._status.setText(f"● live  ·  cell {label_id}  (score {score:.2f})")

    # ── Editing helpers ────────────────────────────────────────────────────────

    def _undo_last(self):
        if self._label_img is None or self._active_id <= 0:
            return
        self._label_img[self._label_img == self._active_id] = 0
        self._labels_layer.data = self._label_img
        self._labels_layer.refresh()
        self._active_id = int(self._label_img.max())
        self._reset_prompt()
        self._update_count()

    def _clear_all(self):
        if self._label_img is None:
            return
        self._label_img[:] = 0
        self._labels_layer.data = self._label_img
        self._labels_layer.refresh()
        self._active_id = 0
        self._reset_prompt()
        self._update_count()

    def _update_count(self):
        n = int(self._label_img.max()) if self._label_img is not None else 0
        # count distinct non-zero labels (ids can have gaps after undo)
        if self._label_img is not None and n > 0:
            n = int(np.count_nonzero(np.bincount(self._label_img.ravel())[1:]))
        self._count_lbl.setText(f"{n} cell{'s' if n != 1 else ''} annotated")
