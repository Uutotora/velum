"""
Cohort window — population-level view of a batch run.

Shows a per-image summary table, a pooled feature-distribution histogram across
every cell in the cohort, and population statistics. Complements the single-
image Measurements window.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QAbstractItemView,
)
from PyQt6.QtCore import Qt

from napari_app.theme import (
    BG, FG, BORDER, TEXT, DIM, LABEL, ACCENT, SUCCESS, CONSOLE, MONO,
    WIDGET_SS, BTN_SECONDARY, BTN_PRIMARY,
)
from napari_app.widgets.measurements_window import Histogram, _TABLE_SS

_DLG = QFileDialog.Option.DontUseNativeDialog

_FEATURES = [
    ("area", "Area"), ("diameter", "Eq. diameter"),
    ("circularity", "Circularity"), ("eccentricity", "Eccentricity"),
    ("solidity", "Solidity"),
]


class CohortWindow(QWidget):
    def __init__(self):
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("CellSeg1 — Cohort analysis")
        self.resize(920, 600)
        self.setMinimumSize(520, 360)
        self.setStyleSheet(WIDGET_SS)
        self._records = []
        self._out_dir = ""

        root = QVBoxLayout(); root.setContentsMargins(14, 12, 14, 12); root.setSpacing(10)

        self._headline = QLabel("No cohort yet")
        self._headline.setStyleSheet(
            f"color:{TEXT}; font-size:15px; font-weight:600; background:transparent;")
        root.addWidget(self._headline)

        self._pop = QLabel("")
        self._pop.setStyleSheet(
            f"color:{LABEL}; font-size:11px; font-family:{MONO};"
            f"background:transparent;")
        self._pop.setWordWrap(True)
        root.addWidget(self._pop)

        body = QHBoxLayout(); body.setSpacing(12)
        self._table = QTableWidget()
        self._table.setStyleSheet(_TABLE_SS)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        body.addWidget(self._table, stretch=3)

        right = QVBoxLayout(); right.setSpacing(8)
        frow = QHBoxLayout(); frow.setSpacing(8)
        flab = QLabel("Pooled distribution")
        flab.setStyleSheet(f"color:{LABEL}; font-size:11px; font-weight:600; background:transparent;")
        frow.addWidget(flab)
        self._feature = QComboBox()
        for key, label in _FEATURES:
            self._feature.addItem(label, key)
        self._feature.currentIndexChanged.connect(self._redraw)
        frow.addWidget(self._feature, stretch=1)
        right.addLayout(frow)
        self._hist = Histogram()
        right.addWidget(self._hist)
        self._feat_stats = QLabel("")
        self._feat_stats.setStyleSheet(
            f"color:{DIM}; font-size:11px; font-family:{MONO};"
            f"background:transparent;")
        self._feat_stats.setWordWrap(True)
        right.addWidget(self._feat_stats)
        right.addStretch()
        body.addLayout(right, stretch=2)
        root.addLayout(body)

        foot = QHBoxLayout(); foot.setSpacing(8); foot.addStretch()
        pc = QPushButton("Export per-cell CSV"); pc.setStyleSheet(BTN_SECONDARY)
        pc.clicked.connect(self._export_cells); foot.addWidget(pc)
        ps = QPushButton("Export summary CSV"); ps.setFixedHeight(30); ps.setStyleSheet(BTN_PRIMARY)
        ps.clicked.connect(self._export_summary); foot.addWidget(ps)
        root.addLayout(foot)
        self.setLayout(root)

    # ── Public ─────────────────────────────────────────────────────────────────

    def set_records(self, records, out_dir=""):
        self._records = records
        self._out_dir = out_dir
        from napari_app import cohort
        pop = cohort.population_stats(records)
        self._headline.setText(
            f"{pop['total_cells']} cells across {pop['n_images']} images")
        feats = pop.get("features", {})
        area = feats.get("area", {})
        diam = feats.get("diameter", {})
        self._pop.setText(
            f"median diameter {diam.get('median', 0):.2f}   "
            f"mean area {area.get('mean', 0):.1f}   "
            f"(pooled over all cells)")

        cols, rows = cohort.per_image_summary(records)
        self._table.setSortingEnabled(False)
        self._table.clear()
        self._table.setColumnCount(len(cols))
        self._table.setRowCount(len(rows))
        self._table.setHorizontalHeaderLabels(cols)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem()
                if isinstance(val, str):
                    item.setData(Qt.ItemDataRole.DisplayRole, val)
                else:
                    item.setData(Qt.ItemDataRole.EditRole, float(val))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(r, c, item)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSortingEnabled(True)
        self._redraw()

    def show_and_raise(self):
        if not getattr(self, "_placed", False):
            self._placed = True
            try:
                from PyQt6.QtGui import QGuiApplication
                geo = QGuiApplication.primaryScreen().availableGeometry()
                self.move(geo.center().x() - self.width() // 2,
                          geo.center().y() - self.height() // 2)
            except Exception:
                pass
        self.show(); self.raise_(); self.activateWindow()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _redraw(self):
        if not self._records:
            return
        from napari_app import cohort
        import numpy as np
        key = self._feature.currentData()
        label = self._feature.currentText()
        vals = cohort.pooled_values(self._records, key)
        if not vals:
            return
        self._hist.plot(vals, label)
        a = np.asarray(vals, dtype=float)
        self._feat_stats.setText(
            f"n={a.size}   mean {a.mean():.3g}   median {np.median(a):.3g}   "
            f"std {a.std():.3g}")

    def _export_cells(self):
        self._export(kind="cells")

    def _export_summary(self):
        self._export(kind="summary")

    def _export(self, kind):
        if not self._records:
            return
        from napari_app import cohort
        default = str(Path(self._out_dir or ".") /
                      ("cohort_measurements.csv" if kind == "cells" else "cohort_summary.csv"))
        p, _ = QFileDialog.getSaveFileName(self, "Export cohort CSV", default,
                                           "CSV (*.csv)", options=_DLG)
        if not p:
            return
        if kind == "cells":
            h, r = cohort.per_cell_long(self._records)
        else:
            h, r = cohort.per_image_summary(self._records)
        cohort._write_csv(p, h, r)


_instance: CohortWindow | None = None


def get_cohort_window() -> CohortWindow:
    global _instance
    if _instance is None:
        _instance = CohortWindow()
    return _instance
