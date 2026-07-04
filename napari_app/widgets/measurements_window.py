"""
Floating Measurements window — a spreadsheet-style view of per-cell
morphometry with an interactive feature histogram and CSV export.

Opened from the Predict panel's "Measurements" button. Kept as a single
reusable instance (like the log window) so re-running prediction refreshes
the same window instead of spawning new ones.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QAbstractItemView,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from napari_app.theme import (
    BG, FG, BORDER, TEXT, DIM, LABEL, ACCENT, SUCCESS, CONSOLE, WIDGET_SS,
    BTN_SECONDARY, BTN_PRIMARY,
)

_DLG = QFileDialog.Option.DontUseNativeDialog


class Histogram(QWidget):
    """Small distribution plot; pyqtgraph if present, else matplotlib."""

    def __init__(self):
        super().__init__()
        self._use_pg = False
        self.setMinimumHeight(150)
        try:
            import pyqtgraph as pg
            self._pg = pg
            self._plot = pg.PlotWidget(background=CONSOLE)
            self._plot.showGrid(x=False, y=True, alpha=0.15)
            for axis in ("bottom", "left"):
                self._plot.getAxis(axis).setTextPen(DIM)
                self._plot.getAxis(axis).setPen(BORDER)
            L = QVBoxLayout(); L.setContentsMargins(0, 0, 0, 0); L.addWidget(self._plot)
            self.setLayout(L)
            self._use_pg = True
        except Exception:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self._fig = Figure(figsize=(4, 1.6), dpi=90)
            self._fig.patch.set_facecolor(BG)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._fig)
            L = QVBoxLayout(); L.setContentsMargins(0, 0, 0, 0); L.addWidget(self._canvas)
            self.setLayout(L)

    def plot(self, values, label: str):
        import numpy as np
        vals = np.asarray([v for v in values], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        bins = max(6, min(40, int(np.sqrt(vals.size)) + 4))
        if self._use_pg:
            pg = self._pg
            y, x = np.histogram(vals, bins=bins)
            self._plot.clear()
            self._plot.addItem(pg.BarGraphItem(
                x0=x[:-1], x1=x[1:], height=y, brush=ACCENT, pen=pg.mkPen(BG, width=0.5)))
            med = float(np.median(vals))
            self._plot.addItem(pg.InfiniteLine(
                pos=med, angle=90, pen=pg.mkPen(SUCCESS, width=1.4, style=Qt.PenStyle.DashLine)))
            self._plot.setLabel("bottom", label, color=DIM)
            self._plot.setLabel("left", "count", color=DIM)
        else:
            self._ax.clear()
            self._ax.set_facecolor(CONSOLE)
            self._ax.hist(vals, bins=bins, color=ACCENT, edgecolor=BG, linewidth=0.4)
            self._ax.axvline(float(np.median(vals)), color=SUCCESS, ls="--", lw=1.3)
            self._ax.tick_params(colors=DIM, labelsize=8)
            self._ax.set_xlabel(label, color=DIM, fontsize=9)
            self._ax.set_ylabel("count", color=DIM, fontsize=9)
            for s in self._ax.spines.values():
                s.set_edgecolor(BORDER)
            self._fig.tight_layout(pad=0.5)
            self._canvas.draw_idle()


class MeasurementsWindow(QWidget):
    def __init__(self):
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("CellSeg1 — Measurements")
        self.resize(880, 560)
        self.setMinimumSize(480, 320)
        self.setStyleSheet(WIDGET_SS)
        self._result: dict[str, Any] | None = None
        self._source_name = ""

        root = QVBoxLayout()
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # Headline summary
        self._headline = QLabel("No measurements yet")
        self._headline.setStyleSheet(
            f"color:{TEXT}; font-size:15px; font-weight:600; background:transparent;")
        root.addWidget(self._headline)

        self._subline = QLabel("")
        self._subline.setStyleSheet(
            f"color:{LABEL}; font-size:11px; font-family:'Menlo','SF Mono',monospace;"
            f"background:transparent;")
        self._subline.setWordWrap(True)
        root.addWidget(self._subline)

        # Body: table (left) + histogram panel (right)
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
        feat_row = QHBoxLayout(); feat_row.setSpacing(8)
        lbl = QLabel("Distribution")
        lbl.setStyleSheet(f"color:{LABEL}; font-size:11px; font-weight:600; background:transparent;")
        feat_row.addWidget(lbl)
        self._feature_combo = QComboBox()
        self._feature_combo.currentIndexChanged.connect(self._redraw_hist)
        feat_row.addWidget(self._feature_combo, stretch=1)
        right.addLayout(feat_row)

        self._hist = Histogram()
        right.addWidget(self._hist)

        self._stat_lbl = QLabel("")
        self._stat_lbl.setStyleSheet(
            f"color:{DIM}; font-size:11px; font-family:'Menlo','SF Mono',monospace;"
            f"background:transparent;")
        self._stat_lbl.setWordWrap(True)
        right.addWidget(self._stat_lbl)
        right.addStretch()

        body.addLayout(right, stretch=2)
        root.addLayout(body)

        # Footer buttons
        foot = QHBoxLayout(); foot.setSpacing(8)
        foot.addStretch()
        copy_btn = QPushButton("Copy TSV")
        copy_btn.setStyleSheet(BTN_SECONDARY)
        copy_btn.clicked.connect(self._copy)
        foot.addWidget(copy_btn)
        csv_btn = QPushButton("Export CSV")
        csv_btn.setFixedHeight(30)
        csv_btn.setStyleSheet(BTN_PRIMARY)
        csv_btn.clicked.connect(self._export)
        foot.addWidget(csv_btn)
        root.addLayout(foot)

        self.setLayout(root)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_result(self, result: dict[str, Any], source_name: str = ""):
        self._result = result
        self._source_name = source_name
        cols = result["columns"]
        rows = result["rows"]

        # Headline
        from napari_app import analysis
        n = result["n_cells"]
        self._headline.setText(f"{n} cells  ·  {source_name}" if source_name else f"{n} cells")
        self._subline.setText(analysis.summary_line(result))

        # Table
        self._table.setSortingEnabled(False)
        self._table.clear()
        headers = [f"{label}\n{unit}" if unit else label for _k, label, unit in cols]
        self._table.setColumnCount(len(cols))
        self._table.setRowCount(len(rows))
        self._table.setHorizontalHeaderLabels(headers)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem()
                if c == 0:
                    item.setData(Qt.ItemDataRole.DisplayRole, int(val))
                else:
                    item.setData(Qt.ItemDataRole.EditRole, float(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(r, c, item)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSortingEnabled(True)

        # Feature combo (numeric columns only, skip id)
        self._feature_combo.blockSignals(True)
        self._feature_combo.clear()
        self._num_cols = []
        for idx, (key, label, unit) in enumerate(cols):
            if key == "cell_id":
                continue
            self._num_cols.append(idx)
            self._feature_combo.addItem(f"{label} ({unit})" if unit else label, idx)
        # Default to "Area" if present
        for i in range(self._feature_combo.count()):
            if self._feature_combo.itemText(i).startswith("Area"):
                self._feature_combo.setCurrentIndex(i); break
        self._feature_combo.blockSignals(False)
        self._redraw_hist()

    _placed = False

    def show_and_raise(self):
        if not self._placed:
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

    def _redraw_hist(self):
        if not self._result or not self._result["rows"]:
            return
        col_idx = self._feature_combo.currentData()
        if col_idx is None:
            return
        key, label, unit = self._result["columns"][col_idx]
        values = [row[col_idx] for row in self._result["rows"]]
        self._hist.plot(values, f"{label} ({unit})" if unit else label)
        s = self._result["summary"].get(key, {})
        if s:
            self._stat_lbl.setText(
                f"mean {s['mean']:.3g}   median {s['median']:.3g}   "
                f"std {s['std']:.3g}\nmin {s['min']:.3g}   max {s['max']:.3g}"
                f"   n={self._result['n_cells']}")

    def _copy(self):
        if not self._result:
            return
        from PyQt6.QtWidgets import QApplication
        text = self._result_as_tsv()
        QApplication.clipboard().setText(text)

    def _result_as_tsv(self) -> str:
        cols = self._result["columns"]
        lines = ["\t".join(f"{label} ({unit})" if unit else label
                           for _k, label, unit in cols)]
        for row in self._result["rows"]:
            lines.append("\t".join(str(v) for v in row))
        return "\n".join(lines)

    def _export(self):
        if not self._result:
            return
        from napari_app import analysis
        from project_root import STORAGE_DIR
        (STORAGE_DIR / "predict_masks").mkdir(parents=True, exist_ok=True)
        stem = Path(self._source_name).stem or "measurements"
        default = str(STORAGE_DIR / "predict_masks" / f"{stem}_measurements.csv")
        p, _ = QFileDialog.getSaveFileName(
            self, "Export measurements", default, "CSV (*.csv)", options=_DLG)
        if not p:
            return
        with open(p, "w", newline="") as f:
            f.write(analysis.rows_as_csv(self._result))


_TABLE_SS = f"""
QTableWidget {{
    background: {CONSOLE};
    alternate-background-color: #12182a;
    color: {TEXT};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 6px;
    font-family: 'Menlo','SF Mono',monospace;
    font-size: 11px;
}}
QTableWidget::item:selected {{ background: rgba(77,143,255,0.28); color: {TEXT}; }}
QHeaderView::section {{
    background: {FG};
    color: {LABEL};
    padding: 4px 6px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
    font-size: 10px;
}}
QTableCornerButton::section {{ background: {FG}; border: none; }}
"""


_instance: MeasurementsWindow | None = None


def get_measurements_window() -> MeasurementsWindow:
    global _instance
    if _instance is None:
        _instance = MeasurementsWindow()
    return _instance
