"""Velum — the Cell Population Analytics view.

The segmentation engines already measure *every* labelled cell (area,
diameter, circularity, aspect ratio, intensity … — see
``velum_core.analysis.compute_measurements``), but the Segment inspector only
ever surfaced three rolled-up numbers. This module turns that per-cell table
into what a "segment, **measure and compare** cells" product should show: the
shape of the whole population — a distribution explorer with per-metric
histograms, mean/median guides, summary stats, and a small-multiples overview
you can click through.

Everything here is fed a plain measurement ``dict`` (the exact shape
``compute_measurements`` returns) plus a theme ``dict`` and two callbacks —
no Qt controllers, no torch, no engine. The binning + metric selection is
pure functions (``histogram`` / ``plottable_metrics`` / ``metric_values`` /
``metric_stats``), unit-tested without a display in
``studio/tests/test_cell_analytics.py``.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import (QPainter, QColor, QPen, QLinearGradient, QFont,
                         QPainterPath)
from PyQt6.QtWidgets import (QWidget, QFrame, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QLabel)

from studio import theme, icons
from studio.components import (label, hline, soft_shadow, PillButton, IconButton,
                               SelectBox, StatTile, GroupLabel)

# Columns that aren't a distribution anyone wants to *see* the shape of.
_NON_METRIC = {"cell_id", "centroid_x", "centroid_y"}
# Dimensionless kinds → no unit suffix on stat tiles.
_UNITLESS = {"", "ratio", "angle", "coord", "id"}
# The metrics the overview strip leads with, best first, when present.
_FEATURED = ["area", "diameter", "circularity", "aspect_ratio",
             "solidity", "eccentricity", "mean_intensity", "perimeter"]


# ── pure logic (unit-tested headless) ────────────────────────────────────────
def plottable_metrics(result: dict[str, Any]) -> list[tuple[str, str, str]]:
    """``(key, label, unit)`` for every numeric column worth a histogram —
    everything except the cell id and centroid coordinates."""
    return [(k, lab, unit) for k, lab, unit in result.get("columns", [])
            if k not in _NON_METRIC]


def _column_index(result: dict[str, Any], key: str) -> int:
    for i, (k, _lab, _unit) in enumerate(result.get("columns", [])):
        if k == key:
            return i
    return -1


def metric_values(result: dict[str, Any], key: str) -> list[float]:
    """Every cell's value for column ``key`` (empty list if unknown)."""
    idx = _column_index(result, key)
    if idx < 0:
        return []
    out: list[float] = []
    for row in result.get("rows", []):
        if idx < len(row) and row[idx] is not None:
            out.append(float(row[idx]))
    return out


def histogram(values: list[float], n_bins: int = 18) -> tuple[list[float], list[int]]:
    """``(edges, counts)`` — ``edges`` has ``len(counts)+1`` entries. Robust to
    the awkward cases: empty input, and a population where every cell has the
    same value (one centred bin rather than a divide-by-zero)."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return [0.0, 1.0], [0]
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-12:
        return [mn - 0.5, mn + 0.5], [len(vals)]
    n = max(1, int(n_bins))
    width = (mx - mn) / n
    edges = [mn + i * width for i in range(n + 1)]
    counts = [0] * n
    for v in vals:
        idx = int((v - mn) / width)
        if idx >= n:
            idx = n - 1
        elif idx < 0:
            idx = 0
        counts[idx] += 1
    return edges, counts


def metric_stats(result: dict[str, Any], key: str) -> dict[str, float]:
    """Rolled-up stats for ``key`` — the engine's own ``summary`` block plus a
    derived coefficient of variation (std/mean, %). Recomputed from the raw
    rows if a summary entry is missing."""
    s = dict(result.get("summary", {}).get(key, {}))
    if not s:
        vals = metric_values(result, key)
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            ordered = sorted(vals)
            mid = len(ordered) // 2
            median = (ordered[mid] if len(ordered) % 2
                      else (ordered[mid - 1] + ordered[mid]) / 2)
            s = {"mean": mean, "median": median, "std": var ** 0.5,
                 "min": ordered[0], "max": ordered[-1]}
    mean = s.get("mean", 0.0)
    s["cv"] = (s.get("std", 0.0) / abs(mean) * 100.0) if mean else 0.0
    return s


def _fmt(v: float) -> str:
    a = abs(v)
    if a == 0:
        return "0"
    if a < 1:
        return f"{v:.3f}"
    if a < 10:
        return f"{v:.2f}"
    if a < 1000:
        return f"{v:.1f}"
    return f"{v:,.0f}"


def _round_top_rect(r: QRectF, radius: float) -> QPainterPath:
    rad = max(0.0, min(radius, r.width() / 2, r.height()))
    path = QPainterPath()
    path.moveTo(r.left(), r.bottom())
    path.lineTo(r.left(), r.top() + rad)
    path.quadTo(r.left(), r.top(), r.left() + rad, r.top())
    path.lineTo(r.right() - rad, r.top())
    path.quadTo(r.right(), r.top(), r.right(), r.top() + rad)
    path.lineTo(r.right(), r.bottom())
    path.closeSubpath()
    return path


# ── chart widgets ────────────────────────────────────────────────────────────
class Histogram(QWidget):
    """A gradient-filled distribution histogram. In ``big`` mode it also draws
    count gridlines, x-axis value labels, and dashed mean/median guide lines."""

    def __init__(self, edges: list[float], counts: list[int], color: str,
                 t: dict, *, big: bool = True,
                 mean: Optional[float] = None, median: Optional[float] = None):
        super().__init__()
        self._edges = edges
        self._counts = counts
        self._color = color
        self._t = t
        self._big = big
        self._mean = mean
        self._median = median
        self.setMinimumHeight(196 if big else 42)

    def paintEvent(self, e):
        counts = self._counts
        if not counts:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        t = self._t
        big = self._big
        W, H = self.width(), self.height()
        lpad = 34 if big else 2
        rpad = 12 if big else 2
        tpad = 10 if big else 2
        bpad = 22 if big else 2
        plot_w = max(1.0, W - lpad - rpad)
        plot_h = max(1.0, H - tpad - bpad)
        peak = max(counts) or 1

        if big:
            grid = QPen(QColor(t["border"]), 1)
            grid.setDashPattern([2, 4])
            lab_font = QFont(); lab_font.setPointSizeF(7.6)
            for k in range(3):
                frac = k / 2
                y = tpad + plot_h * frac
                p.setPen(grid)
                p.drawLine(int(lpad), int(y), int(W - rpad), int(y))
                p.setPen(QColor(t["text_muted"]))
                p.setFont(lab_font)
                p.drawText(QRectF(0, y - 8, lpad - 6, 16),
                           int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                           str(int(round(peak * (1 - frac)))))

        n = len(counts)
        slot = plot_w / n
        gap = 2.0 if big else 1.0
        base = QColor(self._color)
        for i, c in enumerate(counts):
            bh = plot_h * (c / peak)
            x = lpad + slot * i + gap / 2
            bw = max(1.0, slot - gap)
            y = tpad + plot_h - bh
            grad = QLinearGradient(0, y, 0, tpad + plot_h)
            top = QColor(base); top.setAlpha(240 if big else 190)
            bot = QColor(base); bot.setAlpha(95 if big else 70)
            grad.setColorAt(0.0, top)
            grad.setColorAt(1.0, bot)
            p.fillPath(_round_top_rect(QRectF(x, y, bw, bh), 3.0 if big else 1.5), grad)

        if big:
            p.setPen(QPen(QColor(t["border_strong"]), 1))
            p.drawLine(int(lpad), int(tpad + plot_h), int(W - rpad), int(tpad + plot_h))
            e0, e1 = self._edges[0], self._edges[-1]
            span = (e1 - e0) or 1.0

            def X(v):
                return lpad + plot_w * (v - e0) / span

            # x-axis min / max value labels
            xlab = QFont(); xlab.setPointSizeF(7.6)
            p.setFont(xlab)
            p.setPen(QColor(t["text_muted"]))
            p.drawText(QRectF(lpad, tpad + plot_h + 3, 70, 14),
                       int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), _fmt(e0))
            p.drawText(QRectF(W - rpad - 70, tpad + plot_h + 3, 70, 14),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), _fmt(e1))

            gl_font = QFont(); gl_font.setPointSizeF(7.4); gl_font.setBold(True)
            # Stack the two labels on separate rows so a near-coincident
            # mean/median (common in a tight population) never overprints.
            for r_idx, (val, ckey, name) in enumerate((
                    (self._mean, "primary", "mean"),
                    (self._median, "signal", "median"))):
                if val is None or not (e0 <= val <= e1):
                    continue
                gx = X(val)
                gpen = QPen(QColor(t[ckey]), 1.4)
                gpen.setDashPattern([3, 3])
                p.setPen(gpen)
                p.drawLine(int(gx), int(tpad), int(gx), int(tpad + plot_h))
                p.setFont(gl_font)
                p.setPen(QColor(t[ckey]))
                ly = tpad + 1 + r_idx * 12
                left = gx + 3 < W - rpad - 46
                rect = (QRectF(gx + 3, ly, 48, 12) if left
                        else QRectF(gx - 51, ly, 48, 12))
                align = (Qt.AlignmentFlag.AlignLeft if left else Qt.AlignmentFlag.AlignRight)
                p.drawText(rect, int(align | Qt.AlignmentFlag.AlignVCenter), name)
        p.end()


class _MiniMetricCard(QFrame):
    """One clickable tile in the overview strip: metric name, its little
    histogram, and its median — selects that metric in the main chart."""

    def __init__(self, key: str, name: str, median_str: str, edges, counts,
                 color: str, t: dict, selected: bool,
                 on_click: Callable[[str], None]):
        super().__init__()
        self._key = key
        self._on_click = on_click
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("MiniMetricCard")
        bg = t["surface2"] if selected else t["inset"]
        bd = t["primary"] if selected else t["border"]
        self.setStyleSheet(
            f"QFrame#MiniMetricCard{{background:{bg}; border:1px solid {bd}; border-radius:10px;}}"
            f"QFrame#MiniMetricCard:hover{{border-color:{t['border_strong']};}}")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(3)
        top = QHBoxLayout()
        top.setSpacing(6)
        nm = label(name, 11, t["text"] if selected else t["text_subtle"], 600)
        top.addWidget(nm)
        top.addStretch(1)
        med = QLabel(median_str)
        med.setStyleSheet(
            f"color:{t['text_muted']}; font-family:{theme.MONO}; font-size:10.5px; background:transparent;")
        top.addWidget(med)
        v.addLayout(top)
        v.addWidget(Histogram(edges, counts, color, t, big=False))

    def mousePressEvent(self, e) -> None:
        self._on_click(self._key)
        super().mousePressEvent(e)


# ── panel + dialog ───────────────────────────────────────────────────────────
class CellAnalyticsPanel(QFrame):
    """The full analytics card: header · metric selector · distribution
    histogram · summary tiles · clickable overview strip · footer actions."""

    def __init__(self, result: dict[str, Any], t: dict, *,
                 image_name: str = "", on_export: Optional[Callable[[], None]] = None,
                 on_close: Optional[Callable[[], None]] = None):
        super().__init__()
        self._result = result
        self._t = t
        self._image_name = image_name
        self._on_export = on_export
        self._on_close = on_close
        self._metrics = plottable_metrics(result)
        self._label_by_key = {k: lab for k, lab, _u in self._metrics}
        self._unit_by_key = {k: u for k, _lab, u in self._metrics}
        keys = [k for k, _l, _u in self._metrics]
        self._metric_key = next((k for k in _FEATURED if k in keys),
                                keys[0] if keys else "")
        self._mini_cards: dict[str, _MiniMetricCard] = {}

        self.setFixedWidth(760)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("CellAnalyticsPanel")
        self.setStyleSheet(
            f"QFrame#CellAnalyticsPanel{{background:{t['surface']}; "
            f"border:1px solid {t['border_strong']}; border-radius:16px;}}")
        soft_shadow(self, 34, 46, 12)

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(24, 20, 24, 18)
        self._root.setSpacing(14)
        self._build()

    # -- construction --
    def _build(self) -> None:
        t = self._t
        n_cells = int(self._result.get("n_cells", 0))

        header = QHBoxLayout()
        header.setSpacing(12)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_box.addWidget(label("Cell population", 17, t["text"], 600))
        sub = f"{n_cells:,} cells measured"
        if self._image_name:
            sub += f"  ·  {self._image_name}"
        title_box.addWidget(label(sub, 12, t["text_muted"]))
        header.addLayout(title_box)
        header.addStretch(1)
        close_btn = IconButton("close", t, 30, "Close", self._close)
        header.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignTop)
        self._root.addLayout(header)

        if n_cells == 0 or not self._metrics:
            empty = label("No cells were measured for this image yet — run "
                          "segmentation to explore the population.", 12.5, t["text_muted"])
            empty.setWordWrap(True)
            self._root.addWidget(empty)
            self._root.addLayout(self._footer())
            return

        self._root.addWidget(GroupLabel("Distribution", t))
        sel = SelectBox(self._label_by_key[self._metric_key], t,
                        options=[lab for _k, lab, _u in self._metrics],
                        on_select=self._on_metric_label)
        self._selectbox = sel
        self._root.addWidget(sel)

        # dynamic region (histogram + stat tiles), rebuilt on metric change
        self._dyn = QVBoxLayout()
        self._dyn.setSpacing(12)
        self._root.addLayout(self._dyn)
        self._render_metric()

        self._root.addWidget(hline(t))
        self._root.addWidget(GroupLabel("Overview · click a metric to explore", t))
        self._root.addWidget(self._overview_strip())
        self._root.addLayout(self._footer())

    def _footer(self) -> QHBoxLayout:
        t = self._t
        row = QHBoxLayout()
        hint = label("Every measurement is exportable as per-cell CSV.", 11, t["text_muted"])
        row.addWidget(hint)
        row.addStretch(1)
        if self._on_export is not None:
            exp = PillButton("Export CSV", t, "ghost", "csv", small=True)
            exp.clicked.connect(self._export)
            row.addWidget(exp)
        done = PillButton("Done", t, "primary", small=True)
        done.clicked.connect(self._close)
        row.addWidget(done)
        return row

    # -- dynamic metric region --
    def _render_metric(self) -> None:
        t = self._t
        while self._dyn.count():
            item = self._dyn.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                _clear(item.layout())

        key = self._metric_key
        vals = metric_values(self._result, key)
        edges, counts = histogram(vals)
        stats = metric_stats(self._result, key)
        unit = self._unit_by_key.get(key, "")
        color = t["signal"]

        chart = Histogram(edges, counts, color, t, big=True,
                          mean=stats.get("mean"), median=stats.get("median"))
        self._dyn.addWidget(chart)

        show_unit = unit if unit not in _UNITLESS else ""
        tiles = QGridLayout()
        tiles.setSpacing(6)
        spec = [
            (_fmt(stats.get("mean", 0.0)), show_unit, "MEAN"),
            (_fmt(stats.get("median", 0.0)), show_unit, "MEDIAN"),
            (_fmt(stats.get("std", 0.0)), show_unit, "STD DEV"),
            (f"{stats.get('cv', 0.0):.0f}", "%", "CV"),
            (_fmt(stats.get("min", 0.0)), show_unit, "MIN"),
            (_fmt(stats.get("max", 0.0)), show_unit, "MAX"),
        ]
        for i, (val, u, cap) in enumerate(spec):
            tiles.addWidget(StatTile(val, u, cap, t), 0, i)
        wrap = QWidget()
        wrap.setLayout(tiles)
        self._dyn.addWidget(wrap)

    # -- overview --
    def _overview_strip(self) -> QWidget:
        t = self._t
        keys = [k for k, _l, _u in self._metrics]
        featured = [k for k in _FEATURED if k in keys]
        for k in keys:  # top up with anything else, capped
            if k not in featured:
                featured.append(k)
        featured = featured[:6]

        wrap = QWidget()
        grid = QGridLayout(wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        self._mini_cards.clear()
        for i, key in enumerate(featured):
            vals = metric_values(self._result, key)
            edges, counts = histogram(vals, n_bins=14)
            stats = metric_stats(self._result, key)
            card = _MiniMetricCard(
                key, self._label_by_key.get(key, key),
                _fmt(stats.get("median", 0.0)), edges, counts,
                t["primary"], t, selected=(key == self._metric_key),
                on_click=self._select_metric)
            self._mini_cards[key] = card
            grid.addWidget(card, i // 3, i % 3)
        return wrap

    # -- interaction --
    def _on_metric_label(self, lab: str) -> None:
        for k, klabel, _u in self._metrics:
            if klabel == lab:
                self._select_metric(k)
                return

    def _select_metric(self, key: str) -> None:
        if key == self._metric_key or key not in self._label_by_key:
            return
        prev = self._metric_key
        self._metric_key = key
        self._selectbox.set_text(self._label_by_key[key])
        self._render_metric()
        for k in (prev, key):
            card = self._mini_cards.get(k)
            if card is not None:
                self._restyle_mini(card, selected=(k == key))

    def _restyle_mini(self, card: _MiniMetricCard, selected: bool) -> None:
        t = self._t
        bg = t["surface2"] if selected else t["inset"]
        bd = t["primary"] if selected else t["border"]
        card.setStyleSheet(
            f"QFrame#MiniMetricCard{{background:{bg}; border:1px solid {bd}; border-radius:10px;}}"
            f"QFrame#MiniMetricCard:hover{{border-color:{t['border_strong']};}}")

    def _export(self) -> None:
        if self._on_export is not None:
            self._on_export()

    def _close(self) -> None:
        if self._on_close is not None:
            self._on_close()


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())


class CellAnalyticsDialog(QWidget):
    """Scrim-backed, centred modal wrapping :class:`CellAnalyticsPanel` — same
    construction + self-disposing lifecycle as ``project_dialogs.ConfirmDialog``
    (built fresh per open, deletes itself on hide, click-outside closes)."""

    def __init__(self, parent: QWidget, t: dict, result: dict[str, Any], *,
                 image_name: str = "", on_export: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self._t = t
        self.setStyleSheet(f"background:{theme.SCRIM};")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 54, 0, 40)
        outer.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self._panel = CellAnalyticsPanel(
            result, t, image_name=image_name, on_export=on_export, on_close=self.hide)
        outer.addWidget(self._panel)
        self.hide()

    def place(self) -> None:
        p = self.parentWidget()
        if p:
            self.setGeometry(0, 0, p.width(), p.height())

    def open(self) -> None:
        self.place()
        self.show()
        self.raise_()

    def mousePressEvent(self, e) -> None:
        child = self.childAt(e.position().toPoint())
        if child is None:
            self.hide()
        super().mousePressEvent(e)

    def hideEvent(self, e) -> None:
        super().hideEvent(e)
        self.deleteLater()
