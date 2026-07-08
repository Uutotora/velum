"""CellSeg1 — application shell: a permanent icon-only navigation rail.

A clean activity-bar (VS Code / Linear style): a slim vertical strip of icons,
no labels, no expansion. The active section's icon "lights up" turquoise with a
soft glow and a rounded outline; a minimal ring mark shimmers at the top and a
single status dot sits at the bottom. Only a right-edge separator — no internal
horizontal rules to misalign with the content.

No business logic — the five tab widgets are the same instances as before,
hosted in a QStackedWidget.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QSize, QRect, QPropertyAnimation, QEasingCurve
from PyQt6.QtWidgets import (
    QWidget, QFrame, QHBoxLayout, QVBoxLayout, QLabel, QToolButton,
    QStackedWidget, QButtonGroup, QSizePolicy,
)

import device_utils
from napari_app import icons
from napari_app.motion import fade_in, pulse, glow, clear_effect
from napari_app.theme import (
    BG_APP, BG, FG, BORDER, TEXT, LABEL, DIM, ACCENT, TEAL, TEAL_SOFT, TEAL_LINE,
    SUCCESS,
)

RAIL_W = 58

_NAV_SS = f"""
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
}}
QToolButton:hover {{ background: {FG}; }}
QToolButton:checked {{
    background: {TEAL_SOFT};
    border: 1px solid {TEAL_LINE};
}}
"""


class _Rail(QFrame):
    """Fixed-width, icon-only navigation rail."""

    def __init__(self, on_select):
        super().__init__()
        self._on_select = on_select
        self.setObjectName("Rail")
        self.setFixedWidth(RAIL_W)
        self.setStyleSheet(
            f"#Rail {{ background: {BG_APP}; border-right: 1px solid {BORDER}; }}")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # Sliding active indicator — a teal bar pinned to the left edge that
        # animates to the selected icon.
        self._active_idx = 0
        self._indicator = QFrame(self)
        self._indicator.setObjectName("RailIndicator")
        self._indicator.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._indicator.setStyleSheet(
            f"#RailIndicator {{ background:{TEAL};"
            f" border-top-right-radius:2px; border-bottom-right-radius:2px; }}")
        self._indicator.setFixedWidth(3)
        self._indicator.hide()

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 14, 0, 16)
        v.setSpacing(4)

        # ── brand: a thin ring that gently shimmers ──
        self._brand = QLabel()
        self._brand.setPixmap(icons.pixmap("ring", ACCENT, 22, stroke=1.5))
        self._brand.setFixedSize(RAIL_W, 30)
        self._brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._brand.setToolTip("CellSeg1")
        pulse(self._brand, duration=2600, lo=0.5, hi=1.0)
        v.addWidget(self._brand)
        v.addSpacing(10)

        # ── nav icons ──
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self._buttons: list[tuple[QToolButton, str]] = []
        self._nav_wrap = v  # nav buttons are appended here

        v.addStretch()

        # ── status dot ──
        self._dot = QLabel()
        self._dot.setFixedSize(RAIL_W, 12)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dot.setStyleSheet("background:transparent;")
        self._dot.setToolTip(_device_label())
        self._dot.setPixmap(_dot_pixmap(SUCCESS))
        pulse(self._dot, duration=1900, lo=0.35, hi=1.0)
        v.addWidget(self._dot)

    # nav construction --------------------------------------------------------
    def add_nav(self, icon_name: str, label: str, idx: int) -> QToolButton:
        b = QToolButton()
        b.setIcon(icons.icon(icon_name, LABEL, 20))
        b.setIconSize(QSize(20, 20))
        b.setCheckable(True)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFixedSize(40, 40)
        b.setStyleSheet(_NAV_SS)
        b.setToolTip(label)
        b.clicked.connect(lambda _=False, i=idx: self._on_select(i))
        self.group.addButton(b)
        # insert before the stretch (which is the second-to-last item)
        self._nav_wrap.insertWidget(self._nav_wrap.count() - 2, b,
                                    alignment=Qt.AlignmentFlag.AlignHCenter)
        self._buttons.append((b, icon_name))
        return b

    def select(self, idx: int):
        self._active_idx = idx
        for i, (b, icon_name) in enumerate(self._buttons):
            active = i == idx
            b.setChecked(active)
            b.setIcon(icons.icon(icon_name, TEAL if active else LABEL, 20))
            if active:
                glow(b, TEAL, blur=16)
            else:
                clear_effect(b)
        self._place_indicator(animate=True)

    def _place_indicator(self, animate: bool = True):
        if not (0 <= self._active_idx < len(self._buttons)):
            return
        b = self._buttons[self._active_idx][0]
        if b.height() <= 1:
            return  # layout not settled yet
        bar_h = 22
        y = b.y() + (b.height() - bar_h) // 2
        target = QRect(0, y, 3, bar_h)
        self._indicator.show()
        self._indicator.raise_()
        cur = self._indicator.geometry()
        if animate and cur.height() > 1 and cur != target:
            anim = QPropertyAnimation(self._indicator, b"geometry", self)
            anim.setDuration(220)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.setStartValue(cur)
            anim.setEndValue(target)
            anim.start()
            self._ind_anim = anim
        else:
            self._indicator.setGeometry(target)

    def resizeEvent(self, e):
        self._place_indicator(animate=False)
        super().resizeEvent(e)

    def showEvent(self, e):
        super().showEvent(e)
        self._place_indicator(animate=False)

    def set_status(self, color: str = SUCCESS, tip: str = ""):
        self._dot.setPixmap(_dot_pixmap(color))
        if tip:
            self._dot.setToolTip(tip)


class Shell(QWidget):
    """Icon rail + stacked content area (no overlay, no animation of width)."""

    def __init__(self, pages):
        super().__init__()
        self.setObjectName("Shell")
        self.setStyleSheet(f"#Shell {{ background:{BG}; }}")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background:{BG};")
        self.rail = _Rail(on_select=self._select)

        for i, (icon_name, label, widget) in enumerate(pages):
            self.stack.addWidget(widget)
            self.rail.add_nav(icon_name, label, i)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.rail)
        lay.addWidget(self.stack, stretch=1)

        self.rail.select(0)
        self.stack.setCurrentIndex(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _select(self, idx: int):
        if self.stack.currentIndex() != idx:
            self.stack.setCurrentIndex(idx)
            fade_in(self.stack.currentWidget(), 200)
        self.rail.select(idx)

    def set_status(self, color=SUCCESS, tip=""):
        self.rail.set_status(color, tip)


def _dot_pixmap(color: str):
    """A crisp filled status dot."""
    from PyQt6.QtGui import QPixmap, QPainter, QColor, QBrush
    dpr = 2.0
    size = 9
    px = QPixmap(int(size * dpr), int(size * dpr))
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(color)))
    p.drawEllipse(0, 0, size, size)
    p.end()
    return px


def _device_label() -> str:
    """Best-effort, read-only compute-device string for the status tooltip.

    Checks CUDA is actually usable, not just present -- see device_utils'
    docstring for why (a GPU torch.cuda.is_available() calls "available" but
    ships no kernels for still reports "cuda" without this check).
    """
    try:
        import torch
        if torch.cuda.is_available() and device_utils.is_usable(torch):
            return "cuda · ready"
        if torch.backends.mps.is_available():
            return "mps · ready"
    except Exception:
        pass
    return "cpu · ready"
