"""Velum — custom window chrome (frameless title bar).

Removes the native OS title bar (the grey strip) and replaces it with our own
dark bar carrying the app's own traffic-light controls, exactly like the
north-star mockup — the single biggest "this is a real product, not a Qt
window" signal. Frameless + our chrome is also portable (macOS + Linux) and
needs no extra dependency, unlike the pyobjc native-title-bar route.

Window move/resize are delegated to the platform via Qt's
``QWindow.startSystemMove`` / ``startSystemResize`` (native behaviour: snapping,
multi-monitor, edge magnetism) plus corner ``QSizeGrip``s — no hand-rolled
geometry maths. Everything is constructible headless (only a ``QApplication``).
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QToolButton, QSizeGrip, QSizePolicy,
)

from studio import icons
from studio import theme

# The macOS traffic-light trio, in the native order + colours.
_LIGHTS = [
    ("close", "#ff5f57", "#e0443e", "×"),   # ×
    ("min",   "#febc2e", "#dea123", "−"),   # −
    ("zoom",  "#28c840", "#1aad2f", "+"),
]


class _TrafficButton(QToolButton):
    """A round window-control 'light' that reveals its glyph on hover."""

    def __init__(self, color: str, hover: str, glyph: str, on_click: Callable[[], None]):
        super().__init__()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(13, 13)
        self.setText(glyph)
        self.clicked.connect(lambda: on_click())
        self.setStyleSheet(
            f"QToolButton{{background:{color}; border:none; border-radius:6px;"
            f" color:rgba(0,0,0,0); font-size:10px; font-weight:700; padding-bottom:1px;}}"
            f"QToolButton:hover{{background:{hover}; color:rgba(0,0,0,0.55);}}")


class TitleBar(QWidget):
    """The app's own draggable title bar: traffic lights · title · right slot.

    Dragging anywhere on the bar background moves the window (native); a
    double-click toggles maximise. The three lights close / minimise / zoom the
    real window.
    """

    HEIGHT = 42

    def __init__(self, window: QWidget, t: dict,
                 on_toggle_theme: Optional[Callable[[], None]] = None,
                 on_open_settings: Optional[Callable[[], None]] = None):
        super().__init__()
        self._win = window
        self.setFixedHeight(self.HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("TitleBar")
        self.setStyleSheet(
            f"#TitleBar{{background:{t['bg']}; border-bottom:1px solid {t['border']};}}")

        row = QHBoxLayout(self)
        row.setContentsMargins(14, 0, 12, 0)
        row.setSpacing(8)

        # ── traffic lights (left) ──
        # Late-bound (lambdas) so the current window methods are called.
        self._close = _TrafficButton(_LIGHTS[0][1], _LIGHTS[0][2], _LIGHTS[0][3],
                                     lambda: window.close())
        self._min = _TrafficButton(_LIGHTS[1][1], _LIGHTS[1][2], _LIGHTS[1][3],
                                   lambda: window.showMinimized())
        self._zoom = _TrafficButton(_LIGHTS[2][1], _LIGHTS[2][2], _LIGHTS[2][3],
                                    self._toggle_max)
        for b in (self._close, self._min, self._zoom):
            row.addWidget(b)

        row.addStretch(1)
        title = QLabel("Velum")
        title.setStyleSheet(
            f"color:{t['text_muted']}; font-size:12.5px; font-weight:600; letter-spacing:0.2px;")
        row.addWidget(title)
        row.addStretch(1)

        # ── right slot: settings + theme toggle ──
        def _chrome_button(icon_name: str, tip: str, handler: Callable[[], None]) -> QToolButton:
            b = QToolButton()
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFixedSize(26, 26)
            b.setIcon(icons.icon(icon_name, t["text_muted"], 15))
            b.setIconSize(QSize(15, 15))
            b.setToolTip(tip)
            b.clicked.connect(lambda: handler())
            b.setStyleSheet(
                f"QToolButton{{background:transparent; border:1px solid transparent; border-radius:7px;}}"
                f"QToolButton:hover{{background:{t['surface2']}; border-color:{t['border']};}}")
            return b

        added = 0
        if on_open_settings is not None:
            row.addWidget(_chrome_button("settings", "Settings", on_open_settings))
            added += 1
        if on_toggle_theme is not None:
            row.addWidget(_chrome_button("moon", "Toggle light / dark", on_toggle_theme))
            added += 1
        if added == 0:
            row.addSpacing(52)  # keep the title visually centred

    def _toggle_max(self) -> None:
        w = self._win
        w.showNormal() if w.isMaximized() else w.showMaximized()

    # ── native drag / double-click maximise ─────────────────────────────────
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            handle = self._win.windowHandle()
            if handle is not None:
                handle.startSystemMove()
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._toggle_max()
        super().mouseDoubleClickEvent(e)


def install_corner_grips(window: QWidget) -> list[QSizeGrip]:
    """Add four invisible corner resize handles to a frameless ``window``.

    Returns the grips; the window must reposition them in its ``resizeEvent``
    via :func:`layout_corner_grips`.
    """
    grips = [QSizeGrip(window) for _ in range(4)]
    for g in grips:
        g.setFixedSize(15, 15)
        g.setStyleSheet("background:transparent;")
        g.raise_()
    return grips


def layout_corner_grips(window: QWidget, grips: list[QSizeGrip]) -> None:
    """Pin the four grips to the window corners (call from ``resizeEvent``)."""
    if len(grips) != 4:
        return
    w, h, s = window.width(), window.height(), 15
    tl, tr, bl, br = grips
    tl.move(0, 0)
    tr.move(w - s, 0)
    bl.move(0, h - s)
    br.move(w - s, h - s)
    for g in grips:
        g.raise_()
