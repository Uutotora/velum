"""CellSeg1 — small, safe motion helpers (purely presentational).

Micro-interactions that make the app feel like a modern product: soft card
shadows, fade-ins on panel switches, a count-up for the headline number, an
animated width for the collapsing navigation rail, and a hover-elevation
filter. All implemented with Qt's animation framework — no business logic.

Every helper is defensive: if animations can't run (e.g. offscreen import),
it degrades to the final state instead of raising.
"""
from __future__ import annotations

from PyQt6.QtCore import (
    QEasingCurve, QPropertyAnimation, QVariantAnimation, QObject, QEvent,
    QParallelAnimationGroup, pyqtProperty,
)
from PyQt6.QtWidgets import QGraphicsDropShadowEffect, QGraphicsOpacityEffect
from PyQt6.QtGui import QColor

EASE = QEasingCurve.Type.OutCubic


def add_shadow(widget, blur: int = 26, y: int = 10, alpha: int = 150):
    """Attach a soft drop shadow to a widget (QSS has no box-shadow)."""
    try:
        eff = QGraphicsDropShadowEffect(widget)
        eff.setBlurRadius(blur)
        eff.setXOffset(0)
        eff.setYOffset(y)
        eff.setColor(QColor(0, 0, 0, alpha))
        widget.setGraphicsEffect(eff)
        return eff
    except Exception:
        return None


def fade_in(widget, duration: int = 240):
    """Fade a widget from transparent to opaque (used on panel switch)."""
    try:
        eff = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(eff)
        anim = QPropertyAnimation(eff, b"opacity", widget)
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(EASE)
        anim.finished.connect(lambda: widget.setGraphicsEffect(None))
        anim.start()
        widget._fade_anim = anim  # keep a ref alive
        return anim
    except Exception:
        return None


def count_up(label, to: int, duration: int = 650, fmt="{:d}"):
    """Animate an integer QLabel from 0 → ``to`` (the headline cell count)."""
    try:
        to = int(to)
    except Exception:
        return None
    if to <= 0:
        label.setText(fmt.format(max(to, 0)))
        return None
    try:
        anim = QVariantAnimation(label)
        anim.setDuration(duration)
        anim.setStartValue(0)
        anim.setEndValue(to)
        anim.setEasingCurve(EASE)
        anim.valueChanged.connect(lambda v: label.setText(fmt.format(int(v))))
        anim.finished.connect(lambda: label.setText(fmt.format(to)))
        anim.start()
        label._count_anim = anim
        return anim
    except Exception:
        label.setText(fmt.format(to))
        return None


def animate_width(widget, start: int, end: int, duration: int = 240):
    """Animate a widget's fixed width (min & max together) — the nav rail."""
    try:
        grp = QParallelAnimationGroup(widget)
        for prop in (b"minimumWidth", b"maximumWidth"):
            a = QPropertyAnimation(widget, prop, widget)
            a.setDuration(duration)
            a.setStartValue(start)
            a.setEndValue(end)
            a.setEasingCurve(EASE)
            grp.addAnimation(a)
        grp.start()
        widget._width_anim = grp
        return grp
    except Exception:
        widget.setMinimumWidth(end)
        widget.setMaximumWidth(end)
        return None


class HoverElevate(QObject):
    """Event filter: lift a card's shadow on hover for a tactile feel."""

    def __init__(self, widget, rest_blur=18, hover_blur=32, rest_y=8, hover_y=14):
        super().__init__(widget)
        self._w = widget
        self._rest = (rest_blur, rest_y)
        self._hover = (hover_blur, hover_y)
        self._eff = add_shadow(widget, rest_blur, rest_y, 130)
        widget.installEventFilter(self)

    def _to(self, blur, y):
        if not self._eff:
            return
        try:
            a = QPropertyAnimation(self._eff, b"blurRadius", self._w)
            a.setDuration(150)
            a.setEndValue(blur)
            a.setEasingCurve(EASE)
            a.start()
            self._w._hover_anim = a
            self._eff.setYOffset(y)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Enter:
            self._to(*self._hover)
        elif event.type() == QEvent.Type.Leave:
            self._to(*self._rest)
        return False


def elevate_on_hover(widget, **kw):
    """Convenience: give a card a hover-reactive shadow. Returns the filter."""
    return HoverElevate(widget, **kw)
