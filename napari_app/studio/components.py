"""CellSeg1 Studio — reusable, themed Qt building blocks.

Small presentational widgets shared across the Studio screens: the navigation
sidebar, chips/pills, project cards, section headers. Each takes a *token
dict* (from :mod:`napari_app.studio.theme`) so it renders correctly in either
theme and re-styles on a live theme switch.

No business logic here — widgets emit Qt signals / call callbacks; the screens
and the app wire them to the :class:`~napari_app.studio.project.ProjectStore`.
Everything is constructible under ``QT_QPA_PLATFORM=offscreen`` (only a
``QApplication`` is required, not a display), which is how it's smoke-tested.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QToolButton,
    QPushButton, QSizePolicy, QGraphicsDropShadowEffect,
)
from PyQt6.QtGui import QColor

from napari_app import icons
from napari_app.studio import theme


# ── tiny helpers ─────────────────────────────────────────────────────────────
def hline(t: dict) -> QFrame:
    """A 1px hairline separator in the current border colour."""
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background:{t['border']}; border:none;")
    return f


def soft_shadow(widget: QWidget, blur: int = 18, alpha: int = 34, dy: int = 4) -> None:
    """Apply a subtle blue-tinted drop shadow (the Studio card elevation)."""
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(blur)
    eff.setOffset(0, dy)
    eff.setColor(QColor(28, 42, 120, alpha))
    widget.setGraphicsEffect(eff)


class Chip(QLabel):
    """A rounded pill/badge. ``kind`` selects the colour family."""

    def __init__(self, text: str, t: dict, kind: str = "default",
                 dot: Optional[str] = None):
        super().__init__(text)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        fg, bg, bd = {
            "default": (t["text_subtle"], t["surface2"], t["border"]),
            "primary": (t["primary"], t["primary_weak"], t["primary_line"]),
            "signal":  (t["signal"], t["signal_weak"], t["signal_line"]),
            "success": (t["success"], t["success_weak"], t["success_weak"]),
            "muted":   (t["text_muted"], "transparent", t["border"]),
        }.get(kind, (t["text_subtle"], t["surface2"], t["border"]))
        self.setStyleSheet(
            f"color:{fg}; background:{bg}; border:1px solid {bd};"
            f"border-radius:999px; padding:3px 10px; font-size:11px; font-weight:600;")


class GhostButton(QPushButton):
    """Outline/secondary button with an optional leading icon."""

    def __init__(self, text: str, t: dict, icon_name: Optional[str] = None):
        super().__init__(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(theme.button_qss(t, "ghost"))
        if icon_name:
            self.setIcon(icons.icon(icon_name, t["text_subtle"], 15))
            self.setIconSize(QSize(15, 15))


class PrimaryButton(QPushButton):
    """Accent (iris) call-to-action button with an optional leading icon."""

    def __init__(self, text: str, t: dict, icon_name: Optional[str] = None):
        super().__init__(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(theme.button_qss(t, "primary"))
        if icon_name:
            self.setIcon(icons.icon(icon_name, "#ffffff", 15))
            self.setIconSize(QSize(15, 15))


# ── Sidebar navigation ───────────────────────────────────────────────────────
class _NavItem(QToolButton):
    """One sidebar row: icon + label, with active + hover states."""

    def __init__(self, key: str, icon_name: str, label: str, t: dict,
                 badge: str = "", pip: bool = False):
        super().__init__()
        self.key = key
        self.icon_name = icon_name
        self._t = t
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setText(("   " + label) + (f"      {badge}" if badge else ""))
        self.setIcon(icons.icon(icon_name, t["text_muted"], 18))
        self.setIconSize(QSize(18, 18))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(38)
        self._apply_style(active=False)

    def _apply_style(self, active: bool) -> None:
        t = self._t
        if active:
            self.setIcon(icons.icon(self.icon_name, t["signal"], 18))
            self.setStyleSheet(
                f"QToolButton{{background:{t['surface']}; color:{t['text']};"
                f"border:1px solid {t['border']}; border-radius:9px;"
                f"padding:8px 10px; font-size:13px; font-weight:600; text-align:left;}}")
        else:
            self.setIcon(icons.icon(self.icon_name, t["text_muted"], 18))
            self.setStyleSheet(
                f"QToolButton{{background:transparent; color:{t['text_subtle']};"
                f"border:1px solid transparent; border-radius:9px;"
                f"padding:8px 10px; font-size:13px; font-weight:600; text-align:left;}}"
                f"QToolButton:hover{{background:{t['surface2']}; color:{t['text']};}}")

    def set_active(self, active: bool) -> None:
        self.setChecked(active)
        self._apply_style(active)


class Sidebar(QFrame):
    """The Studio navigation rail: wordmark, sections of nav items, footer.

    Items are ``(key, icon_name, label, section)`` tuples. Selecting one emits
    :attr:`navigate`; the theme toggle emits :attr:`toggle_theme`. No app icon —
    a clean wordmark, per the product direction.
    """

    navigate = pyqtSignal(str)
    toggle_theme = pyqtSignal()

    WIDTH = 232

    def __init__(self, items: list[tuple[str, str, str, str]], t: dict):
        super().__init__()
        self._t = t
        self._items: dict[str, _NavItem] = {}
        self.setObjectName("Sidebar")
        self.setFixedWidth(self.WIDTH)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"#Sidebar{{background:{t['inset']}; border-right:1px solid {t['border']};}}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 16, 12, 12)
        lay.setSpacing(3)

        # ── wordmark (no app icon) ──
        brand = QLabel()
        brand.setText(
            f"<span style='font-size:19px;font-weight:600;letter-spacing:-0.4px;"
            f"color:{t['text']}'>CellSeg<span style='color:{t['primary']}'>1</span></span>"
            f"&nbsp;&nbsp;<span style='font-size:9px;font-weight:600;letter-spacing:1.4px;"
            f"color:{t['text_muted']}'>STUDIO</span>")
        brand.setContentsMargins(6, 2, 0, 12)
        lay.addWidget(brand)

        current_section = None
        for key, icon_name, label, section in items:
            if section and section != current_section:
                current_section = section
                lay.addWidget(self._section_label(section))
            extra = {}
            item = _NavItem(key, icon_name, label, t, **extra)
            item.clicked.connect(lambda _=False, k=key: self.navigate.emit(k))
            self._items[key] = item
            lay.addWidget(item)

        lay.addStretch(1)
        lay.addWidget(hline(t))

        # ── footer: appearance toggle + version ──
        appearance = _NavItem("__theme__", "moon", "Appearance", t)
        appearance.setCheckable(False)
        appearance.clicked.connect(lambda: self.toggle_theme.emit())
        lay.addWidget(appearance)

        ver = QLabel("v0.9.0 · Studio")
        ver.setStyleSheet(
            f"color:{t['text_muted']}; font-size:10.5px; padding:6px 10px 0;"
            f"font-family:{theme.MONO};")
        lay.addWidget(ver)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(
            f"color:{self._t['text_muted']}; font-size:10px; font-weight:600;"
            f"letter-spacing:0.7px; padding:12px 10px 4px;")
        return lbl

    def set_active(self, key: str) -> None:
        """Highlight the nav item ``key`` (no-op for panel/footer keys)."""
        for k, item in self._items.items():
            if item.isCheckable():
                item.set_active(k == key)
