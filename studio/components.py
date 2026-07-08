"""CellSeg1 Studio — the static UI kit (design skeleton).

Presentational Qt widgets that reproduce the north-star mockup one-to-one:
chips, buttons, selects, toggles, sliders, steppers, segmented controls, stat
tiles, collapsible sections, and the navigation sidebar. **No business logic** —
this branch is a pure design skeleton; controls render state but don't act.
Interactivity that's part of the *design* (nav, theme toggle, opening the
command palette, expanding a section, flipping a toggle's look) is kept; real
functionality is wired later, tab by tab (see ``docstudio/``).

Every widget takes a token dict from :mod:`studio.theme`, so it
renders in either theme. All constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QToolButton,
    QPushButton, QSizePolicy, QGraphicsDropShadowEffect,
)
from PyQt6.QtGui import QColor

from studio import icons
from studio import theme


# ── primitives ───────────────────────────────────────────────────────────────
def hline(t: dict) -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background:{t['border']}; border:none;")
    return f


def soft_shadow(w: QWidget, blur: int = 16, alpha: int = 26, dy: int = 3) -> None:
    eff = QGraphicsDropShadowEffect(w)
    eff.setBlurRadius(blur)
    eff.setOffset(0, dy)
    eff.setColor(QColor(28, 42, 120, alpha))
    w.setGraphicsEffect(eff)


def label(text: str, size: float, color: str, weight: int = 400,
          spacing: float = 0.0) -> QLabel:
    lb = QLabel(text)
    ls = f"letter-spacing:{spacing}px;" if spacing else ""
    lb.setStyleSheet(f"color:{color}; font-size:{size}px; font-weight:{weight}; {ls}")
    return lb


class Chip(QLabel):
    """Rounded pill/badge. ``kind`` selects the colour family."""

    def __init__(self, text: str, t: dict, kind: str = "default"):
        super().__init__(text)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        fg, bg, bd = {
            "default": (t["text_subtle"], t["surface2"], t["border"]),
            "primary": (t["primary"], t["primary_weak"], t["primary_line"]),
            "signal":  (t["signal"], t["signal_weak"], t["signal_line"]),
            "success": (t["success"], t["success_weak"], t["success_weak"]),
            "warning": (t["warning"], t["warning_weak"], t["warning_weak"]),
            "muted":   (t["text_muted"], "transparent", t["border"]),
        }.get(kind, (t["text_subtle"], t["surface2"], t["border"]))
        self.setStyleSheet(
            f"color:{fg}; background:{bg}; border:1px solid {bd};"
            f"border-radius:999px; padding:3px 10px; font-size:11px; font-weight:600;")


class Badge(QLabel):
    """Compact monospaced value badge."""

    def __init__(self, text: str, t: dict):
        super().__init__(text)
        self.setStyleSheet(
            f"color:{t['text_subtle']}; background:{t['surface2']}; border-radius:6px;"
            f"padding:2px 7px; font-size:11px; font-weight:600; font-family:{theme.MONO};")


class PillButton(QPushButton):
    """Button in a design ``kind`` (primary | ghost | success | danger), optional icon."""

    def __init__(self, text: str, t: dict, kind: str = "primary",
                 icon_name: Optional[str] = None, small: bool = False):
        super().__init__(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        qss = theme.button_qss(t, kind)
        if small:
            qss += "QPushButton{padding:7px 11px; font-size:12.5px;}"
        self.setStyleSheet(qss)
        if icon_name:
            col = "#ffffff" if kind in ("primary", "danger") else t["text_subtle"]
            self.setIcon(icons.icon(icon_name, col, 15))
            self.setIconSize(QSize(15, 15))


class IconButton(QToolButton):
    """A square icon button with hover surface."""

    def __init__(self, icon_name: str, t: dict, size: int = 30, tip: str = "",
                 on_click: Optional[Callable[[], None]] = None):
        super().__init__()
        self._t = t
        self._icon = icon_name
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(size, size)
        self.setIcon(icons.icon(icon_name, t["text_muted"], int(size * 0.55)))
        self.setIconSize(QSize(int(size * 0.55), int(size * 0.55)))
        if tip:
            self.setToolTip(tip)
        self.setStyleSheet(
            f"QToolButton{{background:transparent; border:1px solid transparent; border-radius:8px;}}"
            f"QToolButton:hover{{background:{t['surface2']}; border-color:{t['border']};}}")
        if on_click:
            self.clicked.connect(lambda: on_click())


class SelectBox(QFrame):
    """A non-interactive combo-box look-alike (value + chevron)."""

    def __init__(self, text: str, t: dict, lead_icon: Optional[str] = None,
                 lead_color: Optional[str] = None):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QFrame{{background:{t['inset']}; border:1px solid {t['border']};"
            f"border-radius:8px;}} QFrame:hover{{border-color:{t['border_strong']};}}")
        row = QHBoxLayout(self)
        row.setContentsMargins(11, 7, 10, 7)
        row.setSpacing(7)
        if lead_icon:
            ic = QLabel()
            ic.setPixmap(icons.pixmap(lead_icon, lead_color or t["primary"], 14))
            row.addWidget(ic)
        val = QLabel(text)
        val.setStyleSheet(f"color:{t['text']}; font-size:12.5px; font-weight:600;")
        row.addWidget(val)
        row.addStretch(1)
        chev = QLabel()
        chev.setPixmap(icons.pixmap("chevron_down", t["text_muted"], 13))
        row.addWidget(chev)


class Toggle(QFrame):
    """A pill toggle whose look reflects on/off. Clicking flips the look only."""

    toggled = pyqtSignal(bool)

    def __init__(self, t: dict, on: bool = False):
        super().__init__()
        self._t = t
        self._on = on
        self.setFixedSize(36, 21)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._knob = QFrame(self)
        self._knob.setFixedSize(17, 17)
        self._render()

    def _render(self):
        t = self._t
        self.setStyleSheet(
            f"background:{t['signal'] if self._on else t['border_strong']}; border-radius:10px;")
        self._knob.setStyleSheet("background:#ffffff; border-radius:8px;")
        self._knob.move(17 if self._on else 2, 2)

    def is_on(self) -> bool:
        return self._on

    def set_on(self, on: bool):
        self._on = on
        self._render()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._on = not self._on
            self._render()
            self.toggled.emit(self._on)
        super().mouseReleaseEvent(e)


class Slider(QFrame):
    """A static track with a fill + knob at ``value`` (0..1). Presentational."""

    def __init__(self, t: dict, value: float = 0.5, color: Optional[str] = None):
        super().__init__()
        self._t = t
        self._v = max(0.0, min(1.0, value))
        self._color = color or t["primary"]
        self.setFixedHeight(14)
        self.setMinimumWidth(80)
        self._track = QFrame(self)
        self._fill = QFrame(self)
        self._knob = QFrame(self)
        self._track.setStyleSheet(f"background:{t['border']}; border-radius:2px;")
        self._fill.setStyleSheet(f"background:{self._color}; border-radius:2px;")
        self._knob.setStyleSheet(
            f"background:#ffffff; border:2px solid {self._color}; border-radius:7px;")

    def resizeEvent(self, e):
        w, h = self.width(), self.height()
        self._track.setGeometry(0, h // 2 - 2, w, 4)
        fw = int(w * self._v)
        self._fill.setGeometry(0, h // 2 - 2, fw, 4)
        self._knob.setGeometry(max(0, min(w - 14, fw - 7)), h // 2 - 7, 14, 14)
        super().resizeEvent(e)


class Stepper(QFrame):
    """− value + control (presentational; buttons don't change the value)."""

    def __init__(self, value: str, t: dict):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QFrame{{background:{t['inset']}; border:1px solid {t['border']}; border-radius:8px;}}")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        for glyph in ("−", None, "+"):
            if glyph is None:
                val = QLabel(value)
                val.setAlignment(Qt.AlignmentFlag.AlignCenter)
                val.setMinimumWidth(42)
                val.setStyleSheet(
                    f"color:{t['text']}; font-family:{theme.MONO}; font-size:12.5px; font-weight:600;")
                row.addWidget(val)
            else:
                b = QToolButton()
                b.setText(glyph)
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.setFixedSize(26, 28)
                b.setStyleSheet(
                    f"QToolButton{{background:transparent; border:none; color:{t['text_muted']};"
                    f"font-size:15px;}} QToolButton:hover{{background:{t['surface2']}; color:{t['text']};}}")
                row.addWidget(b)


class SegControl(QFrame):
    """Segmented buttons; one is 'on'. Clicking moves the selection (visual).

    Pass ``icons_`` (parallel to ``options``, entries optional) for icon-only
    segments — e.g. a grid/list view switch — instead of text labels; the icon
    is recoloured to match the on/off state, same as text.
    """

    changed = pyqtSignal(int)

    def __init__(self, options: list[str], t: dict, active: int = 0, compact: bool = False,
                 icons_: Optional[list[Optional[str]]] = None):
        super().__init__()
        self._t = t
        self._btns: list[QToolButton] = []
        self._icon_names: list[Optional[str]] = list(icons_) if icons_ else [None] * len(options)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QFrame{{background:{t['inset'] if compact else t['surface2']};"
            f"border:1px solid {t['border']}; border-radius:8px;}}")
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(2)
        for i, opt in enumerate(options):
            b = QToolButton()
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setCheckable(True)
            b.setChecked(i == active)
            b.setMinimumHeight(26)
            if self._icon_names[i]:
                b.setIconSize(QSize(14, 14))
                b.setFixedWidth(32)
            else:
                b.setText(opt)
                b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.clicked.connect(lambda _=False, idx=i: self._select(idx))
            self._btns.append(b)
            row.addWidget(b, 0 if self._icon_names[i] else 1)
        self._restyle()

    def _select(self, idx: int):
        for i, b in enumerate(self._btns):
            b.setChecked(i == idx)
        self._restyle()
        self.changed.emit(idx)

    def _restyle(self):
        t = self._t
        for name, b in zip(self._icon_names, self._btns):
            on = b.isChecked()
            if name:
                b.setIcon(icons.icon(name, t["text"] if on else t["text_muted"], 14))
            if on:
                b.setStyleSheet(
                    f"QToolButton{{background:{t['surface']}; color:{t['text']};"
                    f"border:none; border-radius:6px; font-size:12px; font-weight:600; padding:5px 10px;}}")
            else:
                b.setStyleSheet(
                    f"QToolButton{{background:transparent; color:{t['text_muted']};"
                    f"border:none; border-radius:6px; font-size:12px; font-weight:600; padding:5px 10px;}}"
                    f"QToolButton:hover{{color:{t['text_subtle']};}}")


class StatTile(QFrame):
    """A value + label tile (results stats)."""

    def __init__(self, value: str, unit: str, caption: str, t: dict, ok: bool = False):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QFrame{{background:{t['inset']}; border:1px solid {t['border']}; border-radius:10px;}}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 11, 12, 11)
        lay.setSpacing(2)
        v = QLabel(f"{value}<span style='font-size:11px;color:{t['text_muted']}'> {unit}</span>" if unit else value)
        col = t["success"] if ok else t["text"]
        v.setStyleSheet(f"color:{col}; font-family:{theme.MONO}; font-size:19px; font-weight:600; letter-spacing:-0.5px;")
        c = QLabel(caption)
        c.setStyleSheet(f"color:{t['text_muted']}; font-size:10.5px; font-weight:600; letter-spacing:0.5px;")
        lay.addWidget(v)
        lay.addWidget(c)


class FieldRow(QWidget):
    """A label ⟷ control row."""

    def __init__(self, name: str, control: QWidget, t: dict):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        lb = QLabel(name)
        lb.setStyleSheet(f"color:{t['text_subtle']}; font-size:12.5px; font-weight:500;")
        row.addWidget(lb)
        row.addStretch(1)
        row.addWidget(control)


class GroupLabel(QLabel):
    """Uppercase group heading."""

    def __init__(self, text: str, t: dict):
        super().__init__(text.upper())
        self.setStyleSheet(
            f"color:{t['text_muted']}; font-size:10.5px; font-weight:600; letter-spacing:0.6px;")


class Accordion(QFrame):
    """A collapsible section with a leading icon, title, chevron, and body."""

    def __init__(self, title: str, t: dict, lead: str = "check", open_: bool = False):
        super().__init__()
        self._t = t
        self._open = open_
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QFrame#Acc{{background:{t['inset']}; border:1px solid {t['border']}; border-radius:10px;}}")
        self.setObjectName("Acc")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)

        self._head = QToolButton()
        self._head.setCursor(Qt.CursorShape.PointingHandCursor)
        self._head.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._head_row = QWidget()
        hr = QHBoxLayout(self._head_row)
        hr.setContentsMargins(13, 12, 13, 12)
        hr.setSpacing(9)
        licon = QLabel()
        licon.setPixmap(icons.pixmap(lead, t["signal"], 15))
        title_lb = QLabel(title.upper())
        title_lb.setStyleSheet(
            f"color:{t['text_subtle']}; font-size:11.5px; font-weight:600; letter-spacing:0.5px;")
        self._chev = QLabel()
        self._chev.setPixmap(icons.pixmap("chevron" if not open_ else "chevron_down", t["text_muted"], 14))
        hr.addWidget(licon)
        hr.addWidget(title_lb)
        hr.addStretch(1)
        hr.addWidget(self._chev)
        self._head_row.setStyleSheet("background:transparent;")
        self._head_row.mouseReleaseEvent = lambda e: self.toggle()
        self._lay.addWidget(self._head_row)

        self._body = QWidget()
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(13, 0, 13, 14)
        self._body_lay.setSpacing(10)
        self._body.setVisible(open_)
        self._lay.addWidget(self._body)

    def add(self, w: QWidget):
        self._body_lay.addWidget(w)

    def add_layout(self, l):
        self._body_lay.addLayout(l)

    def toggle(self):
        self._open = not self._open
        self._body.setVisible(self._open)
        self._chev.setPixmap(icons.pixmap(
            "chevron_down" if self._open else "chevron", self._t["text_muted"], 14))


# ── Sidebar navigation ───────────────────────────────────────────────────────
class _NavItem(QToolButton):
    def __init__(self, key: str, icon_name: str, text: str, t: dict, pip: bool = False):
        super().__init__()
        self.key = key
        self.icon_name = icon_name
        self._t = t
        self._pip = pip
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setText("   " + text)
        self.setIcon(icons.icon(icon_name, t["text_muted"], 18))
        self.setIconSize(QSize(18, 18))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(38)
        self._apply(False)

    def _apply(self, active: bool):
        t = self._t
        if active:
            self.setIcon(icons.icon(self.icon_name, t["signal"], 18))
            self.setStyleSheet(
                f"QToolButton{{background:{t['surface']}; color:{t['text']};"
                f"border:1px solid {t['border']}; border-radius:9px; padding:8px 10px;"
                f"font-size:13px; font-weight:600; text-align:left;}}")
        else:
            self.setIcon(icons.icon(self.icon_name, t["text_muted"], 18))
            self.setStyleSheet(
                f"QToolButton{{background:transparent; color:{t['text_subtle']};"
                f"border:1px solid transparent; border-radius:9px; padding:8px 10px;"
                f"font-size:13px; font-weight:600; text-align:left;}}"
                f"QToolButton:hover{{background:{t['surface2']}; color:{t['text']};}}")

    def set_active(self, active: bool):
        self.setChecked(active)
        self._apply(active)


class Sidebar(QFrame):
    """Navigation rail: wordmark, sectioned nav items, footer (guide, appearance)."""

    navigate = pyqtSignal(str)
    toggle_theme = pyqtSignal()
    open_guide = pyqtSignal()

    WIDTH = 232

    def __init__(self, items: list[tuple[str, str, str, str]], t: dict):
        super().__init__()
        self._t = t
        self._items: dict[str, _NavItem] = {}
        self.setObjectName("Sidebar")
        self.setFixedWidth(self.WIDTH)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"#Sidebar{{background:{t['inset']}; border-right:1px solid {t['border']};}}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(3)

        brand = QLabel(
            f"<span style='font-size:19px;font-weight:600;letter-spacing:-0.4px;color:{t['text']}'>"
            f"CellSeg<span style='color:{t['primary']}'>1</span></span>&nbsp;&nbsp;"
            f"<span style='font-size:9px;font-weight:600;letter-spacing:1.4px;color:{t['text_muted']}'>STUDIO</span>")
        brand.setContentsMargins(6, 2, 0, 12)
        lay.addWidget(brand)

        section = None
        for key, icon_name, text, sec in items:
            if sec and sec != section:
                section = sec
                lay.addWidget(self._section(sec))
            item = _NavItem(key, icon_name, text, t)
            item.clicked.connect(lambda _=False, k=key: self.navigate.emit(k))
            self._items[key] = item
            lay.addWidget(item)

        lay.addStretch(1)
        lay.addWidget(hline(t))

        guide = _NavItem("__guide__", "guide", "Guide & Docs", t)
        guide.setCheckable(False)
        guide.clicked.connect(lambda: self.open_guide.emit())
        lay.addWidget(guide)

        appearance = _NavItem("__theme__", "moon", "Appearance", t)
        appearance.setCheckable(False)
        appearance.clicked.connect(lambda: self.toggle_theme.emit())
        lay.addWidget(appearance)

        ver = QLabel("v0.9.0 · Studio")
        ver.setStyleSheet(f"color:{t['text_muted']}; font-size:10.5px; padding:6px 10px 0; font-family:{theme.MONO};")
        lay.addWidget(ver)

    def _section(self, text: str) -> QLabel:
        lb = QLabel(text.upper())
        lb.setStyleSheet(
            f"color:{self._t['text_muted']}; font-size:10px; font-weight:600;"
            f"letter-spacing:0.7px; padding:12px 10px 4px;")
        return lb

    def set_active(self, key: str):
        for k, item in self._items.items():
            if item.isCheckable():
                item.set_active(k == key)
