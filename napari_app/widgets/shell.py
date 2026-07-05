"""CellSeg1 — application shell: a collapsing icon-rail with a brand header
and a status footer, replacing the old top tab bar.

Behaviour (all presentational):
- Collapsed to a slim icon rail by default.
- Hovering the rail expands it **over** the content (no reflow) to reveal
  labels; leaving collapses it again.
- A pin keeps it expanded and switches to push mode, so the content reflows
  to the narrower width instead of being overlaid.
- Switching sections cross-fades the panel.

No business logic — the five tab widgets are the same instances as before,
merely hosted in a QStackedWidget instead of a QTabWidget.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QVariantAnimation, QEasingCurve, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QFrame, QHBoxLayout, QVBoxLayout, QLabel, QToolButton,
    QStackedWidget, QButtonGroup, QSizePolicy,
)

from napari_app import icons
from napari_app.motion import fade_in
from napari_app.theme import (
    BG_APP, BG, FG, BORDER, TEXT, LABEL, DIM, ACCENT, ACCENT_SOFT, SUCCESS,
    MONO, R_SM,
)

RAIL_COLLAPSED = 62
RAIL_EXPANDED = 202
INNER_W = 202

_NAV_SS = f"""
QToolButton {{
    background: transparent; border: none; border-left: 3px solid transparent;
    color: {LABEL}; font-size: 13px; font-weight: 600; text-align: left;
    padding: 10px 12px 10px 18px; border-radius: {R_SM}px;
}}
QToolButton:hover {{ background: {FG}; color: {TEXT}; }}
QToolButton:checked {{
    background: {ACCENT_SOFT}; color: {TEXT}; border-left: 3px solid {ACCENT};
}}
"""


class _Rail(QFrame):
    """The rail frame; emits hover signals so the shell can expand/collapse."""

    hoverIn = pyqtSignal()
    hoverOut = pyqtSignal()

    def __init__(self, on_pin):
        super().__init__()
        self.setObjectName("Rail")
        self.setStyleSheet(
            f"#Rail {{ background: {BG_APP}; border-right: 1px solid {BORDER}; }}")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # Fixed-width inner content; the rail clips it as it animates.
        self.inner = QWidget(self)
        self.inner.setFixedWidth(INNER_W)
        v = QVBoxLayout(self.inner)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # ── brand header ──
        brand = QWidget()
        brand.setFixedHeight(58)
        brand.setStyleSheet(f"border-bottom: 1px solid {BORDER};")
        hb = QHBoxLayout(brand)
        hb.setContentsMargins(15, 0, 10, 0)
        hb.setSpacing(11)
        logo = QLabel()
        logo.setPixmap(icons.brand_pixmap(26))
        logo.setFixedSize(28, 28)
        hb.addWidget(logo)
        name_box = QVBoxLayout()
        name_box.setContentsMargins(0, 0, 0, 0)
        name_box.setSpacing(0)
        nm = QLabel("CellSeg1")
        nm.setStyleSheet(f"color:{TEXT}; font-size:13.5px; font-weight:700; background:transparent;")
        ver = QLabel("v1.0 · lab")
        ver.setStyleSheet(f"color:{DIM}; font-size:10px; font-family:{MONO}; background:transparent;")
        name_box.addWidget(nm)
        name_box.addWidget(ver)
        hb.addLayout(name_box)
        hb.addStretch()
        self.pin = QToolButton()
        self.pin.setCheckable(True)
        self.pin.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pin.setIcon(icons.icon("pin", DIM))
        self.pin.setIconSize(QSize(15, 15))
        self.pin.setToolTip("Закрепить панель раскрытой")
        self.pin.setStyleSheet("QToolButton{border:none;background:transparent;padding:4px;}")
        self.pin.toggled.connect(on_pin)
        self.pin.toggled.connect(self._sync_pin_icon)
        hb.addWidget(self.pin, alignment=Qt.AlignmentFlag.AlignVCenter)
        v.addWidget(brand)

        # ── nav ──
        nav = QWidget()
        self._nav_l = QVBoxLayout(nav)
        self._nav_l.setContentsMargins(9, 12, 9, 12)
        self._nav_l.setSpacing(3)
        v.addWidget(nav)
        v.addStretch()

        # ── status footer ──
        foot = QWidget()
        foot.setStyleSheet(f"border-top: 1px solid {BORDER};")
        fb = QHBoxLayout(foot)
        fb.setContentsMargins(19, 12, 12, 12)
        fb.setSpacing(11)
        self._dot = QLabel()
        self._dot.setFixedSize(9, 9)
        self._dot.setStyleSheet(f"background:{SUCCESS}; border-radius:4px;")
        self._status = QLabel(_device_label())
        self._status.setStyleSheet(f"color:{LABEL}; font-size:11.5px; font-family:{MONO}; background:transparent;")
        fb.addWidget(self._dot)
        fb.addWidget(self._status)
        fb.addStretch()
        v.addWidget(foot)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self._buttons: list[tuple[QToolButton, str]] = []

    # nav construction --------------------------------------------------------
    def add_nav(self, icon_name: str, label: str) -> QToolButton:
        b = QToolButton()
        b.setText("  " + label)
        b.setIcon(icons.icon(icon_name, LABEL))
        b.setIconSize(QSize(18, 18))
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        b.setCheckable(True)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFixedWidth(INNER_W - 18)
        b.setMinimumHeight(40)
        b.setStyleSheet(_NAV_SS)
        b.setToolTip(label)
        self.group.addButton(b)
        self._nav_l.addWidget(b)
        self._buttons.append((b, icon_name))
        return b

    def select(self, idx: int):
        for i, (b, icon_name) in enumerate(self._buttons):
            active = i == idx
            b.setChecked(active)
            b.setIcon(icons.icon(icon_name, TEXT if active else LABEL))

    def _sync_pin_icon(self, on: bool):
        self.pin.setIcon(icons.icon("pin", ACCENT if on else DIM))

    def set_status(self, text: str, color: str = SUCCESS):
        self._status.setText(text)
        self._dot.setStyleSheet(f"background:{color}; border-radius:4px;")

    # geometry / hover --------------------------------------------------------
    def resizeEvent(self, e):
        self.inner.setGeometry(0, 0, INNER_W, self.height())
        super().resizeEvent(e)

    def enterEvent(self, e):
        self.hoverIn.emit()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.hoverOut.emit()
        super().leaveEvent(e)


class Shell(QWidget):
    """Hosts the rail (overlaid) and a stacked content area."""

    def __init__(self, pages):
        super().__init__()
        self.setObjectName("Shell")
        self.setStyleSheet(f"#Shell {{ background:{BG}; }}")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._pinned = False
        self._w = RAIL_COLLAPSED

        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background:{BG};")
        self.rail = _Rail(on_pin=self._toggle_pin)

        for i, (icon_name, label, widget) in enumerate(pages):
            self.stack.addWidget(widget)
            btn = self.rail.add_nav(icon_name, label)
            btn.clicked.connect(lambda _=False, idx=i: self._select(idx))

        lay = QHBoxLayout(self)
        lay.setContentsMargins(RAIL_COLLAPSED, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.stack)
        self._lay = lay

        self.rail.setParent(self)
        self.rail.raise_()
        self.rail.hoverIn.connect(self._expand)
        self.rail.hoverOut.connect(self._collapse)

        self.rail.select(0)
        self.stack.setCurrentIndex(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # selection ---------------------------------------------------------------
    def _select(self, idx: int):
        if self.stack.currentIndex() != idx:
            self.stack.setCurrentIndex(idx)
            fade_in(self.stack.currentWidget(), 220)
        self.rail.select(idx)

    # rail expand / collapse --------------------------------------------------
    def _animate_rail(self, to: int):
        anim = QVariantAnimation(self)
        anim.setDuration(220)
        anim.setStartValue(self._w)
        anim.setEndValue(to)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(self._apply_rail_width)
        anim.start()
        self._rail_anim = anim

    def _apply_rail_width(self, v):
        self._w = int(v)
        self.rail.setGeometry(0, 0, self._w, self.height())

    def _expand(self):
        if not self._pinned:
            self._animate_rail(RAIL_EXPANDED)

    def _collapse(self):
        if not self._pinned:
            self._animate_rail(RAIL_COLLAPSED)

    def _toggle_pin(self, checked: bool):
        self._pinned = checked
        margin = RAIL_EXPANDED if checked else RAIL_COLLAPSED
        self._lay.setContentsMargins(margin, 0, 0, 0)
        self._animate_rail(margin)

    def set_status(self, text, color=SUCCESS):
        self.rail.set_status(text, color)

    def resizeEvent(self, e):
        self.rail.setGeometry(0, 0, self._w, self.height())
        super().resizeEvent(e)


def _device_label() -> str:
    """Best-effort, read-only compute-device string for the status footer."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda · ready"
        if torch.backends.mps.is_available():
            return "mps · ready"
    except Exception:
        pass
    return "cpu · ready"
