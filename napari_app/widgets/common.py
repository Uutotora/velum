from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve
from napari_app.theme import (
    BORDER, BORDER_STRONG, DIM, LABEL, TEXT, ACCENT, ACCENT_SOFT, ACCENT_LINE,
    FG, CARD_HEADER, INPUT, SUCCESS, SUCCESS_SOFT, WARNING, WARNING_SOFT,
    DANGER, DANGER_SOFT, MONO, R_MD, R_LG,
)
from napari_app import icons


# ── Reusable chips / pills / badges ─────────────────────────────────────────

def make_chip(text: str, fg: str = ACCENT, bg: str = ACCENT_SOFT,
              border: str = ACCENT_LINE) -> QLabel:
    """A small rounded chip (e.g. a metric like ``mAP 0.917``)."""
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{fg}; background:{bg}; border:1px solid {border};"
        f"border-radius:999px; padding:2px 10px; font-size:11px; font-weight:600;")
    lbl.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return lbl


def status_pill(text: str, color: str = SUCCESS, soft: str = SUCCESS_SOFT) -> QWidget:
    """A dot + label status pill for session / model state."""
    w = QWidget()
    w.setStyleSheet(
        f"background:{soft}; border-radius:999px;")
    row = QHBoxLayout(w)
    row.setContentsMargins(9, 3, 11, 3)
    row.setSpacing(7)
    dot = QLabel()
    dot.setFixedSize(8, 8)
    dot.setStyleSheet(f"background:{color}; border-radius:4px;")
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color:{color}; font-size:11px; font-weight:600; background:transparent;")
    row.addWidget(dot)
    row.addWidget(lbl)
    w.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return w


# ── Section header (bar-style, kept for the legacy Predict panel) ────────────

def section_header(text: str) -> QWidget:
    container = QWidget()
    container.setContentsMargins(0, 0, 0, 0)
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 22, 0, 8)
    row.setSpacing(9)
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {LABEL}; font-size: 11px; font-weight: 700; letter-spacing: 1.4px;")
    row.addWidget(lbl)
    row.addStretch()
    return container


def divider() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER}; border: none; margin: 8px 0;")
    return f


def param_row(label_text: str, widget, tip: str = "", label_width: int = 110) -> QHBoxLayout:
    """Label–value pair on the 4px rhythm."""
    row = QHBoxLayout()
    row.setSpacing(10)
    row.setContentsMargins(0, 3, 0, 3)
    lbl = QLabel(label_text)
    lbl.setStyleSheet(f"color: {LABEL}; font-size: 11px; font-weight: 500;")
    lbl.setFixedWidth(label_width)
    if tip:
        lbl.setToolTip(tip)
        widget.setToolTip(tip)
    row.addWidget(lbl)
    row.addWidget(widget)
    return row


# ── Card system v2 ──────────────────────────────────────────────────────────
# Flat header with an accent tick + optional icon + micro-label — no heavy
# "zebra" header band. The card is cleanly framed on all four sides.

def _make_card_frame() -> QFrame:
    f = QFrame()
    f.setObjectName("CardFrame")
    f.setStyleSheet(
        f"QFrame#CardFrame {{"
        f"  background: {FG};"
        f"  border: 1px solid {BORDER};"
        f"  border-radius: {R_LG}px;"
        f"}}")
    return f


def _make_header(title: str, icon_name: str | None = None,
                 clickable: bool = False, accent: str = ACCENT):
    """Return (header_widget, arrow_label). Flat, with an accent tick."""
    hdr = QWidget()
    hdr.setObjectName("CardHdr")
    hdr.setStyleSheet(
        f"QWidget#CardHdr {{ background: transparent; border: none;"
        f" border-bottom: 1px solid {BORDER}; }}")
    if clickable:
        hdr.setCursor(Qt.CursorShape.PointingHandCursor)
    row = QHBoxLayout(hdr)
    row.setContentsMargins(14, 10, 12, 10)
    row.setSpacing(9)

    if icon_name:
        ic = QLabel()
        ic.setPixmap(icons.pixmap(icon_name, LABEL, 15))
        ic.setFixedSize(16, 16)
        ic.setStyleSheet("background:transparent;")
        row.addWidget(ic)

    lbl = QLabel(title.upper())
    lbl.setStyleSheet(
        f"color:{LABEL}; font-size:11px; font-weight:700; letter-spacing:1.3px;"
        f"background:transparent; border:none;")
    row.addWidget(lbl)
    row.addStretch()

    arrow = None
    if clickable:
        arrow = QLabel()
        arrow.setPixmap(icons.pixmap("chevron_down", DIM, 14))
        arrow.setFixedSize(15, 15)
        arrow.setStyleSheet("background:transparent;")
        row.addWidget(arrow)
    return hdr, row, lbl, arrow


class SectionCard(QWidget):
    """Flat card: accent tick + optional icon in a flat header, then content."""

    def __init__(self, title: str, accent_color: str = None, parent=None,
                 icon: str = None):
        super().__init__(parent)
        color = accent_color or ACCENT

        outer = QVBoxLayout(self)
        outer.setSpacing(0)
        outer.setContentsMargins(0, 14, 0, 0)

        frame = _make_card_frame()
        fbox = QVBoxLayout(frame)
        fbox.setSpacing(0)
        fbox.setContentsMargins(0, 0, 0, 0)

        hdr, _row, _lbl, _ = _make_header(title, icon, clickable=False, accent=color)
        fbox.addWidget(hdr)

        self._cw = QWidget()
        self._cw.setObjectName("CardBody")
        self._cw.setStyleSheet(f"#CardBody {{ background: {FG}; border: none; }}")
        self._cl = QVBoxLayout(self._cw)
        self._cl.setSpacing(8)
        self._cl.setContentsMargins(13, 11, 13, 13)
        fbox.addWidget(self._cw)

        outer.addWidget(frame)

    def addWidget(self, w):
        self._cl.addWidget(w)

    def addLayout(self, lay):
        self._cl.addLayout(lay)


class CollapsibleCard(QWidget):
    """Card that shows/hides content on title-band click, with an animation."""

    def __init__(self, title: str, collapsed: bool = False,
                 accent_color: str = None, parent=None, icon: str = None):
        super().__init__(parent)
        self._collapsed = collapsed
        self._accent = accent_color or ACCENT

        outer = QVBoxLayout(self)
        outer.setSpacing(0)
        outer.setContentsMargins(0, 14, 0, 0)

        frame = _make_card_frame()
        fbox = QVBoxLayout(frame)
        fbox.setSpacing(0)
        fbox.setContentsMargins(0, 0, 0, 0)

        self._hdr, _row, _lbl, self._arrow = _make_header(
            title, icon, clickable=True, accent=self._accent)
        self._hdr.mousePressEvent = lambda _e: self._toggle()
        fbox.addWidget(self._hdr)

        self._content = QWidget()
        self._content.setObjectName("CardBody")
        self._content.setStyleSheet(f"#CardBody {{ background: {FG}; border: none; }}")
        self._cl = QVBoxLayout(self._content)
        self._cl.setSpacing(8)
        self._cl.setContentsMargins(13, 11, 13, 13)
        fbox.addWidget(self._content)

        outer.addWidget(frame)

        self._anim = QPropertyAnimation(self._content, b"maximumHeight", self)
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._apply_initial()

    def _apply_initial(self):
        if self._collapsed:
            self._content.setMaximumHeight(0)
            self._content.setVisible(False)
            self._arrow.setPixmap(icons.pixmap("chevron", DIM, 14))
            self._hdr.setStyleSheet(
                f"QWidget#CardHdr {{ background: transparent; border: none; }}")
        else:
            self._content.setMaximumHeight(16777215)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._arrow.setPixmap(
            icons.pixmap("chevron" if self._collapsed else "chevron_down", DIM, 14))
        self._hdr.setStyleSheet(
            "QWidget#CardHdr { background: transparent; border: none; }"
            if self._collapsed else
            f"QWidget#CardHdr {{ background: transparent; border: none;"
            f" border-bottom: 1px solid {BORDER}; }}")
        try:
            self._anim.stop()
            if self._collapsed:
                self._anim.setStartValue(self._content.sizeHint().height())
                self._anim.setEndValue(0)
                try:
                    self._anim.finished.disconnect()
                except Exception:
                    pass
                self._anim.finished.connect(lambda: self._content.setVisible(False))
                self._anim.start()
            else:
                self._content.setVisible(True)
                self._content.setMaximumHeight(0)
                self._anim.setStartValue(0)
                self._anim.setEndValue(self._content.sizeHint().height())
                try:
                    self._anim.finished.disconnect()
                except Exception:
                    pass
                self._anim.finished.connect(
                    lambda: self._content.setMaximumHeight(16777215))
                self._anim.start()
        except Exception:
            # Fallback: no animation.
            self._content.setVisible(not self._collapsed)
            self._content.setMaximumHeight(
                0 if self._collapsed else 16777215)

    def addWidget(self, w):
        self._cl.addWidget(w)

    def addLayout(self, lay):
        self._cl.addLayout(lay)


# ── Legacy bar-style collapsible (Predict panel) ────────────────────────────

class CollapsibleSection(QWidget):
    """Bar-style collapsible section. Use CollapsibleCard for new code."""

    def __init__(self, title: str, collapsed: bool = False, parent=None):
        super().__init__(parent)

        vbox = QVBoxLayout(self)
        vbox.setSpacing(0)
        vbox.setContentsMargins(0, 0, 0, 0)

        self._hdr = QWidget()
        self._hdr.setCursor(Qt.CursorShape.PointingHandCursor)
        hrow = QHBoxLayout(self._hdr)
        hrow.setContentsMargins(0, 22, 0, 8)
        hrow.setSpacing(9)

        self._lbl = QLabel(title.upper())
        self._lbl.setStyleSheet(
            f"color: {LABEL}; font-size: 11px; font-weight: 700; letter-spacing: 1.4px;")
        self._arrow = QLabel()
        self._arrow.setPixmap(icons.pixmap("chevron_down", DIM, 13))
        self._arrow.setFixedSize(14, 14)

        hrow.addWidget(self._lbl)
        hrow.addStretch()
        hrow.addWidget(self._arrow)
        self._hdr.mousePressEvent = lambda _e: self._toggle()
        vbox.addWidget(self._hdr)

        self._content = QWidget()
        self._cl = QVBoxLayout(self._content)
        self._cl.setSpacing(6)
        self._cl.setContentsMargins(0, 0, 0, 6)
        vbox.addWidget(self._content)

        if collapsed:
            self._content.setVisible(False)
            self._arrow.setPixmap(icons.pixmap("chevron", DIM, 13))

    def _toggle(self):
        visible = not self._content.isVisible()
        self._content.setVisible(visible)
        self._arrow.setPixmap(
            icons.pixmap("chevron_down" if visible else "chevron", DIM, 13))

    def addWidget(self, w):
        self._cl.addWidget(w)

    def addLayout(self, lay):
        self._cl.addLayout(lay)

    def _on_toggle(self, checked: bool = False):
        pass
