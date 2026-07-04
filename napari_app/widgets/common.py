from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from napari_app.theme import BORDER, DIM, LABEL, TEXT, ACCENT, FG, CARD_HEADER


def section_header(text: str) -> QWidget:
    """Bar-style section label (kept for compatibility with Predict panel)."""
    container = QWidget()
    container.setContentsMargins(0, 0, 0, 0)
    row = QHBoxLayout()
    row.setContentsMargins(0, 24, 0, 8)
    row.setSpacing(8)

    bar = QFrame()
    bar.setFixedWidth(2)
    bar.setFixedHeight(12)
    bar.setStyleSheet(f"background: {ACCENT}; border-radius: 1px;")

    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {LABEL}; font-size: 11px; font-weight: 700; letter-spacing: 1.6px;"
    )
    row.addWidget(bar)
    row.addWidget(lbl)
    row.addStretch()
    container.setLayout(row)
    return container


def divider() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER}; border: none; margin: 8px 0;")
    return f


def param_row(label_text: str, widget, tip: str = "", label_width: int = 110) -> QHBoxLayout:
    """Label–value pair, 3px vertical padding each side."""
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


# ── Card widgets ──────────────────────────────────────────────────────────────

def _make_card_frame(accent: str) -> QFrame:
    """QFrame styled as a card: a thin border fully enclosing the block.

    ``accent`` is kept for API compatibility but no longer drawn as a heavy
    left bar — the accent now lives only in the header band, so every card is
    cleanly framed on all four sides.
    """
    f = QFrame()
    f.setObjectName("CardFrame")
    f.setStyleSheet(
        f"QFrame#CardFrame {{"
        f"  background: {FG};"
        f"  border: 1px solid {BORDER};"
        f"  border-radius: 8px;"
        f"}}"
    )
    return f


def _make_header_band(title: str) -> QWidget:
    """Title band inside a card."""
    hdr = QWidget()
    hdr.setObjectName("CardHdr")
    hdr.setStyleSheet(
        f"QWidget#CardHdr {{"
        f"  background: {CARD_HEADER};"
        f"  border-bottom: 1px solid {BORDER};"
        f"  border-top-left-radius: 6px;"
        f"  border-top-right-radius: 6px;"
        f"}}"
    )
    row = QHBoxLayout()
    row.setContentsMargins(12, 8, 12, 8)
    row.setSpacing(8)
    lbl = QLabel(title.upper())
    lbl.setStyleSheet(
        f"color: {LABEL}; font-size: 11px; font-weight: 700; letter-spacing: 1.5px;"
        f"background: transparent; border: none;"
    )
    row.addWidget(lbl)
    return hdr, row, lbl


class SectionCard(QWidget):
    """Flat card: colored left border + title band + content.
    accent_color: left border color (defaults to ACCENT blue)."""

    def __init__(self, title: str, accent_color: str = None, parent=None):
        super().__init__(parent)
        color = accent_color or ACCENT

        outer = QVBoxLayout()
        outer.setSpacing(0)
        outer.setContentsMargins(0, 16, 0, 0)

        frame = _make_card_frame(color)
        fbox = QVBoxLayout()
        fbox.setSpacing(0)
        fbox.setContentsMargins(0, 0, 0, 0)

        hdr, hrow, _ = _make_header_band(title)
        hrow.addStretch()
        hdr.setLayout(hrow)
        fbox.addWidget(hdr)

        self._cw = QWidget()
        self._cw.setObjectName("CardBody")
        self._cw.setStyleSheet(f"#CardBody {{ background: {FG}; }}")
        self._cl = QVBoxLayout()
        self._cl.setSpacing(7)
        self._cl.setContentsMargins(12, 10, 12, 12)
        self._cw.setLayout(self._cl)
        fbox.addWidget(self._cw)

        frame.setLayout(fbox)
        outer.addWidget(frame)
        self.setLayout(outer)

    def addWidget(self, w):
        self._cl.addWidget(w)

    def addLayout(self, lay):
        self._cl.addLayout(lay)


class CollapsibleCard(QWidget):
    """Card that shows/hides content on title-band click."""

    def __init__(self, title: str, collapsed: bool = False,
                 accent_color: str = None, parent=None):
        super().__init__(parent)
        self._collapsed = collapsed
        self._accent = accent_color or ACCENT

        outer = QVBoxLayout()
        outer.setSpacing(0)
        outer.setContentsMargins(0, 16, 0, 0)

        self._frame = _make_card_frame(self._accent)
        fbox = QVBoxLayout()
        fbox.setSpacing(0)
        fbox.setContentsMargins(0, 0, 0, 0)

        # Clickable header band
        self._hdr = QWidget()
        self._hdr.setObjectName("ColCardHdr")
        self._hdr.setCursor(Qt.CursorShape.PointingHandCursor)

        hrow = QHBoxLayout()
        hrow.setContentsMargins(12, 8, 12, 8)
        hrow.setSpacing(8)
        t = QLabel(title.upper())
        t.setStyleSheet(
            f"color: {LABEL}; font-size: 11px; font-weight: 700; letter-spacing: 1.5px;"
            f"background: transparent; border: none;"
        )
        self._arrow = QLabel("▾")
        self._arrow.setStyleSheet(
            f"color: {DIM}; font-size: 12px; background: transparent; border: none;"
        )
        hrow.addWidget(t)
        hrow.addStretch()
        hrow.addWidget(self._arrow)
        self._hdr.setLayout(hrow)
        self._hdr.mousePressEvent = lambda _e: self._toggle()
        fbox.addWidget(self._hdr)

        # Content
        self._content = QWidget()
        self._content.setObjectName("CardBody")
        self._content.setStyleSheet(f"#CardBody {{ background: {FG}; }}")
        self._cl = QVBoxLayout()
        self._cl.setSpacing(7)
        self._cl.setContentsMargins(12, 10, 12, 12)
        self._content.setLayout(self._cl)
        fbox.addWidget(self._content)

        self._frame.setLayout(fbox)
        outer.addWidget(self._frame)
        self.setLayout(outer)

        self._apply_state()

    def _apply_state(self):
        if self._collapsed:
            self._content.hide()
        else:
            self._content.show()
        self._arrow.setText("▸" if self._collapsed else "▾")
        if self._collapsed:
            hdr_style = (
                f"QWidget#ColCardHdr {{"
                f"  background: {CARD_HEADER};"
                f"  border-radius: 6px;"
                f"}}"
            )
        else:
            hdr_style = (
                f"QWidget#ColCardHdr {{"
                f"  background: {CARD_HEADER};"
                f"  border-bottom: 1px solid {BORDER};"
                f"  border-top-left-radius: 6px;"
                f"  border-top-right-radius: 6px;"
                f"}}"
            )
        self._hdr.setStyleSheet(hdr_style)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._apply_state()

    def addWidget(self, w):
        self._cl.addWidget(w)

    def addLayout(self, lay):
        self._cl.addLayout(lay)


# ── Legacy bar-style collapsible (Predict panel) ──────────────────────────────

class CollapsibleSection(QWidget):
    """Bar-style collapsible section. Use CollapsibleCard for new code."""

    def __init__(self, title: str, collapsed: bool = False, parent=None):
        super().__init__(parent)

        vbox = QVBoxLayout()
        vbox.setSpacing(0)
        vbox.setContentsMargins(0, 0, 0, 0)

        self._hdr = QWidget()
        self._hdr.setCursor(Qt.CursorShape.PointingHandCursor)
        hrow = QHBoxLayout()
        hrow.setContentsMargins(0, 24, 0, 8)
        hrow.setSpacing(8)

        bar = QFrame()
        bar.setFixedWidth(2)
        bar.setFixedHeight(12)
        bar.setStyleSheet(f"background: {ACCENT}; border-radius: 1px;")

        self._lbl = QLabel(title.upper())
        self._lbl.setStyleSheet(
            f"color: {LABEL}; font-size: 11px; font-weight: 700; letter-spacing: 1.6px;"
        )
        self._arrow = QLabel("▾")
        self._arrow.setStyleSheet(f"color: {DIM}; font-size: 12px;")

        hrow.addWidget(bar)
        hrow.addWidget(self._lbl)
        hrow.addStretch()
        hrow.addWidget(self._arrow)
        self._hdr.setLayout(hrow)
        self._hdr.mousePressEvent = lambda _e: self._toggle()
        vbox.addWidget(self._hdr)

        self._content = QWidget()
        self._cl = QVBoxLayout()
        self._cl.setSpacing(5)
        self._cl.setContentsMargins(0, 0, 0, 6)
        self._content.setLayout(self._cl)
        vbox.addWidget(self._content)
        self.setLayout(vbox)

        if collapsed:
            self._content.setVisible(False)
            self._arrow.setText("▸")

    def _toggle(self):
        visible = not self._content.isVisible()
        self._content.setVisible(visible)
        self._arrow.setText("▾" if visible else "▸")

    def addWidget(self, w):
        self._cl.addWidget(w)

    def addLayout(self, lay):
        self._cl.addLayout(lay)

    def _on_toggle(self, checked: bool = False):
        pass
