from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QToolButton, QSizePolicy,
)
from napari_app.theme import BORDER, DIM, TEXT


class CollapsibleSection(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        vbox = QVBoxLayout()
        vbox.setSpacing(0)
        vbox.setContentsMargins(0, 4, 0, 4)

        self._toggle = QToolButton()
        self._toggle.setText(f"▾  {title}")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.clicked.connect(self._on_toggle)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        vbox.addWidget(self._toggle)

        self._content = QWidget()
        self._cl = QVBoxLayout()
        self._cl.setSpacing(5)
        self._cl.setContentsMargins(0, 4, 0, 0)
        self._content.setLayout(self._cl)
        vbox.addWidget(self._content)

        self.setLayout(vbox)

    def addWidget(self, w):
        self._cl.addWidget(w)

    def addLayout(self, l):
        self._cl.addLayout(l)

    def _on_toggle(self, checked):
        self._content.setVisible(checked)
        title = self._toggle.text()[3:]
        self._toggle.setText(f"{'▾' if checked else '▸'}  {title}")


def divider():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {BORDER}; border: none; margin: 0;")
    return f


def param_row(label_text, widget, tip="", min_label_width=115):
    row = QHBoxLayout()
    row.setSpacing(8)
    lbl = QLabel(label_text)
    lbl.setStyleSheet(f"color: {DIM}; font-size: 12px;")
    lbl.setMinimumWidth(min_label_width)
    if tip:
        lbl.setToolTip(tip)
        widget.setToolTip(tip)
    row.addWidget(lbl)
    row.addWidget(widget)
    return row
