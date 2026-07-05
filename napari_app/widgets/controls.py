"""Shared, product-grade controls.

``Combo`` is a drop-in QComboBox that:
- uses a real Qt popup (a QListView) so the styled, graphite drop-down shows on
  every platform (macOS otherwise falls back to a native, unstyled menu);
- clamps that popup to the width of the box it drops from;
- draws a crisp chevron icon (a transparent overlay) instead of the flaky CSS
  triangle, so the arrow never renders as a stray dash.

Purely presentational — the QComboBox API is unchanged, so all existing
populate/currentData/signal code keeps working (and it stays a QComboBox
subclass, so the wheel-guard isinstance check still matches).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QListView, QLabel

from napari_app import icons
from napari_app.theme import LABEL


class Combo(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setView(QListView(self))
        self._chev = QLabel(self)
        self._chev.setPixmap(icons.pixmap("chevron_down", LABEL, 13))
        self._chev.setFixedSize(14, 14)
        self._chev.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._chev.setStyleSheet("background:transparent; border:none;")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._chev.move(self.width() - 23, (self.height() - self._chev.height()) // 2)
        self._chev.raise_()

    def showPopup(self):
        super().showPopup()
        try:
            self.view().window().setFixedWidth(self.width())
        except Exception:
            pass
