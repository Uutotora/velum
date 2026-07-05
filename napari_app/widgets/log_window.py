"""Shared floating log window — used by both Train and Predict panels."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor

from napari_app.theme import BG, BORDER, BORDER_STRONG, TEXT, DIM, ACCENT, CONSOLE, WIDGET_SS


class LogWindow(QWidget):
    def __init__(self):
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("CellSeg1 — Log")
        self.resize(720, 320)
        self.setMinimumSize(280, 120)
        self.setStyleSheet(WIDGET_SS)

        L = QVBoxLayout()
        L.setContentsMargins(0, 0, 0, 0)
        L.setSpacing(0)

        # ── Header bar ────────────────────────────────────────────────────────
        from PyQt6.QtWidgets import QFrame
        hdr = QWidget()
        hdr.setFixedHeight(36)
        hdr.setStyleSheet(f"background:{BG}; border-bottom:1px solid {BORDER};")
        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(13, 0, 10, 0)
        hdr_row.setSpacing(9)

        tick = QFrame()
        tick.setFixedSize(3, 12)
        tick.setStyleSheet(f"background:{ACCENT}; border-radius:2px;")

        lbl = QLabel("LOG")
        lbl.setStyleSheet(
            f"color:{DIM}; font-size:10px; font-weight:700; letter-spacing:1.4px; background:transparent;")

        _btn_ss = (
            f"QPushButton {{ color:{DIM}; background:transparent;"
            f" border:1px solid {BORDER_STRONG}; border-radius:5px;"
            f" padding:0 10px; font-size:11px; font-weight:600; }}"
            f"QPushButton:hover {{ color:{TEXT}; border-color:{ACCENT}; }}"
            f"QPushButton:checked {{ color:{TEXT}; border-color:{ACCENT}; }}")

        wrap_btn = QPushButton("Wrap")
        wrap_btn.setCheckable(True)
        wrap_btn.setChecked(True)
        wrap_btn.setFixedHeight(24)
        wrap_btn.setStyleSheet(_btn_ss)
        wrap_btn.toggled.connect(self._set_wrap)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(24)
        clear_btn.setStyleSheet(_btn_ss)

        hdr_row.addWidget(tick)
        hdr_row.addWidget(lbl)
        hdr_row.addStretch()
        hdr_row.addWidget(wrap_btn)
        hdr_row.addWidget(clear_btn)
        hdr.setLayout(hdr_row)
        L.addWidget(hdr)

        # ── Text area ─────────────────────────────────────────────────────────
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._text.setStyleSheet(
            f"border:none; border-radius:0; padding:8px 12px;"
            f"background:{CONSOLE}; color:{TEXT};"
        )
        L.addWidget(self._text)

        self.setLayout(L)
        clear_btn.clicked.connect(self._text.clear)

    def append(self, text: str):
        self._text.append(text)
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_wrap(self, checked: bool):
        self._text.setLineWrapMode(
            QTextEdit.LineWrapMode.WidgetWidth if checked
            else QTextEdit.LineWrapMode.NoWrap
        )

    _placed = False

    def _place(self):
        if self._placed:
            return
        self._placed = True
        try:
            from PyQt6.QtGui import QGuiApplication
            geo = QGuiApplication.primaryScreen().availableGeometry()
            # Dock to the bottom-left of the screen, clear of the napari panel.
            self.move(geo.left() + 24, geo.bottom() - self.height() - 40)
        except Exception:
            pass

    def show(self):
        self._place()
        super().show()

    def show_and_raise(self):
        self._place()
        super().show()
        self.raise_()
        self.activateWindow()


_instance: LogWindow | None = None


def get_log_window() -> LogWindow:
    global _instance
    if _instance is None:
        _instance = LogWindow()
    return _instance
