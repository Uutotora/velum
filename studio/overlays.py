"""CellSeg1 Studio — overlay surfaces: Logs console, command palette, toast.

Created as children of the main window; the window shows/hides and positions
them. Logs and the palette still render static ``demo`` content — buttons
give visual feedback only. The Assistant drawer (real chat, real diagnostics,
real model management) has grown into its own module,
``studio/assistant_panel.py`` — imported from there, not here; see its
docstring.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QLineEdit,
    QScrollArea,
)

from studio import icons
from studio import theme, demo
from studio.components import IconButton, hline, label


def _scroll(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setWidget(inner)
    return sa


class LogsConsole(QFrame):
    """Bottom console with a monospaced log."""

    HEIGHT = 210

    def __init__(self, parent: QWidget, t: dict):
        super().__init__(parent)
        self._t = t
        self.setFixedHeight(self.HEIGHT)
        # Qualified selector: an unqualified background+border rule here
        # would cascade to every descendant that doesn't more specifically
        # override `border` (bare QWidget/QLabel have no such override) —
        # confirmed as a real, visible stray-line bug for this exact
        # pattern on AssistantDrawer (see its own comment); fixed
        # proactively here too rather than waiting to find it by screenshot
        # once Logs is wired for real.
        self.setObjectName("LogsConsole")
        self.setStyleSheet(
            f"QFrame#LogsConsole{{background:{t['surface']}; border-top:1px solid {t['border']};}}")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        head = QWidget()
        head.setStyleSheet(f"background:{t['inset']};")
        hr = QHBoxLayout(head)
        hr.setContentsMargins(14, 9, 12, 9)
        hr.setSpacing(10)
        hr.addWidget(label("LOGS", 11.5, t["text_subtle"], 600, 0.6))
        from studio.components import Badge
        hr.addWidget(Badge("run · nuclei-dapi-r8", t))
        hr.addStretch(1)
        hr.addWidget(IconButton("close", t, 27, "Close", self.hide))
        v.addWidget(head)
        v.addWidget(hline(t))

        body = QWidget()
        body.setStyleSheet(f"background:{t['scope']};")
        bv = QVBoxLayout(body)
        bv.setContentsMargins(14, 10, 14, 10)
        bv.setSpacing(3)
        lv_col = {"ok": t["signal"], "info": "#8b9bf4", "warn": t["warning"]}
        for ts, lv, msg in demo.LOGS:
            lvtxt = lv.upper().ljust(5).replace(" ", "&nbsp;")
            line = QLabel(
                f"<span style='color:#5b6472'>{ts}</span>&nbsp;&nbsp;"
                f"<span style='color:{lv_col[lv]};font-weight:700'>{lvtxt}</span>&nbsp;&nbsp;"
                f"<span style='color:#aeb9c7'>{msg}</span>")
            line.setStyleSheet(f"font-family:{theme.MONO}; font-size:11.5px;")
            bv.addWidget(line)
        bv.addStretch(1)
        v.addWidget(_scroll(body), 1)
        self.hide()

    def place(self):
        p = self.parentWidget()
        if p:
            from studio.components import Sidebar
            x = Sidebar.WIDTH
            self.setGeometry(x, p.height() - self.HEIGHT, p.width() - x, self.HEIGHT)


class CommandPalette(QWidget):
    """Centered ⌘K command palette over a scrim."""

    def __init__(self, parent: QWidget, t: dict):
        super().__init__(parent)
        self._t = t
        self.setStyleSheet("background:rgba(8,10,20,0.34);")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 96, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self._panel())
        self.hide()

    def _panel(self) -> QFrame:
        t = self._t
        panel = QFrame()
        panel.setFixedWidth(560)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setObjectName("PalettePanel")   # qualified -- see AssistantDrawer's comment
        panel.setStyleSheet(
            f"QFrame#PalettePanel{{background:{t['surface']}; border:1px solid {t['border_strong']};"
            f" border-radius:14px;}}")
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        inp_wrap = QWidget()
        ir = QHBoxLayout(inp_wrap)
        ir.setContentsMargins(17, 15, 17, 15)
        ir.setSpacing(11)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("diagnose", t["text_muted"], 17))
        ir.addWidget(ic)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Search actions, projects, engines…")
        self.input.setStyleSheet("QLineEdit{border:none; background:transparent; font-size:15px;}")
        ir.addWidget(self.input, 1)
        esc = QLabel("ESC")
        esc.setStyleSheet(
            f"color:{t['text_muted']}; font-family:{theme.MONO}; font-size:10.5px;"
            f"border:1px solid {t['border']}; border-radius:5px; padding:2px 6px;")
        ir.addWidget(esc)
        v.addWidget(inp_wrap)
        v.addWidget(hline(t))

        section = None
        for i, (sec, icon_name, text, hint) in enumerate(demo.PALETTE):
            if sec != section:
                section = sec
                sl = label(sec.upper(), 10.5, t["text_muted"], 600, 0.6)
                sl.setContentsMargins(17, 12, 17, 5)
                v.addWidget(sl)
            v.addWidget(self._item(icon_name, text, hint, highlighted=(i == 0)))

        foot = QWidget()
        fr = QHBoxLayout(foot)
        fr.setContentsMargins(17, 10, 17, 10)
        fr.setSpacing(16)
        for k, act in [("↑↓", "navigate"), ("⏎", "run"), ("esc", "close")]:
            fr.addWidget(label(f"<span style='font-family:{theme.MONO}'>{k}</span> {act}", 11, t["text_muted"]))
        fr.addStretch(1)
        v.addWidget(hline(t))
        v.addWidget(foot)
        return panel

    def _item(self, icon_name: str, text: str, hint: str, highlighted: bool) -> QFrame:
        t = self._t
        row = QFrame()
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setStyleSheet(
            (f"QFrame{{background:{t['primary_weak']};}}" if highlighted else "QFrame{background:transparent;}") +
            f"QFrame:hover{{background:{t['primary_weak']};}}")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(17, 10, 17, 10)
        lay.setSpacing(12)
        ic = QLabel()
        ic.setPixmap(icons.pixmap(icon_name, t["primary"] if highlighted else t["text_muted"], 16))
        lay.addWidget(ic)
        lay.addWidget(label(text, 13.5, t["text"] if highlighted else t["text_subtle"]))
        lay.addStretch(1)
        if hint:
            lay.addWidget(label(hint, 10.5, t["text_muted"]))
        return row

    def place(self):
        p = self.parentWidget()
        if p:
            self.setGeometry(0, 0, p.width(), p.height())

    def open(self):
        self.place()
        self.show()
        self.raise_()
        self.input.setFocus()

    def mousePressEvent(self, e):
        # click on the scrim (outside the panel) closes
        child = self.childAt(e.position().toPoint())
        if child is None:
            self.hide()
        super().mousePressEvent(e)


class Toast(QFrame):
    """Bottom-right success toast. Static by default; ``announce()`` for real use."""

    def __init__(self, parent: QWidget, t: dict):
        super().__init__(parent)
        self._t = t
        self.setObjectName("Toast")   # qualified -- see AssistantDrawer's comment
        self.setStyleSheet(
            f"QFrame#Toast{{background:{t['surface']}; border:1px solid {t['border']};"
            f"border-left:3px solid {t['success']}; border-radius:11px;}}")
        row = QHBoxLayout(self)
        row.setContentsMargins(15, 12, 15, 12)
        row.setSpacing(12)
        ic = QLabel()
        ic.setFixedSize(30, 30)
        ic.setPixmap(icons.pixmap("check", t["success"], 16))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic.setStyleSheet(f"background:{t['success_weak']}; border-radius:8px;")
        row.addWidget(ic)
        col = QVBoxLayout()
        col.setSpacing(1)
        self._title = label("Segmentation complete", 13, t["text"], 600)
        col.addWidget(self._title)
        self._subtitle = label("247 cells · F1 0.94 vs ground truth · 3.2 s", 11.5, t["text_muted"])
        self._subtitle.setWordWrap(True)
        self._subtitle.setMaximumWidth(280)  # wrap long messages instead of clipping/overflowing
        col.addWidget(self._subtitle)
        row.addLayout(col)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
        self.hide()

    def place(self):
        p = self.parentWidget()
        if p:
            self.adjustSize()
            self.move(p.width() - self.width() - 22, p.height() - self.height() - 22)

    def announce(self, title: str, subtitle: str, duration_ms: int = 3200) -> None:
        """Show a real, timed confirmation with the given text."""
        self._title.setText(title)
        self._subtitle.setText(subtitle)
        self.place()
        self.show()
        self.raise_()
        self._hide_timer.start(duration_ms)
