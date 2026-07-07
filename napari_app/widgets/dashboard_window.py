"""Shared floating "Dashboard" window — Aim's own run-history UI, embedded in
an in-app window when `PyQt6-WebEngine` is installed, with an "Open in
browser" escape hatch that always works (even without that extra) since it
just launches the user's system browser against the same local URL.

Follows the exact singleton pattern `log_window.py`/`measurements_window.py`
already use: one shared instance per process, reused across every "Dashboard"
button in Predict/Train/Assistant so they all point at the same window
instead of spawning three independent ones. Because it's a singleton, whether
the embedded view or the fallback message gets built is re-checked on every
open (`_upgrade_to_embedded_view_if_possible`), not only once at first
construction — otherwise installing `PyQt6-WebEngine` into an already-running
app would keep showing the fallback until the whole app was restarted.
"""
from __future__ import annotations

import webbrowser

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QTimer

from napari_app.theme import (
    BG, BORDER, BORDER_STRONG, TEXT, DIM, ACCENT, CONSOLE, WIDGET_SS,
    BTN_PRIMARY, BTN_SECONDARY,
)

# aim up spawns a real web server in a subprocess; the very first load
# attempt right after starting it can race the server still coming up
# (connection refused, not yet listening) -- retry a few times rather than
# showing a permanently blank view for what's usually a ~1-2s startup.
_LOAD_RETRY_MS = 700
_LOAD_MAX_ATTEMPTS = 8


def _has_webengine() -> bool:
    try:
        import PyQt6.QtWebEngineWidgets  # noqa: F401
        return True
    except Exception:
        return False


class DashboardWindow(QWidget):
    def __init__(self):
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("CellSeg1 — Dashboard")
        self.resize(960, 640)
        self.setMinimumSize(420, 320)
        self.setStyleSheet(WIDGET_SS)

        L = QVBoxLayout(); L.setContentsMargins(0, 0, 0, 0); L.setSpacing(0)

        hdr = QWidget()
        hdr.setFixedHeight(36)
        hdr.setStyleSheet(f"background:{BG}; border-bottom:1px solid {BORDER};")
        hdr_row = QHBoxLayout(); hdr_row.setContentsMargins(13, 0, 10, 0); hdr_row.setSpacing(9)
        lbl = QLabel("DASHBOARD")
        lbl.setStyleSheet(
            f"color:{DIM}; font-size:10px; font-weight:700; letter-spacing:1.4px; background:transparent;")
        self._status = QLabel("")
        self._status.setStyleSheet(f"color:{DIM}; font-size:10.5px; background:transparent;")
        btn_ss = (
            f"QPushButton {{ color:{DIM}; background:transparent;"
            f" border:1px solid {BORDER_STRONG}; border-radius:5px;"
            f" padding:0 10px; font-size:11px; font-weight:600; }}"
            f"QPushButton:hover {{ color:{TEXT}; border-color:{ACCENT}; }}")
        self._browser_btn = QPushButton("Open in browser")
        self._browser_btn.setFixedHeight(24)
        self._browser_btn.setStyleSheet(btn_ss)
        self._browser_btn.setEnabled(False)   # nothing to open until a URL exists
        self._browser_btn.setToolTip("Waiting for the dashboard to start…")
        self._browser_btn.clicked.connect(self._open_in_browser)
        hdr_row.addWidget(lbl)
        hdr_row.addWidget(self._status, stretch=1)
        hdr_row.addWidget(self._browser_btn)
        hdr.setLayout(hdr_row)
        L.addWidget(hdr)

        self._url: str | None = None
        self._view = None
        self._fallback = None
        self._load_attempt = 0
        if _has_webengine():
            self._build_embedded_view(L)
        else:
            self._build_fallback(L)

        self.setLayout(L)

    def _build_fallback(self, layout: QVBoxLayout):
        self._fallback = QWidget()
        fl = QVBoxLayout(self._fallback)
        fl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg = QLabel(
            "Embedded view needs the optional PyQt6-WebEngine package\n"
            "(pip install -e \".[tracking-ui]\").\n\n"
            "Use “Open in browser” above to view the dashboard "
            "in your system browser instead.")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet(f"color:{DIM}; font-size:12px; background:transparent;")
        fl.addWidget(msg)
        self._fallback.setStyleSheet(f"background:{CONSOLE};")
        layout.addWidget(self._fallback)

    def _build_embedded_view(self, layout: QVBoxLayout):
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        self._view = QWebEngineView()
        self._view.loadFinished.connect(self._on_embedded_load_finished)
        layout.addWidget(self._view)

    def _upgrade_to_embedded_view_if_possible(self):
        """PyQt6-WebEngine can be installed into a running app's env *after*
        this window was first built — its absence was otherwise baked into
        the layout once, at construction, permanently (this window is a
        singleton), so the fallback message would keep showing even once
        the package genuinely becomes available, until the whole app was
        restarted. Re-check on every open instead and swap the fallback out
        for a real view the moment it's possible, no restart needed."""
        if self._view is not None or not _has_webengine():
            return
        if self._fallback is not None:
            self.layout().removeWidget(self._fallback)
            self._fallback.deleteLater()
            self._fallback = None
        self._build_embedded_view(self.layout())

    def open_dashboard(self):
        """Start Aim's dashboard server (if needed) and point this window
        at it — the embedded view if available, else just enables the
        "Open in browser" button and says so."""
        self._upgrade_to_embedded_view_if_possible()
        from napari_app.core import experiment_tracking as tracking
        try:
            self._url = tracking.ensure_dashboard_running()
        except Exception as e:
            self._url = None
            self._browser_btn.setEnabled(False)
            self._browser_btn.setToolTip("Waiting for the dashboard to start…")
            self._status.setText(f"✗ {e}")
            return
        self._status.setText(self._url)
        self._browser_btn.setEnabled(True)
        self._browser_btn.setToolTip(f"Open {self._url} in your system browser")
        if self._view is not None:
            self._load_attempt = 0
            self._load_embedded_view()

    def _load_embedded_view(self):
        from PyQt6.QtCore import QUrl
        self._view.setUrl(QUrl(self._url))

    def _on_embedded_load_finished(self, ok: bool):
        if ok or self._url is None:
            return
        self._load_attempt += 1
        if self._load_attempt >= _LOAD_MAX_ATTEMPTS:
            self._status.setText(
                f"{self._url}  (embedded load failed — try “Open in browser”)")
            return
        QTimer.singleShot(_LOAD_RETRY_MS, self._load_embedded_view)

    def _open_in_browser(self):
        if self._url:
            webbrowser.open(self._url)

    _placed = False

    def _place(self):
        if self._placed:
            return
        self._placed = True
        try:
            from PyQt6.QtGui import QGuiApplication
            geo = QGuiApplication.primaryScreen().availableGeometry()
            self.move(geo.center().x() - self.width() // 2,
                     geo.center().y() - self.height() // 2)
        except Exception:
            pass

    def show_and_raise(self):
        self._place()
        self.open_dashboard()
        self.show()
        self.raise_()
        self.activateWindow()


_instance: DashboardWindow | None = None


def get_dashboard_window() -> DashboardWindow:
    global _instance
    if _instance is None:
        _instance = DashboardWindow()
    return _instance
