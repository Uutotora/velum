"""CellSeg1 Studio — the application shell and entry point.

This app is a faithful, native-Qt reproduction of the north-star mockup, with
functionality wired back tab by tab (see ``docstudio/`` — OVERVIEW,
ARCHITECTURE, BACKLOG, AGENT_PROMPT). Home/Projects are backed by a real
``ProjectController``; Models & Train and Dashboard are backed by a real
``TrainController``/``DashboardController`` (one-shot LoRA training, a real
trained-models list, real experiment history). Segment (the workspace canvas)
still renders static ``demo`` content pending its own tab — the flagship item
left in ``docstudio/BACKLOG.md``. No napari, no torch — those are reused
lazily, only inside the tab that needs them.

``StudioWindow`` owns a frameless, rounded window with our own dark title bar,
a navigation sidebar, a stack of screens (Home · Projects · Segment · Models &
Train · Dashboard), and overlay surfaces (Assistant drawer, Logs console, ⌘K
command palette, toast). The classic napari-plugin app
(``napari_app.main`` / ``run_napari.sh`` / ``cellseg1``) is untouched.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Make the repo root importable before the ``studio`` imports below, so
# ``python studio/app.py`` works from any cwd (``python -m`` would prepend the
# caller's cwd ahead of PYTHONPATH and import the wrong package). studio/app.py
# lives at <root>/studio/app.py → the repo root is two levels up.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QFontDatabase, QFont, QRegion, QPainterPath, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QApplication,
)

from studio import theme
from studio.components import Sidebar
from studio.project_controller import ProjectController
from studio.train_controller import TrainController
from studio.new_project_dialog import NewProjectDialog
from studio.screens import HomeScreen, ProjectsScreen
from studio.workspace import WorkspaceScreen
from studio.extra_screens import ModelsScreen, DashboardScreen
from studio.guide_screen import GuideScreen
from studio.overlays import AssistantDrawer, LogsConsole, CommandPalette, Toast
from studio.window_chrome import (
    TitleBar, install_corner_grips, layout_corner_grips,
)

_FONT_DIR = Path(__file__).parent / "fonts"
_CORNER_RADIUS = 12

# (key, icon, label, section). assistant/logs toggle overlays; the rest switch
# the main stack.
_NAV = [
    ("home",      "home",      "Home",           ""),
    ("projects",  "projects",  "Projects",       ""),
    ("workspace", "workspace", "Segment",        "Current project"),
    ("train",     "models",    "Models & Train", "Current project"),
    ("dashboard", "dashboard", "Dashboard",      "Current project"),
    ("assistant", "assistant", "Assistant",      "Tools"),
    ("logs",      "log",       "Logs",           "Tools"),
]
_STACK_KEYS = ("home", "projects", "workspace", "train", "dashboard", "guide")


def load_fonts() -> str:
    """Register the bundled Figtree faces; return the resolved family name."""
    family = "Figtree"
    for name in ("Figtree-Regular.ttf", "Figtree-SemiBold.ttf"):
        path = _FONT_DIR / name
        if path.exists():
            fid = QFontDatabase.addApplicationFont(str(path))
            fams = QFontDatabase.applicationFontFamilies(fid) if fid != -1 else []
            if fams:
                family = fams[0]
    return family


class StudioWindow(QMainWindow):
    """Frameless rounded window: title bar + sidebar + screen stack + overlays."""

    def __init__(self, theme_name: str = "dark",
                 project_controller: Optional[ProjectController] = None,
                 train_controller: Optional[TrainController] = None):
        super().__init__()
        self._theme_name = theme_name
        self._projects = project_controller or ProjectController()
        self._train = train_controller or TrainController()
        self._screens: dict[str, QWidget] = {}
        self.setWindowTitle("CellSeg1 Studio")
        self.resize(1320, 860)
        self.setMinimumSize(1040, 680)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)

        self._build_ui()
        self.apply_theme(theme_name)
        self._grips = install_corner_grips(self)
        self.navigate("home")

        QShortcut(QKeySequence("Ctrl+K"), self, activated=self._toggle_palette)
        QShortcut(QKeySequence("Meta+K"), self, activated=self._toggle_palette)
        QShortcut(QKeySequence("Escape"), self, activated=self._close_overlays)

    @property
    def tokens(self) -> dict:
        return theme.tokens_for(self._theme_name)

    # ── construction ────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        t = self.tokens
        central = QWidget()
        central.setObjectName("Central")
        central.setStyleSheet(f"#Central{{background:{t['bg']};}}")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(central)

        outer.addWidget(TitleBar(self, t, on_toggle_theme=self.toggle_theme))

        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self._sidebar = Sidebar(_NAV, t)
        self._sidebar.navigate.connect(self.navigate)
        self._sidebar.toggle_theme.connect(self.toggle_theme)
        self._sidebar.open_guide.connect(lambda: self.navigate("guide"))
        row.addWidget(self._sidebar)

        self._new_project_dialog = NewProjectDialog(
            self, t, self._projects.store, on_created=self._on_project_created)
        # built before _screens: Models & Train / Dashboard announce through it
        self._toast = Toast(self, t)

        self._stack = QStackedWidget()
        self._screens = {
            "home": HomeScreen(t, self._projects, self.navigate, self._open_project,
                               self._new_project_dialog.open),
            "projects": ProjectsScreen(t, self._projects, self.navigate, self._open_project,
                                      self._new_project_dialog.open),
            "workspace": WorkspaceScreen(t),
            "train": ModelsScreen(t, self._train, self._projects, self._toast.announce),
            "dashboard": DashboardScreen(t, self._train, self._projects, self._toast.announce),
            "guide": GuideScreen(t, self._projects, self.navigate, self._open_project,
                                 self._new_project_dialog.open),
        }
        # survives theme-toggle rebuilds, which recreate the workspace screen
        self._screens["workspace"].set_active_project(self._projects.get_active())
        for key in _STACK_KEYS:
            self._stack.addWidget(self._screens[key])
        row.addWidget(self._stack, 1)
        outer.addWidget(body, 1)

        # overlays (children of the window, positioned on resize / open)
        self._assistant = AssistantDrawer(self, t)
        self._logs = LogsConsole(self, t)
        self._palette = CommandPalette(self, t)
        self._overlays = [self._assistant, self._logs, self._palette, self._toast,
                           self._new_project_dialog]

    # ── navigation ──────────────────────────────────────────────────────────
    def navigate(self, key: str) -> None:
        if key.startswith("guide:"):
            self._screens["guide"].open_article(key.split(":", 1)[1])
            key = "guide"
        if key == "assistant":
            self._toggle_drawer(self._assistant)
            return
        if key == "logs":
            self._toggle_drawer(self._logs)
            return
        if key in self._screens:
            screen = self._screens[key]
            refresh = getattr(screen, "refresh", None)
            if refresh is not None:
                refresh()
            self._stack.setCurrentWidget(screen)
            self._sidebar.set_active(key)
            if key != "workspace":
                try:
                    from studio.motion import fade_in
                    fade_in(screen, 170)
                except Exception:
                    pass

    def _open_project(self, project_id: str) -> None:
        project = self._projects.set_active(project_id)
        self._screens["workspace"].set_active_project(project)
        self.navigate("workspace")

    def _on_project_created(self, project_id: str) -> None:
        project = self._projects.store.load(project_id)
        n = len(project.image_paths)
        self._toast.announce(
            "Project created",
            f"“{project.name}” · {n} image{'s' if n != 1 else ''} · {project.settings.engine}")
        self._open_project(project_id)

    def _toggle_drawer(self, drawer) -> None:
        # isHidden() is the explicit flag (works even before the window is shown)
        if not drawer.isHidden():
            drawer.hide()
        else:
            drawer.place()
            drawer.show()
            drawer.raise_()

    def _toggle_palette(self) -> None:
        if not self._palette.isHidden():
            self._palette.hide()
        else:
            self._palette.open()

    def _close_overlays(self) -> None:
        for o in (self._palette, self._assistant, self._logs, self._new_project_dialog):
            o.hide()

    # ── theming ─────────────────────────────────────────────────────────────
    def apply_theme(self, theme_name: str) -> None:
        self._theme_name = theme_name
        qapp = QApplication.instance()
        if qapp is not None:
            qapp.setStyleSheet(theme.build_qss(self.tokens))

    def toggle_theme(self) -> None:
        new = "light" if self._theme_name == "dark" else "dark"
        self._theme_name = new
        # Tear the old static UI down synchronously (setParent(None) removes it
        # from the child tree now; deleteLater frees it) so nothing double-stacks.
        old = self.takeCentralWidget()
        if old is not None:
            old.setParent(None)
            old.deleteLater()
        for o in getattr(self, "_overlays", []):
            o.setParent(None)
            o.deleteLater()
        for g in getattr(self, "_grips", []):
            g.setParent(None)
            g.deleteLater()
        self._build_ui()            # rebuild the static UI in the new palette
        self.apply_theme(new)
        self._grips = install_corner_grips(self)
        self.navigate("home")
        self._round()

    # ── window shape ────────────────────────────────────────────────────────
    def _round(self) -> None:
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width(), self.height()),
                            _CORNER_RADIUS, _CORNER_RADIUS)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._round()
        grips = getattr(self, "_grips", None)
        if grips:
            layout_corner_grips(self, grips)
        for o in (self._assistant, self._logs, self._palette, self._toast, self._new_project_dialog):
            if o.isVisible():
                o.place()

    def showEvent(self, e):
        super().showEvent(e)
        self._round()


def _install_exception_hook() -> None:
    """Log unhandled exceptions from Qt callbacks instead of the default hook.

    PyQt6 treats an exception that escapes a Qt-invoked Python callback
    (an event override, a signal slot) as fatal by default — it prints the
    traceback and then aborts the *entire process* rather than just failing
    that one interaction (confirmed the hard way: a stale
    QGraphicsDropShadowEffect touched from a hover `enterEvent` after its
    widget was torn down — see motion.py). That specific hazard is now
    guarded at the source; this is defense in depth for anything not yet
    found, so a future bug prints a traceback instead of a bare crash log.
    """
    import traceback

    def _hook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def main() -> None:
    """Launch CellSeg1 Studio (pure-design skeleton — no napari/torch needed)."""
    _install_exception_hook()
    app = QApplication.instance() or QApplication(sys.argv)
    family = load_fonts()
    app.setFont(QFont(family, 10))
    win = StudioWindow(theme_name="dark")
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
