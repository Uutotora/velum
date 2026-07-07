"""CellSeg1 Studio — the application shell and entry point.

``StudioWindow`` is a plain ``QMainWindow`` that *owns the top-level window*:
a navigation sidebar plus a stack of screens (Home, Projects, and the heavier
Workspace / Train / Dashboard / Assistant / Logs screens registered by
``main``). This is the architectural pivot away from "napari plugin in a dock"
— here napari's canvas is embedded as one component the shell hosts, not the
other way round.

Designed so the shell is importable and constructible headless (only a
``QApplication``, no display, no napari): the napari import and canvas
embedding live entirely inside :func:`main` / :func:`build_workspace`, never in
``StudioWindow.__init__``. That keeps the smoke tests light and the classic
``napari_app.main`` entry point completely untouched — launch that to revert.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Make the repo root importable *before* the ``napari_app`` imports below, so
# the app works when launched as a script (``python .../studio/app.py``) from
# any working directory — mirrors ``napari_app/main.py``. This is deliberate:
# ``python -m napari_app.studio.app`` injects the *current directory* at the
# front of ``sys.path`` ahead of ``PYTHONPATH``, so running it from a different
# checkout (e.g. the main tree while the code lives in a worktree) would import
# the wrong ``napari_app``. Running the file + bootstrapping here avoids that.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QLabel,
)
from PyQt6.QtGui import QFontDatabase, QFont

from napari_app.studio import theme
from napari_app.studio.components import Sidebar
from napari_app.studio.screens import HomeScreen, ProjectsScreen
from napari_app.studio.project import ProjectStore, default_store_root
from napari_app.studio.window_chrome import (
    TitleBar, install_corner_grips, layout_corner_grips,
)

_FONT_DIR = Path(__file__).parent / "fonts"

# Sidebar layout: (key, icon, label, section). Panel keys (assistant/logs) are
# toggled as overlays; the rest switch the main stack.
_NAV = [
    ("home",      "home",      "Home",           ""),
    ("projects",  "projects",  "Projects",       ""),
    ("workspace", "workspace", "Segment",        "Current project"),
    ("train",     "models",    "Models & Train", "Current project"),
    ("dashboard", "dashboard", "Dashboard",      "Current project"),
    ("assistant", "assistant", "Assistant",      "Tools"),
    ("logs",      "log",       "Logs",           "Tools"),
]


def load_fonts() -> str:
    """Register the bundled Figtree faces; return the resolved family name.

    Falls back silently to the system stack (the tokens already chain to
    ``-apple-system``) if the files are missing or Qt can't load them.
    """
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
    """Sidebar + stacked screens. Heavier screens are registered post-construction."""

    def __init__(self, store: ProjectStore, theme_name: str = "dark"):
        super().__init__()
        self._store = store
        self._theme_name = theme_name
        self._screens: dict[str, QWidget] = {}
        self.setWindowTitle("CellSeg1 Studio")
        self.resize(1320, 860)
        self.setMinimumSize(1040, 680)

        # Frameless: drop the native grey OS title bar and wear our own dark one
        # (with our own traffic lights), so the app reads as a product, not a
        # generic Qt window. Move/resize stay native via startSystemMove +
        # corner grips (see window_chrome).
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(central)

        # Title bar sits in its own holder so a theme switch can rebuild just it.
        self._titlebar_holder = QWidget()
        self._titlebar_lay = QVBoxLayout(self._titlebar_holder)
        self._titlebar_lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._titlebar_holder)

        body = QWidget()
        self._root = QHBoxLayout(body)          # sidebar + stack
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(0)
        outer.addWidget(body, 1)

        self._stack = QStackedWidget()
        self._build_chrome()
        self._build_titlebar()
        self.apply_theme(theme_name)
        self._grips = install_corner_grips(self)
        self.navigate("home")

    # ── construction ────────────────────────────────────────────────────────
    @property
    def tokens(self) -> dict:
        return theme.tokens_for(self._theme_name)

    def _build_chrome(self) -> None:
        """(Re)build the sidebar + the always-present Home/Projects screens."""
        t = self.tokens
        # clear previous root children (theme rebuild)
        while self._root.count():
            item = self._root.takeAt(0)
            w = item.widget()
            if w and w is not self._stack:
                w.setParent(None)      # remove from child tree synchronously
                w.deleteLater()

        self._sidebar = Sidebar(_NAV, t)
        self._sidebar.navigate.connect(self.navigate)
        self._sidebar.toggle_theme.connect(self.toggle_theme)
        self._root.addWidget(self._sidebar)
        self._root.addWidget(self._stack, 1)

        # Home + Projects are cheap and store-backed → owned by the shell.
        home = HomeScreen(self._store, t, on_new=self._new_project,
                          on_open=self.open_project, on_navigate=self.navigate)
        projects = ProjectsScreen(self._store, t, on_new=self._new_project,
                                  on_open=self.open_project)
        self.register_screen("home", home)
        self.register_screen("projects", projects)

    def _build_titlebar(self) -> None:
        """(Re)build the custom title bar for the current theme."""
        while self._titlebar_lay.count():
            item = self._titlebar_lay.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)      # remove from child tree synchronously
                w.deleteLater()
        self._titlebar_lay.addWidget(
            TitleBar(self, self.tokens, on_toggle_theme=self.toggle_theme))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        grips = getattr(self, "_grips", None)
        if grips:
            layout_corner_grips(self, grips)

    def register_screen(self, key: str, widget: QWidget) -> None:
        """Register (or replace) a screen widget under a nav ``key``."""
        if key in self._screens:
            old = self._screens[key]
            idx = self._stack.indexOf(old)
            if idx != -1:
                self._stack.removeWidget(old)
            old.deleteLater()
        self._screens[key] = widget
        self._stack.addWidget(widget)

    def placeholder(self, title: str, subtitle: str) -> QWidget:
        """A tidy 'not wired yet' screen for keys main() hasn't registered."""
        t = self.tokens
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h = QLabel(title)
        h.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.setStyleSheet(f"font-size:18px; font-weight:600; color:{t['text']};")
        s = QLabel(subtitle)
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s.setStyleSheet(f"font-size:13px; color:{t['text_muted']};")
        lay.addWidget(h)
        lay.addWidget(s)
        return w

    # ── navigation ──────────────────────────────────────────────────────────
    def navigate(self, key: str) -> None:
        # Assistant/Logs are overlay panels in the full app; if not registered
        # as stack screens, ignore here (main() wires the drawers).
        if key not in self._screens:
            if key in ("assistant", "logs"):
                return
            self.register_screen(key, self.placeholder(
                key.title(), "This screen is wired in the full app."))
        screen = self._screens[key]
        if hasattr(screen, "refresh"):
            try:
                screen.refresh()
            except Exception:
                pass
        self._stack.setCurrentWidget(screen)
        self._sidebar.set_active(key)
        # A soft fade on the incoming screen — the "beautiful transitions" from
        # the mockup. Skipped for the napari-hosting workspace (an opacity
        # effect on a live GL canvas is expensive and can flicker).
        if key not in ("workspace",):
            try:
                from napari_app.motion import fade_in
                fade_in(screen, 170)
            except Exception:
                pass

    def open_project(self, project_id: str) -> None:
        """Open a project into the Workspace (Segment) screen."""
        self._active_project_id = project_id
        ws = self._screens.get("workspace")
        if ws is not None and hasattr(ws, "load_project"):
            try:
                ws.load_project(project_id)
            except Exception:
                pass
        self.navigate("workspace")

    def _new_project(self) -> None:
        """Create a project and open it. (A rich dialog comes next; this keeps
        the flow working end-to-end today.)"""
        existing = len(self._store.list())
        p = self._store.create(
            name=f"New Project {existing + 1}",
            description="Import images and pick an engine to begin.",
        )
        # refresh library views
        for key in ("home", "projects"):
            s = self._screens.get(key)
            if s and hasattr(s, "refresh"):
                s.refresh()
        self.open_project(p.id)

    # ── theming ─────────────────────────────────────────────────────────────
    def apply_theme(self, theme_name: str) -> None:
        self._theme_name = theme_name
        app = self.parent() or self
        from PyQt6.QtWidgets import QApplication
        qapp = QApplication.instance()
        if qapp is not None:
            qapp.setStyleSheet(theme.build_qss(self.tokens))

    def toggle_theme(self) -> None:
        new = "light" if self._theme_name == "dark" else "dark"
        # Rebuild the custom-styled chrome for the new palette, keep heavy
        # screens (they carry their own dark styling) registered.
        heavy = {k: v for k, v in self._screens.items()
                 if k not in ("home", "projects")}
        self._screens = {}
        self._theme_name = new
        self._build_chrome()
        self._build_titlebar()
        for k, v in heavy.items():
            self._screens[k] = v
            self._stack.addWidget(v)
        self.apply_theme(new)
        self.navigate("home")


# ── workspace embedding (only reached from main(), needs napari) ─────────────
def build_workspace(viewer, predict_widget, t: dict) -> QWidget:
    """Embed napari's canvas next to the existing Predict controls.

    ``viewer`` is a ``napari.Viewer`` created with ``show=False``; we reparent
    its ``_qt_viewer`` (a plain ``QWidget``) into our layout so the canvas
    lives inside the Studio window. The proven ``PredictWidget`` provides the
    full segment/results/GT/measurements controls unchanged — that is the
    "preserve every existing feature" guarantee, achieved by reuse not rewrite.
    """
    host = QWidget()
    lay = QHBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)

    canvas = getattr(viewer.window, "_qt_viewer", None)
    if canvas is not None:
        canvas.setParent(host)
        lay.addWidget(canvas, 1)
    else:  # pragma: no cover - defensive; API drift
        msg = QLabel("napari canvas unavailable")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(msg, 1)

    panel = QWidget()
    panel.setFixedWidth(360)
    panel.setStyleSheet(f"background:{t['surface']}; border-left:1px solid {t['border']};")
    pl = QVBoxLayout(panel)
    pl.setContentsMargins(0, 0, 0, 0)
    pl.addWidget(predict_widget)
    lay.addWidget(panel)
    return host


def main() -> None:
    """Launch CellSeg1 Studio (the standalone desktop app)."""
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from PyQt6.QtCore import QCoreApplication, QLocale
    # WebEngine (embedded Dashboard) needs this before any QApplication exists.
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    import napari
    from PyQt6.QtWidgets import QApplication
    from napari_app.widgets.predict_widget import PredictWidget
    from napari_app.widgets.train_widget import TrainWidget

    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))

    # napari owns the QApplication; create the viewer hidden so *our* window is
    # the one the user sees.
    viewer = napari.Viewer(show=False, title="CellSeg1 Studio")
    app = QApplication.instance()

    family = load_fonts()
    if app is not None:
        app.setFont(QFont(family, 10))
        try:
            from napari_app.ui_utils import install_wheel_guard
            install_wheel_guard(app)
        except Exception:
            pass

    store = ProjectStore(default_store_root())
    _seed_samples_if_empty(store)

    win = StudioWindow(store, theme_name="dark")

    # Register the heavier screens (these need the viewer / existing widgets).
    predict_widget = PredictWidget(viewer)
    try:
        win.register_screen("workspace", build_workspace(viewer, predict_widget, win.tokens))
    except Exception as exc:  # pragma: no cover
        win.register_screen("workspace", win.placeholder(
            "Workspace", f"Could not embed the canvas: {exc}"))
    try:
        win.register_screen("train", TrainWidget(viewer))
    except Exception as exc:  # pragma: no cover
        win.register_screen("train", win.placeholder("Models & Train", str(exc)))

    win.navigate("home")
    win.show()
    win.raise_()
    win.activateWindow()
    napari.run()


def _seed_samples_if_empty(store: ProjectStore) -> None:
    """First-run nicety: populate a few sample projects so the library isn't
    an empty void. Harmless JSON the user can delete; skipped once any project
    exists."""
    if store.list():
        return
    from napari_app.studio.project import ProjectSettings, ProjectStats
    samples = [
        ("Fluorescence Nuclei — DAPI", "384-well DAPI screen, one-shot LoRA fine-tuned.",
         ["fluorescence", "nuclei"], ProjectSettings(engine="cellseg1"),
         ProjectStats(n_images=128, n_cells=31400, last_f1=0.94, progress=96)),
        ("H&E Tissue Cohort", "Whole-slide H&E biopsies, tiled at native resolution.",
         ["histology", "H&E"], ProjectSettings(engine="cellpose", tiled=True),
         ProjectStats(n_images=342, n_cells=188000, progress=41)),
        ("Live-cell Mitosis", "Confocal z-stacks tracked across time with SAM 2.",
         ["time-lapse", "3D"], ProjectSettings(engine="sam2", sam2_tracking_mode="propagate"),
         ProjectStats(n_images=24, n_cells=9700, last_f1=0.90, progress=70)),
    ]
    for name, desc, tags, settings, stats in samples:
        p = store.create(name, description=desc, tags=tags, settings=settings)
        p.stats = stats
        store.save(p)


if __name__ == "__main__":
    main()
