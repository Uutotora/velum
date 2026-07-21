"""Velum — the application shell and entry point.

This app is a faithful, native-Qt reproduction of the north-star mockup, with
functionality wired back tab by tab (see ``docs/velum/`` — OVERVIEW,
ARCHITECTURE, BACKLOG, AGENT_PROMPT). Every screen is real: Home/Projects
(``ProjectController``), Segment (``SegmentController`` + our own canvas/layer
model), Models & Train/Dashboard (``TrainController``/``DashboardController``),
Assistant (``AssistantController`` + a real chat), Logs (``studio.log_bus`` —
a live stream, not a static transcript), and the ⌘K command palette
(``studio.command_registry`` — ``_build_commands`` below is the real, live
action registry every tab feeds). No napari, no torch — those are reused
lazily, only inside the tab that needs them.

``StudioWindow`` owns a frameless, rounded window with our own dark title bar,
a navigation sidebar, a stack of screens (Home · Projects · Segment · Models &
Train · Dashboard), and overlay surfaces (Assistant drawer, Logs console, ⌘K
command palette, toast). Launch it with ``run_studio.sh`` or the ``velum`` /
``cellseg1`` console command.
"""
from __future__ import annotations

import logging
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
from PyQt6.QtGui import (QFontDatabase, QFont, QIcon, QPixmap, QPainter, QRegion,
                         QPainterPath, QShortcut, QKeySequence)
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QApplication,
)

from studio import theme
from studio import screens
from studio.components import Sidebar
from studio.project_controller import ProjectController
from studio.train_controller import TrainController
from studio.segment_controller import SegmentController
from studio.assistant_controller import AssistantController, BACKENDS, BACKEND_LABELS
from studio.new_project_dialog import NewProjectDialog
from studio.screens import HomeScreen, ProjectsScreen
from studio.workspace import WorkspaceScreen, QUALITY_PRESET_NAMES
from studio.project import ENGINE_LABELS
from studio.extra_screens import ModelsScreen, DashboardScreen
from studio.guide_screen import GuideScreen
from studio.assistant_panel import AssistantDrawer
from studio.overlays import LogsConsole, CommandPalette, Toast
from studio.window_chrome import (
    TitleBar, install_corner_grips, layout_corner_grips,
)
from studio.log_bus import install_handler
from studio.command_registry import Command

_log = logging.getLogger("studio.app")

_FONT_DIR = Path(__file__).parent / "fonts"
_ICON_PATH = Path(__file__).parent / "assets" / "icon.png"
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


# macOS Dock/app-icon grid: system icons fill ~0.875 of the square canvas, the
# rest transparent margin. scripts/make_app.sh bakes the .app's .icns at this
# ratio, so it looks right at rest -- but QApplication.setWindowIcon() overrides
# the Dock tile the moment the app runs, and if it drew the raw full-bleed
# icon.png the tile ballooned on launch (reported: "иконка стала огромной").
_ICON_GRID_RATIO = 0.875


def load_icon() -> QIcon:
    """The app icon for ``QApplication.setWindowIcon`` (the running Dock tile /
    window icon). Pads the full-bleed source art to the macOS icon grid so the
    running tile matches the bundled ``.icns`` and sits at the same size as
    system icons. Returns a null QIcon (Qt's safe default) if the asset is
    missing, matching load_fonts()'s degrade-quietly pattern.
    """
    if not _ICON_PATH.exists():
        return QIcon()
    src = QPixmap(str(_ICON_PATH))
    if src.isNull():
        return QIcon()
    canvas = max(src.width(), src.height()) or 1024
    content = round(canvas * _ICON_GRID_RATIO)
    art = src.scaled(content, content, Qt.AspectRatioMode.KeepAspectRatio,
                     Qt.TransformationMode.SmoothTransformation)
    padded = QPixmap(canvas, canvas)
    padded.fill(Qt.GlobalColor.transparent)
    p = QPainter(padded)
    x = (canvas - art.width()) // 2
    y = round((canvas - art.height()) * 18 / 32)  # slight downward bias, like macOS
    p.drawPixmap(x, y, art)
    p.end()
    return QIcon(padded)


class StudioWindow(QMainWindow):
    """Frameless rounded window: title bar + sidebar + screen stack + overlays."""

    def __init__(self, theme_name: str = "dark",
                 project_controller: Optional[ProjectController] = None,
                 train_controller: Optional[TrainController] = None,
                 segment_controller: Optional[SegmentController] = None,
                 assistant_controller: Optional[AssistantController] = None):
        super().__init__()
        # Idempotent -- safe no matter how many StudioWindows exist in this
        # process (every test constructs its own), and is what actually
        # makes an ordinary logging.getLogger(__name__).info(...) call
        # anywhere in the process (this module, the Assistant, the reused ML
        # core) reach the Logs console's shared LogBus.
        install_handler()
        self._theme_name = theme_name
        self._projects = project_controller or ProjectController()
        self._train = train_controller or TrainController()
        self._segment = segment_controller or SegmentController()
        # Set once, not rebuilt in _build_ui() -- survives toggle_theme()'s
        # full UI teardown/rebuild the same way _projects/_train/_segment do,
        # so a chosen backend/model/API key isn't lost on a theme switch.
        self._assistant_controller = assistant_controller or AssistantController()
        self._screens: dict[str, QWidget] = {}
        self.setWindowTitle("Velum")
        self.resize(1320, 860)
        self.setMinimumSize(1040, 680)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)

        self._build_ui()
        self.apply_theme(theme_name)
        self._grips = install_corner_grips(self)
        self.navigate("home")

        QShortcut(QKeySequence("Ctrl+K"), self, activated=self._toggle_palette)
        QShortcut(QKeySequence("Meta+K"), self, activated=self._toggle_palette)
        QShortcut(QKeySequence("Ctrl+T"), self, activated=lambda: self.navigate("assistant"))
        QShortcut(QKeySequence("Meta+T"), self, activated=lambda: self.navigate("assistant"))
        QShortcut(QKeySequence("Ctrl+L"), self, activated=lambda: self.navigate("logs"))
        QShortcut(QKeySequence("Meta+L"), self, activated=lambda: self.navigate("logs"))
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

        # Constructed (not yet populated with pages -- that happens below,
        # after _screens exists) before NewProjectDialog specifically so it
        # can be that dialog's parent: NewProjectDialog is one shared
        # instance triggered from every screen, so it must scrim/centre
        # over whatever's currently showing in *this* stack, not over the
        # whole window including the sidebar (ConfirmDialog/
        # ProjectSettingsDialog already get this for free, being parented to
        # their own screen -- itself always exactly the stack's bounds --
        # rather than the window).
        self._stack = QStackedWidget()

        self._new_project_dialog = NewProjectDialog(
            self._stack, t, self._projects.store, on_created=self._on_project_created)
        # built before _screens: Models & Train / Dashboard announce through it
        self._toast = Toast(self, t)

        self._screens = {
            "home": HomeScreen(t, self._projects, self.navigate, self._open_project,
                               self._new_project_dialog.open),
            "projects": ProjectsScreen(t, self._projects, self.navigate, self._open_project,
                                      self._new_project_dialog.open, on_toast=self._toast.announce),
            "workspace": WorkspaceScreen(t, self._segment, self._projects, self._toast.announce,
                                        on_toggle_logs=lambda: self._toggle_drawer(self._logs),
                                        on_navigate=self.navigate,
                                        on_new_project=self._new_project_dialog.open),
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
        self._assistant = AssistantDrawer(
            self, t, self._assistant_controller, self._screens["workspace"])
        self._logs = LogsConsole(self, t)
        self._palette = CommandPalette(self, t, get_commands=self._build_commands)
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
            # "home" excluded too, not just "workspace": HomeScreen carries
            # its own, cheaper, more deliberate motion (a scoped fade on
            # just its recent-projects list + a waving-hand greeting, both
            # in HomeScreen.refresh(), already called above) rather than a
            # blanket QGraphicsOpacityEffect fade of the whole screen on
            # every single visit -- see that method's own docstring for why.
            if key not in ("workspace", "home"):
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
        _log.info("project created: %r (%d image%s, engine=%s)",
                   project.name, n, "" if n == 1 else "s", project.settings.engine)
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
            end_geom = drawer.geometry()
            # Slide in from whichever edge this overlay is anchored to
            # (AssistantDrawer: the right edge; LogsConsole: the bottom) —
            # a real product cue instead of an instant pop-in. Both overlays'
            # *right* edge touches the window's right edge (LogsConsole
            # spans full remaining width too, not just AssistantDrawer), so
            # "which edge does it touch" doesn't disambiguate them — compare
            # which dimension dominates instead: a right-anchored vertical
            # panel is tall relative to the window's height (its width is
            # comparatively narrow); a bottom-anchored horizontal one is
            # wide relative to the window's width (its height is narrow).
            height_frac = end_geom.height() / max(1, self.height())
            width_frac = end_geom.width() / max(1, self.width())
            if height_frac >= width_frac:
                start_geom = end_geom.translated(end_geom.width(), 0)   # slide in from the right
            else:
                start_geom = end_geom.translated(0, end_geom.height())  # slide up from the bottom
            drawer.setGeometry(start_geom)
            drawer.show()
            drawer.raise_()
            from studio.motion import slide_in
            slide_in(drawer, start_geom, end_geom)

    def _toggle_palette(self) -> None:
        if not self._palette.isHidden():
            self._palette.hide()
        else:
            self._palette.open()

    # ── ⌘K command registry ─────────────────────────────────────────────────
    # Built fresh every time the palette opens (CommandPalette's own
    # get_commands callback) so availability always reflects the current
    # project/theme/backend/running-state — never a stale snapshot. Every
    # command is a real, already-existing action reached through the same
    # narrow public aliases the Assistant integration established
    # (workspace.py/extra_screens.py/assistant_panel.py's "Command palette
    # integration" sections) — nothing here is invented for the palette.
    def _build_commands(self) -> list[Command]:
        ws = self._screens["workspace"]
        train_screen = self._screens["train"]
        dashboard_screen = self._screens["dashboard"]
        project = self._projects.get_active()

        commands: list[Command] = []

        # Navigate -- derived straight from _NAV so it can never drift from
        # the sidebar's own list; "assistant"/"logs" already toggle their
        # drawer via navigate() itself, same as a sidebar click would.
        _SHORTCUT_HINTS = {"assistant": "⌘T", "logs": "⌘L"}
        _NAV_EMOJI = {
            "home": "🏠", "projects": "🗂️", "workspace": "🔬", "train": "🧠",
            "dashboard": "📊", "assistant": "💬", "logs": "📜",
        }
        for key, icon_name, nav_label, _section in _NAV:
            commands.append(Command(
                id=f"nav.{key}", label=f"Go to {nav_label}", section="Navigate",
                icon=icon_name, emoji=_NAV_EMOJI.get(key, ""), hint=_SHORTCUT_HINTS.get(key, ""),
                handler=lambda k=key: self.navigate(k)))
        commands.append(Command(
            id="nav.guide", label="Go to Guide & Docs", section="Navigate",
            icon="guide", emoji="📘", handler=lambda: self.navigate("guide")))

        # Segment -- Run/Save/Export are always listed (greyed out with no
        # active project) since _start_predict/_save_masks/etc. already
        # guard every precondition with a toast; "Switch engine"/"Apply
        # preset" only make sense (and are only generated) with a project
        # to compare "current" against.
        commands += [
            Command(id="segment.run", label="Run segmentation", section="Segment",
                    icon="run", emoji="▶️", handler=ws.rerun_predict, enabled=project is not None),
            Command(id="segment.batch", label="Run batch prediction", section="Segment",
                    icon="batch", emoji="🗃️", handler=ws.run_batch, enabled=project is not None),
            Command(id="segment.benchmark", label="Run benchmark vs. ground truth", section="Segment",
                    icon="chart", emoji="🎯", handler=ws.run_benchmark, enabled=project is not None),
            Command(id="segment.save", label="Save masks", section="Segment",
                    icon="save", emoji="💾", handler=ws.save_masks, enabled=project is not None),
            Command(id="segment.export_csv", label="Export measurements → CSV", section="Segment",
                    icon="csv", emoji="📤", handler=ws.export_measurements, enabled=project is not None),
        ]
        if project is not None:
            current_engine = project.settings.engine
            for key, _elabel, available in self._segment.list_available_engines():
                if available and key != current_engine:
                    # list_available_engines()'s own label is the long,
                    # descriptive combo-box text ("Cellpose-SAM (zero-shot,
                    # generalist)") -- ENGINE_LABELS (project.py) is the
                    # short display name the mockup itself uses ("Switch
                    # engine → SAM 2"), a much better fit for one palette row.
                    short_label = ENGINE_LABELS.get(key, key)
                    keywords = "zstack timelapse z-stack" if key == "sam2" else ""
                    commands.append(Command(
                        id=f"segment.engine.{key}", label=f"Switch engine → {short_label}",
                        section="Segment", icon="workspace", emoji="🔁", keywords=keywords,
                        handler=lambda k=key: ws.switch_engine(k)))
            current_preset = project.settings.quality_preset
            for name in QUALITY_PRESET_NAMES:
                if name != current_preset:
                    commands.append(Command(
                        id=f"segment.preset.{name}", label=f"Apply preset → {name}",
                        section="Segment", icon="chart", emoji="🎛️",
                        handler=lambda n=name: ws.apply_preset(n)))

        # Models & Train
        is_training = self._train.is_training()
        commands += [
            Command(id="train.start", label="Start training", section="Models & Train",
                    icon="models", emoji="🎓", handler=train_screen.start_training,
                    enabled=not is_training),
            Command(id="train.stop", label="Stop training", section="Models & Train",
                    icon="close", emoji="⏹️", handler=train_screen.stop_training,
                    enabled=is_training),
            Command(id="train.import", label="Import a trained model…", section="Models & Train",
                    icon="folder", emoji="📥", handler=train_screen.import_model),
        ]

        # Dashboard
        commands.append(Command(
            id="dashboard.aim", label="Open in Aim", section="Dashboard",
            icon="chart", emoji="📈", handler=dashboard_screen.open_in_aim))

        # Assistant -- opens the drawer, then acts, so the effect is visible
        # immediately rather than happening silently behind a closed panel.
        commands.append(Command(
            id="assistant.diagnose", label="Diagnose current result", section="Assistant",
            icon="diagnose", emoji="🩺", handler=self._cmd_diagnose))
        current_backend = self._assistant_controller.settings.backend
        for idx, key in enumerate(BACKENDS):
            if key != current_backend:
                commands.append(Command(
                    id=f"assistant.backend.{key}",
                    label=f"Switch Assistant backend → {BACKEND_LABELS[key]}",
                    section="Assistant", icon="assistant", emoji="🤖",
                    handler=lambda i=idx: self._cmd_switch_backend(i)))

        # Appearance -- names the concrete destination, not a generic toggle,
        # the same "Switch engine → X" naming convention as Segment above.
        other_theme = "Light" if self._theme_name == "dark" else "Dark"
        commands.append(Command(
            id="appearance.theme", label=f"Switch to {other_theme} theme", section="Appearance",
            icon="sun" if other_theme == "Light" else "moon",
            emoji="🌞" if other_theme == "Light" else "🌙", handler=self.toggle_theme))

        # Projects
        commands.append(Command(
            id="projects.new", label="New Project…", section="Projects",
            icon="plus", emoji="➕", handler=self._new_project_dialog.open))
        commands.append(Command(
            id="projects.sample", label="Open Sample", section="Projects",
            icon="chart", emoji="🧪", handler=self._cmd_open_sample))

        # Help -- mirrors Home's own Resources links exactly (see
        # screens.py's HomeScreen._resources_card), including the
        # already-established redundancy against "Go to Assistant" above.
        commands += [
            Command(id="help.docs", label="Documentation", section="Help",
                    icon="guide", emoji="📖", handler=lambda: self.navigate("guide")),
            Command(id="help.getting_started", label="Getting started guide", section="Help",
                    icon="guide", emoji="🚀", handler=lambda: self.navigate("guide:getting-started")),
            Command(id="help.ask_assistant", label="Ask the Assistant", section="Help",
                    icon="assistant", emoji="💬", handler=lambda: self.navigate("assistant")),
            Command(id="help.github", label="GitHub", section="Help",
                    icon="settings", emoji="🐙", handler=screens._open_github),
        ]
        return commands

    def _cmd_diagnose(self) -> None:
        self.navigate("assistant")
        self._assistant.run_diagnose()

    def _cmd_switch_backend(self, idx: int) -> None:
        self.navigate("assistant")
        self._assistant.switch_backend(idx)

    def _cmd_open_sample(self) -> None:
        projects = self._projects.list_projects()
        if projects:
            self._open_project(projects[0].id)
        else:
            self._new_project_dialog.open()

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
        _log.debug("theme toggled to %s", new)
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
        # Also a real, visible CRITICAL entry in the Logs console -- a crash
        # shouldn't only be discoverable by whoever happened to have a
        # terminal open behind the app.
        _log.critical("Unhandled exception: %s", exc_value, exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _hook


def main() -> None:
    """Launch Velum (pure-design skeleton — no napari/torch needed)."""
    _install_exception_hook()
    install_handler()
    _log.info("Velum starting…")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setWindowIcon(load_icon())
    family = load_fonts()
    app.setFont(QFont(family, 10))
    win = StudioWindow(theme_name="dark")
    win.setWindowIcon(app.windowIcon())
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
