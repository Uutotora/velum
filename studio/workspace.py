"""Velum — the Workspace (Segment) screen, wired for real.

The signature screen: an evented **Layers** panel (list + full controls,
napari-Labels-faithful) driving our own **Canvas** (``studio/canvas.py`` —
not embedded napari), and the **inspector** (Segment settings · Results).
Real segmentation is reused from the classic app via ``SegmentController``
(``studio/segment_controller.py``), which wraps
``velum_core.predict_controller.PredictController`` unmodified.

Every reactive region rebuilds its own small container on demand rather than
tearing down the whole screen (unlike ``ModelsScreen``/``DashboardScreen``,
which cheaply rebuild everything on every tab visit) — the Canvas's pan/zoom
and the Layers' in-progress edits must survive a tab switch. A `Slider`
mid-drag must never have its own container rebuilt (that would sever the
mouse grab mid-gesture): value-only changes update an inline `Badge`
directly and never call the container rebuild; only a *structural* change
(add/remove/select a layer, switch engine, switch project/image) rebuilds a
container. See ``docs/velum/ARCHITECTURE.md``'s "Segment tab specifically".
"""
from __future__ import annotations

import html
import time
from dataclasses import fields
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QStackedWidget, QToolButton, QLineEdit, QSizePolicy, QScrollArea, QFileDialog,
    QSplitter,
)

from studio import icons
from studio import theme, demo
from studio.project import (
    ENGINE_LABELS, ENGINE_KIND, IMAGE_FILE_FILTER, Project, ProjectSettings,
    is_supported_image_path,
)
from studio.paint import nuclei_pixmap
from studio.components import (
    Chip, Badge, EngineChip, PillButton, IconButton, SelectBox, Toggle, Slider, Stepper,
    SegControl, StatTile, FieldRow, GroupLabel, Accordion, SmoothScrollArea, SwipeRow,
    hline, label, bare_widget,
)
from studio.canvas import Canvas
from studio.layer_model import (
    BLENDING_MODES, ImageLayer, IMAGE_COLORMAPS, LabelsLayer, LayerList,
    PAN_ZOOM, PAINT, ERASE, FILL, POLYGON, PICK, PointsLayer, ShapesLayer,
)
from studio.segment_controller import SegmentController, apply_quality_preset
from studio.log_bus import get_log_bus, emit_prefixed

LAYER_TYPE_ICON = {"labels": "layers", "shapes": "shapes", "points": "points", "image": "image"}
# (icon, tooltip, mode) for the Labels layer's 8-icon tool row. "__shuffle__"
# is an action, not a mode — mirrors napari's Mode enum plus that one action.
MODE_ICONS = [
    ("target", "Pan / zoom", PAN_ZOOM),
    ("brush", "Paint brush", PAINT),
    ("eraser", "Eraser", ERASE),
    ("fill", "Fill bucket", FILL),
    ("polygon", "Polygon", POLYGON),
    ("pick", "Pick label colour", PICK),
    ("shuffle", "Shuffle colours", "__shuffle__"),
]
QUALITY_PRESET_NAMES = ["Fast", "Balanced", "Accurate"]
# Every real ProjectSettings field name — apply_assistant_changes() below
# whitelists against this so a changes dict can only ever touch real
# settings, never shadow a method (e.g. a stray "to_dict" key) on the
# instance via setattr.
_SETTINGS_FIELD_NAMES = {f.name for f in fields(ProjectSettings)}
COLOR_BY_OPTIONS = ["Instance ID (default)", "Area (heatmap)", "Diameter (heatmap)",
                    "Solidity (heatmap)", "Mean intensity (heatmap)"]
_COLOR_BY_KEYS = {"Area (heatmap)": "area", "Diameter (heatmap)": "diameter",
                  "Solidity (heatmap)": "solidity", "Mean intensity (heatmap)": "mean_intensity"}
_DLG = QFileDialog.Option.DontUseNativeDialog
# Ground truth's fixed colour — matches the classic app's own GT convention
# (predict_widget.py's solid_rgba=(0.0, 1.0, 0.35, 1.0)): a uniform green,
# not per-instance random hues, so GT and predictions read as visually
# distinct roles instead of competing for the same rainbow.
_GT_COLOR = (0, 255, 89)
# One dot hue per engine for the centred topbar badge — matches the Projects
# tab's card badge (screens.py's _ENGINE_DOT) so an engine reads the same
# colour everywhere it appears.
_ENGINE_DOT = {"cellseg1": theme.VIZ[0], "cellpose": theme.VIZ[1], "sam2": theme.VIZ[5]}


def _scroll(inner: QWidget) -> QScrollArea:
    sa = SmoothScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setWidget(inner)
    return sa


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
            continue
        # Nested layouts (added via addLayout -- the results pane's hero/tiles/
        # button-grid rows) hold widgets that are *not* returned by item.widget()
        # here; without recursing into them, every rebuild orphaned those child
        # widgets (they keep the container as parent and stay visible), so each
        # _rebuild_results_pane() stacked a fresh copy on top of the last -- the
        # reported "Refine…/Measurements overlap" and the ghost "Measure" text
        # bleeding over the calibration hint. Recurse, then drop the empty layout.
        child = item.layout()
        if child is not None:
            _clear_layout(child)
            child.deleteLater()


class WorkspaceScreen(QWidget):
    _predict_result_signal = pyqtSignal(object, object, object)
    _predict_log_signal = pyqtSignal(str)
    _predict_finish_signal = pyqtSignal()

    _batch_progress_signal = pyqtSignal(int, int)
    _batch_log_signal = pyqtSignal(str)
    _batch_cohort_signal = pyqtSignal(object, object)
    _batch_finish_signal = pyqtSignal()

    _bench_row_signal = pyqtSignal(str)
    _bench_log_signal = pyqtSignal(str)
    _bench_done_signal = pyqtSignal(object, object)

    def __init__(self, t: dict, segment: SegmentController, projects, on_toast,
                on_toggle_logs=None, on_navigate=None, on_new_project=None,
                on_open_sample=None):
        super().__init__()
        self._t = t
        self._segment = segment
        self._projects = projects
        self._toast = on_toast
        self._on_toggle_logs = on_toggle_logs
        self._on_navigate = on_navigate
        self._on_new_project = on_new_project
        self._on_open_sample = on_open_sample

        self._project: Optional[Project] = None
        self._layers = LayerList()
        # Cache thumbnails by path: the images pane is rebuilt wholesale on
        # every select / add / project-load, and re-decoding each file every
        # time is what produced the repeated "can't open/read file" storm in
        # the logs for an unreadable source. Both real and fallback pixmaps
        # are cached, so a file that can't be read is attempted once, not on
        # every rebuild.
        self._thumb_cache: dict[str, QPixmap] = {}
        self._current_image_path: Optional[str] = None
        self._current_image_array: Optional[np.ndarray] = None
        self._last_result: Optional[dict] = None
        self._gt_metrics: Optional[dict] = None
        self._bench_rows: list[tuple[str, str]] = []
        self._predicting = False
        self._batching = False
        self._benching = False
        self._run_started_at: Optional[float] = None

        # Keep the Results panel + the project card's cell count in sync with
        # live mask edits (paint/erase/fill/undo/redo), not just fresh predict
        # runs. `notify()` fires on every mouse-move tick of a paint drag, so
        # the actual recompute (regionprops) is debounced to fire once after a
        # burst settles; `_results_sig` is the cheap content fingerprint that
        # gates it, so pure UI churn (selection, visibility, reorder) never
        # triggers a recompute. Without this, erasing cells dropped the legend
        # count but left the Results stats and the stored stats stale -- the
        # reported "card says 122 cells, Results says 45" drift.
        self._results_sig: Optional[tuple[int, int]] = None
        self._results_sync_timer = QTimer(self)
        self._results_sync_timer.setSingleShot(True)
        self._results_sync_timer.setInterval(250)
        self._results_sync_timer.timeout.connect(self._sync_results_after_edit)

        self._layers.on_change(self._on_layers_changed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._topbar_widget = self._topbar()
        outer.addWidget(self._topbar_widget)

        # Resizable three-pane body: drag either handle to rebalance, and each
        # side pane can be collapsed to give the canvas the full width (the
        # topbar's two panel toggles do this). The canvas itself is never
        # collapsible. This replaces the old fixed 240/‑/320 rail layout.
        self._left_panel_w = self._left_panel()
        self._inspector_w = self._inspector()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("BodySplitter")
        splitter.setChildrenCollapsible(True)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(
            "QSplitter#BodySplitter::handle{background:transparent;}"
            f"QSplitter#BodySplitter::handle:hover{{background:{t['primary_weak']};}}")
        splitter.addWidget(self._left_panel_w)
        splitter.addWidget(self._viewport())
        splitter.addWidget(self._inspector_w)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(1, False)  # the canvas always stays visible
        splitter.setSizes([240, 900, 340])
        self._body_splitter = splitter
        main = splitter
        # Both panels start visible -> their topbar toggles start "on".
        self._style_panel_toggle(self._toggle_left_btn, "panel_left", True)
        self._style_panel_toggle(self._toggle_right_btn, "panel_right", True)

        # Two full alternatives for the body, not one body with an empty
        # state layered into a corner of it -- see _no_project_view()'s own
        # docstring for why the three-panel layout itself (not just the
        # canvas) is wrong to show with nothing open.
        self._body_stack = QStackedWidget()
        self._body_stack.addWidget(main)
        self._body_stack.addWidget(self._no_project_view())
        outer.addWidget(self._body_stack, 1)

        self._predict_result_signal.connect(self._on_predict_result)
        self._predict_log_signal.connect(self._on_predict_log)
        self._predict_finish_signal.connect(self._on_predict_finished)
        self._batch_progress_signal.connect(self._on_batch_progress)
        self._batch_log_signal.connect(self._on_predict_log)
        self._batch_cohort_signal.connect(self._on_batch_cohort_ready)
        self._batch_finish_signal.connect(self._on_batch_finished)
        self._bench_row_signal.connect(self._on_bench_row)
        self._bench_log_signal.connect(self._on_predict_log)
        self._bench_done_signal.connect(self._on_bench_done)

        # Undo / redo for mask edits. StandardKey resolves per-platform (⌘Z /
        # ⇧⌘Z on macOS, Ctrl+Z / Ctrl+Y elsewhere); the handlers are no-ops
        # with an empty history, so the shortcuts are always safe to fire.
        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._undo_shortcut.activated.connect(self._undo)
        self._redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        self._redo_shortcut.activated.connect(self._redo)

        self._load_project(None)  # establish the empty state now everything exists
        self._sync_layers_pane_state()  # no layers yet -> show the Layers empty-state

    # ── project lifecycle ────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Called by app.navigate() on every visit to this tab. Only a real
        project *switch* resets the session (layers/canvas/settings); simply
        revisiting the tab with the same project leaves in-progress work —
        including unsaved settings tweaks — alone. A cross-tab settings
        change to the *same, already-open* project (e.g. selecting a model
        in Models & Train) needs reopening the project to pick up — a known,
        deliberate simplification, not a bug."""
        project = self._projects.get_active() if self._projects else None
        current_id = self._project.id if self._project else None
        new_id = project.id if project else None
        if new_id != current_id:
            self._load_project(project)

    def set_active_project(self, project: Optional[Project]) -> None:
        """Reflect the active project in the breadcrumb + engine chip only —
        cheap and safe to call anytime, including from app.py before the
        rest of this screen is built. The heavier "switch projects" reset
        is _load_project's job, triggered from refresh()."""
        t = self._t
        if project is None:
            name, engine_key, engine_label = "No project selected", None, "No project"
        else:
            name, engine_key = project.name, project.engine
            engine_label = ENGINE_LABELS.get(engine_key, engine_key)
        self._crumb_name.setText(html.escape(name))
        old = self._engine_badge
        self._engine_badge = self._make_engine_badge(engine_key, engine_label)
        self._engine_badge_layout.addWidget(self._engine_badge)
        self._engine_badge_layout.removeWidget(old)
        old.setParent(None)
        old.deleteLater()

    def _make_engine_badge(self, engine_key: Optional[str], engine_label: str) -> QWidget:
        """The centred topbar engine badge: an EngineChip (rounded pill + hued
        dot) when a project is open, greyed with a muted dot when none is."""
        t = self._t
        dot = _ENGINE_DOT.get(engine_key, t["text_muted"]) if engine_key else t["text_muted"]
        return EngineChip(engine_label, dot, bg=t["surface2"], fg=t["text_subtle"], border=t["border"])

    def _go_to_projects(self) -> None:
        """The breadcrumb's "Projects" segment — a real link back to the
        Projects tab, matching Label Studio's breadcrumb (only the ancestor
        segment navigates; the current project's own name doesn't, since
        you're already looking at it)."""
        if self._on_navigate:
            self._on_navigate("projects")

    def _load_project(self, project: Optional[Project]) -> None:
        self._project = project
        self.set_active_project(project)
        self._body_stack.setCurrentIndex(0 if project else 1)
        # The whole topbar, not just Export/Run disabled inside it -- the
        # breadcrumb ("No project selected"), engine chip ("No project") and
        # both buttons are all meaningless with nothing open, and the
        # no-project view (_body_stack index 1) already has its own "Open a
        # Project" action, making the breadcrumb's "Projects" link
        # redundant too. Safe to hide outright, not just grey out:
        # _start_predict/_export_csv already guard their own preconditions
        # with a toast regardless of what triggers them (see app.py's
        # command-palette comment on the identical Run/Export commands
        # there), so hiding their one topbar entry point removes a redundant
        # UI-level guard, not the real one.
        self._topbar_widget.setVisible(project is not None)
        self._layers.clear()
        self._current_image_path = None
        self._current_image_array = None
        self._last_result = None
        self._gt_metrics = None
        self._bench_rows = []
        if project and project.image_paths:
            self._select_image(project.image_paths[0])
        self._refresh_images_pane()
        self._rebuild_layer_controls()
        self._rebuild_segment_pane()
        self._rebuild_results_pane()
        if self._canvas is not None:
            self._canvas.home()
            self._sync_toolbars()

    def _no_project_view(self) -> QWidget:
        """The Segment tab's own "no project" screen -- replaces the entire
        three-panel workspace body (``_body_stack`` index 1), not a message
        layered into a corner of the canvas while the rest of the body
        stays on screen. The three-panel IDE layout (Images/Layers · canvas
        with its own floating tool strip/viewer bar · Segment/Results) is
        only meaningful once there's a real project/image/layers behind it
        -- with none of that, showing all three panels at once (an empty
        list, a bare canvas with tools that do nothing yet, an inspector
        that just says "open or create a project") reads as broken chrome,
        not a clean empty state. Reported directly against a real
        screenshot of exactly that ("канвас боковые панели... убрать все" --
        remove the canvas and side panels, all of it). One centred,
        theme-aware message instead, matching how Home/Projects look with
        nothing to show -- not the viewport's own always-dark canvas
        colours (``_viewport``'s ``#07090c`` backdrop and the floating
        legend/tools/vbar/status chrome, all deliberately theme-independent
        since they overlay real image content): this view isn't inside that
        canvas, it *replaces* the whole body, so it should look like any
        other page in the app, light or dark.
        """
        t = self._t
        w = QWidget()
        w.setStyleSheet(f"background:{t['bg']};")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)

        emoji = label("🙂", 52, t["text"])
        emoji.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(emoji)

        title = label("No project open", 19, t["text"], 600)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(title)

        # An explicit \n, not setWordWrap(True) -- a wrapping QLabel with no
        # fixed width anchor has an ambiguous heightForWidth negotiation
        # (see overlays.Toast's own subtitle fix, docs/velum/CHANGELOG.md
        # 2026-07-20); picking the line break ourselves for this short,
        # fixed copy sidesteps that whole bug class instead of re-risking it.
        sub = label("Open an existing project, or create a new one\nto start segmenting cells.",
                   13, t["text_muted"])
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(sub)

        v.addSpacing(8)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch(1)
        new_btn = PillButton("New Project", t, "ghost", "plus", small=True)
        new_btn.clicked.connect(self._trigger_new_project)
        btn_row.addWidget(new_btn)
        open_btn = PillButton("Open a Project", t, "primary", "folder", small=True)
        open_btn.clicked.connect(self._go_to_projects)
        btn_row.addWidget(open_btn)
        btn_row.addStretch(1)
        v.addLayout(btn_row)

        self._no_project_new_btn = new_btn
        self._no_project_open_btn = open_btn

        # A one-click way into a real, already-segmented field -- the fastest
        # path from a blank workspace to "oh, *this* is what it does". Only
        # offered when the host wired a sample handler (the app always does).
        if self._on_open_sample is not None:
            v.addSpacing(4)
            sample_row = QHBoxLayout()
            sample_row.addStretch(1)
            sample_btn = PillButton("Explore the sample dataset", t, "ghost", "spark", small=True)
            sample_btn.clicked.connect(self._trigger_open_sample)
            sample_row.addWidget(sample_btn)
            sample_row.addStretch(1)
            v.addLayout(sample_row)
            self._no_project_sample_btn = sample_btn
        return w

    def _trigger_open_sample(self) -> None:
        if self._on_open_sample:
            self._on_open_sample()

    def _trigger_new_project(self) -> None:
        if self._on_new_project:
            self._on_new_project()

    # ── top bar ──────────────────────────────────────────────────────────────
    def _topbar(self) -> QWidget:
        t = self._t
        bar = QWidget()
        bar.setStyleSheet(f"background:{t['surface']};")
        bar.setFixedHeight(52)
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 0, 14, 0)
        row.setSpacing(8)

        # ── left-panel toggle ─────────────────────────────────────────────────
        self._toggle_left_btn = IconButton("panel_left", t, 30, "Hide / show the Images · Layers panel",
                                            self._toggle_left_panel)
        row.addWidget(self._toggle_left_btn)

        # ── breadcrumb (left): Projects › name ────────────────────────────────
        # "Projects" is a real link back to the tab, styled like Label Studio's
        # breadcrumb ancestor: muted at rest, brightening on hover (QSS has no
        # :hover for a bare QLabel's colour that survives our re-styling, so the
        # enter/leave handlers set it directly).
        self._crumb_projects = QLabel("Projects")
        self._crumb_base_css = f"font-size:13px; font-weight:600; color:{t['text_muted']}; background:transparent;"
        self._crumb_hot_css = f"font-size:13px; font-weight:600; color:{t['text']}; background:transparent;"
        self._crumb_projects.setStyleSheet(self._crumb_base_css)
        self._crumb_projects.setCursor(Qt.CursorShape.PointingHandCursor)
        self._crumb_projects.setToolTip("Back to Projects")
        self._crumb_projects.enterEvent = lambda e: self._crumb_projects.setStyleSheet(self._crumb_hot_css)
        self._crumb_projects.leaveEvent = lambda e: self._crumb_projects.setStyleSheet(self._crumb_base_css)
        self._crumb_projects.mouseReleaseEvent = lambda e: self._go_to_projects()
        row.addWidget(self._crumb_projects)
        self._crumb_sep = QLabel("/")
        self._crumb_sep.setStyleSheet(f"font-size:13px; color:{t['border_strong']}; background:transparent;")
        row.addWidget(self._crumb_sep)
        self._crumb_name = QLabel()
        self._crumb_name.setStyleSheet(f"font-size:13px; font-weight:600; color:{t['text']}; background:transparent;")
        row.addWidget(self._crumb_name)

        # ── engine badge (centre): a rounded pill with an engine-hued dot ──────
        # Centred between two equal stretches instead of sitting in a square
        # next to the name -- the engine is a property of the whole workspace,
        # so it reads as a standalone status badge, not part of the title.
        row.addStretch(1)
        self._engine_badge_holder = QWidget()
        self._engine_badge_layout = QHBoxLayout(self._engine_badge_holder)
        self._engine_badge_layout.setContentsMargins(0, 0, 0, 0)
        self._engine_badge_layout.setSpacing(0)
        self._engine_badge = self._make_engine_badge(None, "No project")
        self._engine_badge_layout.addWidget(self._engine_badge)
        row.addWidget(self._engine_badge_holder)
        row.addStretch(1)

        # ── actions (right) ───────────────────────────────────────────────────
        self._export_btn_topbar = PillButton("Export", t, "ghost", "export", small=True)
        self._export_btn_topbar.clicked.connect(self._export_csv)
        row.addWidget(self._export_btn_topbar)
        self._run_btn_topbar = PillButton("Run", t, "primary", "run", small=True)
        self._run_btn_topbar.clicked.connect(self._start_predict)
        row.addWidget(self._run_btn_topbar)
        # ── right-panel (inspector) toggle ────────────────────────────────────
        self._toggle_right_btn = IconButton("panel_right", t, 30, "Hide / show the Segment · Results panel",
                                             self._toggle_inspector)
        row.addWidget(self._toggle_right_btn)
        bottom = QFrame()
        bottom.setFixedHeight(1)
        bottom.setStyleSheet(f"background:{t['border']};")
        wrap = QWidget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(0)
        wl.addWidget(bar)
        wl.addWidget(bottom)
        return wrap

    def _toggle_left_panel(self) -> None:
        """Collapse / restore the Images·Layers panel (topbar toggle). The
        splitter hands the reclaimed width to the canvas; re-showing restores
        a sensible width. Button tints active while the panel is visible."""
        vis = self._left_panel_w.isHidden()  # currently hidden -> show it (works offscreen too)
        self._left_panel_w.setVisible(vis)
        self._style_panel_toggle(self._toggle_left_btn, "panel_left", vis)

    def _toggle_inspector(self) -> None:
        vis = self._inspector_w.isHidden()
        self._inspector_w.setVisible(vis)
        self._style_panel_toggle(self._toggle_right_btn, "panel_right", vis)

    def _style_panel_toggle(self, btn: QToolButton, icon_name: str, on: bool) -> None:
        t = self._t
        if on:
            btn.setStyleSheet(f"QToolButton{{background:{t['primary_weak']}; border-radius:8px;}}")
            btn.setIcon(icons.icon(icon_name, t["primary"], 16))
        else:
            btn.setStyleSheet(
                f"QToolButton{{background:transparent; border:1px solid transparent; border-radius:8px;}}"
                f"QToolButton:hover{{background:{t['surface2']}; border-color:{t['border']};}}")
            btn.setIcon(icons.icon(icon_name, t["text_muted"], 16))

    # ── left: Images | Layers ────────────────────────────────────────────────
    def _left_panel(self) -> QWidget:
        t = self._t
        panel = QFrame()
        # Resizable (drag the splitter handle) + collapsible, so a wide image
        # gets the whole canvas -- not a locked 240px rail. Min keeps the tabs
        # legible; the splitter owns the actual width.
        panel.setMinimumWidth(210)
        panel.setMaximumWidth(460)
        # No border-right here: the divider is drawn on the canvas edge instead
        # (see _viewport) so this panel's text doesn't pick up a HiDPI border seam.
        panel.setStyleSheet(f"background:{t['inset']};")
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        tabs = SegControl(["Images", "Layers"], t, 1, compact=True)
        tabwrap = QWidget()
        tw = QHBoxLayout(tabwrap)
        tw.setContentsMargins(10, 10, 10, 8)
        tw.addWidget(tabs)
        v.addWidget(tabwrap)

        stack = QStackedWidget()
        stack.addWidget(self._images_pane())
        stack.addWidget(self._layers_pane())
        stack.setCurrentIndex(1)
        tabs.changed.connect(stack.setCurrentIndex)
        v.addWidget(stack, 1)
        return panel

    # ── Images pane ──────────────────────────────────────────────────────────
    def _images_pane(self) -> QWidget:
        t = self._t
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 2, 8, 8)
        v.setSpacing(4)
        search_row = QHBoxLayout()
        search_row.setSpacing(6)
        self._image_search = QLineEdit()
        self._image_search.setPlaceholderText("Filter images…")
        self._image_search.addAction(icons.icon("diagnose", t["text_muted"], 14),
                                     QLineEdit.ActionPosition.LeadingPosition)
        self._image_search.textChanged.connect(lambda _=None: self._refresh_images_pane())
        search_row.addWidget(self._image_search, 1)
        search_row.addWidget(IconButton("plus", t, 30, "Add images…", self._add_images))
        v.addLayout(search_row)
        self._images_list_container = bare_widget()
        self._images_list_layout = QVBoxLayout(self._images_list_container)
        self._images_list_layout.setContentsMargins(0, 4, 0, 0)
        self._images_list_layout.setSpacing(2)
        drop_hint = label("Drop images here, or click + to add", 10.5, t["text_muted"])
        drop_hint.setWordWrap(True)
        drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._images_drop_hint = drop_hint
        v.addWidget(_scroll(self._images_list_container), 1)
        v.addWidget(drop_hint)
        w.setAcceptDrops(True)
        w.dragEnterEvent = self._images_drag_enter
        w.dropEvent = self._images_drop
        return w

    def _refresh_images_pane(self) -> None:
        layout = self._images_list_layout
        _clear_layout(layout)
        t = self._t
        if self._project is None:
            empty = label("No project open.", 12, t["text_muted"])
            empty.setWordWrap(True)
            layout.addWidget(empty)
            return
        query = (self._image_search.text() or "").strip().lower()
        paths = [p for p in self._project.image_paths if not query or query in Path(p).name.lower()]
        if not paths:
            msg = "No images match." if query else "This project has no images yet."
            empty = label(msg, 12, t["text_muted"])
            empty.setWordWrap(True)
            layout.addWidget(empty)
        for p in paths:
            layout.addWidget(self._image_row(p))
        layout.addStretch(1)

    def _add_images(self) -> None:
        if self._project is None:
            self._toast("No project open", "Open or create a project first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add images to project", "", IMAGE_FILE_FILTER, options=_DLG)
        if paths:
            self._add_image_paths(paths)

    def _add_image_paths(self, paths: list[str]) -> None:
        """Append new image files to the active project — the Images pane's
        "+"/drag-drop entry point. A project's images were previously only
        ever set once, at creation, via the New Project dialog; there was no
        way back into this list afterward."""
        if self._project is None:
            return
        existing = set(self._project.image_paths)
        new = [p for p in paths if p not in existing and is_supported_image_path(p)]
        if not new:
            self._toast("No new images",
                       "Those are already in this project, or aren't a supported image format.")
            return
        # Copy into the project (survives the source moving + macOS's
        # per-folder privacy gate); falls back to the original path per file
        # if the copy can't happen -- see ProjectStore.import_images.
        new = self._projects.store.import_images(self._project.id, new)
        self._project.image_paths.extend(new)
        self._project.stats.n_images = len(self._project.image_paths)
        self._projects.store.save(self._project)
        self._refresh_images_pane()
        self._toast("Images added",
                   f"{len(new)} image{'s' if len(new) != 1 else ''} added to “{self._project.name}”")
        if self._current_image_path is None:
            self._select_image(self._project.image_paths[0])

    def _images_drag_enter(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def _images_drop(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        paths = [p for p in paths if is_supported_image_path(p)]
        if paths:
            self._add_image_paths(paths)
        e.acceptProposedAction()

    def _image_row(self, path: str) -> QFrame:
        t = self._t
        sel = path == self._current_image_path
        has_gt = self._segment.find_gt_for_image(path) is not None
        has_saved_result = self._project is not None and self._segment.has_result_mask(self._project, path)
        if (sel and self._last_result is not None) or has_saved_result:
            status, dcol = "predicted", t["signal"]
        elif has_gt:
            status, dcol = "annotated", t["success"]
        else:
            status, dcol = "new", t["border_strong"]
        # The foreground content -- opaque (inset when unselected, so it fully
        # covers the SwipeRow's red delete backdrop until you actually swipe).
        # Qualified objectName -- see components.EngineChip's comment; the
        # inner QLabels set no border, so an unscoped rule would leak this
        # frame's border onto them.
        content = QFrame()
        content.setObjectName("ImageRow")
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content.setStyleSheet(
            f"QFrame#ImageRow{{background:{t['surface'] if sel else t['inset']};"
            f" border:1px solid {t['border'] if sel else 'transparent'}; border-radius:8px;}}")
        lay = QHBoxLayout(content)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(10)
        thumb = QLabel()
        thumb.setFixedSize(38, 30)
        thumb.setPixmap(self._thumbnail(path))
        thumb.setScaledContents(True)
        lay.addWidget(thumb)
        col = QVBoxLayout()
        col.setSpacing(1)
        fnl = QLabel(Path(path).name)
        fnl.setStyleSheet(f"color:{t['text']}; font-family:{theme.MONO}; font-size:12px; font-weight:600; background:transparent;")
        col.addWidget(fnl)
        col.addWidget(label(status, 10.5, t["text_muted"]))
        lay.addLayout(col, 1)
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{dcol}; border-radius:4px;")
        lay.addWidget(dot)
        # Swipe left to reveal Delete (iOS-style); a plain tap selects. Both
        # callbacks are deferred one event-loop tick: each ends up rebuilding
        # _images_list_layout (this very row's parent) via
        # _refresh_images_pane(), and tearing a widget down while its own
        # mouse handler is still on the stack corrupts PyQt/SIP's virtual-call
        # bookkeeping ("invalid argument to sipBadCatcherResult()", a hard
        # process abort in the real app).
        return SwipeRow(
            content, t,
            on_click=lambda p=path: QTimer.singleShot(0, lambda: self._select_image(p)),
            on_delete=lambda p=path: QTimer.singleShot(0, lambda: self._remove_image(p)))

    def _remove_image(self, path: str) -> None:
        """Drop an image from the project (the swipe-left Delete). Also tidies
        the in-project *copy* on disk (only ever a file under this project's
        own images dir -- never an external source we merely reference). If it
        was the open image, moves to the next one (or clears the canvas)."""
        if self._project is None or path not in self._project.image_paths:
            return
        self._project.image_paths.remove(path)
        self._project.stats.n_images = len(self._project.image_paths)
        self._projects.store.save(self._project)
        try:
            store_imgs = str(self._projects.store.image_dir(self._project.id).resolve())
            if str(Path(path).resolve()).startswith(store_imgs) and Path(path).exists():
                Path(path).unlink()  # our own copy -- safe to delete
        except Exception:
            pass  # best-effort cleanup; the reference is already gone
        self._thumb_cache.pop(path, None)
        if path == self._current_image_path:
            self._current_image_path = None
            self._current_image_array = None
            self._last_result = None
            self._layers.clear()
            remaining = self._project.image_paths
            if remaining:
                self._select_image(remaining[0])
            else:
                self._rebuild_layer_controls()
                self._update_legend()
        self._refresh_images_pane()
        self._toast("Image removed", f"“{Path(path).name}” removed from the project.")

    def _thumbnail(self, path: str) -> QPixmap:
        cached = self._thumb_cache.get(path)
        if cached is not None:
            return cached
        try:
            # Use the same controller path as the canvas. OpenCV cannot read
            # ND2/CZI/LIF and mishandles many OME-TIFF channel layouts.
            img = self._segment.load_preview_image(path)
            import cv2
            img = cv2.resize(img, (38, 30), interpolation=cv2.INTER_AREA)
            img = np.ascontiguousarray(img)
            qimg = QImage(img.data, 38, 30, 38 * 3, QImage.Format.Format_RGB888).copy()
            pm = QPixmap.fromImage(qimg)
        except Exception:
            pm = nuclei_pixmap(38, 30, abs(hash(path)) % 1000, density=2.0)
        self._thumb_cache[path] = pm
        return pm

    @staticmethod
    def _read_error_hint(path: str) -> Optional[str]:
        """A specific, actionable reason ``path`` won't load, or None if it
        looks readable (so the caller shows the raw error instead). Separates
        the two real-world causes behind a bare cv2 ``None`` — a moved/deleted
        file vs macOS's per-folder privacy block — because the fix differs."""
        p = Path(path)
        if not p.exists():
            return ("The file no longer exists at this path — it was moved or deleted. "
                    "Re-import it (new imports are copied into the project).")
        try:
            with open(p, "rb") as f:
                f.read(1)
        except PermissionError:
            return ("macOS is blocking access to this folder — Downloads, Desktop and "
                    "Documents are privacy-protected. Grant your terminal Full Disk Access "
                    "in System Settings › Privacy & Security, or re-import the image "
                    "(new imports are copied into the project and always readable).")
        except OSError:
            pass
        return None

    def _select_image(self, path: str) -> None:
        if path == self._current_image_path:
            return
        try:
            img, pixel_size_um = self._segment.load_preview_with_metadata(path)
        except Exception as e:
            hint = self._read_error_hint(path)
            self._toast("Can't load image", hint or str(e))
            return
        self._current_image_path = path
        self._current_image_array = img
        # Never replace a deliberate manual calibration, but use microscope
        # metadata as the sensible default for a new, uncalibrated project.
        if pixel_size_um is not None and self._project is not None and self._project.settings.pixel_size_um <= 0:
            self._project.settings.pixel_size_um = pixel_size_um
            self._projects.store.save(self._project)
        self._last_result = None
        self._gt_metrics = None
        self._layers.clear()
        self._layers.add(ImageLayer(Path(path).stem or "Image", img), select=False)

        # A previous Run (or batch) on this exact image left a real result on
        # disk (SegmentController.save_result_mask) — load it back instead of
        # starting from an empty mask, so "reopen the project" doesn't throw
        # the work away. A cache from a differently-shaped image (a stale
        # entry from some past bug, or a manually-edited image file) is
        # ignored rather than crashing the whole screen on a shape mismatch.
        saved_mask = self._segment.load_result_mask(self._project, path) if self._project else None
        if saved_mask is not None and saved_mask.shape != img.shape[:2]:
            saved_mask = None
        seg_data = saved_mask if saved_mask is not None else np.zeros(img.shape[:2], dtype=np.int32)
        self._layers.add(LabelsLayer("Segmentation", seg_data))

        gt_path = self._segment.find_gt_for_image(path)
        if gt_path is not None:
            try:
                gt = self._segment.load_gt_mask(gt_path, img.shape[:2])
                gt_layer = LabelsLayer("Ground truth", gt)
                gt_layer.visible = False
                gt_layer.opacity = 0.9
                gt_layer.set_uniform_color(_GT_COLOR)
                self._layers.add(gt_layer, select=False)
            except Exception:
                pass
        if self._canvas is not None:
            self._canvas.mip = False  # the previous image's volume context (if any) no longer applies
            self._canvas.home()
            self._sync_toolbars()
        self._refresh_images_pane()
        self._rebuild_layer_controls()

        if saved_mask is not None:
            self._recompute_results()  # a previous run's real result — show its stats immediately
            if self._layers.find("Ground truth") is not None:
                self._evaluate_gt()
        else:
            self._rebuild_results_pane()

    # ── Layers pane ──────────────────────────────────────────────────────────
    def _layers_pane(self) -> QWidget:
        t = self._t
        # Two states: real content (toolbar + list + controls) once an image is
        # loaded, or a single clean empty-state when nothing is. Before this,
        # the empty pane showed a floating add-layer toolbar over a blank area
        # with a stray scrollbar groove -- the "непонятные линии если там ничего
        # нету" a user reported.
        self._layers_stack = QStackedWidget()

        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        tb = QWidget()
        tbl = QHBoxLayout(tb)
        tbl.setContentsMargins(10, 8, 10, 6)
        tbl.setSpacing(3)
        tbl.addWidget(IconButton("points", t, 30, "New points layer", self._add_points_layer))
        tbl.addWidget(IconButton("shapes", t, 30, "New shapes layer", self._add_shapes_layer))
        tbl.addWidget(IconButton("new_labels", t, 30, "New labels layer", self._add_labels_layer))
        tbl.addStretch(1)
        tbl.addWidget(IconButton("shuffle", t, 30, "Shuffle label colours", self._shuffle_colors))
        del_btn = IconButton("trash", t, 30, "Delete selected layer", self._delete_selected_layer)
        del_btn.setStyleSheet(del_btn.styleSheet() + f"QToolButton:hover{{background:{t['danger_weak']};}}")
        tbl.addWidget(del_btn)
        v.addWidget(tb)

        self._layers_list_container = bare_widget()
        self._layers_list_layout = QVBoxLayout(self._layers_list_container)
        self._layers_list_layout.setContentsMargins(8, 0, 8, 8)
        self._layers_list_layout.setSpacing(2)
        v.addWidget(self._layers_list_container)

        self._layer_controls_container = bare_widget()
        self._layer_controls_layout = QVBoxLayout(self._layer_controls_container)
        self._layer_controls_layout.setContentsMargins(0, 0, 0, 0)
        self._layer_controls_layout.setSpacing(0)
        v.addWidget(_scroll(self._layer_controls_container), 1)

        self._layers_stack.addWidget(content)
        self._layers_stack.addWidget(self._pane_empty_state(
            "image", "No image loaded",
            "Add images to this project, then pick one to segment.",
            "Add images", self._add_images))
        return self._layers_stack

    def _pane_empty_state(self, icon_name: str, title: str, subtitle: str,
                          cta: str, on_cta) -> QWidget:
        """A centred empty-state for a side pane: soft icon, title, one line of
        guidance, and a single call-to-action button. Replaces the old blank
        area + floating toolbar + stray scrollbar groove."""
        t = self._t
        w = bare_widget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(22, 22, 22, 22)
        outer.addStretch(1)
        badge = QLabel()
        badge.setFixedSize(46, 46)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setPixmap(icons.pixmap(icon_name, t["text_muted"], 22))
        badge.setStyleSheet(f"background:{t['surface2']}; border-radius:12px;")
        outer.addWidget(badge, alignment=Qt.AlignmentFlag.AlignHCenter)
        outer.addSpacing(12)
        ttl = label(title, 13.5, t["text"], 600)
        ttl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(ttl)
        outer.addSpacing(3)
        sub = label(subtitle, 12, t["text_muted"])
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(sub)
        outer.addSpacing(14)
        btn = PillButton(cta, t, "ghost", "plus", small=True)
        btn.clicked.connect(lambda: on_cta())
        outer.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)
        return w

    def _sync_layers_pane_state(self) -> None:
        """Show the real Layers content only when there's something to show;
        otherwise the clean empty-state. Called on every layer-list change."""
        stack = getattr(self, "_layers_stack", None)
        if stack is not None:
            stack.setCurrentIndex(0 if len(list(self._layers)) else 1)

    def _on_layers_changed(self) -> None:
        """The one generic subscriber to every LayerList mutation — kept
        cheap (row text/labels only) so it's safe to fire on every value
        tick of a drag, not just structural changes. Never rebuilds
        _layer_controls_container: that would sever an in-progress Slider
        drag whose Slider lives inside that very container."""
        self._refresh_layers_list()
        self._sync_layers_pane_state()
        self._update_legend()
        self._schedule_results_sync()

    def _schedule_results_sync(self) -> None:
        """Debounce a Results/stats recompute when the segmentation mask's
        content actually changes. Gated on a cheap fingerprint so selection,
        visibility and reorder churn (which also fire `notify()`) don't trigger
        a recompute, and only once a result already exists to keep in sync."""
        seg = self._layers.find("Segmentation")
        if seg is None or self._last_result is None or self._current_image_array is None:
            return
        sig = (int(seg.max_label), int((seg.data > 0).sum()))
        if sig == self._results_sig:
            return
        self._results_sig = sig
        self._results_sync_timer.start()

    def _sync_results_after_edit(self) -> None:
        """Fired (debounced) after a mask edit settles: recompute the Results
        panel from the edited mask and persist the new cell count so the
        project card can't drift from what's on screen."""
        if self._layers.find("Segmentation") is None or self._current_image_array is None:
            return
        self._recompute_results()
        self._persist_cell_count()

    def _persist_cell_count(self) -> None:
        if self._project is None or self._last_result is None:
            return
        n = int(self._last_result.get("n_cells", 0))
        if n != self._project.stats.n_cells:
            self._segment.record_run(self._project, n_cells=n)
            self._projects.store.save(self._project)

    def _refresh_layers_list(self) -> None:
        layout = self._layers_list_layout
        _clear_layout(layout)
        for i, layer in enumerate(self._layers):
            layout.addWidget(self._layer_row(i, layer))
        layout.addStretch(1)  # keep rows compact + top-aligned (don't stretch to fill)

    def _layer_row(self, i: int, layer) -> QFrame:
        t = self._t
        sel = i == self._layers.selected_index
        row = QFrame()
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setFixedHeight(44)  # compact + predictable for drag-reorder step maths
        # Qualified -- see _image_row's comment (same pattern, same fix).
        row.setObjectName("LayerRow")
        row.setStyleSheet(
            (f"QFrame#LayerRow{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:8px;}}"
             if sel else
             f"QFrame#LayerRow{{background:transparent; border:1px solid transparent; border-radius:8px;}}"
             f"QFrame#LayerRow:hover{{background:{t['surface2']};}}"))
        lay = QHBoxLayout(row)
        lay.setContentsMargins(8, 7, 8, 7)
        lay.setSpacing(9)
        eye = IconButton("eye" if layer.visible else "eye_off", t, 22, "Toggle visibility",
                        lambda idx=i: self._toggle_layer_visible(idx))
        if layer.visible:
            eye.setIcon(icons.icon("eye", t["signal"], 14))
        lay.addWidget(eye)
        kind = demo.LAYER_TYPE_KIND.get(layer.kind, "muted")
        colm = {"signal": t["signal"], "primary": t["primary"], "warning": t["warning"], "muted": t["text_subtle"]}
        weakm = {"signal": t["signal_weak"], "primary": t["primary_weak"], "warning": t["warning_weak"], "muted": t["surface2"]}
        ty = QLabel()
        ty.setFixedSize(24, 24)
        ty.setPixmap(icons.pixmap(LAYER_TYPE_ICON[layer.kind], colm[kind], 14))
        ty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ty.setStyleSheet(f"background:{weakm[kind]}; border-radius:6px;")
        lay.addWidget(ty)
        name, count, _visible = layer.to_summary()
        nm = QLabel(name)
        nm.setStyleSheet(f"color:{t['text']}; font-size:12.5px; font-weight:600; background:transparent;")
        lay.addWidget(nm, 1)
        lay.addWidget(label(count, 10.5, t["text_muted"]))
        # Drag a row up/down to reorder (z-order = list order); a plain click
        # selects. Both outcomes are deferred one tick -- select and move each
        # rebuild _layers_list_layout (this row's own parent) synchronously,
        # the same SIP hazard the image rows guard.
        def _press(e, r=row):
            r._drag_from_y = e.position().y()
        def _release(e, idx=i, r=row):
            start = getattr(r, "_drag_from_y", None)
            r._drag_from_y = None
            n = len(list(self._layers))
            row_h = r.height() + max(self._layers_list_layout.spacing(), 0)
            steps = int(round((e.position().y() - start) / max(row_h, 1))) if start is not None else 0
            dst = max(0, min(n - 1, idx + steps))
            if dst != idx:
                QTimer.singleShot(0, lambda: self._move_layer(idx, dst))
            else:
                QTimer.singleShot(0, lambda: self._select_layer(idx))
        row.mousePressEvent = _press
        row.mouseReleaseEvent = _release
        return row

    def _select_layer(self, idx: int) -> None:
        self._layers.select(idx)
        self._rebuild_layer_controls()

    def _move_layer(self, src: int, dst: int) -> None:
        """Reorder layers (drag in the Layers list). z-order = list order, so
        this changes what draws on top; selection follows the moved layer."""
        self._layers.move(src, dst)
        self._rebuild_layer_controls()

    def _toggle_layer_visible(self, idx: int) -> None:
        self._layers.toggle_visible(idx)

    def _add_points_layer(self) -> None:
        self._layers.add(PointsLayer(self._layers.unique_name("Points")))
        self._rebuild_layer_controls()

    def _add_shapes_layer(self) -> None:
        self._layers.add(ShapesLayer(self._layers.unique_name("Shapes")))
        self._rebuild_layer_controls()

    def _add_labels_layer(self) -> None:
        shape = self._canvas._base_shape() if self._canvas is not None else None
        if shape is None:
            self._toast("No image loaded", "Load an image before adding a labels layer.")
            return
        self._layers.add(LabelsLayer(self._layers.unique_name("Labels"), np.zeros(shape, dtype=np.int32)))
        self._rebuild_layer_controls()

    def _delete_selected_layer(self) -> None:
        if self._layers.selected is not None:
            self._layers.remove_selected()
            self._rebuild_layer_controls()

    def _shuffle_colors(self) -> None:
        target = self._canvas.edit_target() if self._canvas is not None else None
        if target is not None:
            target.shuffle_colors()
            self._layers.notify()

    def _undo(self) -> None:
        """Undo the last mask edit (paint/erase/fill/polygon) — ⌘Z / Ctrl+Z and
        the canvas bar's undo button. A no-op with nothing to undo."""
        if self._canvas is not None and self._canvas.undo():
            self._update_legend()

    def _redo(self) -> None:
        if self._canvas is not None and self._canvas.redo():
            self._update_legend()

    # ── layer controls (dispatch by kind) ────────────────────────────────────
    def _rebuild_layer_controls(self) -> None:
        layout = self._layer_controls_layout
        _clear_layout(layout)
        t = self._t
        layer = self._layers.selected
        if layer is None:
            empty = label("Select a layer to see its settings.", 12, t["text_muted"])
            empty.setWordWrap(True)
            wrap = QWidget()
            wv = QVBoxLayout(wrap)
            wv.setContentsMargins(12, 12, 12, 12)
            wv.addWidget(empty)
            layout.addWidget(wrap)
            return
        if isinstance(layer, LabelsLayer):
            layout.addWidget(self._labels_controls(layer))
        elif isinstance(layer, ImageLayer):
            layout.addWidget(self._image_controls(layer))
        elif isinstance(layer, PointsLayer):
            layout.addWidget(self._points_controls(layer))
        elif isinstance(layer, ShapesLayer):
            layout.addWidget(self._shapes_controls(layer))

    def _labels_controls(self, layer: LabelsLayer) -> QWidget:
        t = self._t
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(11)

        title = QLabel(f"● {html.escape(layer.name)} · labels")
        title.setStyleSheet(f"color:{t['text_muted']}; font-size:10.5px; font-weight:600; letter-spacing:0.6px; background:transparent;")
        v.addWidget(title)

        tools = QGridLayout()
        tools.setSpacing(3)
        for i, (icon_name, tip, mode) in enumerate(MODE_ICONS):
            b = QToolButton()
            b.setToolTip(tip)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFixedHeight(30)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            on = mode != "__shuffle__" and self._canvas is not None and self._canvas.mode == mode
            b.setIcon(icons.icon(icon_name, "#fff" if on else t["text_subtle"], 15))
            b.setIconSize(QSize(15, 15))
            if on:
                b.setStyleSheet(f"QToolButton{{background:{t['primary']}; border:none; border-radius:7px;}}")
            else:
                b.setStyleSheet(
                    f"QToolButton{{background:{t['surface2']}; border:1px solid transparent; border-radius:7px;}}"
                    f"QToolButton:hover{{background:{t['surface']}; border-color:{t['border']};}}")
            if mode == "__shuffle__":
                b.clicked.connect(self._shuffle_colors)
            else:
                b.clicked.connect(lambda _=False, m=mode: self._set_canvas_mode(m))
            tools.addWidget(b, 0, i)
        v.addLayout(tools)

        opacity_badge = Badge(f"{layer.opacity:.2f}", t)
        v.addWidget(FieldRow("opacity", opacity_badge, t))
        opacity_slider = Slider(t, layer.opacity, t["signal"])
        opacity_slider.changed.connect(
            lambda val, ly=layer, bd=opacity_badge: self._set_layer_opacity(ly, val, bd))
        v.addWidget(opacity_slider)

        # label swatch + stepper + a "new label" button (selects max+1 -- the
        # fast way to start a fresh instance, matching napari's "+")
        labelrow = QHBoxLayout()
        labelrow.setSpacing(8)
        chip = QFrame()
        chip.setFixedSize(22, 22)
        r, g, b = layer.get_color(layer.selected_label if layer.selected_label > 0 else 1)
        chip.setStyleSheet(f"background:rgb({r},{g},{b}); border:1px solid {t['border_strong']}; border-radius:5px;")
        labelrow.addWidget(chip)
        label_stepper = Stepper(layer.selected_label, t, step=1, minimum=0, maximum=100000)
        label_stepper.changed.connect(lambda val, ly=layer: self._set_selected_label(ly, val))
        labelrow.addWidget(label_stepper)
        labelrow.addWidget(IconButton("plus", t, 26, "New label (max + 1)",
                                      lambda ly=layer: self._new_label(ly)))
        v.addWidget(FieldRow("label", self._wrap(labelrow), t))

        v.addWidget(GroupLabel("label colours · more choices", t))
        pal = QGridLayout()
        pal.setSpacing(4)
        for i, col in enumerate(demo.LABEL_COLORS):
            sw = QFrame()
            sw.setFixedHeight(16)
            sw.setCursor(Qt.CursorShape.PointingHandCursor)
            border = f"1px solid rgba(128,128,128,0.3)"
            sw.setStyleSheet(f"background:{col}; border:{border}; border-radius:4px;")
            # Deferred for the same reason as the layer/image rows:
            # _pick_label_color ends in _rebuild_layer_controls(), which
            # tears down this very swatch's own parent container.
            sw.mouseReleaseEvent = lambda e, c=col, ly=layer: QTimer.singleShot(
                0, lambda: self._pick_label_color(ly, c))
            pal.addWidget(sw, i // 9, i % 9)
        v.addLayout(pal)

        brush_badge = Badge(str(layer.brush_size), t)
        v.addWidget(FieldRow("brush size", brush_badge, t))
        brush_slider = Slider(t, min(1.0, layer.brush_size / 100.0))
        brush_slider.changed.connect(lambda val, ly=layer, bd=brush_badge: self._set_brush_size(ly, val, bd))
        v.addWidget(brush_slider)

        # The less-often-touched knobs (blending, direct-colour mode, contour
        # outline width, 3-D edit reach, and the fill-behaviour flags) move
        # into a collapsed "Advanced" section so the pane isn't a dense wall of
        # ten fields -- the common ones (tools, opacity, label, palette, brush)
        # stay visible; napari keeps these same knobs one fold deeper too.
        adv = Accordion("Advanced", t, lead="settings", open_=False)
        adv.add(FieldRow("blending", SelectBox(
            layer.blending, t, options=list(BLENDING_MODES),
            on_select=lambda choice, ly=layer: self._set_layer_blending(ly, choice)), t))
        adv.add(FieldRow("colour mode", SelectBox(
            "direct" if layer.color_overrides else "auto", t, options=["auto", "direct"],
            on_select=lambda c, ly=layer: self._set_color_mode(ly, c)), t))
        contour_stepper = Stepper(layer.contour, t, step=1, minimum=0, maximum=20)
        contour_stepper.changed.connect(lambda val, ly=layer: self._set_layer_int_attr(ly, "contour", val))
        adv.add(FieldRow("contour", contour_stepper, t))
        dim_stepper = Stepper(layer.n_edit_dimensions, t, step=1, minimum=2, maximum=3)
        dim_stepper.changed.connect(lambda val, ly=layer: self._set_layer_int_attr(ly, "n_edit_dimensions", val))
        adv.add(FieldRow("n edit dim", dim_stepper, t))
        for name, attr in [("contiguous", "contiguous"), ("preserve labels", "preserve_labels"),
                           ("show selected", "show_selected_label")]:
            adv.add(self._check(name, getattr(layer, attr),
                                lambda ly=layer, a=attr: self._toggle_layer_bool(ly, a)))
        v.addWidget(adv)
        v.addStretch(1)
        return w

    def _new_label(self, layer: LabelsLayer) -> None:
        """Select the next free label id (max + 1) so a fresh instance can be
        painted without hunting for an unused number -- napari's "increment
        selected label". Rebuilds the controls so the stepper + swatch update."""
        layer.selected_label = layer.max_label + 1
        self._rebuild_layer_controls()

    def _image_controls(self, layer: ImageLayer) -> QWidget:
        t = self._t
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(11)
        title = QLabel(f"● {html.escape(layer.name)} · image")
        title.setStyleSheet(f"color:{t['text_muted']}; font-size:10.5px; font-weight:600; letter-spacing:0.6px; background:transparent;")
        v.addWidget(title)

        opacity_badge = Badge(f"{layer.opacity:.2f}", t)
        v.addWidget(FieldRow("opacity", opacity_badge, t))
        sl = Slider(t, layer.opacity, t["primary"])
        sl.changed.connect(lambda val, ly=layer, bd=opacity_badge: self._set_layer_opacity(ly, val, bd))
        v.addWidget(sl)

        v.addWidget(FieldRow("blending", SelectBox(
            layer.blending, t, options=list(BLENDING_MODES),
            on_select=lambda c, ly=layer: self._set_layer_blending(ly, c)), t))
        v.addWidget(FieldRow("colormap", SelectBox(
            layer.colormap, t, options=list(IMAGE_COLORMAPS),
            on_select=lambda c, ly=layer: self._set_image_colormap(ly, c)), t))

        gamma_badge = Badge(f"{layer.gamma:.2f}", t)
        v.addWidget(FieldRow("gamma", gamma_badge, t))
        gsl = Slider(t, min(1.0, layer.gamma / 3.0))
        gsl.changed.connect(lambda val, ly=layer, bd=gamma_badge: self._set_image_gamma(ly, val, bd))
        v.addWidget(gsl)

        # Contrast limits: min/max sliders + an Auto (1–99 percentile) button,
        # replacing the old read-only "lo – hi" badge -- napari's editable
        # contrast. Sliders map 0..1 across the image's own data range.
        lo, hi = layer.contrast_limits
        dmin = float(layer.data.min()) if layer.data.size else 0.0
        dmax = float(layer.data.max()) if layer.data.size else 255.0
        drange = (dmax - dmin) or 1.0
        auto_btn = PillButton("Auto", t, "ghost", None, small=True)
        auto_btn.clicked.connect(lambda _=False, ly=layer: self._auto_contrast(ly))
        v.addWidget(FieldRow("contrast", auto_btn, t))
        lo_badge = Badge(f"{lo:.0f}", t)
        v.addWidget(FieldRow("min", lo_badge, t))
        lo_sl = Slider(t, min(1.0, max(0.0, (lo - dmin) / drange)), t["primary"])
        lo_sl.changed.connect(lambda val, ly=layer, bd=lo_badge:
                              self._set_contrast(ly, "lo", val, dmin, drange, bd))
        v.addWidget(lo_sl)
        hi_badge = Badge(f"{hi:.0f}", t)
        v.addWidget(FieldRow("max", hi_badge, t))
        hi_sl = Slider(t, min(1.0, max(0.0, (hi - dmin) / drange)), t["primary"])
        hi_sl.changed.connect(lambda val, ly=layer, bd=hi_badge:
                              self._set_contrast(ly, "hi", val, dmin, drange, bd))
        v.addWidget(hi_sl)
        v.addStretch(1)
        return w

    def _points_controls(self, layer: PointsLayer) -> QWidget:
        t = self._t
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(11)
        title = QLabel(f"● {html.escape(layer.name)} · points")
        title.setStyleSheet(f"color:{t['text_muted']}; font-size:10.5px; font-weight:600; letter-spacing:0.6px; background:transparent;")
        v.addWidget(title)
        v.addWidget(FieldRow("count", Badge(str(len(layer.points)), t), t))

        opacity_badge = Badge(f"{layer.opacity:.2f}", t)
        v.addWidget(FieldRow("opacity", opacity_badge, t))
        sl = Slider(t, layer.opacity, t["warning"])
        sl.changed.connect(lambda val, ly=layer, bd=opacity_badge: self._set_layer_opacity(ly, val, bd))
        v.addWidget(sl)

        size_badge = Badge(str(int(layer.size)), t)
        v.addWidget(FieldRow("point size", size_badge, t))
        ssl = Slider(t, min(1.0, layer.size / 40.0))
        ssl.changed.connect(lambda val, ly=layer, bd=size_badge: self._set_point_size(ly, val, bd))
        v.addWidget(ssl)

        clear_btn = PillButton("Clear all points", t, "ghost", "trash", small=True)
        clear_btn.clicked.connect(lambda ly=layer: self._clear_points(ly))
        v.addWidget(clear_btn)
        v.addStretch(1)
        return w

    def _shapes_controls(self, layer: ShapesLayer) -> QWidget:
        t = self._t
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(11)
        title = QLabel(f"● {html.escape(layer.name)} · shapes")
        title.setStyleSheet(f"color:{t['text_muted']}; font-size:10.5px; font-weight:600; letter-spacing:0.6px; background:transparent;")
        v.addWidget(title)
        v.addWidget(FieldRow("count", Badge(str(len(layer.shapes)), t), t))

        opacity_badge = Badge(f"{layer.opacity:.2f}", t)
        v.addWidget(FieldRow("opacity", opacity_badge, t))
        sl = Slider(t, layer.opacity, t["primary"])
        sl.changed.connect(lambda val, ly=layer, bd=opacity_badge: self._set_layer_opacity(ly, val, bd))
        v.addWidget(sl)

        width_badge = Badge(f"{layer.edge_width:.1f}", t)
        v.addWidget(FieldRow("edge width", width_badge, t))
        wsl = Slider(t, min(1.0, layer.edge_width / 10.0))
        wsl.changed.connect(lambda val, ly=layer, bd=width_badge: self._set_edge_width(ly, val, bd))
        v.addWidget(wsl)

        clear_btn = PillButton("Clear all shapes", t, "ghost", "trash", small=True)
        clear_btn.clicked.connect(lambda ly=layer: self._clear_shapes(ly))
        v.addWidget(clear_btn)
        v.addStretch(1)
        return w

    def _check(self, name: str, on: bool, on_toggle=None) -> QWidget:
        t = self._t
        row = QFrame()
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(9)
        box = QLabel()
        box.setFixedSize(16, 16)
        if on:
            box.setPixmap(icons.pixmap("check", "#04211f", 11))
            box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.setStyleSheet(f"background:{t['signal']}; border-radius:4px;")
        else:
            box.setStyleSheet(f"background:{t['inset']}; border:1px solid {t['border_strong']}; border-radius:4px;")
        lay.addWidget(box)
        lay.addWidget(label(name, 12, t["text_subtle"]))
        lay.addStretch(1)
        if on_toggle is not None:
            # Deferred for the same reason as the layer/image rows and the
            # colour swatches above: rebuilding _layer_controls_container
            # (via _rebuild_layer_controls()) while this very row's own
            # mouseReleaseEvent is still executing crashes PyQt/SIP's
            # bookkeeping for that virtual call outright (a hard process
            # abort — "TypeError: invalid argument to sipBadCatcherResult()"
            # — not a catchable exception). Confirmed against a real running
            # session: clicking "preserve labels"/"show selected" reproduced
            # it every time.
            row.mouseReleaseEvent = lambda e: QTimer.singleShot(
                0, lambda: (on_toggle(), self._rebuild_layer_controls()))
        return row

    def _wrap(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        return w

    # ── layer-control handlers ───────────────────────────────────────────────
    def _set_canvas_mode(self, mode: str) -> None:
        self._canvas.set_mode(mode)
        self._rebuild_layer_controls()
        self._sync_toolbars()

    def _on_canvas_mode_changed(self, mode: str) -> None:
        """The canvas switched tool itself (a single-key shortcut like B/E/F).
        It already called set_mode; just refresh the UI that mirrors the mode
        (the labels tool-row highlight + the floating strip / viewer bar)."""
        self._rebuild_layer_controls()
        self._sync_toolbars()

    def _set_layer_opacity(self, layer, value: float, badge: Badge) -> None:
        layer.opacity = value
        badge.setText(f"{value:.2f}")
        self._layers.notify()

    def _set_layer_blending(self, layer, choice: str) -> None:
        layer.blending = choice
        self._layers.notify()

    def _set_selected_label(self, layer: LabelsLayer, value: float) -> None:
        layer.selected_label = int(value)
        self._layers.notify()
        self._rebuild_layer_controls()  # updates the colour-chip swatch

    def _pick_label_color(self, layer: LabelsLayer, hex_color: str) -> None:
        h = hex_color.lstrip("#")
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        overrides = dict(layer.color_overrides)
        overrides[layer.selected_label if layer.selected_label > 0 else 1] = rgb
        layer.set_color_overrides(overrides)
        self._layers.notify()
        self._rebuild_layer_controls()

    def _set_brush_size(self, layer: LabelsLayer, value: float, badge: Badge) -> None:
        size = max(1, round(value * 100))
        layer.brush_size = size
        badge.setText(str(size))
        self._layers.notify()

    def _set_color_mode(self, layer: LabelsLayer, choice: str) -> None:
        if choice == "auto":
            layer.clear_color_overrides()
            self._layers.notify()

    def _set_layer_int_attr(self, layer, attr: str, value: float) -> None:
        setattr(layer, attr, int(value))
        self._layers.notify()

    def _toggle_layer_bool(self, layer, attr: str) -> None:
        setattr(layer, attr, not getattr(layer, attr))
        self._layers.notify()

    def _set_image_colormap(self, layer: ImageLayer, choice: str) -> None:
        layer.colormap = choice
        self._layers.notify()

    def _set_image_gamma(self, layer: ImageLayer, value: float, badge: Badge) -> None:
        gamma = max(0.05, value * 3.0)
        layer.gamma = gamma
        badge.setText(f"{gamma:.2f}")
        self._layers.notify()

    def _set_contrast(self, layer: ImageLayer, which: str, value01: float,
                      dmin: float, drange: float, badge: Badge) -> None:
        """Move one contrast limit. Slider 0..1 maps across the image's data
        range; the two limits are kept from crossing."""
        value = dmin + value01 * drange
        lo, hi = layer.contrast_limits
        if which == "lo":
            lo = min(value, hi - 1e-3)
        else:
            hi = max(value, lo + 1e-3)
        layer.contrast_limits = (lo, hi)
        badge.setText(f"{value:.0f}")
        self._layers.notify()

    def _auto_contrast(self, layer: ImageLayer) -> None:
        """Robust auto-contrast: stretch to the 1–99 percentile (napari's
        auto-contrast), falling back to full min/max on a flat image. Rebuilds
        the controls so the min/max sliders jump to the new limits."""
        data = layer.data
        if data.size:
            lo = float(np.percentile(data, 1))
            hi = float(np.percentile(data, 99))
            if hi <= lo:
                lo, hi = float(data.min()), float(data.max())
            if hi <= lo:
                hi = lo + 1.0
            layer.contrast_limits = (lo, hi)
            self._layers.notify()
            self._rebuild_layer_controls()

    def _set_point_size(self, layer: PointsLayer, value: float, badge: Badge) -> None:
        size = max(2, round(value * 40))
        layer.size = size
        badge.setText(str(size))
        self._layers.notify()

    def _set_edge_width(self, layer: ShapesLayer, value: float, badge: Badge) -> None:
        width = max(0.5, round(value * 10, 1))
        layer.edge_width = width
        badge.setText(f"{width:.1f}")
        self._layers.notify()

    def _clear_points(self, layer: PointsLayer) -> None:
        layer.points = []
        self._layers.notify()
        self._rebuild_layer_controls()

    def _clear_shapes(self, layer: ShapesLayer) -> None:
        layer.shapes = []
        self._layers.notify()
        self._rebuild_layer_controls()

    # ── centre: viewport ─────────────────────────────────────────────────────
    def _viewport(self) -> QWidget:
        t = self._t
        vp = QFrame()
        # The 1px dividers between the canvas and the two side panels live here,
        # on the canvas edges, NOT as a border-left/right on the panels. A 1px
        # CSS border on a WA_StyledBackground panel offsets its content and, at
        # HiDPI, leaves a `border`-coloured hairline seam down the left edge of
        # every text block inside it (the reported "чёрточки у начала текста" all
        # over the Segment/Results inspector -- isolated to exactly this border).
        # On the canvas (a dark image, no text) the same hairline is invisible,
        # so the divider reads identically without seaming the panels' text.
        vp.setStyleSheet(
            f"background:#07090c; border-left:1px solid {t['border']};"
            f" border-right:1px solid {t['border']};")
        self._canvas = Canvas(t, self._layers, on_status=self._on_canvas_status,
                             on_label_picked=self._on_label_picked,
                             on_mode_change=self._on_canvas_mode_changed)
        lay = QVBoxLayout(vp)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._canvas)

        legend = QWidget(vp)
        leg = QHBoxLayout(legend)
        leg.setContentsMargins(0, 0, 0, 0)
        leg.setSpacing(7)
        self._legend_detected = QLabel()
        self._legend_label = QLabel()
        for lbl in (self._legend_detected, self._legend_label):
            lbl.setStyleSheet(
                "color:#eaf7f5; background:rgba(8,12,16,0.6); border:1px solid rgba(255,255,255,0.1);"
                "border-radius:999px; padding:3px 9px; font-size:11px; font-weight:600;")
            leg.addWidget(lbl)
        legend.move(14, 14)
        self._legend = legend

        tools = QFrame(vp)
        # Qualified -- see components.EngineChip's comment. Every current
        # child here is a self-styled IconButton (protected regardless),
        # but an unscoped rule is a silent trap for whoever adds a plain
        # label here later -- fixed the same way regardless of current
        # visible impact.
        tools.setObjectName("FloatingToolStrip")
        tools.setStyleSheet(
            f"QFrame#FloatingToolStrip{{background:rgba(21,24,30,0.86); border:1px solid {t['border']};"
            f" border-radius:11px;}}")
        tl = QVBoxLayout(tools)
        tl.setContentsMargins(5, 5, 5, 5)
        tl.setSpacing(4)
        # Navigation + prompts only -- Paint used to sit here too, duplicating
        # the Labels tool row (and now the B shortcut), and Home reused the
        # very same "target" glyph as Pan two rows up. Distinct icons now:
        # pan, add-a-prompt-point, home.
        self._floating_tool_buttons: list[tuple[QToolButton, str, str]] = []
        for icon_name, action in [("target", PAN_ZOOM),
                                   ("points", "__add_point__"), ("home", "__home__")]:
            b = IconButton(icon_name, t, 30)
            b.clicked.connect(lambda _=False, a=action: self._on_floating_tool(a))
            tl.addWidget(b)
            self._floating_tool_buttons.append((b, action, icon_name))
        self._vp_tools = tools

        vbar = QFrame(vp)
        # Qualified -- see `tools` above, same reasoning.
        vbar.setObjectName("ViewerBar")
        vbar.setStyleSheet(
            f"QFrame#ViewerBar{{background:rgba(21,24,30,0.86); border:1px solid {t['border']};"
            f" border-radius:11px;}}")
        vl = QHBoxLayout(vbar)
        vl.setContentsMargins(5, 5, 5, 5)
        vl.setSpacing(3)
        vbar_defs = [
            ("undo", "Undo  ⌘Z", self._undo),
            ("redo", "Redo  ⇧⌘Z", self._redo),
            ("console", "Toggle console", self._toggle_logs_console),
            ("cube3d", "Toggle 2D / 3D", self._toggle_mip),
            ("refresh", "Roll dimensions", self._roll_channel),
            ("transpose", "Transpose", self._toggle_transpose),
            ("grid", "Grid mode", self._toggle_grid),
            ("home", "Reset view", self._canvas.home),
        ]
        self._vbar_buttons: dict[str, QToolButton] = {}
        for icon_name, tip, handler in vbar_defs:
            b = IconButton(icon_name, t, 30, tip, handler)
            vl.addWidget(b)
            self._vbar_buttons[icon_name] = b
        self._vp_bar = vbar

        self._vp_status = QLabel("", vp)
        self._vp_status.setStyleSheet(
            "color:#dbe6ee; background:rgba(8,12,16,0.6); border:1px solid rgba(255,255,255,0.1);"
            "border-radius:999px; padding:5px 11px; font-size:11.5px; font-weight:600;")

        vp._overlays = (legend, tools, vbar, self._vp_status)
        vp.resizeEvent = lambda e: self._place_overlays(vp)
        self._update_legend()
        self._sync_toolbars()
        return vp

    def _sync_toolbars(self) -> None:
        """Re-style (never rebuild) the floating tool strip + viewer bar so
        their "on" highlight always matches live Canvas state — mirrors the
        Labels controls' mode-tool-grid highlight treatment exactly.

        Uses each button's own stored icon name, never re-derived from its
        mode/action string: "pan_zoom"/"paint" aren't real icons.PATHS keys
        (only their semantic names "target"/"brush" are), so a previous
        version that reconstructed the name from the action here silently
        redrew the wrong glyph — icons.py falls back to a generic chevron
        for any unknown name — on every single restyle, i.e. on almost
        every interaction."""
        t = self._t
        mode = self._canvas.mode if self._canvas is not None else PAN_ZOOM
        for btn, action, icon_name in self._floating_tool_buttons:
            on = action == mode  # only PAN_ZOOM/PAINT persist as a "mode"; the
            # add-point/home actions are one-shot and never show as active
            self._style_toolbar_button(btn, icon_name, on)
        if self._canvas is not None:
            self._style_toolbar_button(self._vbar_buttons["cube3d"], "cube3d", self._canvas.mip)
            self._style_toolbar_button(self._vbar_buttons["grid"], "grid", self._canvas.grid)
            self._style_toolbar_button(self._vbar_buttons["transpose"], "transpose", self._canvas.transposed)

    def _style_toolbar_button(self, btn: QToolButton, icon_name: str, on: bool) -> None:
        t = self._t
        if on:
            btn.setStyleSheet(f"QToolButton{{background:{t['signal_weak']}; border-radius:7px;}}")
            btn.setIcon(icons.icon(icon_name, t["signal"], 16))
        else:
            btn.setStyleSheet(
                f"QToolButton{{background:transparent; border:1px solid transparent; border-radius:8px;}}"
                f"QToolButton:hover{{background:{t['surface2']}; border-color:{t['border']};}}")
            btn.setIcon(icons.icon(icon_name, t["text_muted"], 16))

    def _place_overlays(self, vp) -> None:
        legend, tools, vbar, status = vp._overlays
        w, h = vp.width(), vp.height()
        legend.adjustSize()
        tools.adjustSize()
        vbar.adjustSize()
        status.adjustSize()
        tools.move(w - tools.width() - 14, 14)
        vbar.move(14, h - vbar.height() - 14)
        status.move(w - status.width() - 14, h - status.height() - 14)

    def _on_canvas_status(self, text: str) -> None:
        self._vp_status.setText(f"●  {text}")
        self._vp_status.adjustSize()
        if self._vp_status.parentWidget():
            self._place_overlays(self._vp_status.parentWidget())

    def _on_label_picked(self, label_id: int) -> None:
        self._update_legend()
        if isinstance(self._layers.selected, LabelsLayer):
            self._rebuild_layer_controls()

    def _update_legend(self, detected: Optional[int] = None) -> None:
        labels_layers = self._layers.by_kind("labels")
        primary = labels_layers[0] if labels_layers else None
        # Live path (fires on every paint-drag tick) uses the cheap max id;
        # the exact distinct-cell count (n_labels / regionprops is O(n log n))
        # is passed in by _recompute_results after an edit settles, so the
        # legend lands on the same number the Results panel shows without
        # paying np.unique on every mouse-move (16ms on a 2k² mask).
        if detected is None:
            detected = primary.max_label if primary else 0
        self._legend_detected.setText(f"● {detected} detected")
        target = self._canvas.edit_target() if self._canvas is not None else None
        self._legend_label.setText(f"● label {target.selected_label if target else 0}")
        self._legend_detected.adjustSize()
        self._legend_label.adjustSize()
        self._legend.adjustSize()

    def _on_floating_tool(self, action: str) -> None:
        if action == "__home__":
            self._canvas.home()
            self._sync_toolbars()
            return
        if action == "__add_point__":
            points = self._layers.by_kind("points")
            if points:
                self._layers.select(self._layers.index_of(points[0]))
            else:
                self._layers.add(PointsLayer(self._layers.unique_name("Prompts")))
            self._canvas.set_mode(PAINT)  # any non-pan_zoom mode routes clicks to the selected layer
            self._rebuild_layer_controls()
            self._sync_toolbars()
            return
        self._canvas.set_mode(action)
        self._rebuild_layer_controls()
        self._sync_toolbars()

    def _toggle_logs_console(self) -> None:
        if self._on_toggle_logs:
            self._on_toggle_logs()

    def _toggle_mip(self) -> None:
        """Matches real napari's own ndisplay toggle: no dimensionality
        guard, always does something (a real max-intensity projection for a
        loaded z-stack; a perspective tilt — Canvas._draw_pseudo_3d — for a
        plain 2-D image, since this canvas has no GPU 3-D camera)."""
        self._canvas.toggle_mip()
        self._sync_toolbars()

    def _roll_channel(self) -> None:
        if not self._canvas.roll_channel():
            self._toast("Only one channel loaded",
                       "Roll dimensions cycles which image channel is visible — "
                       "this image only has one.")

    def _toggle_transpose(self) -> None:
        self._canvas.toggle_transpose()
        self._sync_toolbars()

    def _toggle_grid(self) -> None:
        self._canvas.toggle_grid()
        self._sync_toolbars()

    # ── right: inspector ─────────────────────────────────────────────────────
    def _inspector(self) -> QWidget:
        t = self._t
        panel = QFrame()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(520)
        # No border-left here: the divider is drawn on the canvas edge instead
        # (see _viewport) so the inspector's text doesn't pick up a HiDPI border
        # seam -- the root cause of the "чёрточки у начала текста" report.
        panel.setStyleSheet(f"background:{t['surface']};")
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        tabs = SegControl(["Segment", "Results"], t, 0, compact=True)
        tw = QWidget()
        twl = QHBoxLayout(tw)
        twl.setContentsMargins(12, 10, 12, 4)
        twl.addWidget(tabs)
        v.addWidget(tw)

        stack = QStackedWidget()
        self._segment_container = bare_widget()
        self._segment_container_layout = QVBoxLayout(self._segment_container)
        self._segment_container_layout.setContentsMargins(14, 12, 14, 14)
        self._segment_container_layout.setSpacing(14)
        stack.addWidget(_scroll(self._segment_container))

        self._results_container = bare_widget()
        self._results_container_layout = QVBoxLayout(self._results_container)
        self._results_container_layout.setContentsMargins(14, 12, 14, 14)
        self._results_container_layout.setSpacing(14)
        stack.addWidget(_scroll(self._results_container))

        tabs.changed.connect(stack.setCurrentIndex)
        v.addWidget(stack, 1)
        v.addWidget(self._runbar())
        return panel

    # ── Segment settings pane ────────────────────────────────────────────────
    def _rebuild_segment_pane(self) -> None:
        layout = self._segment_container_layout
        _clear_layout(layout)
        t = self._t
        if self._project is None:
            empty = label("Open or create a project to configure segmentation.", 12, t["text_muted"])
            empty.setWordWrap(True)
            layout.addWidget(empty)
            return
        s = self._project.settings

        layout.addWidget(GroupLabel("Engine", t))
        engines = self._segment.list_available_engines()
        self._engine_label_to_key = {}
        options = []
        current_text = ENGINE_LABELS.get(s.engine, s.engine)
        for key, elabel, available in engines:
            text = elabel if available else f"{elabel}  (not installed)"
            self._engine_label_to_key[text] = key
            options.append(text)
            if key == s.engine:
                current_text = text
        layout.addWidget(SelectBox(current_text, t, "models", t["primary"],
                                   options=options, on_select=self._on_engine_select))

        model_row = self._model_field(s)
        if model_row is not None:
            layout.addWidget(model_row)

        layout.addWidget(GroupLabel("Quality preset", t))
        active_idx = (QUALITY_PRESET_NAMES.index(s.quality_preset)
                     if s.quality_preset in QUALITY_PRESET_NAMES else 1)
        preset_ctrl = SegControl(QUALITY_PRESET_NAMES, t, active_idx, compact=True)
        preset_ctrl.changed.connect(self._on_quality_preset)
        layout.addWidget(preset_ctrl)

        layout.addWidget(hline(t))
        layout.addWidget(GroupLabel("Detection thresholds", t))
        pps_stepper = Stepper(s.points_per_side, t, step=4, minimum=4, maximum=128)
        pps_stepper.changed.connect(lambda v: self._set_setting_custom("points_per_side", int(v)))
        layout.addWidget(FieldRow("Points / side", pps_stepper, t))

        iou_badge = Badge(f"{s.pred_iou_thresh:.2f}", t)
        layout.addWidget(FieldRow("IoU threshold", iou_badge, t))
        iou_slider = Slider(t, s.pred_iou_thresh)
        iou_slider.changed.connect(
            lambda v, bd=iou_badge: self._on_threshold_change("pred_iou_thresh", v, bd))
        layout.addWidget(iou_slider)

        stab_badge = Badge(f"{s.stability_score_thresh:.2f}", t)
        layout.addWidget(FieldRow("Stability score", stab_badge, t))
        stab_slider = Slider(t, s.stability_score_thresh)
        stab_slider.changed.connect(
            lambda v, bd=stab_badge: self._on_threshold_change("stability_score_thresh", v, bd))
        layout.addWidget(stab_slider)

        area_stepper = Stepper(s.min_mask_area, t, step=5, minimum=0, maximum=5000)
        area_stepper.changed.connect(lambda v: self._set_setting_custom("min_mask_area", int(v)))
        layout.addWidget(FieldRow("Min mask area", area_stepper, t))

        layout.addWidget(hline(t))
        layout.addWidget(GroupLabel("Image", t))
        resize_opts = ["256 px", "384 px", "512 px", "768 px", "1024 px"]
        layout.addWidget(FieldRow("Resize", SelectBox(
            f"{s.resize_size} px", t, options=resize_opts,
            on_select=lambda choice: self._set_setting("resize_size", int(choice.split()[0]))), t))
        pixel_edit = QLineEdit(f"{s.pixel_size_um:.3f}")
        pixel_edit.setFixedWidth(90)
        pixel_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        pixel_edit.editingFinished.connect(lambda le=pixel_edit: self._on_pixel_size_edited(le))
        layout.addWidget(FieldRow("Pixel size (µm/px)", pixel_edit, t))
        n_ch = len(s.channels) if s.channels else "RGB"
        layout.addWidget(FieldRow("Channels", Badge(str(n_ch), t), t))

        clahe_toggle = Toggle(t, s.clahe)
        clahe_toggle.toggled.connect(lambda on: self._set_setting("clahe", on))
        layout.addWidget(FieldRow("CLAHE contrast", clahe_toggle, t))

        tiled_toggle = Toggle(t, s.tiled)
        tiled_toggle.toggled.connect(lambda on: self._set_setting("tiled", on))
        layout.addWidget(FieldRow("Large image (tiling)", tiled_toggle, t))

        layout.addWidget(hline(t))
        layout.addWidget(GroupLabel("Overlays", t))
        seg_layers = self._layers.by_kind("labels")
        pred_visible = any(l.visible for l in seg_layers if l.name != "Ground truth")
        pred_toggle = Toggle(t, pred_visible)
        pred_toggle.toggled.connect(self._toggle_show_predictions)
        layout.addWidget(FieldRow("Show predictions", pred_toggle, t))
        gt_layer = self._layers.find("Ground truth")
        gt_toggle = Toggle(t, gt_layer.visible if gt_layer else False)
        gt_toggle.toggled.connect(self._toggle_show_gt)
        layout.addWidget(FieldRow("Show ground truth", gt_toggle, t))

        layout.addWidget(self._engine_settings_accordion(s))
        layout.addStretch(1)

    def _model_field(self, s) -> Optional[QWidget]:
        t = self._t
        if s.engine == "cellseg1":
            models = self._segment.list_lora_models()
            current = Path(s.model_name).stem if s.model_name else "Choose model…"
            options = [m.name for m in models] + ["Browse…"]
            box = SelectBox(current, t, options=options, on_select=self._on_model_select)
            return FieldRow("Model", box, t)
        if s.engine == "sam2":
            labels = {"tiny": "Tiny", "small": "Small", "base_plus": "Base+", "large": "Large"}
            box = SelectBox(labels.get(s.sam2_model, "Large"), t, options=list(labels.values()),
                            on_select=self._on_sam2_model_select)
            return FieldRow("Model size", box, t)
        return None  # cellpose is zero-shot — no model field

    def _engine_settings_accordion(self, s) -> Accordion:
        t = self._t
        title = f"Engine settings · {ENGINE_LABELS.get(s.engine, s.engine)}"
        acc = Accordion(title, t, lead="settings", open_=False)
        if s.engine == "cellseg1":
            from studio.train_controller import available_backbones
            backbones = available_backbones(self._segment.storage_dir / "sam_backbone")
            labels = {"vit_h": "ViT-H", "vit_l": "ViT-L", "vit_b": "ViT-B"}
            options = [lbl for _key, lbl in backbones] or list(labels.values())
            box = SelectBox(labels.get(s.vit_name, s.vit_name), t, options=options,
                            on_select=self._on_backbone_select)
            acc.add(FieldRow("SAM backbone", box, t))
            rank_stepper = Stepper(s.lora_rank, t, step=4, minimum=4, maximum=64)
            rank_stepper.changed.connect(lambda v: self._set_setting("lora_rank", int(v)))
            acc.add(FieldRow("LoRA rank", rank_stepper, t))
            nms_stepper = Stepper(s.box_nms_thresh, t, step=0.01, minimum=0, maximum=1, decimals=2)
            nms_stepper.changed.connect(lambda v: self._set_setting("box_nms_thresh", round(v, 3)))
            acc.add(FieldRow("Box NMS", nms_stepper, t))
        elif s.engine == "cellpose":
            diam_stepper = Stepper(s.cp_diameter, t, step=1, minimum=0, maximum=500)
            diam_stepper.changed.connect(lambda v: self._set_setting("cp_diameter", float(v)))
            acc.add(FieldRow("Diameter (0=auto)", diam_stepper, t))
            flow_stepper = Stepper(s.cp_flow_threshold, t, step=0.05, minimum=0, maximum=3, decimals=2)
            flow_stepper.changed.connect(lambda v: self._set_setting("cp_flow_threshold", round(v, 3)))
            acc.add(FieldRow("Flow threshold", flow_stepper, t))
            prob_stepper = Stepper(s.cp_cellprob_threshold, t, step=0.1, minimum=-6, maximum=6, decimals=1)
            prob_stepper.changed.connect(lambda v: self._set_setting("cp_cellprob_threshold", round(v, 2)))
            acc.add(FieldRow("Cell probability threshold", prob_stepper, t))
        elif s.engine == "sam2":
            mode_ctrl = SegControl(["Independent", "Propagate"], t,
                                   1 if s.sam2_tracking_mode == "propagate" else 0, compact=True)
            mode_ctrl.changed.connect(self._on_sam2_tracking_mode)
            acc.add(FieldRow("Z-stack tracking", mode_ctrl, t))
            stitch_stepper = Stepper(s.stitch_iou, t, step=0.05, minimum=0, maximum=1, decimals=2)
            stitch_stepper.changed.connect(lambda v: self._set_setting("stitch_iou", round(v, 3)))
            acc.add(FieldRow("Stitch IoU", stitch_stepper, t))
        return acc

    def _on_engine_select(self, text: str) -> None:
        key = self._engine_label_to_key.get(text)
        if key is None or self._project is None:
            return
        self._project.settings.engine = key
        self._rebuild_segment_pane()

    def _on_model_select(self, text: str) -> None:
        if self._project is None:
            return
        if text == "Browse…":
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose a LoRA checkpoint", "", "PyTorch (*.pth);;All files (*)", options=_DLG)
            if path:
                self._project.settings.model_name = path
                self._rebuild_segment_pane()
            return
        for m in self._segment.list_lora_models():
            if m.name == text:
                self._project.settings.model_name = str(m.checkpoint)
                break
        self._rebuild_segment_pane()

    def _on_sam2_model_select(self, text: str) -> None:
        inverse = {"Tiny": "tiny", "Small": "small", "Base+": "base_plus", "Large": "large"}
        if self._project is not None:
            self._project.settings.sam2_model = inverse.get(text, "large")

    def _on_backbone_select(self, text: str) -> None:
        inverse = {"ViT-H": "vit_h", "ViT-L": "vit_l", "ViT-B": "vit_b"}
        if self._project is not None and text in inverse:
            self._project.settings.vit_name = inverse[text]

    def _on_sam2_tracking_mode(self, idx: int) -> None:
        if self._project is not None:
            self._project.settings.sam2_tracking_mode = "propagate" if idx == 1 else "independent"

    def _on_quality_preset(self, idx: int) -> None:
        if self._project is None or not (0 <= idx < len(QUALITY_PRESET_NAMES)):
            return
        apply_quality_preset(self._project.settings, QUALITY_PRESET_NAMES[idx])
        self._rebuild_segment_pane()

    def _set_setting(self, attr: str, value) -> None:
        if self._project is not None:
            setattr(self._project.settings, attr, value)

    def _set_setting_custom(self, attr: str, value) -> None:
        """Like _set_setting, but also marks the quality preset "Custom" —
        the three detection-threshold controls are also driven by the Fast/
        Balanced/Accurate preset, so hand-tweaking one means the preset no
        longer exactly matches (ProjectSettings.quality_preset's own
        contract: "Fast | Balanced | Accurate | Custom")."""
        if self._project is None:
            return
        setattr(self._project.settings, attr, value)
        self._project.settings.quality_preset = "Custom"

    def _on_threshold_change(self, attr: str, value: float, badge: Badge) -> None:
        self._set_setting_custom(attr, round(value, 3))
        badge.setText(f"{value:.2f}")

    def _on_pixel_size_edited(self, line_edit: QLineEdit) -> None:
        try:
            value = max(0.0, float(line_edit.text()))
        except ValueError:
            value = 0.0
        line_edit.setText(f"{value:.3f}")
        self._set_setting("pixel_size_um", value)
        if self._last_result is not None:
            self._recompute_results()

    def _toggle_show_predictions(self, on: bool) -> None:
        for lyr in self._layers.by_kind("labels"):
            if lyr.name != "Ground truth":
                lyr.visible = on
        self._layers.notify()

    def _toggle_show_gt(self, on: bool) -> None:
        gt = self._layers.find("Ground truth")
        if gt is not None:
            gt.visible = on
            self._layers.notify()
        elif on:
            self._toast("No ground truth loaded",
                       "Pick a ground-truth mask in Results → Ground truth & evaluation.")

    # ── run bar + predict flow ───────────────────────────────────────────────
    def _runbar(self) -> QWidget:
        t = self._t
        bar = QFrame()
        bar.setStyleSheet(f"background:{t['surface']}; border-top:1px solid {t['border']};")
        v = QVBoxLayout(bar)
        v.setContentsMargins(14, 12, 14, 14)
        v.setSpacing(10)
        self._progress_frame = QFrame()
        self._progress_frame.setFixedHeight(6)
        self._progress_frame.setStyleSheet(
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {t['primary']}, stop:1 {t['signal']}); border-radius:3px;")
        self._progress_frame.setVisible(False)
        v.addWidget(self._progress_frame)
        row = QHBoxLayout()
        row.setSpacing(9)
        self._run_btn_bar = PillButton("Run segmentation", t, "primary", "run")
        self._run_btn_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._run_btn_bar.clicked.connect(self._start_predict)
        row.addWidget(self._run_btn_bar, 1)
        row.addWidget(IconButton("batch", t, 38, "Batch predict", self._start_batch))
        v.addLayout(row)
        return bar

    def _set_running(self, running: bool) -> None:
        self._predicting = running
        self._run_btn_topbar.setEnabled(not running)
        self._run_btn_bar.setEnabled(not running)
        self._run_btn_bar.setText("Segmenting…" if running else "Run segmentation")
        self._progress_frame.setVisible(running)

    def _start_predict(self) -> None:
        if self._project is None:
            self._toast("No project open", "Open or create a project first.")
            return
        if self._current_image_path is None:
            self._toast("No image selected", "Pick an image from the Images panel first.")
            return
        if self._predicting:
            return
        try:
            self._projects.store.save(self._project)
            config_ok = self._segment.build_config(self._project, self._current_image_path)
        except ValueError as e:
            self._toast("Can't run segmentation", str(e))
            return
        del config_ok  # only built to validate synchronously before spawning the thread
        self._run_started_at = time.monotonic()
        self._set_running(True)
        self._segment.run_predict_async(
            self._project, self._current_image_path,
            on_result=self._safe_emit_predict_result, on_log=self._safe_emit_predict_log,
            on_finish=self._safe_emit_predict_finish)

    def _safe_emit_predict_result(self, img, mask, stack) -> None:
        try:
            self._predict_result_signal.emit(img, mask, stack)
        except RuntimeError:
            pass

    def _safe_emit_predict_log(self, msg: str) -> None:
        try:
            self._predict_log_signal.emit(msg)
        except RuntimeError:
            pass

    def _safe_emit_predict_finish(self) -> None:
        try:
            self._predict_finish_signal.emit()
        except RuntimeError:
            pass

    def _on_predict_result(self, img, mask, _stack) -> None:
        self._current_image_array = img
        mask = np.ascontiguousarray(mask).astype(np.int32)
        seg = self._layers.find("Segmentation")
        if seg is not None:
            seg.data = mask
            seg.visible = True
        else:
            seg = LabelsLayer("Segmentation", mask)
            self._layers.add(seg, select=False)
        self._recompute_results()
        if self._project is not None:
            self._segment.record_run(self._project, n_cells=self._last_result.get("n_cells", 0))
            if self._current_image_path is not None:
                try:
                    self._segment.save_result_mask(self._project, self._current_image_path, mask)
                except OSError as e:
                    self._toast("Result not saved to disk", str(e))
        if self._layers.find("Ground truth") is not None:
            self._evaluate_gt()
        self._layers.notify()

    def _recompute_results(self) -> None:
        seg = self._layers.find("Segmentation")
        if seg is None or self._current_image_array is None:
            return
        pixel_size = self._project.settings.pixel_size_um if self._project else 0.0
        self._last_result = self._segment.compute_measurements(
            seg.data, image=self._current_image_array, pixel_size_um=pixel_size)
        # Refresh the edit-sync fingerprint here (not only in _schedule_results_sync)
        # so the recompute this method just did -- and the notify() that predict
        # fires right after -- don't immediately re-trigger the debounced sync.
        self._results_sig = (int(seg.max_label), int((seg.data > 0).sum()))
        # Land the canvas legend on the same exact count the Results panel shows
        # (the live per-tick path only had the cheap max id).
        self._update_legend(int(self._last_result.get("n_cells", 0)))
        self._rebuild_results_pane()
        self._refresh_images_pane()

    def _on_predict_log(self, msg: str) -> None:
        # Shared by predict/batch/benchmark -- previously every line but
        # [ERROR]/[HINT] was thrown away the instant this ran; now the whole
        # stream (the reused PredictController's real operational log) also
        # reaches the Logs console, not just a transient toast.
        emit_prefixed(get_log_bus(), msg, source="studio.segment")
        if msg.startswith("[ERROR]"):
            first_line = msg.splitlines()[0]
            self._toast("Segmentation failed", first_line[len("[ERROR] "):])
        elif msg.startswith("[HINT]"):
            self._toast("Hint", msg[len("[HINT] "):])

    def _on_predict_finished(self) -> None:
        self._set_running(False)
        if self._project is not None:
            self._projects.store.save(self._project)
        elapsed = time.monotonic() - self._run_started_at if self._run_started_at else 0.0
        if self._last_result is not None:
            n = self._last_result.get("n_cells", 0)
            self._on_canvas_status(f"Segmented in {elapsed:.1f} s")
            self._toast("Segmentation complete", f"{n} cells · {elapsed:.1f} s")
        self._refresh_images_pane()

    # ── Assistant integration ───────────────────────────────────────────────
    # The narrow read/write surface the Assistant drawer uses to see the
    # active session and act on it — mirrors the classic app's
    # PredictWidget.last_context()/current_params()/apply_params()/rerun()
    # contract, so the two apps' Assistants are built against the same shape.
    def assistant_context(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
        """``(image, mask, params)`` for the Assistant's diagnose/chat.

        Degrades to ``(None, None, {})`` with no project/image open —
        ``advisor.diagnose`` handles that combination without crashing (it
        just asks for a prediction first), so this never needs its own
        empty-state branching.
        """
        # _select_image() always creates a "Segmentation" layer (zero-filled
        # until something real is predicted), so its mere existence doesn't
        # mean a result exists — gate on _last_result, the same signal the
        # Results pane already uses (_export_csv's "Nothing to export", "Show
        # measurements" without a run) to tell "genuinely segmented" apart
        # from "just an empty placeholder", so an unpredicted image reads to
        # the advisor as "no mask yet" rather than "0 cells found".
        seg = self._layers.find("Segmentation")
        mask = seg.data if (seg is not None and self._last_result is not None) else None
        params: dict = {}
        if self._project is not None:
            image_path = self._current_image_path or (
                self._project.image_paths[0] if self._project.image_paths else None)
            if image_path is not None:
                params = self._segment.build_params(self._project, image_path)
        return self._current_image_array, mask, params

    def apply_assistant_changes(self, changes: dict) -> Optional[list[str]]:
        """Apply a ``{param: value}`` dict (an advisor finding's, or a chat
        model's parsed ``SUGGEST:`` lines) to the active project's settings —
        the same convention a manual threshold edit uses
        (``_set_setting_custom``): marks ``quality_preset`` "Custom" and
        persists immediately, then rebuilds the Segment pane so it reflects
        the new values without needing to reopen the project.

        Returns the keys actually applied, or ``None`` when there's no
        active project — lets the caller (the Assistant) tell "nothing to
        apply" apart from "nowhere to apply it".
        """
        if self._project is None:
            return None
        applied = [k for k in changes if k in _SETTINGS_FIELD_NAMES]
        for k in applied:
            setattr(self._project.settings, k, changes[k])
        if applied:
            self._project.settings.quality_preset = "Custom"
            self._projects.store.save(self._project)
            self._rebuild_segment_pane()
        return applied

    def rerun_predict(self) -> None:
        """The Assistant's "Apply & re-run" — ``_start_predict`` already
        guards every precondition (no project/image selected, already
        running), so this is a thin, honestly-named alias rather than a
        second copy of that logic."""
        self._start_predict()

    # ── Command palette integration ─────────────────────────────────────────
    # A narrow set of real actions the ⌘K palette can trigger from outside —
    # same principle as the Assistant integration above (own UI, act through
    # a small public surface, never poke at a private attribute from another
    # module). Each one is a thin alias over an already-self-guarding
    # private method (toasts on a bad precondition, no-ops if already
    # running) rather than a second copy of that logic.
    def run_batch(self) -> None:
        self._start_batch()

    def run_benchmark(self) -> None:
        self._start_benchmark()

    def save_masks(self) -> None:
        self._save_masks()

    def export_measurements(self) -> None:
        self._export_csv()

    def switch_engine(self, key: str) -> None:
        """Set the active engine by registry key — the palette already has
        the key straight from ``segment_controller.list_available_engines()``,
        so this is ``_on_engine_select``'s same two-line effect without the
        label-string round-trip that method needs for its own combo box."""
        if self._project is None:
            return
        self._project.settings.engine = key
        self._rebuild_segment_pane()

    def apply_preset(self, name: str) -> None:
        """Apply a quality preset by name (Fast/Balanced/Accurate) — the
        palette's equivalent of picking it from the SegControl."""
        if self._project is None or name not in QUALITY_PRESET_NAMES:
            return
        apply_quality_preset(self._project.settings, name)
        self._rebuild_segment_pane()

    # ── Results pane ─────────────────────────────────────────────────────────
    def _rebuild_results_pane(self) -> None:
        layout = self._results_container_layout
        _clear_layout(layout)
        t = self._t
        if self._project is None:
            empty = label("Open or create a project to see results.", 12, t["text_muted"])
            empty.setWordWrap(True)
            layout.addWidget(empty)
            return
        if self._last_result is None:
            empty = label("Run segmentation to see results here.", 12, t["text_muted"])
            empty.setWordWrap(True)
            layout.addWidget(empty)
            layout.addWidget(self._batch_accordion())
            layout.addWidget(self._benchmark_accordion())
            layout.addStretch(1)
            return

        r = self._last_result
        hero = QHBoxLayout()
        hero.setSpacing(14)
        num = QLabel(str(r["n_cells"]))
        num.setStyleSheet(
            f"color:{t['success']}; font-family:{theme.MONO}; font-size:40px; font-weight:600; letter-spacing:-1.5px;")
        hero.addWidget(num)
        hero.addWidget(label("cells\ndetected", 13, t["text_muted"], 600))
        hero.addStretch(1)
        layout.addLayout(hero)

        summary = r.get("summary", {})
        diam = summary.get("diameter", {})
        area = summary.get("area", {})
        coverage = 0.0
        seg = self._layers.find("Segmentation")
        if seg is not None and seg.data.size:
            coverage = float((seg.data > 0).sum()) / seg.data.size * 100.0
        tiles = QHBoxLayout()
        tiles.setSpacing(6)
        diam_unit = next((u for k, _l, u in r["columns"] if k == "diameter"), "px")
        area_unit = next((u for k, _l, u in r["columns"] if k == "area"), "px²")
        tiles.addWidget(StatTile(f"{diam.get('median', 0):.1f}", diam_unit, "MEDIAN Ø", t))
        tiles.addWidget(StatTile(f"{area.get('mean', 0):.0f}", area_unit, "MEAN AREA", t))
        tiles.addWidget(StatTile(f"{coverage:.1f}", "%", "COVERAGE", t))
        layout.addLayout(tiles)

        explore = PillButton("Explore cell population", t, "primary", "chart")
        explore.clicked.connect(self._show_measurements)
        layout.addWidget(explore)

        layout.addWidget(GroupLabel("Pixel calibration", t))
        pixel_size = self._project.settings.pixel_size_um
        cal_edit = QLineEdit(f"{pixel_size:.3f}")
        cal_edit.editingFinished.connect(lambda le=cal_edit: self._on_pixel_size_edited(le))
        layout.addWidget(cal_edit)
        hint = label("Enter your microscope's µm-per-pixel to get real-world units. 0 = pixels.",
                    11, t["text_muted"])
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btns = QGridLayout()
        btns.setSpacing(8)
        save_btn = PillButton("Save masks", t, "ghost", "save", small=True)
        save_btn.clicked.connect(self._save_masks)
        export_btn = PillButton("Export CSV", t, "ghost", "csv", small=True)
        export_btn.clicked.connect(self._export_csv)
        refine_btn = PillButton("Refine…", t, "ghost", "spark", small=True)
        refine_btn.clicked.connect(self._refine_coming_soon)
        measure_btn = PillButton("Analytics", t, "ghost", "measure", small=True)
        measure_btn.clicked.connect(self._show_measurements)
        for i, b in enumerate([save_btn, export_btn, refine_btn, measure_btn]):
            btns.addWidget(b, i // 2, i % 2)
        layout.addLayout(btns)

        layout.addWidget(hline(t))
        layout.addWidget(GroupLabel("Display · colour cells by", t))
        colorby_box = SelectBox(COLOR_BY_OPTIONS[0], t, options=COLOR_BY_OPTIONS, on_select=self._on_color_by)
        layout.addWidget(colorby_box)
        heat = QFrame()
        hv = QVBoxLayout(heat)
        hv.setContentsMargins(0, 4, 0, 0)
        hv.setSpacing(3)
        grad = QFrame()
        grad.setFixedHeight(10)
        grad.setStyleSheet(
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #440154, stop:0.25 #3b528b, stop:0.5 #21918c, stop:0.75 #5ec962, stop:1 #fde725);"
            "border-radius:5px;")
        hv.addWidget(grad)
        layout.addWidget(heat)

        layout.addWidget(self._gt_accordion())
        layout.addWidget(self._batch_accordion())
        layout.addWidget(self._benchmark_accordion())
        layout.addStretch(1)

    def _gt_accordion(self) -> Accordion:
        t = self._t
        gt_layer = self._layers.find("Ground truth")
        acc = Accordion("Ground truth & evaluation", t, lead="check", open_=bool(self._gt_metrics))
        gt_text = Path(self._segment.find_gt_for_image(self._current_image_path) or "").name \
            if self._current_image_path and gt_layer is None else (gt_layer.name if gt_layer else "")
        box = SelectBox(gt_text or "Choose ground-truth mask…", t, on_click=self._pick_gt_mask)
        acc.add(FieldRow("Ground-truth mask", box, t))
        gt_toggle = Toggle(t, gt_layer.visible if gt_layer else False)
        gt_toggle.toggled.connect(self._toggle_show_gt)
        acc.add(FieldRow("Show ground truth", gt_toggle, t))
        if self._gt_metrics:
            for name, value in self._format_gt_metrics(self._gt_metrics):
                mv = QLabel(value)
                mv.setStyleSheet(f"color:{t['success']}; font-family:{theme.MONO}; font-size:12.5px; font-weight:600; background:transparent;")
                acc.add(FieldRow(name, mv, t))
        else:
            note = label("Pick a ground-truth mask to evaluate against.", 11, t["text_muted"])
            note.setWordWrap(True)
            acc.add(note)
        return acc

    @staticmethod
    def _format_gt_metrics(m: dict) -> list[tuple[str, str]]:
        tp, fp, fn = m.get("tp", 0), m.get("fp", 0), m.get("fn", 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return [("F1 score", f"{m.get('f1', 0):.2f}"), ("Precision", f"{precision:.2f}"),
               ("Recall", f"{recall:.2f}"), ("AP @ 0.50", f"{m.get('ap@0.5', 0):.2f}")]

    def _pick_gt_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a ground-truth mask", "",
            "Masks (*.png *.tif *.tiff *.npy);;All files (*)", options=_DLG)
        if path:
            self._load_gt(path)

    def _load_gt(self, path: str) -> None:
        if self._current_image_array is None:
            self._toast("No image loaded", "Load an image before adding ground truth.")
            return
        shape = self._current_image_array.shape[:2]
        try:
            gt = self._segment.load_gt_mask(path, shape)
        except Exception as e:
            self._toast("Can't read ground truth", str(e))
            return
        existing = self._layers.find("Ground truth")
        if existing is not None:
            self._layers.remove(self._layers.index_of(existing))
        gt_layer = LabelsLayer("Ground truth", gt)
        gt_layer.opacity = 0.9
        gt_layer.set_uniform_color(_GT_COLOR)
        self._layers.add(gt_layer, select=False)
        if self._last_result is not None:
            self._evaluate_gt()
        self._rebuild_results_pane()

    def _evaluate_gt(self) -> None:
        gt_layer = self._layers.find("Ground truth")
        seg_layer = self._layers.find("Segmentation")
        if gt_layer is None or seg_layer is None:
            return
        self._gt_metrics = self._segment.evaluate_masks(gt_layer.data, seg_layer.data)
        if self._project is not None:
            self._segment.record_run(self._project, f1=self._gt_metrics.get("f1"))
            self._projects.store.save(self._project)
        self._rebuild_results_pane()

    def _on_color_by(self, text: str) -> None:
        seg = self._layers.find("Segmentation")
        if seg is None or self._last_result is None:
            return
        key = _COLOR_BY_KEYS.get(text)
        if key is None:
            seg.clear_color_overrides()
        else:
            overrides = self._segment.color_overrides_for(self._last_result, key)
            if not overrides:
                self._toast("Can't colour by that", f"No {key.replace('_', ' ')} data available.")
                return
            seg.set_color_overrides(overrides)
        self._layers.notify()

    def _save_masks(self) -> None:
        seg = self._layers.find("Segmentation")
        if seg is None or seg.max_label == 0 or self._project is None or self._current_image_path is None:
            self._toast("Nothing to save", "Run segmentation first.")
            return
        default = str(self._segment.project_run_dir(self._project) /
                      f"{Path(self._current_image_path).stem}_mask.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save masks", default, "PNG (*.png)", options=_DLG)
        if not path:
            return
        self._segment.save_mask(seg.data, path)
        self._toast("Masks saved", path)

    def _export_csv(self) -> None:
        if self._last_result is None or self._project is None or self._current_image_path is None:
            self._toast("Nothing to export", "Run segmentation first.")
            return
        default = str(self._segment.project_run_dir(self._project) /
                      f"{Path(self._current_image_path).stem}_measurements.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export measurements", default, "CSV (*.csv)", options=_DLG)
        if not path:
            return
        self._segment.export_measurements_csv(self._last_result, path)
        self._toast("Measurements exported", path)

    def _refine_coming_soon(self) -> None:
        self._toast("Refine — coming soon", "Interactive point-prompt refinement isn't wired up yet.")

    def _show_measurements(self) -> None:
        """Open the Cell Population Analytics explorer over the current
        result — the per-cell morphometry the engine already computed, shown
        as distribution histograms + summary stats (see
        ``studio/cell_analytics.py``)."""
        if self._last_result is None:
            self._toast("No measurements yet", "Run segmentation first.")
            return
        from studio.cell_analytics import CellAnalyticsDialog
        name = Path(self._current_image_path).name if self._current_image_path else ""
        dlg = CellAnalyticsDialog(self, self._t, self._last_result,
                                  image_name=name, on_export=self._export_csv)
        dlg.open()

    # ── Batch prediction ─────────────────────────────────────────────────────
    def _batch_accordion(self) -> Accordion:
        t = self._t
        acc = Accordion("Batch prediction", t, lead="batch", open_=False)
        n = len(self._project.image_paths) if self._project else 0
        note = label(
            f"Run the current engine & settings across all {n} images in this project, "
            "then aggregate cohort statistics.", 11.5, t["text_muted"])
        note.setWordWrap(True)
        acc.add(note)
        btn = PillButton("Running batch…" if self._batching else f"Run batch ({n} images)",
                        t, "ghost", "run", small=True)
        btn.setEnabled(n > 0 and not self._batching and not self._predicting)
        btn.clicked.connect(self._start_batch)
        acc.add(btn)
        return acc

    def _start_batch(self) -> None:
        if self._project is None:
            self._toast("No project open", "Open or create a project first.")
            return
        if self._batching:
            return
        self._projects.store.save(self._project)
        try:
            self._batching = True
            self._segment.run_batch_async(
                self._project, on_log=self._safe_emit_batch_log,
                on_progress=self._safe_emit_batch_progress,
                on_cohort_ready=self._safe_emit_batch_cohort,
                on_finish=self._safe_emit_batch_finish)
        except ValueError as e:
            self._batching = False
            self._toast("Can't start batch", str(e))
            return
        self._rebuild_results_pane()

    def _safe_emit_batch_log(self, msg: str) -> None:
        try:
            self._batch_log_signal.emit(msg)
        except RuntimeError:
            pass

    def _safe_emit_batch_progress(self, done: int, total: int) -> None:
        try:
            self._batch_progress_signal.emit(done, total)
        except RuntimeError:
            pass

    def _safe_emit_batch_cohort(self, records, out_dir) -> None:
        try:
            self._batch_cohort_signal.emit(records, out_dir)
        except RuntimeError:
            pass

    def _safe_emit_batch_finish(self) -> None:
        try:
            self._batch_finish_signal.emit()
        except RuntimeError:
            pass

    def _on_batch_progress(self, done: int, total: int) -> None:
        self._on_canvas_status(f"Batch {done}/{total}")

    def _on_batch_cohort_ready(self, records, out_dir) -> None:
        if self._project is not None:
            self._projects.store.save(self._project)
        pop = self._segment.population_stats(records)
        self._toast("Batch complete",
                    f"{pop.get('total_cells', 0)} cells across {pop.get('n_images', 0)} images · {out_dir}")

    def _on_batch_finished(self) -> None:
        self._batching = False
        self._refresh_images_pane()
        self._rebuild_results_pane()

    # ── Benchmark engines vs GT ──────────────────────────────────────────────
    def _benchmark_accordion(self) -> Accordion:
        t = self._t
        acc = Accordion("Benchmark engines vs GT", t, lead="chart", open_=False)
        if self._bench_rows:
            for name, val in self._bench_rows:
                mv = QLabel(val)
                mv.setStyleSheet(f"color:{t['text_subtle']}; font-family:{theme.MONO}; font-size:12.5px; font-weight:600; background:transparent;")
                acc.add(FieldRow(name, mv, t))
        else:
            note = label("Runs every available engine against this project's ground-truth masks.",
                        11.5, t["text_muted"])
            note.setWordWrap(True)
            acc.add(note)
        btn = PillButton("Running…" if self._benching else "Run benchmark", t, "ghost", "chart", small=True)
        btn.setEnabled(not self._benching)
        btn.clicked.connect(self._start_benchmark)
        acc.add(btn)
        return acc

    def _start_benchmark(self) -> None:
        if self._project is None or self._benching:
            return
        try:
            self._benching = True
            self._bench_rows = []
            self._segment.run_benchmark_async(
                self._project, on_row=self._safe_emit_bench_row,
                on_log=self._safe_emit_bench_log, on_done=self._safe_emit_bench_done)
        except ValueError as e:
            self._benching = False
            self._toast("Can't run benchmark", str(e))
            return
        self._rebuild_results_pane()

    def _safe_emit_bench_row(self, text: str) -> None:
        try:
            self._bench_row_signal.emit(text)
        except RuntimeError:
            pass

    def _safe_emit_bench_log(self, msg: str) -> None:
        try:
            self._bench_log_signal.emit(msg)
        except RuntimeError:
            pass

    def _safe_emit_bench_done(self, cols, rows) -> None:
        try:
            self._bench_done_signal.emit(cols, rows)
        except RuntimeError:
            pass

    def _on_bench_row(self, text: str) -> None:
        self._on_canvas_status(f"Benchmark {text}")

    def _on_bench_done(self, cols, rows) -> None:
        self._benching = False
        # cols = ["engine", "images", "F1@0.5", "AP@0.5", "AP@0.75", "AP@0.9", "mAP"]
        self._bench_rows = [(r[0], f"{r[2]:.2f}") for r in rows]
        self._rebuild_results_pane()
        self._toast("Benchmark complete", f"{len(rows)} engine(s) scored")
