"""Velum — the screens.

Native reproductions of the mockup, one screen per class, driven by the
``components``/``paint`` kits. Home and Projects are wired to the real
``ProjectController`` (live projects, search/filter/favourites); the other
screens still render ``demo.py`` static content pending their own tab in
``docs/velum/BACKLOG.md``. The Workspace is the signature screen (adapted-napari
layers · canvas · inspector). See ``docs/velum/`` for the per-tab wiring plan.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QSize, QUrl
from PyQt6.QtGui import QDesktopServices, QAction
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QScrollArea, QLineEdit, QSizePolicy, QStackedWidget, QToolButton, QMenu,
)

from studio import hardware
from studio import icons
from studio import theme
from studio import project_controller
from studio.project import ENGINE_LABELS, ENGINES
from studio.motion import install_hover_lift, fade_in
from studio.components import (
    Chip, Badge, EngineChip, PillButton, IconButton, SelectBox, Toggle, Slider, Stepper,
    SegControl, StatTile, FieldRow, GroupLabel, Accordion, SmoothScrollArea, WavingEmoji,
    hline, soft_shadow, label,
)
from studio.project_dialogs import ProjectSettingsDialog

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _github_url() -> Optional[str]:
    """The real origin remote, converted to an https:// web URL (or None)."""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True,
            text=True, timeout=2, cwd=str(_REPO_ROOT))
        url = out.stdout.strip()
    except Exception:
        return None
    if not url:
        return None
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:"):]
    if url.endswith(".git"):
        url = url[:-4]
    return url


def _open_github() -> None:
    url = _github_url()
    if url:
        QDesktopServices.openUrl(QUrl(url))


# ── shared helpers ───────────────────────────────────────────────────────────
def scroll(inner: QWidget) -> QScrollArea:
    sa = SmoothScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setWidget(inner)
    return sa


def page_header(title: str, subtitle: str, t: dict, action: Optional[QWidget] = None,
                 title_extra: Optional[QWidget] = None) -> QWidget:
    """``title_extra`` sits beside the title text in the same row (e.g.
    Home's waving-hand greeting) -- optional and unused by every other
    caller, so their layout is byte-for-byte unchanged."""
    head = QWidget()
    row = QHBoxLayout(head)
    row.setContentsMargins(34, 30, 34, 18)
    col = QVBoxLayout()
    col.setSpacing(5)
    if title_extra is not None:
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(4)
        title_row.addWidget(label(title, 26, t["text"], 600, -0.6))
        title_row.addWidget(title_extra, alignment=Qt.AlignmentFlag.AlignVCenter)
        title_row.addStretch(1)
        col.addLayout(title_row)
    else:
        col.addWidget(label(title, 26, t["text"], 600, -0.6))
    if subtitle:
        col.addWidget(label(subtitle, 14, t["text_muted"]))
    row.addLayout(col)
    row.addStretch(1)
    if action is not None:
        row.addWidget(action, alignment=Qt.AlignmentFlag.AlignBottom)
    return head


# ── Home ─────────────────────────────────────────────────────────────────────
class HomeScreen(QWidget):
    def __init__(self, t: dict, controller: "project_controller.ProjectController",
                 on_navigate: Callable[[str], None], on_open: Callable[[str], None],
                 on_new_project: Callable[[], None]):
        super().__init__()
        self._t = t
        self._controller = controller
        self._nav = on_navigate
        self._open = on_open
        self._new_project = on_new_project

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._new_project_cta = PillButton("New Project", t, "primary", "plus")
        self._new_project_cta.clicked.connect(lambda: on_new_project())
        self._wave = WavingEmoji(t)
        outer.addWidget(page_header("Welcome back",
                                    "Segment, measure and compare cells across your projects.",
                                    t, self._new_project_cta, title_extra=self._wave))

        body = QWidget()
        grid = QHBoxLayout(body)
        grid.setContentsMargins(34, 4, 34, 40)
        grid.setSpacing(24)

        left = QVBoxLayout()
        left.setSpacing(24)
        self._quick_grid = self._quick()
        left.addLayout(self._quick_grid)
        self._recent_widget = self._recent_section()
        left.addWidget(self._recent_widget)
        left.addStretch(1)
        self._left = left
        grid.addLayout(left, 1)
        grid.addLayout(self._aside(), 0)

        outer.addWidget(scroll(body))
        # Snapshot what the recent list currently shows, so the very first
        # navigate("home") (same content) is a calm no-op instead of an
        # entrance animation -- refresh() only rebuilds + fades when this
        # signature actually changes.
        self._recent_sig = self._current_recent_sig()

    def _current_recent_sig(self) -> tuple:
        """A cheap fingerprint of the recent-projects list: enough to tell a
        genuine content change (a new/renamed project, new images or cells, a
        re-order) from an identical revisit. Built from raw store fields, not
        the rendered ``when`` string, so it stays stable across visits unless
        the data really moved."""
        return tuple(
            (p.id, p.updated_at, p.stats.n_images, p.stats.n_cells, p.engine, p.name)
            for p in self._controller.recent(limit=4)
        )

    def refresh(self) -> None:
        """Rebuild the recent-projects list from the store's current state,
        called by app.navigate() on every visit to Home (including the very
        first one).

        HomeScreen is built once and kept alive across navigation (the stack
        just swaps the visible page), so anything that changes the store
        elsewhere -- most importantly creating a project -- needs this
        called before Home is shown again, or it'd still show what the store
        looked like at construction time.

        Deliberately does *not* fade the whole screen in -- app.navigate()
        used to do that unconditionally on every screen switch, but Home's
        quick-cards, recent rows and aside cards each carry their own
        ``install_hover_lift``/``soft_shadow`` QGraphicsDropShadowEffect
        (up to ~10 of them at once), and animating a QGraphicsOpacityEffect
        on their shared ancestor forces Qt to re-rasterise every one of
        those nested effects on every frame of the fade -- the same
        composited-effects-are-expensive mechanism already root-caused for
        the Projects grid's scroll stutter (docs/velum/BACKLOG.md's "Projects
        tab v2" entry), just triggered by a repeated opacity animation
        instead of scrolling. Worse, it re-played on *every single visit* to
        an already-built, mostly-unchanged screen, not just the first time —
        reported directly as "recent projects... каждый раз с ужасной
        анимацией" (shows with a terrible animation every time).

        Two things make it quiet and smooth now:

        1. **It only animates on a real change.** ``_recent_sig`` fingerprints
           the list; an identical revisit (the common case -- you tab away
           and back without creating anything) returns early, so the recent
           section is neither rebuilt nor re-faded. That kills the
           "every single visit" replay outright.

        2. **When it does change, the fade carries no shadows.** The rebuilt
           rows are created *without* their hover-lift ``QGraphicsDropShadow``
           effects (``with_hover=False``); the fade therefore composites plain
           rows, not ~4 nested shadow effects per frame, and the shadows are
           installed once, after the fade settles (``_install_recent_hover``
           via ``fade_in``'s ``on_finished`` hook). Same visible result, none
           of the per-frame re-rasterisation that made it stutter.

        The waving-hand greeting still plays once per visit (cheap, no nested
        effects) -- an unconditional, deliberate cue independent of whether
        the list changed.
        """
        self._wave.play()
        sig = self._current_recent_sig()
        if sig == self._recent_sig:
            return  # identical list -- don't rebuild or re-animate it
        self._recent_sig = sig

        idx = self._left.indexOf(self._recent_widget)
        old = self._recent_widget
        new_widget = self._recent_section(with_hover=False)
        self._recent_widget = new_widget
        self._left.insertWidget(idx, new_widget)
        self._left.removeWidget(old)
        old.setParent(None)
        old.deleteLater()
        fade_in(new_widget, 220,
                on_finished=lambda w=new_widget: self._install_recent_hover(w))

    def _install_recent_hover(self, widget: QWidget) -> None:
        """Give the freshly-faded recent rows their hover-lift shadows, once
        the entrance fade is done -- deferred so the fade never has to
        re-rasterise them (see ``refresh``). ``install_hover_lift`` is itself
        guarded, and ``getattr`` tolerates the widget having been torn down."""
        for row in getattr(widget, "_rows", ()):  # RRow frames collected below
            install_hover_lift(row)

    def _quick(self) -> QGridLayout:
        t = self._t
        g = QGridLayout()
        g.setSpacing(14)
        cards = [
            ("folder", "New Project", "Name it, import images, pick an engine.", "primary", self._new_project),
            ("image", "Import Images", "TIFF · OME-TIFF · ND2 · CZI · PNG. Drag & drop.", "signal", self._new_project),
            ("models", "Train a Model", "One-shot LoRA from a single annotated image.", "warning", lambda: self._nav("train")),
            ("chart", "Open Sample", "Nuclei, tissue & mitosis datasets to explore.", "success", self._open_sample),
        ]
        for i, c in enumerate(cards):
            g.addWidget(self._quick_card(*c), i // 2, i % 2)
        return g

    def _open_sample(self) -> None:
        projects = self._controller.list_projects()
        if projects:
            self._open(projects[0].id)
        else:
            self._new_project()

    def _quick_card(self, icon_name, title, sub, kind, on_click: Callable[[], None]) -> QFrame:
        t = self._t
        card = QFrame()
        card.setObjectName("QCard")
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            f"#QCard{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;}}"
            f"#QCard:hover{{border-color:{t['border_strong']};}}")
        install_hover_lift(card, base=(14, 22, 3), hover=(22, 34, 6))
        row = QHBoxLayout(card)
        row.setContentsMargins(18, 16, 18, 16)
        row.setSpacing(14)
        colm = {"primary": t["primary"], "signal": t["signal"], "warning": t["warning"], "success": t["success"]}
        weakm = {"primary": t["primary_weak"], "signal": t["signal_weak"], "warning": t["warning_weak"], "success": t["success_weak"]}
        badge = QLabel()
        badge.setFixedSize(38, 38)
        badge.setPixmap(icons.pixmap(icon_name, colm[kind], 19))
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(f"background:{weakm[kind]}; border-radius:10px;")
        col = QVBoxLayout()
        col.setSpacing(3)
        col.addWidget(label(title, 14.5, t["text"], 600))
        s = label(sub, 12.5, t["text_muted"])
        s.setWordWrap(True)
        col.addWidget(s)
        row.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)
        row.addLayout(col, 1)
        card.mouseReleaseEvent = lambda e, cb=on_click: cb()
        return card

    def _recent_section(self, with_hover: bool = True) -> QWidget:
        """``with_hover=False`` builds the rows without their hover-lift
        shadow effects -- used by ``refresh`` so an entrance fade doesn't have
        to composite a per-row ``QGraphicsDropShadowEffect`` on every frame;
        the shadows are installed afterwards via ``_install_recent_hover``.
        The built rows are collected on ``widget._rows`` for exactly that."""
        t = self._t
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        head = QHBoxLayout()
        head.addWidget(label("Recent projects", 15, t["text"], 600))
        head.addStretch(1)
        link = QLabel(f"<a href='#' style='color:{t['primary']};text-decoration:none;'>View all →</a>")
        link.setStyleSheet("font-size:12.5px; font-weight:600;")
        link.linkActivated.connect(lambda: self._nav("projects"))
        head.addWidget(link)
        v.addLayout(head)
        recent = [project_controller.to_card(p) for p in self._controller.recent(limit=4)]
        if not recent:
            empty = label("No projects yet — create one to get started.", 12.5, t["text_muted"])
            v.addWidget(empty)
        rows = []
        for card in recent:
            row = self._recent_row(card, with_hover=with_hover)
            rows.append(row)
            v.addWidget(row)
        w._rows = rows
        return w

    def _recent_row(self, p: "project_controller.ProjectCard", with_hover: bool = True) -> QFrame:
        t = self._t
        row = QFrame()
        row.setObjectName("RRow")
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet(
            f"#RRow{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:10px;}}"
            f"#RRow:hover{{border-color:{t['border_strong']};}}")
        if with_hover:
            install_hover_lift(row)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(14)
        meta = QVBoxLayout()
        meta.setSpacing(2)
        meta.addWidget(label(p.name, 13.5, t["text"], 600))
        meta.addWidget(label(f"{p.engine_label} · {p.n_images} images · {p.n_cells} cells", 11.5, t["text_muted"]))
        lay.addLayout(meta, 1)
        when = QLabel(p.when)
        when.setStyleSheet(f"color:{t['text_muted']}; font-size:11.5px; font-family:{theme.MONO};")
        lay.addWidget(when)
        row.mouseReleaseEvent = lambda e, pid=p.id: self._open(pid)
        return row

    def _aside(self) -> QVBoxLayout:
        t = self._t
        col = QVBoxLayout()
        col.setSpacing(16)
        col.setContentsMargins(0, 0, 0, 0)

        tip = QFrame()
        tip.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified -- see components.EngineChip's comment. This is the
        # exact instance docs/velum/CHANGELOG.md's 2026-07-08 entry already
        # found and screenshotted ("the identical double-box on the Tip
        # card's text") but deliberately left unfixed as out of scope for
        # that change -- fixed now.
        tip.setObjectName("HomeTip")
        tip.setStyleSheet(
            f"QFrame#HomeTip{{background:{t['primary_weak']}; border:1px solid {t['primary_line']}; border-radius:14px;}}")
        tl = QVBoxLayout(tip)
        tl.setContentsMargins(16, 16, 16, 16)
        tl.setSpacing(6)
        tl.addWidget(label("✦ Tip — press ⌘K anywhere", 13, t["primary"], 600))
        tp = label("Run segmentation, switch engine, apply a preset or export — the "
                   "command palette reaches every action without leaving the image.", 12.5, t["text_subtle"])
        tp.setWordWrap(True)
        tl.addWidget(tp)
        col.addWidget(tip)

        res = self._card("Resources")
        for name, icon_name, ext, on_click in [
            ("Documentation", "guide", False, lambda: self._nav("guide")),
            ("Getting started guide", "guide", False, lambda: self._nav("guide:getting-started")),
            ("Ask the Assistant", "assistant", False, lambda: self._nav("assistant")),
            ("GitHub", "settings", True, _open_github)]:
            res.layout().addWidget(self._res_link(name, icon_name, ext, on_click))
        col.addWidget(res)

        dev = self._card("This device")
        compute = hardware.detect().label
        for name, val in [("Compute", compute), ("SAM backbone", "ViT-H · cached"),
                          ("Storage", "data_store · 3.1 GB")]:
            dev.layout().addWidget(FieldRow(name, Badge(val, t), t))
        col.addWidget(dev)
        col.addStretch(1)
        return col

    def _card(self, title: str) -> QFrame:
        t = self._t
        c = QFrame()
        c.setFixedWidth(300)
        c.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified -- see components.EngineChip's comment. This is the
        # exact instance docs/velum/CHANGELOG.md's 2026-07-08 entry already
        # found and screenshotted ("HomeScreen._card()... carry this same
        # *latent, currently invisible* bug") but deliberately left unfixed
        # as out of scope for that change -- fixed now. Shared objectName
        # across both callers ("Resources", "This device").
        c.setObjectName("HomeCard")
        c.setStyleSheet(f"QFrame#HomeCard{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;}}")
        soft_shadow(c, 14, 20, 3)
        v = QVBoxLayout(c)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)
        v.addWidget(label(title, 13.5, t["text"], 600))
        return c

    def _res_link(self, name: str, icon_name: str, ext: bool, on_click: Callable[[], None]) -> QFrame:
        t = self._t
        row = QFrame()
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setStyleSheet("QFrame:hover{background:%s; border-radius:8px;}" % t["surface2"])
        lay = QHBoxLayout(row)
        lay.setContentsMargins(6, 8, 6, 8)
        lay.setSpacing(11)
        ic = QLabel()
        ic.setPixmap(icons.pixmap(icon_name, t["text_muted"], 16))
        lay.addWidget(ic)
        lay.addWidget(label(name, 13, t["text_subtle"], 500))
        lay.addStretch(1)
        if ext:
            e = QLabel()
            e.setPixmap(icons.pixmap("settings", t["text_muted"], 12))
            lay.addWidget(e)
        row.mouseReleaseEvent = lambda e, cb=on_click: cb()
        return row


# ── Projects ─────────────────────────────────────────────────────────────────
_SCOPES = ("all", "favorites", "shared")

# The engine-chip dot colour on a project card's header — each engine gets
# its own hue (iris / teal / fig) distinct from ENGINE_KIND's fg/bg "kind"
# (primary/signal), which only carries two families. Reuses theme.VIZ, the
# app's one stable categorical palette, rather than inventing new tokens.
_ENGINE_DOT = {"cellseg1": theme.VIZ[0], "cellpose": theme.VIZ[1], "sam2": theme.VIZ[5]}


class ProjectsScreen(QWidget):
    def __init__(self, t: dict, controller: "project_controller.ProjectController",
                 on_navigate: Callable[[str], None], on_open: Callable[[str], None],
                 on_new_project: Callable[[], None],
                 on_toast: Optional[Callable[..., None]] = None):
        super().__init__()
        self._t = t
        self._controller = controller
        self._nav = on_navigate
        self._open = on_open
        self._new_project = on_new_project
        self._toast = on_toast
        self._query = ""
        self._scope = "all"
        self._engines: set[str] = set()
        self._sort = "modified"
        self._view = "grid"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._outer = outer
        self._header_widget = self._build_header()
        outer.addWidget(self._header_widget)
        outer.addWidget(self._toolbar())

        self._empty_label = label("", 12.5, t["text_muted"])
        self._empty_label.setWordWrap(True)
        self._empty_label.setContentsMargins(34, 0, 34, 16)
        self._empty_label.setVisible(False)
        outer.addWidget(self._empty_label)

        body = QWidget()
        host = QVBoxLayout(body)
        host.setContentsMargins(34, 0, 34, 40)
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(16)
        for col in range(3):
            self._grid.setColumnStretch(col, 1)
        self._list_host = QWidget()
        self._list = QVBoxLayout(self._list_host)
        self._list.setContentsMargins(0, 0, 0, 0)
        self._list.setSpacing(10)
        host.addWidget(self._grid_host)
        host.addWidget(self._list_host)
        host.addStretch(1)
        outer.addWidget(scroll(body))

        self._populate()

    def _build_header(self) -> QWidget:
        t = self._t
        cta = PillButton("New Project", t, "primary", "plus")
        cta.clicked.connect(lambda: self._new_project())
        n_projects, n_images, n_engines = self._controller.summary()
        subtitle = (f"{n_projects} project{'s' if n_projects != 1 else ''} · "
                    f"{n_images} images · {n_engines} engine{'s' if n_engines != 1 else ''}")
        return page_header("Projects", subtitle, t, cta)

    def refresh(self) -> None:
        """Recompute the header counts and repopulate the grid/list.

        ProjectsScreen is built once and kept alive across navigation (see
        HomeScreen.refresh for why), so anything that changes the store
        elsewhere needs this called before Projects is shown again. The
        current search query / scope tab / engine filter / grid-or-list view
        are all preserved (the toolbar itself is never rebuilt).
        """
        idx = self._outer.indexOf(self._header_widget)
        old = self._header_widget
        self._header_widget = self._build_header()
        self._outer.insertWidget(idx, self._header_widget)
        self._outer.removeWidget(old)
        old.setParent(None)
        old.deleteLater()
        self._populate()

    # Every toolbar control except the page header's primary "New Project"
    # CTA sits at this one height — the mockup's own CSS gets All/Favorites/
    # Shared, Filter, the view toggle and the search box to line up "for
    # free" from near-identical padding/font-size across simple CSS boxes,
    # but Qt's per-widget sizeHint (padding + real Figtree font metrics) does
    # not converge on its own the same way; pinning every control to the
    # same explicit height is what actually guarantees they line up.
    _TOOLBAR_H = 34

    def _toolbar(self) -> QWidget:
        t = self._t
        H = self._TOOLBAR_H
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(34, 2, 34, 20)
        row.setSpacing(10)

        search = QLineEdit()
        search.setPlaceholderText("Search projects, tags, engines…")
        search.setClearButtonEnabled(True)
        search.setFixedHeight(H)
        search.setMaximumWidth(560)
        search.addAction(icons.icon("search", t["text_muted"], 15), QLineEdit.ActionPosition.LeadingPosition)
        search.textChanged.connect(self._on_search)
        row.addWidget(search, 1)

        self._scope_seg = SegControl(["All", "Favorites", "Shared"], t, 0)
        self._scope_seg.setFixedHeight(H)
        self._scope_seg.changed.connect(self._on_scope_changed)
        row.addWidget(self._scope_seg)

        row.addStretch(1)

        self._filter_btn = PillButton("Filter", t, "ghost", "filter", small=True)
        self._filter_btn.setToolTip("Filter by engine")
        self._filter_btn.setFixedHeight(H)
        self._filter_btn.clicked.connect(self._open_filter_menu)
        row.addWidget(self._filter_btn)

        sort_labels = list(project_controller.ProjectController.SORT_OPTIONS.keys())
        self._sort_box = SelectBox(sort_labels[0], t, options=sort_labels, on_select=self._on_sort_changed)
        self._sort_box.setFixedHeight(H)
        # SelectBox's own value label (_ElidingLabel) uses a horizontally
        # "Ignored" size policy -- right for its original fixed-width-panel
        # use (Segment inspector rows), but it means SelectBox's *own*
        # sizeHint doesn't reserve room for its text, so a bare
        # setFixedHeight() alone collapsed this to just the chevron in a
        # toolbar with room to spare (confirmed by screenshot: no visible
        # label text at all). 152px comfortably fits the longest option,
        # "Last modified" (measured ~107px of text + margins/chevron).
        self._sort_box.setMinimumWidth(152)
        self._sort_box.setToolTip("Sort by")
        row.addWidget(self._sort_box)

        self._view_seg = SegControl(["", ""], t, 0, icons_=["grid", "list"])
        self._view_seg.setFixedHeight(H)
        self._view_seg.changed.connect(self._on_view_changed)
        row.addWidget(self._view_seg)
        return bar

    # ── toolbar handlers ─────────────────────────────────────────────────────
    def _on_search(self, text: str) -> None:
        self._query = text
        self._populate()

    def _on_scope_changed(self, idx: int) -> None:
        self._scope = _SCOPES[idx]
        self._populate()

    def _on_view_changed(self, idx: int) -> None:
        self._view = "grid" if idx == 0 else "list"
        self._grid_host.setVisible(self._view == "grid")
        self._list_host.setVisible(self._view == "list")

    def _on_sort_changed(self, label: str) -> None:
        self._sort = project_controller.ProjectController.SORT_OPTIONS[label]
        self._populate()

    def _open_filter_menu(self) -> None:
        """A checkable engine multi-select, anchored under the Filter button.

        ``popup()`` (not ``exec()``) so opening it never blocks — the filter
        is applied via the ``toggled`` signal, no return value needed.
        """
        menu = QMenu(self)
        for key in ENGINES:
            act = QAction(ENGINE_LABELS[key], menu)
            act.setCheckable(True)
            act.setChecked(key in self._engines)
            act.toggled.connect(lambda checked, k=key: self._on_engine_toggled(k, checked))
            menu.addAction(act)
        if self._engines:
            menu.addSeparator()
            menu.addAction("Clear engine filter", self._clear_engine_filter)
        menu.popup(self._filter_btn.mapToGlobal(self._filter_btn.rect().bottomLeft()))
        self._filter_menu = menu  # keep a ref alive while it's open

    def _on_engine_toggled(self, engine_key: str, checked: bool) -> None:
        if checked:
            self._engines.add(engine_key)
        else:
            self._engines.discard(engine_key)
        self._restyle_filter_button()
        self._populate()

    def _clear_engine_filter(self) -> None:
        self._engines.clear()
        self._restyle_filter_button()
        self._populate()

    def _restyle_filter_button(self) -> None:
        kind = "primary" if self._engines else "ghost"
        self._filter_btn.setStyleSheet(
            theme.button_qss(self._t, kind) + "QPushButton{padding:7px 11px; font-size:12.5px;}")

    def _toggle_favorite(self, project_id: str) -> None:
        self._controller.toggle_favorite(project_id)
        self._populate()

    # ── overflow (⋯) menu + settings ─────────────────────────────────────────
    def _open_card_menu(self, anchor: QWidget, project_id: str, project_name: str) -> None:
        """A per-card/row overflow menu, anchored under ``anchor`` (the ⋯
        button that opened it) -- same on-demand-QMenu construction as
        ``_open_filter_menu``/``SelectBox._open_menu`` elsewhere in this file.
        Deliberately short (Open · Duplicate · Settings) -- rename and
        delete both live in Settings now, matching Label Studio's own
        minimal card overflow menu (Settings / Label) rather than listing
        every action here.
        """
        menu = QMenu(self)
        menu.addAction("Open", lambda: self._open(project_id))
        menu.addAction("Duplicate", lambda: self._duplicate(project_id))
        menu.addAction("Settings", lambda: self._open_settings(project_id))
        menu.popup(anchor.mapToGlobal(anchor.rect().bottomRight()))
        self._card_menu = menu  # keep a ref alive while it's open

    def _duplicate(self, project_id: str) -> None:
        dup = self._controller.duplicate_project(project_id)
        self.refresh()
        if self._toast:
            self._toast("Project duplicated", f"“{dup.name}” created.")

    def _open_settings(self, project_id: str) -> None:
        project = self._controller.store.load(project_id)
        # Keep a ref alive while it's open -- with nothing else referencing
        # the dialog Python-side (only Qt's own C++ parent-child ownership),
        # it's fair game for garbage collection before a click.
        self._active_dialog = ProjectSettingsDialog(
            self, self._t, project,
            on_saved=lambda name, desc: self._save_settings(project_id, name, desc),
            on_delete=lambda: self._delete_project(project_id, project.name))
        self._active_dialog.open()

    def _save_settings(self, project_id: str, name: str, description: str) -> None:
        self._controller.rename_project(project_id, name)
        project = self._controller.store.load(project_id)
        project.description = description
        self._controller.store.save(project)
        self.refresh()

    def _delete_project(self, project_id: str, project_name: str) -> None:
        self._controller.delete_project(project_id)
        self.refresh()
        if self._toast:
            self._toast("Project deleted", f"“{project_name}” was permanently deleted.")

    # ── data + rendering ─────────────────────────────────────────────────────
    def _filtered_projects(self) -> list:
        if self._scope == "shared":
            return []  # no multi-user/sharing backend yet — always empty, honestly
        return self._controller.list_projects(
            query=self._query, favorites_only=(self._scope == "favorites"),
            engines=self._engines, sort=self._sort)

    def _empty_message(self, has_cards: bool) -> str:
        if has_cards:
            return ""
        if self._scope == "shared":
            return "Studio doesn't support shared projects yet — everything stays on this device."
        if self._query:
            return f"No projects match “{self._query}”."
        if self._scope == "favorites":
            return "No favorites yet — star a project to see it here."
        if self._engines:
            return "No projects use the selected engine filter."
        return ""

    @staticmethod
    def _clear_layout(lay) -> None:
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _populate(self) -> None:
        cards = [project_controller.to_card(p) for p in self._filtered_projects()]

        self._empty_label.setText(self._empty_message(bool(cards)))
        self._empty_label.setVisible(not cards)

        self._clear_layout(self._grid)
        for i, card in enumerate(cards):
            self._grid.addWidget(self._card(card), i // 3, i % 3)
        idx = len(cards)
        self._grid.addWidget(self._ghost(), idx // 3, idx % 3)

        self._clear_layout(self._list)
        for card in cards:
            self._list.addWidget(self._list_row(card))
        self._list.addWidget(self._ghost_row())

        self._grid_host.setVisible(self._view == "grid")
        self._list_host.setVisible(self._view == "list")

    def _card(self, p: "project_controller.ProjectCard") -> QFrame:
        """A clean, text-only card -- no thumbnail/cover art. Matches Label
        Studio's own project card (reference screenshots supplied by the
        product owner): identity + stats + a footer, nothing decorative.
        The earlier version had a live-painted "nuclei art" cover; removed
        entirely (not hidden behind a flag -- genuinely not wanted), which
        also removes the single most expensive thing this grid used to
        repaint on every scroll frame (on top of the NucleiView caching and
        eased-wheel-scroll work already done -- this is the rest of that
        same performance story, and the more important half of it).
        """
        t = self._t
        card = QFrame()
        card.setObjectName("PCard")
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            f"#PCard{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;}}"
            f"#PCard:hover{{border-color:{t['border_strong']};}}")
        install_hover_lift(card, base=(14, 22, 3), hover=(22, 34, 6))
        b = QVBoxLayout(card)
        b.setContentsMargins(16, 16, 16, 16)
        b.setSpacing(3)

        head = QHBoxLayout()
        head.setSpacing(8)
        eng = EngineChip(p.engine_label, _ENGINE_DOT.get(p.engine_key, theme.VIZ[0]),
                        bg=t["surface2"], fg=t["text_subtle"], border=t["border"])
        head.addWidget(eng)
        head.addStretch(1)
        star = IconButton("star", t, 26, "Favourite",
                           on_click=lambda pid=p.id: self._toggle_favorite(pid))
        star.setIcon(icons.icon("star", "#f0b357" if p.favorite else t["text_muted"], 15))
        head.addWidget(star)
        more = IconButton("more", t, 26, "More")  # handler wired below (needs `more` itself as the menu anchor)
        more.clicked.connect(lambda _=False, w=more, pid=p.id, name=p.name: self._open_card_menu(w, pid, name))
        head.addWidget(more)
        b.addLayout(head)
        b.addSpacing(6)

        b.addWidget(label(p.name, 14.5, t["text"], 600, -0.2))
        desc = label(p.description, 12, t["text_muted"])
        desc.setWordWrap(True)
        desc.setFixedHeight(34)
        b.addWidget(desc)
        b.addSpacing(12)
        b.addWidget(hline(t))
        b.addSpacing(12)
        b.addLayout(self._stats(p))
        if p.tags:
            tags = QHBoxLayout()
            tags.setSpacing(6)
            for tag in p.tags[:3]:
                tags.addWidget(Chip(tag, t, "muted"))
            tags.addStretch(1)
            b.addSpacing(12)
            b.addLayout(tags)

        b.addSpacing(12)
        b.addWidget(hline(t))
        b.addSpacing(8)
        foot = QHBoxLayout()
        foot.addWidget(label(f"{p.progress}% segmented", 11, t["text_muted"], 600))
        foot.addStretch(1)
        when = QLabel(p.when)
        when.setStyleSheet(f"color:{t['text_muted']}; font-size:11px; font-family:{theme.MONO};")
        foot.addWidget(when)
        b.addLayout(foot)

        card.mouseReleaseEvent = lambda e, pid=p.id: self._open(pid)
        return card

    def _stats(self, p: "project_controller.ProjectCard") -> QHBoxLayout:
        t = self._t
        row = QHBoxLayout()
        row.setSpacing(14)
        f1 = p.f1 or "—"
        for value, cap, ok in [(str(p.n_images), "Images", False), (p.n_cells, "Cells", False),
                               (f1, "F1 vs GT", p.f1 is not None)]:
            cell = QVBoxLayout()
            cell.setSpacing(1)
            col = t["success"] if ok else t["text"]
            v = QLabel(value)
            v.setStyleSheet(f"color:{col}; font-family:{theme.MONO}; font-size:14px; font-weight:600;")
            cell.addWidget(v)
            cell.addWidget(label(cap.upper(), 10, t["text_muted"], 600, 0.5))
            row.addLayout(cell)
        row.addStretch(1)
        return row

    def _ghost(self) -> QFrame:
        t = self._t
        card = QFrame()
        card.setObjectName("Ghost")
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        # Matches a real card's own height (measured: 265px with tags, now
        # that cards are cover-art-free and shorter) so the grid's last
        # (ghost) cell doesn't stand out as a visibly different size.
        card.setMinimumHeight(265)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            f"#Ghost{{background:transparent; border:1px dashed {t['border_strong']}; border-radius:14px;}}"
            f"#Ghost:hover{{border-color:{t['primary_line']}; background:{t['primary_weak']};}}"
            f"#Ghost QFrame#PlusBox{{background:{t['surface2']}; border-radius:12px;}}"
            f"#Ghost:hover QFrame#PlusBox{{background:{t['surface']};}}")
        v = QVBoxLayout(card)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plus_box = QFrame()
        plus_box.setObjectName("PlusBox")
        plus_box.setFixedSize(44, 44)
        plus_box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        pl = QVBoxLayout(plus_box)
        pl.setContentsMargins(0, 0, 0, 0)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("plus", t["text_muted"], 22))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(ic)
        v.addWidget(plus_box, alignment=Qt.AlignmentFlag.AlignCenter)
        v.addSpacing(12)
        title = label("New Project", 13.5, t["text_subtle"], 600)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(title)
        sub = label("Import images & pick an engine", 12, t["text_muted"])
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(sub)
        card.mouseReleaseEvent = lambda e: self._new_project()
        return card

    # ── list view ────────────────────────────────────────────────────────────
    def _list_row(self, p: "project_controller.ProjectCard") -> QFrame:
        t = self._t
        row = QFrame()
        row.setObjectName("PRow")
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet(
            f"#PRow{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:10px;}}"
            f"#PRow:hover{{border-color:{t['border_strong']};}}")
        # No install_hover_lift() here (unlike other rows/cards): this row's
        # container starts out hidden (list view isn't the default), and a
        # QGraphicsDropShadowEffect installed while hidden leaves Qt's effect
        # source cache stale once the container is later shown — rows render
        # with wildly inflated spacing. Reproduced and confirmed by disabling
        # install_hover_lift() alone; the QSS :hover border-color above still
        # gives real hover feedback without the broken effect.
        lay = QHBoxLayout(row)
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(14)

        meta = QVBoxLayout()
        meta.setSpacing(2)
        meta.addWidget(label(p.name, 13.5, t["text"], 600))
        meta.addWidget(label(f"{p.engine_label} · {p.n_images} images · {p.n_cells} cells",
                             11.5, t["text_muted"]))
        lay.addLayout(meta, 1)

        f1_col = QVBoxLayout()
        f1_col.setSpacing(1)
        f1_val = QLabel(p.f1 or "—")
        f1_val.setAlignment(Qt.AlignmentFlag.AlignRight)
        f1_val.setStyleSheet(
            f"color:{t['success'] if p.f1 else t['text_muted']}; font-family:{theme.MONO};"
            f"font-size:13px; font-weight:600;")
        f1_col.addWidget(f1_val)
        f1_cap = label("F1 VS GT", 9.5, t["text_muted"], 600, 0.5)
        f1_cap.setAlignment(Qt.AlignmentFlag.AlignRight)
        f1_col.addWidget(f1_cap)
        lay.addLayout(f1_col)

        star = IconButton("star", t, 30, "Favourite",
                           on_click=lambda pid=p.id: self._toggle_favorite(pid))
        star.setIcon(icons.icon("star", "#f0b357" if p.favorite else t["text_muted"], 15))
        lay.addWidget(star)

        more = IconButton("more", t, 30, "More")  # handler wired below (needs `more` itself as the menu anchor)
        more.clicked.connect(lambda _=False, w=more, pid=p.id, name=p.name: self._open_card_menu(w, pid, name))
        lay.addWidget(more)

        row.mouseReleaseEvent = lambda e, pid=p.id: self._open(pid)
        return row

    def _ghost_row(self) -> QFrame:
        t = self._t
        row = QFrame()
        row.setObjectName("GhostRow")
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet(
            f"#GhostRow{{background:transparent; border:1px dashed {t['border_strong']}; border-radius:10px;}}"
            f"#GhostRow:hover{{border-color:{t['primary_line']}; background:{t['primary_weak']};}}")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(11)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("plus", t["text_muted"], 16))
        lay.addWidget(ic)
        lay.addWidget(label("New Project", 13, t["text_subtle"], 600))
        lay.addWidget(label("· Import images & pick an engine", 12, t["text_muted"]))
        lay.addStretch(1)
        row.mouseReleaseEvent = lambda e: self._new_project()
        return row
