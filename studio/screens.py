"""CellSeg1 Studio — the screens.

Native reproductions of the mockup, one screen per class, driven by the
``components``/``paint`` kits. Home and Projects are wired to the real
``ProjectController`` (live projects, search/filter/favourites); the other
screens still render ``demo.py`` static content pending their own tab in
``docstudio/BACKLOG.md``. The Workspace is the signature screen (adapted-napari
layers · canvas · inspector). See ``docstudio/`` for the per-tab wiring plan.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QSize, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QScrollArea, QLineEdit, QSizePolicy, QStackedWidget, QToolButton,
)

from studio import icons
from studio import theme
from studio import project_controller
from studio.project import ENGINE_KIND
from studio.paint import nuclei_pixmap, NucleiView
from studio.motion import install_hover_lift
from studio.components import (
    Chip, Badge, PillButton, IconButton, SelectBox, Toggle, Slider, Stepper,
    SegControl, StatTile, FieldRow, GroupLabel, Accordion, hline, soft_shadow, label,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _open_local_doc(*relative_parts: str) -> None:
    path = _REPO_ROOT.joinpath(*relative_parts)
    if path.exists():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


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
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setWidget(inner)
    return sa


def page_header(title: str, subtitle: str, t: dict, action: Optional[QWidget] = None) -> QWidget:
    head = QWidget()
    row = QHBoxLayout(head)
    row.setContentsMargins(34, 30, 34, 18)
    col = QVBoxLayout()
    col.setSpacing(5)
    col.addWidget(label(title, 26, t["text"], 600, -0.6))
    if subtitle:
        col.addWidget(label(subtitle, 14, t["text_muted"]))
    row.addLayout(col)
    row.addStretch(1)
    if action is not None:
        row.addWidget(action, alignment=Qt.AlignmentFlag.AlignBottom)
    return head


def cover_label(seed: int, w: int, h: int, density: float, big: bool = False) -> QLabel:
    lb = QLabel()
    lb.setFixedSize(w, h)
    lb.setPixmap(nuclei_pixmap(w, h, seed, density=density, big=big))
    lb.setScaledContents(True)
    return lb


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
        outer.addWidget(page_header("Welcome back  👋",
                                    "Segment, measure and compare cells across your projects.",
                                    t, self._new_project_cta))

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

    def refresh(self) -> None:
        """Rebuild the recent-projects list from the store's current state.

        HomeScreen is built once and kept alive across navigation (the stack
        just swaps the visible page), so anything that changes the store
        elsewhere -- most importantly creating a project -- needs this
        called before Home is shown again, or it'd still show what the store
        looked like at construction time.
        """
        idx = self._left.indexOf(self._recent_widget)
        old = self._recent_widget
        self._recent_widget = self._recent_section()
        self._left.insertWidget(idx, self._recent_widget)
        self._left.removeWidget(old)
        old.setParent(None)
        old.deleteLater()

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

    def _recent_section(self) -> QWidget:
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
        for card in recent:
            v.addWidget(self._recent_row(card))
        return w

    def _recent_row(self, p: "project_controller.ProjectCard") -> QFrame:
        t = self._t
        row = QFrame()
        row.setObjectName("RRow")
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet(
            f"#RRow{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:10px;}}"
            f"#RRow:hover{{border-color:{t['border_strong']};}}")
        install_hover_lift(row)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(14)
        lay.addWidget(cover_label(p.seed, 52, 40, 1.5))
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
        tip.setStyleSheet(
            f"background:{t['primary_weak']}; border:1px solid {t['primary_line']}; border-radius:14px;")
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
            ("Documentation", "guide", True, lambda: _open_local_doc("README.md")),
            ("Getting started guide", "guide", False, lambda: _open_local_doc("docstudio", "OVERVIEW.md")),
            ("Ask the Assistant", "assistant", False, lambda: self._nav("assistant")),
            ("GitHub", "settings", True, _open_github)]:
            res.layout().addWidget(self._res_link(name, icon_name, ext, on_click))
        col.addWidget(res)

        dev = self._card("This device")
        for name, val in [("Compute", "Apple M-series · MPS"), ("SAM backbone", "ViT-H · cached"),
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
        c.setStyleSheet(f"background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;")
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
class ProjectsScreen(QWidget):
    def __init__(self, t: dict, controller: "project_controller.ProjectController",
                 on_navigate: Callable[[str], None], on_open: Callable[[str], None]):
        super().__init__()
        self._t = t
        self._controller = controller
        self._nav = on_navigate
        self._open = on_open
        self._query = ""
        self._favorites_only = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._outer = outer
        self._header_widget = self._build_header()
        outer.addWidget(self._header_widget)
        outer.addWidget(self._toolbar())

        body = QWidget()
        host = QVBoxLayout(body)
        host.setContentsMargins(34, 0, 34, 40)
        self._grid = QGridLayout()
        self._grid.setSpacing(16)
        host.addLayout(self._grid)
        host.addStretch(1)
        outer.addWidget(scroll(body))

        self._populate_grid()

    def _build_header(self) -> QWidget:
        t = self._t
        cta = PillButton("New Project", t, "primary", "plus")
        cta.clicked.connect(lambda: self._nav("workspace"))
        n_projects, n_images, n_engines = self._controller.summary()
        subtitle = (f"{n_projects} project{'s' if n_projects != 1 else ''} · "
                    f"{n_images} images · {n_engines} engine{'s' if n_engines != 1 else ''}")
        return page_header("Projects", subtitle, t, cta)

    def refresh(self) -> None:
        """Recompute the header counts and repopulate the grid.

        ProjectsScreen is built once and kept alive across navigation (see
        HomeScreen.refresh for why), so anything that changes the store
        elsewhere needs this called before Projects is shown again. The
        current search query / favourites-only filter are preserved.
        """
        idx = self._outer.indexOf(self._header_widget)
        old = self._header_widget
        self._header_widget = self._build_header()
        self._outer.insertWidget(idx, self._header_widget)
        self._outer.removeWidget(old)
        old.setParent(None)
        old.deleteLater()
        self._populate_grid()

    def _toolbar(self) -> QWidget:
        t = self._t
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(34, 2, 34, 20)
        row.setSpacing(10)
        search = QLineEdit()
        search.setPlaceholderText("Search projects, tags, engines…")
        search.setClearButtonEnabled(True)
        search.setMaximumWidth(420)
        search.addAction(icons.icon("diagnose", t["text_muted"], 15), QLineEdit.ActionPosition.LeadingPosition)
        search.textChanged.connect(self._on_search)
        row.addWidget(search)
        row.addStretch(1)
        self._filter_btn = PillButton("Filter", t, "ghost", "filter", small=True)
        self._filter_btn.setToolTip("Show favourites only")
        self._filter_btn.clicked.connect(self._on_toggle_filter)
        row.addWidget(self._filter_btn)
        view = SegControl(["▦", "☰"], t, 0, compact=True)
        view.setFixedWidth(78)
        row.addWidget(view)
        return bar

    def _on_search(self, text: str) -> None:
        self._query = text
        self._populate_grid()

    def _on_toggle_filter(self) -> None:
        self._favorites_only = not self._favorites_only
        kind = "primary" if self._favorites_only else "ghost"
        self._filter_btn.setStyleSheet(
            theme.button_qss(self._t, kind) + "QPushButton{padding:7px 11px; font-size:12.5px;}")
        self._populate_grid()

    def _toggle_favorite(self, project_id: str) -> None:
        self._controller.toggle_favorite(project_id)
        self._populate_grid()

    def _populate_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        projects = self._controller.list_projects(query=self._query, favorites_only=self._favorites_only)
        cards = [project_controller.to_card(p) for p in projects]
        for i, card in enumerate(cards):
            self._grid.addWidget(self._card(card), i // 3, i % 3)
        idx = len(cards)
        self._grid.addWidget(self._ghost(), idx // 3, idx % 3)

    def _card(self, p: "project_controller.ProjectCard") -> QFrame:
        t = self._t
        card = QFrame()
        card.setObjectName("PCard")
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            f"#PCard{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;}}"
            f"#PCard:hover{{border-color:{t['border_strong']};}}")
        soft_shadow(card, 16, 26, 3)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # cover
        cover = QLabel()
        cover.setFixedHeight(120)
        cover.setPixmap(nuclei_pixmap(360, 120, p.seed, density=1.1, big=False))
        cover.setScaledContents(True)
        cover.setStyleSheet("border-top-left-radius:14px; border-top-right-radius:14px;")
        cwrap = QFrame()
        cwrap.setFixedHeight(120)
        cwl = QVBoxLayout(cwrap)
        cwl.setContentsMargins(0, 0, 0, 0)
        cover.setParent(cwrap)

        top = QHBoxLayout()
        top.setContentsMargins(8, 8, 8, 0)
        top.addStretch(1)
        star = IconButton("star", t, 26, "Favourite",
                           on_click=lambda pid=p.id: self._toggle_favorite(pid))
        star_color = "#f0b357" if p.favorite else "rgba(255,255,255,0.65)"
        star.setIcon(icons.icon("star", star_color, 15))
        star.setStyleSheet(
            "QToolButton{background:rgba(8,12,16,0.4); border:1px solid transparent; border-radius:8px;}"
            "QToolButton:hover{background:rgba(8,12,16,0.6);}")
        top.addWidget(star)
        cwl.addLayout(top)

        overlay = QHBoxLayout()
        overlay.setContentsMargins(12, 0, 12, 10)
        eng = Chip(p.engine_label, t, ENGINE_KIND.get(p.engine_key, "primary"))
        eng.setStyleSheet(eng.styleSheet() + "background:rgba(8,12,16,0.55);color:#eaf0f8;border-color:rgba(255,255,255,0.15);")
        overlay.addWidget(eng, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        overlay.addStretch(1)
        prog = QLabel(f"{p.progress}%")
        prog.setStyleSheet(f"color:#fff; font-family:{theme.MONO}; font-size:10.5px; font-weight:600;"
                           f"background:rgba(8,12,16,0.55); border-radius:6px; padding:2px 7px;")
        overlay.addWidget(prog, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        cwl.addStretch(1)
        cwl.addLayout(overlay)
        lay.addWidget(cwrap)

        body = QWidget()
        b = QVBoxLayout(body)
        b.setContentsMargins(15, 14, 15, 15)
        b.setSpacing(3)
        b.addWidget(label(p.name, 14.5, t["text"], 600, -0.2))
        desc = label(p.description, 12, t["text_muted"])
        desc.setWordWrap(True)
        desc.setMaximumHeight(34)
        b.addWidget(desc)
        b.addSpacing(8)
        b.addWidget(hline(t))
        b.addSpacing(10)
        b.addLayout(self._stats(p))
        if p.tags:
            tags = QHBoxLayout()
            tags.setSpacing(6)
            for tag in p.tags[:3]:
                tags.addWidget(Chip(tag, t, "muted"))
            tags.addStretch(1)
            b.addSpacing(10)
            b.addLayout(tags)
        lay.addWidget(body)
        card.mouseReleaseEvent = lambda e, pid=p.id: self._open(pid)
        return card

    def _stats(self, p: "project_controller.ProjectCard") -> QHBoxLayout:
        t = self._t
        row = QHBoxLayout()
        row.setSpacing(16)
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
        card.setMinimumHeight(240)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            f"#Ghost{{background:transparent; border:1px dashed {t['border_strong']}; border-radius:14px;}}"
            f"#Ghost:hover{{border-color:{t['primary_line']}; background:{t['primary_weak']};}}")
        v = QVBoxLayout(card)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("plus", t["text_muted"], 26))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(ic)
        v.addSpacing(8)
        title = label("New Project", 13.5, t["text_subtle"], 600)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(title)
        sub = label("Import images & pick an engine", 12, t["text_muted"])
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(sub)
        card.mouseReleaseEvent = lambda e: self._nav("workspace")
        return card


from studio.workspace import WorkspaceScreen  # noqa: E402
from studio.extra_screens import ModelsScreen, DashboardScreen  # noqa: E402
