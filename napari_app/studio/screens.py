"""CellSeg1 Studio — the Home and Projects screens.

Data-driven views over the :class:`~napari_app.studio.project.ProjectStore`:
Home (welcome, quick actions, recent projects, resources) and Projects (a
searchable card grid). Both call :meth:`refresh` when shown so they always
reflect the store on disk. Interaction is delegated up via callbacks the app
supplies (open a project, create a project, navigate).
"""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QScrollArea, QSizePolicy, QLineEdit,
)

from napari_app import icons
from napari_app.studio import theme
from napari_app.studio.components import (
    Chip, GhostButton, PrimaryButton, hline, soft_shadow,
)
from napari_app.studio.project import Project, ProjectStore

# engine key → (display label, accent token) for card chips
_ENGINE_META = {
    "cellseg1": ("CellSeg1 · LoRA", "primary"),
    "cellpose": ("Cellpose-SAM", "signal"),
    "sam2":     ("SAM 2", "primary"),
}


def _scroll(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setWidget(inner)
    return sa


def _page_header(title: str, subtitle: str, t: dict, action: QWidget | None = None) -> QWidget:
    head = QWidget()
    row = QHBoxLayout(head)
    row.setContentsMargins(34, 30, 34, 18)
    col = QVBoxLayout()
    col.setSpacing(5)
    h = QLabel(title)
    h.setStyleSheet(f"font-size:26px; font-weight:600; letter-spacing:-0.6px; color:{t['text']};")
    s = QLabel(subtitle)
    s.setStyleSheet(f"font-size:14px; color:{t['text_muted']};")
    col.addWidget(h)
    col.addWidget(s)
    row.addLayout(col)
    row.addStretch(1)
    if action is not None:
        row.addWidget(action, alignment=Qt.AlignmentFlag.AlignBottom)
    return head


# ── Project card ─────────────────────────────────────────────────────────────
class ProjectCard(QFrame):
    """A library card for one project: cover, name, description, stats, tags."""

    def __init__(self, project: Project, t: dict,
                 on_open: Callable[[str], None]):
        super().__init__()
        self.project = project
        self._t = t
        self._on_open = on_open
        self.setObjectName("PCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"#PCard{{background:{t['surface']}; border:1px solid {t['border']};"
            f"border-radius:14px;}} "
            f"#PCard:hover{{border-color:{t['border_strong']};}}")
        soft_shadow(self, blur=16, alpha=26, dy=3)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        lay.addWidget(self._cover())
        body = QWidget()
        b = QVBoxLayout(body)
        b.setContentsMargins(15, 14, 15, 15)
        b.setSpacing(3)

        name = QLabel(project.name)
        name.setStyleSheet(f"font-size:14.5px; font-weight:600; letter-spacing:-0.2px; color:{t['text']};")
        b.addWidget(name)

        desc = QLabel(project.description or "No description")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"font-size:12px; color:{t['text_muted']};")
        desc.setMaximumHeight(34)
        b.addWidget(desc)

        b.addSpacing(8)
        b.addWidget(hline(t))
        b.addSpacing(10)
        b.addLayout(self._stats_row())

        if project.tags:
            tagrow = QHBoxLayout()
            tagrow.setSpacing(6)
            for tag in project.tags[:3]:
                tagrow.addWidget(Chip(tag, t, "muted"))
            tagrow.addStretch(1)
            b.addSpacing(10)
            b.addLayout(tagrow)

        lay.addWidget(body)

    def _cover(self) -> QWidget:
        t = self._t
        label, kind = _ENGINE_META.get(self.project.engine, ("Engine", "primary"))
        accent = t["primary"] if kind == "primary" else t["signal"]
        cover = QFrame()
        cover.setFixedHeight(96)
        cover.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        cover.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            f"stop:0 {t['scope']}, stop:1 {accent}); "
            f"border-top-left-radius:14px; border-top-right-radius:14px;")
        cl = QHBoxLayout(cover)
        cl.setContentsMargins(12, 10, 12, 10)
        eng = Chip(label, t, "primary" if kind == "primary" else "signal")
        eng.setStyleSheet(eng.styleSheet() +
                          "background:rgba(8,12,16,0.55); color:#eaf0f8; border-color:rgba(255,255,255,0.15);")
        cl.addWidget(eng, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        cl.addStretch(1)
        if self.project.stats.progress:
            prog = QLabel(f"{self.project.stats.progress}%")
            prog.setStyleSheet(
                f"font-family:{theme.MONO}; font-size:10.5px; font-weight:600; color:#fff;"
                f"background:rgba(8,12,16,0.55); border-radius:6px; padding:2px 7px;")
            cl.addWidget(prog, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        if self.project.favorite:
            star = QLabel()
            star.setPixmap(icons.pixmap("star", "#f0b357", 15))
            cl.addWidget(star, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        return cover

    def _stats_row(self) -> QHBoxLayout:
        t = self._t
        row = QHBoxLayout()
        row.setSpacing(16)
        st = self.project.stats
        f1 = "—" if st.last_f1 is None else f"{st.last_f1:.2f}"
        for value, label, ok in (
            (str(st.n_images), "Images", False),
            (f"{st.n_cells:,}" if st.n_cells else "0", "Cells", False),
            (f1, "F1 vs GT", st.last_f1 is not None),
        ):
            cell = QVBoxLayout()
            cell.setSpacing(1)
            v = QLabel(value)
            col = t["success"] if ok else t["text"]
            v.setStyleSheet(f"font-size:14px; font-weight:600; font-family:{theme.MONO}; color:{col};")
            l = QLabel(label.upper())
            l.setStyleSheet(f"font-size:10px; color:{t['text_muted']}; font-weight:600; letter-spacing:0.5px;")
            cell.addWidget(v)
            cell.addWidget(l)
            row.addLayout(cell)
        row.addStretch(1)
        return row

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._on_open(self.project.id)
        super().mouseReleaseEvent(e)


class NewProjectCard(QFrame):
    """The dashed 'ghost' card that creates a new project."""

    def __init__(self, t: dict, on_new: Callable[[], None]):
        super().__init__()
        self._on_new = on_new
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("NewCard")
        self.setMinimumHeight(240)
        self.setStyleSheet(
            f"#NewCard{{background:transparent; border:1px dashed {t['border_strong']};"
            f"border-radius:14px;}} "
            f"#NewCard:hover{{border-color:{t['primary_line']}; background:{t['primary_weak']};}}")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = QLabel()
        icon.setPixmap(icons.pixmap("plus", t["text_muted"], 26))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("New Project")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"font-size:13.5px; font-weight:600; color:{t['text_subtle']};")
        sub = QLabel("Import images & pick an engine")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(f"font-size:12px; color:{t['text_muted']};")
        lay.addWidget(icon)
        lay.addSpacing(8)
        lay.addWidget(title)
        lay.addWidget(sub)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._on_new()
        super().mouseReleaseEvent(e)


# ── Home screen ──────────────────────────────────────────────────────────────
class HomeScreen(QWidget):
    def __init__(self, store: ProjectStore, t: dict,
                 on_new: Callable[[], None],
                 on_open: Callable[[str], None],
                 on_navigate: Callable[[str], None]):
        super().__init__()
        self._store = store
        self._t = t
        self._on_open = on_open
        self._on_navigate = on_navigate

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(_page_header(
            "Welcome back  👋",
            "Segment, measure and compare cells across your projects.",
            t, PrimaryButton("New Project", t, "plus")))

        content = QWidget()
        self._content = QVBoxLayout(content)
        self._content.setContentsMargins(34, 4, 34, 40)
        self._content.setSpacing(24)
        outer.addWidget(_scroll(content))

        # quick actions
        self._content.addLayout(self._quick_actions())
        # recent
        self._recent_title = self._section_title("Recent projects", "View all →", "projects")
        self._content.addWidget(self._recent_title)
        self._recent_box = QVBoxLayout()
        self._recent_box.setSpacing(10)
        self._content.addLayout(self._recent_box)
        self._content.addStretch(1)

        # wire the header CTA
        for btn in self.findChildren(PrimaryButton):
            btn.clicked.connect(on_new)

        self.refresh()

    def _section_title(self, text: str, link: str, link_key: str) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        h = QLabel(text)
        h.setStyleSheet(f"font-size:15px; font-weight:600; color:{self._t['text']};")
        a = QLabel(f"<a href='#' style='color:{self._t['primary']}; text-decoration:none;'>{link}</a>")
        a.setStyleSheet("font-size:12.5px; font-weight:600;")
        a.linkActivated.connect(lambda: self._on_navigate(link_key))
        row.addWidget(h)
        row.addStretch(1)
        row.addWidget(a)
        return w

    def _quick_actions(self) -> QGridLayout:
        t = self._t
        grid = QGridLayout()
        grid.setSpacing(14)
        cards = [
            ("folder", "New Project", "Name it, import images, pick an engine.", "primary"),
            ("image", "Import Images", "TIFF · OME-TIFF · ND2 · CZI · PNG. Drag & drop.", "signal"),
            ("models", "Train a Model", "One-shot LoRA from a single annotated image.", "warning"),
            ("chart", "Open Sample", "Nuclei, tissue & mitosis datasets to explore.", "success"),
        ]
        for i, (icon_name, title, sub, kind) in enumerate(cards):
            grid.addWidget(self._quick_card(icon_name, title, sub, kind), i // 2, i % 2)
        return grid

    def _quick_card(self, icon_name, title, sub, kind) -> QFrame:
        t = self._t
        card = QFrame()
        card.setObjectName("QCard")
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            f"#QCard{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:14px;}}"
            f"#QCard:hover{{border-color:{t['border_strong']};}}")
        soft_shadow(card, blur=14, alpha=22, dy=3)
        row = QHBoxLayout(card)
        row.setContentsMargins(18, 16, 18, 16)
        row.setSpacing(14)
        col_map = {"primary": t["primary"], "signal": t["signal"],
                   "warning": t["warning"], "success": t["success"]}
        weak_map = {"primary": t["primary_weak"], "signal": t["signal_weak"],
                    "warning": t["warning_weak"], "success": t["success_weak"]}
        badge = QLabel()
        badge.setFixedSize(38, 38)
        badge.setPixmap(icons.pixmap(icon_name, col_map[kind], 19))
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(f"background:{weak_map[kind]}; border-radius:10px;")
        col = QVBoxLayout()
        col.setSpacing(3)
        h = QLabel(title)
        h.setStyleSheet(f"font-size:14.5px; font-weight:600; color:{t['text']};")
        s = QLabel(sub)
        s.setWordWrap(True)
        s.setStyleSheet(f"font-size:12.5px; color:{t['text_muted']};")
        col.addWidget(h)
        col.addWidget(s)
        row.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)
        row.addLayout(col, 1)
        return card

    def _clear(self, box):
        while box.count():
            item = box.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def refresh(self) -> None:
        self._clear(self._recent_box)
        recents = self._store.recent(limit=4)
        if not recents:
            empty = QLabel("No projects yet — create your first one to get started.")
            empty.setStyleSheet(f"color:{self._t['text_muted']}; font-size:13px; padding:8px 2px;")
            self._recent_box.addWidget(empty)
            return
        for p in recents:
            self._recent_box.addWidget(self._recent_row(p))

    def _recent_row(self, p: Project) -> QFrame:
        t = self._t
        row = QFrame()
        row.setObjectName("RRow")
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet(
            f"#RRow{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:10px;}}"
            f"#RRow:hover{{border-color:{t['border_strong']};}}")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(14)
        label, kind = _ENGINE_META.get(p.engine, ("Engine", "primary"))
        swatch = QLabel()
        swatch.setFixedSize(40, 30)
        accent = t["primary"] if kind == "primary" else t["signal"]
        swatch.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 {t['scope']}, stop:1 {accent});"
            f"border-radius:6px;")
        meta = QVBoxLayout()
        meta.setSpacing(2)
        nm = QLabel(p.name)
        nm.setStyleSheet(f"font-size:13.5px; font-weight:600; color:{t['text']};")
        mm = QLabel(f"{label} · {p.stats.n_images} images · {p.stats.n_cells:,} cells")
        mm.setStyleSheet(f"font-size:11.5px; color:{t['text_muted']};")
        meta.addWidget(nm)
        meta.addWidget(mm)
        lay.addWidget(swatch)
        lay.addLayout(meta, 1)
        row.mouseReleaseEvent = lambda e, pid=p.id: self._on_open(pid)
        return row


# ── Projects screen ──────────────────────────────────────────────────────────
class ProjectsScreen(QWidget):
    def __init__(self, store: ProjectStore, t: dict,
                 on_new: Callable[[], None],
                 on_open: Callable[[str], None]):
        super().__init__()
        self._store = store
        self._t = t
        self._on_new = on_new
        self._on_open = on_open
        self._query = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        cta = PrimaryButton("New Project", t, "plus")
        cta.clicked.connect(on_new)
        self._header = _page_header("Projects", "", t, cta)
        outer.addWidget(self._header)
        outer.addWidget(self._toolbar())

        content = QWidget()
        self._grid_host = QVBoxLayout(content)
        self._grid_host.setContentsMargins(34, 0, 34, 40)
        self._grid = QGridLayout()
        self._grid.setSpacing(16)
        self._grid_host.addLayout(self._grid)
        self._grid_host.addStretch(1)
        outer.addWidget(_scroll(content))

        self.refresh()

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
        search.textChanged.connect(self._on_search)
        row.addWidget(search)
        row.addStretch(1)
        row.addWidget(GhostButton("Filter", t, "filter"))
        return bar

    def _on_search(self, text: str) -> None:
        self._query = text.strip().lower()
        self.refresh()

    def _match(self, p: Project) -> bool:
        if not self._query:
            return True
        hay = " ".join([p.name, p.description, p.engine, *p.tags]).lower()
        return self._query in hay

    def _clear_grid(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def refresh(self) -> None:
        self._clear_grid()
        projects = [p for p in self._store.list() if self._match(p)]
        self._header.findChild(QLabel)  # keep ref alive; subtitle updated below
        # update subtitle count
        subs = self._header.findChildren(QLabel)
        if len(subs) >= 2:
            total = len(self._store.list())
            subs[1].setText(f"{total} project{'s' if total != 1 else ''}")

        cols = 3
        idx = 0
        for p in projects:
            self._grid.addWidget(ProjectCard(p, self._t, self._on_open), idx // cols, idx % cols)
            idx += 1
        # ghost 'new' card only when not filtering
        if not self._query:
            self._grid.addWidget(NewProjectCard(self._t, self._on_new), idx // cols, idx % cols)
