"""Velum — the Datasets tab: build, browse, and re-use your own datasets.

A first-class surface for the product's core loop — *collect → segment →
proofread → **curate a dataset** → re-use / train*. Three pieces:

  * ``DatasetsScreen`` — a list of the datasets you've built (empty state → a
    grid of cards) with a list→detail navigation; the detail view offers
    "Open in a new project" (re-import for more proofreading), "Train on this",
    "Reveal folder", and "Delete".
  * ``NewDatasetDialog`` — the *interactive build*: pick a source project, then
    tick exactly which of its images to include (each row shows segmented /
    cell-count / ground-truth status), name it, set an optional val split, and
    create. This is what makes dataset-building a curated workflow, not a
    single blind "export" button.

All logic is the Qt-free ``DatasetController`` (unit-tested); this file is only
the view. Follows the app's hard-won rendering rules: every bordered ``QFrame``
scopes its stylesheet to an ``#ObjectName``; every plain grouping widget is
transparent; the modal uses ``theme.SCRIM`` (see ``studio.memory`` lessons in
``docstudio/CHANGELOG.md``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QSizePolicy, QVBoxLayout, QWidget,
)

from studio import icons, theme
from studio.components import (
    Badge, Chip, EngineChip, GroupLabel, IconButton, PillButton, SegControl,
    SelectBox, StatTile, Toggle, bare_widget, hline, label, soft_shadow,
)
from studio.dataset import DatasetInfo
from studio.dataset_controller import (
    BuildCandidate, DatasetController, ImportScan,
)
from studio.project import ENGINE_KIND, ENGINE_LABELS, Project
from studio.project_controller import ProjectController, relative_time
from studio.screens import page_header, scroll


def _clear_layout(layout) -> None:
    """Empty a layout synchronously — ``setParent(None)`` detaches each widget
    at once (no brief double-render on rebuild), recursing into nested layouts
    (rows/grids) whose ``item.widget()`` is ``None``."""
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        else:
            child = item.layout()
            if child is not None:
                _clear_layout(child)
                child.deleteLater()


def _engine_chip(engine: str, t: dict) -> EngineChip:
    kind = ENGINE_KIND.get(engine, "primary")
    hue = t.get(kind, t["primary"])
    return EngineChip(ENGINE_LABELS.get(engine, engine or "—"),
                      hue, t["surface2"], t["text_subtle"], t["border"])


class DatasetsScreen(QWidget):
    def __init__(self, t: dict, datasets: DatasetController,
                 projects: ProjectController,
                 on_toast: Callable[[str, str], None],
                 on_open_project: Callable[[Project], None],
                 on_navigate: Callable[[str], None]):
        super().__init__()
        self._t = t
        self._datasets = datasets
        self._projects = projects
        self._toast = on_toast
        self._on_open_project = on_open_project
        self._on_navigate = on_navigate
        self._dialog: Optional[NewDatasetDialog] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        new_btn = PillButton("New dataset", t, "primary", "plus", small=True)
        new_btn.clicked.connect(self.open_new_dialog)
        self._new_btn = new_btn
        root.addWidget(page_header(
            "Datasets",
            "Curate and re-use your own image + mask collections.", t,
            action=new_btn))

        self._body = QVBoxLayout()
        self._body.setContentsMargins(34, 4, 34, 28)
        self._body.setSpacing(0)
        body_wrap = bare_widget(self._body)
        root.addWidget(scroll(body_wrap), 1)
        self.refresh()

    # ── public API (called by app on navigate + after a build) ───────────────
    def _clear_body(self) -> None:
        _clear_layout(self._body)

    def refresh(self) -> None:
        self._clear_body()
        infos = self._datasets.list_datasets()
        if not infos:
            self._body.addWidget(self._empty_state())
            self._body.addStretch(1)
            return
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)
        for i, info in enumerate(infos):
            grid.addWidget(self._card(info), i // 2, i % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self._body.addLayout(grid)
        self._body.addStretch(1)

    def open_new_dialog(self) -> None:
        if self._dialog is None:
            self._dialog = NewDatasetDialog(
                self, self._t, self._datasets, self._projects,
                on_toast=self._toast, on_built=self._after_build)
        self._dialog.open()

    # ── empty state ──────────────────────────────────────────────────────────
    def _empty_state(self) -> QWidget:
        t = self._t
        card = QFrame()
        card.setObjectName("DSEmpty")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            f"QFrame#DSEmpty{{background:{t['surface']};border:1px solid {t['border']};"
            f"border-radius:16px;}}")
        v = QVBoxLayout(card)
        v.setContentsMargins(40, 46, 40, 46)
        v.setSpacing(10)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = label("🗂️", 40, t["text"])
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(icon)
        title = label("No datasets yet", 17, t["text"], 600)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(title)
        sub = label("Build one from a project's segmented images — pick exactly "
                    "which to include, then re-use or train on it.",
                    12.5, t["text_muted"])
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        sub.setFixedWidth(360)
        v.addWidget(sub, alignment=Qt.AlignmentFlag.AlignCenter)
        cta = PillButton("New dataset", t, "primary", "plus", small=True)
        cta.clicked.connect(self.open_new_dialog)
        v.addWidget(cta, alignment=Qt.AlignmentFlag.AlignCenter)
        return card

    # ── list card ────────────────────────────────────────────────────────────
    def _card(self, info: DatasetInfo) -> QWidget:
        t = self._t
        card = QFrame()
        card.setObjectName("DSCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            f"QFrame#DSCard{{background:{t['surface']};border:1px solid {t['border']};"
            f"border-radius:14px;}}")
        soft_shadow(card, 14, 20, 3)
        v = QVBoxLayout(card)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(_engine_chip(info.engine, t))
        top.addStretch(1)
        if info.created:
            top.addWidget(label(relative_time(info.created), 11, t["text_muted"]))
        v.addLayout(top)

        v.addWidget(label(info.name, 16, t["text"], 600))

        stats = QHBoxLayout()
        stats.setSpacing(16)
        stats.addWidget(label(f"{info.n_images} images", 12, t["text_muted"]))
        stats.addWidget(label(f"{info.n_cells} cells", 12, t["text_muted"]))
        split = f"{info.n_train} train"
        if info.n_val:
            split += f" · {info.n_val} val"
        stats.addWidget(label(split, 12, t["text_muted"]))
        stats.addStretch(1)
        v.addLayout(stats)

        v.addWidget(hline(t))
        actions = QHBoxLayout()
        actions.setSpacing(8)
        open_btn = PillButton("Open", t, "ghost", "chevron", small=True)
        open_btn.clicked.connect(lambda _=False, i=info: self._show_detail(i))
        proj_btn = PillButton("New project", t, "ghost", "projects", small=True)
        proj_btn.clicked.connect(lambda _=False, i=info: self._import_to_project(i))
        actions.addWidget(open_btn)
        actions.addWidget(proj_btn)
        actions.addStretch(1)
        v.addLayout(actions)
        return card

    # ── detail (rendered inline, replacing the list) ─────────────────────────
    def _show_detail(self, info: DatasetInfo) -> None:
        # Re-read from disk so counts/images reflect the current folder.
        fresh = self._datasets.get(info.id) or info
        self._clear_body()
        self._body.addWidget(self._detail(fresh))
        self._body.addStretch(1)

    def _detail(self, info: DatasetInfo) -> QWidget:
        t = self._t
        wrap = bare_widget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(16)

        back = PillButton("← Datasets", t, "ghost", small=True)
        back.clicked.connect(self.refresh)
        head = QHBoxLayout()
        head.addWidget(back)
        head.addStretch(1)
        v.addLayout(head)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title_row.addWidget(label(info.name, 22, t["text"], 600))
        title_row.addWidget(_engine_chip(info.engine, t))
        title_row.addStretch(1)
        v.addLayout(title_row)

        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        px = f"{info.pixel_size_um}" if info.pixel_size_um else "—"
        for value, unit, caption in (
            (str(info.n_images), "", "Images"),
            (str(info.n_cells), "", "Cells"),
            (f"{info.n_train}/{info.n_val}", "", "Train / Val"),
            (px, "µm/px" if info.pixel_size_um else "", "Pixel size"),
        ):
            tile = StatTile(value, unit, caption, t)
            tile.setMinimumWidth(104)     # wide enough that captions never elide
            tiles.addWidget(tile)
        tiles.addStretch(1)
        v.addLayout(tiles)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        open_proj = PillButton("Open in a new project", t, "primary", "projects", small=True)
        open_proj.clicked.connect(lambda _=False, i=info: self._import_to_project(i))
        train_btn = PillButton("Train on this", t, "ghost", "models", small=True)
        train_btn.clicked.connect(lambda _=False, i=info: self._train_on(i))
        reveal_btn = PillButton("Reveal folder", t, "ghost", "folder", small=True)
        reveal_btn.clicked.connect(lambda _=False, i=info: self._reveal(i))
        del_btn = PillButton("Delete", t, "ghost", "trash", small=True)
        del_btn.clicked.connect(lambda _=False, i=info: self._delete(i))
        for b in (open_proj, train_btn, reveal_btn):
            actions.addWidget(b)
        actions.addStretch(1)
        actions.addWidget(del_btn)
        v.addLayout(actions)

        v.addWidget(label("IMAGES", 11, t["text_muted"], 600))
        list_card = QFrame()
        list_card.setObjectName("DSImages")
        list_card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        list_card.setStyleSheet(
            f"QFrame#DSImages{{background:{t['surface']};border:1px solid {t['border']};"
            f"border-radius:12px;}}")
        lv = QVBoxLayout(list_card)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.setSpacing(0)
        images = info.images
        for idx, rec in enumerate(images):
            if idx:
                lv.addWidget(hline(t))
            lv.addWidget(self._image_row(rec))
        if not images:
            lv.addWidget(label("This dataset has no images.", 12, t["text_muted"]))
        v.addWidget(list_card)
        return wrap

    def _image_row(self, rec: dict) -> QWidget:
        t = self._t
        row = bare_widget()
        h = QHBoxLayout(row)
        h.setContentsMargins(12, 9, 12, 9)
        h.setSpacing(10)
        name = Path(str(rec.get("image", "image"))).name
        h.addWidget(label(name, 12.5, t["text"]))
        h.addStretch(1)
        h.addWidget(label(f"{int(rec.get('cells', 0))} cells", 11.5, t["text_muted"]))
        split = str(rec.get("split", "train"))
        h.addWidget(Chip("val" if split == "val" else "train", t,
                         "warning" if split == "val" else "default"))
        return row

    # ── actions ──────────────────────────────────────────────────────────────
    def _import_to_project(self, info: DatasetInfo) -> None:
        try:
            project = self._datasets.import_to_project(info, self._projects.store)
        except Exception as exc:   # noqa: BLE001
            self._toast("Import failed", str(exc))
            return
        self._toast("Project created",
                    f"'{project.name}' — {project.stats.n_images} images, masks loaded")
        self._on_open_project(project)

    def _train_on(self, info: DatasetInfo) -> None:
        images_dir, masks_dir = DatasetController.train_target(info)
        self._on_navigate("train")
        self._toast("Train on dataset",
                    f"Point Images at {images_dir} and Masks at {masks_dir}")

    def _reveal(self, info: DatasetInfo) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(info.path)))

    def _delete(self, info: DatasetInfo) -> None:
        from studio.project_dialogs import ConfirmDialog
        def do_delete():
            self._datasets.delete(info.id)
            self._toast("Dataset deleted", info.name)
            self.refresh()
        dlg = ConfirmDialog(
            self, self._t,
            "Delete dataset?",
            f"'{info.name}' and its images/masks will be permanently removed.",
            confirm_label="Delete", on_confirm=do_delete)
        dlg.open()

    def _after_build(self, info: DatasetInfo) -> None:
        self._toast("Dataset created",
                    f"'{info.name}' — {info.n_images} images, {info.n_cells} cells")
        self._show_detail(info)


class _ImportDropZone(QFrame):
    """Drag-and-drop target (folder or files) + native pickers — the
    upload-first entry point every comparable tool leads with (Roboflow's
    drop panel, Label Studio's Upload Files)."""

    def __init__(self, t: dict, on_paths: Callable[[list[str]], None]):
        super().__init__()
        self._t = t
        self._on_paths = on_paths
        self.setAcceptDrops(True)
        self.setMinimumHeight(118)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("ImportDrop")
        self._style(False)
        v = QVBoxLayout(self)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.setSpacing(7)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("download", t["text_muted"], 22))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic.setStyleSheet("background:transparent;")
        v.addWidget(ic)
        head = label("Drop a folder or images here", 12.5, t["text_subtle"], 600)
        head.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(head)
        sub = label("A folder of images + masks, or an exported Velum dataset",
                    11, t["text_muted"])
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(sub)
        btns = QHBoxLayout()
        btns.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btns.setSpacing(8)
        folder = PillButton("Choose folder…", t, "ghost", "folder", small=True)
        folder.clicked.connect(self._pick_folder)
        files = PillButton("Choose files…", t, "ghost", "image", small=True)
        files.clicked.connect(self._pick_files)
        btns.addWidget(folder)
        btns.addWidget(files)
        v.addLayout(btns)

    def _style(self, active: bool) -> None:
        t = self._t
        border = t["primary_line"] if active else t["border_strong"]
        bg = t["primary_weak"] if active else "transparent"
        self.setStyleSheet(
            f"QFrame#ImportDrop{{background:{bg};border:1px dashed {border};"
            f"border-radius:12px;}}")

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose a folder to import")
        if d:
            self._on_paths([d])

    def _pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose images to import", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.npy *.nd2 *.czi);;"
            "All files (*)")
        if paths:
            self._on_paths(paths)

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            self._style(True)
            e.acceptProposedAction()

    def dragLeaveEvent(self, e) -> None:
        self._style(False)

    def dropEvent(self, e) -> None:
        self._style(False)
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile()]
        if paths:
            self._on_paths(paths)


# ── create-dataset modal (build from a project OR import from disk) ───────────
class NewDatasetDialog(QWidget):
    """Create a dataset two ways — curate a project's segmented images, or
    **import image+mask pairs from disk** — over a scrim, like
    ``NewProjectDialog``. A compact, fixed-height panel (never stretches to the
    window: the panel's vertical policy is ``Maximum``)."""

    def __init__(self, parent: QWidget, t: dict, datasets: DatasetController,
                 projects: ProjectController, *,
                 on_toast: Callable[[str, str], None],
                 on_built: Callable[[DatasetInfo], None]):
        super().__init__(parent)
        self._t = t
        self._datasets = datasets
        self._projects = projects
        self._toast = on_toast
        self._on_built = on_built
        self._mode = "project"                       # "project" | "import"
        # project-mode state
        self._project: Optional[Project] = None
        self._candidates: list[BuildCandidate] = []
        self._selected: set[str] = set()
        # import-mode state
        self._scan: Optional[ImportScan] = None
        # shared
        self._include_measurements = True
        self._val_pct = 0
        self._name_input: Optional[QLineEdit] = None
        self._name_override: Optional[str] = None

        self.setStyleSheet(f"background:{theme.SCRIM};")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 56, 0, 40)
        outer.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self._build_panel())
        self.hide()

    def _build_panel(self) -> QFrame:
        t = self._t
        panel = QFrame()
        panel.setFixedWidth(560)
        # Maximum vertical policy is the fix for the "huge gaps" bug: without it
        # the fullscreen scrim's layout stretched the panel to the window height
        # and the inner QVBoxLayout distributed the slack into gaps + ballooned
        # the SelectBoxes. Now the panel sits at its own sizeHint.
        panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setObjectName("NewDatasetPanel")
        panel.setStyleSheet(
            f"QFrame#NewDatasetPanel{{background:{t['surface']};border:1px solid "
            f"{t['border_strong']};border-radius:14px;}}")
        soft_shadow(panel, 28, 40, 10)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # header
        header = bare_widget()
        hrow = QHBoxLayout(header)
        hrow.setContentsMargins(22, 15, 14, 14)
        hcol = QVBoxLayout()
        hcol.setSpacing(1)
        hcol.addWidget(label("New dataset", 15.5, t["text"], 600))
        hcol.addWidget(label("Build from a project, or import your own from disk.",
                             11, t["text_muted"]))
        hrow.addLayout(hcol)
        hrow.addStretch(1)
        hrow.addWidget(IconButton("close", t, 27, "Close", self.hide))
        v.addWidget(header)
        v.addWidget(hline(t))

        # mode switch
        modewrap = bare_widget()
        mrow = QHBoxLayout(modewrap)
        mrow.setContentsMargins(22, 12, 22, 2)
        self._mode_seg = SegControl(["From a project", "Import from disk"], t, active=0)
        self._mode_seg.changed.connect(self._on_mode_changed)
        mrow.addWidget(self._mode_seg)
        v.addWidget(modewrap)

        # mode-specific + shared body
        body = bare_widget()
        self._body = QVBoxLayout(body)
        self._body.setContentsMargins(22, 12, 22, 8)
        self._body.setSpacing(11)
        v.addWidget(body)

        v.addWidget(hline(t))
        footer = bare_widget()
        frow = QHBoxLayout(footer)
        frow.setContentsMargins(22, 11, 20, 13)
        self._summary = label("", 11.5, t["text_muted"])
        frow.addWidget(self._summary)
        frow.addStretch(1)
        cancel = PillButton("Cancel", t, "ghost", small=True)
        cancel.clicked.connect(self.hide)
        self._create_btn = PillButton("Create dataset", t, "primary", small=True)
        self._create_btn.clicked.connect(self._create)
        frow.addWidget(cancel)
        frow.addWidget(self._create_btn)
        v.addWidget(footer)
        self._panel = panel
        return panel

    # ── render ────────────────────────────────────────────────────────────────
    def _render(self) -> None:
        _clear_layout(self._body)
        if self._mode == "project":
            self._render_project_mode()
        else:
            self._render_import_mode()
        self._render_name_and_options()
        self._update_summary()

    def _section_label(self, text: str) -> None:
        self._body.addWidget(GroupLabel(text, self._t))

    def _list_scroll(self, rows: list[QWidget], row_h: int = 44,
                     max_h: int = 196) -> QWidget:
        t = self._t
        frame = QFrame()
        frame.setObjectName("DSChecklist")
        frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        frame.setStyleSheet(
            f"QFrame#DSChecklist{{background:{t['inset']};border:1px solid "
            f"{t['border']};border-radius:10px;}}")
        lv = QVBoxLayout(frame)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.setSpacing(2)
        for r in rows:
            lv.addWidget(r)
        wrapped = scroll(frame)
        wrapped.setFixedHeight(min(max_h, row_h * max(1, len(rows)) + 14))
        wrapped.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        return wrapped

    # ── project mode ──────────────────────────────────────────────────────────
    def _render_project_mode(self) -> None:
        t = self._t
        eligible = self._eligible_projects()
        if not eligible:
            self._body.addWidget(label(
                "No project has segmented images yet — segment some first, or "
                "import a dataset you already have:", 12.5, t["text_muted"]))
            switch = PillButton("Import from disk instead", t, "ghost", "download",
                                small=True)
            switch.clicked.connect(lambda: self._set_mode("import"))
            row = QHBoxLayout()
            row.addWidget(switch)
            row.addStretch(1)
            self._body.addLayout(row)
            return

        self._section_label("SOURCE PROJECT")
        names = [p.name for p in eligible]
        cur = self._project.name if self._project else names[0]
        picker = SelectBox(cur, t, lead_icon="projects", options=names,
                           on_select=self._on_project_selected)
        picker.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._body.addWidget(picker)

        seg = [c for c in self._candidates if c.segmented]
        head = QHBoxLayout()
        head.addWidget(label(f"{len(seg)} segmented · {len(self._selected)} selected",
                             11.5, t["text_muted"]))
        head.addStretch(1)
        all_btn = PillButton("Select all", t, "ghost", small=True)
        all_btn.clicked.connect(self._select_all)
        none_btn = PillButton("Clear", t, "ghost", small=True)
        none_btn.clicked.connect(self._clear_all)
        head.addWidget(all_btn)
        head.addWidget(none_btn)
        self._section_label("IMAGES TO INCLUDE")
        self._body.addLayout(head)
        rows = [self._candidate_row(c)
                for c in seg + [c for c in self._candidates if not c.segmented]]
        self._body.addWidget(self._list_scroll(rows))

    def _candidate_row(self, c: BuildCandidate) -> QWidget:
        t = self._t
        row = bare_widget()
        h = QHBoxLayout(row)
        h.setContentsMargins(8, 5, 8, 5)
        h.setSpacing(9)
        selected = c.image_path in self._selected
        check = IconButton("check_square" if selected else "square", t, 26,
                           "Toggle", None)
        if c.segmented:
            check.clicked.connect(lambda _=False, p=c.image_path: self._toggle(p))
        else:
            check.setEnabled(False)
        h.addWidget(check)
        h.addWidget(label(c.name, 12.5, t["text"] if c.segmented else t["text_muted"]))
        h.addStretch(1)
        if c.has_gt:
            h.addWidget(Badge("GT", t))
        if c.segmented:
            h.addWidget(label(f"{c.cells} cells", 11, t["text_muted"]))
        else:
            h.addWidget(Chip("not segmented", t, "default"))
        return row

    # ── import mode ───────────────────────────────────────────────────────────
    def _render_import_mode(self) -> None:
        t = self._t
        self._body.addWidget(_ImportDropZone(t, self._on_import_paths))
        if self._scan is None:
            return
        s = self._scan
        self._section_label("FOUND")
        if s.is_velum_dataset:
            total_cells = sum(c.cells for c in s.candidates)
            summary = (f"Velum dataset · {s.n_images} image"
                       f"{'' if s.n_images == 1 else 's'} · {total_cells} cells")
        else:
            skipped = s.n_images - s.n_with_mask
            summary = (f"{s.source_label} · {s.n_with_mask} with masks"
                       + (f" · {skipped} skipped (no mask)" if skipped else ""))
        self._body.addWidget(label(summary, 11.5, t["text_muted"]))
        rows = [self._import_row(c) for c in s.candidates]
        if rows:
            self._body.addWidget(self._list_scroll(rows))

    def _import_row(self, c) -> QWidget:
        t = self._t
        row = bare_widget()
        h = QHBoxLayout(row)
        h.setContentsMargins(8, 5, 8, 5)
        h.setSpacing(9)
        has = c.mask_path is not None
        check = IconButton("check_square" if has else "square", t, 26, "", None)
        check.setEnabled(False)
        h.addWidget(check)
        h.addWidget(label(c.name, 12.5, t["text"] if has else t["text_muted"]))
        h.addStretch(1)
        if has:
            h.addWidget(label(f"{c.cells} cells", 11, t["text_muted"]))
        else:
            h.addWidget(Chip("no mask", t, "default"))
        return row

    def _on_import_paths(self, paths: list[str]) -> None:
        try:
            self._scan = self._datasets.scan_import(paths)
        except Exception as exc:   # noqa: BLE001
            self._toast("Couldn't read that", str(exc))
            return
        self._name_override = None    # let the name default to the new source
        self._render()

    # ── shared: name + options ────────────────────────────────────────────────
    def _render_name_and_options(self) -> None:
        t = self._t
        self._section_label("NAME")
        self._name_input = QLineEdit(self._default_name())
        self._name_input.setPlaceholderText("Dataset name")
        self._name_input.textEdited.connect(self._set_name_override)
        self._body.addWidget(self._name_input)

        meas = QHBoxLayout()
        meas.setSpacing(10)
        toggle = Toggle(t, on=self._include_measurements)
        toggle.toggled.connect(self._set_measurements)
        meas.addWidget(toggle)
        meas.addWidget(label("Include per-cell measurements", 12.5, t["text"]))
        meas.addStretch(1)
        self._body.addLayout(meas)

        val = QHBoxLayout()
        val.setSpacing(10)
        val.addWidget(label("Validation split", 12.5, t["text"]))
        val.addStretch(1)
        opts = ["0%", "10%", "20%", "30%"]
        seg = SegControl(opts, t, active=opts.index(f"{self._val_pct}%")
                         if f"{self._val_pct}%" in opts else 0, compact=True)
        seg.changed.connect(self._set_val)
        val.addWidget(seg)
        self._body.addLayout(val)

    # ── state ────────────────────────────────────────────────────────────────
    def _eligible_projects(self) -> list[Project]:
        return [p for p in self._projects.list_projects()
                if self._datasets.segmented_count(p) > 0]

    def _select_project(self, project: Project) -> None:
        self._project = project
        self._candidates = self._datasets.build_candidates(project)
        self._selected = {c.image_path for c in self._candidates if c.segmented}

    def _on_project_selected(self, name: str) -> None:
        for p in self._eligible_projects():
            if p.name == name:
                self._select_project(p)
                break
        self._name_override = None
        self._render()

    def _on_mode_changed(self, idx: int) -> None:
        self._mode = "project" if idx == 0 else "import"
        self._name_override = None
        self._render()

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._mode_seg.blockSignals(True)
        self._mode_seg._select(0 if mode == "project" else 1)
        self._mode_seg.blockSignals(False)
        self._name_override = None
        self._render()

    def _toggle(self, path: str) -> None:
        self._selected.symmetric_difference_update({path})
        self._render()

    def _select_all(self) -> None:
        self._selected = {c.image_path for c in self._candidates if c.segmented}
        self._render()

    def _clear_all(self) -> None:
        self._selected = set()
        self._render()

    def _set_measurements(self, on: bool) -> None:
        self._include_measurements = on

    def _set_val(self, idx: int) -> None:
        self._val_pct = [0, 10, 20, 30][idx] if 0 <= idx < 4 else 0
        self._update_summary()

    def _set_name_override(self, text: str) -> None:
        self._name_override = text

    def _default_name(self) -> str:
        if self._name_override:
            return self._name_override
        if self._mode == "project":
            base = self._project.name if self._project else "Dataset"
            return f"{base} dataset"
        if self._scan is not None:
            if self._scan.is_velum_dataset:
                return f"{self._scan.source_label.split('· ', 1)[-1]} (copy)"
            return f"{self._scan.source_label} dataset"
        return "Imported dataset"

    def _ready_count(self) -> int:
        if self._mode == "project":
            return len(self._selected)
        return self._scan.n_with_mask if self._scan else 0

    def _update_summary(self) -> None:
        n = self._ready_count()
        verb = "selected" if self._mode == "project" else "ready"
        self._summary.setText(f"{n} image{'' if n == 1 else 's'} {verb}")
        self._create_btn.setEnabled(n > 0)

    # ── create ───────────────────────────────────────────────────────────────
    def _create(self) -> None:
        name = (self._name_input.text().strip() if self._name_input else "") \
            or self._default_name()
        try:
            if self._mode == "project":
                if self._project is None or not self._selected:
                    return
                info = self._datasets.build_from_project(
                    self._project, sorted(self._selected), name=name,
                    include_measurements=self._include_measurements,
                    val_fraction=self._val_pct / 100.0)
            else:
                if self._scan is None or self._scan.n_with_mask == 0:
                    return
                info = self._datasets.import_as_dataset(
                    self._scan, name=name,
                    include_measurements=self._include_measurements,
                    val_fraction=self._val_pct / 100.0)
        except ValueError as exc:
            self._toast("Nothing to create", str(exc))
            return
        except Exception as exc:   # noqa: BLE001
            self._toast("Failed", str(exc))
            return
        self.hide()
        self._on_built(info)

    # ── open / place (mirrors NewProjectDialog) ──────────────────────────────
    def open(self) -> None:
        self._scan = None
        self._name_override = None
        eligible = self._eligible_projects()
        active = self._projects.get_active()
        if active is not None and self._datasets.segmented_count(active) > 0:
            self._select_project(active)
        elif eligible:
            self._select_project(eligible[0])
        else:
            self._project = None
            self._candidates = []
            self._selected = set()
        # Default to import mode when there's nothing to curate — so "bring your
        # own dataset" is the first thing offered, not a dead end.
        self._set_mode("project" if eligible else "import")
        self.place()
        self.show()
        self.raise_()

    def place(self) -> None:
        p = self.parentWidget()
        if p:
            self.setGeometry(0, 0, p.width(), p.height())

    def mousePressEvent(self, e) -> None:
        if self.childAt(e.position().toPoint()) is None:
            self.hide()
        super().mousePressEvent(e)
