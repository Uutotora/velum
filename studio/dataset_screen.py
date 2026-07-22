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
    QFrame, QGridLayout, QHBoxLayout, QLineEdit, QVBoxLayout, QWidget,
)

from studio import icons, theme
from studio.components import (
    Badge, Chip, EngineChip, GroupLabel, IconButton, PillButton, SelectBox,
    StatTile, Toggle, bare_widget, hline, label, soft_shadow,
)
from studio.dataset import DatasetInfo
from studio.dataset_controller import BuildCandidate, DatasetController
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


# ── interactive build modal ──────────────────────────────────────────────────
class NewDatasetDialog(QWidget):
    """Pick a project, tick which segmented images to include, name it, set an
    optional val split, and build — over a scrim, like ``NewProjectDialog``."""

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
        self._project: Optional[Project] = None
        self._candidates: list[BuildCandidate] = []
        self._selected: set[str] = set()
        self._include_measurements = True
        self._val_pct = 0

        self.setStyleSheet(f"background:{theme.SCRIM};")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 64, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self._build_panel())
        self.hide()

    def _build_panel(self) -> QFrame:
        t = self._t
        panel = QFrame()
        panel.setFixedWidth(520)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setObjectName("NewDatasetPanel")
        panel.setStyleSheet(
            f"QFrame#NewDatasetPanel{{background:{t['surface']};border:1px solid "
            f"{t['border_strong']};border-radius:14px;}}")
        soft_shadow(panel, 28, 40, 10)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        header = bare_widget()
        hrow = QHBoxLayout(header)
        hrow.setContentsMargins(20, 16, 14, 16)
        hcol = QVBoxLayout()
        hcol.setSpacing(2)
        hcol.addWidget(label("New dataset", 15, t["text"], 600))
        hcol.addWidget(label("Curate a project's segmented images.", 11, t["text_muted"]))
        hrow.addLayout(hcol)
        hrow.addStretch(1)
        hrow.addWidget(IconButton("close", t, 27, "Close", self.hide))
        v.addWidget(header)
        v.addWidget(hline(t))

        body = bare_widget()
        self._body = QVBoxLayout(body)
        self._body.setContentsMargins(22, 18, 22, 8)
        self._body.setSpacing(14)
        v.addWidget(body)

        v.addWidget(hline(t))
        footer = bare_widget()
        frow = QHBoxLayout(footer)
        frow.setContentsMargins(20, 12, 20, 14)
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

    # ── build the body for the currently-selected project ────────────────────
    def _render(self) -> None:
        t = self._t
        _clear_layout(self._body)

        eligible = self._eligible_projects()
        if not eligible:
            self._body.addWidget(label(
                "No project has segmented images yet. Segment some images in a "
                "project first, then build a dataset from them.",
                12.5, t["text_muted"]))
            self._name_input = None
            self._create_btn.setEnabled(False)
            self._summary.setText("")
            return

        # Project picker
        picker = bare_widget()
        pv = QVBoxLayout(picker)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(7)
        pv.addWidget(GroupLabel("SOURCE PROJECT", t))
        names = [p.name for p in eligible]
        cur = self._project.name if self._project else names[0]
        pv.addWidget(SelectBox(cur, t, lead_icon="projects", options=names,
                               on_select=self._on_project_selected))
        self._body.addWidget(picker)

        # Candidate checklist
        self._body.addWidget(GroupLabel("IMAGES TO INCLUDE", t))
        seg = [c for c in self._candidates if c.segmented]
        others = [c for c in self._candidates if not c.segmented]
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
        self._body.addLayout(head)

        list_frame = QFrame()
        list_frame.setObjectName("DSChecklist")
        list_frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        list_frame.setStyleSheet(
            f"QFrame#DSChecklist{{background:{t['inset']};border:1px solid {t['border']};"
            f"border-radius:10px;}}")
        lv = QVBoxLayout(list_frame)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.setSpacing(2)
        for c in seg + others:
            lv.addWidget(self._candidate_row(c))
        wrapped = scroll(list_frame)
        wrapped.setFixedHeight(min(232, 46 * max(1, len(self._candidates)) + 14))
        self._body.addWidget(wrapped)

        # Name + options
        self._name_input = QLineEdit(self._default_name())
        self._name_input.setPlaceholderText("Dataset name")
        name_field = bare_widget()
        nf = QVBoxLayout(name_field)
        nf.setContentsMargins(0, 0, 0, 0)
        nf.setSpacing(7)
        nf.addWidget(GroupLabel("NAME", t))
        nf.addWidget(self._name_input)
        self._body.addWidget(name_field)

        opts = QHBoxLayout()
        opts.setSpacing(12)
        meas_toggle = Toggle(t, on=self._include_measurements)
        meas_toggle.toggled.connect(self._set_measurements)
        opts.addWidget(label("Include per-cell measurements", 12.5, t["text"]))
        opts.addWidget(meas_toggle)
        opts.addStretch(1)
        opts.addWidget(label("Val split", 12.5, t["text"]))
        val_box = SelectBox(f"{self._val_pct}%", t,
                            options=["0%", "10%", "20%", "30%"],
                            on_select=self._set_val)
        val_box.setFixedWidth(88)     # SelectBox has no width of its own; don't let it get crushed
        opts.addWidget(val_box)
        self._body.addLayout(opts)
        self._update_summary()

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
        name = label(c.name, 12.5, t["text"] if c.segmented else t["text_muted"])
        h.addWidget(name)
        h.addStretch(1)
        if c.has_gt:
            h.addWidget(Badge("GT", t))
        if c.segmented:
            h.addWidget(label(f"{c.cells} cells", 11, t["text_muted"]))
        else:
            h.addWidget(Chip("not segmented", t, "default"))
        return row

    # ── state ────────────────────────────────────────────────────────────────
    def _eligible_projects(self) -> list[Project]:
        out = []
        for p in self._projects.list_projects():
            if self._datasets.segmented_count(p) > 0:
                out.append(p)
        return out

    def _select_project(self, project: Project) -> None:
        self._project = project
        self._candidates = self._datasets.build_candidates(project)
        self._selected = {c.image_path for c in self._candidates if c.segmented}

    def _on_project_selected(self, name: str) -> None:
        for p in self._eligible_projects():
            if p.name == name:
                self._select_project(p)
                break
        self._render()

    def _toggle(self, path: str) -> None:
        if path in self._selected:
            self._selected.discard(path)
        else:
            self._selected.add(path)
        self._render()

    def _select_all(self) -> None:
        self._selected = {c.image_path for c in self._candidates if c.segmented}
        self._render()

    def _clear_all(self) -> None:
        self._selected = set()
        self._render()

    def _set_measurements(self, on: bool) -> None:
        self._include_measurements = on

    def _set_val(self, text: str) -> None:
        self._val_pct = int(text.rstrip("%") or "0")
        self._update_summary()

    def _default_name(self) -> str:
        base = self._project.name if self._project else "Dataset"
        return f"{base} dataset"

    def _update_summary(self) -> None:
        n = len(self._selected)
        self._summary.setText(f"{n} image{'s' if n != 1 else ''} selected")
        self._create_btn.setEnabled(n > 0)

    # ── create ───────────────────────────────────────────────────────────────
    def _create(self) -> None:
        if self._project is None or not self._selected:
            return
        name = (self._name_input.text().strip() if self._name_input else "") \
            or self._default_name()
        try:
            info = self._datasets.build_from_project(
                self._project, sorted(self._selected), name=name,
                include_measurements=self._include_measurements,
                val_fraction=self._val_pct / 100.0)
        except ValueError as exc:
            self._toast("Nothing to build", str(exc))
            return
        except Exception as exc:   # noqa: BLE001
            self._toast("Build failed", str(exc))
            return
        self.hide()
        self._on_built(info)

    # ── open / place (mirrors NewProjectDialog) ──────────────────────────────
    def open(self) -> None:
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
        self._render()
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
