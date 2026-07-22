"""Velum — the "New Project" creation flow.

A frameless, scrim-backed modal (same construction as ``overlays.
CommandPalette``: full-window overlay, click-outside-to-close, Escape via the
window's shortcut) implementing the Label Studio 3-step pattern
``docs/velum/BACKLOG.md`` calls for: name & description -> import images ->
pick an engine. The final step writes through the real ``ProjectStore``
(``studio/project.py``). Image files are only ever collected as paths (drag
&drop or a native file picker) — never opened, decoded or previewed — so this
module stays exactly as dependency-free as the rest of the design skeleton
(no napari, no torch, no ``data.utils``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QLineEdit, QFileDialog,
)

from studio import icons, theme
from studio.components import (
    PillButton, IconButton, SegControl, GroupLabel, hline, label, soft_shadow,
)
from studio.project import (
    ENGINES, ENGINE_LABELS, IMAGE_FILE_FILTER, ProjectSettings, ProjectStore,
    is_supported_image_path,
)

STEP_TITLES = ["Name your project", "Import images", "Choose an engine"]

_ENGINE_BLURB = {
    "cellseg1": "One-shot LoRA fine-tuning from a single annotated image — best "
                "with one representative image to train on.",
    "cellpose": "Zero-shot generalist. Strong out-of-the-box accuracy, no "
                "training or checkpoint required.",
    "sam2": "Zero-shot, the flagship choice for z-stacks and time-lapse "
            "sequences.",
}


def _field(title: str, control: QWidget, t: dict) -> QWidget:
    w = QWidget()
    w.setStyleSheet("background:transparent;")
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(7)
    v.addWidget(GroupLabel(title, t))
    v.addWidget(control)
    return w


class _DropZone(QFrame):
    """Drag-and-drop target + a native file-picker fallback."""

    def __init__(self, t: dict, on_files: Callable[[list[str]], None]):
        super().__init__()
        self._t = t
        self._on_files = on_files
        self.setAcceptDrops(True)
        self.setMinimumHeight(112)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._set_active(False)
        v = QVBoxLayout(self)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.setSpacing(7)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("image", t["text_muted"], 22))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(ic)
        head = label("Drag & drop images here", 12.5, t["text_subtle"], 600)
        head.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(head)
        sub = label("TIFF · OME-TIFF · ND2 · CZI · LIF · PNG", 11, t["text_muted"])
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(sub)
        browse = PillButton("Browse files…", t, "ghost", small=True)
        browse.clicked.connect(self._browse)
        v.addWidget(browse, alignment=Qt.AlignmentFlag.AlignCenter)

    def _set_active(self, active: bool) -> None:
        t = self._t
        border = t["primary_line"] if active else t["border_strong"]
        bg = t["primary_weak"] if active else "transparent"
        # Qualified -- see components.EngineChip's comment. The icon/
        # heading/subtitle QLabels above don't set their own `border`, so
        # an unscoped QFrame{...} rule here leaks this zone's dashed border
        # onto each of them individually.
        self.setObjectName("DropZone")
        self.setStyleSheet(
            f"QFrame#DropZone{{background:{bg}; border:1px dashed {border}; border-radius:14px;}}")

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import images", "",
            IMAGE_FILE_FILTER)
        if paths:
            self._on_files(paths)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            self._set_active(True)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self._set_active(False)

    def dropEvent(self, event) -> None:
        self._set_active(False)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self._on_files(paths)
        event.acceptProposedAction()


class NewProjectDialog(QWidget):
    """3-step "create a project" modal, centred over a scrim."""

    def __init__(self, parent: QWidget, t: dict, store: ProjectStore,
                 on_created: Optional[Callable[[str], None]] = None):
        super().__init__(parent)
        self._t = t
        self._store = store
        self._on_created = on_created
        self._reset()

        self.setStyleSheet(f"background:{theme.SCRIM};")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 80, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self._build_panel())
        self.hide()

    # ── construction ─────────────────────────────────────────────────────────
    def _build_panel(self) -> QFrame:
        t = self._t
        panel = QFrame()
        panel.setFixedWidth(480)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified -- see components.EngineChip's comment. This is the
        # *whole dialog body's* container: fully unqualified here means
        # every QLabel anywhere inside the panel (every field caption, step
        # title, hint) inherits this panel's own border-radius-14 box
        # around just its own text -- the most visible instance of this bug
        # class found in the whole app, since it silently double-boxes
        # nearly every line of text in the New Project flow.
        panel.setObjectName("NewProjectPanel")
        panel.setStyleSheet(
            f"QFrame#NewProjectPanel{{background:{t['surface']}; border:1px solid {t['border_strong']};"
            f" border-radius:14px;}}")
        soft_shadow(panel, 28, 40, 10)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._header())
        v.addWidget(hline(t))
        body_wrap = QWidget()
        body_wrap.setStyleSheet("background:transparent;")
        self._body = QVBoxLayout(body_wrap)
        self._body.setContentsMargins(24, 22, 24, 4)
        v.addWidget(body_wrap)
        v.addWidget(hline(t))
        v.addWidget(self._footer())
        self._panel = panel
        self._render_step()
        return panel

    def _header(self) -> QWidget:
        t = self._t
        h = QWidget()
        h.setStyleSheet("background:transparent;")
        row = QHBoxLayout(h)
        row.setContentsMargins(20, 16, 14, 16)
        row.setSpacing(10)
        col = QVBoxLayout()
        col.setSpacing(2)
        self._title_lbl = label(STEP_TITLES[0], 15, t["text"], 600)
        col.addWidget(self._title_lbl)
        self._step_lbl = label("", 11, t["text_muted"], 600)
        col.addWidget(self._step_lbl)
        row.addLayout(col)
        row.addStretch(1)
        row.addWidget(IconButton("close", t, 27, "Close", self.hide))
        return h

    def _footer(self) -> QWidget:
        t = self._t
        f = QWidget()
        f.setStyleSheet("background:transparent;")
        row = QHBoxLayout(f)
        row.setContentsMargins(20, 14, 20, 16)
        row.setSpacing(10)
        self._back_btn = PillButton("Back", t, "ghost", small=True)
        self._back_btn.clicked.connect(self._go_back)
        row.addWidget(self._back_btn)
        row.addStretch(1)
        self._next_btn = PillButton("Next", t, "primary", small=True)
        self._next_btn.clicked.connect(self._go_next)
        row.addWidget(self._next_btn)
        return f

    # ── steps ────────────────────────────────────────────────────────────────
    def _step_name(self) -> QWidget:
        t = self._t
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(16)
        name_input = QLineEdit(self._name)
        name_input.setPlaceholderText("e.g. Fluorescence Nuclei — DAPI")
        name_input.textChanged.connect(self._set_name)
        v.addWidget(_field("PROJECT NAME", name_input, t))
        desc_input = QLineEdit(self._description)
        desc_input.setPlaceholderText("What is this cohort? (optional)")
        desc_input.textChanged.connect(self._set_description)
        v.addWidget(_field("DESCRIPTION", desc_input, t))
        return w

    def _step_import(self) -> QWidget:
        t = self._t
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)
        v.addWidget(_DropZone(t, self._add_files))
        if self._image_paths:
            n = len(self._image_paths)
            v.addWidget(label(f"{n} image{'s' if n != 1 else ''} selected", 12, t["text"], 600))
            shown = self._image_paths[:5]
            for i, path in enumerate(shown):
                v.addWidget(self._file_row(i, path))
            if n > len(shown):
                v.addWidget(label(f"+ {n - len(shown)} more", 11, t["text_muted"]))
        else:
            v.addWidget(label("Optional — you can add images later.", 11.5, t["text_muted"]))
        return w

    def _file_row(self, idx: int, path: str) -> QWidget:
        t = self._t
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        name = label(Path(path).name, 12, t["text_subtle"])
        name.setStyleSheet(name.styleSheet() + f"font-family:{theme.MONO};")
        h.addWidget(name, 1)
        h.addWidget(IconButton("close", t, 22, "Remove", on_click=lambda i=idx: self._remove_file(i)))
        return row

    def _step_engine(self) -> QWidget:
        t = self._t
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        seg = SegControl([ENGINE_LABELS[e] for e in ENGINES], t, active=self._engine_idx)
        seg.changed.connect(self._set_engine)
        v.addWidget(seg)
        desc = label(_ENGINE_BLURB[ENGINES[self._engine_idx]], 12.5, t["text_muted"])
        desc.setWordWrap(True)
        v.addWidget(desc)
        return w

    # ── step state ───────────────────────────────────────────────────────────
    def _set_name(self, text: str) -> None:
        self._name = text
        self._next_btn.setEnabled(bool(text.strip()))

    def _set_description(self, text: str) -> None:
        self._description = text

    def _add_files(self, paths: list[str]) -> None:
        for p in paths:
            if p not in self._image_paths and is_supported_image_path(p):
                self._image_paths.append(p)
        self._render_step()

    def _remove_file(self, idx: int) -> None:
        if 0 <= idx < len(self._image_paths):
            del self._image_paths[idx]
        self._render_step()

    def _set_engine(self, idx: int) -> None:
        self._engine_idx = idx
        self._render_step()

    def _clear_body(self) -> None:
        while self._body.count():
            item = self._body.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _render_step(self) -> None:
        self._title_lbl.setText(STEP_TITLES[self._step])
        self._step_lbl.setText(f"Step {self._step + 1} of {len(STEP_TITLES)}")
        self._back_btn.setVisible(self._step > 0)
        self._next_btn.setText("Create Project" if self._step == len(STEP_TITLES) - 1 else "Next")
        self._clear_body()
        step_widget = (self._step_name, self._step_import, self._step_engine)[self._step]()
        self._body.addWidget(step_widget)
        self._next_btn.setEnabled(bool(self._name.strip()) if self._step == 0 else True)

    # ── navigation ───────────────────────────────────────────────────────────
    def _go_back(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._render_step()

    def _go_next(self) -> None:
        if self._step == 0 and not self._name.strip():
            return  # the Next button is disabled for this case too; belt and braces
        if self._step < len(STEP_TITLES) - 1:
            self._step += 1
            self._render_step()
        else:
            self._create()

    def _create(self) -> None:
        name = self._name.strip()
        if not name:
            self._step = 0
            self._render_step()
            return
        engine = ENGINES[self._engine_idx]
        project = self._store.create(
            name, description=self._description.strip(),
            settings=ProjectSettings(engine=engine),
        )
        # Copy the chosen images into the project so they keep opening after
        # the source moves and aren't blocked by macOS's per-folder privacy
        # gate (see ProjectStore.image_dir) -- done after create() because the
        # copy target is keyed by the new project's id.
        project.image_paths = self._store.import_images(project.id, self._image_paths)
        project.stats.n_images = len(project.image_paths)
        self._store.save(project)
        self.hide()
        if self._on_created:
            self._on_created(project.id)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _reset(self) -> None:
        self._step = 0
        self._name = ""
        self._description = ""
        self._image_paths: list[str] = []
        self._engine_idx = 0

    def open(self) -> None:
        self._reset()
        self._render_step()
        self.place()
        self.show()
        self.raise_()

    def place(self) -> None:
        p = self.parentWidget()
        if p:
            self.setGeometry(0, 0, p.width(), p.height())

    def mousePressEvent(self, e) -> None:
        child = self.childAt(e.position().toPoint())
        if child is None:
            self.hide()
        super().mousePressEvent(e)
