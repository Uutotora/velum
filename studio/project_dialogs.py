"""CellSeg1 Studio — the Projects tab's small modals: a generic confirm
dialog, a rename dialog, and the Trash view.

All scrim-backed, centred panels -- the same construction
``new_project_dialog.NewProjectDialog`` and ``overlays.CommandPalette``
already use (full-window overlay, click-outside-to-close). Kept in their own
module, mirroring how ``new_project_dialog.py`` got its own file rather than
living inside ``screens.py``, since ``ProjectsScreen`` is already large.
"""
from __future__ import annotations

import html
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QFrame, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit

from studio.components import IconButton, PillButton, SmoothScrollArea, hline, label, soft_shadow
from studio.project_controller import ProjectController, relative_time


class ConfirmDialog(QWidget):
    """A scrim-backed confirm/cancel modal for a single, scoped action --
    e.g. "Move to Trash". Not for a truly irreversible action (permanent
    delete already carries its own extra confirm inside ``TrashDialog``) --
    friction should match blast radius, and a reversible soft-delete with an
    Undo toast doesn't need type-to-confirm ceremony on top.

    Constructed fresh per use (mirrors ``ProjectsScreen._open_filter_menu``'s
    "build a QMenu on demand" pattern rather than one persistent, content-
    swapping instance) and disposes itself on hide, whichever way it closes.
    """

    def __init__(self, parent: QWidget, t: dict, title: str, body: str,
                 confirm_label: str = "Confirm", confirm_kind: str = "danger",
                 on_confirm: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self._t = t
        self._on_confirm = on_confirm
        self.setStyleSheet("background:rgba(8,10,20,0.34);")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 140, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self._build_panel(title, body, confirm_label, confirm_kind))
        self.hide()

    def _build_panel(self, title: str, body: str, confirm_label: str, confirm_kind: str) -> QFrame:
        t = self._t
        panel = QFrame()
        panel.setFixedWidth(400)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified by #ObjectName -- see components.EngineChip's comment for
        # why an unscoped QFrame{...} here would leak this panel's border
        # onto the title/body QLabels below.
        panel.setObjectName("ConfirmPanel")
        panel.setStyleSheet(
            f"QFrame#ConfirmPanel{{background:{t['surface']}; border:1px solid {t['border_strong']};"
            f" border-radius:14px;}}")
        soft_shadow(panel, 28, 40, 10)
        v = QVBoxLayout(panel)
        v.setContentsMargins(22, 20, 22, 18)
        v.setSpacing(10)
        v.addWidget(label(title, 15, t["text"], 600))
        body_lbl = QLabel(body)
        body_lbl.setTextFormat(Qt.TextFormat.RichText)
        body_lbl.setWordWrap(True)
        body_lbl.setStyleSheet(f"color:{t['text_muted']}; font-size:12.5px; background:transparent;")
        v.addWidget(body_lbl)
        v.addSpacing(8)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = PillButton("Cancel", t, "ghost", small=True)
        cancel.clicked.connect(self.hide)
        row.addWidget(cancel)
        confirm = PillButton(confirm_label, t, confirm_kind, small=True)
        confirm.clicked.connect(self._confirm)
        row.addWidget(confirm)
        v.addLayout(row)
        return panel

    def _confirm(self) -> None:
        self.hide()
        if self._on_confirm:
            self._on_confirm()

    def place(self) -> None:
        p = self.parentWidget()
        if p:
            self.setGeometry(0, 0, p.width(), p.height())

    def open(self) -> None:
        self.place()
        self.show()
        self.raise_()

    def mousePressEvent(self, e) -> None:
        child = self.childAt(e.position().toPoint())
        if child is None:
            self.hide()
        super().mousePressEvent(e)

    def hideEvent(self, e) -> None:
        super().hideEvent(e)
        self.deleteLater()  # throwaway, one-shot -- see the class docstring


def confirm_trash(parent: QWidget, t: dict, project_name: str,
                   on_confirm: Callable[[], None]) -> ConfirmDialog:
    """Build + open the "Move to Trash?" confirmation for ``project_name``."""
    safe_name = html.escape(project_name)
    dlg = ConfirmDialog(
        parent, t, "Move to Trash?",
        f"<b>{safe_name}</b> will move to Trash. You can restore it later, "
        f"or delete it permanently from there.",
        confirm_label="Move to Trash", confirm_kind="danger", on_confirm=on_confirm)
    dlg.open()
    return dlg


def confirm_delete_forever(parent: QWidget, t: dict, project_name: str,
                            on_confirm: Callable[[], None]) -> ConfirmDialog:
    """Build + open the "Delete Forever?" confirmation -- the one genuinely
    irreversible action in this flow (unlike trashing)."""
    safe_name = html.escape(project_name)
    dlg = ConfirmDialog(
        parent, t, "Delete Forever?",
        f"<b>{safe_name}</b> and everything in it will be permanently "
        f"deleted. This can't be undone.",
        confirm_label="Delete Forever", confirm_kind="danger", on_confirm=on_confirm)
    dlg.open()
    return dlg


class RenameDialog(QWidget):
    """A scrim-backed rename modal: a QLineEdit pre-filled with the current
    name, Cancel/Save. Same construction (and same fresh-per-use,
    self-disposing lifecycle) as ``ConfirmDialog`` -- see its docstring.
    """

    def __init__(self, parent: QWidget, t: dict, current_name: str,
                 on_save: Optional[Callable[[str], None]] = None):
        super().__init__(parent)
        self._t = t
        self._on_save = on_save
        self.setStyleSheet("background:rgba(8,10,20,0.34);")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 140, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self._build_panel(current_name))
        self.hide()

    def _build_panel(self, current_name: str) -> QFrame:
        t = self._t
        panel = QFrame()
        panel.setFixedWidth(400)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified -- see ConfirmDialog._build_panel's identical comment.
        panel.setObjectName("RenamePanel")
        panel.setStyleSheet(
            f"QFrame#RenamePanel{{background:{t['surface']}; border:1px solid {t['border_strong']};"
            f" border-radius:14px;}}")
        soft_shadow(panel, 28, 40, 10)
        v = QVBoxLayout(panel)
        v.setContentsMargins(22, 20, 22, 18)
        v.setSpacing(10)
        v.addWidget(label("Rename project", 15, t["text"], 600))
        self._input = QLineEdit(current_name)
        self._input.selectAll()
        self._input.returnPressed.connect(self._save)
        v.addWidget(self._input)
        v.addSpacing(4)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = PillButton("Cancel", t, "ghost", small=True)
        cancel.clicked.connect(self.hide)
        row.addWidget(cancel)
        save = PillButton("Save", t, "primary", small=True)
        save.clicked.connect(self._save)
        row.addWidget(save)
        v.addLayout(row)
        return panel

    def _save(self) -> None:
        name = self._input.text().strip()
        self.hide()
        if name and self._on_save:
            self._on_save(name)

    def place(self) -> None:
        p = self.parentWidget()
        if p:
            self.setGeometry(0, 0, p.width(), p.height())

    def open(self) -> None:
        self.place()
        self.show()
        self.raise_()
        self._input.setFocus()

    def mousePressEvent(self, e) -> None:
        child = self.childAt(e.position().toPoint())
        if child is None:
            self.hide()
        super().mousePressEvent(e)

    def hideEvent(self, e) -> None:
        super().hideEvent(e)
        self.deleteLater()  # throwaway, one-shot -- see ConfirmDialog's docstring


class TrashDialog(QWidget):
    """Lists trashed projects with Restore / Delete Forever per row --
    reached from the Projects toolbar's Trash entry point. Same scrim+panel
    construction as ``ConfirmDialog``/``NewProjectDialog``, but built once
    and kept alive across opens (like ``NewProjectDialog``), since its list
    genuinely needs to refresh in place after a Restore/Delete without
    closing the dialog.
    """

    def __init__(self, parent: QWidget, t: dict, controller: ProjectController,
                 on_changed: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self._t = t
        self._controller = controller
        self._on_changed = on_changed
        self.setStyleSheet("background:rgba(8,10,20,0.34);")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 70, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self._build_panel())
        self.hide()

    # ── construction ─────────────────────────────────────────────────────────
    def _build_panel(self) -> QFrame:
        t = self._t
        panel = QFrame()
        panel.setFixedWidth(440)
        panel.setFixedHeight(460)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setObjectName("TrashPanel")
        panel.setStyleSheet(
            f"QFrame#TrashPanel{{background:{t['surface']}; border:1px solid {t['border_strong']};"
            f" border-radius:14px;}}")
        soft_shadow(panel, 28, 40, 10)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._header())
        v.addWidget(hline(t))

        # Both the scroll area and its content widget are explicitly
        # re-pinned to this panel's own `surface` token -- theme.build_qss's
        # app-wide "QScrollArea, QScrollArea > QWidget > QWidget { background:
        # bg }" rule would otherwise apply (a visible seam against this
        # panel's `surface`, worse still under this dialog's translucent
        # scrim: confirmed by pixel-sampling a light-theme screenshot, the
        # scrim-over-bg blend is visibly darker than scrim-over-surface would
        # be). Same fix CommandPalette's `_results_container`/`_results_area`
        # already use for the identical reason -- see overlays.py.
        self._rows_host = QWidget()
        self._rows_host.setStyleSheet(f"background:{t['surface']};")
        self._rows = QVBoxLayout(self._rows_host)
        self._rows.setContentsMargins(16, 14, 16, 14)
        self._rows.setSpacing(8)
        self._rows.addStretch(1)
        scroll = SmoothScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea{{background:{t['surface']}; border:none;}}")
        scroll.setWidget(self._rows_host)
        v.addWidget(scroll, 1)

        self._panel = panel
        self.refresh()
        return panel

    def _header(self) -> QWidget:
        t = self._t
        h = QWidget()
        # A bare QWidget() otherwise inherits the app-wide QWidget{background:
        # <bg>} rule and paints an opaque patch here -- see the scroll-area
        # comment above (_build_panel) for why this is worse than usual under
        # this dialog's translucent scrim, confirmed by screenshot.
        h.setStyleSheet("background:transparent;")
        row = QHBoxLayout(h)
        row.setContentsMargins(20, 16, 14, 16)
        row.setSpacing(10)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(label("Trash", 15, t["text"], 600))
        self._sub_lbl = label("", 11, t["text_muted"], 600)
        col.addWidget(self._sub_lbl)
        row.addLayout(col)
        row.addStretch(1)
        row.addWidget(IconButton("close", t, 27, "Close", self.hide))
        return h

    # ── data ─────────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Rebuild the row list from the controller's current trash state."""
        t = self._t
        while self._rows.count():
            item = self._rows.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        trashed = self._controller.list_trashed()
        n = len(trashed)
        self._sub_lbl.setText(f"{n} project{'s' if n != 1 else ''}")
        if not trashed:
            empty = label("Trash is empty.", 12.5, t["text_muted"])
            self._rows.addWidget(empty)
        else:
            for p in trashed:
                self._rows.addWidget(self._row(p))
        self._rows.addStretch(1)

    def _row(self, p) -> QFrame:
        t = self._t
        row = QFrame()
        row.setObjectName("TrashRow")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet(
            f"QFrame#TrashRow{{background:{t['inset']}; border:1px solid {t['border']};"
            f" border-radius:10px;}}")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(10)
        meta = QVBoxLayout()
        meta.setSpacing(2)
        meta.addWidget(label(p.name, 13.5, t["text"], 600))
        when = relative_time(p.trashed_at) if p.trashed_at else ""
        meta.addWidget(label(f"Trashed {when}" if when else "Trashed", 11.5, t["text_muted"]))
        lay.addLayout(meta, 1)

        restore_btn = PillButton("Restore", t, "ghost", small=True)
        restore_btn.clicked.connect(lambda _=False, pid=p.id: self._restore(pid))
        lay.addWidget(restore_btn)

        delete_btn = PillButton("Delete Forever", t, "danger", small=True)
        delete_btn.clicked.connect(lambda _=False, pid=p.id, name=p.name: self._confirm_delete(pid, name))
        lay.addWidget(delete_btn)
        return row

    # ── actions ──────────────────────────────────────────────────────────────
    def _restore(self, project_id: str) -> None:
        self._controller.restore_project(project_id)
        self.refresh()
        if self._on_changed:
            self._on_changed()

    def _confirm_delete(self, project_id: str, project_name: str) -> None:
        # Keep a ref alive while it's open -- see ProjectsScreen._confirm_trash's
        # identical comment; nothing else references this dialog Python-side.
        self._active_dialog = confirm_delete_forever(
            self, self._t, project_name,
            on_confirm=lambda: self._delete_forever(project_id))

    def _delete_forever(self, project_id: str) -> None:
        self._controller.delete_project_permanently(project_id)
        self.refresh()
        if self._on_changed:
            self._on_changed()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def place(self) -> None:
        p = self.parentWidget()
        if p:
            self.setGeometry(0, 0, p.width(), p.height())

    def open(self) -> None:
        self.refresh()
        self.place()
        self.show()
        self.raise_()

    def mousePressEvent(self, e) -> None:
        child = self.childAt(e.position().toPoint())
        if child is None:
            self.hide()
        super().mousePressEvent(e)
