"""Velum — the Projects tab's small modals: a generic confirm
dialog and the Project Settings dialog (General + Danger Zone).

Both are scrim-backed, centred panels -- the same construction
``new_project_dialog.NewProjectDialog`` and ``overlays.CommandPalette``
already use (full-window overlay, click-outside-to-close). Kept in their own
module, mirroring how ``new_project_dialog.py`` got its own file rather than
living inside ``screens.py``, since ``ProjectsScreen`` is already large.

Deletion lives here, in Settings' Danger Zone, not as a quick card-menu
action -- mirrors Label Studio's own project Settings > Danger Zone exactly
(reference screenshots supplied directly by the product owner). An earlier
version of this tab had a soft-delete Trash-with-Undo flow; real usage in
the actual running app (not just offscreen screenshots) surfaced both a
rendering bug in the Trash dialog and, more fundamentally, that it was more
machinery than this product needs -- reverted in favour of this simpler,
Label-Studio-shaped design. See docs/velum/CHANGELOG.md's dated entry for
the full story.
"""
from __future__ import annotations

import html
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QFrame, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit

from studio import theme
from studio.components import GroupLabel, IconButton, PillButton, hline, label, soft_shadow


class ConfirmDialog(QWidget):
    """A scrim-backed confirm/cancel modal for a single action -- e.g.
    "Delete Project?". Constructed fresh per use (mirrors ProjectsScreen.
    _open_filter_menu's "build a QMenu on demand" pattern rather than one
    persistent, content-swapping instance) and disposes itself on hide,
    whichever way it closes.
    """

    def __init__(self, parent: QWidget, t: dict, title: str, body: str,
                 confirm_label: str = "Confirm", confirm_kind: str = "danger",
                 on_confirm: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self._t = t
        self._on_confirm = on_confirm
        self.setStyleSheet(f"background:{theme.SCRIM};")
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


def confirm_delete_project(parent: QWidget, t: dict, project_name: str,
                            on_confirm: Callable[[], None]) -> ConfirmDialog:
    """Build + open the "Delete Project?" confirmation -- the one truly
    irreversible action anywhere in the Projects tab."""
    safe_name = html.escape(project_name)
    dlg = ConfirmDialog(
        parent, t, "Delete Project?",
        f"<b>{safe_name}</b> and everything in it -- images, results, "
        f"settings -- will be permanently deleted. This can't be undone.",
        confirm_label="Delete Project", confirm_kind="danger", on_confirm=on_confirm)
    dlg.open()
    return dlg


class ProjectSettingsDialog(QWidget):
    """The project's own Settings -- General (name, description) and Danger
    Zone (delete) -- reached from the card/row kebab menu's "Settings" item.
    Mirrors Label Studio's own project Settings screen (General Settings /
    Danger Zone, reference screenshots supplied by the product owner)
    collapsed into one compact panel rather than a separate navigated
    screen with its own sidebar -- this project only has two sections worth
    showing, so a second-level nested screen would be more machinery than
    the content needs. Same scrim+panel construction, self-disposing
    lifecycle as ConfirmDialog.
    """

    def __init__(self, parent: QWidget, t: dict, project,
                 on_saved: Optional[Callable[[str, str], None]] = None,
                 on_delete: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self._t = t
        self._project = project
        self._on_saved = on_saved
        self._on_delete = on_delete
        self.setStyleSheet(f"background:{theme.SCRIM};")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 90, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self._build_panel())
        self.hide()

    def _build_panel(self) -> QFrame:
        t = self._t
        panel = QFrame()
        panel.setFixedWidth(460)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setObjectName("SettingsPanel")
        panel.setStyleSheet(
            f"QFrame#SettingsPanel{{background:{t['surface']}; border:1px solid {t['border_strong']};"
            f" border-radius:14px;}}")
        soft_shadow(panel, 28, 40, 10)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._header())
        v.addWidget(hline(t))

        body = QWidget()
        body.setStyleSheet("background:transparent;")  # see the class docstring's bare-QWidget note
        b = QVBoxLayout(body)
        b.setContentsMargins(22, 20, 22, 20)
        b.setSpacing(16)
        b.addWidget(self._general_section())
        b.addWidget(self._danger_zone())
        v.addWidget(body)
        return panel

    def _header(self) -> QWidget:
        t = self._t
        h = QWidget()
        h.setStyleSheet("background:transparent;")
        row = QHBoxLayout(h)
        row.setContentsMargins(20, 16, 14, 16)
        row.setSpacing(10)
        row.addWidget(label("Project Settings", 15, t["text"], 600))
        row.addStretch(1)
        row.addWidget(IconButton("close", t, 27, "Close", self.hide))
        return h

    def _general_section(self) -> QWidget:
        t = self._t
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)

        name_col = QVBoxLayout()
        name_col.setSpacing(7)
        name_col.addWidget(GroupLabel("Project name", t))
        self._name_input = QLineEdit(self._project.name)
        name_col.addWidget(self._name_input)
        v.addLayout(name_col)

        desc_col = QVBoxLayout()
        desc_col.setSpacing(7)
        desc_col.addWidget(GroupLabel("Description", t))
        self._desc_input = QLineEdit(self._project.description)
        self._desc_input.setCursorPosition(0)  # show the start, not the tail, of a long description
        desc_col.addWidget(self._desc_input)
        v.addLayout(desc_col)

        save_row = QHBoxLayout()
        save_row.addStretch(1)
        save_btn = PillButton("Save", t, "primary", small=True)
        save_btn.clicked.connect(self._save_general)
        save_row.addWidget(save_btn)
        v.addLayout(save_row)
        return w

    def _danger_zone(self) -> QFrame:
        t = self._t
        card = QFrame()
        card.setObjectName("DangerZone")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            f"QFrame#DangerZone{{background:{t['danger_weak']}; border:1px solid {t['danger']};"
            f" border-radius:10px;}}")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(8)
        v.addWidget(label("Danger zone", 12, t["danger"], 700, 0.3))
        body = label(
            "Deleting this project removes its images, results, and settings "
            "from this device. This can't be undone.", 12, t["text_muted"])
        body.setWordWrap(True)
        v.addWidget(body)
        row = QHBoxLayout()
        row.addStretch(1)
        delete_btn = PillButton("Delete Project", t, "danger", small=True)
        delete_btn.clicked.connect(self._confirm_delete)
        row.addWidget(delete_btn)
        v.addLayout(row)
        return card

    def _save_general(self) -> None:
        name = self._name_input.text().strip()
        desc = self._desc_input.text().strip()
        self.hide()
        if name and self._on_saved:
            self._on_saved(name, desc)

    def _confirm_delete(self) -> None:
        # Keep a ref alive while it's open -- nothing else references this
        # dialog Python-side (only Qt's own C++ parent-child ownership), so
        # without this it's fair game for garbage collection before a click.
        self._active_dialog = confirm_delete_project(
            self, self._t, self._project.name, on_confirm=self._do_delete)

    def _do_delete(self) -> None:
        self.hide()
        if self._on_delete:
            self._on_delete()

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
        self.deleteLater()  # throwaway, one-shot -- see ConfirmDialog's docstring
