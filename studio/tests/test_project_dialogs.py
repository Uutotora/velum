"""Headless tests for studio/project_dialogs.py -- ConfirmDialog (+ the
confirm_trash/confirm_delete_forever builders) and TrashDialog.

Offscreen Qt, no napari/torch.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
pd = pytest.importorskip("studio.project_dialogs")

from PyQt6 import sip
from PyQt6.QtCore import QPoint
from PyQt6.QtCore import Qt as QtNS
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QWidget

from studio import theme
from studio.project import ProjectStore
from studio.project_controller import ProjectController


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def parent(app):
    w = QWidget()
    w.resize(1200, 800)
    w.show()
    return w


@pytest.fixture
def controller(tmp_path):
    return ProjectController(ProjectStore(tmp_path))


def _click_outside(dialog) -> None:
    # childAt() on a point with no child (far corner, outside the centred
    # panel) returns None -- the same path a real click on the scrim takes.
    # Mirrors test_new_project_dialog.py's test_click_outside_panel_closes.
    pos = QPoint(2, 2)
    assert dialog.childAt(pos) is None
    event = QMouseEvent(QMouseEvent.Type.MouseButtonPress, pos.toPointF(),
                        QtNS.MouseButton.LeftButton, QtNS.MouseButton.LeftButton,
                        QtNS.KeyboardModifier.NoModifier)
    dialog.mousePressEvent(event)


# ── ConfirmDialog ────────────────────────────────────────────────────────────
def test_confirm_dialog_hidden_until_open(parent):
    dlg = pd.ConfirmDialog(parent, theme.DARK, "Move to Trash?", "body",
                           on_confirm=lambda: None)
    assert dlg.isHidden()
    dlg.open()
    assert not dlg.isHidden()


def test_confirm_dialog_click_outside_closes_without_confirming(parent):
    seen = []
    dlg = pd.ConfirmDialog(parent, theme.DARK, "Title", "body",
                           on_confirm=lambda: seen.append(1))
    dlg.open()
    _click_outside(dlg)
    assert dlg.isHidden()
    assert seen == []


def test_confirm_dialog_confirm_button_calls_on_confirm_and_hides(parent):
    seen = []
    dlg = pd.ConfirmDialog(parent, theme.DARK, "Title", "body", confirm_label="Do it",
                           on_confirm=lambda: seen.append(1))
    dlg.open()
    dlg._confirm()
    assert seen == [1]
    assert dlg.isHidden()


def test_confirm_dialog_constructor_time_hide_does_not_self_delete(app, parent):
    """__init__ ends with self.hide() (same construction as NewProjectDialog)
    to start hidden -- must be a no-op (never-shown -> hidden is not a real
    transition) and NOT trigger hideEvent's deleteLater(), or the dialog
    would be destroyed before ever being usable."""
    dlg = pd.ConfirmDialog(parent, theme.DARK, "Title", "body", on_confirm=lambda: None)
    app.processEvents()
    assert not sip.isdeleted(dlg)
    dlg.open()
    assert not dlg.isHidden()


def test_confirm_dialog_self_disposes_after_a_real_hide(app, parent):
    dlg = pd.ConfirmDialog(parent, theme.DARK, "Title", "body", on_confirm=lambda: None)
    dlg.open()
    dlg.hide()
    app.processEvents()  # let the deferred deleteLater() run
    assert sip.isdeleted(dlg)


def test_confirm_trash_names_the_project_and_wires_confirm(parent):
    seen = []
    dlg = pd.confirm_trash(parent, theme.DARK, "My Project", on_confirm=lambda: seen.append(1))
    assert not dlg.isHidden()
    dlg._confirm()
    assert seen == [1]


def test_confirm_trash_escapes_html_in_the_project_name(parent):
    # A project named with HTML-special characters must not break the rich
    # text body or inject markup -- html.escape() before embedding.
    dlg = pd.confirm_trash(parent, theme.DARK, "<script>alert(1)</script>", on_confirm=lambda: None)
    from PyQt6.QtWidgets import QLabel
    labels = dlg.findChildren(QLabel)
    assert any("&lt;script&gt;" in lbl.text() for lbl in labels)
    assert not any("<script>" in lbl.text() for lbl in labels)


def test_confirm_delete_forever_wires_confirm(parent):
    seen = []
    dlg = pd.confirm_delete_forever(parent, theme.DARK, "Doomed", on_confirm=lambda: seen.append(1))
    assert not dlg.isHidden()
    dlg._confirm()
    assert seen == [1]


# ── RenameDialog ─────────────────────────────────────────────────────────────
def test_rename_dialog_hidden_until_open(parent):
    dlg = pd.RenameDialog(parent, theme.DARK, "Old Name", on_save=lambda n: None)
    assert dlg.isHidden()
    dlg.open()
    assert not dlg.isHidden()


def test_rename_dialog_prefills_and_selects_the_current_name(parent):
    dlg = pd.RenameDialog(parent, theme.DARK, "Old Name", on_save=lambda n: None)
    assert dlg._input.text() == "Old Name"
    assert dlg._input.selectedText() == "Old Name"


def test_rename_dialog_click_outside_does_not_save(parent):
    seen = []
    dlg = pd.RenameDialog(parent, theme.DARK, "Old Name", on_save=lambda n: seen.append(n))
    dlg.open()
    _click_outside(dlg)
    assert dlg.isHidden()
    assert seen == []


def test_rename_dialog_save_calls_on_save_with_the_trimmed_name(parent):
    seen = []
    dlg = pd.RenameDialog(parent, theme.DARK, "Old Name", on_save=lambda n: seen.append(n))
    dlg.open()
    dlg._input.setText("  New Name  ")
    dlg._save()
    assert seen == ["New Name"]
    assert dlg.isHidden()


def test_rename_dialog_return_pressed_also_saves(parent):
    """The QLineEdit's returnPressed is wired to the same save path -- Enter
    should work, not just clicking Save."""
    seen = []
    dlg = pd.RenameDialog(parent, theme.DARK, "Old Name", on_save=lambda n: seen.append(n))
    dlg.open()
    dlg._input.setText("Via Enter")
    dlg._input.returnPressed.emit()
    assert seen == ["Via Enter"]


def test_rename_dialog_blank_name_does_not_save(parent):
    seen = []
    dlg = pd.RenameDialog(parent, theme.DARK, "Old Name", on_save=lambda n: seen.append(n))
    dlg.open()
    dlg._input.setText("   ")
    dlg._save()
    assert seen == []
    assert dlg.isHidden()  # still closes -- blank just means "no-op", not "stay open"


def test_rename_dialog_self_disposes_after_a_real_hide(app, parent):
    dlg = pd.RenameDialog(parent, theme.DARK, "Old Name", on_save=lambda n: None)
    dlg.open()
    dlg.hide()
    app.processEvents()
    assert sip.isdeleted(dlg)


# ── TrashDialog ──────────────────────────────────────────────────────────────
def test_trash_dialog_empty_state(parent, controller):
    dlg = pd.TrashDialog(parent, theme.DARK, controller)
    dlg.open()
    assert "empty" in dlg._sub_lbl.text() or dlg._rows.count() >= 1


def test_trash_dialog_lists_trashed_projects(parent, controller):
    p = controller.list_projects()[0]
    controller.trash_project(p.id)
    dlg = pd.TrashDialog(parent, theme.DARK, controller)
    dlg.open()
    assert p.name in dlg._sub_lbl.text() or "1 project" in dlg._sub_lbl.text()
    from PyQt6.QtWidgets import QFrame
    rows = dlg._rows_host.findChildren(QFrame, "TrashRow")
    assert len(rows) == 1


def test_trash_dialog_restore_removes_the_row_and_calls_on_changed(parent, controller):
    p = controller.list_projects()[0]
    controller.trash_project(p.id)
    changed = []
    dlg = pd.TrashDialog(parent, theme.DARK, controller, on_changed=lambda: changed.append(1))
    dlg.open()

    dlg._restore(p.id)

    assert p.id in [x.id for x in controller.list_projects()]
    assert changed == [1]
    from PyQt6.QtWidgets import QFrame
    assert dlg._rows_host.findChildren(QFrame, "TrashRow") == []


def test_trash_dialog_delete_forever_requires_its_own_confirm(parent, controller):
    """Delete Forever must not act immediately -- it opens its own nested
    ConfirmDialog first (the one truly irreversible step in this flow)."""
    p = controller.list_projects()[0]
    controller.trash_project(p.id)
    dlg = pd.TrashDialog(parent, theme.DARK, controller)
    dlg.open()

    dlg._confirm_delete(p.id, p.name)

    assert controller.store.exists(p.id)  # not deleted yet -- only the nested dialog opened
    nested = next(w for w in dlg.findChildren(pd.ConfirmDialog) if w.isVisible())
    nested._confirm()
    assert not controller.store.exists(p.id)


def test_trash_dialog_delete_forever_calls_on_changed(parent, controller):
    p = controller.list_projects()[0]
    controller.trash_project(p.id)
    changed = []
    dlg = pd.TrashDialog(parent, theme.DARK, controller, on_changed=lambda: changed.append(1))
    dlg.open()
    dlg._delete_forever(p.id)
    assert changed == [1]


def test_trash_dialog_labels_sit_on_the_panel_surface_not_a_scrim_blended_patch(app, controller):
    """Regression test: TrashDialog's header (a bare QWidget()) and its
    rows_host/scroll-area content used to rely on the app-wide QWidget{
    background:<bg>}/QScrollArea-viewport rule instead of the panel's own
    `surface` -- glaring specifically because this dialog sits on a
    translucent scrim (rgba(8,10,20,0.34)): a scrim-over-bg blend is
    visibly darker than a scrim-over-surface blend, so "Trash", "N
    project(s)", and every trashed row's name/timestamp rendered inside a
    boxed patch. Confirmed by an actual light-theme screenshot before the
    fix. Same bug family, same fix (overlays.CommandPalette's
    `_results_container`/`_results_area` already do this) as
    test_new_project_dialog.py's own analogous regression test.
    """
    import time as _time
    p = controller.list_projects()[0]
    controller.trash_project(p.id)

    t = theme.LIGHT  # see the NewProjectDialog test's comment for why light, not dark
    app.setStyleSheet(theme.build_qss(t))
    try:
        win = QWidget()
        win.resize(1400, 900)
        win.setStyleSheet(f"background:{t['bg']};")
        win.show()
        dlg = pd.TrashDialog(win, t, controller)
        dlg.open()
        for _ in range(30):
            app.processEvents()
            _time.sleep(0.01)

        img = win.grab().toImage()
        from PyQt6.QtWidgets import QLabel
        labels = [w for w in dlg.findChildren(QLabel) if w.text().strip() and w.isVisible()]
        assert len(labels) >= 4  # "Trash", "N project(s)", the row's name + "Trashed …"
        offenders = []
        for lbl in labels:
            pt = lbl.mapTo(win, lbl.rect().topLeft())
            sample = img.pixelColor(pt.x(), pt.y())
            if sample.name() != t["surface"]:
                offenders.append((lbl.text(), sample.name()))
        assert not offenders, f"labels not sitting on the panel's own surface fill: {offenders}"
    finally:
        app.setStyleSheet("")  # process-wide QApplication singleton -- don't leak


def test_trash_dialog_refresh_updates_after_external_change(parent, controller):
    p = controller.list_projects()[0]
    dlg = pd.TrashDialog(parent, theme.DARK, controller)
    dlg.open()
    from PyQt6.QtWidgets import QFrame
    assert dlg._rows_host.findChildren(QFrame, "TrashRow") == []

    controller.trash_project(p.id)
    dlg.refresh()

    assert len(dlg._rows_host.findChildren(QFrame, "TrashRow")) == 1
