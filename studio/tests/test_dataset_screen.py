"""Offscreen wiring tests for the Datasets tab (studio/dataset_screen.py).
Constructs the real screen + build dialog under offscreen Qt with the real
QSS applied, and drives the interactive build + detail flow. No torch/napari;
pixel rendering itself is checked by the screenshot pass, not here."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QFrame, QWidget

from studio import theme
from studio.dataset_controller import DatasetController
from studio.dataset_screen import DatasetsScreen, NewDatasetDialog
from studio.project_controller import ProjectController
from studio.segment_controller import SegmentController


@pytest.fixture
def app():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(theme.build_qss(theme.DARK))   # the app-wide cascade real bugs need
    yield a
    a.setStyleSheet("")                            # QApplication is process-wide; reset


@pytest.fixture
def parent(app):
    w = QWidget(); w.resize(1200, 820); w.show()
    return w


def _seed_project(tmp_path):
    pc = ProjectController(store_root=tmp_path / "projects")
    seg = SegmentController(storage_dir=tmp_path / "storage")
    dc = DatasetController(segment=seg, datasets_root=tmp_path / "datasets")
    img_dir = tmp_path / "imgs"; img_dir.mkdir()
    paths = []
    for i in range(3):
        p = img_dir / f"cell{i}.png"
        cv2.imwrite(str(p), (np.random.default_rng(i).random((16, 16)) * 255).astype(np.uint8))
        paths.append(str(p))
    project = pc.store.create("Cohort", image_paths=paths)
    mask = np.zeros((16, 16), np.int32); mask[0] = 1; mask[1] = 2
    seg.save_result_mask(project, paths[0], mask)   # only 2 of 3 segmented
    seg.save_result_mask(project, paths[2], mask)
    pc.store.save(project)
    return pc, dc, project


def _find(widget, cls, name):
    for w in widget.findChildren(cls):
        if w.objectName() == name:
            return w
    return None


def test_empty_state_then_card_after_build(parent, tmp_path):
    pc, dc, project = _seed_project(tmp_path)
    screen = DatasetsScreen(theme.DARK, dc, pc, on_toast=lambda *a: None,
                            on_open_project=lambda p: None, on_navigate=lambda k: None)
    screen.setParent(parent); screen.show()

    assert _find(screen, QFrame, "DSEmpty") is not None   # no datasets yet
    assert _find(screen, QFrame, "DSCard") is None

    # build one straight through the controller, then refresh the screen
    info = dc.build_from_project(project, project.image_paths, name="Set A")
    screen.refresh()
    assert _find(screen, QFrame, "DSEmpty") is None
    assert _find(screen, QFrame, "DSCard") is not None
    assert info.n_images == 2                              # 2 segmented of 3


def test_build_dialog_populates_and_creates(parent, tmp_path):
    pc, dc, project = _seed_project(tmp_path)
    pc.set_active(project.id)
    screen = DatasetsScreen(theme.DARK, dc, pc, on_toast=lambda *a: None,
                            on_open_project=lambda p: None, on_navigate=lambda k: None)
    screen.setParent(parent); screen.show()

    screen.open_new_dialog()
    dlg = screen._dialog
    assert isinstance(dlg, NewDatasetDialog)
    # active project auto-selected; its 2 segmented images pre-checked
    assert dlg._project is not None and dlg._project.id == project.id
    assert len(dlg._selected) == 2
    assert len([c for c in dlg._candidates if c.segmented]) == 2

    dlg._create()
    assert dc.list_datasets()                              # a dataset now exists
    assert dlg.isHidden()                                  # modal closed on success


def test_dialog_no_eligible_projects_message(parent, tmp_path):
    # a project with zero segmented images -> not eligible
    pc = ProjectController(store_root=tmp_path / "p")
    dc = DatasetController(segment=SegmentController(storage_dir=tmp_path / "s"),
                           datasets_root=tmp_path / "d")
    pc.store.create("Empty", image_paths=[str(tmp_path / "x.png")])
    screen = DatasetsScreen(theme.DARK, dc, pc, on_toast=lambda *a: None,
                            on_open_project=lambda p: None, on_navigate=lambda k: None)
    screen.setParent(parent); screen.show()
    screen.open_new_dialog()
    assert screen._dialog._eligible_projects() == []
    assert screen._dialog._create_btn.isEnabled() is False


# ── import-from-disk mode ────────────────────────────────────────────────────

def _external_folder(tmp_path):
    ext = tmp_path / "external"; ext.mkdir()
    for n in ("A", "B"):
        cv2.imwrite(str(ext / f"{n}.png"),
                    (np.random.default_rng(0).random((16, 16)) * 255).astype(np.uint8))
        cv2.imwrite(str(ext / f"{n}_mask.png"), np.array([[0, 1], [2, 3]], np.uint16))
    cv2.imwrite(str(ext / "lonely.png"),
                (np.random.default_rng(1).random((16, 16)) * 255).astype(np.uint8))
    return str(ext)


def test_dialog_defaults_to_import_when_no_projects(parent, tmp_path):
    pc = ProjectController(store_root=tmp_path / "p")   # no segmented projects
    dc = DatasetController(segment=SegmentController(storage_dir=tmp_path / "s"),
                           datasets_root=tmp_path / "d")
    screen = DatasetsScreen(theme.DARK, dc, pc, on_toast=lambda *a: None,
                            on_open_project=lambda p: None, on_navigate=lambda k: None)
    screen.setParent(parent); screen.show()
    screen.open_new_dialog()
    assert screen._dialog._mode == "import"        # bring-your-own is offered first


def test_import_scan_and_create(parent, tmp_path):
    pc, dc, project = _seed_project(tmp_path)
    screen = DatasetsScreen(theme.DARK, dc, pc, on_toast=lambda *a: None,
                            on_open_project=lambda p: None, on_navigate=lambda k: None)
    screen.setParent(parent); screen.show()
    screen.open_new_dialog()
    dlg = screen._dialog
    dlg._set_mode("import")
    assert dlg._mode == "import"
    dlg._on_import_paths([_external_folder(tmp_path)])
    assert dlg._scan is not None
    assert dlg._scan.n_with_mask == 2 and dlg._scan.n_images == 3
    assert dlg._ready_count() == 2 and dlg._create_btn.isEnabled()
    dlg._create()
    assert dlg.isHidden()
    assert any(d.engine == "imported" for d in dc.list_datasets())
