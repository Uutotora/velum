"""Tests for studio/dataset_controller.py — the Datasets tab logic:
build-candidate inspection, building a dataset from a curated selection, and
the datasets → project import round-trip. Pure logic + cv2, light CI group."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from studio.dataset_controller import DatasetController
from studio.project import Project, ProjectSettings, ProjectStore
from studio.segment_controller import SegmentController


def _img(path: Path, seed=0):
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), (rng.random((16, 16)) * 255).astype(np.uint8))


def _mask(ids):
    m = np.zeros((16, 16), np.int32)
    for k, i in enumerate(ids):
        m[k, :] = i
    return m


@pytest.fixture
def ctrl(tmp_path):
    seg = SegmentController(storage_dir=tmp_path / "storage")
    return DatasetController(segment=seg, datasets_root=tmp_path / "datasets")


def _project_with_masks(ctrl, tmp_path):
    a, b, c = (tmp_path / f"cell{n}.png" for n in range(3))
    for p in (a, b, c):
        _img(p)
    project = Project(id="p1", name="Cohort",
                      image_paths=[str(a), str(b), str(c)],
                      settings=ProjectSettings(engine="cellseg1", pixel_size_um=0.3))
    ctrl._segment.save_result_mask(project, a, _mask([1, 2, 3]))
    ctrl._segment.save_result_mask(project, c, _mask([9]))   # b left unsegmented
    return project, (a, b, c)


def test_build_candidates_reports_status(ctrl, tmp_path):
    project, (a, b, c) = _project_with_masks(ctrl, tmp_path)
    cands = ctrl.build_candidates(project)
    by_name = {x.name: x for x in cands}
    assert by_name["cell0.png"].segmented and by_name["cell0.png"].cells == 3
    assert not by_name["cell1.png"].segmented and by_name["cell1.png"].cells == 0
    assert by_name["cell2.png"].segmented and by_name["cell2.png"].cells == 1
    assert ctrl.segmented_count(project) == 2


def test_build_from_project_writes_dataset(ctrl, tmp_path):
    project, (a, b, c) = _project_with_masks(ctrl, tmp_path)
    info = ctrl.build_from_project(
        project, [str(a), str(b), str(c)], name="Nuclei v1")
    # b has no mask -> silently excluded; only a + c make it in
    assert info.n_images == 2 and info.n_cells == 4
    assert info.engine == "cellseg1"
    assert (info.path / "dataset.json").is_file()
    assert info in [] or ctrl.list_datasets()[0].id == info.id


def test_build_from_project_raises_when_none_segmented(ctrl, tmp_path):
    project, (a, b, c) = _project_with_masks(ctrl, tmp_path)
    with pytest.raises(ValueError):
        ctrl.build_from_project(project, [str(b)], name="Empty")     # b unsegmented
    assert ctrl.list_datasets() == []          # nothing half-written left behind


def test_import_to_project_seeds_masks(ctrl, tmp_path):
    project, (a, b, c) = _project_with_masks(ctrl, tmp_path)
    info = ctrl.build_from_project(project, [str(a), str(c)], name="Set A")

    pstore = ProjectStore(tmp_path / "projects")
    imported = ctrl.import_to_project(info, pstore, name="Reopened")

    assert imported.name == "Reopened"
    assert len(imported.image_paths) == 2
    assert imported.settings.engine == "cellseg1"
    assert imported.settings.pixel_size_um == 0.3
    # every imported image opens with its mask already there
    for p in imported.image_paths:
        m = ctrl._segment.load_result_mask(imported, p)
        assert m is not None and m.max() > 0
    assert imported.stats.n_images == 2 and imported.stats.progress == 100


def test_train_target_points_at_images_and_masks(ctrl, tmp_path):
    project, (a, b, c) = _project_with_masks(ctrl, tmp_path)
    info = ctrl.build_from_project(project, [str(a)], name="One")
    images_dir, masks_dir = DatasetController.train_target(info)
    assert images_dir == info.path / "images"
    assert masks_dir == info.path / "masks"
    assert (masks_dir / "cell0.png").is_file()   # <stem>.png beside images/
