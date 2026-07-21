"""Tests for the bundled demo-project generator (studio/sample_data.py).

The geometry/metrics half is pure numpy and always runs (it must work in the
light CI test group). The wiring half (``ensure_sample_project``) touches disk
via cv2, so it's guarded with ``importorskip`` — it exercises the real
ProjectStore + SegmentController the app uses, just against a temp store.
"""
from pathlib import Path

import numpy as np
import pytest

from studio.sample_data import (
    SAMPLE_PROJECT_ID,
    SampleField,
    perturb_prediction,
    synthesize_nuclei_field,
)


# ── pure synth (numpy only) ──────────────────────────────────────────────────

def test_synthesize_shapes_and_dtypes():
    f = synthesize_nuclei_field(size=256, n=80, seed=3)
    assert isinstance(f, SampleField)
    assert f.image.shape == (256, 256, 3)
    assert f.image.dtype == np.uint8
    assert f.image.min() >= 0 and f.image.max() <= 255
    assert f.ground_truth.shape == (256, 256)
    assert f.ground_truth.dtype == np.int32
    assert f.prediction.shape == (256, 256)


def test_synthesize_is_deterministic():
    a = synthesize_nuclei_field(size=192, n=60, seed=11)
    b = synthesize_nuclei_field(size=192, n=60, seed=11)
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.ground_truth, b.ground_truth)
    assert np.array_equal(a.prediction, b.prediction)


def test_synthesize_has_many_instances():
    f = synthesize_nuclei_field(size=512, n=190, seed=7)
    n_gt = len(set(np.unique(f.ground_truth)) - {0})
    # rejection sampling may place slightly fewer than requested, never more
    assert 130 <= n_gt <= 190


def test_image_reads_as_a_field_not_a_blank():
    # the synthesized field must actually carry signal, not be a flat colour
    f = synthesize_nuclei_field(size=256, n=80, seed=5)
    assert f.image.std() > 8


def test_prediction_is_imperfect_but_good():
    """The stand-in engine result should differ from GT (some misses/merges)
    yet score a believable, high F1 — never a fake perfect 1.0."""
    benchmark = pytest.importorskip("velum_core.benchmark")
    f = synthesize_nuclei_field(size=512, n=190, seed=7)
    assert not np.array_equal(f.prediction, f.ground_truth)
    n_gt = len(set(np.unique(f.ground_truth)) - {0})
    n_pred = len(set(np.unique(f.prediction)) - {0})
    assert n_pred < n_gt  # drops + merges reduce the count
    m = benchmark.evaluate(f.ground_truth, f.prediction)
    f1 = float(m.get("f1") or m.get("F1") or 0.0)
    assert 0.85 <= f1 < 1.0


def test_perturb_empty_mask_is_safe():
    empty = np.zeros((16, 16), dtype=np.int32)
    assert np.array_equal(perturb_prediction(empty), empty)


# ── wiring (needs cv2 for image IO) ──────────────────────────────────────────

def _store_and_controller(tmp_path: Path):
    from studio.project import ProjectStore
    from studio.segment_controller import SegmentController
    store = ProjectStore(tmp_path / "projects")
    controller = SegmentController(storage_dir=tmp_path / "store")
    return store, controller


def test_ensure_sample_project_creates_hero(tmp_path):
    pytest.importorskip("cv2")
    from studio.sample_data import ensure_sample_project

    store, controller = _store_and_controller(tmp_path)
    sid = ensure_sample_project(store, controller)
    assert sid == SAMPLE_PROJECT_ID
    assert store.exists(sid)

    project = store.load(sid)
    assert project.image_paths and Path(project.image_paths[0]).exists()
    assert project.stats.n_images == 1
    assert project.stats.n_cells > 100
    assert project.stats.last_f1 is not None and 0.85 <= project.stats.last_f1 < 1.0
    assert project.stats.progress == 100
    assert "sample" in project.tags

    # the workspace loads this back as the Segmentation layer
    saved = controller.load_result_mask(project, project.image_paths[0])
    assert saved is not None
    assert len(set(np.unique(saved)) - {0}) > 100

    # a ground-truth sibling is discoverable for the F1 evaluation
    gt = controller.find_gt_for_image(project.image_paths[0])
    assert gt is not None and Path(gt).exists()


def test_ensure_sample_project_is_idempotent(tmp_path):
    pytest.importorskip("cv2")
    from studio.sample_data import ensure_sample_project

    store, controller = _store_and_controller(tmp_path)
    sid1 = ensure_sample_project(store, controller)
    img1 = store.load(sid1).image_paths[0]
    mtime1 = Path(img1).stat().st_mtime_ns

    sid2 = ensure_sample_project(store, controller)  # second call: no rebuild
    assert sid2 == sid1
    assert store.load(sid2).image_paths[0] == img1
    assert Path(img1).stat().st_mtime_ns == mtime1  # untouched
