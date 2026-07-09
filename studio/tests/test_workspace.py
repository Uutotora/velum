"""Tests for the Segment workspace screen, wired for real (studio/workspace.py).

Offscreen Qt. Every predict/batch/benchmark test monkeypatches
napari_app.inference_cache.predict_cached (same seam as
test_segment_controller.py) so the real UI -> controller -> engine chain
runs end to end without a GPU/SAM weights/torch actually loading anything.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")
workspace = pytest.importorskip("studio.workspace")

from PyQt6.QtWidgets import QApplication, QFileDialog

from studio import theme
from studio.layer_model import ImageLayer, LabelsLayer, PAINT, PAN_ZOOM, PointsLayer, ShapesLayer
from studio.project import Project, ProjectSettings, ProjectStore
from studio.project_controller import ProjectController
from studio.segment_controller import SegmentController
from studio.workspace import WorkspaceScreen


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _write_image(path: Path, size=48, seed=0) -> None:
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), (rng.random((size, size, 3)) * 255).astype(np.uint8))


def _write_mask(path: Path, *labels_and_boxes, size=48) -> None:
    mask = np.zeros((size, size), dtype=np.uint16)
    for lbl, (y0, y1, x0, x1) in labels_and_boxes:
        mask[y0:y1, x0:x1] = lbl
    cv2.imwrite(str(path), mask)


def _fake_predict_cached(config, image_rgb):
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[h // 8: h // 2, w // 8: w // 2] = 1
    mask[h // 2: h - h // 8, w // 2: w - w // 8] = 2
    return mask


@pytest.fixture(autouse=True)
def _fake_engine(monkeypatch):
    monkeypatch.setattr("napari_app.inference_cache.predict_cached", _fake_predict_cached)
    # cellpose may be genuinely installed in whatever env runs these tests —
    # without this, the benchmark test sees it as available and actually
    # runs real Cellpose inference instead of treating it as absent (see the
    # identical note in test_segment_controller.py's own _fake_engine).
    monkeypatch.setattr("napari_app.engines.cellpose_available", lambda: False)


@pytest.fixture
def storage(tmp_path):
    d = tmp_path / "storage"
    (d / "sam_backbone").mkdir(parents=True)
    (d / "sam_backbone" / "sam_vit_h_4b8939.pth").write_bytes(b"fake")
    (d / "loras").mkdir(parents=True)
    (d / "loras" / "nuclei-dapi-r8.pth").write_bytes(b"fake-lora")
    return d


@pytest.fixture
def segment(storage):
    return SegmentController(storage_dir=storage)


@pytest.fixture
def projects(tmp_path):
    return ProjectController(ProjectStore(tmp_path / "projects"), seed_if_empty=False)


@pytest.fixture
def toasts():
    return []


def _ws(app, segment, projects, toasts, on_toggle_logs=None):
    return WorkspaceScreen(theme.DARK, segment, projects,
                           lambda title, sub: toasts.append((title, sub)), on_toggle_logs)


def _make_project(tmp_path, projects, storage, *, n_images=1, with_gt=False, size=48):
    imgs = []
    for i in range(n_images):
        p = tmp_path / f"img_{i}.png"
        _write_image(p, size=size, seed=i)
        imgs.append(str(p))
    if with_gt:
        _write_mask(tmp_path / "img_0_mask.png", (1, (2, 10, 2, 10)), (2, (20, 30, 20, 30)), size=size)
    lora = storage / "loras" / "nuclei-dapi-r8.pth"
    project = projects.store.create(
        "Test Project", "x", image_paths=imgs,
        settings=ProjectSettings(engine="cellseg1", model_name=str(lora)))
    return project


def _pump(app, ws, timeout=10):
    t0 = time.monotonic()
    while (ws._predicting or ws._batching or ws._benching) and time.monotonic() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(5):
        app.processEvents()


# ── construction / empty state ────────────────────────────────────────────────
def test_construct_with_no_project_shows_empty_states(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    assert ws._project is None
    assert ws._canvas._base_shape() is None


def test_refresh_is_a_noop_when_no_active_project(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    ws.refresh()  # should not raise
    assert ws._project is None


# ── project / image loading ──────────────────────────────────────────────────
def test_load_project_selects_first_image_and_builds_layers(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=2)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert ws._current_image_path == project.image_paths[0]
    kinds = [l.kind for l in ws._layers]
    assert kinds == ["image", "labels"]
    assert ws._layers.selected.kind == "labels"  # Segmentation layer selected by default


def test_load_project_auto_discovers_ground_truth(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1, with_gt=True)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    gt = ws._layers.find("Ground truth")
    assert gt is not None
    assert gt.visible is False  # auto-discovered GT starts hidden


def test_ground_truth_layers_get_the_fixed_gt_colour_not_random_hues(
        app, segment, projects, toasts, tmp_path, storage):
    """Matches the classic app's own GT convention: a fixed colour (not
    per-instance random), so GT and predictions read as distinct roles."""
    from studio.workspace import _GT_COLOR

    project = _make_project(tmp_path, projects, storage, n_images=1, with_gt=True)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    gt = ws._layers.find("Ground truth")
    assert gt.opacity == 0.9
    ids = [i for i in set(gt.data.ravel().tolist()) if i > 0]
    assert ids  # the fixture's GT mask actually has labelled cells
    assert all(gt.get_color(i) == _GT_COLOR for i in ids)


def test_refresh_reloads_only_on_project_switch(app, segment, projects, toasts, tmp_path, storage):
    p1 = _make_project(tmp_path, projects, storage, n_images=1)
    p2_dir = tmp_path / "p2"
    p2_dir.mkdir()
    p2 = _make_project(p2_dir, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)

    projects.set_active(p1.id)
    ws.refresh()
    assert ws._project.id == p1.id
    seg = ws._layers.find("Segmentation")
    seg.data[0, 0] = 77  # simulate in-progress unsaved work

    ws.refresh()  # same project still active -> must NOT reset layers
    assert ws._layers.find("Segmentation").data[0, 0] == 77

    projects.set_active(p2.id)
    ws.refresh()
    assert ws._project.id == p2.id
    assert ws._layers.find("Segmentation").data[0, 0] == 0  # fresh layers for the new project


def test_select_image_switches_layers(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=2)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._select_image(project.image_paths[1])
    assert ws._current_image_path == project.image_paths[1]
    assert ws._last_result is None


# ── adding images to an existing project ──────────────────────────────────────
def test_add_image_paths_appends_and_persists(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    new_img = tmp_path / "extra.png"
    _write_image(new_img, seed=99)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._add_image_paths([str(new_img)])
    assert str(new_img) in project.image_paths
    assert len(project.image_paths) == 2
    reloaded = projects.store.load(project.id)
    assert str(new_img) in reloaded.image_paths
    assert reloaded.stats.n_images == 2
    assert any("Images added" in t[0] for t in toasts)


def test_add_image_paths_dedupes_and_filters_unsupported(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    existing = project.image_paths[0]
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("hello")
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._add_image_paths([existing, str(not_an_image)])
    assert len(project.image_paths) == 1  # nothing new actually added
    assert any("No new images" in t[0] for t in toasts)


def test_add_image_paths_auto_selects_first_image_for_empty_project(
        app, segment, projects, toasts, tmp_path, storage):
    project = projects.store.create(
        "Empty", "", settings=ProjectSettings(
            engine="cellseg1", model_name=str(storage / "loras" / "nuclei-dapi-r8.pth")))
    new_img = tmp_path / "first.png"
    _write_image(new_img)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert ws._current_image_path is None
    ws._add_image_paths([str(new_img)])
    assert ws._current_image_path == str(new_img)


def test_add_images_without_a_project_toasts(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    ws._add_images()
    assert toasts and "No project open" in toasts[-1][0]


def test_images_drop_adds_only_local_image_files(app, segment, projects, toasts, tmp_path, storage):
    from PyQt6.QtCore import QUrl

    project = _make_project(tmp_path, projects, storage, n_images=1)
    new_img = tmp_path / "dropped.png"
    _write_image(new_img, seed=7)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)

    class _FakeMime:
        def hasUrls(self_inner):
            return True

        def urls(self_inner):
            return [QUrl.fromLocalFile(str(new_img)), QUrl("https://example.com/not-local")]

    class _FakeDropEvent:
        def mimeData(self_inner):
            return _FakeMime()

        def acceptProposedAction(self_inner):
            pass

    ws._images_drop(_FakeDropEvent())
    assert str(new_img) in project.image_paths
    assert len(project.image_paths) == 2


# ── layer panel actions ───────────────────────────────────────────────────────
def test_add_points_and_shapes_layers(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._add_points_layer()
    ws._add_shapes_layer()
    kinds = [l.kind for l in ws._layers]
    assert kinds.count("points") == 1
    assert kinds.count("shapes") == 1
    assert ws._layers.selected.kind == "shapes"  # most-recently-added is selected


def test_add_labels_layer_needs_a_loaded_image(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    ws._add_labels_layer()
    assert not any(l.kind == "labels" for l in ws._layers)
    assert toasts and "No image loaded" in toasts[-1][0]


def test_delete_selected_layer(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    n_before = len(ws._layers)
    ws._delete_selected_layer()
    assert len(ws._layers) == n_before - 1


def test_toggle_layer_visible(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    assert seg.visible is True
    ws._toggle_layer_visible(ws._layers.index_of(seg))
    assert seg.visible is False


def test_select_layer_rebuilds_controls_for_the_right_kind(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    image_layer = ws._layers.find(Path(project.image_paths[0]).stem)
    ws._select_layer(ws._layers.index_of(image_layer))
    assert ws._layers.selected is image_layer


def test_set_canvas_mode_updates_target_layer_mode(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._set_canvas_mode(PAINT)
    assert ws._canvas.mode == PAINT
    assert ws._layers.find("Segmentation").mode == PAINT


def test_shuffle_colors_changes_seed(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    before = seg.color_seed
    ws._shuffle_colors()
    assert seg.color_seed != before


# ── Segment settings pane ─────────────────────────────────────────────────────
def test_engine_select_switches_engine_and_rebuilds_model_field(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    # this file's _fake_engine fixture reports cellpose as unavailable (see
    # its docstring), so its option text carries the "(not installed)" note
    ws._on_engine_select("Cellpose-SAM (zero-shot, generalist)  (not installed)")
    assert project.settings.engine == "cellpose"


def test_quality_preset_updates_thresholds(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._on_quality_preset(2)  # Accurate
    assert project.settings.quality_preset == "Accurate"
    assert project.settings.points_per_side == 48


def test_manual_threshold_change_marks_preset_custom(app, segment, projects, toasts, tmp_path, storage):
    from studio.components import Badge
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    badge = Badge("0.80", theme.DARK)
    ws._on_threshold_change("pred_iou_thresh", 0.91, badge)
    assert project.settings.pred_iou_thresh == pytest.approx(0.91)
    assert project.settings.quality_preset == "Custom"
    assert badge.text() == "0.91"


def test_pixel_size_edit_recomputes_results(app, segment, projects, toasts, tmp_path, storage):
    from PyQt6.QtWidgets import QLineEdit
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._last_result = segment.compute_measurements(np.zeros((10, 10), dtype=np.int32))
    edit = QLineEdit("0.5")
    ws._on_pixel_size_edited(edit)
    assert project.settings.pixel_size_um == pytest.approx(0.5)


# ── predict flow ─────────────────────────────────────────────────────────────
def test_start_predict_without_project_toasts(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    ws._start_predict()
    assert toasts and "No project open" in toasts[-1][0]


def test_start_predict_runs_and_populates_results(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    assert ws._predicting is True
    _pump(app, ws)
    assert ws._predicting is False
    assert ws._last_result is not None
    assert ws._last_result["n_cells"] == 2
    seg = ws._layers.find("Segmentation")
    assert seg.max_label == 2
    reloaded = projects.store.load(project.id)
    assert reloaded.stats.n_cells == 2
    assert any("Segmentation complete" in t[0] for t in toasts)


def test_predict_result_survives_switching_images_and_back(app, segment, projects, toasts, tmp_path, storage):
    """The core "reopen and see your result" requirement, exercised via an
    in-session round trip: run predict on image A, switch to B (a fresh,
    unsegmented image), switch back to A — A's mask and stats must come
    back exactly, not an empty layer."""
    project = _make_project(tmp_path, projects, storage, n_images=2)
    img_a, img_b = project.image_paths
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert ws._current_image_path == img_a
    ws._start_predict()
    _pump(app, ws)
    assert ws._layers.find("Segmentation").max_label == 2

    ws._select_image(img_b)
    assert ws._layers.find("Segmentation").max_label == 0  # a fresh image starts empty
    assert ws._last_result is None

    ws._select_image(img_a)
    assert ws._layers.find("Segmentation").max_label == 2
    assert ws._last_result is not None
    assert ws._last_result["n_cells"] == 2


def test_predict_result_survives_a_full_project_reload(app, segment, projects, toasts, tmp_path, storage):
    """The literal "close and reopen the project" case: a brand new
    WorkspaceScreen (as if the app were restarted) must still show the
    previous session's result for an image that was segmented."""
    project = _make_project(tmp_path, projects, storage)
    ws1 = _ws(app, segment, projects, toasts)
    ws1._load_project(project)
    ws1._start_predict()
    _pump(app, ws1)
    assert ws1._layers.find("Segmentation").max_label == 2

    # a second, independent screen instance re-loading the same project —
    # nothing shared except what's actually on disk
    ws2 = _ws(app, segment, projects, [])
    ws2._load_project(projects.store.load(project.id))
    seg2 = ws2._layers.find("Segmentation")
    assert seg2.max_label == 2
    assert ws2._last_result is not None
    assert ws2._last_result["n_cells"] == 2


def test_images_pane_shows_predicted_status_for_a_saved_but_unselected_image(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=2)
    img_a, img_b = project.image_paths
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    ws._select_image(img_b)  # img_a is no longer the selected image...
    assert ws._current_image_path != img_a
    assert segment.has_result_mask(project, img_a) is True  # ...but its result is still on disk


def test_reopening_an_image_with_gt_also_shows_evaluation_immediately(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, with_gt=True)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    assert ws._gt_metrics is not None
    first_f1 = ws._gt_metrics["f1"]

    ws2 = _ws(app, segment, projects, [])
    ws2._load_project(projects.store.load(project.id))
    assert ws2._gt_metrics is not None
    assert ws2._gt_metrics["f1"] == first_f1


def test_start_predict_bad_config_toasts_synchronously(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    project.settings.model_name = ""  # no LoRA -> build_config raises
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    assert ws._predicting is False
    assert any("Can't run segmentation" in t[0] for t in toasts)


# ── ground truth evaluation ───────────────────────────────────────────────────
def test_load_gt_and_evaluate_updates_metrics_and_dashboard_stats(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, with_gt=False)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)

    gt_path = tmp_path / "gt.png"
    _write_mask(gt_path, (1, (6, 24, 6, 24)), (2, (24, 42, 24, 42)))
    ws._load_gt(str(gt_path))
    assert ws._gt_metrics is not None
    reloaded = projects.store.load(project.id)
    assert reloaded.stats.last_f1 == ws._gt_metrics["f1"]


def test_color_by_recolors_segmentation(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    seg = ws._layers.find("Segmentation")
    ws._on_color_by("Area (heatmap)")
    assert seg.color_overrides
    ws._on_color_by("Instance ID (default)")
    assert seg.color_overrides == {}


# ── save / export (QFileDialog monkeypatched) ────────────────────────────────
def test_save_masks_without_a_result_toasts(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._save_masks()
    assert toasts and "Nothing to save" in toasts[-1][0]


def test_save_masks_writes_a_file(app, segment, projects, toasts, tmp_path, storage, monkeypatch):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    out = tmp_path / "out_mask.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    ws._save_masks()
    assert out.exists()
    assert toasts[-1][0] == "Masks saved"


def test_export_csv_writes_a_file(app, segment, projects, toasts, tmp_path, storage, monkeypatch):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    out = tmp_path / "out.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    ws._export_csv()
    assert out.exists()
    assert "Area" in out.read_text()


# ── batch ─────────────────────────────────────────────────────────────────────
def test_start_batch_runs_and_updates_dashboard_stats(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=3)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_batch()
    assert ws._batching is True
    _pump(app, ws)
    assert ws._batching is False
    reloaded = projects.store.load(project.id)
    assert reloaded.stats.n_cells == 3 * 2
    assert reloaded.stats.progress == 100
    assert any("Batch complete" in t[0] for t in toasts)


def test_start_batch_empty_project_toasts(app, segment, projects, toasts, tmp_path, storage):
    project = projects.store.create("Empty", "", settings=ProjectSettings(
        engine="cellseg1", model_name=str(storage / "loras" / "nuclei-dapi-r8.pth")))
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_batch()
    assert toasts and "Can't start batch" in toasts[-1][0]


# ── benchmark ─────────────────────────────────────────────────────────────────
def test_start_benchmark_without_gt_toasts(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, with_gt=False)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_benchmark()
    assert toasts and "Can't run benchmark" in toasts[-1][0]


def test_start_benchmark_scores_engines(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, with_gt=True)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_benchmark()
    assert ws._benching is True
    _pump(app, ws)
    assert ws._benching is False
    assert ws._bench_rows
    assert any("Benchmark complete" in t[0] for t in toasts)


# ── viewer bar actions ─────────────────────────────────────────────────────────
def test_toggle_mip_works_without_a_volume_like_real_napari(app, segment, projects, toasts, tmp_path, storage):
    """Real napari's ndisplay toggle has no dimensionality guard — it must
    not silently refuse (or toast an excuse) on a plain 2-D image; the
    Canvas gets the pseudo-3D tilt instead of a real MIP projection, but the
    *toggle itself* always works."""
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._toggle_mip()
    assert ws._canvas.mip is True


def test_roll_channel_with_one_channel_toasts_instead_of_silent_noop(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._roll_channel()
    assert toasts and "Only one channel loaded" in toasts[-1][0]


def test_toggle_grid_and_transpose(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._toggle_grid()
    assert ws._canvas.grid is True
    ws._toggle_transpose()
    assert ws._canvas.transposed is True


# ── toolbar active-state sync ─────────────────────────────────────────────────
def _is_toolbar_button_on(btn) -> bool:
    """True if _style_toolbar_button styled this button as "on" (signal_weak
    background) rather than the default transparent/off look."""
    return "background:transparent" not in btn.styleSheet()


def test_grid_button_highlights_when_grid_is_on(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    grid_btn = ws._vbar_buttons["grid"]
    assert not _is_toolbar_button_on(grid_btn)
    ws._toggle_grid()
    assert _is_toolbar_button_on(grid_btn)
    ws._toggle_grid()
    assert not _is_toolbar_button_on(grid_btn)


def test_mip_button_highlights_with_or_without_a_volume(app, segment, projects, toasts, tmp_path, storage):
    """Matches real napari's own ndisplay toggle: works — and so highlights
    — regardless of whether the loaded data has a volume to project."""
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    cube_btn = ws._vbar_buttons["cube3d"]
    ws._toggle_mip()  # no volume loaded -> still toggles (pseudo-3D tilt), still highlights
    assert ws._canvas.mip is True
    assert _is_toolbar_button_on(cube_btn)
    ws._toggle_mip()
    assert not _is_toolbar_button_on(cube_btn)

    volume = LabelsLayer("Segmentation", np.zeros((3, 20, 20), dtype=np.int32))
    ws._layers.remove(ws._layers.index_of(ws._layers.find("Segmentation")))
    ws._layers.add(volume, select=False)
    ws._toggle_mip()
    assert ws._canvas.mip is True
    assert _is_toolbar_button_on(cube_btn)


def test_floating_tool_strip_highlights_active_mode(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    pan_btn = ws._floating_tool_buttons[0][0]  # (target, PAN_ZOOM)
    brush_btn = ws._floating_tool_buttons[1][0]  # (brush, PAINT)
    assert _is_toolbar_button_on(pan_btn)  # PAN_ZOOM is the default mode
    assert not _is_toolbar_button_on(brush_btn)
    ws._set_canvas_mode(PAINT)
    assert not _is_toolbar_button_on(pan_btn)
    assert _is_toolbar_button_on(brush_btn)


def test_selecting_a_new_image_resets_mip(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=2)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._layers.remove(ws._layers.index_of(ws._layers.find("Segmentation")))
    ws._layers.add(LabelsLayer("Segmentation", np.zeros((3, 20, 20), dtype=np.int32)), select=False)
    ws._toggle_mip()
    assert ws._canvas.mip is True
    ws._current_image_path = None  # force _select_image to treat it as a real switch
    ws._select_image(project.image_paths[1])
    assert ws._canvas.mip is False


def test_toggle_logs_console_calls_the_injected_callback(app, segment, projects, toasts):
    calls = []
    ws = _ws(app, segment, projects, toasts, on_toggle_logs=lambda: calls.append(True))
    ws._toggle_logs_console()
    assert calls == [True]


# ── breadcrumb navigation ─────────────────────────────────────────────────────
def test_breadcrumb_projects_click_navigates_back(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    seen = []
    ws = WorkspaceScreen(theme.DARK, segment, projects,
                         lambda title, sub: toasts.append((title, sub)),
                         on_navigate=lambda key: seen.append(key))
    ws._load_project(project)
    ws._go_to_projects()
    assert seen == ["projects"]


def test_breadcrumb_shows_project_name_not_in_the_projects_link(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert ws._crumb_projects.text() == "Projects"
    assert project.name in ws._crumb_name.text()


def test_floating_add_point_selects_or_creates_a_points_layer(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._on_floating_tool("__add_point__")
    assert isinstance(ws._layers.selected, PointsLayer)
    assert ws._canvas.mode == PAINT

    ws._on_floating_tool(PAN_ZOOM)
    assert ws._canvas.mode == PAN_ZOOM


# ── labels-layer settings (contour/dims/checkboxes/colour) ───────────────────
def test_set_selected_label_updates_layer_and_rebuilds(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    ws._set_selected_label(seg, 7)
    assert seg.selected_label == 7


def test_pick_label_color_then_auto_clears_overrides(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    seg.selected_label = 3
    ws._pick_label_color(seg, "#ff8800")
    assert seg.color_overrides.get(3) == (0xFF, 0x88, 0x00)
    ws._set_color_mode(seg, "auto")
    assert seg.color_overrides == {}


def test_set_layer_int_attr_updates_contour_and_edit_dims(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    ws._set_layer_int_attr(seg, "contour", 5)
    assert seg.contour == 5
    ws._set_layer_int_attr(seg, "n_edit_dimensions", 3)
    assert seg.n_edit_dimensions == 3


def test_toggle_layer_bool_flips_each_labels_checkbox(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    for attr in ("contiguous", "preserve_labels", "show_selected_label"):
        before = getattr(seg, attr)
        ws._toggle_layer_bool(seg, attr)
        assert getattr(seg, attr) is not before


def test_set_brush_size_updates_layer_and_badge(app, segment, projects, toasts, tmp_path, storage):
    from studio.components import Badge
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    badge = Badge("1", theme.DARK)
    ws._set_brush_size(seg, 0.5, badge)
    assert seg.brush_size == 50
    assert badge.text() == "50"


# ── image-layer settings (gamma/colormap) ────────────────────────────────────
def test_set_image_gamma_updates_layer_and_badge(app, segment, projects, toasts, tmp_path, storage):
    from studio.components import Badge
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    img_layer = ws._layers.find(Path(project.image_paths[0]).stem)
    badge = Badge("1.00", theme.DARK)
    ws._set_image_gamma(img_layer, 0.5, badge)
    assert img_layer.gamma == pytest.approx(1.5)
    assert badge.text() == "1.50"


def test_set_image_colormap_updates_layer(app, segment, projects, toasts, tmp_path, storage):
    from studio.layer_model import IMAGE_COLORMAPS
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    img_layer = ws._layers.find(Path(project.image_paths[0]).stem)
    other = next(c for c in IMAGE_COLORMAPS if c != img_layer.colormap)
    ws._set_image_colormap(img_layer, other)
    assert img_layer.colormap == other


# ── points/shapes settings ────────────────────────────────────────────────────
def test_set_point_size_and_clear_points(app, segment, projects, toasts, tmp_path, storage):
    from studio.components import Badge
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._add_points_layer()
    pts = ws._layers.selected
    pts.add(5, 5)
    pts.add(10, 10)
    badge = Badge("10", theme.DARK)
    ws._set_point_size(pts, 0.5, badge)
    assert pts.size == 20
    assert badge.text() == "20"
    ws._clear_points(pts)
    assert pts.points == []


def test_set_edge_width_and_clear_shapes(app, segment, projects, toasts, tmp_path, storage):
    from studio.components import Badge
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._add_shapes_layer()
    shapes = ws._layers.selected
    shapes.add("polygon", [(1, 1), (1, 5), (5, 5)])
    badge = Badge("1.0", theme.DARK)
    ws._set_edge_width(shapes, 0.5, badge)
    assert shapes.edge_width == pytest.approx(5.0)
    assert badge.text() == "5.0"
    ws._clear_shapes(shapes)
    assert shapes.shapes == []


# ── overlay visibility toggles ────────────────────────────────────────────────
def test_toggle_show_predictions_hides_segmentation_not_gt(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, with_gt=True)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    gt_path = tmp_path / "img_0_mask.png"
    ws._load_gt(str(gt_path))
    ws._toggle_show_predictions(False)
    assert ws._layers.find("Segmentation").visible is False
    assert ws._layers.find("Ground truth").visible is True
    ws._toggle_show_predictions(True)
    assert ws._layers.find("Segmentation").visible is True


def test_toggle_show_gt_without_gt_toasts(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._toggle_show_gt(True)
    assert toasts and "No ground truth loaded" in toasts[-1][0]


def test_toggle_show_gt_with_gt_flips_visibility(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, with_gt=True)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._load_gt(str(tmp_path / "img_0_mask.png"))
    assert ws._layers.find("Ground truth").visible is True
    ws._toggle_show_gt(False)
    assert ws._layers.find("Ground truth").visible is False


# ── engine-specific settings (SAM2 / backbone) ────────────────────────────────
def test_backbone_select_updates_vit_name(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._on_backbone_select("ViT-B")
    assert project.settings.vit_name == "vit_b"


def test_sam2_model_and_tracking_mode_selects(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    project.settings.engine = "sam2"
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._on_sam2_model_select("Small")
    assert project.settings.sam2_model == "small"
    ws._on_sam2_tracking_mode(1)
    assert project.settings.sam2_tracking_mode == "propagate"
    ws._on_sam2_tracking_mode(0)
    assert project.settings.sam2_tracking_mode == "independent"


def test_generic_set_setting_roundtrips_a_plain_toggle(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert project.settings.clahe is False
    ws._set_setting("clahe", True)
    assert project.settings.clahe is True
    assert project.settings.quality_preset == "Balanced"  # plain _set_setting never marks Custom


# ── refine / measurements ─────────────────────────────────────────────────────
def test_refine_button_toasts_honest_coming_soon(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._refine_coming_soon()
    assert toasts and "coming soon" in toasts[-1][0].lower()


def test_show_measurements_without_a_result_toasts(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._show_measurements()
    assert toasts and "No measurements yet" in toasts[-1][0]


def test_show_measurements_after_predict_shows_a_summary(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    ws._show_measurements()
    assert toasts[-1][0] == "Measurements"


def test_export_csv_without_a_result_toasts(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._export_csv()
    assert toasts and "Nothing to export" in toasts[-1][0]
