"""Tests for the Segment tab's predict/GT/batch/benchmark controller
(studio/segment_controller.py).

Pure-logic — no Qt, and no *real* torch/model either: every test that
exercises a predict run monkeypatches ``velum_core.inference_cache
.predict_cached`` (the one seam the "cellseg1" engine calls into,
regardless of which higher-level path reaches it), so the exact real
production call chain (build_params -> build_config -> engine dispatch ->
callbacks) runs for real, at real cv2/numpy speed, without a GPU, SAM
weights, or torch actually loading anything. Only the file *paths* for the
LoRA/SAM backbone need to exist (checked with Path.exists(), never opened).
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from studio.project import Project, ProjectSettings, ProjectStats
from studio.segment_controller import (
    QUALITY_PRESETS,
    SegmentController,
    apply_quality_preset,
)


def _write_image(path: Path, size: int = 32, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), (rng.random((size, size, 3)) * 255).astype(np.uint8))


def _write_mask(path: Path, *labels_and_boxes, size: int = 32) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint16)
    for label, (y0, y1, x0, x1) in labels_and_boxes:
        mask[y0:y1, x0:x1] = label
    cv2.imwrite(str(path), mask)
    return mask


def _fake_predict_cached(config, image_rgb):
    """Stand-in for velum_core.inference_cache.predict_cached: two fixed
    blobs regardless of input, fast and fully deterministic.

    Runs on the *resized* image (default 512x512 -- see config["resize_size"])
    and its result is then nearest-neighbour-downsampled back to the
    original image size by the real _predict_cached — so each blob is sized
    as a fraction of the frame, not a fixed pixel count, or it can vanish
    entirely under a large downsampling ratio (e.g. 512 -> a 32px test image).
    """
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[h // 8 : h // 2, w // 8 : w // 2] = 1
    mask[h // 2 : h - h // 8, w // 2 : w - w // 8] = 2
    return mask


@pytest.fixture(autouse=True)
def _fake_engine(monkeypatch):
    monkeypatch.setattr("velum_core.inference_cache.predict_cached", _fake_predict_cached)
    # cellpose is a real, heavy [project.dependencies] package that may well
    # be installed in whatever env runs these tests (it is, in the full conda
    # env) — without this, list_available_engines()/run_benchmark_async see
    # it as real and available and actually run genuine (slow, possibly
    # model-downloading) Cellpose inference instead of the fake above, which
    # only stands in for the "cellseg1" engine's predict_cached seam.
    monkeypatch.setattr("velum_core.engines.cellpose_available", lambda: False)


@pytest.fixture
def storage(tmp_path):
    d = tmp_path / "storage"
    (d / "sam_backbone").mkdir(parents=True)
    (d / "sam_backbone" / "sam_vit_h_4b8939.pth").write_bytes(b"fake-weights")
    (d / "loras").mkdir(parents=True)
    lora = d / "loras" / "nuclei-dapi-r8.pth"
    lora.write_bytes(b"fake-lora")
    return d


@pytest.fixture
def ctrl(storage):
    return SegmentController(storage_dir=storage)


@pytest.fixture
def project_with_image(tmp_path, storage):
    img = tmp_path / "img_001.png"
    _write_image(img, seed=1)
    lora = storage / "loras" / "nuclei-dapi-r8.pth"
    settings = ProjectSettings(engine="cellseg1", model_name=str(lora))
    return Project(id="p1", name="P1", image_paths=[str(img)], settings=settings)


def _join(thread, timeout=10):
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "background thread did not finish in time"


# ── quality presets ──────────────────────────────────────────────────────────
def test_apply_quality_preset_sets_thresholds_and_name():
    s = ProjectSettings()
    apply_quality_preset(s, "Accurate")
    assert s.quality_preset == "Accurate"
    assert s.points_per_side == QUALITY_PRESETS["Accurate"]["points_per_side"]
    assert s.pred_iou_thresh == QUALITY_PRESETS["Accurate"]["pred_iou_thresh"]


def test_apply_quality_preset_unknown_name_is_a_noop():
    s = ProjectSettings(quality_preset="Balanced")
    apply_quality_preset(s, "Nonsense")
    assert s.quality_preset == "Balanced"


# ── build_params / build_config ──────────────────────────────────────────────
def test_build_params_maps_every_settings_field(ctrl, project_with_image):
    params = ctrl.build_params(project_with_image, project_with_image.image_paths[0])
    assert params["engine"] == "cellseg1"
    assert params["lora_custom_text"] == project_with_image.settings.model_name
    assert params["resize_size"] == project_with_image.settings.resize_size
    assert params["storage_dir"] == str(ctrl.storage_dir)
    assert params["zstack"] is False
    assert params["device"] in ("cpu", "mps", "0")


def test_build_config_resolves_a_real_engine_config(ctrl, project_with_image):
    config = ctrl.build_config(project_with_image, project_with_image.image_paths[0])
    assert config["engine"] == "cellseg1"
    assert Path(config["result_pth_path"]) == Path(project_with_image.settings.model_name)
    assert Path(config["model_path"]).name == "sam_vit_h_4b8939.pth"


def test_build_config_missing_lora_raises_clear_error(ctrl, project_with_image):
    project_with_image.settings.model_name = ""
    with pytest.raises(ValueError, match="LoRA"):
        ctrl.build_config(project_with_image, project_with_image.image_paths[0])


# ── engines / models listing ─────────────────────────────────────────────────
def test_list_available_engines_includes_the_three_built_ins(ctrl):
    keys = {key for key, _label, _avail in ctrl.list_available_engines()}
    assert {"cellseg1", "cellpose", "sam2"} <= keys


def test_list_lora_models_finds_checkpoints_with_sidecars(ctrl, storage):
    sidecar = {"vit_name": "vit_h", "image_encoder_lora_rank": 8, "saved_at": "2026-01-01T00:00:00"}
    (storage / "loras" / "nuclei-dapi-r8.json").write_text(json.dumps(sidecar))
    names = [m.name for m in ctrl.list_lora_models()]
    assert "nuclei-dapi-r8" in names


# ── single-image predict (real pipeline, fake engine) ────────────────────────
def test_run_predict_async_calls_back_with_mask_and_logs(ctrl, project_with_image):
    results = []
    logs = []
    finished = []
    thread = ctrl.run_predict_async(
        project_with_image, project_with_image.image_paths[0],
        on_result=lambda img, mask, stack: results.append((img, mask, stack)),
        on_log=logs.append, on_finish=lambda: finished.append(True))
    _join(thread)
    assert finished == [True]
    assert len(results) == 1
    img, mask, _stack = results[0]
    assert mask.max() == 2  # the two fake blobs
    assert any("cells" in line for line in logs)


def test_run_predict_async_raises_synchronously_for_a_missing_image(ctrl, project_with_image, tmp_path):
    """Config-building errors (bad path, missing LoRA, ...) raise synchronously,
    *before* any thread starts — matching TrainController.build_config's own
    contract, so the caller (workspace.py) can catch and toast immediately,
    exactly like ModelsScreen._start_training already does for training."""
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(ValueError, match="Image not found"):
        ctrl.run_predict_async(project_with_image, str(missing))


def test_run_predict_async_reports_a_read_failure_via_on_log(ctrl, project_with_image, tmp_path):
    """A file that exists but can't actually be decoded fails *inside* the
    background thread instead — reported through on_log, not raised."""
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a real image")
    project_with_image.image_paths = [str(corrupt)]
    logs = []
    finished = []
    thread = ctrl.run_predict_async(
        project_with_image, str(corrupt), on_log=logs.append,
        on_finish=lambda: finished.append(True))
    _join(thread)
    assert finished == [True]
    assert any("[ERROR]" in line for line in logs)


# ── measurements / colour-by ──────────────────────────────────────────────────
def test_compute_measurements_reports_two_cells(ctrl):
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[2:6, 2:6] = 1
    mask[10:16, 10:16] = 2
    result = ctrl.compute_measurements(mask)
    assert result["n_cells"] == 2
    assert "area" in ctrl.summary_line(result)  # the 2-D summary line's shape


def test_color_overrides_and_measurement_range(ctrl):
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[2:6, 2:6] = 1
    mask[10:16, 10:16] = 2
    result = ctrl.compute_measurements(mask)
    overrides = ctrl.color_overrides_for(result, "area")
    assert set(overrides) == {1, 2}
    lo, hi = ctrl.measurement_range(result, "area")
    assert hi > lo


# ── ground truth + evaluation ─────────────────────────────────────────────────
def test_find_gt_for_image_sibling_convention(tmp_path):
    img = tmp_path / "a.png"
    _write_image(img)
    _write_mask(tmp_path / "a_mask.png", (1, (2, 6, 2, 6)))
    assert SegmentController.find_gt_for_image(img) == tmp_path / "a_mask.png"


def test_load_gt_mask_resizes_to_match_prediction(tmp_path, ctrl):
    gt_path = tmp_path / "gt.png"
    _write_mask(gt_path, (1, (2, 6, 2, 6)), size=32)
    gt = ctrl.load_gt_mask(gt_path, target_shape=(64, 64))
    assert gt.shape == (64, 64)
    assert gt.max() == 1


def test_evaluate_against_gt_perfect_match_scores_f1_one(tmp_path, ctrl):
    pred = np.zeros((20, 20), dtype=np.int32)
    pred[2:10, 2:10] = 1
    gt_path = tmp_path / "gt.png"
    cv2.imwrite(str(gt_path), pred.astype(np.uint16))
    metrics = ctrl.evaluate_against_gt(gt_path, pred)
    assert metrics["f1"] == pytest.approx(1.0)


def test_evaluate_masks_perfect_match_scores_f1_one(ctrl):
    pred = np.zeros((20, 20), dtype=np.int32)
    pred[2:10, 2:10] = 1
    metrics = ctrl.evaluate_masks(pred.copy(), pred)
    assert metrics["f1"] == pytest.approx(1.0)


def test_discover_gt_pairs_only_includes_images_with_a_mask(tmp_path, ctrl):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _write_image(a)
    _write_image(b)
    _write_mask(tmp_path / "a_mask.png", (1, (2, 6, 2, 6)))
    project = Project(id="p", name="P", image_paths=[str(a), str(b)])
    pairs = ctrl.discover_gt_pairs(project)
    assert [p[0] for p in pairs] == [a]


# ── save / export ─────────────────────────────────────────────────────────────
def test_save_mask_writes_a_readable_file(tmp_path, ctrl):
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[1:3, 1:3] = 5
    out = ctrl.save_mask(mask, tmp_path / "out" / "mask.png")
    assert out.exists()
    reread = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert reread.max() == 5


# ── persisted per-(project, image) results ───────────────────────────────────
def test_load_result_mask_is_none_before_any_save(ctrl, tmp_path):
    project = Project(id="p1", name="P1")
    assert ctrl.has_result_mask(project, tmp_path / "img.png") is False
    assert ctrl.load_result_mask(project, tmp_path / "img.png") is None


def test_save_then_load_result_mask_roundtrips(ctrl, tmp_path):
    project = Project(id="p1", name="P1")
    mask = np.zeros((12, 14), dtype=np.int32)
    mask[2:5, 2:5] = 7
    ctrl.save_result_mask(project, tmp_path / "img.png", mask)
    assert ctrl.has_result_mask(project, tmp_path / "img.png") is True
    reloaded = ctrl.load_result_mask(project, tmp_path / "img.png")
    assert reloaded is not None
    assert reloaded.shape == mask.shape
    assert reloaded.max() == 7
    assert (reloaded == mask).all()


def test_mask_path_differs_by_project_and_by_image(ctrl, tmp_path):
    p1 = Project(id="p1", name="P1")
    p2 = Project(id="p2", name="P2")
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    assert ctrl.mask_path_for_image(p1, img_a) != ctrl.mask_path_for_image(p2, img_a)
    assert ctrl.mask_path_for_image(p1, img_a) != ctrl.mask_path_for_image(p1, img_b)
    # same project + same image path -> same cache file, deterministically
    assert ctrl.mask_path_for_image(p1, img_a) == ctrl.mask_path_for_image(p1, img_a)


def test_same_filename_different_folders_do_not_collide(ctrl, tmp_path):
    """Two images that happen to share a filename in different folders must
    not overwrite each other's cached result."""
    project = Project(id="p1", name="P1")
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    mask_a = np.full((5, 5), 1, dtype=np.int32)
    mask_b = np.full((5, 5), 2, dtype=np.int32)
    ctrl.save_result_mask(project, dir_a / "img.png", mask_a)
    ctrl.save_result_mask(project, dir_b / "img.png", mask_b)
    assert ctrl.load_result_mask(project, dir_a / "img.png").max() == 1
    assert ctrl.load_result_mask(project, dir_b / "img.png").max() == 2


def test_export_measurements_csv_has_a_header_row(tmp_path, ctrl):
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[1:3, 1:3] = 1
    result = ctrl.compute_measurements(mask)
    out = ctrl.export_measurements_csv(result, tmp_path / "out" / "m.csv")
    text = out.read_text()
    assert "Area" in text
    assert text.count("\n") >= 1


def test_project_run_dir_is_per_project_and_created(ctrl):
    p1 = ctrl.project_run_dir(Project(id="alpha", name="Alpha"))
    p2 = ctrl.project_run_dir(Project(id="beta", name="Beta"))
    assert p1 != p2
    assert p1.is_dir() and p2.is_dir()


# ── batch prediction ─────────────────────────────────────────────────────────
def test_run_batch_async_updates_stats_and_writes_cohort_csvs(ctrl, tmp_path, storage):
    imgs = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        _write_image(p, seed=i)
        imgs.append(str(p))
    lora = storage / "loras" / "nuclei-dapi-r8.pth"
    project = Project(id="batchp", name="BatchP", image_paths=imgs,
                      settings=ProjectSettings(engine="cellseg1", model_name=str(lora)))

    cohort_ready = []
    finished = []
    thread = ctrl.run_batch_async(
        project, on_cohort_ready=lambda records, out_dir: cohort_ready.append((records, out_dir)),
        on_finish=lambda: finished.append(True))
    _join(thread)

    assert finished == [True]
    assert len(cohort_ready) == 1
    records, out_dir = cohort_ready[0]
    assert len(records) == 3
    assert (Path(out_dir) / "cohort_measurements.csv").exists()
    assert (Path(out_dir) / "cohort_summary.csv").exists()
    assert project.stats.n_cells == 3 * 2  # two fake blobs per image
    assert project.stats.progress == 100
    assert project.stats.n_images == 3
    # each batch-computed mask is also reloadable later, same as a single Run
    for img in imgs:
        assert ctrl.has_result_mask(project, img)
        assert ctrl.load_result_mask(project, img).max() == 2


def test_run_batch_async_raises_for_an_empty_project(ctrl):
    with pytest.raises(ValueError, match="no images"):
        ctrl.run_batch_async(Project(id="empty", name="Empty"))


# ── benchmark engines vs GT ───────────────────────────────────────────────────
def test_run_benchmark_async_raises_without_any_gt(ctrl, tmp_path, storage):
    img = tmp_path / "a.png"
    _write_image(img)
    lora = storage / "loras" / "nuclei-dapi-r8.pth"
    project = Project(id="p", name="P", image_paths=[str(img)],
                      settings=ProjectSettings(engine="cellseg1", model_name=str(lora)))
    with pytest.raises(ValueError, match="ground-truth"):
        ctrl.run_benchmark_async(project)


def test_run_benchmark_async_scores_the_available_engine(ctrl, tmp_path, storage):
    img = tmp_path / "img_001.png"
    _write_image(img, size=32)
    _write_mask(tmp_path / "img_001_mask.png", (1, (2, 8, 2, 8)), (2, (22, 30, 22, 30)), size=32)
    lora = storage / "loras" / "nuclei-dapi-r8.pth"
    project = Project(id="p", name="P", image_paths=[str(img)],
                      settings=ProjectSettings(engine="cellseg1", model_name=str(lora)))

    done = []
    thread = ctrl.run_benchmark_async(project, on_done=lambda cols, rows: done.append((cols, rows)))
    _join(thread)
    assert len(done) == 1
    cols, rows = done[0]
    assert "engine" in cols
    names = [r[0] for r in rows]
    assert any("CellSeg1" in n for n in names)


# ── Dashboard integration hook ────────────────────────────────────────────────
def test_record_run_mutates_stats_without_saving():
    project = Project(id="p", name="P", image_paths=["a.png", "b.png"],
                      stats=ProjectStats(n_cells=0, progress=0))
    SegmentController.record_run(project, n_cells=42, f1=0.91, progress=150)
    assert project.stats.n_cells == 42
    assert project.stats.last_f1 == pytest.approx(0.91)
    assert project.stats.progress == 100  # clamped
    assert project.stats.n_images == 2


def test_record_run_only_touches_given_fields():
    project = Project(id="p", name="P", stats=ProjectStats(n_cells=7, last_f1=0.5, progress=50))
    SegmentController.record_run(project, n_cells=9)
    assert project.stats.n_cells == 9
    assert project.stats.last_f1 == 0.5
    assert project.stats.progress == 50
