"""Tests for the Studio Models & Train tab controller (studio/train_controller.py).

Pure-logic, no Qt/torch/napari — runs under the light CI `test` group (numpy
and opencv-python-headless are in that group; heavy deps like torch are only
ever imported lazily, inside ``TrainController.start_training``, never here).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from studio.train_controller import (
    TrainController,
    available_backbones,
    count_cells_in_dir,
    count_cells_in_mask,
    duration_str,
    find_mask_for_image,
    guess_vit_name,
    list_recent_runs,
    list_trained_models,
)


def _write_image(path, size=16):
    cv2.imwrite(str(path), (np.random.rand(size, size, 3) * 255).astype(np.uint8))


def _write_mask(path, *labels_and_boxes):
    """``labels_and_boxes``: (label, (y0, y1, x0, x1)) pairs on a 16x16 canvas."""
    mask = np.zeros((16, 16), dtype=np.uint16)
    for label, (y0, y1, x0, x1) in labels_and_boxes:
        mask[y0:y1, x0:x1] = label
    cv2.imwrite(str(path), mask)
    return mask


@pytest.fixture
def storage(tmp_path):
    return tmp_path / "storage"


@pytest.fixture
def ctrl(storage):
    return TrainController(storage_dir=storage)


def _add_backbone(storage, vit_name="vit_h"):
    d = storage / "sam_backbone"
    d.mkdir(parents=True, exist_ok=True)
    names = {"vit_h": "sam_vit_h_4b8939.pth", "vit_l": "sam_vit_l_0b3195.pth", "vit_b": "sam_vit_b_01ec64.pth"}
    (d / names[vit_name]).write_bytes(b"fake-weights")


# ── duration_str ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seconds,expected", [
    (0, "0m 00s"), (5, "0m 05s"), (62, "1m 02s"), (372, "6m 12s"), (-3, "0m 00s"),
])
def test_duration_str(seconds, expected):
    assert duration_str(seconds) == expected


# ── available_backbones ──────────────────────────────────────────────────────
def test_available_backbones_empty_when_none_present(storage):
    assert available_backbones(storage / "sam_backbone") == []


def test_available_backbones_lists_only_existing_files(storage):
    _add_backbone(storage, "vit_h")
    _add_backbone(storage, "vit_b")
    result = available_backbones(storage / "sam_backbone")
    assert result == [("vit_h", "ViT-H"), ("vit_b", "ViT-B")]


# ── find_mask_for_image ──────────────────────────────────────────────────────
def test_find_mask_for_image_sibling_suffix(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    _write_image(work / "a.png")
    _write_mask(work / "a_mask.png", (1, (2, 6, 2, 6)))
    assert find_mask_for_image(work / "a.png") == work / "a_mask.png"


def test_find_mask_for_image_masks_subfolder(tmp_path):
    work = tmp_path / "work"
    (work / "masks").mkdir(parents=True)
    _write_image(work / "a.png")
    _write_mask(work / "masks" / "a.png", (1, (2, 6, 2, 6)))
    assert find_mask_for_image(work / "a.png") == work / "masks" / "a.png"


def test_find_mask_for_image_shared_mask_dir_param(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    shared_masks = tmp_path / "shared_masks"
    shared_masks.mkdir()
    _write_image(work / "a.png")
    _write_mask(shared_masks / "a.png", (1, (2, 6, 2, 6)))
    assert find_mask_for_image(work / "a.png", mask_dir=shared_masks) == shared_masks / "a.png"
    assert find_mask_for_image(work / "a.png") is None  # not found without the mask_dir hint


def test_find_mask_for_image_returns_none_when_missing(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    _write_image(work / "a.png")
    assert find_mask_for_image(work / "a.png") is None


# ── cell counting ────────────────────────────────────────────────────────────
def test_count_cells_in_mask_reads_real_mask(tmp_path):
    p = tmp_path / "m.png"
    _write_mask(p, (1, (0, 4, 0, 4)), (2, (5, 9, 5, 9)), (3, (10, 14, 10, 14)))
    assert count_cells_in_mask(p) == 3


def test_count_cells_in_mask_returns_none_for_bad_path(tmp_path):
    assert count_cells_in_mask(tmp_path / "nope.png") is None


def test_count_cells_in_dir_sums_and_ignores_non_mask_files(tmp_path):
    d = tmp_path / "masks"
    d.mkdir()
    _write_mask(d / "a.png", (1, (0, 4, 0, 4)), (2, (5, 9, 5, 9)))  # 2 cells
    _write_mask(d / "b.png", (1, (0, 4, 0, 4)))  # 1 cell
    (d / "readme.txt").write_text("not a mask")
    assert count_cells_in_dir(d) == 3


def test_count_cells_in_dir_returns_none_for_missing_dir(tmp_path):
    assert count_cells_in_dir(tmp_path / "nope") is None


# ── backbone resolution ──────────────────────────────────────────────────────
def test_resolve_backbone_raises_when_missing(ctrl):
    with pytest.raises(ValueError, match="not found"):
        ctrl.resolve_backbone("vit_h")


def test_resolve_backbone_returns_path_when_present(ctrl, storage):
    _add_backbone(storage, "vit_h")
    path = ctrl.resolve_backbone("vit_h")
    assert path == storage / "sam_backbone" / "sam_vit_h_4b8939.pth"


# ── guess_vit_name ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("name,expected", [
    ("sam_vit_h_4b8939.pth", "vit_h"),
    ("sam_vit_l_0b3195.pth", "vit_l"),
    ("sam_vit_b_01ec64.pth", "vit_b"),
    ("my-custom-vit-l-weights.pth", "vit_l"),
    ("totally_unrelated_name.pth", "vit_h"),  # no hint -> flagship default
])
def test_guess_vit_name(name, expected, tmp_path):
    assert guess_vit_name(tmp_path / name) == expected


# ── build_config ─────────────────────────────────────────────────────────────
@pytest.fixture
def annotated_image(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    image = work / "cells.png"
    _write_image(image)
    mask = work / "cells_mask.png"
    _write_mask(mask, (1, (2, 6, 2, 6)), (2, (8, 12, 8, 12)))
    return image, mask


def test_build_config_raises_on_missing_image(ctrl, tmp_path):
    with pytest.raises(ValueError, match="Image not found"):
        ctrl.build_config(image_path=tmp_path / "nope.png", mask_path=tmp_path / "nope_mask.png",
                          vit_name="vit_h", lora_rank=8, epochs=100)


def test_build_config_raises_on_missing_mask(ctrl, annotated_image, tmp_path):
    image, _ = annotated_image
    with pytest.raises(ValueError, match="Mask not found"):
        ctrl.build_config(image_path=image, mask_path=tmp_path / "nope_mask.png",
                          vit_name="vit_h", lora_rank=8, epochs=100)


def test_build_config_raises_on_missing_backbone(ctrl, annotated_image):
    image, mask = annotated_image
    with pytest.raises(ValueError, match="backbone"):
        ctrl.build_config(image_path=image, mask_path=mask, vit_name="vit_h", lora_rank=8, epochs=100)


def test_build_config_raises_cleanly_when_nothing_picked_yet(ctrl):
    """Regression test: a screen calling this before the user has picked an
    image (both paths still None, e.g. a defensive/direct call bypassing the
    UI's own disabled-button guard) must get a clean ValueError, not an
    uncaught TypeError from Path(None) — the caller only catches ValueError."""
    with pytest.raises(ValueError, match="No annotated image"):
        ctrl.build_config(image_path=None, mask_path=None, vit_name="vit_h", lora_rank=8, epochs=100)


def test_build_config_raises_when_no_backbone_selected_at_all(ctrl, annotated_image):
    """Regression test for the reported bug: with nothing auto-detected in
    sam_backbone_dir and no manual backbone_path given either, this must
    fail with a clear message -- not silently resolve to some default."""
    image, mask = annotated_image
    with pytest.raises(ValueError, match="No SAM backbone selected"):
        ctrl.build_config(image_path=image, mask_path=mask, vit_name=None,
                          lora_rank=8, epochs=100)


def test_build_config_manual_backbone_path_used_directly(ctrl, annotated_image, tmp_path):
    """The browse-for-a-checkpoint fallback: no vit_name, an explicit
    backbone_path instead -- used as-is, with the architecture guessed from
    its filename."""
    image, mask = annotated_image
    custom = tmp_path / "my_sam_vit_l_backbone.pth"
    custom.write_bytes(b"fake-weights")
    config = ctrl.build_config(image_path=image, mask_path=mask, vit_name=None,
                               backbone_path=custom, lora_rank=8, epochs=100)
    assert config["model_path"] == str(custom)
    assert config["vit_name"] == "vit_l"


def test_build_config_manual_backbone_path_respects_explicit_vit_name(ctrl, annotated_image, tmp_path):
    image, mask = annotated_image
    custom = tmp_path / "ambiguous_name.pth"
    custom.write_bytes(b"fake-weights")
    config = ctrl.build_config(image_path=image, mask_path=mask, vit_name="vit_b",
                               backbone_path=custom, lora_rank=8, epochs=100)
    assert config["vit_name"] == "vit_b"  # explicit override wins over the filename guess


def test_build_config_manual_backbone_path_raises_when_missing(ctrl, annotated_image, tmp_path):
    image, mask = annotated_image
    with pytest.raises(ValueError, match="backbone not found"):
        ctrl.build_config(image_path=image, mask_path=mask, vit_name=None,
                          backbone_path=tmp_path / "nope.pth", lora_rank=8, epochs=100)


def test_build_config_copies_into_an_isolated_run_dir_without_touching_originals(ctrl, storage, annotated_image):
    _add_backbone(storage, "vit_h")
    image, mask = annotated_image
    config = ctrl.build_config(image_path=image, mask_path=mask, vit_name="vit_h", lora_rank=8, epochs=100)
    assert Path(config["train_image_dir"]).is_relative_to(ctrl.run_data_dir)
    assert Path(config["train_mask_dir"]).is_relative_to(ctrl.run_data_dir)
    assert (Path(config["train_image_dir"]) / image.name).exists()
    assert (Path(config["train_mask_dir"]) / mask.name).exists()
    assert image.exists() and mask.exists()  # originals untouched
    assert Path(config["train_image_dir"]) != image.parent  # not the classic shared folder


def test_build_config_reflects_rank_and_epochs(ctrl, storage, annotated_image):
    _add_backbone(storage, "vit_h")
    image, mask = annotated_image
    config = ctrl.build_config(image_path=image, mask_path=mask, vit_name="vit_h", lora_rank=16, epochs=250)
    assert config["image_encoder_lora_rank"] == 16
    assert config["mask_decoder_lora_rank"] == 16
    assert config["epoch_max"] == 250
    assert config["vit_name"] == "vit_h"
    assert config["train_id"] == [0]


def test_build_config_two_runs_get_distinct_isolated_dirs(ctrl, storage, annotated_image):
    _add_backbone(storage, "vit_h")
    image, mask = annotated_image
    c1 = ctrl.build_config(image_path=image, mask_path=mask, vit_name="vit_h", lora_rank=8, epochs=100)
    c2 = ctrl.build_config(image_path=image, mask_path=mask, vit_name="vit_h", lora_rank=8, epochs=100)
    assert c1["train_image_dir"] != c2["train_image_dir"]


# ── list_trained_models ──────────────────────────────────────────────────────
def _write_sidecar(lora_dir, stem, *, vit_name="vit_h", rank=8, n_images=1,
                   saved_at="2026-01-01T00:00:00+00:00", loss_history=None,
                   train_mask_dir=None, epoch_max=100):
    lora_dir.mkdir(parents=True, exist_ok=True)
    (lora_dir / f"{stem}.pth").write_bytes(b"fake")
    sidecar = {
        "vit_name": vit_name, "image_encoder_lora_rank": rank,
        "train_id": list(range(n_images)), "saved_at": saved_at,
        "epoch_max": epoch_max, "loss_history": loss_history or [],
    }
    if train_mask_dir is not None:
        sidecar["train_mask_dir"] = str(train_mask_dir)
    (lora_dir / f"{stem}.json").write_text(json.dumps(sidecar))


def test_list_trained_models_empty_when_no_sidecars(tmp_path):
    assert list_trained_models(tmp_path / "loras") == []


def test_list_trained_models_parses_sidecar_fields(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_sidecar(lora_dir, "lora_vit_h_r8", vit_name="vit_h", rank=8, n_images=1)
    models = list_trained_models(lora_dir)
    assert len(models) == 1
    m = models[0]
    assert m.name == "lora_vit_h_r8"
    assert m.checkpoint == lora_dir / "lora_vit_h_r8.pth"
    assert m.meta == "ViT-H · rank 8 · 1 image"
    assert m.f1 is None  # never computed at train time


def test_list_trained_models_pluralises_image_count(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_sidecar(lora_dir, "many", n_images=5)
    assert "5 images" in list_trained_models(lora_dir)[0].meta


def test_list_trained_models_computes_real_cell_count_from_mask_dir(tmp_path):
    lora_dir = tmp_path / "loras"
    mask_dir = tmp_path / "masks_for_run"
    mask_dir.mkdir()
    _write_mask(mask_dir / "a.png", (1, (0, 4, 0, 4)), (2, (5, 9, 5, 9)), (3, (10, 14, 10, 14)))
    _write_sidecar(lora_dir, "m", train_mask_dir=mask_dir)
    assert list_trained_models(lora_dir)[0].n_cells == 3


def test_list_trained_models_skips_corrupt_sidecar(tmp_path):
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir(parents=True)
    (lora_dir / "broken.json").write_text("{not valid json")
    (lora_dir / "broken.pth").write_bytes(b"x")
    assert list_trained_models(lora_dir) == []


def test_list_trained_models_sorted_newest_first(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_sidecar(lora_dir, "older", saved_at="2026-01-01T00:00:00+00:00")
    _write_sidecar(lora_dir, "newer", saved_at="2026-06-01T00:00:00+00:00")
    names = [m.name for m in list_trained_models(lora_dir)]
    assert names == ["newer", "older"]


# ── list_recent_runs ─────────────────────────────────────────────────────────
def test_list_recent_runs_empty_when_no_history(ctrl):
    assert list_recent_runs(ctrl.state_manager) == []


def test_list_recent_runs_formats_completed_and_stopped(ctrl):
    ctrl.state_manager.append_history_entry({
        "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:06:02+00:00",
        "checkpoint": "/loras/done_run.pth", "status": "completed",
    })
    ctrl.state_manager.append_history_entry({
        "started_at": "2026-01-02T00:00:00+00:00", "finished_at": "2026-01-02T00:02:00+00:00",
        "checkpoint": "/loras/stopped_run.pth", "status": "stopped",
    })
    runs = list_recent_runs(ctrl.state_manager)
    by_name = {r.name: r for r in runs}
    assert by_name["done_run"].meta == "done · 6m 02s"
    assert by_name["done_run"].state == "done"
    assert by_name["stopped_run"].meta == "stopped · 2m 00s"
    assert by_name["stopped_run"].state == "stopped"


def test_list_recent_runs_respects_limit(ctrl):
    for i in range(5):
        ctrl.state_manager.append_history_entry({
            "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:01:00+00:00",
            "checkpoint": f"/loras/run{i}.pth", "status": "completed",
        })
    assert len(list_recent_runs(ctrl.state_manager, limit=2)) == 2


# ── TrainController: model management ────────────────────────────────────────
def test_import_model_copies_checkpoint_and_sidecar(ctrl, tmp_path):
    src = tmp_path / "external.pth"
    src.write_bytes(b"weights")
    src.with_suffix(".json").write_text(json.dumps({"vit_name": "vit_b"}))
    dst = ctrl.import_model(src)
    assert dst == ctrl.lora_out_dir / "external.pth"
    assert dst.exists()
    assert dst.with_suffix(".json").exists()


def test_import_model_without_sidecar_still_copies_checkpoint(ctrl, tmp_path):
    src = tmp_path / "external.pth"
    src.write_bytes(b"weights")
    dst = ctrl.import_model(src)
    assert dst.exists()
    assert not dst.with_suffix(".json").exists()


def test_import_model_raises_when_source_missing(ctrl, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        ctrl.import_model(tmp_path / "nope.pth")


def test_select_model_for_project_updates_settings(ctrl, tmp_path):
    lora_dir = ctrl.lora_out_dir
    _write_sidecar(lora_dir, "nuclei-r8", vit_name="vit_l", rank=16)
    model = ctrl.list_trained_models()[0]

    from studio.project import Project
    project = Project(id="p", name="P")
    ctrl.select_model_for_project(model, project)
    assert project.settings.engine == "cellseg1"
    assert project.settings.model_name == str(model.checkpoint)
    assert project.settings.vit_name == "vit_l"
    assert project.settings.lora_rank == 16


# ── live state when idle ─────────────────────────────────────────────────────
def test_is_training_false_and_current_run_none_when_idle(ctrl):
    assert ctrl.is_training() is False
    assert ctrl.current_run() is None
    assert ctrl.current_loss_history() == []
    assert ctrl.active_epoch_max() is None
