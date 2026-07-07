"""Unit tests for napari_app.core.predict_controller.PredictController.

Covers what PredictWidget used to do inline and untested: config building
(engine dispatch, LoRA/SAM path resolution, validation errors) and the
predict/batch/benchmark orchestration (threading + callback sequencing).
Unlike tests/test_predict_*_wiring.py this module needs neither PyQt6 nor
torch — PredictController and its dependencies (engines.py, inference_cache.py)
only import them lazily inside the functions that actually need them — so
these tests run in the lightweight CI job too.
"""
import threading

import numpy as np
import pytest
from scipy import ndimage

from napari_app.core.predict_controller import PredictController, ENGINE_LABELS


def _cc(tile_img: np.ndarray) -> np.ndarray:
    """Fake engine: connected components of the non-zero pixels."""
    fg = tile_img[..., 0] > 0 if tile_img.ndim == 3 else tile_img > 0
    return ndimage.label(fg)[0].astype(np.int32)


def _base_params(tmp_path, engine="cellseg1", **overrides):
    img = tmp_path / "img.png"
    img.write_bytes(b"not a real image - build_config only checks existence")
    params = {
        "engine": engine,
        "image_path": str(img),
        "resize_size": 512,
        "vit_name": "vit_h",
        "sam_path_text": "",
        "storage_dir": tmp_path,
        "lora_custom_text": "",
        "lora_combo_text": "mylora",
        "lora_paths": {},
        "lora_rank": 4,
        "device": "cpu",
        "half_precision": False,
        "compile_decoder": False,
        "points_per_side": 32,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.6,
        "box_nms_thresh": 0.05,
        "min_mask_area": 20,
        "clahe": False,
        "tiled": False,
        "cp_diameter": 0,
        "cp_flow_threshold": 0.4,
        "cp_cellprob_threshold": 0.0,
        "channels": None,
    }
    params.update(overrides)
    return params


# ── Config building ───────────────────────────────────────────────────────────

def test_build_config_missing_image_raises(tmp_path):
    params = _base_params(tmp_path)
    params["image_path"] = str(tmp_path / "nope.png")
    with pytest.raises(ValueError, match="Image not found"):
        PredictController.build_config(params)


def test_build_config_cellpose_shape(tmp_path, monkeypatch):
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "cellpose_available", lambda: True)
    params = _base_params(tmp_path, engine="cellpose", resize_size=256)
    cfg = PredictController.build_config(params)
    assert cfg == {
        "engine": "cellpose", "image_path": params["image_path"],
        "resize_size": [256, 256],
        "cp_diameter": 0, "cp_flow_threshold": 0.4, "cp_cellprob_threshold": 0.0,
        "selected_device": "cpu", "clahe": False, "tiled": False,
        "tile_size": 256, "tile_overlap": 0,
        "vit_name": "vit_h", "image_encoder_lora_rank": 4,
        "sam_image_size": 256, "result_pth_path": "",
        "channels": None,
        "zstack": False, "stitch_iou": 0.25,
    }


def test_build_config_cellpose_threads_zstack_through(tmp_path, monkeypatch):
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "cellpose_available", lambda: True)
    params = _base_params(tmp_path, engine="cellpose", zstack=True, stitch_iou=0.4)
    cfg = PredictController.build_config(params)
    assert cfg["zstack"] is True
    assert cfg["stitch_iou"] == 0.4


def test_build_config_cellpose_unavailable_raises(tmp_path, monkeypatch):
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "cellpose_available", lambda: False)
    params = _base_params(tmp_path, engine="cellpose")
    with pytest.raises(ValueError, match="Cellpose is not installed"):
        PredictController.build_config(params)


def _with_sam_backbone(tmp_path, vit_name="vit_h"):
    """Create the SAM backbone file resolve_sam expects, so a test can get
    past it and exercise the LoRA-checking logic beyond."""
    names = {"vit_h": "sam_vit_h_4b8939.pth", "vit_l": "sam_vit_l_0b3195.pth",
             "vit_b": "sam_vit_b_01ec64.pth"}
    backbone_dir = tmp_path / "sam_backbone"; backbone_dir.mkdir(exist_ok=True)
    sam = backbone_dir / names[vit_name]; sam.write_bytes(b"x")
    return sam


def test_build_config_non_cellpose_dispatches_to_sam_config(tmp_path):
    _with_sam_backbone(tmp_path)
    lora = tmp_path / "my.pth"; lora.write_bytes(b"x")
    params = _base_params(tmp_path, lora_paths={"mylora": str(lora)})
    cfg = PredictController.build_config(params)
    assert cfg["engine"] == "cellseg1"
    assert cfg["result_pth_path"] == str(lora)


def test_sam_config_missing_lora_raises(tmp_path):
    _with_sam_backbone(tmp_path)
    params = _base_params(tmp_path)  # lora_combo_text unresolved, lora_paths empty
    with pytest.raises(ValueError, match="LoRA checkpoint not found"):
        PredictController.sam_config(params)


def test_sam_config_missing_sam_backbone_raises(tmp_path):
    lora = tmp_path / "my.pth"; lora.write_bytes(b"x")
    params = _base_params(tmp_path, lora_paths={"mylora": str(lora)})
    with pytest.raises(ValueError, match="SAM backbone not found"):
        PredictController.sam_config(params)


def test_sam_config_resolves_paths_and_shape(tmp_path):
    lora = tmp_path / "my.pth"; lora.write_bytes(b"x")
    backbone_dir = tmp_path / "sam_backbone"; backbone_dir.mkdir()
    sam = backbone_dir / "sam_vit_h_4b8939.pth"; sam.write_bytes(b"x")
    params = _base_params(tmp_path, lora_paths={"mylora": str(lora)},
                           resize_size=1024, lora_rank=8)
    cfg = PredictController.sam_config(params)
    assert cfg["engine"] == "cellseg1"
    assert cfg["model_path"] == str(sam)
    assert cfg["result_pth_path"] == str(lora)
    assert cfg["resize_size"] == [1024, 1024]
    assert cfg["image_encoder_lora_rank"] == 8
    assert cfg["mask_decoder_lora_rank"] == 8
    assert cfg["points_per_side"] == 32
    assert cfg["freeze_image_encoder"] is True   # LoRA-only fine-tune assumption


def test_sam_config_perf_flags_default_off(tmp_path):
    lora = tmp_path / "my.pth"; lora.write_bytes(b"x")
    backbone_dir = tmp_path / "sam_backbone"; backbone_dir.mkdir()
    (backbone_dir / "sam_vit_h_4b8939.pth").write_bytes(b"x")
    params = _base_params(tmp_path, lora_paths={"mylora": str(lora)})
    cfg = PredictController.sam_config(params)
    assert cfg["half_precision"] is False
    assert cfg["compile_decoder"] is False


def test_sam_config_threads_perf_flags_through(tmp_path):
    lora = tmp_path / "my.pth"; lora.write_bytes(b"x")
    backbone_dir = tmp_path / "sam_backbone"; backbone_dir.mkdir()
    (backbone_dir / "sam_vit_h_4b8939.pth").write_bytes(b"x")
    params = _base_params(tmp_path, lora_paths={"mylora": str(lora)},
                           half_precision=True, compile_decoder=True)
    cfg = PredictController.sam_config(params)
    assert cfg["half_precision"] is True
    assert cfg["compile_decoder"] is True


# ── sam2_config ───────────────────────────────────────────────────────────────

def _fake_sam2_available(monkeypatch, value=True):
    import napari_app.engines_sam2 as es2
    monkeypatch.setattr(es2, "sam2_available", lambda: value)


def _with_sam2_checkpoint(tmp_path, model_type="large"):
    names = {"large": "sam2.1_hiera_large.pt", "small": "sam2.1_hiera_small.pt"}
    d = tmp_path / "sam2_checkpoints"; d.mkdir(exist_ok=True)
    ckpt = d / names[model_type]; ckpt.write_bytes(b"x")
    return ckpt


def test_sam2_config_unavailable_raises(tmp_path, monkeypatch):
    _fake_sam2_available(monkeypatch, False)
    params = _base_params(tmp_path, engine="sam2")
    with pytest.raises(ValueError, match="SAM2 is not installed"):
        PredictController.build_config(params)


def test_sam2_config_missing_checkpoint_raises(tmp_path, monkeypatch):
    _fake_sam2_available(monkeypatch, True)
    params = _base_params(tmp_path, engine="sam2")
    with pytest.raises(ValueError, match="checkpoint not found"):
        PredictController.build_config(params)


def test_sam2_config_resolves_paths_and_shape(tmp_path, monkeypatch):
    _fake_sam2_available(monkeypatch, True)
    ckpt = _with_sam2_checkpoint(tmp_path)
    params = _base_params(tmp_path, engine="sam2", resize_size=768,
                          sam2_model_type="large", sam2_checkpoint_text="",
                          sam2_config_text="")
    cfg = PredictController.build_config(params)
    assert cfg["engine"] == "sam2"
    assert cfg["sam2_checkpoint"] == str(ckpt)
    assert cfg["sam2_config_name"] == "configs/sam2.1/sam2.1_hiera_l.yaml"
    assert cfg["resize_size"] == [768, 768]
    assert cfg["points_per_side"] == 32
    assert cfg["zstack"] is False and cfg["stitch_iou"] == 0.25


def test_sam2_config_threads_zstack_through(tmp_path, monkeypatch):
    _fake_sam2_available(monkeypatch, True)
    _with_sam2_checkpoint(tmp_path)
    params = _base_params(tmp_path, engine="sam2", zstack=True, stitch_iou=0.5)
    cfg = PredictController.sam2_config(params)
    assert cfg["zstack"] is True
    assert cfg["stitch_iou"] == 0.5


def test_resolve_lora_prefers_custom_text_over_combo():
    got = PredictController.resolve_lora(" /custom/path.pth ", "combo",
                                         {"combo": "/other.pth"})
    assert got == "/custom/path.pth"


@pytest.mark.parametrize("custom_text", ["", "   "])
def test_resolve_lora_falls_back_to_combo_when_custom_blank(custom_text):
    got = PredictController.resolve_lora(custom_text, "combo", {"combo": "/other.pth"})
    assert got == "/other.pth"


def test_resolve_sam_prefers_explicit_existing_path(tmp_path):
    p = tmp_path / "custom_sam.pth"; p.write_bytes(b"x")
    assert PredictController.resolve_sam(str(p), "vit_h", tmp_path) == str(p)


def test_resolve_sam_ignores_explicit_path_that_does_not_exist(tmp_path):
    backbone_dir = tmp_path / "sam_backbone"; backbone_dir.mkdir()
    real = backbone_dir / "sam_vit_b_01ec64.pth"; real.write_bytes(b"x")
    got = PredictController.resolve_sam(str(tmp_path / "ghost.pth"), "vit_b", tmp_path)
    assert got == str(real)


def test_resolve_sam_raises_when_nothing_found(tmp_path):
    with pytest.raises(ValueError, match="SAM backbone not found"):
        PredictController.resolve_sam("", "vit_h", tmp_path)


# ── run_prediction_async ──────────────────────────────────────────────────────

def test_run_prediction_async_success_sequences_callbacks(tmp_path, monkeypatch):
    import cv2
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[10:30, 10:30] = 200
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)
    config = {"engine": "cellpose", "image_path": str(path),
              "resize_size": [40, 40], "clahe": False, "tile_size": 1024,
              "selected_device": "cpu"}

    events = []
    controller = PredictController()
    t = controller.run_prediction_async(
        config,
        on_result=lambda img_arr, mask, stack: events.append(("result", int(mask.max()), stack)),
        on_log=lambda s: events.append(("log", s)),
        on_finish=lambda: events.append(("finish",)))
    t.join(timeout=10)

    assert [e[0] for e in events] == ["result", "log", "finish"]  # exact order
    assert events[0][1] == 1                 # one connected component
    assert events[0][2] is None              # ordinary path → no channel stack
    assert "1 cells" in events[1][1] and "Cellpose-SAM" in events[1][1]


def test_run_prediction_async_hints_when_large_and_not_tiled(tmp_path, monkeypatch):
    import cv2
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    img = np.zeros((160, 480, 3), dtype=np.uint8)
    img[70:90, 40:60] = 255
    path = tmp_path / "big.png"
    cv2.imwrite(str(path), img)
    config = {"engine": "cellpose", "tile_size": 128, "clahe": False,
              "tiled": False, "image_path": str(path), "resize_size": [256, 256]}

    logs = []
    controller = PredictController()
    t = controller.run_prediction_async(config, on_log=logs.append)
    t.join(timeout=10)
    assert any("[HINT]" in s and "Large image" in s for s in logs)


def test_run_prediction_async_no_hint_when_tiled(tmp_path, monkeypatch):
    import cv2
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    img = np.zeros((160, 480, 3), dtype=np.uint8)
    img[70:90, 40:60] = 255
    path = tmp_path / "big.png"
    cv2.imwrite(str(path), img)
    config = {"engine": "cellpose", "tile_size": 128, "tile_overlap": 48,
              "clahe": False, "tiled": True, "image_path": str(path),
              "resize_size": 256}

    logs = []
    controller = PredictController()
    t = controller.run_prediction_async(config, on_log=logs.append)
    t.join(timeout=10)
    assert not any("[HINT]" in s for s in logs)


def test_run_prediction_async_no_hint_when_small(tmp_path, monkeypatch):
    import cv2
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[10:30, 10:30] = 200
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)
    config = {"engine": "cellpose", "image_path": str(path),
              "resize_size": [40, 40], "clahe": False, "tile_size": 1024,
              "tiled": False, "selected_device": "cpu"}

    logs = []
    controller = PredictController()
    t = controller.run_prediction_async(config, on_log=logs.append)
    t.join(timeout=10)
    assert not any("[HINT]" in s for s in logs)


def test_run_prediction_async_sam_branch_logs_cache_status(tmp_path, monkeypatch):
    import cv2
    import napari_app.inference_cache as ic
    monkeypatch.setattr(ic, "predict_cached", lambda cfg, t: _cc(t))

    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[10:30, 10:30] = 200
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)
    config = {"engine": "cellseg1", "image_path": str(path),
              "resize_size": [40, 40], "clahe": False, "tile_size": 1024}

    logs = []
    controller = PredictController()
    t = controller.run_prediction_async(config, on_log=logs.append)
    t.join(timeout=10)
    assert any("cells" in s and "model:" in s for s in logs)  # cache_status() text


def test_run_prediction_async_error_still_calls_log_and_finish(tmp_path):
    config = {"engine": "cellpose", "image_path": str(tmp_path / "missing.png"),
              "resize_size": [40, 40], "clahe": False, "tile_size": 1024}
    events = []
    controller = PredictController()
    t = controller.run_prediction_async(
        config,
        on_result=lambda *a: events.append(("result",)),
        on_log=lambda s: events.append(("log", s)),
        on_finish=lambda: events.append(("finish",)))
    t.join(timeout=10)
    assert events[0][0] == "log" and "[ERROR]" in events[0][1]
    assert events[1] == ("finish",)          # on_finish still fires after an error


def test_run_prediction_async_forwards_on_tile(tmp_path, monkeypatch):
    import cv2
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    img = np.zeros((160, 480, 3), dtype=np.uint8)
    img[70:90, 40:60] = 255
    path = tmp_path / "big.png"
    cv2.imwrite(str(path), img)
    config = {"engine": "cellpose", "tile_size": 128, "tile_overlap": 48,
              "clahe": False, "tiled": True, "image_path": str(path),
              "resize_size": 256}

    calls = []
    controller = PredictController()
    t = controller.run_prediction_async(config, on_tile=lambda d, n: calls.append((d, n)))
    t.join(timeout=10)
    assert calls and calls[-1][0] == calls[-1][1]  # ends at total/total


# ── run_tuning_loop_async (agentic predict -> score -> adjust loop) ──────────

def _cellpose_blob_setup(tmp_path):
    """A real one-blob PNG + the matching perfect-GT label mask, so a
    prediction through the fake connected-components engine scores 1.0."""
    import cv2
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[10:30, 10:30] = 200
    path = tmp_path / "real.png"
    cv2.imwrite(str(path), img)
    gt = np.zeros((40, 40), dtype=np.int32)
    gt[10:30, 10:30] = 1
    return path, gt


def test_run_tuning_loop_async_sequences_steps_then_finish(tmp_path, monkeypatch):
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))
    monkeypatch.setattr(engines, "cellpose_available", lambda: True)
    path, gt = _cellpose_blob_setup(tmp_path)
    params = _base_params(tmp_path, engine="cellpose", image_path=str(path), resize_size=40)

    events = []
    controller = PredictController()
    t = controller.run_tuning_loop_async(
        params, gt, patience=1,
        on_step=lambda step: events.append(("step", step)),
        on_finish=lambda: events.append(("finish",)))
    t.join(timeout=10)

    assert [e[0] for e in events] == ["step", "step", "finish"]  # plateaus after 1 flat repeat
    steps = [e[1] for e in events if e[0] == "step"]
    assert steps[0].step == 0 and steps[1].step == 1
    assert steps[0].score == pytest.approx(1.0)   # perfect match -> mAP 1.0
    assert steps[0].n_cells == 1
    assert steps[1].score == pytest.approx(1.0)   # cellpose config ignores advisor's
                                                   # suggested keys -> identical re-run


def test_run_tuning_loop_async_error_still_calls_log_and_finish(tmp_path):
    params = _base_params(tmp_path, engine="cellpose",
                          image_path=str(tmp_path / "missing.png"))
    gt = np.zeros((10, 10), dtype=np.int32)
    events = []
    controller = PredictController()
    t = controller.run_tuning_loop_async(
        params, gt,
        on_step=lambda step: events.append(("step", step)),
        on_log=lambda s: events.append(("log", s)),
        on_finish=lambda: events.append(("finish",)))
    t.join(timeout=10)
    assert events[0][0] == "log" and "[ERROR]" in events[0][1]
    assert events[1] == ("finish",)
    assert not any(e[0] == "step" for e in events)   # failed before a single round completed


def test_run_tuning_loop_async_stop_tuning_halts_the_loop(tmp_path, monkeypatch):
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))
    monkeypatch.setattr(engines, "cellpose_available", lambda: True)
    path, gt = _cellpose_blob_setup(tmp_path)
    params = _base_params(tmp_path, engine="cellpose", image_path=str(path), resize_size=40)

    controller = PredictController()
    steps = []

    def on_step(step):
        steps.append(step)
        controller.stop_tuning()   # request cancellation right after round 1

    events = []
    # patience=10 so a plateau alone would not stop the loop this early —
    # isolates stop_tuning() as the actual reason it halts at one round.
    t = controller.run_tuning_loop_async(
        params, gt, patience=10, max_steps=10,
        on_step=on_step, on_finish=lambda: events.append("finish"))
    t.join(timeout=10)

    assert len(steps) == 1
    assert events == ["finish"]


# ── z-stack / time-lapse orchestration (_predict_volume / run_volume_prediction_async) ──

def _write_zstack(tmp_path):
    """A 3-plane grayscale z-stack: one square tracked across all 3 planes
    (shifting slightly each time, so IoU-linking keeps its id) plus a second,
    unrelated square that only appears in the last plane."""
    import tifffile
    arr = np.zeros((3, 40, 40), dtype=np.uint8)
    arr[0, 10:30, 10:30] = 200
    arr[1, 10:30, 12:32] = 200
    arr[2, 10:30, 14:34] = 200
    arr[2, 2:6, 2:6] = 200
    path = tmp_path / "zstack.tif"
    tifffile.imwrite(path, arr, photometric="minisblack", metadata={"axes": "ZYX"})
    return path


def test_predict_volume_stitches_across_planes(tmp_path, monkeypatch):
    import napari_app.engines as engines
    from napari_app.core.predict_controller import _predict_volume
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    path = _write_zstack(tmp_path)
    config = {"engine": "cellpose", "image_path": str(path), "resize_size": [40, 40],
              "clahe": False, "channels": None, "stitch_iou": 0.25, "min_mask_area": 0}

    img_vol, mask_vol = _predict_volume(config)
    assert img_vol.shape == (3, 40, 40, 3)
    assert mask_vol.shape == (3, 40, 40)
    assert int(mask_vol.max()) == 2                     # tracked square + one new appearance
    tracked = mask_vol[0, 15, 20], mask_vol[1, 15, 20], mask_vol[2, 15, 20]
    assert tracked[0] == tracked[1] == tracked[2] != 0   # same id, all 3 planes
    assert mask_vol[2, 3, 3] not in (0, tracked[0])      # new instance, distinct id
    assert mask_vol[0, 3, 3] == 0 and mask_vol[1, 3, 3] == 0  # absent from earlier planes


def test_predict_volume_reports_on_slice_progress(tmp_path, monkeypatch):
    import napari_app.engines as engines
    from napari_app.core.predict_controller import _predict_volume
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    path = _write_zstack(tmp_path)
    config = {"engine": "cellpose", "image_path": str(path), "resize_size": [40, 40],
              "clahe": False, "channels": None}
    calls = []
    _predict_volume(config, on_slice=lambda d, n: calls.append((d, n)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_predict_volume_composes_with_tiling_for_large_planes(tmp_path, monkeypatch):
    """A plane large enough that should_tile recommends it (and tiled=True)
    must go through _predict_tiled instead of a plain resize+predict — the
    z-stack path composes with the existing tiling feature instead of
    ignoring config["tiled"]."""
    import napari_app.engines as engines
    from napari_app.core import predict_controller as pc
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    calls = []
    real_tiled = pc._predict_tiled

    def spy_tiled(config, img, on_tile=None):
        calls.append(img.shape)
        return real_tiled(config, img, on_tile=on_tile)

    monkeypatch.setattr(pc, "_predict_tiled", spy_tiled)

    path = _write_zstack(tmp_path)   # 3 planes, 40x40 each
    config = {"engine": "cellpose", "image_path": str(path), "resize_size": [40, 40],
              "clahe": False, "channels": None, "tiled": True, "tile_size": 20,
              "tile_overlap": 8, "stitch_iou": 0.25, "min_mask_area": 0}

    img_vol, mask_vol = pc._predict_volume(config)
    assert len(calls) == 3                     # every plane went through the tiled path
    assert all(shape == (40, 40, 3) for shape in calls)
    assert img_vol.shape == (3, 40, 40, 3)
    assert mask_vol.shape == (3, 40, 40)
    assert int(mask_vol.max()) >= 1


def test_predict_volume_untiled_when_tiled_flag_off_even_if_large(tmp_path, monkeypatch):
    import napari_app.engines as engines
    from napari_app.core import predict_controller as pc
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    calls = []
    monkeypatch.setattr(pc, "_predict_tiled", lambda *a, **k: calls.append(1))

    path = _write_zstack(tmp_path)
    config = {"engine": "cellpose", "image_path": str(path), "resize_size": [40, 40],
              "clahe": False, "channels": None, "tiled": False, "tile_size": 20}
    pc._predict_volume(config)
    assert calls == []                         # tiled=False -> never routed to _predict_tiled


def test_predict_volume_dispatches_to_sam2_propagate_mode(tmp_path, monkeypatch):
    """engine=sam2 + sam2_tracking_mode=propagate must call
    predict_sam2_propagate directly and skip the generic per-plane-predict +
    stitch_slices path entirely (no per-plane spec.predict, no stitching)."""
    import napari_app.engines_sam2 as es2
    from napari_app.core import predict_controller as pc

    calls = []

    def fake_propagate(frames, config, on_slice=None):
        calls.append((len(frames), config.get("sam2_tracking_mode")))
        if on_slice:
            for i in range(len(frames)):
                on_slice(i + 1, len(frames))
        return np.ones((len(frames), 40, 40), dtype=np.int32)

    monkeypatch.setattr(es2, "predict_sam2_propagate", fake_propagate)

    path = _write_zstack(tmp_path)
    config = {"engine": "sam2", "image_path": str(path), "resize_size": [40, 40],
              "clahe": False, "channels": None, "sam2_tracking_mode": "propagate"}

    slice_calls = []
    img_vol, mask_vol = pc._predict_volume(config, on_slice=lambda d, n: slice_calls.append((d, n)))

    assert calls == [(3, "propagate")]
    assert img_vol.shape == (3, 40, 40, 3)
    assert mask_vol.shape == (3, 40, 40)
    assert slice_calls == [(1, 3), (2, 3), (3, 3)]


def test_predict_volume_sam2_automatic_mode_never_calls_propagate(tmp_path, monkeypatch):
    import napari_app.engines_sam2 as es2
    from napari_app.core import predict_controller as pc

    calls = []
    monkeypatch.setattr(es2, "predict_sam2_propagate", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(es2, "predict_sam2", lambda image, config: _cc(image))

    path = _write_zstack(tmp_path)
    config = {"engine": "sam2", "image_path": str(path), "resize_size": [40, 40],
              "clahe": False, "channels": None, "sam2_tracking_mode": "automatic",
              "stitch_iou": 0.25, "min_mask_area": 0}
    pc._predict_volume(config)
    assert calls == []                          # "automatic" -> generic per-plane path, not propagate


def test_run_volume_prediction_async_success_sequences_callbacks(tmp_path, monkeypatch):
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    path = _write_zstack(tmp_path)
    config = {"engine": "cellpose", "image_path": str(path), "resize_size": [40, 40],
              "clahe": False, "channels": None}

    events = []
    controller = PredictController()
    t = controller.run_volume_prediction_async(
        config,
        on_result=lambda img_vol, mask_vol, stack: events.append(
            ("result", mask_vol.shape, int(mask_vol.max()), stack)),
        on_log=lambda s: events.append(("log", s)),
        on_finish=lambda: events.append(("finish",)))
    t.join(timeout=10)

    assert [e[0] for e in events] == ["result", "log", "finish"]
    assert events[0][1] == (3, 40, 40)
    assert events[0][2] == 2
    assert events[0][3] is not None              # the VolumeStack read along the way
    assert "2 cells across 3 planes" in events[1][1]
    assert "Cellpose-SAM" in events[1][1]


def test_run_volume_prediction_async_forwards_on_slice(tmp_path, monkeypatch):
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    path = _write_zstack(tmp_path)
    config = {"engine": "cellpose", "image_path": str(path), "resize_size": [40, 40],
              "clahe": False, "channels": None}

    calls = []
    controller = PredictController()
    t = controller.run_volume_prediction_async(config, on_slice=lambda d, n: calls.append((d, n)))
    t.join(timeout=10)
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_run_volume_prediction_async_error_still_calls_log_and_finish(tmp_path):
    config = {"engine": "cellpose", "image_path": str(tmp_path / "missing.tif"),
              "resize_size": [40, 40], "clahe": False, "channels": None}
    events = []
    controller = PredictController()
    t = controller.run_volume_prediction_async(
        config,
        on_result=lambda *a: events.append(("result",)),
        on_log=lambda s: events.append(("log", s)),
        on_finish=lambda: events.append(("finish",)))
    t.join(timeout=10)
    assert events[0][0] == "log" and "[ERROR]" in events[0][1]
    assert events[1] == ("finish",)


# ── run_batch_async ───────────────────────────────────────────────────────────

def _write_images(tmp_path, n, size=30):
    import cv2
    paths = []
    for i in range(n):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[5:15, 5:15] = 200
        p = tmp_path / f"img{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def test_run_batch_async_processes_all_and_writes_cohort(tmp_path, monkeypatch):
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    images = _write_images(tmp_path, 3)
    out_dir = tmp_path / "out"; out_dir.mkdir()
    config = {"engine": "cellpose", "resize_size": [30, 30], "clahe": False,
              "tile_size": 1024, "selected_device": "cpu"}

    logs, progress, cohort_ready = [], [], []
    finished = threading.Event()
    controller = PredictController()
    t = controller.run_batch_async(
        config, images, out_dir, pixel_size_um=0.0,
        on_log=logs.append, on_progress=lambda d, n: progress.append((d, n)),
        on_cohort_ready=lambda records, out: cohort_ready.append((records, out)),
        on_finish=finished.set)
    t.join(timeout=10)

    assert finished.is_set()
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert len(cohort_ready) == 1
    records, out = cohort_ready[0]
    assert len(records) == 3 and out == out_dir
    assert (out_dir / "img0_mask.png").exists()
    assert (out_dir / "cohort_measurements.csv").exists()
    assert (out_dir / "cohort_summary.csv").exists()
    assert any("Batch done" in s for s in logs)


def test_run_batch_async_stop_skips_cohort_step(tmp_path, monkeypatch):
    import napari_app.engines as engines
    controller = PredictController()

    call_count = {"n": 0}

    def fake_predict(t, **k):
        call_count["n"] += 1
        if call_count["n"] == 1:
            controller.stop_batch()   # simulate the user clicking Stop mid-image
        return _cc(t)

    monkeypatch.setattr(engines, "predict_cellpose", fake_predict)
    images = _write_images(tmp_path, 3)
    out_dir = tmp_path / "out"; out_dir.mkdir()
    config = {"engine": "cellpose", "resize_size": [30, 30], "clahe": False,
              "tile_size": 1024, "selected_device": "cpu"}

    logs, cohort_ready = [], []
    finished = threading.Event()
    t = controller.run_batch_async(
        config, images, out_dir, 0.0, on_log=logs.append,
        on_cohort_ready=lambda *a: cohort_ready.append(a), on_finish=finished.set)
    t.join(timeout=10)

    assert finished.is_set()
    assert call_count["n"] == 1              # loop broke before the 2nd image
    assert any("Stopped at 1/3" in s for s in logs)
    assert cohort_ready == []                # for/else: stop skips the cohort step


def test_run_batch_async_continues_after_a_per_image_error(tmp_path, monkeypatch):
    import napari_app.engines as engines

    calls = {"n": 0}

    def flaky(t, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return _cc(t)

    monkeypatch.setattr(engines, "predict_cellpose", flaky)
    images = _write_images(tmp_path, 2)
    out_dir = tmp_path / "out"; out_dir.mkdir()
    config = {"engine": "cellpose", "resize_size": [30, 30], "clahe": False,
              "tile_size": 1024, "selected_device": "cpu"}

    logs = []
    cohort_ready = []
    finished = threading.Event()
    controller = PredictController()
    t = controller.run_batch_async(
        config, images, out_dir, 0.0, on_log=logs.append,
        on_cohort_ready=lambda *a: cohort_ready.append(a), on_finish=finished.set)
    t.join(timeout=10)

    assert finished.is_set()
    assert any("[ERROR] boom" in s for s in logs)
    assert len(cohort_ready) == 1
    records, _out = cohort_ready[0]
    assert len(records) == 1                 # only the 2nd image succeeded


# ── run_benchmark_async ───────────────────────────────────────────────────────

def test_run_benchmark_async_aggregates_and_calls_on_done(tmp_path, monkeypatch):
    import cv2
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    img_dir = tmp_path / "images"; img_dir.mkdir()
    gt_dir = tmp_path / "gt"; gt_dir.mkdir()
    for i in range(2):
        img = np.zeros((30, 30, 3), dtype=np.uint8)
        img[5:15, 5:15] = 200
        cv2.imwrite(str(img_dir / f"img{i}.png"), img)
        gt = np.zeros((30, 30), dtype=np.uint16)
        gt[5:15, 5:15] = 1
        cv2.imwrite(str(gt_dir / f"img{i}.png"), gt)

    pairs = [(img_dir / "img0.png", gt_dir / "img0.png"),
             (img_dir / "img1.png", gt_dir / "img1.png")]
    bases = {"cellpose": {"engine": "cellpose", "resize_size": [30, 30],
                          "image_path": "", "clahe": False}}

    rows_seen = []
    done = {}
    controller = PredictController()
    t = controller.run_benchmark_async(
        ["cellpose"], bases, pairs, img_dir,
        on_row=rows_seen.append,
        on_done=lambda cols, rows: done.update(cols=cols, rows=rows))
    t.join(timeout=10)

    assert len(rows_seen) == 2                # one row per (engine × image) pair
    assert all(ENGINE_LABELS["cellpose"] in r for r in rows_seen)
    assert done["cols"][0] == "engine"
    assert done["rows"][0][0] == ENGINE_LABELS["cellpose"]
    assert done["rows"][0][1] == 2            # n_images
    assert (img_dir / "benchmark.csv").exists()


def test_run_benchmark_async_logs_per_pair_errors(tmp_path, monkeypatch):
    import cv2
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose",
                        lambda t, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    img_dir = tmp_path / "images"; img_dir.mkdir()
    gt_dir = tmp_path / "gt"; gt_dir.mkdir()
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "img0.png"), img)
    gt = np.zeros((20, 20), dtype=np.uint16)
    cv2.imwrite(str(gt_dir / "img0.png"), gt)

    pairs = [(img_dir / "img0.png", gt_dir / "img0.png")]
    bases = {"cellpose": {"engine": "cellpose", "resize_size": [20, 20],
                          "image_path": "", "clahe": False}}

    logs = []
    controller = PredictController()
    t = controller.run_benchmark_async(["cellpose"], bases, pairs, img_dir,
                                       on_log=logs.append)
    t.join(timeout=10)
    assert any("[ERROR] cellpose img0.png: boom" in s for s in logs)
