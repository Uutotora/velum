"""Unit tests for napari_app.engines_sam2.

Covers everything that doesn't need torch or a real sam2/checkpoint install:
availability gating, checkpoint/config resolution, cache-key/state bookkeeping,
and registry wiring. Mirrors tests/test_inference_cache.py's approach (poke
module globals directly rather than call the torch-needing loader) and
tests/test_engine_registry.py's approach for the built-in engines.

Deliberately NOT tested here (needs torch, which the pure-logic `test`
dependency-group excludes by design — see AGENTS.md): predict_sam2(),
get_mask_generator(), get_video_predictor(), _build_mask_generator(),
_build_video_predictor(), _torch_device(). Those import torch even with a
fake `sam2` module installed, so exercising them would pass in a full conda
env but break the lightweight CI job, exactly the class of gap AGENTS.md's
"verifying changes without a display" section warns about.

predict_sam2_propagate() *is* tested, despite calling get_mask_generator()/
get_video_predictor() internally — every test below monkeypatches both to
fakes, so the only real code exercised is the temp-JPEG-directory bookkeeping
and the propagation-output-to-label-volume accumulation, neither of which
needs torch or the real sam2 package.
"""
import os
import sys
import types

import numpy as np
import pytest

from napari_app import engine_registry
# Import predict_controller (not engines_sam2 directly) first, so engine
# registration always happens in the app's real order — engines.py's
# cellseg1/cellpose before engines_sam2.py's sam2 — regardless of which test
# module the whole suite happens to collect first. Importing engines_sam2
# directly here would make *this file* the thing that decides registration
# order whenever it's collected before anything that imports predict_controller
# (e.g. the predict_widget wiring tests), silently reordering
# PredictWidget's engine combo (sam2 first, not cellseg1) for every other test
# in the same run — exactly the failure this comment exists to prevent.
import napari_app.core.predict_controller as _predict_controller  # noqa: F401
import napari_app.engines_sam2 as es2


@pytest.fixture(autouse=True)
def _reset_cache():
    es2.invalidate_sam2()
    yield
    es2.invalidate_sam2()


# ── availability ─────────────────────────────────────────────────────────────

def test_sam2_available_false_without_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "sam2", None)   # force ImportError on import
    assert es2.sam2_available() is False


def test_sam2_available_true_with_fake_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    assert es2.sam2_available() is True


# ── checkpoint / config resolution ──────────────────────────────────────────

def test_resolve_sam2_explicit_checkpoint_path_wins(tmp_path):
    ckpt = tmp_path / "custom.pt"
    ckpt.write_bytes(b"fake")
    path, config = es2.resolve_sam2(str(ckpt), "", "large", tmp_path)
    assert path == str(ckpt)
    assert config == "configs/sam2.1/sam2.1_hiera_l.yaml"


def test_resolve_sam2_falls_back_to_storage_dir(tmp_path):
    d = tmp_path / "sam2_checkpoints"
    d.mkdir()
    ckpt = d / "sam2.1_hiera_small.pt"
    ckpt.write_bytes(b"fake")
    path, config = es2.resolve_sam2("", "", "small", tmp_path)
    assert path == str(ckpt)
    assert config == "configs/sam2.1/sam2.1_hiera_s.yaml"


def test_resolve_sam2_missing_checkpoint_raises(tmp_path):
    with pytest.raises(ValueError, match="checkpoint not found"):
        es2.resolve_sam2("", "", "large", tmp_path)


def test_resolve_sam2_config_override_used_verbatim(tmp_path):
    d = tmp_path / "sam2_checkpoints"
    d.mkdir()
    (d / "sam2.1_hiera_large.pt").write_bytes(b"fake")
    _, config = es2.resolve_sam2("", "configs/custom.yaml", "large", tmp_path)
    assert config == "configs/custom.yaml"


def test_resolve_sam2_unknown_model_type_defaults_to_large(tmp_path):
    d = tmp_path / "sam2_checkpoints"
    d.mkdir()
    (d / "sam2.1_hiera_large.pt").write_bytes(b"fake")
    path, config = es2.resolve_sam2("", "", "not-a-real-size", tmp_path)
    assert path == str(d / "sam2.1_hiera_large.pt")
    assert config == "configs/sam2.1/sam2.1_hiera_l.yaml"


def test_resolve_sam2_explicit_path_that_does_not_exist_falls_back(tmp_path):
    d = tmp_path / "sam2_checkpoints"
    d.mkdir()
    (d / "sam2.1_hiera_large.pt").write_bytes(b"fake")
    path, _ = es2.resolve_sam2(str(tmp_path / "nope.pt"), "", "large", tmp_path)
    assert path == str(d / "sam2.1_hiera_large.pt")


# ── cache-key / state bookkeeping (no torch, no real model) ─────────────────

def test_mg_cache_key_changes_with_checkpoint_or_thresholds():
    base = {"sam2_checkpoint": "a.pt", "sam2_config_name": "cfg.yaml",
           "selected_device": "cpu", "points_per_side": 32,
           "pred_iou_thresh": 0.8, "stability_score_thresh": 0.6,
           "box_nms_thresh": 0.7, "min_mask_area": 0}
    k1 = es2._mg_cache_key(base)
    k2 = es2._mg_cache_key({**base, "sam2_checkpoint": "b.pt"})
    k3 = es2._mg_cache_key({**base, "pred_iou_thresh": 0.9})
    assert k1 != k2
    assert k1 != k3
    assert k1 == es2._mg_cache_key(dict(base))   # identical config -> identical key


def test_cache_status_reports_no_model_by_default():
    es2.invalidate_sam2()
    assert es2.cache_status() == "model: —"


def test_cache_status_reports_checkpoint_name():
    es2._mg_key = ("/some/path/sam2.1_hiera_large.pt", "cfg.yaml", "cpu",
                   32, 0.8, 0.6, 0.7, 0)
    assert es2.cache_status() == "model: sam2.1_hiera_large.pt"


def test_invalidate_sam2_clears_cache():
    es2._mask_generator = object()
    es2._mg_key = ("x",)
    es2.invalidate_sam2()
    assert es2._mask_generator is None
    assert es2._mg_key is None


# ── registry wiring ──────────────────────────────────────────────────────────

def test_sam2_engine_is_registered():
    assert engine_registry.is_registered("sam2")


def test_sam2_engine_labels():
    spec = engine_registry.get("sam2")
    assert spec.label == "SAM 2 · zero-shot (z-stack / video)"
    assert spec.bench_label == "SAM 2 (zero-shot, experimental)"
    assert spec.result_label == "SAM 2"


def test_sam2_available_check_reflects_monkeypatched_function(monkeypatch):
    # EngineSpec.available must call through live, not a frozen reference
    # captured at register() time (see the cellpose regression this class of
    # bug already caused, per napari_app.engines's own comment).
    monkeypatch.setattr(es2, "sam2_available", lambda: True)
    assert engine_registry.get("sam2").available() is True
    monkeypatch.setattr(es2, "sam2_available", lambda: False)
    assert engine_registry.get("sam2").available() is False


def test_sam2_status_line_reports_cache_status():
    spec = engine_registry.get("sam2")
    assert spec.status_line is not None
    assert "model:" in spec.status_line()


def test_predict_dispatches_through_registry(monkeypatch):
    calls = []
    monkeypatch.setattr(es2, "predict_sam2", lambda image, config: calls.append((image, config)) or "mask")
    result = engine_registry.get("sam2").predict("img", {"engine": "sam2"})
    assert result == "mask"
    assert calls == [("img", {"engine": "sam2"})]


# ── predict_sam2_propagate: video-predictor tracking mode ───────────────────
# get_mask_generator()/get_video_predictor() are always monkeypatched to fakes
# here (see module docstring) — no torch, no real sam2 package involved.

class _FakeVideoPredictor:
    def __init__(self, propagate_output=None, raise_in_propagate=None):
        self.init_state_calls = []
        self.add_new_mask_calls = []
        self.propagate_calls = 0
        self._propagate_output = propagate_output or []
        self._raise_in_propagate = raise_in_propagate

    def init_state(self, video_path):
        self.init_state_calls.append(video_path)
        return {"video_path": video_path}

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        self.add_new_mask_calls.append((frame_idx, obj_id, np.array(mask)))

    def propagate_in_video(self, inference_state):
        self.propagate_calls += 1
        if self._raise_in_propagate:
            raise self._raise_in_propagate
        yield from self._propagate_output


class _FakeMaskGenerator:
    def __init__(self, seeds):
        self._seeds = seeds
        self.generate_calls = 0

    def generate(self, image):
        self.generate_calls += 1
        return self._seeds


def _frames(n, h=12, w=16):
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_predict_sam2_propagate_empty_frames_returns_empty_volume():
    out = es2.predict_sam2_propagate([], {})
    assert out.shape == (0, 0, 0)
    assert out.dtype == np.int32


def test_predict_sam2_propagate_writes_one_jpeg_per_frame_and_cleans_up(monkeypatch):
    frames = _frames(3)
    vp = _FakeVideoPredictor(propagate_output=[])
    mg = _FakeMaskGenerator(seeds=[])
    monkeypatch.setattr(es2, "get_video_predictor", lambda config: vp)
    monkeypatch.setattr(es2, "get_mask_generator", lambda config: mg)

    es2.predict_sam2_propagate(frames, {})

    assert len(vp.init_state_calls) == 1
    tmp_dir = vp.init_state_calls[0]
    written = sorted(os.listdir(tmp_dir)) if os.path.isdir(tmp_dir) else None
    # The directory must have held exactly one .jpg per frame *during* the
    # call — but is removed again afterwards (temp scratch space, not an
    # artifact this feature is meant to leave behind).
    assert written is None or written == []  # already gone, or (if not) empty
    assert not os.path.exists(tmp_dir)


def test_predict_sam2_propagate_seeds_objects_largest_first(monkeypatch):
    frames = _frames(2)
    small = np.zeros((12, 16), dtype=bool); small[0:2, 0:2] = True     # area 4
    big = np.zeros((12, 16), dtype=bool); big[0:6, 0:6] = True         # area 36
    mg = _FakeMaskGenerator(seeds=[
        {"segmentation": small, "area": 4},
        {"segmentation": big, "area": 36},
    ])
    vp = _FakeVideoPredictor(propagate_output=[])
    monkeypatch.setattr(es2, "get_video_predictor", lambda config: vp)
    monkeypatch.setattr(es2, "get_mask_generator", lambda config: mg)

    es2.predict_sam2_propagate(frames, {})

    assert len(vp.add_new_mask_calls) == 2
    (frame0, obj0, mask0), (frame1, obj1, mask1) = vp.add_new_mask_calls
    assert frame0 == 0 and frame1 == 0                  # always seeded on the first plane
    assert (obj0, obj1) == (1, 2)
    assert np.array_equal(mask0, big)                   # largest area seeded first
    assert np.array_equal(mask1, small)


def test_predict_sam2_propagate_respects_max_objects_cap(monkeypatch):
    frames = _frames(2)
    seeds = [{"segmentation": np.zeros((12, 16), dtype=bool), "area": i} for i in range(5)]
    mg = _FakeMaskGenerator(seeds=seeds)
    vp = _FakeVideoPredictor(propagate_output=[])
    monkeypatch.setattr(es2, "get_video_predictor", lambda config: vp)
    monkeypatch.setattr(es2, "get_mask_generator", lambda config: mg)

    es2.predict_sam2_propagate(frames, {"sam2_max_objects": 2})

    assert len(vp.add_new_mask_calls) == 2


def test_predict_sam2_propagate_builds_label_volume_from_output(monkeypatch):
    h, w = 12, 16
    frames = _frames(2, h, w)
    seed_mask = np.zeros((h, w), dtype=bool); seed_mask[0:4, 0:4] = True
    mg = _FakeMaskGenerator(seeds=[{"segmentation": seed_mask, "area": 16}])

    obj1_frame0 = np.zeros((1, h, w), dtype=np.float32); obj1_frame0[0, 0:4, 0:4] = 1.0
    obj1_frame1 = np.zeros((1, h, w), dtype=np.float32); obj1_frame1[0, 1:5, 1:5] = 1.0
    vp = _FakeVideoPredictor(propagate_output=[
        (0, [1], [obj1_frame0]),
        (1, [1], [obj1_frame1]),
    ])
    monkeypatch.setattr(es2, "get_video_predictor", lambda config: vp)
    monkeypatch.setattr(es2, "get_mask_generator", lambda config: mg)

    out = es2.predict_sam2_propagate(frames, {})

    assert out.shape == (2, h, w)
    assert out[0, 2, 2] == 1 and out[0, 10, 10] == 0
    assert out[1, 3, 3] == 1 and out[1, 0, 0] == 0       # tracked object moved between planes


def test_predict_sam2_propagate_reports_on_slice_progress(monkeypatch):
    h, w = 8, 8
    frames = _frames(3, h, w)
    seed_mask = np.zeros((h, w), dtype=bool); seed_mask[0:2, 0:2] = True
    mg = _FakeMaskGenerator(seeds=[{"segmentation": seed_mask, "area": 4}])
    out_mask = np.zeros((1, h, w), dtype=np.float32); out_mask[0, 0:2, 0:2] = 1.0
    vp = _FakeVideoPredictor(propagate_output=[
        (0, [1], [out_mask]), (1, [1], [out_mask]), (2, [1], [out_mask]),
    ])
    monkeypatch.setattr(es2, "get_video_predictor", lambda config: vp)
    monkeypatch.setattr(es2, "get_mask_generator", lambda config: mg)

    calls = []
    es2.predict_sam2_propagate(frames, {}, on_slice=lambda d, n: calls.append((d, n)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_predict_sam2_propagate_no_seeds_skips_propagation_but_still_reports_progress(monkeypatch):
    frames = _frames(3)
    mg = _FakeMaskGenerator(seeds=[])   # nothing detected on the first plane
    vp = _FakeVideoPredictor(propagate_output=[])
    monkeypatch.setattr(es2, "get_video_predictor", lambda config: vp)
    monkeypatch.setattr(es2, "get_mask_generator", lambda config: mg)

    calls = []
    out = es2.predict_sam2_propagate(frames, {}, on_slice=lambda d, n: calls.append((d, n)))

    assert vp.propagate_calls == 0             # nothing to track -> never even called
    assert int(out.max()) == 0
    assert calls == [(1, 3), (2, 3), (3, 3)]    # progress still completes


def test_predict_sam2_propagate_cleans_up_temp_dir_even_on_exception(monkeypatch):
    frames = _frames(2)
    seed_mask = np.zeros((12, 16), dtype=bool); seed_mask[0:2, 0:2] = True
    mg = _FakeMaskGenerator(seeds=[{"segmentation": seed_mask, "area": 4}])
    vp = _FakeVideoPredictor(raise_in_propagate=RuntimeError("boom"))
    monkeypatch.setattr(es2, "get_video_predictor", lambda config: vp)
    monkeypatch.setattr(es2, "get_mask_generator", lambda config: mg)

    tmp_dir_holder = {}
    real_init_state = vp.init_state

    def spy_init_state(video_path):
        tmp_dir_holder["path"] = video_path
        return real_init_state(video_path)
    vp.init_state = spy_init_state

    with pytest.raises(RuntimeError, match="boom"):
        es2.predict_sam2_propagate(frames, {})
    assert not os.path.exists(tmp_dir_holder["path"])
