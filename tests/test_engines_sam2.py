"""Unit tests for napari_app.engines_sam2.

Covers everything that doesn't need torch or a real sam2/checkpoint install:
availability gating, checkpoint/config resolution, cache-key/state bookkeeping,
and registry wiring. Mirrors tests/test_inference_cache.py's approach (poke
module globals directly rather than call the torch-needing loader) and
tests/test_engine_registry.py's approach for the built-in engines.

Deliberately NOT tested here (needs torch, which the pure-logic `test`
dependency-group excludes by design — see AGENTS.md): predict_sam2(),
get_mask_generator(), _build_mask_generator(), _torch_device(). Those import
torch even with a fake `sam2` module installed, so exercising them would pass
in a full conda env but break the lightweight CI job, exactly the class of
gap AGENTS.md's "verifying changes without a display" section warns about.
"""
import sys
import types

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
    assert spec.label == "SAM 2 · zero-shot (z-stack / video, experimental)"
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
