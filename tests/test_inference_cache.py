"""Pure-logic tests for velum_core.inference_cache's fp16/torch.compile gating.

inference_cache.py keeps torch imports lazy inside its functions specifically
so it (and its config-only helpers) can be imported and tested without torch
installed — see the module docstring. predict_cached/_load_model themselves
touch a real SAM model and aren't covered here (no GPU/checkpoint in this
suite); that mirrors every other test in this repo, none of which unit-test
those two functions directly — only the pure decision logic around them.
"""
import velum_core.inference_cache as ic


# ── _is_cuda_device ───────────────────────────────────────────────────────────

def test_is_cuda_device_true_for_cuda_index():
    assert ic._is_cuda_device("0") is True
    assert ic._is_cuda_device("1") is True


def test_is_cuda_device_false_for_cpu_mps_or_empty():
    assert ic._is_cuda_device("cpu") is False
    assert ic._is_cuda_device("mps") is False
    assert ic._is_cuda_device("") is False
    assert ic._is_cuda_device(None) is False


# ── use_amp / use_compile ──────────────────────────────────────────────────────

def test_use_amp_requires_both_flag_and_cuda_device():
    assert ic.use_amp({"half_precision": True, "selected_device": "0"}) is True
    assert ic.use_amp({"half_precision": True, "selected_device": "cpu"}) is False
    assert ic.use_amp({"half_precision": True, "selected_device": "mps"}) is False
    assert ic.use_amp({"half_precision": False, "selected_device": "0"}) is False
    assert ic.use_amp({"selected_device": "0"}) is False  # default off


def test_use_compile_requires_both_flag_and_cuda_device():
    assert ic.use_compile({"compile_decoder": True, "selected_device": "0"}) is True
    assert ic.use_compile({"compile_decoder": True, "selected_device": "cpu"}) is False
    assert ic.use_compile({"compile_decoder": True, "selected_device": "mps"}) is False
    assert ic.use_compile({"compile_decoder": False, "selected_device": "0"}) is False
    assert ic.use_compile({"selected_device": "0"}) is False  # default off


# ── _mk_model_key ──────────────────────────────────────────────────────────────

def _cfg(**overrides):
    base = {
        "result_pth_path": "/x/lora.pth", "vit_name": "vit_h",
        "image_encoder_lora_rank": 4, "sam_image_size": 512,
        "selected_device": "cpu", "compile_decoder": False,
    }
    base.update(overrides)
    return base


def test_model_key_unaffected_by_compile_toggle_off_cuda():
    # use_compile is already False on cpu/mps regardless of the raw flag, so
    # toggling it shouldn't force a spurious model reload.
    a = ic._mk_model_key(_cfg(selected_device="cpu", compile_decoder=False))
    b = ic._mk_model_key(_cfg(selected_device="cpu", compile_decoder=True))
    assert a == b


def test_model_key_changes_when_compile_toggled_on_cuda():
    a = ic._mk_model_key(_cfg(selected_device="0", compile_decoder=False))
    b = ic._mk_model_key(_cfg(selected_device="0", compile_decoder=True))
    assert a != b


def test_model_key_unaffected_by_half_precision():
    # half_precision only gates the per-call autocast context, not the cached
    # model object, so it deliberately isn't part of the key.
    a = ic._mk_model_key(_cfg(selected_device="0"))
    b = ic._mk_model_key({**_cfg(selected_device="0"), "half_precision": True})
    assert a == b


# ── cache_status ───────────────────────────────────────────────────────────────

def test_cache_status_shows_compiled_marker_only_when_set():
    ic.invalidate_model()
    assert "compiled" not in ic.cache_status()
    ic._compiled = True
    try:
        assert "compiled" in ic.cache_status()
    finally:
        ic._compiled = False


def test_invalidate_model_resets_compiled_flag():
    ic._compiled = True
    ic.invalidate_model()
    assert ic._compiled is False
