"""Tests for device_utils (shared CUDA-capability check).

Pure-logic, no real torch/GPU needed -- every function takes an
already-imported ``torch``-like object, so tests pass a lightweight fake
(same technique tests/test_engines_sam2.py uses for a fake ``sam2`` module).
"""
import types

import device_utils


def _fake_torch(cuda_available=True, device_count=1,
                 capability_by_index=None, arch_list=()):
    capability_by_index = capability_by_index or {}

    def get_capability(idx=0):
        return capability_by_index.get(idx, (0, 0))

    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: device_count,
        get_device_capability=get_capability,
        get_arch_list=lambda: list(arch_list),
    )
    return mod


# ── is_usable ────────────────────────────────────────────────────────────────

def test_is_usable_true_when_capability_meets_minimum_supported():
    torch_mod = _fake_torch(capability_by_index={0: (7, 5)},
                            arch_list=["sm_75", "sm_80"])
    assert device_utils.is_usable(torch_mod, 0) is True


def test_is_usable_false_for_pascal_gpu_against_modern_cuda13_wheel():
    """Regression: a real GTX 1070 (Pascal, capability 6.1) against a torch
    build whose CUDA wheel only ships kernels for capability >= 7.5 --
    torch.cuda.is_available() would still report True, but any real op
    raises "CUDA error: no kernel image is available for execution"."""
    torch_mod = _fake_torch(
        capability_by_index={0: (6, 1)},
        arch_list=["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"])
    assert device_utils.is_usable(torch_mod, 0) is False


def test_is_usable_true_when_no_arch_list_reported():
    torch_mod = _fake_torch(capability_by_index={0: (6, 1)}, arch_list=[])
    assert device_utils.is_usable(torch_mod, 0) is True


def test_is_usable_true_when_capability_check_raises():
    torch_mod = _fake_torch()
    torch_mod.cuda.get_device_capability = lambda idx=0: (_ for _ in ()).throw(RuntimeError("boom"))
    assert device_utils.is_usable(torch_mod, 0) is True


# ── usable_cuda_indices ────────────────────────────────────────────────────────

def test_usable_cuda_indices_empty_when_cuda_unavailable():
    torch_mod = _fake_torch(cuda_available=False)
    assert device_utils.usable_cuda_indices(torch_mod) == []


def test_usable_cuda_indices_filters_out_unsupported_devices():
    # A mixed multi-GPU box: index 0 is an old Pascal card (unusable against
    # this arch list), index 1 is a modern Ampere card (usable).
    torch_mod = _fake_torch(
        device_count=2,
        capability_by_index={0: (6, 1), 1: (8, 6)},
        arch_list=["sm_75", "sm_80", "sm_86", "sm_90"])
    assert device_utils.usable_cuda_indices(torch_mod) == [1]


def test_usable_cuda_indices_all_present_when_all_supported():
    torch_mod = _fake_torch(
        device_count=2,
        capability_by_index={0: (8, 0), 1: (8, 6)},
        arch_list=["sm_75", "sm_80", "sm_86"])
    assert device_utils.usable_cuda_indices(torch_mod) == [0, 1]
