"""Tests for studio.hardware (the Home screen's "This device" card).

Pure-logic, no Qt — runs under the light CI `test` group even though torch
isn't in that dependency group: detect() imports torch lazily and degrades to
"CPU" if it's missing, so the "no torch installed" path is exercised for
real here. The CUDA/MPS branches are exercised with a fake `torch` module
injected into sys.modules (same technique tests/test_engines_sam2.py uses for
a fake `sam2`), so this suite needs no GPU and no real torch install.
"""
import sys
import types

import pytest

from studio import hardware


def _fake_torch(cuda_available=False, device_name="", capability=(0, 0),
                 arch_list=(), mps_available=False):
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_name=lambda idx=0: device_name,
        get_device_capability=lambda idx=0: capability,
        get_arch_list=lambda: list(arch_list),
    )
    mod.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps_available))
    return mod


@pytest.fixture(autouse=True)
def _fresh_torch_module():
    # detect()/_cuda_is_usable() do `import torch` inside the function, which
    # resolves from sys.modules first -- swapping the entry there is enough,
    # no need to monkeypatch the import machinery itself.
    had = "torch" in sys.modules
    old = sys.modules.get("torch")
    yield
    if had:
        sys.modules["torch"] = old
    else:
        sys.modules.pop("torch", None)


# ── os name ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("system,expected", [
    ("Linux", "Linux"),
    ("Darwin", "macOS"),
    ("Windows", "Windows"),
    ("FreeBSD", "FreeBSD"),
    ("", "Unknown"),
])
def test_os_name(monkeypatch, system, expected):
    monkeypatch.setattr(hardware.platform, "system", lambda: system)
    assert hardware._os_name() == expected


# ── detect() ─────────────────────────────────────────────────────────────────

def test_detect_reports_cpu_when_torch_not_installed(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError on `import torch`
    info = hardware.detect()
    assert info == hardware.DeviceInfo("cpu", "Linux · CPU", "Linux")


def test_detect_reports_cuda_when_usable(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(
        cuda_available=True, device_name="NVIDIA GeForce RTX 4090",
        capability=(8, 9), arch_list=["sm_75", "sm_80", "sm_86", "sm_89", "sm_90"]))
    info = hardware.detect()
    assert info.kind == "cuda"
    assert info.label == "NVIDIA GeForce RTX 4090 · CUDA"
    assert info.os_name == "Linux"


def test_detect_falls_back_to_cpu_when_cuda_present_but_unsupported(monkeypatch):
    """Regression: a real GTX 1070 (Pascal, capability 6.1) against a torch
    build whose CUDA wheel only ships kernels for capability >= 7.5.
    torch.cuda.is_available() reports True and get_device_name() works, but
    any real op raises "CUDA error: no kernel image is available for
    execution on the device" -- detect() must not report "CUDA" for this.
    """
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(
        cuda_available=True, device_name="NVIDIA GeForce GTX 1070",
        capability=(6, 1),
        arch_list=["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]))
    info = hardware.detect()
    assert info.kind == "cpu"
    assert info.label == "Linux · CPU"


def test_detect_reports_mps_when_available_and_no_cuda(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(
        cuda_available=False, mps_available=True))
    info = hardware.detect()
    assert info == hardware.DeviceInfo("mps", "Apple M-series · MPS", "macOS")


def test_detect_reports_cpu_when_neither_cuda_nor_mps(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    info = hardware.detect()
    assert info == hardware.DeviceInfo("cpu", "Windows · CPU", "Windows")


def test_detect_never_raises_on_broken_capability_check(monkeypatch):
    """If get_device_capability/get_arch_list misbehave, assume usable rather
    than crash the Home screen -- best-effort display, not a hard gate."""
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    torch_mod = _fake_torch(cuda_available=True, device_name="Mystery GPU")
    torch_mod.cuda.get_device_capability = lambda idx=0: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    info = hardware.detect()
    assert info == hardware.DeviceInfo("cuda", "Mystery GPU · CUDA", "Linux")


# ── _cuda_is_usable ───────────────────────────────────────────────────────────

def test_cuda_is_usable_false_when_cuda_unavailable():
    assert hardware._cuda_is_usable(_fake_torch(cuda_available=False)) == (False, "")


def test_cuda_is_usable_true_at_exact_min_supported_capability():
    torch_mod = _fake_torch(cuda_available=True, device_name="X",
                            capability=(7, 5), arch_list=["sm_75", "sm_80"])
    assert hardware._cuda_is_usable(torch_mod) == (True, "X")
