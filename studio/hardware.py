"""Read-only, best-effort compute-hardware detection for the Home screen's
"This device" card (see ``screens.py``'s ``HomeScreen._aside``).

That card used to hard-code ``("Compute", "Apple M-series · MPS")`` — true on
the machine it was designed on, wrong everywhere else (including every Linux
box). This module replaces it with a real, cross-platform check: CUDA on
Linux/Windows, MPS on Apple Silicon, CPU as the honest fallback.

The CUDA-vs-fake-CUDA distinction (a device can be *present* without the
installed torch build shipping kernels for it) is the repo-root
``device_utils.is_usable`` check — shared with the classic app's device
dropdowns/status label, see that module's docstring for the real-hardware
case (a Linux GTX 1070) that motivated it.

torch is imported lazily inside ``detect()``, not at module top level: studio's
shared modules stay import-light (see ``studio/__init__.py``) even though torch
is a real project dependency (``pyproject.toml``) that ends up installed.
"""
from __future__ import annotations

import platform
import warnings
from dataclasses import dataclass

import device_utils

_OS_NAMES = {"Darwin": "macOS", "Linux": "Linux", "Windows": "Windows"}


@dataclass(frozen=True)
class DeviceInfo:
    kind: str       # "cuda" | "mps" | "cpu"
    label: str      # e.g. "NVIDIA GeForce GTX 1070 · CUDA", "Linux · CPU"
    os_name: str     # "macOS" | "Linux" | "Windows" | whatever platform.system() said


def _os_name() -> str:
    system = platform.system()
    return _OS_NAMES.get(system, system or "Unknown")


def _cuda_is_usable(torch) -> tuple[bool, str]:
    """(True, device name) only if CUDA is present AND this build ships its kernels."""
    if not torch.cuda.is_available():
        return False, ""
    name = torch.cuda.get_device_name(0)
    return device_utils.is_usable(torch, 0), name


def detect() -> DeviceInfo:
    """Best-effort compute device for display. Never raises."""
    os_name = _os_name()
    try:
        with warnings.catch_warnings():
            # torch itself warns (via UserWarning) when a CUDA device's
            # capability doesn't match any shipped kernel -- we already handle
            # that case explicitly above, so don't let it reach the console.
            warnings.simplefilter("ignore")
            import torch
            cuda_ok, cuda_name = _cuda_is_usable(torch)
            if cuda_ok:
                return DeviceInfo("cuda", f"{cuda_name} · CUDA", os_name)
            if torch.backends.mps.is_available():
                return DeviceInfo("mps", "Apple M-series · MPS", os_name)
    except Exception:
        pass
    return DeviceInfo("cpu", f"{os_name} · CPU", os_name)
