"""Shared, Qt-free compute-device capability check.

Every "is CUDA available" call site in this repo (set_environment.py,
studio/hardware.py) used to ask only
``torch.cuda.is_available()``. That call answers "is a CUDA device present",
not "will this torch build actually run on it" -- and those two questions
diverge for real hardware. Confirmed on a Linux box with a GTX 1070 (Pascal,
compute capability 6.1) against a current CUDA 13 wheel, which only ships
kernels for capability >= 7.5: ``is_available()`` returns ``True`` and
``get_device_name()`` works, but any real op raises "CUDA error: no kernel
image is available for execution on the device". ``is_usable`` below catches
that in advance via ``torch.cuda.get_arch_list()`` -- no kernel launch needed
-- so callers can offer/report a device that will actually work instead of
one that crashes the first real Run.

Every function here takes an already-imported ``torch`` module as its first
argument rather than importing torch itself, so callers (all of which import
it lazily, inside a widget method or a UI-build function, never at a shared
module's top level) keep control of when that heavy import happens.
"""
from __future__ import annotations


def is_usable(torch, index: int = 0) -> bool:
    """True if CUDA device ``index`` is present and has shipped kernels.

    Best-effort: if the capability/arch-list check itself fails for any
    reason, assume usable rather than hide a device that might be fine.
    """
    try:
        capability = torch.cuda.get_device_capability(index)
        supported = []
        for arch in torch.cuda.get_arch_list():
            if not arch.startswith("sm_"):
                continue
            n = int(arch.split("_", 1)[1])
            supported.append((n // 10, n % 10))
        if supported and capability < min(supported):
            return False  # present, but this torch build ships no kernels for it
    except Exception:
        pass
    return True


def usable_cuda_indices(torch) -> list:
    """CUDA device indices that are both present and actually usable."""
    if not torch.cuda.is_available():
        return []
    return [i for i in range(torch.cuda.device_count()) if is_usable(torch, i)]
