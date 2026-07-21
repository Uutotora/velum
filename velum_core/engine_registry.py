"""Segmentation engine registry.

A registered engine is a small :class:`EngineSpec`: a stable ``key``, a
display ``label``, and a ``predict(image, config) -> label_mask`` callable —
the one interface :mod:`velum_core.predict_controller` needs to run
inference, regardless of which engine is selected. The Predict tab's engine
selector and benchmark checklist both list whatever is registered here
instead of hardcoding engine names, so adding a new engine (StarDist,
InstanSeg, Micro-SAM, DeepCell, ...) is a single :func:`register` call in the
module that defines it — see :mod:`velum_core.engines` for the two built-in
engines.

Heavy per-engine dependencies (torch, cellpose, ...) are never imported here;
each engine's ``predict``/``available``/``status_line`` callables import them
lazily, exactly as the built-in engines already did before this registry
existed. Config *building* (the very different parameter shapes SAM+LoRA vs.
Cellpose need) and per-engine settings UI are intentionally not part of this
interface — those stay bespoke per engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class EngineSpec:
    """One registered segmentation engine.

    ``predict`` is the only required behaviour. ``available`` probes whether
    the engine's dependency is installed (defaults to always-available).
    ``status_line`` is an optional live status string for the post-run log
    line (e.g. cache state); when absent, ``result_label`` is used instead.
    ``bench_label``/``result_label`` are shorter display strings for the
    benchmark checklist and results table respectively; both default to
    ``label`` when not given.
    """
    key: str
    label: str
    predict: Callable[[np.ndarray, dict], np.ndarray]
    available: Callable[[], bool] = lambda: True
    status_line: Callable[[], str] | None = None
    bench_label: str = ""
    result_label: str = ""

    def __post_init__(self):
        self.bench_label = self.bench_label or self.label
        self.result_label = self.result_label or self.label


_registry: dict[str, EngineSpec] = {}


def register(spec: EngineSpec) -> None:
    """Register (or replace) an engine under ``spec.key``."""
    _registry[spec.key] = spec


def get(key: str) -> EngineSpec:
    """Look up a registered engine, raising a clear error if unknown."""
    try:
        return _registry[key]
    except KeyError:
        raise ValueError(f"Unknown engine {key!r}. Registered: {sorted(_registry)}")


def all_engines() -> list[EngineSpec]:
    """Every registered engine, in registration order."""
    return list(_registry.values())


def is_registered(key: str) -> bool:
    return key in _registry
