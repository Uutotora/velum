"""
Segmentation engines.

CellSeg1's own engine is the SAM + LoRA one-shot pipeline (see predict.py /
inference_cache.py). This module adds a second, complementary engine:

  Cellpose-SAM (Pachitariu, Rariden & Stringer, 2025) — a generalist foundation
  model that pairs a SAM backbone with the Cellpose framework and gives strong
  *zero-shot* accuracy across imaging modalities, with no training or checkpoint
  required. It is the recommended choice when you don't have a fine-tuned LoRA
  for your data.

The two engines are interchangeable at the mask level, so everything downstream
(measurements, the Assistant, export) works the same regardless of which ran.
"""
from __future__ import annotations

import numpy as np

_cp_model = None
_cp_key: str | None = None


def cellpose_available() -> bool:
    try:
        import cellpose  # noqa: F401
        return True
    except Exception:
        return False


def _cellpose_device(device: str):
    """Map the app's device string to a torch device cellpose accepts."""
    import torch
    if device == "mps" and torch.backends.mps.is_available():
        return dict(gpu=True, device=torch.device("mps"))
    if device not in ("cpu", "mps") and torch.cuda.is_available():
        return dict(gpu=True)          # a CUDA index → let cellpose pick
    return dict(gpu=False)


def predict_cellpose(image_rgb: np.ndarray, diameter: float = 0.0,
                     flow_threshold: float = 0.4, cellprob_threshold: float = 0.0,
                     device: str = "cpu") -> np.ndarray:
    """Zero-shot instance segmentation with Cellpose-SAM.

    image_rgb : H×W×3 uint8.
    diameter  : expected cell diameter in px; 0 → let the model estimate.
    Returns an int32 label mask (0 = background).
    """
    global _cp_model, _cp_key
    from cellpose import models

    if _cp_model is None or _cp_key != device:
        _cp_model = models.CellposeModel(**_cellpose_device(device))
        _cp_key = device

    diam = None if not diameter or diameter <= 0 else float(diameter)
    kwargs = dict(flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
    if diam is not None:
        kwargs["diameter"] = diam
    try:
        out = _cp_model.eval(image_rgb, **kwargs)
    except TypeError:
        # Older/newer signature mismatch — retry with only the image.
        out = _cp_model.eval(image_rgb)

    masks = out[0] if isinstance(out, (tuple, list)) else out
    return np.ascontiguousarray(masks).astype(np.int32)


def invalidate_cellpose():
    global _cp_model, _cp_key
    _cp_model = None
    _cp_key = None
