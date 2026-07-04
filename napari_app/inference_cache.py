"""
Model + embedding cache for the napari Predict widget.

Two levels:
  1. Model cache  — avoids re-loading SAM + LoRA from disk on every run.
  2. Embedding cache — avoids re-running the ViT encoder (the slow part)
     when only inference thresholds change (IoU, stability, NMS, min_area).

The embedding cache works by subclassing SamPredictor with a CachingPredictor
that intercepts every set_image() call. Each call hashes the image crop, looks
up cached features, and skips the encoder if the crop was already processed.
This works correctly even with crop_n_layers=1 (multiple crops per image).

Cache is invalidated when:
  - checkpoint path, vit_name, lora_rank, sam_image_size, or device changes
    → full model reload + full embedding invalidation
  - image path or mtime changes
    → embedding cache cleared, model kept
  - only inference params change (IoU, NMS thresholds, etc.)
    → both caches kept, decoder re-runs on cached embeddings (~1-2 sec)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

# ── Module-level singletons ────────────────────────────────────────────────────

_model: Any = None          # LoRA_Sam
_model_key: tuple | None = None

# embed_cache: maps sha256(image_crop_bytes) → features tensor (on device)
_embed_cache: dict = {}
_embed_model_key: tuple | None = None   # which model these embeddings belong to
_embed_img_key:   tuple | None = None   # (image_path, mtime, resize_w, resize_h)


# ── Key helpers ───────────────────────────────────────────────────────────────

def _mk_model_key(config: dict) -> tuple:
    return (
        config["result_pth_path"],
        config["vit_name"],
        config["image_encoder_lora_rank"],
        config["sam_image_size"],
        config.get("selected_device", "cpu"),
    )


def _mk_img_key(config: dict) -> tuple:
    p = config["image_path"]
    try:
        import os
        mtime = os.path.getmtime(p)
    except OSError:
        mtime = 0.0
    rs = config["resize_size"]
    return (p, mtime, rs[0], rs[1])


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ── CachingPredictor ──────────────────────────────────────────────────────────

class CachingPredictor:
    """Wraps SamPredictor; intercepts set_image to use cached encoder output."""

    def __init__(self, predictor, embed_cache: dict):
        self._p = predictor
        self._cache = embed_cache

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        key = _hash_array(image)
        if key in self._cache:
            # Restore cached state without running the encoder
            cached = self._cache[key]
            self._p.reset_image()
            self._p.original_size = cached["original_size"]
            self._p.input_size    = cached["input_size"]
            self._p.features      = cached["features"]
            self._p.is_image_set  = True
        else:
            # Run the encoder normally and store the result
            self._p.set_image(image, image_format)
            self._cache[key] = {
                "original_size": self._p.original_size,
                "input_size":    self._p.input_size,
                "features":      self._p.features,
            }

    def reset_image(self) -> None:
        self._p.reset_image()

    # Forward every other attribute access to the wrapped predictor
    def __getattr__(self, name: str):
        return getattr(self._p, name)


# ── Public API ────────────────────────────────────────────────────────────────

def predict_cached(config: dict, image_rgb: np.ndarray) -> np.ndarray:
    """
    Run prediction using cached model and embeddings where possible.

    image_rgb : uint8 H×W×3, already resized to config["resize_size"].
    Returns   : instance mask (uint16/int64).
    """
    global _model, _model_key, _embed_cache, _embed_model_key, _embed_img_key

    mkey = _mk_model_key(config)
    ikey = _mk_img_key(config)

    # ── Level 1: model cache ──────────────────────────────────────────────────
    if _model is None or _model_key != mkey:
        _model = _load_model(config)
        _model_key = mkey
        _embed_cache.clear()
        _embed_model_key = mkey
        _embed_img_key   = None

    # ── Level 2: embedding cache — invalidate on image change ─────────────────
    if ikey != _embed_img_key:
        _embed_cache.clear()
        _embed_img_key = ikey

    # ── Run prediction with caching predictor ─────────────────────────────────
    from predict import sam_output_to_mask
    from segment_anything import SamAutomaticMaskGeneratorOptMaskNMS
    from segment_anything.predictor import SamPredictor

    model_sam = _model.sam if hasattr(_model, "sam") else _model

    # Build mask generator; swap its internal predictor with caching wrapper
    mg = SamAutomaticMaskGeneratorOptMaskNMS(
        model=model_sam,
        points_per_side=config["points_per_side"],
        points_per_batch=config["points_per_batch"],
        crop_n_layers=config["crop_n_layers"],
        crop_n_points_downscale_factor=config["crop_n_points_downscale_factor"],
        box_nms_thresh=config["box_nms_thresh"],
        crop_nms_thresh=config["crop_nms_thresh"],
        pred_iou_thresh=config["pred_iou_thresh"],
        min_mask_region_area=config["min_mask_region_area"],
        max_mask_region_area_ratio=config["max_mask_region_area_ratio"],
        stability_score_thresh=config["stability_score_thresh"],
        stability_score_offset=config["stability_score_offset"],
    )
    # Replace predictor with our caching wrapper
    mg.predictor = CachingPredictor(mg.predictor, _embed_cache)

    import torch
    with torch.no_grad():
        output = mg.generate(image_rgb)

    if not output:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint16)
    return sam_output_to_mask(output)


def get_model(config: dict):
    """Return the cached LoRA_Sam for ``config``, loading it if needed.

    Shares the singleton used by predict_cached so the interactive segmenter
    and the automatic generator never hold two copies of the weights.
    """
    global _model, _model_key, _embed_cache, _embed_model_key, _embed_img_key
    mkey = _mk_model_key(config)
    if _model is None or _model_key != mkey:
        _model = _load_model(config)
        _model_key = mkey
        _embed_cache.clear()
        _embed_model_key = mkey
        _embed_img_key = None
    return _model


def _load_model(config: dict):
    import os
    dev = config.get("selected_device", "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if dev in ("cpu", "mps") else dev
    if dev == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

    from predict import load_model_from_config
    from set_environment import set_env
    set_env(
        config["deterministic"], config["seed"],
        config["allow_tf32_on_cudnn"], config["allow_tf32_on_matmul"],
    )
    model = load_model_from_config(config, empty_lora=False)
    model.eval()
    return model


def invalidate_model():
    """Call when the user switches to a different checkpoint."""
    global _model, _model_key, _embed_cache, _embed_model_key, _embed_img_key
    _model = None
    _model_key = None
    _embed_cache.clear()
    _embed_model_key = None
    _embed_img_key = None


def invalidate_embeddings():
    """Call when image path or resize changes but checkpoint stays the same."""
    global _embed_cache, _embed_img_key
    _embed_cache.clear()
    _embed_img_key = None


def cache_status() -> str:
    n_crops = len(_embed_cache)
    model_name = Path(_model_key[0]).name if _model_key else "—"
    img_name   = Path(_embed_img_key[0]).name if _embed_img_key else "—"
    return (
        f"model: {model_name}  |  "
        f"embed: {img_name} ({n_crops} crop{'s' if n_crops != 1 else ''} cached)"
    )
