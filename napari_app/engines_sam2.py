"""
SAM 2 (Meta, 2024) segmentation engine.

SAM2 is a zero-shot foundation model like Cellpose-SAM, generalising to new
imaging modalities with no checkpoint of your own required — but unlike SAM1
and Cellpose it was trained for *video* (and by extension volumetric z-stack)
segmentation with temporal/depth consistency as a first-class goal, which is
why it is CellSeg1's flagship engine for z-stacks and time-lapse (see
:mod:`napari_app.core.predict_controller`'s ``_predict_volume`` for the
plane-by-plane-plus-stitch orchestration that actually drives a stack through
this engine).

This module only ever does the *2-D, single-plane* half of that: given one
RGB frame, run SAM2's automatic mask generator (the SAM1
``SamAutomaticMaskGenerator`` equivalent — same "segment everything, no
prompt needed" contract, same per-mask ``segmentation``/``area`` output
shape) and return an instance label mask. Volume orchestration is a layer
above and is completely engine-agnostic: it calls this exactly like it calls
any other registered engine's ``predict(image, config)``, one plane at a
time, and stitches the results — so nothing in this module has any notion of
"z-stack" at all.

Not implemented (documented rather than silently half-done): SAM2's other
headline capability, the *video predictor* (memory-bank-conditioned mask
*propagation* from a prompted first frame, tracking through occlusion far
better than independent-per-plane detection + IoU stitching can). That is a
strictly stronger but substantially more complex mode — a prompted object
per video, not "segment everything on every frame" — and is left as a future
enhancement; the automatic-mask-generator-per-plane path implemented here is
the same trade-off Cellpose users already make for their own 3-D mode, and
needs no interactive prompting step.

The ``sam2`` package is never imported at module level — only lazily inside
the functions that need it, exactly like ``napari_app.engines`` does for
``cellpose`` — so this module (and the registry it populates at import time)
stays free of the dependency, and ``available()`` reports the plain truth of
whether it's installed rather than crashing the app on import when it isn't.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from napari_app.engine_registry import EngineSpec, register

# Best-effort filename/config conventions for the official checkpoints
# (github.com/facebookresearch/sam2). A user whose installed sam2 package
# ships different config paths can override either via the SAM2 settings
# card's checkpoint/config text fields — these are only the zero-config
# defaults, not a hard requirement.
_SAM2_CHECKPOINTS = {
    "large":     "sam2.1_hiera_large.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "small":     "sam2.1_hiera_small.pt",
    "tiny":      "sam2.1_hiera_tiny.pt",
}
_SAM2_CONFIGS = {
    "large":     "configs/sam2.1/sam2.1_hiera_l.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "small":     "configs/sam2.1/sam2.1_hiera_s.yaml",
    "tiny":      "configs/sam2.1/sam2.1_hiera_t.yaml",
}

_mask_generator = None
_mg_key: tuple | None = None


def sam2_available() -> bool:
    try:
        import sam2  # noqa: F401
        return True
    except Exception:
        return False


def resolve_sam2(checkpoint_text: str, config_text: str, model_type: str,
                 storage_dir) -> tuple[str, str]:
    """Resolve ``(checkpoint_path, config_name)`` for the SAM2 engine.

    Mirrors ``PredictController.resolve_sam``'s convention: an explicit path
    wins, otherwise fall back to ``<storage_dir>/sam2_checkpoints/<name>``
    for the checkpoint. The Hydra config name has no equivalent on-disk
    lookup (it is a package-relative name, not a file the user downloads), so
    an empty override always falls back to the built-in guess.
    """
    model_type = model_type if model_type in _SAM2_CHECKPOINTS else "large"

    ckpt = (checkpoint_text or "").strip()
    if not ckpt or not Path(ckpt).exists():
        name = _SAM2_CHECKPOINTS[model_type]
        candidate = Path(storage_dir) / "sam2_checkpoints" / name
        if candidate.exists():
            ckpt = str(candidate)
        else:
            raise ValueError(
                f"SAM2 checkpoint not found. Place {name} in "
                f"{Path(storage_dir) / 'sam2_checkpoints'}/ (download from "
                "github.com/facebookresearch/sam2), or set a custom path above.")

    config_name = (config_text or "").strip() or _SAM2_CONFIGS[model_type]
    return ckpt, config_name


def _torch_device(device: str):
    import torch
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device not in ("cpu", "mps") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _mg_cache_key(config: dict) -> tuple:
    return (
        config["sam2_checkpoint"],
        config["sam2_config_name"],
        config.get("selected_device", "cpu"),
        config.get("points_per_side", 32),
        config.get("pred_iou_thresh", 0.8),
        config.get("stability_score_thresh", 0.6),
        config.get("box_nms_thresh", 0.7),
        config.get("min_mask_area", 0),
    )


def _build_mask_generator(config: dict):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = _torch_device(config.get("selected_device", "cpu"))
    model = build_sam2(config["sam2_config_name"], config["sam2_checkpoint"], device=device)
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=config.get("points_per_side", 32),
        pred_iou_thresh=config.get("pred_iou_thresh", 0.8),
        stability_score_thresh=config.get("stability_score_thresh", 0.6),
        box_nms_thresh=config.get("box_nms_thresh", 0.7),
        min_mask_region_area=config.get("min_mask_area", 0),
    )


def get_mask_generator(config: dict):
    """Return the cached SAM2 automatic mask generator for ``config``,
    building it if the checkpoint/config/device/thresholds changed."""
    global _mask_generator, _mg_key
    key = _mg_cache_key(config)
    if _mask_generator is None or _mg_key != key:
        _mask_generator = _build_mask_generator(config)
        _mg_key = key
    return _mask_generator


def predict_sam2(image_rgb: np.ndarray, config: dict) -> np.ndarray:
    """Zero-shot instance segmentation with SAM2's automatic mask generator.

    image_rgb : H×W×3 uint8. Returns an int32 label mask (0 = background).
    """
    from predict import sam_output_to_mask

    mg = get_mask_generator(config)
    output = mg.generate(image_rgb)
    if not output:
        return np.zeros(image_rgb.shape[:2], dtype=np.int32)
    return sam_output_to_mask(output).astype(np.int32)


def invalidate_sam2():
    global _mask_generator, _mg_key
    _mask_generator = None
    _mg_key = None


def cache_status() -> str:
    if _mg_key is None:
        return "model: —"
    return f"model: {Path(_mg_key[0]).name}"


# ── Registry wiring ───────────────────────────────────────────────────────────

def _predict_sam2_engine(image: np.ndarray, config: dict) -> np.ndarray:
    return predict_sam2(image, config)


def _sam2_available_check() -> bool:
    # A thin wrapper (not sam2_available directly) so tests that monkeypatch
    # napari_app.engines_sam2.sam2_available still take effect — EngineSpec
    # would otherwise hold a frozen reference to whatever function object
    # existed at register() time (see napari_app.engines for the same fix).
    return sam2_available()


register(EngineSpec(
    key="sam2",
    label="SAM 2 · zero-shot (z-stack / video, experimental)",
    predict=_predict_sam2_engine,
    available=_sam2_available_check,
    status_line=cache_status,
    bench_label="SAM 2 (zero-shot, experimental)",
    result_label="SAM 2",
))
