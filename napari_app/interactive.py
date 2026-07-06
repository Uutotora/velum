"""
Interactive prompt-based segmentation — the engine behind click-to-segment.

Wraps a SamPredictor around the loaded LoRA-SAM and keeps the (expensive)
image embedding resident so each click only runs the lightweight mask decoder.
Positive/negative point prompts segment or refine a single object at a time,
mirroring the interactive workflow SAM was designed for. A box prompt (drag a
rectangle around a cell instead of clicking it) works the same way — SAM's own
predictor already accepts one alongside or instead of points, so this is a
thin pass-through, not a second code path.

The heavy `set_image` call is done once per image; subsequent `predict` calls
are fast enough to feel live. Coordinates are passed in original-image pixel
(x, y) — the predictor's ResizeLongestSide transform handles scaling to the
fine-tuned input resolution internally.
"""
from __future__ import annotations

import numpy as np


class InteractiveSession:
    """A SamPredictor bound to one image, reusing the cached model weights."""

    def __init__(self, config: dict, image_rgb: np.ndarray):
        from segment_anything.predictor import SamPredictor
        from napari_app import inference_cache

        model = inference_cache.get_model(config)
        sam = model.sam if hasattr(model, "sam") else model
        self.predictor = SamPredictor(sam)
        img = np.ascontiguousarray(image_rgb)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        self.image_shape = img.shape[:2]
        self.predictor.set_image(img, image_format="RGB")

    def predict(self, coords_xy, labels, mask_input=None, box=None):
        """Segment from point and/or box prompts.

        coords_xy : list of (x, y) in original-image pixels; empty for a
                    pure box prompt with no points.
        labels    : list of 1 (foreground) / 0 (background), aligned with
                    coords_xy; empty when coords_xy is.
        mask_input: optional 1×256×256 low-res logits from the previous step,
                    used to refine the same object (from either a point or a
                    box prompt — SAM doesn't distinguish once it has one).
        box       : optional (x0, y0, x1, y1) in original-image pixels — a
                    single box prompt, SAM's XYXY convention.
        Returns (mask_bool HxW, low_res 1x256x256, score).
        """
        import torch

        pc = np.asarray(coords_xy, dtype=np.float32) if len(coords_xy) else None
        pl = np.asarray(labels, dtype=np.int32) if len(labels) else None
        box_arr = np.asarray(box, dtype=np.float32) if box is not None else None
        # A box (or a refine step building on a previous mask) is a far less
        # ambiguous prompt than a single click — SAM's own docs recommend
        # multimask_output=False for exactly these cases, letting the model
        # commit to one mask instead of returning three candidates to choose
        # between.
        multimask = mask_input is None and box_arr is None
        with torch.no_grad():
            masks, scores, low = self.predictor.predict(
                point_coords=pc,
                point_labels=pl,
                box=box_arr,
                mask_input=mask_input,
                multimask_output=multimask,
            )
        best = int(np.argmax(scores))
        return masks[best].astype(bool), low[best][None, :, :], float(scores[best])
