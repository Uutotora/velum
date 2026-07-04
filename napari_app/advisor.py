"""
Local, offline segmentation advisor — the "agent" behind the Assistant tab.

Two independent layers, both fully local (nothing leaves the machine):

1. A deterministic diagnostic engine (`diagnose`) that inspects the current
   image + predicted mask + inference parameters and returns concrete,
   evidence-backed recommendations. Each recommendation carries a `changes`
   dict the UI can apply with one click and re-run — that is what makes it
   agentic rather than a static tip sheet.

2. An optional bridge to a locally-running Ollama LLM (`ollama_available`,
   `ollama_models`, `ollama_chat`) for natural-language Q&A. It is discovered
   at runtime and degrades gracefully to the diagnostic engine when no local
   model is serving. Uses only the standard library (urllib) — no new deps.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

OLLAMA_HOST = "http://localhost:11434"


# ── Diagnostic engine ──────────────────────────────────────────────────────────

@dataclass
class Finding:
    severity: str            # "good" | "info" | "warn"
    title: str
    detail: str
    changes: dict[str, Any] = field(default_factory=dict)
    action: str = ""         # button label; empty → no action button


def _gray(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    return arr.astype(np.float64)


def image_stats(image: np.ndarray | None) -> dict[str, float]:
    g = _gray(image)
    if g is None or g.size == 0:
        return {}
    if g.max() <= 1.0:
        g = g * 255.0
    lo, hi = np.percentile(g, [2, 98])
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "contrast": float(hi - lo),
        "h": float(g.shape[0]),
        "w": float(g.shape[1]),
    }


def mask_stats(mask: np.ndarray | None) -> dict[str, float]:
    if mask is None:
        return {"n_cells": 0}
    mask = np.asarray(mask)
    n = int(mask.max())
    if n == 0:
        return {"n_cells": 0, "coverage": 0.0}
    counts = np.bincount(mask.ravel())
    areas = counts[1:][counts[1:] > 0].astype(np.float64)
    median = float(np.median(areas)) if areas.size else 0.0
    frag = float((areas < 0.3 * median).sum()) / areas.size if median > 0 else 0.0
    return {
        "n_cells": n,
        "coverage": float(counts[1:].sum()) / mask.size,
        "median_area": median,
        "max_area_frac": float(areas.max()) / mask.size if areas.size else 0.0,
        "fragment_ratio": frag,
    }


def diagnose(image: np.ndarray | None, mask: np.ndarray | None,
             params: dict[str, Any]) -> dict[str, Any]:
    """Return {'findings': [Finding], 'image': {..}, 'mask': {..}}."""
    im = image_stats(image)
    ms = mask_stats(mask)
    findings: list[Finding] = []

    pps = int(params.get("points_per_side", 32))
    iou = float(params.get("pred_iou_thresh", 0.8))
    stab = float(params.get("stability_score_thresh", 0.6))
    nms = float(params.get("box_nms_thresh", 0.05))
    min_area = int(params.get("min_mask_area", 20))
    resize = int(params.get("resize_size", 512))

    n = ms.get("n_cells", 0)

    if mask is None:
        findings.append(Finding(
            "info", "Run a prediction first",
            "Predict on an image, then diagnose the result here for tuning advice."))
        return {"findings": findings, "image": im, "mask": ms}

    # ── No detections ──────────────────────────────────────────────────────────
    if n == 0:
        changes = {
            "pred_iou_thresh": round(max(0.5, iou - 0.15), 2),
            "stability_score_thresh": round(max(0.5, stab - 0.15), 2),
        }
        if pps < 48:
            changes["points_per_side"] = 48
        if resize < 1024:
            changes["resize_size"] = 1024
        findings.append(Finding(
            "warn", "No cells were detected",
            "Thresholds may be too strict, or the cells are too small for the "
            "current resolution. Loosen acceptance thresholds, sample more "
            "points, and increase the inference resolution.",
            changes, "Loosen & upsample, re-run"))
        if im.get("contrast", 255) < 40:
            findings.append(Finding(
                "info", "Low image contrast",
                f"Dynamic range is only ~{im['contrast']:.0f}/255. Faint cells "
                "segment poorly — consider contrast-enhancing the image before "
                "prediction (e.g. CLAHE/normalisation)."))
        return {"findings": findings, "image": im, "mask": ms}

    # ── Over-segmentation / fragments ─────────────────────────────────────────
    frag = ms.get("fragment_ratio", 0.0)
    med = ms.get("median_area", 0.0)
    if frag > 0.22 or (med and med < 40):
        new_min = max(min_area, int(med * 0.35)) if med else max(min_area, 40)
        findings.append(Finding(
            "warn", "Likely over-segmentation",
            f"{frag*100:.0f}% of objects are smaller than a third of the median "
            f"cell ({med:.0f} px). These are usually debris or split cells. "
            "Raise the minimum area and NMS overlap to merge/drop fragments.",
            {"min_mask_area": new_min,
             "box_nms_thresh": round(min(0.3, nms + 0.05), 3)},
            "Filter fragments, re-run"))

    # ── Under-segmentation / merged cells ─────────────────────────────────────
    if ms.get("max_area_frac", 0.0) > 0.12 and n < 15:
        findings.append(Finding(
            "warn", "Cells may be merged",
            "At least one object covers a large fraction of the field while the "
            "total count is low — a sign that touching cells were merged. "
            "Lower the NMS overlap and sample more points to separate them.",
            {"box_nms_thresh": round(max(0.02, nms - 0.03), 3),
             "points_per_side": min(96, pps + 16)},
            "Split merged cells, re-run"))

    # ── Sparse sampling on a dense field ──────────────────────────────────────
    if pps < 32 and ms.get("coverage", 0) > 0.25:
        findings.append(Finding(
            "info", "Grid density looks low for this field",
            f"Coverage is {ms['coverage']*100:.0f}% but only {pps} points/side "
            "are sampled. Denser sampling finds cells the grid currently skips.",
            {"points_per_side": 48}, "Increase density, re-run"))

    # ── Small cells vs resolution ─────────────────────────────────────────────
    if med and med < 120 and resize < 1024:
        findings.append(Finding(
            "info", "Small cells at low resolution",
            f"Median cell is only ~{med:.0f} px. Predicting at {resize}px loses "
            "detail on small objects — 1024px usually recovers more of them.",
            {"resize_size": 1024}, "Predict at 1024, re-run"))

    if not any(f.severity == "warn" for f in findings):
        findings.insert(0, Finding(
            "good", f"Segmentation looks healthy — {n} cells",
            "No structural problems detected in the mask. Fine-tune thresholds "
            "only if you can see specific errors in the viewer."))

    return {"findings": findings, "image": im, "mask": ms}


def findings_to_text(diag: dict[str, Any]) -> str:
    icon = {"good": "✓", "info": "•", "warn": "⚠"}
    lines = []
    for f in diag["findings"]:
        lines.append(f"{icon.get(f.severity, '•')} {f.title}\n    {f.detail}")
        if f.changes:
            chg = ", ".join(f"{k}={v}" for k, v in f.changes.items())
            lines.append(f"    → suggested: {chg}")
    return "\n".join(lines)


# ── Optional local LLM bridge (Ollama) ─────────────────────────────────────────

def ollama_available(timeout: float = 0.4) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ollama_models(timeout: float = 1.0) -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def build_context_prompt(diag: dict[str, Any], params: dict[str, Any]) -> str:
    """System prompt grounding the LLM in CellSeg1 and the live result."""
    im, ms = diag.get("image", {}), diag.get("mask", {})
    im_txt = (f"image {im.get('w',0):.0f}×{im.get('h',0):.0f} px, "
              f"mean {im.get('mean',0):.0f}, contrast {im.get('contrast',0):.0f}/255"
              if im else "no image loaded")
    ms_txt = (f"{ms.get('n_cells',0)} cells, coverage {ms.get('coverage',0)*100:.0f}%, "
              f"median area {ms.get('median_area',0):.0f} px, "
              f"fragment ratio {ms.get('fragment_ratio',0)*100:.0f}%"
              if ms.get("n_cells") else "no cells detected yet")
    par_txt = ", ".join(f"{k}={v}" for k, v in params.items())
    return (
        "You are the built-in assistant for CellSeg1, a napari app that segments "
        "cells with a SAM (Segment Anything) backbone fine-tuned via LoRA. You help "
        "microscopists tune segmentation and interpret morphometry. Be concise, "
        "practical, and specific about which parameter to change and in which "
        "direction. Parameters you can advise on: points_per_side, pred_iou_thresh, "
        "stability_score_thresh, box_nms_thresh, min_mask_area, resize_size, "
        "lora_rank, and training epochs/learning-rate.\n\n"
        f"Current image: {im_txt}.\n"
        f"Current result: {ms_txt}.\n"
        f"Current parameters: {par_txt}.\n"
        f"Automated diagnosis:\n{findings_to_text(diag)}"
    )


def ollama_chat(model: str, messages: list[dict[str, str]],
                on_token: Callable[[str], None], stop: Callable[[], bool] | None = None) -> str:
    """Stream a chat completion from Ollama. Returns the full text.

    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    on_token: called with each text chunk as it arrives.
    """
    payload = json.dumps({"model": model, "messages": messages, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat", data=payload,
        headers={"Content-Type": "application/json"})
    full = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            if stop and stop():
                break
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = obj.get("message", {}).get("content", "")
            if chunk:
                full.append(chunk)
                on_token(chunk)
            if obj.get("done"):
                break
    return "".join(full)
