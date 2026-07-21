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
                "segment poorly — enhance contrast before prediction.",
                {"clahe": True}, "Enable CLAHE, re-run"))
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


# Static persona/instructions — used both for live chat and for baking a
# task-specialised Ollama model via a Modelfile (see ollama_create_agent).
AGENT_SYSTEM = (
    "You are CellSeg1 Assistant, the built-in expert for CellSeg1 — a napari "
    "application that segments cells in microscopy images with a Segment Anything "
    "(SAM) backbone fine-tuned via LoRA, then reports per-cell morphometry.\n\n"
    "Your users are microscopists and cell biologists, not ML engineers. Give "
    "concise, concrete, actionable answers. When a parameter should change, name "
    "it, give the new value, and say why in one clause.\n\n"
    "Tunable inference parameters and their effects:\n"
    "• points_per_side (4-96): sampling grid density. Higher finds more/denser "
    "cells but is slower.\n"
    "• pred_iou_thresh (0-1): mask-quality gate. Lower to recover missed cells, "
    "raise to drop false positives.\n"
    "• stability_score_thresh (0-1): confidence gate. Same direction as IoU.\n"
    "• box_nms_thresh (0-1): overlap suppression. Lower separates touching cells, "
    "higher merges duplicates.\n"
    "• min_mask_area (px): discards objects smaller than this — use to remove debris.\n"
    "• resize_size (256/512/768/1024): inference resolution. 1024 recovers small "
    "cells at the cost of speed.\n"
    "Training levers: lora_rank (adapter capacity), epochs, learning rate.\n\n"
    "ACTION PROTOCOL: whenever you recommend a concrete parameter value, append a "
    "line of the exact form `SUGGEST: <param>=<value>` (one per line, machine-"
    "readable, in addition to your prose). The app turns these into one-click "
    "Apply buttons. Only suggest parameters from the list above. Never invent "
    "values you cannot justify from the current diagnosis."
)


def _context_block(diag: dict[str, Any], params: dict[str, Any]) -> str:
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
        f"Current image: {im_txt}.\n"
        f"Current result: {ms_txt}.\n"
        f"Current parameters: {par_txt}.\n"
        f"Automated diagnosis:\n{findings_to_text(diag)}"
    )


def build_context_prompt(diag: dict[str, Any], params: dict[str, Any]) -> str:
    """Full system prompt = persona + live context (for a generic base model)."""
    return f"{AGENT_SYSTEM}\n\n---\nLIVE CONTEXT\n{_context_block(diag, params)}"


def build_live_message(diag: dict[str, Any], params: dict[str, Any]) -> str:
    """Just the live context, to prepend when using a pre-baked agent model."""
    return f"[live context]\n{_context_block(diag, params)}"


def build_tuning_prompt(diag: dict[str, Any], params: dict[str, Any],
                        trajectory: list[Any]) -> str:
    """Prompt for the agentic tuning loop's LLM-driven strategy
    (:func:`velum_core.tuning_loop.llm_propose_fn`): like
    :func:`build_context_prompt`, but framed as one round of an autonomous
    tuning session with the score trajectory so far, so the model reasons
    about what it already tried instead of re-suggesting the same thing
    blind — the ReAct pattern (reason, then act) applied to this specific
    loop rather than free-form chat. ``trajectory`` items need only
    ``step``/``score``/``n_cells``/``changes`` attributes (a
    ``tuning_loop.TuningStep``, not imported here to avoid a cross-module
    coupling in the other direction).
    """
    lines = []
    for s in trajectory:
        chg = ", ".join(f"{k}={v}" for k, v in s.changes.items()) or "(baseline)"
        lines.append(f"  round {s.step}: score={s.score:.3f}  cells={s.n_cells}  change={chg}")
    hist_txt = "\n".join(lines) if lines else "  (no rounds yet — this is the baseline)"
    return (
        f"{AGENT_SYSTEM}\n\n---\n"
        "AUTO-TUNE MODE: you are running an autonomous tuning loop against a "
        "ground-truth mask, not chatting with a person. Each round you either "
        "suggest exactly one parameter change to try next, or decide tuning "
        "has plateaued.\n\n"
        f"Score so far (0-1, higher is better — mean instance AP against "
        f"ground truth):\n{hist_txt}\n\n"
        f"{_context_block(diag, params)}\n\n"
        "Reply with a short reason (one sentence), then either one line of "
        "the exact form `SUGGEST: <param>=<value>` for the single change you "
        "want to try next, or a line `STOP: <one-sentence reason>` if you "
        "believe further rounds are unlikely to improve the score. Never "
        "repeat a change already listed in the history above."
    )


_SUGGEST_RE = None


def parse_suggestions(text: str) -> dict[str, Any]:
    """Extract `SUGGEST: key=value` lines from an LLM reply into a changes dict."""
    global _SUGGEST_RE
    if _SUGGEST_RE is None:
        import re
        _SUGGEST_RE = re.compile(r"SUGGEST:\s*([a-z_]+)\s*=\s*([0-9.]+)", re.IGNORECASE)
    allowed = {
        "points_per_side", "pred_iou_thresh", "stability_score_thresh",
        "box_nms_thresh", "min_mask_area", "resize_size", "lora_rank",
    }
    ints = {"points_per_side", "min_mask_area", "resize_size", "lora_rank"}
    out: dict[str, Any] = {}
    for key, val in _SUGGEST_RE.findall(text):
        key = key.lower()
        if key not in allowed:
            continue
        try:
            out[key] = int(float(val)) if key in ints else float(val)
        except ValueError:
            continue
    return out


def parse_stop(text: str) -> str | None:
    """Extract a `STOP: <reason>` line from an LLM reply (the auto-tune
    loop's "finish" action — see :func:`build_tuning_prompt`), or ``None``
    if the reply doesn't contain one."""
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("STOP:"):
            return line.split(":", 1)[1].strip() or "the model asked to stop."
    return None


# Curated local models that run well on a scientist's laptop and are strong at
# following the action protocol. Sizes are approximate download footprints.
RECOMMENDED_MODELS = [
    {"name": "llama3.2:3b",  "size": "2.0 GB", "note": "Fast, great default on 8–16 GB RAM"},
    {"name": "qwen2.5:7b",   "size": "4.7 GB", "note": "Sharper reasoning, needs ~16 GB RAM"},
    {"name": "phi3.5",       "size": "2.2 GB", "note": "Tiny and quick, good on low-RAM machines"},
    {"name": "mistral:7b",   "size": "4.1 GB", "note": "Balanced general-purpose model"},
]

AGENT_MODEL_NAME = "cellseg1-assistant"


def ollama_pull(model: str, on_progress: Callable[[str, float], None],
                stop: Callable[[], bool] | None = None) -> bool:
    """Download a model via Ollama, reporting (status, fraction) as it streams."""
    payload = json.dumps({"name": model, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/pull", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                if stop and stop():
                    return False
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                status = obj.get("status", "")
                total = obj.get("total") or 0
                done = obj.get("completed") or 0
                frac = (done / total) if total else 0.0
                on_progress(status, frac)
                if obj.get("error"):
                    return False
        return True
    except Exception as e:
        on_progress(f"error: {e}", 0.0)
        return False


def ollama_create_agent(base_model: str,
                        on_progress: Callable[[str], None] | None = None) -> bool:
    """Bake a task-specialised model on top of ``base_model``.

    This is the idiomatic way to configure a local agent for a specific job:
    a Modelfile that pins the persona (SYSTEM) and sampling parameters so the
    assistant is deterministic and domain-focused rather than a generic chatbot.
    Produces the model ``cellseg1-assistant``.
    """
    params = {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1}
    modelfile = (
        f"FROM {base_model}\n"
        f"SYSTEM \"\"\"{AGENT_SYSTEM}\"\"\"\n"
        "PARAMETER temperature 0.2\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER repeat_penalty 1.1\n"
    )
    # Newer Ollama expects structured fields; older builds want a raw modelfile.
    payloads = [
        {"model": AGENT_MODEL_NAME, "from": base_model,
         "system": AGENT_SYSTEM, "parameters": params, "stream": True},
        {"name": AGENT_MODEL_NAME, "modelfile": modelfile, "stream": True},
    ]
    last_err = "unknown error"
    for payload in payloads:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/create", data=data,
            headers={"Content-Type": "application/json"})
        try:
            failed = False
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if on_progress and obj.get("status"):
                        on_progress(obj["status"])
                    if obj.get("error"):
                        last_err = obj["error"]
                        failed = True
                        break
            if not failed:
                return True
        except Exception as e:
            last_err = str(e)
            continue
    if on_progress:
        on_progress(f"error: {last_err}")
    return False


def ollama_chat(model: str, messages: list[dict[str, str]],
                on_token: Callable[[str], None], stop: Callable[[], bool] | None = None,
                temperature: float = 0.2) -> str:
    """Stream a chat completion from Ollama. Returns the full text.

    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    on_token: called with each text chunk as it arrives.
    temperature: low by default so tuning advice is precise and repeatable.
    """
    payload = json.dumps({
        "model": model, "messages": messages, "stream": True,
        "options": {"temperature": temperature},
    }).encode("utf-8")
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
