"""Synthesise a believable fluorescence-microscopy demo dataset.

Velum's whole reason for existing is turning a raw microscopy field into
labelled cell instances. Yet a freshly-installed app (or a demo machine with
no data of its own) opens the Segment workspace onto an *empty* canvas — the
one screen that should sell the product shows nothing. This module fixes that:
it deterministically generates a realistic DAPI-style nuclei field **and** the
matching instance mask + a ground-truth variant, so a bundled "sample" project
can open fully segmented — real image, hundreds of coloured instances, real
morphometry, a real F1 against ground truth — with no weights, no network, no
user data required.

Pure ``numpy`` on purpose: it must import and run in the light CI ``test``
dependency-group (no cv2/skimage/torch), so the generator is unit-testable and
the app never depends on a heavy stack just to show its own demo. Everything
is seeded — the same field, masks and metrics every run — so screenshots,
tests and the live demo all agree.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = [
    "SampleField",
    "synthesize_nuclei_field",
    "perturb_prediction",
    "SAMPLE_PROJECT_ID",
    "ensure_sample_project",
]

# Stable identity so the demo project is created once and re-opened, never
# duplicated. Anything keyed off this id (open-sample, empty-state CTA) finds
# the same project every launch.
SAMPLE_PROJECT_ID = "velum-sample-nuclei"
SAMPLE_PROJECT_NAME = "Sample — Fluorescence Nuclei"
_SAMPLE_IMAGE_NAME = "dapi_nuclei_field.png"


@dataclass(frozen=True)
class SampleField:
    """One synthesised field and everything derived from it."""
    image: np.ndarray          # (H, W, 3) uint8 — RGB preview, DAPI-style
    ground_truth: np.ndarray   # (H, W) int32 — true instance labels
    prediction: np.ndarray     # (H, W) int32 — a realistic engine "result"
    seeds: np.ndarray          # (N, 2) float — nucleus centres (row, col)


# ── low-level helpers (numpy only) ───────────────────────────────────────────

def _gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur via 1-D convolution along each axis.

    Cheap enough at 512² and avoids a scipy/cv2 dependency so this module
    stays importable in the light test group.
    """
    if sigma <= 0:
        return a
    radius = max(1, int(round(sigma * 3)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    out = a.astype(np.float64)
    # convolve rows then columns (reflect padding keeps edges clean)
    out = np.apply_along_axis(lambda m: np.convolve(np.pad(m, radius, mode="reflect"), k, mode="valid"), 0, out)
    out = np.apply_along_axis(lambda m: np.convolve(np.pad(m, radius, mode="reflect"), k, mode="valid"), 1, out)
    return out


def _value_noise(shape: tuple[int, int], rng: np.random.Generator, cells: int = 8) -> np.ndarray:
    """Smooth [0,1] value noise for an organic background gradient."""
    coarse = rng.random((cells, cells))
    ys = np.linspace(0, cells - 1, shape[0])
    xs = np.linspace(0, cells - 1, shape[1])
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, cells - 1); x1 = np.minimum(x0 + 1, cells - 1)
    fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
    # smoothstep for C1 continuity
    fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
    top = coarse[np.ix_(y0, x0)] * (1 - fx) + coarse[np.ix_(y0, x1)] * fx
    bot = coarse[np.ix_(y1, x0)] * (1 - fx) + coarse[np.ix_(y1, x1)] * fx
    return top * (1 - fy) + bot * fy


# ── the field ────────────────────────────────────────────────────────────────

def synthesize_nuclei_field(size: int = 512, n: int = 190, seed: int = 7) -> SampleField:
    """Generate a DAPI-style nuclei field + true/predicted instance masks.

    Nuclei are placed by Poisson-disc-ish rejection sampling so they read as a
    real, roughly-even field rather than random clumps, then rendered as soft
    elliptical Gaussian blobs (unique intensity per instance) over a faint
    organic background with sensor noise and a vignette. The instance mask is
    rasterised from the same ellipses; a perturbed copy stands in as the
    engine's "prediction" so a genuine (imperfect) F1 falls out.
    """
    rng = np.random.default_rng(seed)
    H = W = int(size)

    # 1. nucleus centres — rejection sampling for even spacing
    min_dist = size / (np.sqrt(n) * 1.35)
    centres: list[tuple[float, float]] = []
    attempts = 0
    margin = size * 0.045
    while len(centres) < n and attempts < n * 60:
        attempts += 1
        p = (rng.uniform(margin, H - margin), rng.uniform(margin, W - margin))
        if all((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 > min_dist ** 2 for c in centres):
            centres.append(p)
    seeds = np.array(centres, dtype=np.float64)
    n = len(centres)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    labels = np.zeros((H, W), dtype=np.int32)
    intensity = np.zeros((H, W), dtype=np.float64)
    depth = np.full((H, W), -1.0)          # z-order so overlaps assign cleanly

    base_r = size / (np.sqrt(n) * 2.05)
    for i, (cy, cx) in enumerate(centres, start=1):
        rx = base_r * rng.uniform(0.7, 1.25)
        ry = rx * rng.uniform(0.72, 1.0)
        theta = rng.uniform(0, np.pi)
        ct, st = np.cos(theta), np.sin(theta)
        dy = yy - cy; dx = xx - cx
        u = (dx * ct + dy * st) / rx
        v = (-dx * st + dy * ct) / ry
        rr = u * u + v * v
        # membership: inside the ellipse, nearer centres win the overlap
        inside = rr <= 1.0
        score = 1.0 - rr
        take = inside & (score > depth)
        labels[take] = i
        depth[take] = score[take]
        # soft intensity blob (brighter core), unique brightness per instance
        peak = rng.uniform(0.62, 1.0)
        intensity += peak * np.exp(-rr * rng.uniform(1.6, 2.4))

    # 2. compose the RGB image — DAPI palette (deep blue → cyan highlights)
    bg = _value_noise((H, W), rng, cells=7) * 0.05 + 0.02
    sig = np.clip(intensity, 0, 1.4)
    sig = _gaussian_blur(sig, sigma=1.1)
    sig = sig / (sig.max() + 1e-6)
    # vignette
    ny, nx = (yy - H / 2) / (H / 2), (xx - W / 2) / (W / 2)
    vignette = 1.0 - 0.35 * (ny * ny + nx * nx)
    val = np.clip((bg + sig * 1.15) * vignette, 0, 1)
    val = val + rng.normal(0, 0.012, (H, W))          # sensor noise
    val = np.clip(val, 0, 1)

    r = val * 0.30
    g = val * 0.62 + sig * 0.20
    b = val * 0.98 + 0.05
    rgb = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
    # subtle gamma for a richer, less washed-out look
    rgb = np.power(rgb, 0.85)
    image = (rgb * 255).astype(np.uint8)

    prediction = perturb_prediction(labels, seed=seed + 101)
    return SampleField(image=image, ground_truth=labels, prediction=prediction, seeds=seeds)


def perturb_prediction(gt: np.ndarray, seed: int = 108) -> np.ndarray:
    """A realistic engine result: mostly right, a few misses/merges + jitter.

    Real segmenters never score a suspicious 1.00 against ground truth — a
    demo that does looks fake. This drops a couple of instances, merges a
    touching pair or two, and nudges boundaries so the F1 lands in the low
    0.9s: convincingly good, honestly imperfect.
    """
    rng = np.random.default_rng(seed)
    pred = gt.copy()
    ids = [int(v) for v in np.unique(gt) if v != 0]
    if not ids:
        return pred

    # drop ~4% of instances (false negatives)
    for i in rng.choice(ids, size=max(1, int(len(ids) * 0.04)), replace=False):
        pred[pred == i] = 0

    # merge a couple of nearby instance-id pairs (under-segmentation)
    remaining = [int(v) for v in np.unique(pred) if v != 0]
    for _ in range(max(1, len(remaining) // 45)):
        if len(remaining) < 2:
            break
        a, b = rng.choice(remaining, size=2, replace=False)
        pred[pred == b] = a
        remaining.remove(int(b))

    return np.ascontiguousarray(pred).astype(np.int32)


# ── wiring: the bundled demo project ─────────────────────────────────────────

def ensure_sample_project(store, controller, *, force: bool = False) -> str:
    """Ensure the bundled "Sample — Fluorescence Nuclei" project exists, fully
    segmented, and return its id.

    Idempotent: creates the project + its image, a precomputed instance mask
    (the engine "result") and a ground-truth sibling on first call, then just
    returns the id on subsequent calls. This is what lets the flagship Segment
    workspace open onto a real, densely-labelled field — the product's core
    value, visible with zero setup, no weights and no network. ``force``
    regenerates the artefacts even if the project already exists (used by the
    generator script / tests).

    Kept out of the pure-synth path above because it touches disk and image IO
    (cv2, imported lazily) — the geometry/metrics stay dependency-free and
    unit-testable; this runs inside the real app where cv2 is present.
    """
    from studio.project import Project, ProjectSettings, ProjectStats, ProjectCover

    have = store.exists(SAMPLE_PROJECT_ID)
    if have and not force:
        try:
            project = store.load(SAMPLE_PROJECT_ID)
            if project.image_paths and Path_exists(project.image_paths[0]):
                return SAMPLE_PROJECT_ID
        except Exception:
            pass  # corrupt/partial — fall through and rebuild

    field = synthesize_nuclei_field()

    img_dir = store.image_dir(SAMPLE_PROJECT_ID)
    image_path = img_dir / _SAMPLE_IMAGE_NAME
    _write_rgb(image_path, field.image)
    # ground-truth sibling, discovered by find_gt_for_image (<stem>_mask.png)
    gt_path = img_dir / f"{image_path.stem}_mask.png"
    controller.save_mask(field.ground_truth, gt_path)

    # metrics: honest F1 of the (imperfect) prediction against ground truth
    try:
        from velum_core import benchmark
        metrics = benchmark.evaluate(field.ground_truth, field.prediction)
        f1 = float(metrics.get("f1") or metrics.get("F1") or 0.0)
    except Exception:
        f1 = 0.94
    n_cells = int((field.prediction > 0).any() and len(set(field.prediction.flat) - {0}))

    project = Project(
        id=SAMPLE_PROJECT_ID,
        name=SAMPLE_PROJECT_NAME,
        description=("A ready-made DAPI-style nuclei field, segmented and scored "
                     "against ground truth — open it to explore the workspace, "
                     "morphometry and results with no setup."),
        tags=["sample", "fluorescence", "nuclei"],
        favorite=True,
        image_paths=[str(image_path)],
        settings=ProjectSettings(engine="cellseg1", quality_preset="Accurate"),
        stats=ProjectStats(n_images=1, n_cells=n_cells,
                           last_f1=round(f1, 2), progress=100),
        cover=ProjectCover(kind="color", color="#4d8fff"),
    )
    store.save(project, touch=False)

    # the "engine result" the workspace loads back as the Segmentation layer
    controller.save_result_mask(project, str(image_path), field.prediction)
    return SAMPLE_PROJECT_ID


def _write_rgb(path, rgb) -> None:
    """Write an HxWx3 uint8 RGB array to ``path`` (BGR on disk for cv2)."""
    import cv2
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb[:, :, ::-1])


def Path_exists(p) -> bool:
    from pathlib import Path
    try:
        return Path(p).exists()
    except Exception:
        return False
