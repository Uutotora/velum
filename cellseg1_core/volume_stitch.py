"""
3-D instance stitching for z-stacks and time-lapse sequences.

Some engines (SAM, SAM2's automatic mask generator, Cellpose in 2-D mode)
segment one plane at a time with no notion of object identity between planes:
run independently on slice ``z`` and slice ``z+1``, the same physical cell
gets an unrelated label id in each. This module links those per-slice
instance masks into one volume with a single consistent id per physical
object, so a z-stack or time-lapse produces real 3-D (or 2-D+t) instances
instead of a stack of unrelated 2-D ones.

Design
------
Slices are linked adjacent-pair-wise, in order: for slice ``z``, every local
instance is matched against the *already-linked* global ids of slice ``z-1``
by IoU (intersection over union of their pixel masks). A match at or above
``iou_thresh`` carries the global id forward (same physical cell); anything
left unmatched starts a new global id (a cell entering the stack, or the
z-stack equivalent of a cell that just wasn't detected one slice earlier).
Candidate matches are resolved greedily by descending IoU with each side used
at most once, so a cell that splits into two regions in the next slice hands
its id to whichever child overlaps it most, not both.

This is the same overlap-linking idea Cellpose uses for its own "stitch 2-D
masks across z" mode — a simple, well-precedented alternative to a full
tracker (Kalman/Hungarian) that needs no motion model and is cheap enough to
run on every adjacent pair with plain numpy.

Limitations (documented rather than silently accepted):
  * Linking only looks one slice back. An object entirely absent from one
    slice (e.g. a true gap, or a slice that failed to segment it) breaks the
    chain — it gets a new id when it reappears rather than being reconnected.
  * Matching is greedy, not a global optimum (no Hungarian algorithm) — for
    the common case (adjacent slices of the same z-stack are highly similar)
    this makes no practical difference, but a pathological arrangement of
    many equally-overlapping candidates could in principle pick a
    less-than-optimal assignment.
  * No motion model: this assumes slice-to-slice displacement is small enough
    that IoU overlap is meaningful (true for z-stacks and most time-lapse
    imaging; not true for fast-moving objects between frames).

The engine that produced each per-slice mask is irrelevant here — this module
only ever looks at label geometry, never at how the labels were produced, so
it works identically regardless of which segmentation engine ran each slice.
"""
from __future__ import annotations

import numpy as np


def _label_areas(slice_: np.ndarray) -> dict[int, int]:
    """``{label: pixel_count}`` for every positive label in one 2-D slice."""
    ids, counts = np.unique(slice_[slice_ > 0], return_counts=True)
    return dict(zip(ids.tolist(), counts.tolist()))


def _match_iou(prev_global: np.ndarray, curr_local: np.ndarray,
              iou_thresh: float) -> dict[int, int]:
    """Best-effort one-to-one match of ``curr_local``'s labels to
    ``prev_global``'s ids.

    Returns ``{curr_label: prev_global_id}`` for every pair whose IoU clears
    ``iou_thresh``, resolved greedily by descending IoU so each side is used
    at most once. Only pixel geometry is compared — the actual label *values*
    in either slice are irrelevant beyond identifying which pixels belong to
    which object.
    """
    if prev_global.size == 0 or curr_local.size == 0:
        return {}
    overlap_mask = (prev_global > 0) & (curr_local > 0)
    if not np.any(overlap_mask):
        return {}

    pairs = np.stack([prev_global[overlap_mask], curr_local[overlap_mask]], axis=1)
    uniq_pairs, inter_counts = np.unique(pairs, axis=0, return_counts=True)

    area_prev = _label_areas(prev_global)
    area_curr = _label_areas(curr_local)

    candidates = []
    for (p, c), inter in zip(uniq_pairs.tolist(), inter_counts.tolist()):
        union = area_prev[p] + area_curr[c] - inter
        iou = inter / union if union > 0 else 0.0
        if iou >= iou_thresh:
            candidates.append((iou, p, c))
    # Descending IoU so the strongest matches claim their labels first.
    candidates.sort(key=lambda t: t[0], reverse=True)

    matched: dict[int, int] = {}
    used_prev: set[int] = set()
    used_curr: set[int] = set()
    for iou, p, c in candidates:
        if p in used_prev or c in used_curr:
            continue
        matched[c] = p
        used_prev.add(p)
        used_curr.add(c)
    return matched


def _relabel_consecutive(volume: np.ndarray) -> np.ndarray:
    """Remap arbitrary positive labels to a dense ``1..N`` range (0 stays 0)."""
    ids = np.unique(volume)
    ids = ids[ids > 0]
    if ids.size == 0:
        return volume.astype(np.int32)
    lut = np.zeros(int(volume.max()) + 1, dtype=np.int32)
    lut[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    return lut[volume]


def stitch_slices(slices, iou_thresh: float = 0.25,
                  min_size: int = 0) -> np.ndarray:
    """Link independently-labelled 2-D slices into one 3-D instance volume.

    Parameters
    ----------
    slices : sequence of Z 2-D int label arrays, each H×W (0 = background).
        Labels need not be consistent between slices — only overlap geometry
        matters (see module docstring).
    iou_thresh : minimum IoU for two regions in adjacent slices to be treated
        as the same physical object. Lower links more eagerly (more tolerant
        of shape change between slices); higher requires near-identical
        cross-sections.
    min_size : drop instances whose total voxel count (summed across every
        slice) is below this, after linking.

    Returns
    -------
    (Z, H, W) int32 label volume with one consistent id per physical object,
    relabelled to a dense ``1..N`` range.
    """
    slices = [np.asarray(s) for s in slices]
    if not slices:
        raise ValueError("slices must be non-empty")
    h, w = slices[0].shape
    for s in slices:
        if s.ndim != 2 or s.shape != (h, w):
            raise ValueError(
                f"all slices must share one 2-D H×W shape; got {s.shape} vs {(h, w)}")
    if not 0.0 <= iou_thresh <= 1.0:
        raise ValueError("iou_thresh must be in [0, 1]")

    z = len(slices)
    volume = np.zeros((z, h, w), dtype=np.int32)
    next_id = 1

    ids0 = np.unique(slices[0])
    ids0 = ids0[ids0 > 0]
    for local in ids0.tolist():
        volume[0][slices[0] == local] = next_id
        next_id += 1

    for zi in range(1, z):
        prev_global = volume[zi - 1]
        curr_local = slices[zi]
        matched = _match_iou(prev_global, curr_local, iou_thresh)
        ids = np.unique(curr_local)
        ids = ids[ids > 0]
        for local in ids.tolist():
            glob = matched.get(local)
            if glob is None:
                glob = next_id
                next_id += 1
            volume[zi][curr_local == local] = glob

    if min_size > 0:
        ids, counts = np.unique(volume[volume > 0], return_counts=True)
        drop = ids[counts < min_size]
        if drop.size:
            volume[np.isin(volume, drop)] = 0

    return _relabel_consecutive(volume)
