"""
Tiled instance segmentation for large images.

The CellSeg1 pipeline resizes every image to a fixed inference size
(``resize_size``), which destroys detail on whole-slide / high-content
images: a 40 000 px slide squeezed into 1024 px loses every small cell.
This module runs the segmenter on overlapping tiles at native resolution
and stitches the per-tile instance masks back into one global mask,
merging cells that straddle a tile boundary.

Design
------
1. ``plan_tiles`` lays a regular grid of overlapping tiles that fully
   covers the image; the last tile in each axis is flushed to the edge so
   there is never an uncovered strip.
2. The caller runs its segmenter on each tile crop and returns a local
   instance-label mask (0 = background).
3. ``stitch`` composits the tiles greedily. For every object in a tile it
   looks at what is already painted underneath: if it overlaps an existing
   global cell by more than ``merge_frac`` of its own area it is treated as
   the *same* cell (the seam-straddling case) and grows that label;
   otherwise it becomes a new cell. Only background pixels are written, so
   the first tile to claim a pixel keeps it and distinct touching cells are
   not fused.

The pipeline is engine-agnostic: it only needs a ``predict_fn`` that maps an
``H×W×C`` tile to an ``H×W`` int label mask, so it works identically for the
CellSeg1 (SAM+LoRA) and Cellpose engines.

Reliability note: ``overlap`` must exceed the diameter of the largest cell,
otherwise a cell bigger than the overlap band can be seen only partially by
each tile and will not be merged. ``recommend_overlap`` derives a safe value
from an estimated cell diameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Tile:
    """Half-open global bounds of a tile: rows [y0, y1), cols [x0, x1)."""
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y0:self.y1, self.x0:self.x1]


def _axis_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Tile start offsets covering ``length`` with the last flush to the edge."""
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, step))
    if not starts or starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def plan_tiles(h: int, w: int, tile: int = 1024, overlap: int = 128) -> list[Tile]:
    """Grid of overlapping tiles fully covering an ``h×w`` image.

    A single tile covering the whole image is returned when it already fits,
    which makes tiled prediction a safe no-op wrapper for small images.
    """
    if tile <= 0:
        raise ValueError("tile must be positive")
    if not 0 <= overlap < tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")
    ys = _axis_starts(h, tile, overlap)
    xs = _axis_starts(w, tile, overlap)
    th, tw = min(tile, h), min(tile, w)
    return [Tile(y, x, y + th, x + tw) for y in ys for x in xs]


def should_tile(shape: tuple[int, ...], tile: int = 1024, margin: float = 1.5) -> bool:
    """Whether an image is large enough that tiling beats a plain resize.

    ``margin`` avoids tiling images only marginally larger than one tile,
    where the resize loss is negligible and the seam handling is not worth it.
    """
    h, w = shape[0], shape[1]
    return max(h, w) > tile * margin


def should_warn_no_tiling(shape: tuple[int, ...], tiled: bool,
                          tile: int = 1024, margin: float = 1.5) -> bool:
    """Whether to hint at "Large image" mode after a run.

    True when ``should_tile`` would have recommended tiling but ``tiled`` was
    off, meaning the image was silently shrunk for inference and may have
    lost small cells.
    """
    return not tiled and should_tile(shape, tile=tile, margin=margin)


def recommend_overlap(cell_diameter_px: float, tile: int = 1024) -> int:
    """A safe overlap: at least one full cell diameter, capped below the tile."""
    if cell_diameter_px <= 0:
        return max(64, tile // 8)
    ov = int(np.ceil(cell_diameter_px * 1.25))
    return int(min(max(ov, 32), tile - 1))


def _relabel_consecutive(mask: np.ndarray) -> np.ndarray:
    """Remap arbitrary positive labels to a dense 1..N range (0 stays 0)."""
    labels = np.unique(mask)
    labels = labels[labels > 0]
    if labels.size == 0:
        return mask.astype(np.int32)
    lut = np.zeros(int(mask.max()) + 1, dtype=np.int32)
    lut[labels] = np.arange(1, labels.size + 1, dtype=np.int32)
    return lut[mask]


def stitch(shape: tuple[int, int],
           placements: list[tuple[Tile, np.ndarray]],
           merge_frac: float = 0.25,
           min_area: int = 0) -> np.ndarray:
    """Composite per-tile instance masks into one global instance mask.

    Parameters
    ----------
    shape : (H, W) of the full image.
    placements : list of (tile, local_mask) where ``local_mask`` is the tile's
        instance labels (0 = background), same H×W as the tile.
    merge_frac : an incoming object that overlaps an existing global cell by
        at least this fraction of its own area is merged into it (same cell
        seen from two tiles). Lower → merges more eagerly.
    min_area : drop objects smaller than this many pixels after stitching.
    """
    H, W = shape
    canvas = np.zeros((H, W), dtype=np.int32)
    next_label = 1

    for tile, local in placements:
        local = np.ascontiguousarray(local)
        if local.size == 0 or local.max() == 0:
            continue
        sub = canvas[tile.y0:tile.y1, tile.x0:tile.x1]  # a view into canvas
        # Process larger objects first so a big straddling cell claims the
        # seam before small neighbours nibble at it.
        ids, counts = np.unique(local[local > 0], return_counts=True)
        for lbl in ids[np.argsort(-counts)]:
            obj = local == lbl
            area = int(obj.sum())
            if area == 0:
                continue
            under = sub[obj]
            occupied = under[under > 0]
            if occupied.size:
                vals, vcounts = np.unique(occupied, return_counts=True)
                best = int(vals[vcounts.argmax()])
                if vcounts.max() / area >= merge_frac:
                    target = best            # same cell across the seam
                else:
                    target = next_label
                    next_label += 1
            else:
                target = next_label
                next_label += 1
            # Only claim background: never overwrite a cell already placed.
            sub[obj & (sub == 0)] = target

    if min_area > 0:
        ids, counts = np.unique(canvas[canvas > 0], return_counts=True)
        drop = ids[counts < min_area]
        if drop.size:
            canvas[np.isin(canvas, drop)] = 0

    return _relabel_consecutive(canvas)


def tiled_predict(image: np.ndarray,
                  predict_fn: Callable[[np.ndarray], np.ndarray],
                  tile: int = 1024,
                  overlap: int = 128,
                  merge_frac: float = 0.25,
                  min_area: int = 0,
                  on_tile: Callable[[int, int], None] | None = None) -> np.ndarray:
    """Segment a large image tile-by-tile at native resolution and stitch.

    ``predict_fn`` maps an ``H×W×C`` (or ``H×W``) tile to an ``H×W`` int label
    mask. ``on_tile(done, total)`` is called after each tile for progress.
    Returns a global instance-label mask the same H×W as ``image``.
    """
    h, w = image.shape[:2]
    tiles = plan_tiles(h, w, tile=tile, overlap=overlap)
    placements: list[tuple[Tile, np.ndarray]] = []
    for i, t in enumerate(tiles):
        local = predict_fn(t.crop(image))
        placements.append((t, np.asarray(local)))
        if on_tile is not None:
            on_tile(i + 1, len(tiles))
    return stitch((h, w), placements, merge_frac=merge_frac, min_area=min_area)
