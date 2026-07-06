"""Unit tests for napari_app.tiling (tiled large-image segmentation).

The stitching logic is the load-bearing part — it decides whether a cell cut
by a tile boundary comes back as one cell or two. These tests pin that with a
fake connected-component predictor so no GPU/model is needed.
"""
import numpy as np
import pytest
from scipy import ndimage

from napari_app.tiling import (
    Tile,
    plan_tiles,
    recommend_overlap,
    should_tile,
    should_warn_no_tiling,
    stitch,
    tiled_predict,
)


# ── plan_tiles ────────────────────────────────────────────────────────────────

def test_single_tile_when_image_fits():
    tiles = plan_tiles(500, 500, tile=1024, overlap=128)
    assert tiles == [Tile(0, 0, 500, 500)]


def test_tiles_cover_every_pixel():
    h, w, tile, overlap = 300, 460, 128, 32
    tiles = plan_tiles(h, w, tile=tile, overlap=overlap)
    cover = np.zeros((h, w), dtype=bool)
    for t in tiles:
        cover[t.y0:t.y1, t.x0:t.x1] = True
    assert cover.all()                       # no uncovered strip anywhere


def test_last_tile_flush_to_edge():
    tiles = plan_tiles(200, 200, tile=120, overlap=40)
    assert max(t.y1 for t in tiles) == 200   # bottom edge reached exactly
    assert max(t.x1 for t in tiles) == 200
    assert all(t.h == 120 and t.w == 120 for t in tiles)


def test_overlap_must_be_valid():
    with pytest.raises(ValueError):
        plan_tiles(100, 100, tile=64, overlap=64)   # overlap == tile
    with pytest.raises(ValueError):
        plan_tiles(100, 100, tile=0, overlap=0)


# ── helpers ───────────────────────────────────────────────────────────────────

def test_should_tile_threshold():
    assert should_tile((4000, 4000), tile=1024)
    assert not should_tile((512, 512), tile=1024)


def test_should_warn_no_tiling_only_when_large_and_untiled():
    assert should_warn_no_tiling((4000, 4000), False, tile=1024)      # large, tiling off
    assert not should_warn_no_tiling((4000, 4000), True, tile=1024)   # large, tiling on
    assert not should_warn_no_tiling((512, 512), False, tile=1024)    # small, tiling off
    assert not should_warn_no_tiling((512, 512), True, tile=1024)     # small, tiling on


def test_recommend_overlap_covers_cell():
    assert recommend_overlap(80, tile=1024) >= 80
    assert recommend_overlap(0, tile=1024) > 0            # sane default
    assert recommend_overlap(10_000, tile=1024) < 1024    # capped below tile


# ── stitch ────────────────────────────────────────────────────────────────────

def test_stitch_single_tile_passthrough():
    local = np.zeros((20, 20), dtype=np.int32)
    local[2:8, 2:8] = 5
    local[12:18, 12:18] = 9
    out = stitch((20, 20), [(Tile(0, 0, 20, 20), local)])
    assert out.max() == 2                    # relabelled to a dense 1..N range
    assert (out > 0).sum() == (local > 0).sum()


def test_stitch_merges_cell_split_across_seam():
    # One horizontal bar spanning the overlap of two side-by-side tiles.
    full = np.zeros((40, 120), dtype=np.int32)
    full[15:25, 20:100] = 1                  # single wide cell
    left = Tile(0, 0, 40, 80)
    right = Tile(0, 40, 40, 120)             # overlaps cols 40..80
    lloc = left.crop(full).copy()
    rloc = right.crop(full).copy()
    out = stitch((40, 120), [(left, lloc), (right, rloc)], merge_frac=0.25)
    assert out.max() == 1                    # one cell, not two halves
    assert (out > 0).sum() == (full > 0).sum()


def test_stitch_keeps_distinct_cells_separate():
    # Two clearly separated cells, one per tile, with an overlapping band
    # that neither cell enters → must remain two labels.
    left = Tile(0, 0, 40, 80)
    right = Tile(0, 40, 40, 120)
    lloc = np.zeros((40, 80), dtype=np.int32)
    lloc[10:20, 5:20] = 3                     # cell fully inside left tile
    rloc = np.zeros((40, 80), dtype=np.int32)
    rloc[10:20, 60:75] = 7                    # cell fully inside right tile
    out = stitch((40, 120), [(left, lloc), (right, rloc)], merge_frac=0.25)
    assert out.max() == 2


def test_stitch_min_area_drops_fragments():
    local = np.zeros((30, 30), dtype=np.int32)
    local[2:12, 2:12] = 1                     # 100 px cell
    local[20:22, 20:22] = 2                   # 4 px fragment
    out = stitch((30, 30), [(Tile(0, 0, 30, 30), local)], min_area=10)
    assert out.max() == 1                     # fragment removed


# ── end-to-end tiled_predict ──────────────────────────────────────────────────

def _cc_predictor(tile_img: np.ndarray) -> np.ndarray:
    """Fake engine: label connected components of the foreground."""
    fg = tile_img[..., 0] > 0 if tile_img.ndim == 3 else tile_img > 0
    lab, _ = ndimage.label(fg)
    return lab.astype(np.int32)


def test_tiled_predict_reassembles_large_object():
    # A big rectangle spanning several tiles must come back as ONE cell.
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img[60:140, 40:360] = 255                 # 80 x 320 object across tiles
    out = tiled_predict(img, _cc_predictor, tile=128, overlap=64)
    assert out.shape == (200, 400)
    assert out.max() == 1
    assert (out > 0).sum() == (img[..., 0] > 0).sum()


def test_tiled_predict_counts_multiple_objects():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    # three well-separated blobs, spacing > overlap so none get merged
    for cx in (40, 200, 350):
        img[90:110, cx:cx + 20] = 255
    out = tiled_predict(img, _cc_predictor, tile=128, overlap=48)
    assert out.max() == 3


def test_tiled_predict_progress_callback():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    seen = []
    tiled_predict(img, _cc_predictor, tile=128, overlap=32,
                  on_tile=lambda done, total: seen.append((done, total)))
    assert seen[-1][0] == seen[-1][1]         # ends at 100%
    assert seen[0][0] == 1
