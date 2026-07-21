"""Pure-logic tests for velum_core.volume_stitch (z-stack/time-lapse linking).

No torch/napari/GPU/engine of any kind — every test feeds hand-built 2-D
label arrays directly to stitch_slices(), the same shape any registered
engine's per-plane predict() output would have.
"""
import numpy as np
import pytest

from velum_core.volume_stitch import stitch_slices


def _blank(h=10, w=20):
    return np.zeros((h, w), dtype=np.int32)


# ── input validation ─────────────────────────────────────────────────────────

def test_empty_slices_raises():
    with pytest.raises(ValueError, match="non-empty"):
        stitch_slices([])


def test_mismatched_shapes_raises():
    with pytest.raises(ValueError, match="shape"):
        stitch_slices([_blank(10, 20), _blank(12, 20)])


def test_non_2d_slice_raises():
    with pytest.raises(ValueError, match="shape"):
        stitch_slices([_blank(10, 20), np.zeros((10, 20, 3), dtype=np.int32)])


def test_bad_iou_thresh_raises():
    with pytest.raises(ValueError, match="iou_thresh"):
        stitch_slices([_blank()], iou_thresh=1.5)


# ── basic shape / relabeling ─────────────────────────────────────────────────

def test_single_slice_relabels_consecutively():
    s = _blank()
    s[0:3, 0:3] = 5
    s[5:8, 5:8] = 2
    out = stitch_slices([s])
    assert out.shape == (1, 10, 20)
    assert out.dtype == np.int32
    # Ascending original label order -> ascending new ids.
    assert out[0, 6, 6] == 1     # was label 2 (smaller original id)
    assert out[0, 1, 1] == 2     # was label 5
    assert set(np.unique(out).tolist()) == {0, 1, 2}


def test_all_background_slice_stays_all_zero():
    out = stitch_slices([_blank(), _blank()])
    assert out.shape == (2, 10, 20)
    assert out.max() == 0


# ── cross-slice linking ──────────────────────────────────────────────────────

def test_translating_object_keeps_same_id_across_slices_despite_different_labels():
    """Same physical object, deliberately given unrelated label numbers in
    each slice, must still resolve to one global id — only geometry (IoU)
    drives linking, never the raw label value."""
    s0 = _blank(); s0[2:8, 2:8] = 100          # 6x6 block, label 100
    s1 = _blank(); s1[2:8, 3:9] = 7            # shifted 1px right, label 7
    s2 = _blank(); s2[2:8, 4:10] = 42          # shifted again, label 42

    out = stitch_slices([s0, s1, s2], iou_thresh=0.25)
    g0 = out[0, 4, 4]
    g1 = out[1, 4, 5]
    g2 = out[2, 4, 6]
    assert g0 > 0
    assert g0 == g1 == g2
    assert set(np.unique(out).tolist()) == {0, g0}


def test_no_overlap_between_slices_gets_distinct_ids():
    s0 = _blank(); s0[0:4, 0:4] = 1
    s1 = _blank(); s1[6:10, 15:19] = 1         # far away, zero overlap with s0's object
    out = stitch_slices([s0, s1], iou_thresh=0.1)
    g0 = out[0, 1, 1]
    g1 = out[1, 8, 17]
    assert g0 != g1
    assert set(np.unique(out).tolist()) == {0, g0, g1}


def test_gap_slice_breaks_the_chain():
    """Documented limitation: linking only looks one slice back, so an object
    missing from just one slice gets a *new* id when it reappears, even at
    the exact same location."""
    s0 = _blank(); s0[2:8, 2:8] = 9
    s1 = _blank()                              # nothing detected this slice
    s2 = _blank(); s2[2:8, 2:8] = 9             # same location as s0

    out = stitch_slices([s0, s1, s2], iou_thresh=0.25)
    g0 = out[0, 4, 4]
    g2 = out[2, 4, 4]
    assert g0 > 0 and g2 > 0
    assert g0 != g2                            # chain broken by the gap


def test_split_object_hands_id_to_higher_iou_child():
    """One parent region splits into two children in the next slice; both
    clear the IoU threshold, so the assignment is a genuine conflict — the
    higher-overlap child must inherit the parent's id, the other gets a new
    one (not both claiming it, and not neither)."""
    parent = _blank(10, 20)
    parent[0:10, 0:20] = 5                     # area 200

    child = _blank(10, 20)
    child[0:10, 0:12] = 1                      # area 120, IoU = 120/200 = 0.6
    child[0:10, 12:20] = 2                     # area 80,  IoU = 80/200  = 0.4

    out = stitch_slices([parent, child], iou_thresh=0.25)
    parent_id = out[0, 0, 0]
    a_id = out[1, 0, 0]      # higher-IoU child
    b_id = out[1, 0, 15]     # lower-IoU child

    assert a_id == parent_id                   # inherits
    assert b_id != parent_id and b_id != 0      # new id, not dropped
    assert a_id != b_id


def test_candidate_below_threshold_never_links():
    parent = _blank(10, 20); parent[0:10, 0:20] = 5
    weak = _blank(10, 20); weak[0:10, 18:20] = 1   # tiny sliver, low IoU
    out = stitch_slices([parent, weak], iou_thresh=0.9)
    assert out[0, 0, 0] != out[1, 0, 19]


# ── min_size filtering ───────────────────────────────────────────────────────

def test_min_size_drops_small_instances_across_the_whole_volume():
    s0 = _blank()
    s0[0:6, 0:6] = 1                           # big object: 36 px in this slice
    s0[9, 19] = 2                              # small object: 1 px total, ever

    s1 = _blank()
    s1[0:6, 0:6] = 7                           # big object continues (72 px total)

    out = stitch_slices([s0, s1], iou_thresh=0.25, min_size=5)
    assert out[0, 9, 19] == 0                  # small instance dropped everywhere
    assert out[0, 2, 2] > 0 and out[1, 2, 2] > 0
    assert out[0, 2, 2] == out[1, 2, 2]        # big instance untouched and linked
    assert set(np.unique(out).tolist()) == {0, out[0, 2, 2]}   # relabelled densely


def test_min_size_zero_keeps_everything():
    s = _blank(); s[0, 0] = 1
    out = stitch_slices([s], min_size=0)
    assert out[0, 0, 0] == 1
