"""Unit tests for napari_app.core.tuning_loop (the agentic predict -> score ->
adjust loop).

``run_tuning_loop`` itself is tested with fully scripted fakes so the
looping/plateau/stop logic is deterministic and independent of any real
image or model. ``default_score_fn``/``default_propose_fn`` — the real
``benchmark.evaluate``/``advisor.diagnose``-backed defaults used in
production — are each tested separately against small synthetic label
arrays, mirroring tests/test_benchmark.py and tests/test_advisor.py.
"""
import numpy as np
import pytest

from napari_app.core.tuning_loop import (
    TuningStep,
    best_step,
    default_propose_fn,
    default_score_fn,
    run_tuning_loop,
)

_DUMMY_MASK = np.zeros((2, 2), dtype=np.int32)


def _predict_fn(params):
    return None, _DUMMY_MASK


def _scripted_score_fn(scores):
    """A score_fn that yields each value in ``scores`` in turn."""
    it = iter(scores)

    def score(mask):
        s = next(it)
        return s, {"f1": s}

    return score


def _const_propose_fn(changes):
    return lambda image, mask, params: dict(changes)


# ── run_tuning_loop: plateau / stop / budget logic ──────────────────────────

def test_stops_after_patience_flat_rounds():
    # step0=0.5 (best) -> step1=0.6 (>best+delta, new best) -> step2=0.6
    # (not > 0.6+0.01, plateau=1) -> step3=0.6 (plateau=2=patience -> stop).
    # Each round proposes a *different* change (an incrementing counter) so
    # the plateau rule is what stops the loop, not the seen-signature guard.
    score_fn = _scripted_score_fn([0.5, 0.6, 0.6, 0.6, 0.99])
    propose_fn = lambda image, mask, params: {"knob": params.get("knob", 0) + 1}
    traj = run_tuning_loop(
        {"pred_iou_thresh": 0.8}, _predict_fn, score_fn, propose_fn=propose_fn,
        max_steps=10, patience=2, min_delta=0.01)
    assert [round(s.score, 2) for s in traj] == [0.5, 0.6, 0.6, 0.6]
    assert [s.step for s in traj] == [0, 1, 2, 3]


def test_records_full_param_snapshot_and_change_delta_per_step():
    score_fn = _scripted_score_fn([0.1, 0.1, 0.1])
    propose_fn = _const_propose_fn({"pred_iou_thresh": 0.5})
    traj = run_tuning_loop(
        {"pred_iou_thresh": 0.8, "resize_size": 512}, _predict_fn, score_fn,
        propose_fn=propose_fn, patience=2, min_delta=0.01)
    assert traj[0].changes == {}                       # baseline: no change applied yet
    assert traj[0].params["pred_iou_thresh"] == 0.8     # baseline params untouched
    assert traj[1].changes == {"pred_iou_thresh": 0.5}  # the delta that produced step 1
    assert traj[1].params["pred_iou_thresh"] == 0.5     # and it's reflected in the snapshot
    assert traj[1].params["resize_size"] == 512         # untouched keys carry forward


def test_stops_when_propose_fn_has_nothing_left():
    score_fn = _scripted_score_fn([0.4, 0.9])
    traj = run_tuning_loop(
        {"a": 1}, _predict_fn, score_fn, propose_fn=lambda i, m, p: None,
        max_steps=10, patience=5)
    assert len(traj) == 1   # improved at step 0 (vs -inf), but nothing to try next


def test_stops_when_proposed_change_is_a_noop():
    # advisor-style filtering already drops no-ops in default_propose_fn, but
    # run_tuning_loop must also cope with a custom propose_fn that doesn't.
    score_fn = _scripted_score_fn([0.4, 0.9])
    propose_fn = _const_propose_fn({})
    traj = run_tuning_loop({"a": 1}, _predict_fn, score_fn, propose_fn=propose_fn,
                           max_steps=10, patience=5)
    assert len(traj) == 1


def test_guards_against_repeating_the_same_change_forever():
    # A pathological propose_fn that always suggests the identical change
    # would otherwise oscillate/loop forever without the seen-signature guard.
    score_fn = _scripted_score_fn([0.5, 0.5, 0.5, 0.5, 0.5])
    propose_fn = _const_propose_fn({"box_nms_thresh": 0.1})
    traj = run_tuning_loop({"box_nms_thresh": 0.05}, _predict_fn, score_fn,
                           propose_fn=propose_fn, max_steps=10, patience=10)
    assert len(traj) == 2   # step0 (baseline) + step1 (first try) then repeat detected


def test_respects_max_steps_when_still_improving():
    scores = [i * 0.1 for i in range(1, 21)]   # always improving, never plateaus
    score_fn = _scripted_score_fn(scores)
    propose_fn = lambda image, mask, params: {"n": params.get("n", 0) + 1}
    traj = run_tuning_loop({"n": 0}, _predict_fn, score_fn, propose_fn=propose_fn,
                           max_steps=4, patience=10)
    assert len(traj) == 4


def test_should_stop_cancels_cooperatively():
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 1   # let the first round through, stop before the second

    score_fn = _scripted_score_fn([0.5, 0.9, 0.9])
    propose_fn = _const_propose_fn({"a": 1})
    traj = run_tuning_loop({}, _predict_fn, score_fn, propose_fn=propose_fn,
                           max_steps=10, patience=10, should_stop=should_stop)
    assert len(traj) == 1


def test_on_step_fires_once_per_recorded_round():
    score_fn = _scripted_score_fn([0.5, 0.6, 0.6, 0.6])
    propose_fn = _const_propose_fn({"a": 1})
    seen = []
    traj = run_tuning_loop({}, _predict_fn, score_fn, propose_fn=propose_fn,
                           patience=2, min_delta=0.01, on_step=seen.append)
    assert seen == traj
    assert all(isinstance(s, TuningStep) for s in seen)


def test_never_mutates_initial_params():
    original = {"pred_iou_thresh": 0.8}
    score_fn = _scripted_score_fn([0.5, 0.6, 0.6])
    propose_fn = _const_propose_fn({"pred_iou_thresh": 0.1})
    run_tuning_loop(original, _predict_fn, score_fn, propose_fn=propose_fn,
                    patience=2, min_delta=0.01)
    assert original == {"pred_iou_thresh": 0.8}


def test_n_cells_reads_mask_max():
    def predict_fn(params):
        m = np.zeros((3, 3), dtype=np.int32)
        m[0, 0] = 5
        return None, m

    score_fn = _scripted_score_fn([0.5, 0.5])
    traj = run_tuning_loop({}, predict_fn, score_fn, propose_fn=lambda *a: None, patience=1)
    assert traj[0].n_cells == 5


# ── best_step ────────────────────────────────────────────────────────────────

def test_best_step_picks_highest_score_not_last():
    traj = [
        TuningStep(0, {}, {}, 0.5),
        TuningStep(1, {}, {}, 0.9),
        TuningStep(2, {}, {}, 0.7),
    ]
    assert best_step(traj).step == 1


def test_best_step_empty_trajectory_is_none():
    assert best_step([]) is None


# ── default_score_fn (real benchmark.evaluate) ──────────────────────────────

def _two_cells(offset=0):
    m = np.zeros((60, 60), dtype=np.int32)
    m[5 + offset:15 + offset, 5 + offset:15 + offset] = 1
    m[40:50, 40:50] = 2
    return m


def test_default_score_fn_perfect_prediction_scores_near_one():
    gt = _two_cells()
    score, metrics = default_score_fn(gt)(gt.copy())
    assert score == pytest.approx(1.0)
    assert metrics["ap@0.5"] == 1.0


def test_default_score_fn_empty_prediction_scores_zero():
    gt = _two_cells()
    score, metrics = default_score_fn(gt)(np.zeros_like(gt))
    assert score == 0.0
    assert metrics["fn"] == 2


# ── default_propose_fn (real advisor.diagnose) ──────────────────────────────

BASE_PARAMS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.6,
    "box_nms_thresh": 0.05,
    "min_mask_area": 20,
    "resize_size": 512,
}


def _grid_mask(n, side=16, gap=6, canvas=500):
    m = np.zeros((canvas, canvas), dtype=np.int32)
    idx = 1
    r = c = 5
    while idx <= n:
        m[r:r + side, c:c + side] = idx
        idx += 1
        c += side + gap
        if c > canvas - side:
            c = 5
            r += side + gap
        if r > canvas - side:
            break
    return m


def test_default_propose_fn_suggests_loosening_when_no_cells():
    img = np.full((100, 100), 128, dtype=np.uint8)
    changes = default_propose_fn(img, np.zeros((100, 100), dtype=np.int32), BASE_PARAMS)
    assert changes is not None
    assert changes["pred_iou_thresh"] < BASE_PARAMS["pred_iou_thresh"]


def test_default_propose_fn_returns_none_for_a_healthy_mask():
    img = np.random.RandomState(0).randint(0, 255, (500, 500), dtype=np.uint8)
    m = _grid_mask(20)   # uniform, well-separated -> advisor reports "good", no changes
    assert default_propose_fn(img, m, BASE_PARAMS) is None


def test_default_propose_fn_filters_out_current_values():
    # A finding whose suggested value already equals the current parameter
    # must not count as a proposal (would spin the loop with no real change).
    img = np.full((100, 100), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.int32)
    params = dict(BASE_PARAMS, pred_iou_thresh=0.5, stability_score_thresh=0.45,
                 points_per_side=48, resize_size=1024)
    changes = default_propose_fn(img, mask, params)
    # No-cells findings would suggest iou/stability <= current and pps/resize
    # already at their ceiling -> nothing left that actually differs.
    assert changes is None or all(params.get(k) != v for k, v in changes.items())


# ── run_tuning_loop wired to the real defaults end-to-end ───────────────────

def test_end_to_end_with_real_score_and_propose_fns():
    """A tiny integration check: real advisor.diagnose + real benchmark.evaluate,
    with only predict_fn faked, plateau on a fixed 'good enough' mask."""
    gt = _two_cells()

    def predict_fn(params):
        # Always predicts the two GT cells regardless of params -> the score
        # is stable immediately, so the loop should plateau quickly.
        img = np.zeros((60, 60), dtype=np.uint8)
        return img, gt.copy()

    score_fn = default_score_fn(gt)
    traj = run_tuning_loop(BASE_PARAMS, predict_fn, score_fn,
                           propose_fn=default_propose_fn, max_steps=8, patience=2)
    assert traj[0].score == pytest.approx(1.0)
    assert best_step(traj).score == pytest.approx(1.0)
    # A perfect, unchanging score is its own plateau: advisor reports "good"
    # (no findings with changes) once the mask looks healthy, so the loop
    # should stop quickly rather than churning for max_steps.
    assert len(traj) <= 3
