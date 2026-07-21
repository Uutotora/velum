"""Unit tests for velum_core.tuning_loop (the agentic predict -> score ->
adjust loop).

``run_tuning_loop`` itself is tested with fully scripted fakes so the
looping/plateau/stop/round-start logic is deterministic and independent of
any real image or model. ``default_score_fn``/``default_propose_fn`` (the
real ``benchmark.evaluate``/``advisor.diagnose``-backed defaults) and
``llm_propose_fn`` (the real tool-calling strategy, backed by
``advisor.ollama_chat`` — monkeypatched here, never a real network call) are
each tested against small synthetic inputs, mirroring tests/test_benchmark.py
and tests/test_advisor.py. ``parameter_importance``/``write_trajectory_csv``/
``describe_stop_reason`` are plain pure-data helpers, tested directly.
"""
import numpy as np
import pytest

from velum_core import tuning_loop
from velum_core.tuning_loop import (
    Proposal,
    TuningStep,
    best_step,
    default_propose_fn,
    default_score_fn,
    describe_stop_reason,
    parameter_importance,
    run_tuning_loop,
    write_trajectory_csv,
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


def _const_propose_fn(changes, reason=""):
    return lambda image, mask, params, trajectory: Proposal(dict(changes), reason)


# ── run_tuning_loop: plateau / stop / budget logic ──────────────────────────

def test_stops_after_patience_flat_rounds():
    # step0=0.5 (best) -> step1=0.6 (>best+delta, new best) -> step2=0.6
    # (not > 0.6+0.01, plateau=1) -> step3=0.6 (plateau=2=patience -> stop).
    # Each round proposes a *different* change (an incrementing counter) so
    # the plateau rule is what stops the loop, not the seen-signature guard.
    score_fn = _scripted_score_fn([0.5, 0.6, 0.6, 0.6, 0.99])
    propose_fn = lambda image, mask, params, trajectory: Proposal({"knob": params.get("knob", 0) + 1})
    result = run_tuning_loop(
        {"pred_iou_thresh": 0.8}, _predict_fn, score_fn, propose_fn=propose_fn,
        max_steps=10, patience=2, min_delta=0.01)
    assert [round(s.score, 2) for s in result.trajectory] == [0.5, 0.6, 0.6, 0.6]
    assert [s.step for s in result.trajectory] == [0, 1, 2, 3]
    assert result.stop_reason == "plateau"


def test_records_full_param_snapshot_change_and_reason_per_step():
    # A propose_fn that always suggests the same change trips the seen-
    # signature guard on its second attempt (see
    # test_guards_against_repeating_the_same_change_forever) — exactly right
    # here, since this test only needs to inspect steps 0 and 1.
    score_fn = _scripted_score_fn([0.1, 0.1])
    propose_fn = _const_propose_fn({"pred_iou_thresh": 0.5}, "try lowering IoU")
    result = run_tuning_loop(
        {"pred_iou_thresh": 0.8, "resize_size": 512}, _predict_fn, score_fn,
        propose_fn=propose_fn, patience=2, min_delta=0.01)
    traj = result.trajectory
    assert traj[0].changes == {} and traj[0].reason == ""     # baseline: no change applied yet
    assert traj[0].params["pred_iou_thresh"] == 0.8            # baseline params untouched
    assert traj[1].changes == {"pred_iou_thresh": 0.5}          # the delta that produced step 1
    assert traj[1].reason == "try lowering IoU"                # why it was tried
    assert traj[1].params["pred_iou_thresh"] == 0.5
    assert traj[1].params["resize_size"] == 512                 # untouched keys carry forward
    assert result.stop_reason == "repeated_change"


def test_stops_when_propose_fn_has_nothing_left():
    score_fn = _scripted_score_fn([0.4, 0.9])
    result = run_tuning_loop(
        {"a": 1}, _predict_fn, score_fn,
        propose_fn=lambda i, m, p, t: Proposal(None, "looks healthy"),
        max_steps=10, patience=5)
    assert len(result.trajectory) == 1   # improved at step 0 (vs -inf), but nothing to try next
    assert result.stop_reason == "no_more_suggestions"


def test_stops_when_proposed_change_is_a_noop():
    # advisor-style filtering already drops no-ops in default_propose_fn, but
    # run_tuning_loop must also cope with a custom propose_fn that doesn't.
    score_fn = _scripted_score_fn([0.4, 0.9])
    propose_fn = _const_propose_fn({})
    result = run_tuning_loop({"a": 1}, _predict_fn, score_fn, propose_fn=propose_fn,
                            max_steps=10, patience=5)
    assert len(result.trajectory) == 1
    assert result.stop_reason == "no_more_suggestions"


def test_guards_against_repeating_the_same_change_forever():
    # A pathological propose_fn that always suggests the identical change
    # would otherwise oscillate/loop forever without the seen-signature guard.
    score_fn = _scripted_score_fn([0.5, 0.5, 0.5, 0.5, 0.5])
    propose_fn = _const_propose_fn({"box_nms_thresh": 0.1})
    result = run_tuning_loop({"box_nms_thresh": 0.05}, _predict_fn, score_fn,
                            propose_fn=propose_fn, max_steps=10, patience=10)
    assert len(result.trajectory) == 2   # step0 (baseline) + step1 (first try) then repeat detected
    assert result.stop_reason == "repeated_change"


def test_respects_max_steps_when_still_improving():
    scores = [i * 0.1 for i in range(1, 21)]   # always improving, never plateaus
    score_fn = _scripted_score_fn(scores)
    propose_fn = lambda image, mask, params, trajectory: Proposal({"n": params.get("n", 0) + 1})
    result = run_tuning_loop({"n": 0}, _predict_fn, score_fn, propose_fn=propose_fn,
                            max_steps=4, patience=10)
    assert len(result.trajectory) == 4
    assert result.stop_reason == "max_steps"


def test_should_stop_cancels_cooperatively():
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 1   # let the first round through, stop before the second

    score_fn = _scripted_score_fn([0.5, 0.9, 0.9])
    propose_fn = _const_propose_fn({"a": 1})
    result = run_tuning_loop({}, _predict_fn, score_fn, propose_fn=propose_fn,
                            max_steps=10, patience=10, should_stop=should_stop)
    assert len(result.trajectory) == 1
    assert result.stop_reason == "cancelled"


def test_on_step_fires_once_per_recorded_round():
    score_fn = _scripted_score_fn([0.5, 0.6, 0.6, 0.6])
    propose_fn = _const_propose_fn({"a": 1})
    seen = []
    result = run_tuning_loop({}, _predict_fn, score_fn, propose_fn=propose_fn,
                             patience=2, min_delta=0.01, on_step=seen.append)
    assert seen == result.trajectory
    assert all(isinstance(s, TuningStep) for s in seen)


def test_on_round_start_fires_before_every_round_with_the_budget():
    score_fn = _scripted_score_fn([0.5, 0.6, 0.6])
    propose_fn = lambda image, mask, params, trajectory: Proposal({"n": params.get("n", 0) + 1})
    calls = []
    run_tuning_loop({"n": 0}, _predict_fn, score_fn, propose_fn=propose_fn,
                    max_steps=8, patience=1, min_delta=0.01,
                    on_round_start=lambda step, total: calls.append((step, total)))
    assert calls == [(0, 8), (1, 8), (2, 8)]


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
    result = run_tuning_loop({}, predict_fn, score_fn,
                             propose_fn=lambda *a: Proposal(None), patience=1)
    assert result.trajectory[0].n_cells == 5


# ── best_step / describe_stop_reason ─────────────────────────────────────────

def test_best_step_picks_highest_score_not_last():
    traj = [
        TuningStep(0, {}, {}, 0.5),
        TuningStep(1, {}, {}, 0.9),
        TuningStep(2, {}, {}, 0.7),
    ]
    assert best_step(traj).step == 1


def test_best_step_empty_trajectory_is_none():
    assert best_step([]) is None


def test_describe_stop_reason_known_codes_are_human_readable():
    for code in tuning_loop.STOP_REASONS:
        text = describe_stop_reason(code)
        assert text and text != code


def test_describe_stop_reason_unknown_code_passes_through():
    assert describe_stop_reason("something_new") == "something_new"


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
    proposal = default_propose_fn(img, np.zeros((100, 100), dtype=np.int32), BASE_PARAMS, [])
    assert proposal.changes is not None
    assert proposal.changes["pred_iou_thresh"] < BASE_PARAMS["pred_iou_thresh"]
    assert proposal.reason   # non-empty, names the finding


def test_default_propose_fn_returns_stop_proposal_for_a_healthy_mask():
    img = np.random.RandomState(0).randint(0, 255, (500, 500), dtype=np.uint8)
    m = _grid_mask(20)   # uniform, well-separated -> advisor reports "good", no changes
    proposal = default_propose_fn(img, m, BASE_PARAMS, [])
    assert proposal.changes is None
    assert proposal.reason   # names the "healthy" assessment


def test_default_propose_fn_filters_out_current_values():
    # A finding whose suggested value already equals the current parameter
    # must not count as a proposal (would spin the loop with no real change).
    img = np.full((100, 100), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.int32)
    params = dict(BASE_PARAMS, pred_iou_thresh=0.5, stability_score_thresh=0.45,
                 points_per_side=48, resize_size=1024)
    proposal = default_propose_fn(img, mask, params, [])
    changes = proposal.changes or {}
    assert all(params.get(k) != v for k, v in changes.items())


# ── llm_propose_fn (tool-calling strategy, backed by a monkeypatched
#    advisor.ollama_chat — no real network/model involved) ──────────────────

def test_llm_propose_fn_parses_a_suggest_line(monkeypatch):
    from velum_core import advisor
    monkeypatch.setattr(
        advisor, "ollama_chat",
        lambda model, messages, on_token: "Cells look merged.\nSUGGEST: box_nms_thresh=0.02")
    propose = tuning_loop.llm_propose_fn("my-model")
    img = np.random.RandomState(0).randint(0, 255, (200, 200), dtype=np.uint8)
    mask = _grid_mask(20)
    proposal = propose(img, mask, dict(BASE_PARAMS), [])
    assert proposal.changes == {"box_nms_thresh": 0.02}
    assert "merged" in proposal.reason.lower()


def test_llm_propose_fn_parses_a_stop_line(monkeypatch):
    from velum_core import advisor
    monkeypatch.setattr(
        advisor, "ollama_chat",
        lambda model, messages, on_token: "Looks converged.\nSTOP: further rounds unlikely to help.")
    propose = tuning_loop.llm_propose_fn("my-model")
    img = np.random.RandomState(0).randint(0, 255, (200, 200), dtype=np.uint8)
    mask = _grid_mask(20)
    proposal = propose(img, mask, dict(BASE_PARAMS), [])
    assert proposal.changes is None
    assert "unlikely to help" in proposal.reason.lower()


def test_llm_propose_fn_ignores_a_suggest_line_matching_the_current_value(monkeypatch):
    from velum_core import advisor
    monkeypatch.setattr(
        advisor, "ollama_chat",
        lambda model, messages, on_token: f"SUGGEST: pred_iou_thresh={BASE_PARAMS['pred_iou_thresh']}")
    propose = tuning_loop.llm_propose_fn("my-model")
    img = np.full((100, 100), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.int32)   # advisor's own fallback: "no cells" -> real changes
    proposal = propose(img, mask, dict(BASE_PARAMS), [])
    assert proposal.changes is not None
    assert "rule-based advisor" in proposal.reason.lower()


def test_llm_propose_fn_falls_back_to_advisor_when_model_errors(monkeypatch):
    from velum_core import advisor

    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(advisor, "ollama_chat", boom)
    propose = tuning_loop.llm_propose_fn("my-model")
    img = np.full((100, 100), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.int32)
    proposal = propose(img, mask, dict(BASE_PARAMS), [])
    assert proposal.changes is not None
    assert proposal.changes["pred_iou_thresh"] < BASE_PARAMS["pred_iou_thresh"]
    assert "unreachable" in proposal.reason.lower()


def test_llm_propose_fn_falls_back_when_reply_has_no_usable_suggestion(monkeypatch):
    from velum_core import advisor
    monkeypatch.setattr(
        advisor, "ollama_chat",
        lambda model, messages, on_token: "I think it looks fine, no changes needed.")
    propose = tuning_loop.llm_propose_fn("my-model")
    img = np.full((100, 100), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.int32)
    proposal = propose(img, mask, dict(BASE_PARAMS), [])
    assert proposal.changes is not None
    assert "rule-based advisor" in proposal.reason.lower()


def test_run_tuning_loop_with_llm_propose_fn_end_to_end(monkeypatch):
    from velum_core import advisor
    replies = iter([
        "Try raising the minimum area.\nSUGGEST: min_mask_area=40",
        "That's enough.\nSTOP: the score has plateaued.",
    ])
    monkeypatch.setattr(advisor, "ollama_chat", lambda *a, **k: next(replies))
    score_fn = _scripted_score_fn([0.5, 0.6])
    result = run_tuning_loop(
        {"min_mask_area": 20}, _predict_fn, score_fn,
        propose_fn=tuning_loop.llm_propose_fn("my-model"),
        max_steps=10, patience=10)
    assert [s.score for s in result.trajectory] == [0.5, 0.6]
    assert result.trajectory[1].changes == {"min_mask_area": 40}
    assert result.stop_reason == "no_more_suggestions"
    assert result.stop_detail == "the score has plateaued."


# ── parameter_importance ─────────────────────────────────────────────────────

def test_parameter_importance_ranks_a_perfectly_correlated_param_first():
    traj = [
        TuningStep(0, {"a": 1, "b": 5}, {}, 0.5),
        TuningStep(1, {"a": 2, "b": 5}, {}, 0.6),
        TuningStep(2, {"a": 3, "b": 5}, {}, 0.7),
    ]
    ranked = parameter_importance(traj)
    keys = [k for k, _ in ranked]
    assert keys == ["a"]   # b never varied -> excluded
    assert ranked[0][1] == pytest.approx(1.0)


def test_parameter_importance_needs_at_least_three_rounds():
    traj = [TuningStep(0, {"a": 1}, {}, 0.5), TuningStep(1, {"a": 2}, {}, 0.9)]
    assert parameter_importance(traj) == []


def test_parameter_importance_ignores_non_numeric_params():
    traj = [
        TuningStep(0, {"a": 1, "engine": "cellseg1"}, {}, 0.5),
        TuningStep(1, {"a": 2, "engine": "cellseg1"}, {}, 0.6),
        TuningStep(2, {"a": 3, "engine": "cellseg1"}, {}, 0.4),
    ]
    ranked = parameter_importance(traj)
    assert all(k != "engine" for k, _ in ranked)


def test_parameter_importance_empty_when_score_never_varies():
    traj = [TuningStep(i, {"a": i}, {}, 0.5) for i in range(4)]
    assert parameter_importance(traj) == []


# ── write_trajectory_csv ─────────────────────────────────────────────────────

def test_write_trajectory_csv_writes_expected_rows(tmp_path):
    traj = [
        TuningStep(0, {"pred_iou_thresh": 0.8}, {}, 0.5, n_cells=3, reason=""),
        TuningStep(1, {"pred_iou_thresh": 0.6}, {"pred_iou_thresh": 0.6}, 0.7,
                  n_cells=5, reason="loosen"),
    ]
    path = tmp_path / "traj.csv"
    write_trajectory_csv(str(path), traj)

    import csv
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["step", "score", "n_cells", "reason", "pred_iou_thresh"]
    assert rows[1] == ["0", "0.5", "3", "", "0.8"]
    assert rows[2] == ["1", "0.7", "5", "loosen", "0.6"]


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
    result = run_tuning_loop(BASE_PARAMS, predict_fn, score_fn,
                            propose_fn=default_propose_fn, max_steps=8, patience=2)
    assert result.trajectory[0].score == pytest.approx(1.0)
    assert best_step(result.trajectory).score == pytest.approx(1.0)
    # A perfect, unchanging score is its own plateau: advisor reports "good"
    # (no findings with changes) once the mask looks healthy, so the loop
    # should stop quickly rather than churning for max_steps.
    assert len(result.trajectory) <= 3
    assert result.stop_reason in ("no_more_suggestions", "plateau")
