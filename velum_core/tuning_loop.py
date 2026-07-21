"""Agentic tuning loop: predict -> score -> adjust -> repeat.

Automates the cycle a user already runs by hand from the Assistant tab —
Diagnose, Apply & re-run, Evaluate against ground truth, look, repeat.
Two interchangeable strategies decide the "adjust" step:

  * ``default_propose_fn`` — the existing rule-based diagnostic engine
    (``advisor.diagnose``). Deterministic, free, always available.
  * ``llm_propose_fn`` — a real tool-calling loop: a connected local Ollama
    model sees the score trajectory so far and decides the next change (or
    that tuning has plateaued) itself, using the same ``SUGGEST:``/chat
    machinery the Assistant's chat already speaks. Falls back to
    ``default_propose_fn`` if the model errors or returns nothing usable.

Both plug into the same ``run_tuning_loop``, which scores every round with
``benchmark.evaluate`` (``default_score_fn``) against a ground-truth mask —
"AP" (the backlog's own stopping criterion) is only meaningful with one.

Pure Python/numpy at module scope (importing this module is cheap); cross-
module deps (``advisor``/``benchmark``) are reached lazily, exactly like
every other cross-module import in this package. Threading, the actual
prediction call, and Qt all live in the caller (see
``PredictController.run_tuning_loop_async``) — this module only knows how to
run the loop given a ``predict_fn``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class TuningStep:
    """One round of the loop: the full parameter snapshot used, the delta
    that produced it, why, and how it scored."""

    step: int
    params: dict[str, Any]
    changes: dict[str, Any]
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    n_cells: int = 0
    reason: str = ""    # why `changes` was proposed (advisor finding title, or the model's own words)


@dataclass
class Proposal:
    """What a propose_fn decided for the next round."""

    changes: dict[str, Any] | None    # None/{} => nothing left to try, loop stops
    reason: str = ""


@dataclass
class TuningResult:
    """Everything a run of the loop produced."""

    trajectory: list[TuningStep]
    stop_reason: str      # one of STOP_REASONS' keys
    stop_detail: str = ""  # free text, populated when the stop *had* a reason worth keeping
                           # (e.g. "no_more_suggestions": the advisor's/model's own words for
                           # why — otherwise "" for a purely mechanical stop like max_steps)


PredictFn = Callable[[dict], "tuple[Any, np.ndarray]"]              # params -> (image, mask)
ScoreFn = Callable[[np.ndarray], "tuple[float, dict[str, float]]"]  # mask -> (score, metrics)
ProposeFn = Callable[[Any, np.ndarray, dict, "list[TuningStep]"], Proposal]  # image, mask, params, history -> Proposal


STOP_REASONS = {
    "plateau": "the score stopped improving",
    "max_steps": "reached the round budget",
    "no_more_suggestions": "nothing further to try",
    "repeated_change": "the next suggestion repeated one already tried",
    "cancelled": "stopped by the user",
    "error": "a round failed with an error",
}


def describe_stop_reason(code: str) -> str:
    return STOP_REASONS.get(code, code)


def default_score_fn(gt_mask: np.ndarray) -> ScoreFn:
    """Score a predicted mask against ``gt_mask`` with ``benchmark.evaluate``.

    The primary scalar is the mean of AP@0.5/0.75/0.9 — the same "mAP" the
    engine-benchmark table already reports — matching how
    ``docs/BACKLOG.md`` frames the stopping criterion ("until AP plateaus").
    """
    from velum_core import benchmark

    def score(mask: np.ndarray) -> tuple[float, dict[str, float]]:
        metrics = benchmark.evaluate(gt_mask, mask)
        primary = float(np.mean([metrics[f"ap@{t}"] for t in benchmark.THRESHOLDS]))
        return primary, metrics

    return score


def _merge_finding_changes(diag: dict[str, Any], params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    changes: dict[str, Any] = {}
    reasons: list[str] = []
    for f in diag["findings"]:
        if f.changes:
            changes.update(f.changes)
            reasons.append(f.title)
    changes = {k: v for k, v in changes.items() if params.get(k) != v}
    return changes, reasons


def default_propose_fn(image: Any, mask: np.ndarray, params: dict[str, Any],
                       trajectory: list[TuningStep]) -> Proposal:
    """The diagnostic engine's suggested next change, or a stop `Proposal`
    when it has nothing left to try.

    Reuses ``advisor.diagnose`` verbatim (the same engine the Assistant's
    manual "Diagnose" button already runs) so the loop's judgement is
    identical to a human clicking "Apply & re-run" themselves — merged
    across every finding, filtered to changes that would actually move a
    parameter from its current value (a no-op suggestion isn't progress and
    would otherwise loop forever).
    """
    from velum_core import advisor

    diag = advisor.diagnose(image, mask, params)
    changes, reasons = _merge_finding_changes(diag, params)
    if not changes:
        reason = diag["findings"][0].title if diag["findings"] else "No further suggestions."
        return Proposal(None, reason)
    return Proposal(changes, "; ".join(reasons))


def _first_sentence(text: str, limit: int = 160) -> str:
    """A short, UI-friendly slice of a model's (possibly long) reply."""
    text = " ".join(text.split())
    for sep in (". ", "! ", "? ", "\n"):
        i = text.find(sep)
        if 0 < i < limit:
            return text[:i + 1].strip()
    return (text[:limit] + "…") if len(text) > limit else text


def llm_propose_fn(model: str, *, on_token: Callable[[str], None] | None = None) -> ProposeFn:
    """A tool-calling propose_fn backed by a local Ollama model.

    Each round, the model sees the score trajectory so far and the
    advisor's own diagnosis (:func:`velum_core.advisor.build_tuning_prompt`),
    then either proposes one change (a ``SUGGEST:`` line, the same protocol
    the Assistant's chat already parses) or decides to stop (a ``STOP:``
    line) — a real ReAct-style reason-then-act round, not a fixed rule
    table. Falls back to :func:`default_propose_fn` whenever the model
    errors, is unreachable, or replies with nothing usable, so a flaky or
    slow local model degrades the loop instead of crashing it.
    """
    def propose(image: Any, mask: np.ndarray, params: dict[str, Any],
               trajectory: list[TuningStep]) -> Proposal:
        from velum_core import advisor

        diag = advisor.diagnose(image, mask, params)
        try:
            prompt = advisor.build_tuning_prompt(diag, params, trajectory)
            reply = advisor.ollama_chat(
                model, [{"role": "user", "content": prompt}],
                on_token or (lambda _t: None))
        except Exception as e:
            fallback = default_propose_fn(image, mask, params, trajectory)
            prefix = f"(local model unreachable — used the rule-based advisor instead: {e}) "
            return Proposal(fallback.changes, prefix + fallback.reason)

        stop_reason = advisor.parse_stop(reply)
        if stop_reason:
            return Proposal(None, stop_reason)

        changes = advisor.parse_suggestions(reply)
        changes = {k: v for k, v in changes.items() if params.get(k) != v}
        if not changes:
            fallback = default_propose_fn(image, mask, params, trajectory)
            prefix = "(model reply had no usable suggestion — used the rule-based advisor instead) "
            return Proposal(fallback.changes, prefix + fallback.reason)
        return Proposal(changes, _first_sentence(reply))

    return propose


def run_tuning_loop(
    initial_params: dict[str, Any],
    predict_fn: PredictFn,
    score_fn: ScoreFn,
    *,
    propose_fn: ProposeFn = default_propose_fn,
    max_steps: int = 8,
    patience: int = 2,
    min_delta: float = 0.005,
    on_step: Callable[[TuningStep], None] | None = None,
    on_round_start: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TuningResult:
    """Predict, score, ask ``propose_fn`` what to change, repeat.

    Stops on the first of: ``patience`` consecutive steps that fail to beat
    the best score so far by more than ``min_delta`` (``"plateau"``);
    ``max_steps`` reached (``"max_steps"``); ``propose_fn`` returning
    nothing new to try (``"no_more_suggestions"``); a change repeating one
    already tried (``"repeated_change"`` — guards an oscillation looping
    forever); or ``should_stop()`` returning true (``"cancelled"``,
    cooperative cancellation, checked once per round). The returned
    :class:`TuningResult` names exactly which — see :data:`STOP_REASONS`.

    ``on_round_start(step, max_steps)`` fires right before each round's
    (potentially slow, real-model) predict call, so a caller can show live
    progress ("round 3/8…") instead of going quiet mid-round.

    Every round is recorded — not just the winner — each with its own full
    parameter snapshot, so a caller can revert to *any* step, not only the
    best one (see :func:`best_step`). ``initial_params`` is never mutated; a
    shallow copy is the loop's running working set.
    """
    params = dict(initial_params)
    tried: set[frozenset] = set()
    trajectory: list[TuningStep] = []
    best_score = float("-inf")
    plateau_count = 0
    changes: dict[str, Any] = {}
    reason = ""
    stop_detail = ""

    for step in range(max_steps):
        if should_stop and should_stop():
            stop_reason = "cancelled"
            break

        if on_round_start:
            on_round_start(step, max_steps)
        image, mask = predict_fn(params)
        score, metrics = score_fn(mask)
        n_cells = int(mask.max()) if mask.size else 0
        record = TuningStep(step, dict(params), dict(changes), score, metrics, n_cells, reason)
        trajectory.append(record)
        if on_step:
            on_step(record)

        if score > best_score + min_delta:
            best_score = score
            plateau_count = 0
        else:
            plateau_count += 1
        if plateau_count >= patience:
            stop_reason = "plateau"
            break

        proposal = propose_fn(image, mask, params, trajectory)
        changes = proposal.changes or {}
        reason = proposal.reason
        if not changes:
            stop_reason = "no_more_suggestions"
            stop_detail = proposal.reason
            break
        signature = frozenset(changes.items())
        if signature in tried:
            stop_reason = "repeated_change"
            break
        tried.add(signature)
        params.update(changes)
    else:
        stop_reason = "max_steps"

    return TuningResult(trajectory, stop_reason, stop_detail)


def best_step(trajectory: list[TuningStep]) -> TuningStep | None:
    """The highest-scoring round, or ``None`` for an empty trajectory."""
    return max(trajectory, key=lambda s: s.score) if trajectory else None


def parameter_importance(trajectory: list[TuningStep]) -> list[tuple[str, float]]:
    """Rank tunable parameters by how strongly their value correlated with
    the score across the trajectory (Pearson correlation coefficient) — the
    same idea Weights & Biases' sweep "parameter importance" panel shows,
    computed here with plain numpy instead of a random-forest model since a
    handful of rounds is nowhere near enough data for one.

    Returns only numeric parameters that actually varied across the
    trajectory, sorted by |correlation| descending; empty if fewer than 3
    rounds ran or nothing varied (both make a correlation meaningless).
    """
    if len(trajectory) < 3:
        return []
    scores = np.array([s.score for s in trajectory], dtype=float)
    if np.std(scores) == 0:
        return []
    keys: set[str] = set()
    for s in trajectory:
        keys.update(s.params.keys())

    out: list[tuple[str, float]] = []
    for k in sorted(keys):
        vals: list[float] = []
        for s in trajectory:
            v = s.params.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                vals = []
                break
            vals.append(float(v))
        if len(vals) != len(trajectory):
            continue
        arr = np.array(vals)
        if np.allclose(arr, arr[0]) or np.std(arr) == 0:
            continue
        corr = float(np.corrcoef(arr, scores)[0, 1])
        if np.isnan(corr):
            continue
        out.append((k, corr))

    out.sort(key=lambda kv: -abs(kv[1]))
    return out


def write_trajectory_csv(path, trajectory: list[TuningStep]) -> None:
    """Export the full trajectory (every round's params + score) as a CSV —
    the same "raw data export" every sweep/AutoML dashboard offers next to
    its charts (Optuna, W&B)."""
    import csv

    cols_seen: list[str] = []
    seen = set()
    for s in trajectory:
        for k in s.params:
            if k not in seen:
                seen.add(k)
                cols_seen.append(k)
    cols = ["step", "score", "n_cells", "reason"] + cols_seen
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in trajectory:
            w.writerow([s.step, round(s.score, 4), s.n_cells, s.reason]
                      + [s.params.get(k, "") for k in cols_seen])
