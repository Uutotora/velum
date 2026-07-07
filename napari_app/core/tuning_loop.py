"""Agentic tuning loop: predict -> score -> adjust -> repeat.

Automates the cycle a user already runs by hand from the Assistant tab —
Diagnose, Apply & re-run, Evaluate against ground truth, look, repeat — by
wiring the same two pieces (``advisor.diagnose``'s rule-based suggestions and
``benchmark.evaluate``'s instance AP/F1) into a closed loop that stops once
the score plateaus instead of waiting for a human to click each round.

Pure Python/numpy at module scope (importing this module is cheap); the two
default callables below reach into ``advisor``/``benchmark`` lazily, exactly
like every other cross-module import in this package. Threading, the actual
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
    that produced it, and how it scored."""

    step: int
    params: dict[str, Any]
    changes: dict[str, Any]
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    n_cells: int = 0


PredictFn = Callable[[dict], "tuple[Any, np.ndarray]"]           # params -> (image, mask)
ScoreFn = Callable[[np.ndarray], "tuple[float, dict[str, float]]"]  # mask -> (score, metrics)
ProposeFn = Callable[[Any, np.ndarray, dict], "dict[str, Any] | None"]  # image, mask, params -> changes


def default_score_fn(gt_mask: np.ndarray) -> ScoreFn:
    """Score a predicted mask against ``gt_mask`` with ``benchmark.evaluate``.

    The primary scalar is the mean of AP@0.5/0.75/0.9 — the same "mAP" the
    engine-benchmark table already reports — matching how
    ``docs/BACKLOG.md`` frames the stopping criterion ("until AP plateaus").
    """
    from napari_app import benchmark

    def score(mask: np.ndarray) -> tuple[float, dict[str, float]]:
        metrics = benchmark.evaluate(gt_mask, mask)
        primary = float(np.mean([metrics[f"ap@{t}"] for t in benchmark.THRESHOLDS]))
        return primary, metrics

    return score


def default_propose_fn(image: Any, mask: np.ndarray, params: dict[str, Any]) -> dict[str, Any] | None:
    """The diagnostic engine's suggested next change, or ``None`` when it has
    nothing left to try.

    Reuses ``advisor.diagnose`` verbatim (the same engine the Assistant's
    manual "Diagnose" button already runs) so the loop's judgement is
    identical to a human clicking "Apply & re-run" themselves — merged
    across every finding, filtered to changes that would actually move a
    parameter from its current value (a no-op suggestion isn't progress and
    would otherwise loop forever).
    """
    from napari_app import advisor

    diag = advisor.diagnose(image, mask, params)
    changes: dict[str, Any] = {}
    for f in diag["findings"]:
        changes.update(f.changes)
    changes = {k: v for k, v in changes.items() if params.get(k) != v}
    return changes or None


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
    should_stop: Callable[[], bool] | None = None,
) -> list[TuningStep]:
    """Predict, score, ask ``propose_fn`` what to change, repeat.

    Stops on the first of: ``patience`` consecutive steps that fail to beat
    the best score so far by more than ``min_delta`` (a plateau); ``max_steps``
    reached; ``propose_fn`` returning nothing new to try; a change repeating
    one already tried (guards against a two-step oscillation looping
    forever); or ``should_stop()`` returning true (cooperative cancellation,
    checked once per round).

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

    for step in range(max_steps):
        if should_stop and should_stop():
            break

        image, mask = predict_fn(params)
        score, metrics = score_fn(mask)
        n_cells = int(mask.max()) if mask.size else 0
        record = TuningStep(step, dict(params), dict(changes), score, metrics, n_cells)
        trajectory.append(record)
        if on_step:
            on_step(record)

        if score > best_score + min_delta:
            best_score = score
            plateau_count = 0
        else:
            plateau_count += 1
        if plateau_count >= patience:
            break

        changes = propose_fn(image, mask, params) or {}
        if not changes:
            break
        signature = frozenset(changes.items())
        if signature in tried:
            break
        tried.add(signature)
        params.update(changes)

    return trajectory


def best_step(trajectory: list[TuningStep]) -> TuningStep | None:
    """The highest-scoring round, or ``None`` for an empty trajectory."""
    return max(trajectory, key=lambda s: s.score) if trajectory else None
