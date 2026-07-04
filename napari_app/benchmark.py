"""
Engine benchmarking against ground truth.

Runs one or more segmentation engines over a folder of images that have
matching ground-truth masks, scores each prediction with instance-level F1 and
Average Precision (Cellpose/StarDist-style, TP/(TP+FP+FN)) at several IoU
thresholds, and aggregates per engine so you can objectively pick the best
method for *your* data.

Only the scoring/aggregation is here (pure and testable). Running the actual
engines is orchestrated by the widget, which owns the prediction pipeline.
"""
from __future__ import annotations

from typing import Any

import numpy as np

THRESHOLDS = (0.5, 0.75, 0.9)


def evaluate(gt: np.ndarray, pred: np.ndarray,
             thresholds=THRESHOLDS) -> dict[str, float]:
    """Instance metrics for one prediction vs ground truth."""
    from metrics import average_precision

    gt = np.ascontiguousarray(gt).astype(np.int32)
    pred = np.ascontiguousarray(pred).astype(np.int32)
    out: dict[str, float] = {"gt_cells": int(gt.max()), "pred_cells": int(pred.max())}
    for th in thresholds:
        # one threshold per call — average_precision's guard is not vectorised
        ap, tp, fp, fn = average_precision(gt, pred, threshold=[th])
        ap = float(np.atleast_1d(ap)[0])
        tp = float(np.atleast_1d(tp)[0]); fp = float(np.atleast_1d(fp)[0])
        fn = float(np.atleast_1d(fn)[0])
        out[f"ap@{th}"] = ap
        if th == thresholds[0]:
            denom = 2 * tp + fp + fn
            out["f1"] = (2 * tp / denom) if denom else 0.0
            out["tp"], out["fp"], out["fn"] = int(tp), int(fp), int(fn)
    return out


def summarize(per_image: list[dict[str, float]],
              thresholds=THRESHOLDS) -> dict[str, float]:
    """Mean metrics over a set of per-image results for one engine."""
    if not per_image:
        return {}
    def _mean(key):
        vals = [d[key] for d in per_image if key in d]
        return float(np.mean(vals)) if vals else 0.0
    summary = {"n_images": len(per_image), "f1": _mean("f1")}
    for th in thresholds:
        summary[f"ap@{th}"] = _mean(f"ap@{th}")
    summary["mAP"] = float(np.mean([summary[f"ap@{th}"] for th in thresholds]))
    return summary


def results_table(engine_summaries: dict[str, dict[str, float]],
                  thresholds=THRESHOLDS) -> tuple[list[str], list[list]]:
    """Build a comparison table: one row per engine, best score highlighted upstream."""
    cols = ["engine", "images", "F1@0.5"] + [f"AP@{t}" for t in thresholds] + ["mAP"]
    rows: list[list] = []
    for name, s in engine_summaries.items():
        rows.append([
            name, s.get("n_images", 0),
            round(s.get("f1", 0.0), 3),
            *[round(s.get(f"ap@{t}", 0.0), 3) for t in thresholds],
            round(s.get("mAP", 0.0), 3),
        ])
    return cols, rows


def write_csv(path, cols, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
