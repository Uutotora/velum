"""Unit tests for velum_core.benchmark (instance-level scoring vs GT)."""
import numpy as np

from velum_core.benchmark import evaluate, results_table, summarize


def _two_cells(offset=0):
    """A 60x60 label mask with two 10px squares; ``offset`` shifts them."""
    m = np.zeros((60, 60), dtype=np.int32)
    m[5 + offset:15 + offset, 5 + offset:15 + offset] = 1
    m[40:50, 40:50] = 2
    return m


def test_perfect_prediction_scores_one():
    gt = _two_cells()
    res = evaluate(gt, gt.copy())
    assert res["gt_cells"] == 2 and res["pred_cells"] == 2
    assert res["f1"] == 1.0
    assert res["ap@0.5"] == 1.0
    assert res["tp"] == 2 and res["fp"] == 0 and res["fn"] == 0


def test_empty_prediction_scores_zero():
    gt = _two_cells()
    pred = np.zeros_like(gt)
    res = evaluate(gt, pred)
    assert res["f1"] == 0.0
    assert res["ap@0.5"] == 0.0
    assert res["fn"] == 2 and res["tp"] == 0


def test_partial_overlap_below_threshold_is_false():
    gt = _two_cells()
    # Shift one cell far enough that IoU with GT drops below 0.5 → miss.
    pred = _two_cells(offset=7)
    res = evaluate(gt, pred)
    assert 0.0 <= res["f1"] <= 1.0
    # at least one true cell is not matched at IoU 0.5
    assert res["fn"] >= 1


def test_summarize_means_across_images():
    per_image = [
        {"f1": 1.0, "ap@0.5": 1.0, "ap@0.75": 0.8, "ap@0.9": 0.6},
        {"f1": 0.0, "ap@0.5": 0.0, "ap@0.75": 0.0, "ap@0.9": 0.0},
    ]
    s = summarize(per_image)
    assert s["n_images"] == 2
    assert s["f1"] == 0.5
    assert s["ap@0.5"] == 0.5
    assert s["mAP"] == (0.5 + 0.4 + 0.3) / 3


def test_summarize_empty_is_empty():
    assert summarize([]) == {}


def test_results_table_shape():
    summaries = {
        "CellSeg1": {"n_images": 3, "f1": 0.9, "ap@0.5": 0.9,
                     "ap@0.75": 0.7, "ap@0.9": 0.5, "mAP": 0.7},
        "Cellpose": {"n_images": 3, "f1": 0.8, "ap@0.5": 0.8,
                     "ap@0.75": 0.6, "ap@0.9": 0.4, "mAP": 0.6},
    }
    cols, rows = results_table(summaries)
    assert cols[0] == "engine" and "mAP" in cols
    assert len(rows) == 2
    assert rows[0][0] == "CellSeg1"
    # every row has one cell per column
    assert all(len(r) == len(cols) for r in rows)
