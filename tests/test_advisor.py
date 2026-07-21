"""Unit tests for velum_core.advisor (the deterministic diagnostic engine).

The LLM/Ollama bridge is intentionally not exercised here (it depends on an
external service). We pin the offline diagnostic engine, which is what makes
the Assistant useful even with no model running.
"""
import numpy as np

from velum_core.advisor import (
    diagnose,
    findings_to_text,
    image_stats,
    mask_stats,
    parse_suggestions,
)

BASE_PARAMS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.6,
    "box_nms_thresh": 0.05,
    "min_mask_area": 20,
    "resize_size": 512,
}


def _grid_mask(n, side=14, gap=6, canvas=500):
    """A canvas with ``n`` well-separated uniform square cells."""
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


def test_image_stats_mean_and_contrast():
    img = np.full((10, 10), 128, dtype=np.uint8)
    st = image_stats(img)
    assert st["mean"] == 128.0
    assert st["contrast"] == 0.0            # uniform image → no dynamic range
    assert st["h"] == 10 and st["w"] == 10


def test_mask_stats_counts_and_coverage():
    m = _grid_mask(10)
    st = mask_stats(m)
    assert st["n_cells"] == 10
    assert 0.0 < st["coverage"] < 1.0
    assert st["median_area"] > 0


def test_diagnose_without_mask_prompts_to_predict():
    diag = diagnose(np.zeros((50, 50), dtype=np.uint8), None, BASE_PARAMS)
    titles = [f.title for f in diag["findings"]]
    assert any("Run a prediction" in t for t in titles)


def test_diagnose_no_cells_suggests_loosening():
    img = np.full((100, 100), 128, dtype=np.uint8)
    empty = np.zeros((100, 100), dtype=np.int32)
    diag = diagnose(img, empty, BASE_PARAMS)
    warn = next(f for f in diag["findings"] if f.severity == "warn")
    assert "No cells" in warn.title
    # thresholds should be loosened relative to the current values
    assert warn.changes["pred_iou_thresh"] < BASE_PARAMS["pred_iou_thresh"]
    assert warn.changes["stability_score_thresh"] < BASE_PARAMS["stability_score_thresh"]


def test_diagnose_healthy_mask_reports_good():
    img = np.random.RandomState(0).randint(0, 255, (500, 500), dtype=np.uint8)
    m = _grid_mask(20, side=16)             # uniform, no fragments, count >= 15
    diag = diagnose(img, m, BASE_PARAMS)
    assert diag["findings"][0].severity == "good"
    assert not any(f.severity == "warn" for f in diag["findings"])


def test_diagnose_detects_over_segmentation():
    # 5 real cells + 12 tiny fragments → median area collapses, triggers warn
    m = np.zeros((300, 300), dtype=np.int32)
    idx = 1
    for i in range(5):
        m[10 + i * 30:30 + i * 30, 10:30] = idx  # 20x20 = 400 px
        idx += 1
    for i in range(12):
        m[200 + i * 5:202 + i * 5, 200:202] = idx  # 2x2 = 4 px fragments
        idx += 1
    diag = diagnose(np.zeros((300, 300), np.uint8), m, BASE_PARAMS)
    assert any("over-segmentation" in f.title.lower() for f in diag["findings"])


def test_parse_suggestions_filters_and_types():
    text = (
        "You should tune the grid.\n"
        "SUGGEST: points_per_side=48\n"
        "SUGGEST: pred_iou_thresh=0.65\n"
        "SUGGEST: not_a_real_param=5\n"     # must be dropped
    )
    out = parse_suggestions(text)
    assert out["points_per_side"] == 48
    assert isinstance(out["points_per_side"], int)
    assert out["pred_iou_thresh"] == 0.65
    assert isinstance(out["pred_iou_thresh"], float)
    assert "not_a_real_param" not in out


def test_findings_to_text_includes_suggested_changes():
    img = np.full((100, 100), 128, dtype=np.uint8)
    diag = diagnose(img, np.zeros((100, 100), np.int32), BASE_PARAMS)
    text = findings_to_text(diag)
    assert "suggested:" in text
    assert "pred_iou_thresh" in text
