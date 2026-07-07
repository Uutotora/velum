"""Wiring test: PredictWidget's Assistant-facing auto-tune hooks —
has_ground_truth / start_auto_tune / stop_auto_tune / restore_tuning_step.

Focuses on PredictWidget's own glue (precondition checks, loading/resizing
the ground-truth mask, gathering params, dispatching to
PredictController.run_tuning_loop_async) — the controller call itself is
monkeypatched out so this doesn't re-exercise the loop (see
tests/test_predict_controller.py, already covered end-to-end with a real
fake-engine prediction) or the Assistant's rendering of it (see
tests/test_assistant_widget_wiring.py).

Skipped in the lightweight CI image (no PyQt6).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pw = pytest.importorskip("napari_app.widgets.predict_widget")

from PyQt6.QtWidgets import QApplication


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widget(app, monkeypatch):
    viewer = MagicMock()
    w = pw.PredictWidget(viewer)
    monkeypatch.setattr(w, "_append_log", lambda *a, **k: None)
    yield w
    w.close()


# ── has_ground_truth ─────────────────────────────────────────────────────────

def test_has_ground_truth_false_with_no_path_set(widget):
    assert widget.has_ground_truth() is False


def test_has_ground_truth_false_when_file_missing(widget):
    widget.gt_path.setText("/nonexistent/gt.png")
    assert widget.has_ground_truth() is False


def test_has_ground_truth_true_when_file_exists(widget, tmp_path):
    p = tmp_path / "gt.png"
    p.write_bytes(b"x")
    widget.gt_path.setText(str(p))
    assert widget.has_ground_truth() is True


# ── start_auto_tune preconditions ────────────────────────────────────────────

def test_start_auto_tune_requires_a_prediction_first(widget, tmp_path):
    p = tmp_path / "gt.png"; p.write_bytes(b"x")
    widget.gt_path.setText(str(p))
    err = widget.start_auto_tune(on_step=lambda s: None, on_finish=lambda: None)
    assert err == "Run a prediction first."


def test_start_auto_tune_requires_ground_truth(widget):
    widget._last_mask = np.ones((10, 10), dtype=np.int32)
    err = widget.start_auto_tune(on_step=lambda s: None, on_finish=lambda: None)
    assert err is not None and "ground" in err.lower()


def test_start_auto_tune_reports_unreadable_gt(widget, tmp_path):
    widget._last_mask = np.ones((10, 10), dtype=np.int32)
    p = tmp_path / "gt.png"; p.write_bytes(b"not a real image")
    widget.gt_path.setText(str(p))
    err = widget.start_auto_tune(on_step=lambda s: None, on_finish=lambda: None)
    assert err is not None


# ── start_auto_tune dispatch ──────────────────────────────────────────────────

def test_start_auto_tune_dispatches_with_gathered_params(widget, monkeypatch, tmp_path):
    import cv2
    widget._last_mask = np.zeros((10, 10), dtype=np.int32)
    widget._last_mask[2:5, 2:5] = 1
    gt = np.zeros((10, 10), dtype=np.uint16)
    gt[2:5, 2:5] = 1
    gt_path = tmp_path / "gt.png"
    cv2.imwrite(str(gt_path), gt)
    widget.gt_path.setText(str(gt_path))
    img_path = tmp_path / "img.png"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))
    widget.image_path.setText(str(img_path))

    calls = []
    monkeypatch.setattr(widget._controller, "run_tuning_loop_async",
                        lambda params, gtm, **kw: calls.append((params, gtm, kw)))
    on_step, on_finish = (lambda s: None), (lambda: None)
    err = widget.start_auto_tune(on_step, on_finish)

    assert err is None
    assert len(calls) == 1
    params, gtm, kw = calls[0]
    assert params["image_path"] == str(img_path)
    assert np.array_equal(gtm, gt.astype(np.int32))
    assert kw["on_step"] is on_step and kw["on_finish"] is on_finish


def test_start_auto_tune_resizes_gt_to_match_prediction_shape(widget, monkeypatch, tmp_path):
    import cv2
    widget._last_mask = np.zeros((20, 20), dtype=np.int32)   # prediction shape 20x20
    gt = np.zeros((10, 10), dtype=np.uint16)                  # GT shape 10x10 -> mismatch
    gt[2:5, 2:5] = 1
    gt_path = tmp_path / "gt.png"
    cv2.imwrite(str(gt_path), gt)
    widget.gt_path.setText(str(gt_path))
    img_path = tmp_path / "img.png"
    cv2.imwrite(str(img_path), np.zeros((20, 20, 3), dtype=np.uint8))
    widget.image_path.setText(str(img_path))

    calls = []
    monkeypatch.setattr(widget._controller, "run_tuning_loop_async",
                        lambda params, gtm, **kw: calls.append(gtm))
    widget.start_auto_tune(lambda s: None, lambda: None)
    assert calls[0].shape == (20, 20)


# ── stop_auto_tune / restore_tuning_step ─────────────────────────────────────

def test_stop_auto_tune_forwards_to_controller(widget, monkeypatch):
    calls = []
    monkeypatch.setattr(widget._controller, "stop_tuning", lambda: calls.append(True))
    widget.stop_auto_tune()
    assert calls == [True]


def test_restore_tuning_step_applies_params_and_reruns(widget, monkeypatch):
    applied = []
    reran = []
    monkeypatch.setattr(widget, "apply_params", lambda p: applied.append(p))
    monkeypatch.setattr(widget, "rerun", lambda: reran.append(True))
    widget.restore_tuning_step({"pred_iou_thresh": 0.5})
    assert applied == [{"pred_iou_thresh": 0.5}]
    assert reran == [True]
