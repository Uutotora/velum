"""Wiring test: the Assistant's Auto-tune card drives the agentic tuning
loop and renders its trajectory (chart + sortable table + parameter
importance) as rounds stream in from a background thread.

AssistantWidget only ever talks to its ``predict`` collaborator through a
small public API (last_context/current_params/apply_params/rerun/
has_ground_truth/start_auto_tune/stop_auto_tune/restore_tuning_step), so this
stubs a fake predict-widget instead of building a real PredictWidget — no
image, model or GT file needed, matching the boundary
PredictController.run_tuning_loop_async and PredictWidget.start_auto_tune
already own (see tests/test_predict_controller.py for those, tested against
a real fake-engine prediction, and tests/test_predict_widget_autotune_wiring.py
for PredictWidget's own glue). ``on_step``/``on_round_start``/``on_finish``
are called directly here (not from a real thread), so the Qt signal each
routes through fires synchronously — no event-loop pump required, though
``app.processEvents()`` is still used where a layout update is being
asserted on.

Skipped in the lightweight CI image (no PyQt6).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
aw = pytest.importorskip("napari_app.widgets.assistant_widget")

from PyQt6.QtWidgets import QApplication

from napari_app.core.tuning_loop import TuningStep


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _FakePredict:
    """Minimal stand-in for PredictWidget's Assistant-facing API."""

    def __init__(self):
        self.applied = []
        self.reran = False
        self.start_error = None
        self.started_with = None       # (on_step, on_round_start, on_finish)
        self.start_kwargs = None        # everything else start_auto_tune was called with
        self.stopped = False

    def last_context(self):
        return None, None

    def current_params(self):
        return {}

    def apply_params(self, changes):
        self.applied.append(dict(changes))
        return [f"{k} → {v}" for k, v in changes.items()]

    def rerun(self):
        self.reran = True

    def has_ground_truth(self):
        return True

    def start_auto_tune(self, on_step, on_finish, *, on_round_start=None, **kwargs):
        if self.start_error:
            return self.start_error
        self.started_with = (on_step, on_round_start, on_finish)
        self.start_kwargs = kwargs
        return None

    def stop_auto_tune(self):
        self.stopped = True

    def restore_tuning_step(self, params):
        self.applied.append(dict(params))
        self.reran = True


@pytest.fixture
def widget(app):
    predict = _FakePredict()
    w = aw.AssistantWidget(viewer=None, predict_widget=predict)
    w.predict_fake = predict
    # Qt's isVisible() is composite (considers ancestors), so an unshown
    # top-level widget reports every child as invisible regardless of its
    # own setVisible() state — needed for the chart/table/importance
    # visibility assertions below (same issue already documented in
    # tests/test_predict_labels_display_wiring.py).
    w.show()
    app.processEvents()
    yield w
    w.close()


# ── starting / stopping ──────────────────────────────────────────────────────

def test_autotune_reports_precondition_error_without_starting(widget):
    widget.predict_fake.start_error = "Run a prediction first."
    widget._toggle_autotune()
    assert widget.predict_fake.started_with is None
    assert widget._autotune_active is False
    assert "Run a prediction" in widget._autotune_status.text()


def test_autotune_button_starts_the_loop_with_the_ui_settings(widget):
    widget._max_steps_sb.setValue(5)
    widget._patience_sb.setValue(3)
    widget._min_delta_sb.setValue(0.02)
    widget._toggle_autotune()

    assert widget.predict_fake.started_with is not None
    assert widget._autotune_active is True
    kw = widget.predict_fake.start_kwargs
    assert kw["strategy"] == "advisor"     # default combo index 0
    assert kw["model"] is None
    assert kw["max_steps"] == 5
    assert kw["patience"] == 3
    assert kw["min_delta"] == pytest.approx(0.02)
    assert widget._autotune_btn.text() == "Stop auto-tune"


def test_llm_strategy_without_a_connected_model_refuses_to_start(widget):
    widget._strategy_combo.setCurrentIndex(1)   # "Local model"
    widget._toggle_autotune()
    assert widget.predict_fake.started_with is None
    assert widget._autotune_active is False
    assert "no local model" in widget._autotune_status.text().lower()


def test_llm_strategy_with_a_connected_model_passes_it_through(widget):
    widget._model_combo.clear()
    widget._model_combo.addItem("qwen2.5:7b")
    widget._model_combo.setEnabled(True)   # mirrors _refresh_models() when Ollama is reachable
    widget._strategy_combo.setCurrentIndex(1)
    widget._toggle_autotune()
    assert widget.predict_fake.started_with is not None
    assert widget.predict_fake.start_kwargs["strategy"] == "llm"
    assert widget.predict_fake.start_kwargs["model"] == "qwen2.5:7b"


def test_second_click_while_active_stops_instead_of_restarting(widget):
    widget._toggle_autotune()
    first_callbacks = widget.predict_fake.started_with
    widget._toggle_autotune()
    assert widget.predict_fake.stopped is True
    assert widget.predict_fake.started_with is first_callbacks   # no second start


def test_round_start_updates_the_status_label(widget):
    widget._toggle_autotune()
    _on_step, on_round_start, _on_finish = widget.predict_fake.started_with
    on_round_start(2, 8)
    assert "3/8" in widget._autotune_status.text()


# ── streaming steps into the chart/table ─────────────────────────────────────

def test_steps_populate_the_table_and_chart_and_track_the_best_score(widget, app):
    widget._toggle_autotune()
    on_step, _on_round_start, _on_finish = widget.predict_fake.started_with

    on_step(TuningStep(0, {"a": 1}, {}, 0.5, n_cells=3))
    on_step(TuningStep(1, {"a": 2}, {"a": 2}, 0.8, n_cells=4, reason="cells may be merged"))
    on_step(TuningStep(2, {"a": 3}, {"a": 3}, 0.7, n_cells=5))
    app.processEvents()

    assert widget._tune_table.rowCount() == 3
    assert widget._tune_table.isVisible()
    assert widget._tune_chart.isVisible()
    assert widget._autotune_best.step == 1
    assert widget._autotune_best.score == 0.8
    assert widget._export_tune_btn.isEnabled()
    # Round 1's reason made it into the table's "Reason" column.
    assert widget._tune_table.item(1, 4).text() == "cells may be merged"


def test_selecting_a_row_enables_use_button_and_restores_that_rounds_params(widget, app):
    widget._toggle_autotune()
    on_step, _rs, _f = widget.predict_fake.started_with
    on_step(TuningStep(0, {"a": 1}, {}, 0.5, n_cells=3))
    on_step(TuningStep(1, {"a": 2}, {"a": 2}, 0.8, n_cells=4))
    app.processEvents()

    assert widget._use_round_btn.isEnabled() is False
    widget._tune_table.selectRow(0)
    app.processEvents()
    assert widget._use_round_btn.isEnabled() is True

    widget._use_selected_tune_row()
    assert widget.predict_fake.applied == [{"a": 1}]
    assert widget.predict_fake.reran is True


def test_export_csv_writes_the_trajectory(widget, app, tmp_path, monkeypatch):
    out = tmp_path / "trajectory.csv"
    monkeypatch.setattr(aw.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))

    widget._toggle_autotune()
    on_step, _rs, _f = widget.predict_fake.started_with
    on_step(TuningStep(0, {"a": 1}, {}, 0.5, n_cells=3))
    on_step(TuningStep(1, {"a": 2}, {"a": 2}, 0.8, n_cells=4))

    widget._export_tune_csv()

    assert out.exists()
    text = out.read_text()
    assert "step,score,n_cells,reason,a" in text.splitlines()[0]
    assert out.name in widget._autotune_status.text()


def test_export_csv_with_no_rounds_does_nothing(widget, monkeypatch):
    calls = []
    monkeypatch.setattr(aw.QFileDialog, "getSaveFileName",
                        lambda *a, **k: calls.append(1) or ("", ""))
    widget._export_tune_csv()
    assert calls == []   # never even opened the dialog


# ── finishing ─────────────────────────────────────────────────────────────────

def test_finish_resets_active_state_and_reports_best_plus_importance(widget, app):
    widget._toggle_autotune()
    on_step, _rs, on_finish = widget.predict_fake.started_with
    on_step(TuningStep(0, {"box_nms_thresh": 0.05}, {}, 0.5, n_cells=3))
    on_step(TuningStep(1, {"box_nms_thresh": 0.03}, {"box_nms_thresh": 0.03}, 0.7, n_cells=4))
    on_step(TuningStep(2, {"box_nms_thresh": 0.01}, {"box_nms_thresh": 0.01}, 0.9, n_cells=5))
    on_finish("plateau", "")
    app.processEvents()

    assert widget._autotune_active is False
    assert widget._autotune_btn.text() == "Start auto-tune"
    assert "round 2" in widget._autotune_status.text().lower()
    assert widget._tune_importance.isVisible()   # 3 rounds, a perfectly-correlated param
    assert "box_nms_thresh" in widget._tune_importance.text()


def test_finish_includes_the_stop_detail_when_present(widget, app):
    widget._toggle_autotune()
    on_step, _rs, on_finish = widget.predict_fake.started_with
    on_step(TuningStep(0, {"a": 1}, {}, 0.62, n_cells=3))
    on_finish("no_more_suggestions", "looks fine already.")
    app.processEvents()
    assert "looks fine already." in widget._autotune_status.text()


def test_finish_with_no_steps_does_not_crash(widget, app):
    widget._toggle_autotune()
    _on_step, _rs, on_finish = widget.predict_fake.started_with
    on_finish("error", "")
    app.processEvents()
    assert widget._autotune_active is False
    assert "no rounds completed" in widget._autotune_status.text().lower()


# ── restore (direct handler, mirrors the "Use selected round" button) ───────

def test_restore_step_applies_full_snapshot_and_reruns(widget):
    widget._tune_rows = [TuningStep(0, {"pred_iou_thresh": 0.5, "resize_size": 1024}, {}, 0.6)]
    widget._tune_table.setRowCount(1)
    widget._fill_tune_row(0, widget._tune_rows[0], None)
    widget._tune_table.selectRow(0)
    widget._use_selected_tune_row()
    assert widget.predict_fake.applied == [{"pred_iou_thresh": 0.5, "resize_size": 1024}]
    assert widget.predict_fake.reran is True
