"""Wiring test: the Assistant's Auto-tune button drives the agentic tuning
loop and renders its trajectory as steps stream in from a background thread.

AssistantWidget only ever talks to its ``predict`` collaborator through a
small public API (last_context/current_params/apply_params/rerun/
has_ground_truth/start_auto_tune/stop_auto_tune/restore_tuning_step), so this
stubs a fake predict-widget instead of building a real PredictWidget — no
image, model or GT file needed, matching the boundary
PredictController.run_tuning_loop_async and PredictWidget.start_auto_tune
already own (see tests/test_predict_controller.py for those, tested against
a real fake-engine prediction). ``on_step``/``on_finish`` are called directly
here (not from a real thread), so the Qt signal they route through fires
synchronously — no event-loop pump required, though ``app.processEvents()``
is still used where a layout update is being asserted on.

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
        self.started_with = None
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

    def start_auto_tune(self, on_step, on_finish):
        if self.start_error:
            return self.start_error
        self.started_with = (on_step, on_finish)
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
    yield w
    w.close()


def test_autotune_reports_precondition_error_without_starting(widget):
    widget.predict_fake.start_error = "Run a prediction first."
    widget._toggle_autotune()
    assert widget.predict_fake.started_with is None
    assert widget._autotune_active is False


def test_autotune_button_starts_the_loop(widget):
    widget._toggle_autotune()
    assert widget.predict_fake.started_with is not None
    assert widget._autotune_active is True


def test_second_click_while_active_stops_instead_of_restarting(widget):
    widget._toggle_autotune()
    first_callbacks = widget.predict_fake.started_with
    widget._toggle_autotune()
    assert widget.predict_fake.stopped is True
    assert widget.predict_fake.started_with is first_callbacks  # no second start


def test_steps_render_as_cards_and_track_the_best_score(widget, app):
    widget._toggle_autotune()
    on_step, _on_finish = widget.predict_fake.started_with

    on_step(TuningStep(0, {"a": 1}, {}, 0.5, {}, 3))
    on_step(TuningStep(1, {"a": 2}, {"a": 2}, 0.8, {}, 4))
    on_step(TuningStep(2, {"a": 3}, {"a": 3}, 0.7, {}, 5))
    app.processEvents()

    cards = widget._chat.findChildren(aw.AutoTuneStepCard)
    assert len(cards) == 3
    assert widget._autotune_best.step == 1
    assert widget._autotune_best.score == 0.8


def test_finish_resets_active_state_and_reports_best(widget, app):
    widget._toggle_autotune()
    on_step, on_finish = widget.predict_fake.started_with
    on_step(TuningStep(0, {"a": 1}, {}, 0.62, {}, 3))
    on_finish()
    app.processEvents()
    assert widget._autotune_active is False


def test_finish_with_no_steps_does_not_crash(widget, app):
    widget._toggle_autotune()
    _on_step, on_finish = widget.predict_fake.started_with
    on_finish()
    app.processEvents()
    assert widget._autotune_active is False


def test_restore_step_applies_full_snapshot_and_reruns(widget):
    widget._restore_autotune_step({"pred_iou_thresh": 0.5, "resize_size": 1024})
    assert widget.predict_fake.applied == [{"pred_iou_thresh": 0.5, "resize_size": 1024}]
    assert widget.predict_fake.reran is True
