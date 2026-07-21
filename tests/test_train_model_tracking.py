"""Unit tests for train_model.py's optional experiment-tracking hooks.

train_model() itself has no prior test coverage (it pulls in real torch
training machinery via cellseg1_train, not designed for easy mocking) — this
file doesn't attempt full training-loop coverage, only the new tracking
wiring, by monkeypatching every external call (cellseg1_train's functions,
set_environment.set_env, train_model.load_dataset, and
experiment_tracking.start_run itself) so the "loop" runs in milliseconds
with a scripted loss sequence.

Skipped in the lightweight CI image (no torch — cellseg1_train needs it at
import time).
"""
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
import cellseg1_train  # noqa: E402
import set_environment  # noqa: E402

from velum_core import train_model  # noqa: E402


def _fake_state_manager():
    sm = MagicMock()
    sm.check_stop_flag.return_value = False
    return sm


def _base_config(tmp_path, epoch_max=3):
    return {
        "epoch_max": epoch_max,
        "result_pth_path": str(tmp_path / "out.pth"),
        "selected_device": "cpu",
        "deterministic": True, "seed": 0,
        "allow_tf32_on_cudnn": True, "allow_tf32_on_matmul": True,
        "vit_name": "vit_h", "image_encoder_lora_rank": 4,
        "finetune_from": None,
    }


@pytest.fixture
def mocked_training(monkeypatch, tmp_path):
    """Stub every heavy dependency train_model() touches so it runs fast and
    deterministically, driven by a scripted per-epoch loss sequence."""
    losses = iter([0.5, 0.3, 0.2, 0.1, 0.05])
    monkeypatch.setattr(cellseg1_train, "prepare_directories", lambda config: None)
    monkeypatch.setattr(cellseg1_train, "load_model", lambda config: MagicMock())
    monkeypatch.setattr(cellseg1_train, "setup_training",
                        lambda config, model, ds: (MagicMock(), MagicMock(), MagicMock()))
    monkeypatch.setattr(cellseg1_train, "train_epoch",
                        lambda model, config, tl, opt, sch: next(losses))
    monkeypatch.setattr(cellseg1_train, "save_model_pth", lambda model, path: None)
    monkeypatch.setattr(set_environment, "set_env", lambda *a, **k: None)
    monkeypatch.setattr(train_model, "load_dataset", lambda config: MagicMock())
    monkeypatch.setattr(train_model, "_save_config_sidecar", lambda config, history: None)
    return losses


def test_train_model_logs_one_tracked_value_per_epoch(mocked_training, monkeypatch, tmp_path):
    tracked = MagicMock()
    starts = []

    def fake_start_run(experiment, hparams):
        starts.append((experiment, hparams))
        return tracked

    from velum_core import experiment_tracking
    monkeypatch.setattr(experiment_tracking, "start_run", fake_start_run)

    config = _base_config(tmp_path, epoch_max=3)
    train_model.train_model(config, _fake_state_manager())

    assert starts and starts[0][0] == "train"
    assert starts[0][1]["epoch_max"] == 3
    # one tracked "loss" call per epoch, in order, matching the scripted losses
    loss_calls = [c for c in tracked.track.call_args_list if c.kwargs.get("name") == "loss"]
    assert [c.args[0] for c in loss_calls] == [0.5, 0.3, 0.2]
    assert [c.kwargs["step"] for c in loss_calls] == [1, 2, 3]
    tracked.close.assert_called_once()


def test_train_model_closes_the_run_even_when_stopped_early(mocked_training, monkeypatch, tmp_path):
    tracked = MagicMock()
    monkeypatch.setattr(
        "velum_core.experiment_tracking.start_run",
        lambda experiment, hparams: tracked)

    config = _base_config(tmp_path, epoch_max=10)
    sm = _fake_state_manager()
    sm.check_stop_flag.side_effect = [False, False, True]   # stop after 2 epochs
    train_model.train_model(config, sm)

    loss_calls = [c for c in tracked.track.call_args_list if c.kwargs.get("name") == "loss"]
    assert len(loss_calls) == 2
    tracked.close.assert_called_once()


def test_train_model_closes_the_run_even_on_a_training_exception(mocked_training, monkeypatch, tmp_path):
    tracked = MagicMock()
    monkeypatch.setattr(
        "velum_core.experiment_tracking.start_run",
        lambda experiment, hparams: tracked)
    monkeypatch.setattr(cellseg1_train, "train_epoch",
                        MagicMock(side_effect=RuntimeError("boom")))

    config = _base_config(tmp_path, epoch_max=3)
    with pytest.raises(RuntimeError):
        train_model.train_model(config, _fake_state_manager())
    tracked.close.assert_called_once()


def test_train_model_works_with_the_real_no_op_tracker_when_aim_is_absent(mocked_training, tmp_path):
    """No monkeypatching of experiment_tracking at all: with the real
    (aim-less, in this sandbox) start_run(), training must still complete —
    the whole point of the _NullRun fallback."""
    config = _base_config(tmp_path, epoch_max=2)
    train_model.train_model(config, _fake_state_manager())   # must not raise
