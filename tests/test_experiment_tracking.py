"""Unit tests for velum_core.experiment_tracking (optional Aim logging).

Aim itself is never installed in this sandbox (it's an optional `tracking`
extra, not part of the pure-logic `test` dependency-group), so the
"available" path is exercised the same way tests/test_engines_sam2.py fakes
an optional heavy package: injecting a bare `types.ModuleType` into
`sys.modules` with just enough attributes (a fake `Run` class) attached to
observe what this module's wrapper does with it — no real aim install or
network/disk activity involved anywhere.

`ensure_dashboard_running`/`stop_dashboard` hold module-level singleton state
(the same class of shared-global fragility already documented against
get_log_window()/get_measurements_window() elsewhere in this repo), so every
test that touches them resets it via an autouse fixture.
"""
import subprocess
import sys
import types

import pytest

from velum_core import experiment_tracking as tracking


@pytest.fixture(autouse=True)
def _reset_dashboard_singleton():
    tracking._dashboard_proc = None
    tracking._dashboard_url = None
    yield
    tracking._dashboard_proc = None
    tracking._dashboard_url = None


def _install_fake_aim(monkeypatch, run_cls):
    fake = types.ModuleType("aim")
    fake.Run = run_cls
    monkeypatch.setitem(sys.modules, "aim", fake)


class _RecordingRun:
    """A fake aim.Run: records every call instead of touching disk."""
    instances = []

    def __init__(self, repo=None, experiment=None):
        self.repo = repo
        self.experiment = experiment
        self.items = {}
        self.tracked = []
        self.closed = False
        _RecordingRun.instances.append(self)

    def __setitem__(self, key, value):
        self.items[key] = value

    def track(self, value, **kw):
        self.tracked.append((value, kw))

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _clear_recording_run():
    _RecordingRun.instances = []
    yield


# ── available() ──────────────────────────────────────────────────────────────

def test_available_false_without_aim(monkeypatch):
    monkeypatch.setitem(sys.modules, "aim", None)   # force ImportError on import
    assert tracking.available() is False


def test_available_true_with_a_fake_package(monkeypatch):
    _install_fake_aim(monkeypatch, _RecordingRun)
    assert tracking.available() is True


# ── start_run(): no-op path (Aim absent) ─────────────────────────────────────

def test_start_run_without_aim_returns_a_safe_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "aim", None)
    run = tracking.start_run("predict", {"engine": "cellseg1"})
    run.track(0.9, name="score")            # must not raise
    run["anything"] = 123                     # must not raise
    run.close()                               # must not raise


def test_start_run_noop_works_as_a_context_manager(monkeypatch):
    monkeypatch.setitem(sys.modules, "aim", None)
    with tracking.start_run("predict") as run:
        run.track(1, name="x")
    # exiting the context must not raise either


# ── start_run(): real (fake-module) path ─────────────────────────────────────

def test_start_run_creates_a_run_with_repo_and_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _RecordingRun)
    tracking.start_run("auto-tune", {"strategy": "advisor"})
    assert len(_RecordingRun.instances) == 1
    run = _RecordingRun.instances[0]
    assert run.experiment == "auto-tune"
    assert run.repo == str(tmp_path / "aim_repo")


def test_start_run_creates_the_repo_directory(monkeypatch, tmp_path):
    repo = tmp_path / "nested" / "aim_repo"
    monkeypatch.setattr(tracking, "repo_path", lambda: repo)
    _install_fake_aim(monkeypatch, _RecordingRun)
    tracking.start_run("predict")
    assert repo.is_dir()


def test_start_run_sets_sanitized_hparams(monkeypatch, tmp_path):
    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _RecordingRun)
    tracking.start_run("predict", {"engine": "cellseg1", "resize_size": [512, 512],
                                   "lora_paths": {"a": "b"}})
    run = _RecordingRun.instances[0]
    assert run.items["hparams"]["engine"] == "cellseg1"
    assert "lora_paths" not in run.items["hparams"]   # nested dict dropped


def test_run_track_delegates_value_name_step_context(monkeypatch, tmp_path):
    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _RecordingRun)
    run = tracking.start_run("auto-tune")
    run.track(0.75, name="score", step=3)
    value, kw = _RecordingRun.instances[0].tracked[0]
    assert value == 0.75 and kw["name"] == "score" and kw["step"] == 3


def test_run_setitem_delegates(monkeypatch, tmp_path):
    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _RecordingRun)
    run = tracking.start_run("train")
    run["status"] = "completed"
    assert _RecordingRun.instances[0].items["status"] == "completed"


def test_run_close_delegates(monkeypatch, tmp_path):
    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _RecordingRun)
    run = tracking.start_run("train")
    run.close()
    assert _RecordingRun.instances[0].closed is True


# ── guarded failure paths: tracking must never break a real run ─────────────

def test_track_call_that_raises_is_swallowed(monkeypatch, tmp_path):
    class _BoomOnTrack(_RecordingRun):
        def track(self, value, **kw):
            raise RuntimeError("disk full")

    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _BoomOnTrack)
    run = tracking.start_run("predict")
    run.track(1, name="x")   # must not raise


def test_setitem_that_raises_is_swallowed(monkeypatch, tmp_path):
    class _BoomOnSetitem(_RecordingRun):
        def __setitem__(self, key, value):
            raise RuntimeError("nope")

    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _BoomOnSetitem)
    tracking.start_run("predict", {"a": 1})   # must not raise despite hparams assignment failing


def test_close_that_raises_is_swallowed(monkeypatch, tmp_path):
    class _BoomOnClose(_RecordingRun):
        def close(self):
            raise RuntimeError("nope")

    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _BoomOnClose)
    run = tracking.start_run("predict")
    run.close()   # must not raise


def test_constructor_that_raises_falls_back_to_noop(monkeypatch, tmp_path):
    class _BoomOnInit(_RecordingRun):
        def __init__(self, repo=None, experiment=None):
            raise RuntimeError("corrupt repo")

    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _BoomOnInit)
    run = tracking.start_run("predict")   # must not raise
    run.track(1, name="x")                # the returned no-op must be usable too


# ── _sanitize ────────────────────────────────────────────────────────────────

def test_sanitize_keeps_scalars_and_homogeneous_lists():
    out = tracking._sanitize({
        "a": 1, "b": 1.5, "c": "x", "d": True, "e": None, "f": [1, 2, 3],
    })
    assert out == {"a": 1, "b": 1.5, "c": "x", "d": True, "e": None, "f": [1, 2, 3]}


def test_sanitize_drops_nested_and_non_scalar_values():
    import numpy as np
    out = tracking._sanitize({
        "ok": 1, "nested": {"x": 1}, "array": np.zeros(3), "mixed_list": [1, "a", {}],
    })
    assert out == {"ok": 1}


# ── repo_path ─────────────────────────────────────────────────────────────────

def test_repo_path_is_under_storage_dir(monkeypatch, tmp_path):
    import project_root
    monkeypatch.setattr(project_root, "STORAGE_DIR", tmp_path)
    assert tracking.repo_path() == tmp_path / "aim_repo"


# ── dashboard subprocess management ─────────────────────────────────────────

class _FakeProc:
    def __init__(self, alive=True):
        self._alive = alive

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False


def test_ensure_dashboard_running_raises_without_aim(monkeypatch):
    monkeypatch.setitem(sys.modules, "aim", None)
    with pytest.raises(RuntimeError, match="not installed"):
        tracking.ensure_dashboard_running()


def test_ensure_dashboard_running_launches_aim_up_with_expected_args(monkeypatch, tmp_path):
    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _RecordingRun)
    calls = []

    def fake_popen(cmd, **kw):
        calls.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    url = tracking.ensure_dashboard_running()
    assert url.startswith("http://127.0.0.1:")
    cmd = calls[0]
    assert cmd[0].endswith("aim") and cmd[1] == "up"
    assert "--repo" in cmd and str(tmp_path / "aim_repo") in cmd
    assert "--host" in cmd and "127.0.0.1" in cmd


# ── _aim_cli_path: a bare "aim" only resolves via PATH, which a real Aim
#    install is not guaranteed to be on (confirmed against a real install in
#    a non-activated venv: `subprocess.Popen(["aim", ...])` raised
#    FileNotFoundError there) ─────────────────────────────────────────────

def test_aim_cli_path_prefers_the_sibling_of_the_running_interpreter(monkeypatch, tmp_path):
    fake_python = tmp_path / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_bytes(b"")
    (tmp_path / "bin" / "aim").write_bytes(b"")   # sibling console-script
    monkeypatch.setattr(sys, "executable", str(fake_python))
    assert tracking._aim_cli_path() == str(tmp_path / "bin" / "aim")


def test_aim_cli_path_falls_back_to_bare_command_without_a_sibling(monkeypatch, tmp_path):
    fake_python = tmp_path / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_bytes(b"")
    # no sibling "aim" file created
    monkeypatch.setattr(sys, "executable", str(fake_python))
    assert tracking._aim_cli_path() == "aim"


def test_ensure_dashboard_running_reuses_a_live_process(monkeypatch, tmp_path):
    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _RecordingRun)
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or _FakeProc())

    url1 = tracking.ensure_dashboard_running()
    url2 = tracking.ensure_dashboard_running()
    assert url1 == url2
    assert len(calls) == 1   # second call reused the running process


def test_stop_dashboard_lets_a_new_call_spawn_again(monkeypatch, tmp_path):
    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _RecordingRun)
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or _FakeProc())

    tracking.ensure_dashboard_running()
    tracking.stop_dashboard()
    tracking.ensure_dashboard_running()
    assert len(calls) == 2


def test_ensure_dashboard_running_restarts_after_the_process_died(monkeypatch, tmp_path):
    monkeypatch.setattr(tracking, "repo_path", lambda: tmp_path / "aim_repo")
    _install_fake_aim(monkeypatch, _RecordingRun)
    procs = [_FakeProc(alive=True), _FakeProc(alive=True)]
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: procs.pop(0))

    tracking.ensure_dashboard_running()
    tracking._dashboard_proc._alive = False   # simulate the server having died
    tracking.ensure_dashboard_running()
    assert procs == []   # both fake processes were consumed -> the second call restarted it
