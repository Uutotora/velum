"""Optional experiment tracking (Aim) for predict/train/auto-tune runs.

`Aim <https://aimstack.io/>`_ is a fully local, self-hosted, open-source
experiment tracker with a modern run-comparison UI — exactly the shape of
data this app already produces (params in, score/metrics out, whether from
one-shot predicts, multi-epoch training, or the multi-round auto-tune loop).
It is an optional dependency (``pip install -e ".[tracking]"``), lazy-
imported and ``available()``-gated the same way sam2/cellpose/Ollama already
are elsewhere in this package: nothing here is a hard dependency, and every
call degrades to a no-op when Aim isn't installed, so the app (and CI's
pure-logic suite) is unaffected.

Every entry point (single/batch predict, benchmark, LoRA training, the
auto-tune loop) logs into *one* shared local repo under
``STORAGE_DIR/aim_repo``, distinguished by ``experiment=``, so Aim's own UI
becomes a single unified history across the whole app instead of separate
custom charts per tab.
"""
from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from typing import Any


def available() -> bool:
    """Whether the optional `aim` package is importable."""
    try:
        import aim  # noqa: F401
        return True
    except Exception:
        return False


def repo_path() -> Path:
    from project_root import STORAGE_DIR
    return Path(STORAGE_DIR) / "aim_repo"


class _NullRun:
    """No-op stand-in used whenever Aim isn't installed or fails to
    initialise, so predict/train/auto-tune never depend on it succeeding."""

    def track(self, *a, **k) -> None:
        pass

    def __setitem__(self, key, value) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _AimRun:
    """Thin wrapper around a real ``aim.Run``; every call is guarded so a
    tracking hiccup (disk full, a corrupt repo, a future Aim API change)
    degrades to a dropped data point instead of an interrupted real run."""

    def __init__(self, experiment: str, hparams: dict[str, Any]):
        import aim
        repo_path().mkdir(parents=True, exist_ok=True)
        self._run = aim.Run(repo=str(repo_path()), experiment=experiment)
        if hparams:
            try:
                self._run["hparams"] = _sanitize(hparams)
            except Exception:
                pass

    def track(self, value, *, name: str | None = None, step: int | None = None,
              context: dict[str, Any] | None = None) -> None:
        try:
            kw: dict[str, Any] = {"step": step, "context": context or {}}
            if name is not None:
                kw["name"] = name
            self._run.track(value, **kw)
        except Exception:
            pass

    def __setitem__(self, key, value) -> None:
        try:
            self._run[key] = value
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._run.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _sanitize(d: dict[str, Any]) -> dict[str, Any]:
    """Aim's hparams want JSON-ish scalars; silently drop anything else
    (nested dicts, numpy arrays, Path objects, ...) rather than risk the one
    call that sets them crashing a real run over an unrelated config value."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None or isinstance(v, (int, float, str, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(
                isinstance(x, (int, float, str, bool)) for x in v):
            out[k] = list(v)
    return out


def start_run(experiment: str, hparams: dict[str, Any] | None = None):
    """Start a tracked run under ``experiment`` (e.g. "predict"/"train"/
    "auto-tune" — Aim's own UI groups/filters runs by this), or a no-op
    stand-in if Aim isn't installed or fails to initialise for any reason.
    Use as a context manager so ``close()`` always runs::

        with start_run("predict", {"engine": "cellseg1"}) as run:
            run.track(0.83, name="n_cells")
    """
    if not available():
        return _NullRun()
    try:
        return _AimRun(experiment, hparams or {})
    except Exception:
        return _NullRun()


# ── Dashboard (Aim's own local web UI, `aim up`) ─────────────────────────────

_dashboard_proc: "subprocess.Popen | None" = None
_dashboard_url: str | None = None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _aim_cli_path() -> str:
    """Locate the ``aim`` console-script reliably: a sibling of the
    currently-running Python interpreter, which is where pip installs it —
    this works even when the environment's ``bin/`` isn't on ``PATH`` (e.g.
    the app launched via a GUI entry point or wrapper script rather than an
    activated shell), unlike a bare ``"aim"`` lookup (confirmed to actually
    fail that way against a real Aim install in a non-activated venv — see
    docs/BACKLOG.md). Falls back to the bare command name for an unusual
    install layout where this sibling doesn't exist.
    """
    import sys
    sibling = Path(sys.executable).parent / "aim"
    return str(sibling) if sibling.exists() else "aim"


def ensure_dashboard_running() -> str:
    """Start Aim's own dashboard server (``aim up``) once, reusing it across
    every call from any widget — a shared singleton, the same idea as this
    app's ``get_log_window()``/``get_measurements_window()``. Returns the
    local URL. Raises ``RuntimeError`` if Aim isn't installed.
    """
    global _dashboard_proc, _dashboard_url
    if _dashboard_proc is not None and _dashboard_proc.poll() is None:
        return _dashboard_url
    if not available():
        raise RuntimeError("Aim is not installed — run: pip install aim")
    repo_path().mkdir(parents=True, exist_ok=True)
    port = _free_port()
    _dashboard_proc = subprocess.Popen(
        [_aim_cli_path(), "up", "--repo", str(repo_path()), "--port", str(port), "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _dashboard_url = f"http://127.0.0.1:{port}"
    return _dashboard_url


def stop_dashboard() -> None:
    """Terminate the dashboard subprocess if one is running."""
    global _dashboard_proc, _dashboard_url
    if _dashboard_proc is not None:
        _dashboard_proc.terminate()
    _dashboard_proc = None
    _dashboard_url = None
