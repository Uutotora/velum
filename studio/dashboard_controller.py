"""Velum — the Dashboard tab controller.

Real experiment data for the Dashboard screen's two charts and runs table,
built from the same plain on-disk JSON this app already writes for real
(training history + per-checkpoint sidecars via ``train_controller.py``,
benchmarked project stats via ``project_controller.py``) rather than
querying Aim's own storage directly: empirically, the installed ``aim``
version's read API (``Repo.get_run``/``Repo.query_runs``) does not return
data even for the real, 484-run ``data_store/aim_repo`` this app has already
produced — ``get_run()`` returns ``None`` for every hash and ``query_runs``
raises ``NotImplementedError`` outside of Aim's own ``aim up`` server
process. "Open in Aim" still shells out to that real server (via
``velum_core.experiment_tracking``) for the full cross-run UI; Studio's
own charts stay fed by the robust, no-extra-process JSON files instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from studio.project import ENGINE_LABELS
from studio.project_controller import ProjectController, format_count, relative_time
from studio.train_controller import TrainController, duration_str, parse_iso


@dataclass
class DashRun:
    """One row of the Dashboard "Runs" table."""

    name: str
    engine_label: str
    f1: Optional[str]
    cells: Optional[str]
    duration: Optional[str]
    when: str
    ok: bool


def _sort_ts(iso: str) -> datetime:
    dt = parse_iso(iso)
    if dt is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class DashboardController:
    """Aggregates ``TrainController`` + ``ProjectController`` data into the
    Dashboard's charts and runs table — independent of Qt."""

    def __init__(self, train_controller: TrainController, project_controller: ProjectController):
        self.train = train_controller
        self.projects = project_controller

    # ── runs table ───────────────────────────────────────────────────────────
    def runs_table(self) -> list[DashRun]:
        rows: list[tuple[datetime, DashRun]] = []

        models_by_name = {m.name: m for m in self.train.list_trained_models()}
        for entry in self.train.state_manager.load_history():
            name = Path(entry.get("checkpoint", "")).stem or "training run"
            started, finished = entry.get("started_at", ""), entry.get("finished_at", "")
            dur = None
            s_dt, f_dt = parse_iso(started), parse_iso(finished)
            if s_dt and f_dt:
                dur = duration_str((f_dt - s_dt).total_seconds())
            model = models_by_name.get(name)
            cells = format_count(model.n_cells) if model and model.n_cells is not None else None
            rows.append((_sort_ts(finished or started), DashRun(
                name=name, engine_label="CellSeg1 · LoRA", f1=None,
                cells=cells, duration=dur, when=relative_time(finished or started), ok=False,
            )))

        for project in self.projects.store.list():
            # Any real segmentation activity shows up here, not only a
            # GT-benchmarked one — a plain predict/batch run with no ground
            # truth just shows "—" for F1 (ok=False, same muted styling as a
            # training run) instead of being invisible until benchmarked.
            if project.stats.last_f1 is None and not project.stats.n_cells:
                continue
            rows.append((_sort_ts(project.updated_at), DashRun(
                name=project.name,
                engine_label=ENGINE_LABELS.get(project.engine, project.engine),
                f1=f"{project.stats.last_f1:.2f}" if project.stats.last_f1 is not None else None,
                cells=format_count(project.stats.n_cells) if project.stats.n_cells else None,
                duration=None,
                when=relative_time(project.updated_at),
                ok=project.stats.last_f1 is not None,
            )))

        rows.sort(key=lambda r: r[0], reverse=True)
        return [r for _, r in rows]

    # ── training loss chart ──────────────────────────────────────────────────
    def loss_curve(self) -> tuple[list[float], str]:
        """``(losses, caption)`` — the live run's curve-so-far if training is
        active right now, else the most recently saved model's full curve,
        else ``([], "")`` (no runs yet)."""
        live = self.train.current_run()
        losses = [e["loss"] for e in self.train.current_loss_history()] if live is not None else []
        if losses:
            total = self.train.active_epoch_max() or "?"
            return losses, f"{live.name} · {total} epochs"

        newest: Optional[dict] = None
        newest_name = ""
        for jf in Path(self.train.lora_out_dir).glob("*.json"):
            try:
                sidecar = json.loads(jf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not sidecar.get("loss_history"):
                continue
            if newest is None or sidecar.get("saved_at", "") > newest.get("saved_at", ""):
                newest, newest_name = sidecar, jf.stem
        if newest is None:
            return [], ""
        losses = [e["loss"] for e in newest["loss_history"]]
        return losses, f"{newest_name} · {newest.get('epoch_max', len(losses))} epochs"

    # ── F1-across-runs bar chart ─────────────────────────────────────────────
    def f1_bars(self) -> list[float]:
        """Benchmarked projects' F1, oldest first (last bar = most recent —
        matches ``_BarChart``'s highlight-the-last-bar convention)."""
        scored = [p for p in self.projects.store.list() if p.stats.last_f1 is not None]
        scored.sort(key=lambda p: _sort_ts(p.updated_at))
        return [p.stats.last_f1 for p in scored]

    # ── Open in Aim ──────────────────────────────────────────────────────────
    def open_in_aim(self) -> str:
        """Start (if needed) and return the URL of Aim's own dashboard
        server. Raises ``RuntimeError`` if the optional ``aim`` package isn't
        installed — callers should show that message rather than crash."""
        from velum_core import experiment_tracking as tracking
        return tracking.ensure_dashboard_running()
