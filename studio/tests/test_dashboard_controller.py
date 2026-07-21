"""Tests for the Studio Dashboard tab controller (studio/dashboard_controller.py).

Pure-logic, no Qt/torch/napari/aim — runs under the light CI `test` group.
"""
import json

import pytest

from studio.dashboard_controller import DashboardController
from studio.project import Project, ProjectSettings, ProjectStats, ProjectStore
from studio.project_controller import ProjectController
from studio.train_controller import TrainController


@pytest.fixture
def train_ctrl(tmp_path):
    return TrainController(storage_dir=tmp_path / "storage")


@pytest.fixture
def project_ctrl(tmp_path):
    return ProjectController(ProjectStore(tmp_path / "projects"), seed_if_empty=False)


@pytest.fixture
def dash(train_ctrl, project_ctrl):
    return DashboardController(train_ctrl, project_ctrl)


def _benchmarked_project(project_ctrl, name, f1, n_cells, updated_at, engine="cellseg1"):
    p = Project(id=name, name=name, updated_at=updated_at,
                settings=ProjectSettings(engine=engine),
                stats=ProjectStats(last_f1=f1, n_cells=n_cells))
    project_ctrl.store.save(p, touch=False)
    return p


def _completed_training_run(train_ctrl, stem, started, finished, *, n_cells_masks=None):
    lora_dir = train_ctrl.lora_out_dir
    lora_dir.mkdir(parents=True, exist_ok=True)
    (lora_dir / f"{stem}.pth").write_bytes(b"fake")
    sidecar = {
        "vit_name": "vit_h", "image_encoder_lora_rank": 8, "train_id": [0],
        "saved_at": finished, "epoch_max": 100,
        "loss_history": [{"epoch": 1, "loss": 0.9}, {"epoch": 2, "loss": 0.4}],
    }
    if n_cells_masks is not None:
        mask_dir = train_ctrl.storage_dir / f"masks_{stem}"
        mask_dir.mkdir(parents=True, exist_ok=True)
        import cv2
        import numpy as np
        for i, n in enumerate(n_cells_masks):
            m = np.zeros((16, 16), dtype=np.uint16)
            for lbl in range(1, n + 1):
                m[lbl, lbl] = lbl  # trivially distinct nonzero pixels, max() == n
            cv2.imwrite(str(mask_dir / f"m{i}.png"), m)
        sidecar["train_mask_dir"] = str(mask_dir)
    (lora_dir / f"{stem}.json").write_text(json.dumps(sidecar))
    train_ctrl.state_manager.append_history_entry({
        "started_at": started, "finished_at": finished,
        "checkpoint": str(lora_dir / f"{stem}.pth"), "status": "completed",
    })


# ── empty state ──────────────────────────────────────────────────────────────
def test_runs_table_empty_when_nothing_tracked(dash):
    assert dash.runs_table() == []


def test_loss_curve_empty_when_nothing_trained(dash):
    assert dash.loss_curve() == ([], "")


def test_f1_bars_empty_when_nothing_benchmarked(dash):
    assert dash.f1_bars() == []


# ── runs_table ───────────────────────────────────────────────────────────────
def test_runs_table_includes_a_completed_training_run(dash, train_ctrl):
    _completed_training_run(train_ctrl, "nuclei-r8",
                            "2026-01-01T00:00:00+00:00", "2026-01-01T00:06:02+00:00",
                            n_cells_masks=[3])
    rows = dash.runs_table()
    assert len(rows) == 1
    row = rows[0]
    assert row.name == "nuclei-r8"
    assert row.engine_label == "CellSeg1 · LoRA"
    assert row.f1 is None
    assert row.duration == "6m 02s"
    assert row.ok is False


def test_runs_table_includes_a_benchmarked_project(dash, project_ctrl):
    _benchmarked_project(project_ctrl, "Tissue Cohort", 0.87, 5000, "2026-01-01T00:00:00+00:00",
                         engine="cellpose")
    rows = dash.runs_table()
    assert len(rows) == 1
    row = rows[0]
    assert row.name == "Tissue Cohort"
    assert row.engine_label == "Cellpose-SAM"
    assert row.f1 == "0.87"
    assert row.cells == "5k"
    assert row.duration is None
    assert row.ok is True


def test_runs_table_ignores_unbenchmarked_projects(dash, project_ctrl):
    p = Project(id="x", name="No F1 yet")  # stats.last_f1 AND n_cells both default empty
    project_ctrl.store.save(p, touch=False)
    assert dash.runs_table() == []


def test_runs_table_includes_a_plain_segmented_project_with_no_gt(dash, project_ctrl):
    """A Segment-tab predict/batch run with no ground truth still shows up
    here (Studio's whole point of logging segmentation activity to the
    Dashboard) — just with F1 "—" (ok=False), not silently invisible until
    someone benchmarks it."""
    p = Project(id="y", name="Segmented, no GT", updated_at="2026-01-02T00:00:00+00:00",
               settings=ProjectSettings(engine="cellseg1"), stats=ProjectStats(n_cells=247))
    project_ctrl.store.save(p, touch=False)
    rows = dash.runs_table()
    assert len(rows) == 1
    row = rows[0]
    assert row.name == "Segmented, no GT"
    assert row.cells == "247"
    assert row.f1 is None
    assert row.ok is False


def test_runs_table_sorted_newest_first_across_both_sources(dash, train_ctrl, project_ctrl):
    _completed_training_run(train_ctrl, "older-run",
                            "2020-01-01T00:00:00+00:00", "2020-01-01T00:05:00+00:00")
    _benchmarked_project(project_ctrl, "Newer Project", 0.9, 100, "2026-06-01T00:00:00+00:00")
    names = [r.name for r in dash.runs_table()]
    assert names == ["Newer Project", "older-run"]


# ── loss_curve ───────────────────────────────────────────────────────────────
def test_loss_curve_reads_the_newest_completed_models_sidecar(dash, train_ctrl):
    _completed_training_run(train_ctrl, "old", "2026-01-01T00:00:00+00:00", "2026-01-01T00:05:00+00:00")
    lora_dir = train_ctrl.lora_out_dir
    (lora_dir / "new.pth").write_bytes(b"fake")
    (lora_dir / "new.json").write_text(json.dumps({
        "vit_name": "vit_h", "epoch_max": 50,
        "loss_history": [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.1}],
        "saved_at": "2026-06-01T00:00:00+00:00",
    }))
    losses, caption = dash.loss_curve()
    assert losses == [0.5, 0.1]
    assert caption == "new · 50 epochs"


def test_loss_curve_skips_sidecars_with_no_loss_history(dash, train_ctrl):
    lora_dir = train_ctrl.lora_out_dir
    lora_dir.mkdir(parents=True, exist_ok=True)
    (lora_dir / "empty.pth").write_bytes(b"fake")
    (lora_dir / "empty.json").write_text(json.dumps({
        "vit_name": "vit_h", "epoch_max": 50, "loss_history": [], "saved_at": "2026-06-01T00:00:00+00:00",
    }))
    assert dash.loss_curve() == ([], "")


# ── f1_bars ──────────────────────────────────────────────────────────────────
def test_f1_bars_oldest_first(dash, project_ctrl):
    _benchmarked_project(project_ctrl, "First", 0.70, 10, "2026-01-01T00:00:00+00:00")
    _benchmarked_project(project_ctrl, "Second", 0.90, 10, "2026-03-01T00:00:00+00:00")
    _benchmarked_project(project_ctrl, "Third", 0.85, 10, "2026-02-01T00:00:00+00:00")
    assert dash.f1_bars() == [0.70, 0.85, 0.90]  # chronological, not insertion order


# ── open_in_aim ──────────────────────────────────────────────────────────────
def test_open_in_aim_delegates_to_experiment_tracking(dash, monkeypatch):
    from velum_core import experiment_tracking as tracking
    monkeypatch.setattr(tracking, "ensure_dashboard_running", lambda: "http://127.0.0.1:1234")
    assert dash.open_in_aim() == "http://127.0.0.1:1234"


def test_open_in_aim_propagates_runtime_error_when_aim_missing(dash, monkeypatch):
    from velum_core import experiment_tracking as tracking

    def _raise():
        raise RuntimeError("Aim is not installed — run: pip install aim")
    monkeypatch.setattr(tracking, "ensure_dashboard_running", _raise)
    with pytest.raises(RuntimeError, match="pip install aim"):
        dash.open_in_aim()
