"""Headless wiring tests for Models & Train / Dashboard (studio/extra_screens.py).

Offscreen Qt, no napari/torch: ``TrainController.start_training`` is always
monkeypatched here so no test ever spawns a real background training thread
or touches torch/cellseg1_train — those are exercised by
``test_train_controller.py`` (pure logic) instead. Mirrors
``test_new_project_dialog.py``'s conventions.
"""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
es = pytest.importorskip("studio.extra_screens")

from PyQt6.QtWidgets import QApplication, QWidget

from studio import theme
from studio.project import Project, ProjectSettings, ProjectStats
from studio.project_controller import ProjectController
from studio.train_controller import TrainController


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def parent(app):
    w = QWidget()
    w.resize(1200, 900)
    w.show()
    return w


@pytest.fixture
def train_ctrl(tmp_path):
    return TrainController(storage_dir=tmp_path / "storage")


@pytest.fixture
def project_ctrl(tmp_path):
    return ProjectController(store_root=tmp_path / "projects", seed_if_empty=False)


@pytest.fixture
def toasts():
    return []


@pytest.fixture
def on_toast(toasts):
    return lambda title, sub: toasts.append((title, sub))


def _add_backbone(train_ctrl, vit_name="vit_h"):
    d = train_ctrl.sam_backbone_dir
    d.mkdir(parents=True, exist_ok=True)
    names = {"vit_h": "sam_vit_h_4b8939.pth", "vit_l": "sam_vit_l_0b3195.pth", "vit_b": "sam_vit_b_01ec64.pth"}
    (d / names[vit_name]).write_bytes(b"fake")


def _annotated_pair(tmp_path):
    import cv2
    import numpy as np
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    image = work / "cells.png"
    cv2.imwrite(str(image), (np.random.rand(16, 16, 3) * 255).astype(np.uint8))
    mask_path = work / "cells_mask.png"
    mask = np.zeros((16, 16), dtype=np.uint16)
    mask[2:6, 2:6] = 1
    mask[8:12, 8:12] = 2
    cv2.imwrite(str(mask_path), mask)
    return image, mask_path


# ── ModelsScreen ─────────────────────────────────────────────────────────────
def test_models_screen_empty_state_disables_start(parent, train_ctrl, project_ctrl, on_toast):
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    scr.setParent(parent)
    assert scr._can_start() is False


def test_models_screen_header_count_reflects_trained_models(parent, train_ctrl, project_ctrl, on_toast, tmp_path):
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    lora_dir = train_ctrl.lora_out_dir
    (lora_dir / "m.pth").write_bytes(b"x")
    (lora_dir / "m.json").write_text(json.dumps({
        "vit_name": "vit_h", "image_encoder_lora_rank": 8, "train_id": [0],
        "saved_at": "2026-01-01T00:00:00+00:00", "epoch_max": 100, "loss_history": [],
    }))
    scr.refresh()
    from PyQt6.QtWidgets import QLabel
    subtitles = [w.text() for w in scr.findChildren(QLabel) if "trained adapter" in w.text()]
    assert subtitles == ["1 trained adapter · one-shot LoRA fine-tuning"]


def test_models_screen_picking_image_finds_mask_and_enables_start(parent, train_ctrl, project_ctrl, on_toast, tmp_path):
    _add_backbone(train_ctrl)
    image, mask = _annotated_pair(tmp_path)
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    scr._image_path = image
    scr._mask_path = train_ctrl.find_mask_for_image(image)
    scr.refresh()
    assert scr._mask_path == mask
    assert scr._can_start() is True
    assert "2 cells" in scr._image_field_text()


def test_models_screen_picking_image_without_mask_shows_warning_and_disables_start(
        parent, train_ctrl, project_ctrl, on_toast, tmp_path):
    _add_backbone(train_ctrl)
    work = tmp_path / "work2"
    work.mkdir()
    import cv2
    import numpy as np
    image = work / "lonely.png"
    cv2.imwrite(str(image), (np.random.rand(8, 8, 3) * 255).astype(np.uint8))
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    scr._image_path = image
    scr._mask_path = train_ctrl.find_mask_for_image(image)
    scr.refresh()
    assert scr._mask_path is None
    assert scr._can_start() is False
    assert "No mask found" in scr._status_text()


def test_models_screen_start_training_calls_controller_and_toasts(
        parent, train_ctrl, project_ctrl, on_toast, toasts, tmp_path, monkeypatch):
    _add_backbone(train_ctrl)
    image, mask = _annotated_pair(tmp_path)
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    scr._image_path, scr._mask_path = image, mask

    calls = []
    monkeypatch.setattr(train_ctrl, "start_training", lambda config, **kw: calls.append(config))
    scr._start_training()
    assert len(calls) == 1
    assert calls[0]["image_encoder_lora_rank"] == int(scr._lora_rank)
    assert any("Training started" in t for t, _ in toasts)


def test_models_screen_start_training_with_invalid_state_toasts_error_without_calling_controller(
        parent, train_ctrl, project_ctrl, on_toast, toasts, monkeypatch):
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    calls = []
    monkeypatch.setattr(train_ctrl, "start_training", lambda config, **kw: calls.append(config))
    scr._start_training()  # no image/mask picked at all
    assert calls == []
    assert any("Can't start training" in t for t, _ in toasts)


def test_models_screen_stop_training_delegates_to_controller(parent, train_ctrl, project_ctrl, on_toast, monkeypatch):
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    calls = []
    monkeypatch.setattr(train_ctrl, "stop_training", lambda: calls.append(True))
    scr._stop_training()
    assert calls == [True]


def test_models_screen_import_model_copies_and_refreshes(parent, train_ctrl, project_ctrl, on_toast, toasts, tmp_path, monkeypatch):
    src = tmp_path / "external.pth"
    src.write_bytes(b"weights")
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    monkeypatch.setattr(es.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(src), "")))
    scr._import_model()
    assert (train_ctrl.lora_out_dir / "external.pth").exists()
    assert any("Model imported" in t for t, _ in toasts)


def test_models_screen_select_model_without_active_project_toasts_hint(
        parent, train_ctrl, project_ctrl, on_toast, toasts):
    lora_dir = train_ctrl.lora_out_dir
    lora_dir.mkdir(parents=True, exist_ok=True)
    (lora_dir / "m.pth").write_bytes(b"x")
    (lora_dir / "m.json").write_text(json.dumps({
        "vit_name": "vit_h", "image_encoder_lora_rank": 8, "train_id": [0],
        "saved_at": "2026-01-01T00:00:00+00:00", "epoch_max": 100, "loss_history": [],
    }))
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    model = train_ctrl.list_trained_models()[0]
    scr._select_model(model)
    assert any("No active project" in t for t, _ in toasts)


def test_models_screen_select_model_with_active_project_updates_settings(
        parent, train_ctrl, project_ctrl, on_toast, toasts):
    lora_dir = train_ctrl.lora_out_dir
    lora_dir.mkdir(parents=True, exist_ok=True)
    (lora_dir / "m.pth").write_bytes(b"x")
    (lora_dir / "m.json").write_text(json.dumps({
        "vit_name": "vit_l", "image_encoder_lora_rank": 16, "train_id": [0],
        "saved_at": "2026-01-01T00:00:00+00:00", "epoch_max": 100, "loss_history": [],
    }))
    project = project_ctrl.store.create("My Project")
    project_ctrl.set_active(project.id)
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    model = train_ctrl.list_trained_models()[0]
    scr._select_model(model)
    reloaded = project_ctrl.store.load(project.id)
    assert reloaded.settings.model_name == str(model.checkpoint)
    assert reloaded.settings.vit_name == "vit_l"
    assert reloaded.settings.lora_rank == 16
    assert any("Model selected" in t for t, _ in toasts)


def test_models_screen_backbone_rank_epoch_selection_updates_state(parent, train_ctrl, project_ctrl, on_toast):
    _add_backbone(train_ctrl, "vit_h")
    _add_backbone(train_ctrl, "vit_l")
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    scr._set_backbone("ViT-L")
    assert scr._vit_name == "vit_l"
    scr._set_rank("32")
    assert scr._lora_rank == "32"
    scr._set_epochs("500")
    assert scr._epochs == "500"


def test_models_screen_on_live_tick_stops_timer_when_idle(parent, train_ctrl, project_ctrl, on_toast):
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    scr._live_timer.start()
    assert scr._live_timer.isActive()
    scr._on_live_tick()
    assert not scr._live_timer.isActive()


def test_models_screen_safe_emit_guards_against_a_deleted_widget(parent, train_ctrl, project_ctrl, on_toast):
    """Regression test for the documented hazard (studio-subproject memory,
    lesson #4): a background thread's completion callback can outlive the
    widget it targets (e.g. torn down by a theme toggle mid-training).
    Emitting a signal on a since-deleted QObject must not raise past the
    guard — reproduced the same way the memory's own investigation did,
    by force-deleting the underlying C++ object with sip.delete()."""
    from PyQt6 import sip
    scr = es.ModelsScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    log_fn, finish_fn = scr._safe_emit_log, scr._safe_emit_finish
    sip.delete(scr)
    log_fn("Training complete")     # must not raise
    finish_fn()                     # must not raise


# ── DashboardScreen ──────────────────────────────────────────────────────────
def test_dashboard_screen_empty_state_constructs_without_crashing(parent, train_ctrl, project_ctrl, on_toast):
    scr = es.DashboardScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    scr.setParent(parent)
    from PyQt6.QtWidgets import QLabel
    labels = [w.text() for w in scr.findChildren(QLabel)]
    assert any("No runs yet" in t for t in labels)
    assert any("No training runs yet" in t for t in labels)


def test_dashboard_screen_shows_real_data_when_present(parent, train_ctrl, project_ctrl, on_toast):
    lora_dir = train_ctrl.lora_out_dir
    lora_dir.mkdir(parents=True, exist_ok=True)
    (lora_dir / "m.pth").write_bytes(b"x")
    (lora_dir / "m.json").write_text(json.dumps({
        "vit_name": "vit_h", "epoch_max": 100,
        "loss_history": [{"epoch": 1, "loss": 0.8}, {"epoch": 2, "loss": 0.3}],
        "saved_at": "2026-01-01T00:00:00+00:00",
    }))
    train_ctrl.state_manager.append_history_entry({
        "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:06:00+00:00",
        "checkpoint": str(lora_dir / "m.pth"), "status": "completed",
    })
    project = Project(id="p", name="Benchmarked Project",
                      settings=ProjectSettings(engine="sam2"),
                      stats=ProjectStats(last_f1=0.91, n_cells=9700))
    project_ctrl.store.save(project, touch=False)

    scr = es.DashboardScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    assert scr.findChildren(es._LineChart)
    assert scr.findChildren(es._BarChart)
    from PyQt6.QtWidgets import QLabel
    labels = [w.text() for w in scr.findChildren(QLabel)]
    assert any("2 tracked runs" in t for t in labels)
    assert any("Benchmarked Project" in t for t in labels)


def test_dashboard_screen_open_in_aim_success_opens_browser_and_toasts(
        parent, train_ctrl, project_ctrl, on_toast, toasts, monkeypatch):
    scr = es.DashboardScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)
    monkeypatch.setattr(scr._dashboard, "open_in_aim", lambda: "http://127.0.0.1:9999")
    opened = []
    monkeypatch.setattr(es.webbrowser, "open", lambda url: opened.append(url))
    scr._open_in_aim()
    assert opened == ["http://127.0.0.1:9999"]
    assert any("Opening Aim" in t for t, _ in toasts)


def test_dashboard_screen_open_in_aim_missing_aim_toasts_hint(
        parent, train_ctrl, project_ctrl, on_toast, toasts, monkeypatch):
    scr = es.DashboardScreen(theme.DARK, train_ctrl, project_ctrl, on_toast)

    def _raise():
        raise RuntimeError("Aim is not installed — run: pip install aim")
    monkeypatch.setattr(scr._dashboard, "open_in_aim", _raise)
    scr._open_in_aim()
    assert any("Aim isn't installed" in t for t, _ in toasts)
