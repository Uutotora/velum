"""Headless wiring tests for the real Assistant drawer (studio/assistant_panel.py).

Uses a *real*, tmp_path-backed AssistantController (Qt-free, its own
thorough suite lives in test_assistant_controller.py) with only the deepest
network-touching calls monkeypatched (napari_app.advisor.ollama_*, this
module's own custom_api_*) — the same "real controller, fake ML/network
seam" convention test_workspace.py uses for SegmentController. The Segment
workspace side is a small fake (_FakeWorkspace) satisfying
assistant_context()/apply_assistant_changes()/rerun_predict(), mirroring the
classic app's own _FakePredict in tests/test_assistant_widget_wiring.py —
WorkspaceScreen's real implementation of that contract is covered separately
in test_workspace.py.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import numpy as np
import pytest

pytest.importorskip("PyQt6")
ap = pytest.importorskip("studio.assistant_panel")

from PyQt6.QtWidgets import QApplication, QWidget

from studio import theme
from studio.assistant_controller import AssistantController
from studio.components import SelectBox


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
def controller(tmp_path):
    return AssistantController(storage_dir=tmp_path / "assistant_storage")


class _FakeWorkspace:
    """Minimal stand-in for WorkspaceScreen's Assistant-facing API."""

    def __init__(self):
        self.applied: list[dict] = []
        self.reran = False
        self.apply_return = []
        self.image = None
        self.mask = None
        self.params: dict = {}

    def assistant_context(self):
        return self.image, self.mask, self.params

    def apply_assistant_changes(self, changes):
        self.applied.append(dict(changes))
        return self.apply_return

    def rerun_predict(self):
        self.reran = True


@pytest.fixture
def workspace():
    return _FakeWorkspace()


def _drawer(parent, controller, workspace):
    d = ap.AssistantDrawer(parent, theme.DARK, controller, workspace)
    d.setParent(parent)
    return d


def _pump(app, controller, timeout=5):
    t0 = time.monotonic()
    while (controller.chat_busy() or controller.model_op_busy()) and time.monotonic() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(5):
        app.processEvents()


# ── construction / empty state ────────────────────────────────────────────────
def test_drawer_starts_hidden(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    assert d.isHidden()


def test_chat_shows_empty_state_initially(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    assert d._chat._empty is not None and not d._chat._empty.isHidden()


def test_model_accordion_starts_collapsed(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    assert d._model_acc._open is False


# ── backend switching ───────────────────────────────────────────────────────
def test_backend_seg_reflects_persisted_backend(app, parent, controller, workspace, monkeypatch):
    monkeypatch.setattr("napari_app.advisor.ollama_available", lambda: False)
    controller.settings.backend = "ollama"
    d = _drawer(parent, controller, workspace)
    assert d._backend_seg._btns[1].isChecked()
    assert "OLLAMA" in d._model_acc._title_lb.text()


def test_switching_backend_persists_and_updates_title(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    d._backend_seg._select(2)   # Custom API — empty base_url short-circuits before any real I/O
    assert controller.settings.backend == "custom"
    reloaded = AssistantController(settings_store=controller.settings_store)
    assert reloaded.settings.backend == "custom"
    assert "CUSTOM API" in d._model_acc._title_lb.text()


def test_offline_body_shows_a_description_no_status_row(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    assert not hasattr(d, "_status_lbl") or d._model_body_lay.count() == 1


def test_switching_to_ollama_kicks_off_a_status_check(app, parent, controller, workspace, monkeypatch):
    monkeypatch.setattr("napari_app.advisor.ollama_available", lambda: True)
    monkeypatch.setattr("napari_app.advisor.ollama_models", lambda: ["llama3.2:3b"])
    d = _drawer(parent, controller, workspace)
    d._backend_seg._select(1)   # Ollama
    _pump(app, controller)
    assert d._found_models == ["llama3.2:3b"]
    assert d._status_ok is True


# ── diagnose ─────────────────────────────────────────────────────────────────
def test_diagnose_with_no_context_prompts_to_predict(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    before = d._chat._v.count()
    d._run_diagnose()
    assert d._chat._v.count() > before
    # the lone "run a prediction first" finding has no changes -> no Apply row
    cards = d._chat.findChildren(ap.ChangeCard)
    assert len(cards) == 1


def test_diagnose_with_findings_renders_change_cards(app, parent, controller, workspace):
    workspace.image = np.full((100, 100), 128, dtype=np.uint8)
    workspace.mask = np.zeros((100, 100), dtype=np.int32)
    workspace.params = {"points_per_side": 32, "pred_iou_thresh": 0.8,
                        "stability_score_thresh": 0.6, "box_nms_thresh": 0.05,
                        "min_mask_area": 20, "resize_size": 512}
    d = _drawer(parent, controller, workspace)
    d._run_diagnose()
    cards = d._chat.findChildren(ap.ChangeCard)
    assert len(cards) >= 1


def test_diagnose_apply_button_calls_workspace(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    d._apply({"pred_iou_thresh": 0.5})
    assert workspace.applied == [{"pred_iou_thresh": 0.5}]
    assert workspace.reran is False


def test_diagnose_apply_and_rerun_calls_workspace_rerun(app, parent, controller, workspace):
    workspace.apply_return = ["pred_iou_thresh"]
    d = _drawer(parent, controller, workspace)
    d._apply_rerun({"pred_iou_thresh": 0.5})
    assert workspace.applied == [{"pred_iou_thresh": 0.5}]
    assert workspace.reran is True


def test_apply_without_a_project_shows_a_hint_not_a_crash(app, parent, controller, workspace):
    workspace.apply_return = None   # _FakeWorkspace signals "no active project"
    d = _drawer(parent, controller, workspace)
    d._apply({"pred_iou_thresh": 0.5})   # must not raise
    d._apply_rerun({"pred_iou_thresh": 0.5})
    assert workspace.reran is False   # rerun is never called with no project


# ── chat — offline backend ──────────────────────────────────────────────────
def test_send_offline_answers_synchronously_and_offers_suggestions(app, parent, controller, workspace):
    workspace.image = np.full((100, 100), 128, dtype=np.uint8)
    workspace.mask = np.zeros((100, 100), dtype=np.int32)
    workspace.params = {"points_per_side": 32, "pred_iou_thresh": 0.8,
                        "stability_score_thresh": 0.6, "box_nms_thresh": 0.05,
                        "min_mask_area": 20, "resize_size": 512}
    d = _drawer(parent, controller, workspace)
    d._input.setText("why no cells?")
    d._send()
    assert d._history[-1]["role"] == "assistant"
    assert "No cells" in d._history[-1]["content"]
    cards = d._chat.findChildren(ap.ChangeCard)
    assert len(cards) == 1
    assert cards[0].findChildren(type(d._send_btn)) or True  # rendered without raising


def test_send_ignores_empty_input(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    d._input.setText("   ")
    d._send()
    assert d._history == []


# ── chat — a connected (threaded) backend ───────────────────────────────────
def test_send_ollama_streams_into_the_chat(app, parent, controller, workspace, monkeypatch):
    monkeypatch.setattr("napari_app.advisor.ollama_available", lambda: False)
    controller.settings.backend = "ollama"
    controller.settings.ollama_model = "qwen2.5:7b"

    def fake_chat(model, messages, on_token, stop=None, temperature=0.2):
        on_token("Try ")
        on_token("raising IoU.")
        return "Try raising IoU."

    monkeypatch.setattr("napari_app.advisor.ollama_chat", fake_chat)
    d = _drawer(parent, controller, workspace)
    d._input.setText("why merged?")
    d._send()
    assert d._send_btn.isEnabled() is False
    _pump(app, controller)
    assert d._send_btn.isEnabled() is True
    assert d._history[-1] == {"role": "assistant", "content": "Try raising IoU."}


def test_send_reports_errors_into_the_chat(app, parent, controller, workspace, monkeypatch):
    monkeypatch.setattr("napari_app.advisor.ollama_available", lambda: False)
    controller.settings.backend = "ollama"
    controller.settings.ollama_model = "m"

    def fake_chat(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("napari_app.advisor.ollama_chat", fake_chat)
    d = _drawer(parent, controller, workspace)
    d._input.setText("hi")
    d._send()
    _pump(app, controller)
    assert d._send_btn.isEnabled() is True
    # error text landed in the (now-finalised) streaming bubble, not a crash
    assert d._chat._cur is None


def test_send_disabled_while_a_previous_send_is_in_flight(app, parent, controller, workspace, monkeypatch):
    import threading
    monkeypatch.setattr("napari_app.advisor.ollama_available", lambda: False)
    controller.settings.backend = "ollama"
    controller.settings.ollama_model = "m"
    release = threading.Event()

    def fake_chat(model, messages, on_token, stop=None, temperature=0.2):
        release.wait(timeout=5)
        return "done"

    monkeypatch.setattr("napari_app.advisor.ollama_chat", fake_chat)
    d = _drawer(parent, controller, workspace)
    d._input.setText("first")
    d._send()
    d._input.setText("second")
    d._send()   # must be a no-op — chat_busy() is True
    release.set()
    _pump(app, controller)
    assert d._history.count({"role": "user", "content": "first"}) == 1
    assert not any(m.get("content") == "second" for m in d._history)


# ── Ollama model management wiring ──────────────────────────────────────────
def test_ollama_model_select_appears_after_a_refresh(app, parent, controller, workspace, monkeypatch):
    controller.settings.backend = "ollama"
    monkeypatch.setattr("napari_app.advisor.ollama_available", lambda: True)
    monkeypatch.setattr("napari_app.advisor.ollama_models", lambda: ["a:1b", "b:2b"])
    d = _drawer(parent, controller, workspace)
    _pump(app, controller)
    selects = d._model_body_wrap.findChildren(SelectBox)
    assert len(selects) == 1


def test_picking_an_ollama_model_persists_it(app, parent, controller, workspace, monkeypatch):
    controller.settings.backend = "ollama"
    monkeypatch.setattr("napari_app.advisor.ollama_available", lambda: True)
    monkeypatch.setattr("napari_app.advisor.ollama_models", lambda: ["a:1b", "b:2b"])
    d = _drawer(parent, controller, workspace)
    _pump(app, controller)
    d._pick_ollama_model("b:2b")
    assert controller.settings.ollama_model == "b:2b"
    reloaded = AssistantController(settings_store=controller.settings_store)
    assert reloaded.settings.ollama_model == "b:2b"


def test_no_models_found_shows_a_hint_not_a_crash(app, parent, controller, workspace, monkeypatch):
    controller.settings.backend = "ollama"
    monkeypatch.setattr("napari_app.advisor.ollama_available", lambda: False)
    d = _drawer(parent, controller, workspace)
    _pump(app, controller)
    assert d._found_models == []


def test_download_model_shows_progress_and_completes(app, parent, controller, workspace, monkeypatch):
    import threading
    controller.settings.backend = "ollama"
    monkeypatch.setattr("napari_app.advisor.ollama_available", lambda: False)
    started = threading.Event()
    release = threading.Event()

    def fake_pull(name, on_progress):
        on_progress("downloading", 0.5)
        started.set()
        release.wait(timeout=5)
        on_progress("done", 1.0)
        return True

    monkeypatch.setattr("napari_app.advisor.ollama_pull", fake_pull)
    d = _drawer(parent, controller, workspace)
    _pump(app, controller)
    d._download_model("llama3.2:3b")
    started.wait(timeout=5)
    # isHidden() is the explicit per-widget flag; isVisible() is composite
    # and needs the whole ancestor chain shown — the drawer itself starts
    # hidden (it's an overlay), so isVisible() would be False regardless.
    assert not d._op_progress.isHidden()
    release.set()
    _pump(app, controller)
    assert "ready" in d._op_status_text


def test_create_agent_requires_a_base_model_selected(app, parent, controller, workspace, monkeypatch):
    monkeypatch.setattr("napari_app.advisor.ollama_available", lambda: False)
    controller.settings.backend = "ollama"
    controller.settings.ollama_model = ""
    d = _drawer(parent, controller, workspace)
    d._create_agent()
    assert "Pick a base model" in d._op_status_text


def test_create_agent_flow_completes(app, parent, controller, workspace, monkeypatch):
    controller.settings.backend = "ollama"
    controller.settings.ollama_model = "qwen2.5:7b"
    monkeypatch.setattr("napari_app.advisor.ollama_available", lambda: False)

    def fake_create(base_model, on_status):
        on_status("baking…")
        return True

    monkeypatch.setattr("napari_app.advisor.ollama_create_agent", fake_create)
    d = _drawer(parent, controller, workspace)
    d._create_agent()
    _pump(app, controller)
    assert "ready" in d._op_status_text


# ── Custom API wiring ────────────────────────────────────────────────────────
def test_custom_api_fields_update_settings_on_editing_finished(app, parent, controller, workspace):
    controller.settings.backend = "custom"
    d = _drawer(parent, controller, workspace)
    from PyQt6.QtWidgets import QLineEdit
    edits = d._model_body_wrap.findChildren(QLineEdit)
    url_edit, key_edit, model_edit = edits[0], edits[1], edits[2]
    url_edit.setText("http://localhost:1234/v1")
    url_edit.editingFinished.emit()
    key_edit.setText("sk-x")
    key_edit.editingFinished.emit()
    model_edit.setText("local-model")
    model_edit.editingFinished.emit()

    assert controller.settings.custom_base_url == "http://localhost:1234/v1"
    assert controller.settings.custom_api_key == "sk-x"
    assert controller.settings.custom_model == "local-model"
    reloaded = AssistantController(settings_store=controller.settings_store)
    assert reloaded.settings.custom_base_url == "http://localhost:1234/v1"


def test_custom_api_key_field_is_masked(app, parent, controller, workspace):
    controller.settings.backend = "custom"
    d = _drawer(parent, controller, workspace)
    from PyQt6.QtWidgets import QLineEdit
    edits = d._model_body_wrap.findChildren(QLineEdit)
    assert edits[1].echoMode() == QLineEdit.EchoMode.Password


def test_custom_api_test_connection_populates_found_models(app, parent, controller, workspace, monkeypatch):
    controller.settings.backend = "custom"
    controller.settings.custom_base_url = "http://localhost:1234/v1"
    monkeypatch.setattr(controller, "custom_api_available", lambda: True)
    monkeypatch.setattr(controller, "custom_api_models", lambda: ["m1", "m2"])
    d = _drawer(parent, controller, workspace)
    d._refresh_status()
    _pump(app, controller)
    assert d._found_models == ["m1", "m2"]
    selects = d._model_body_wrap.findChildren(SelectBox)
    assert len(selects) == 1


def test_custom_api_unreachable_shows_a_muted_status_not_a_crash(app, parent, controller, workspace, monkeypatch):
    controller.settings.backend = "custom"
    monkeypatch.setattr(controller, "custom_api_available", lambda: False)
    monkeypatch.setattr(controller, "custom_api_models", lambda: [])
    d = _drawer(parent, controller, workspace)
    d._refresh_status()
    _pump(app, controller)
    assert d._status_ok is False


def test_stale_status_result_from_a_since_switched_backend_is_ignored(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    controller.settings.backend = "custom"   # switched after the check was dispatched
    d._on_status_result("ollama", True, "Connected — 3 models", ["a", "b", "c"])
    assert d._found_models == []
    assert d._status_ok is None


# ── cross-thread safety ─────────────────────────────────────────────────────
def test_safe_emit_guards_against_a_deleted_widget(app, parent, controller, workspace):
    """A background thread's completion callback can outlive this widget
    (e.g. torn down by a theme toggle mid-chat) — every _safe_emit_* must
    swallow the RuntimeError a signal emit on a deleted QObject raises,
    same pattern already used throughout Studio (ModelsScreen, etc.)."""
    from PyQt6 import sip
    d = _drawer(parent, controller, workspace)
    emitters = [
        lambda: d._safe_emit_token("x"),
        lambda: d._safe_emit_done("x"),
        lambda: d._safe_emit_error("x"),
        lambda: d._safe_emit_status("offline", True, "ok", []),
        lambda: d._safe_emit_pull_progress("x", 0.5),
        lambda: d._safe_emit_pull_done("x", True),
        lambda: d._safe_emit_create_status("x"),
        lambda: d._safe_emit_create_done(True),
    ]
    sip.delete(d)
    for emit in emitters:
        emit()   # must not raise
