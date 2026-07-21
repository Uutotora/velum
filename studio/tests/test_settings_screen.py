"""Headless wiring tests for the Settings screen (studio/settings_screen.py).

Real, tmp_path-backed AssistantController (its Qt-free logic is covered in
test_assistant_controller.py); only the deepest network/model calls
(velum_core.advisor.ollama_*) are monkeypatched. Follows the same
"real controller, fake ML seam" convention as test_assistant_panel.py.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import pytest

pytest.importorskip("PyQt6")
ss = pytest.importorskip("studio.settings_screen")

from PyQt6.QtWidgets import QApplication, QLineEdit

from studio import theme
from studio.assistant_controller import AssistantController, PROVIDERS


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(tmp_path):
    return AssistantController(storage_dir=tmp_path / "assistant_storage")


@pytest.fixture
def toasts():
    return []


@pytest.fixture
def screen(app, controller, toasts):
    return ss.SettingsScreen(theme.DARK, controller, on_toast=lambda *a: toasts.append(a))


def _pump(app, controller, timeout=5):
    t0 = time.monotonic()
    while controller.model_op_busy() and time.monotonic() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(5):
        app.processEvents()


# ── construction / sections ───────────────────────────────────────────────────
def test_screen_builds_all_three_sections(app, screen):
    for idx in (0, 1, 2):
        screen._set_section(idx)   # AI · Compute · About — none must raise
    assert screen._section == 2


def test_ai_section_opens_the_active_provider_by_default(app, screen):
    assert screen._section == 0
    assert screen._open_provider == "offline"


# ── choosing a provider ───────────────────────────────────────────────────────
def test_use_provider_sets_active_and_persists(app, controller, screen, toasts):
    screen._use_provider("openai")
    assert controller.settings.active == "openai"
    reloaded = AssistantController(settings_store=controller.settings_store)
    assert reloaded.settings.active == "openai"
    assert toasts   # a toast announced the change


def test_toggle_provider_expands_and_collapses(app, screen):
    screen._toggle_provider("groq")
    assert screen._open_provider == "groq"
    screen._toggle_provider("groq")
    assert screen._open_provider is None


def test_focus_provider_jumps_to_ai_section_and_expands(app, screen):
    screen._set_section(1)   # go to Compute
    screen.focus_provider("openrouter")
    assert screen._section == 0
    assert screen._open_provider == "openrouter"


# ── OpenAI-compatible fields persist per provider ─────────────────────────────
def test_openai_key_and_model_fields_persist_for_that_provider(app, controller, screen):
    screen.focus_provider("openai")
    edits = screen.findChildren(QLineEdit)
    # key field (password) + model field
    key_edit = next(e for e in edits if e.echoMode() == QLineEdit.EchoMode.Password)
    key_edit.setText("sk-secret")
    key_edit.editingFinished.emit()
    assert controller.key_for("openai") == "sk-secret"
    reloaded = AssistantController(settings_store=controller.settings_store)
    assert reloaded.key_for("openai") == "sk-secret"


def test_custom_provider_exposes_an_editable_base_url(app, controller, screen):
    screen.focus_provider("custom")
    edits = screen.findChildren(QLineEdit)
    # First field is the base URL for the custom provider.
    url_edit = edits[0]
    url_edit.setText("http://localhost:9000/v1")
    url_edit.editingFinished.emit()
    assert controller.settings.custom_base_url == "http://localhost:9000/v1"


# ── Ollama: status check + model management ───────────────────────────────────
def test_opening_ollama_kicks_off_a_status_check(app, controller, monkeypatch):
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: True)
    monkeypatch.setattr("velum_core.advisor.ollama_models", lambda: ["llama3.2:3b"])
    screen = ss.SettingsScreen(theme.DARK, controller)
    screen.focus_provider("ollama")
    _pump(app, controller)
    ok, _msg, models = screen._status.get("ollama", (None, "", []))
    assert ok is True and models == ["llama3.2:3b"]


def test_download_model_shows_progress_and_completes(app, controller, monkeypatch):
    import threading
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: False)
    started = threading.Event()
    release = threading.Event()

    def fake_pull(name, on_progress):
        on_progress("downloading", 0.5)
        started.set()
        release.wait(timeout=5)
        on_progress("done", 1.0)
        return True

    monkeypatch.setattr("velum_core.advisor.ollama_pull", fake_pull)
    screen = ss.SettingsScreen(theme.DARK, controller)
    screen.focus_provider("ollama")
    _pump(app, controller)
    screen._download_model("llama3.2:3b")
    started.wait(timeout=5)
    assert not screen._op_progress.isHidden()
    release.set()
    _pump(app, controller)
    assert "ready" in screen._op_status_text


def test_create_agent_requires_a_base_model(app, controller, monkeypatch):
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: False)
    controller.settings.ollama_model = ""
    screen = ss.SettingsScreen(theme.DARK, controller)
    screen.focus_provider("ollama")
    _pump(app, controller)
    screen._create_agent()
    assert "Pick a base model" in screen._op_status_text


# ── cross-thread safety ───────────────────────────────────────────────────────
def test_safe_emit_guards_against_a_deleted_screen(app, controller):
    from PyQt6 import sip
    screen = ss.SettingsScreen(theme.DARK, controller)
    emitters = [
        lambda: screen._safe_emit_status("ollama", True, "ok", []),
        lambda: screen._safe_emit_pull_progress("x", 0.5),
        lambda: screen._safe_emit_pull_done("x", True),
        lambda: screen._safe_emit_create_status("x"),
        lambda: screen._safe_emit_create_done(True),
    ]
    sip.delete(screen)
    for emit in emitters:
        emit()   # must not raise


def test_provider_registry_renders_a_card_per_provider(app, screen):
    # Every provider id can be expanded (collapse whatever's open first).
    for spec in PROVIDERS:
        screen._open_provider = None
        screen._toggle_provider(spec.id)   # expand — must build without raising
        assert screen._open_provider == spec.id
