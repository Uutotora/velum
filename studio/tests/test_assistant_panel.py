"""Headless wiring tests for the chat-only Assistant drawer
(studio/assistant_panel.py).

The drawer no longer holds any provider/model configuration — that moved to
the Settings screen (tests in test_settings_screen.py). What remains here is a
pure chat surface: a header (Diagnose · Settings · Close), a one-line provider
status strip that jumps to Settings, the chat, and the input row.

Uses a *real*, tmp_path-backed AssistantController (Qt-free, its own thorough
suite lives in test_assistant_controller.py) with only the deepest
network-touching calls monkeypatched (velum_core.advisor.ollama_*). The
Segment workspace side is a small fake (_FakeWorkspace).
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
def styled_app(app):
    """The shared QApplication with the real app-wide stylesheet applied —
    see test_guide_screen.py's identical fixture for why this specific class
    of rendering bug only reproduces with it applied, and why it must be
    reset afterward (a process-wide singleton shared across the session)."""
    app.setStyleSheet(theme.build_qss(theme.DARK))
    yield app
    app.setStyleSheet("")


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


def _drawer(parent, controller, workspace, on_open_settings=None):
    d = ap.AssistantDrawer(parent, theme.DARK, controller, workspace,
                           on_open_settings=on_open_settings)
    d.setParent(parent)
    return d


def _pump(app, controller, timeout=5):
    t0 = time.monotonic()
    while controller.chat_busy() and time.monotonic() - t0 < timeout:
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


# ── provider status strip ────────────────────────────────────────────────────
def test_status_strip_shows_offline_provider_by_default(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    assert "Offline" in d._status_lbl.text()


def test_status_strip_reflects_active_provider_and_model(app, parent, controller, workspace):
    controller.settings.active = "openai"
    controller.set_model("openai", "gpt-4o")
    controller.set_key("openai", "sk-x")
    d = _drawer(parent, controller, workspace)
    assert "OpenAI" in d._status_lbl.text()
    assert "gpt-4o" in d._status_lbl.text()


def test_status_strip_updates_on_show_after_a_settings_change(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    assert "Offline" in d._status_lbl.text()
    # A change made "in Settings" while the drawer is closed…
    controller.settings.active = "ollama"
    controller.settings.ollama_model = "llama3.2:3b"
    d._refresh_status_strip()   # what showEvent calls when the drawer reopens
    assert "Ollama" in d._status_lbl.text() and "llama3.2:3b" in d._status_lbl.text()


def test_settings_button_and_strip_call_open_settings(app, parent, controller, workspace):
    calls = []
    d = _drawer(parent, controller, workspace, on_open_settings=lambda: calls.append(1))
    d.open_settings()
    d._status_wrap.mouseReleaseEvent(None)
    assert len(calls) == 2


# ── command-palette aliases ──────────────────────────────────────────────────
def test_run_diagnose_public_alias_delegates(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    before = d._chat._v.count()
    d.run_diagnose()
    assert d._chat._v.count() > before


# ── real logging ─────────────────────────────────────────────────────────────
def test_chat_error_logs_a_warning(app, parent, controller, workspace, caplog):
    d = _drawer(parent, controller, workspace)
    with caplog.at_level("WARNING", logger="studio.assistant"):
        d._on_error("network unreachable")
    assert any(r.levelname == "WARNING" and "network unreachable" in r.message
               for r in caplog.records)


# ── diagnose ─────────────────────────────────────────────────────────────────
def test_diagnose_with_no_context_prompts_to_predict(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    before = d._chat._v.count()
    d._run_diagnose()
    assert d._chat._v.count() > before
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
    assert workspace.reran is False


# ── chat — offline provider ──────────────────────────────────────────────────
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


def test_send_ignores_empty_input(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    d._input.setText("   ")
    d._send()
    assert d._history == []


# ── chat — a connected (threaded) provider ───────────────────────────────────
def test_send_ollama_streams_into_the_chat(app, parent, controller, workspace, monkeypatch):
    controller.settings.active = "ollama"
    controller.settings.ollama_model = "qwen2.5:7b"

    def fake_chat(model, messages, on_token, stop=None, temperature=0.2):
        on_token("Try ")
        on_token("raising IoU.")
        return "Try raising IoU."

    monkeypatch.setattr("velum_core.advisor.ollama_chat", fake_chat)
    d = _drawer(parent, controller, workspace)
    d._input.setText("why merged?")
    d._send()
    assert d._send_btn.isEnabled() is False
    _pump(app, controller)
    assert d._send_btn.isEnabled() is True
    assert d._history[-1] == {"role": "assistant", "content": "Try raising IoU."}


def test_send_reports_errors_into_the_chat(app, parent, controller, workspace, monkeypatch):
    controller.settings.active = "ollama"
    controller.settings.ollama_model = "m"

    def fake_chat(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("velum_core.advisor.ollama_chat", fake_chat)
    d = _drawer(parent, controller, workspace)
    d._input.setText("hi")
    d._send()
    _pump(app, controller)
    assert d._send_btn.isEnabled() is True
    assert d._chat._cur is None


def test_send_disabled_while_a_previous_send_is_in_flight(app, parent, controller, workspace, monkeypatch):
    import threading
    controller.settings.active = "ollama"
    controller.settings.ollama_model = "m"
    release = threading.Event()

    def fake_chat(model, messages, on_token, stop=None, temperature=0.2):
        release.wait(timeout=5)
        return "done"

    monkeypatch.setattr("velum_core.advisor.ollama_chat", fake_chat)
    d = _drawer(parent, controller, workspace)
    d._input.setText("first")
    d._send()
    d._input.setText("second")
    d._send()   # must be a no-op — chat_busy() is True
    release.set()
    _pump(app, controller)
    assert d._history.count({"role": "user", "content": "first"}) == 1
    assert not any(m.get("content") == "second" for m in d._history)


# ── rendering regressions ────────────────────────────────────────────────────
def test_drawers_own_border_does_not_leak_onto_the_empty_state(styled_app, controller, workspace):
    """AssistantDrawer.setStyleSheet() is a *qualified* rule; regression guard
    that its border-left never leaks onto ChatView's inset empty-state widget
    as a stray 1px vertical line at its own left edge. Only reproduces with
    the real app-wide stylesheet applied and given time to settle."""
    import time as _time
    from PyQt6.QtWidgets import QVBoxLayout

    t = theme.DARK
    host = QWidget()
    host.setStyleSheet(f"background:{t['bg']};")
    host.resize(360, 860)
    lay = QVBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    d = ap.AssistantDrawer(host, t, controller, workspace)
    lay.addWidget(d)
    d.show()
    host.show()
    for _ in range(40):
        styled_app.processEvents()
        _time.sleep(0.01)

    img = d.grab().toImage()
    empty = d._chat._empty
    pt = empty.mapTo(d, empty.rect().topLeft())
    border = t["border"]
    hits = [(dx, dy) for dy in range(5, empty.height() - 5, 5)
            for dx in range(-3, 4)
            if img.pixelColor(pt.x() + dx, pt.y() + dy).name() == border]
    assert not hits, f"stray border-coloured pixel(s) near the empty state's left edge: {hits}"


def test_change_card_title_and_detail_labels_are_not_individually_boxed(styled_app):
    """ChangeCard uses a qualified #ObjectName selector; regression guard that
    its background+border rule never repaints its own title/detail QLabels as
    individually bordered boxes."""
    t = theme.DARK
    card = ap.ChangeCard(t, "Run a prediction first",
                         "Predict on an image, then diagnose the result here for tuning advice.",
                         {}, t["primary"], "", lambda c: None, lambda c: None)
    card.resize(320, card.sizeHint().height())
    card.show()
    for _ in range(10):
        styled_app.processEvents()
    img = card.grab().toImage()

    from PyQt6.QtWidgets import QLabel
    labels = [w for w in card.findChildren(QLabel) if w.text()]
    assert len(labels) >= 2
    border = t["border"]
    for lbl in labels:
        pt = lbl.mapTo(card, lbl.rect().topLeft())
        c = img.pixelColor(pt.x(), pt.y())
        assert c.name() != border, f"{lbl.text()!r} is individually boxed"


# ── cross-thread safety ─────────────────────────────────────────────────────
def test_safe_emit_guards_against_a_deleted_widget(app, parent, controller, workspace):
    """A background thread's completion callback can outlive this widget
    (e.g. torn down by a theme toggle mid-chat) — every _safe_emit_* must
    swallow the RuntimeError a signal emit on a deleted QObject raises."""
    from PyQt6 import sip
    d = _drawer(parent, controller, workspace)
    emitters = [
        lambda: d._safe_emit_token("x"),
        lambda: d._safe_emit_done("x"),
        lambda: d._safe_emit_error("x"),
    ]
    sip.delete(d)
    for emit in emitters:
        emit()   # must not raise
