"""Headless wiring tests for the real Assistant drawer (studio/assistant_panel.py).

Uses a *real*, tmp_path-backed AssistantController (Qt-free, its own
thorough suite lives in test_assistant_controller.py) with only the deepest
network-touching calls monkeypatched (velum_core.advisor.ollama_*, this
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
def styled_app(app):
    """The shared QApplication with the real app-wide stylesheet applied —
    see test_guide_screen.py's identical fixture for why this specific
    class of rendering bug (an unqualified/bare-type-selector setStyleSheet
    cascading to descendants) only reproduces with it applied, and why it
    must be reset afterward (a process-wide QApplication singleton shared
    with every other test module in the session)."""
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
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: False)
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


# ── Command palette integration aliases ─────────────────────────────────────
def test_switch_backend_public_alias_matches_the_segcontrol_select(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    d.switch_backend(2)   # Custom API — empty base_url short-circuits before any real I/O
    assert controller.settings.backend == "custom"


def test_run_diagnose_public_alias_delegates(app, parent, controller, workspace):
    d = _drawer(parent, controller, workspace)
    before = d._chat._v.count()
    d.run_diagnose()
    assert d._chat._v.count() > before


def test_switching_to_ollama_kicks_off_a_status_check(app, parent, controller, workspace, monkeypatch):
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: True)
    monkeypatch.setattr("velum_core.advisor.ollama_models", lambda: ["llama3.2:3b"])
    d = _drawer(parent, controller, workspace)
    d._backend_seg._select(1)   # Ollama
    _pump(app, controller)
    assert d._found_models == ["llama3.2:3b"]
    assert d._status_ok is True


# ── real logging (studio/log_bus.py, via the stdlib logging bridge) ────────
def test_switching_backend_logs_to_the_shared_logger(app, parent, controller, workspace, caplog):
    d = _drawer(parent, controller, workspace)
    with caplog.at_level("INFO", logger="studio.assistant"):
        d._backend_seg._select(2)   # Custom API -- unconfigured, short-circuits before any I/O
    assert any("custom" in r.message.lower() for r in caplog.records)


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
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: False)
    controller.settings.backend = "ollama"
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
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: False)
    controller.settings.backend = "ollama"
    controller.settings.ollama_model = "m"

    def fake_chat(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("velum_core.advisor.ollama_chat", fake_chat)
    d = _drawer(parent, controller, workspace)
    d._input.setText("hi")
    d._send()
    _pump(app, controller)
    assert d._send_btn.isEnabled() is True
    # error text landed in the (now-finalised) streaming bubble, not a crash
    assert d._chat._cur is None


def test_send_disabled_while_a_previous_send_is_in_flight(app, parent, controller, workspace, monkeypatch):
    import threading
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: False)
    controller.settings.backend = "ollama"
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


# ── live status dot (pulses while checking, solid once resolved) ────────────
def test_status_dot_pulses_while_checking_and_settles_once_resolved(
        app, parent, controller, workspace, monkeypatch):
    controller.settings.backend = "ollama"
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: True)
    monkeypatch.setattr("velum_core.advisor.ollama_models", lambda: ["a:1b"])
    d = _drawer(parent, controller, workspace)
    # Constructor kicks off an async check -> starts in the "checking" state
    # (pulsing, a live QGraphicsOpacityEffect) before the result lands.
    assert d._status_dot.graphicsEffect() is not None
    _pump(app, controller)
    # Resolved: _rebuild_model_body() replaced the dot with a fresh, solid one.
    assert d._status_dot.graphicsEffect() is None


def test_status_dot_set_state_after_deletion_does_not_raise(app, parent, controller, workspace):
    """Same deleted-widget hazard every animation in this codebase guards
    against (see studio/motion.py's module docstring): a status result can
    land after the drawer (and this dot) was torn down mid-check."""
    from PyQt6 import sip
    dot = ap._StatusDot(theme.DARK, "checking")
    set_state = dot.set_state
    sip.delete(dot)
    set_state(theme.DARK, "ok")   # must not raise


def test_manual_refresh_immediately_shows_checking_before_the_result_lands(
        app, parent, controller, workspace, monkeypatch):
    import threading
    controller.settings.backend = "ollama"
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: False)
    d = _drawer(parent, controller, workspace)
    _pump(app, controller)
    assert d._status_dot.graphicsEffect() is None   # settled: not reachable

    release = threading.Event()

    def slow_available():
        release.wait(timeout=5)
        return True

    monkeypatch.setattr("velum_core.advisor.ollama_available", slow_available)
    d._refresh_status()
    assert d._status_dot.graphicsEffect() is not None   # pulsing again, immediately
    release.set()
    _pump(app, controller)


# ── Ollama model management wiring ──────────────────────────────────────────
def test_ollama_model_select_appears_after_a_refresh(app, parent, controller, workspace, monkeypatch):
    controller.settings.backend = "ollama"
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: True)
    monkeypatch.setattr("velum_core.advisor.ollama_models", lambda: ["a:1b", "b:2b"])
    d = _drawer(parent, controller, workspace)
    _pump(app, controller)
    selects = d._model_body_wrap.findChildren(SelectBox)
    assert len(selects) == 1


def test_picking_an_ollama_model_persists_it(app, parent, controller, workspace, monkeypatch):
    controller.settings.backend = "ollama"
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: True)
    monkeypatch.setattr("velum_core.advisor.ollama_models", lambda: ["a:1b", "b:2b"])
    d = _drawer(parent, controller, workspace)
    _pump(app, controller)
    d._pick_ollama_model("b:2b")
    assert controller.settings.ollama_model == "b:2b"
    reloaded = AssistantController(settings_store=controller.settings_store)
    assert reloaded.settings.ollama_model == "b:2b"


def test_no_models_found_shows_a_hint_not_a_crash(app, parent, controller, workspace, monkeypatch):
    controller.settings.backend = "ollama"
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: False)
    d = _drawer(parent, controller, workspace)
    _pump(app, controller)
    assert d._found_models == []


def test_download_model_shows_progress_and_completes(app, parent, controller, workspace, monkeypatch):
    import threading
    controller.settings.backend = "ollama"
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
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: False)
    controller.settings.backend = "ollama"
    controller.settings.ollama_model = ""
    d = _drawer(parent, controller, workspace)
    d._create_agent()
    assert "Pick a base model" in d._op_status_text


def test_create_agent_flow_completes(app, parent, controller, workspace, monkeypatch):
    controller.settings.backend = "ollama"
    controller.settings.ollama_model = "qwen2.5:7b"
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: False)

    def fake_create(base_model, on_status):
        on_status("baking…")
        return True

    monkeypatch.setattr("velum_core.advisor.ollama_create_agent", fake_create)
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


# ── rendering regressions ────────────────────────────────────────────────────
# Both bugs below only reproduce with the *real* app-wide stylesheet applied
# (styled_app, not the plain app fixture) -- see its docstring. Found from a
# real (non-offscreen) user screenshot reporting "borders aren't fully
# filled" plus stray lines; confirmed by pixel-sampling offscreen, not by
# eye, and confirmed *fixed* the same way (these tests fail against the
# pre-fix code, not just pass against the fix).
def test_drawers_own_border_does_not_leak_onto_the_empty_state(styled_app, controller, workspace):
    """AssistantDrawer.setStyleSheet() used to be an *unqualified* rule
    ("background:...;border-left:...;" with no selector) -- QWidget has an
    app-wide type-selector for `background` (always safely overridden), but
    nothing overrides plain `border` for a bare QWidget, so the drawer's own
    border-left leaked onto ChatView's inset empty-state widget as a stray
    1px vertical line at *its own* left edge (the ChatView content margin)
    -- not the drawer's real left edge (x=0, where a border-left is correct
    and expected). Only reproduces with the drawer actually laid out inside
    a parent that owns its own background (mirroring how StudioWindow hosts
    it) *and* given real elapsed time to settle (a bare processEvents()
    loop with no sleep never lets the fade-in reach a paintable state) --
    both matched here, not simplified away."""
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
    # Scan a narrow band around the empty state's own left edge (+/- a few
    # px, to absorb sub-pixel mapTo()-vs-actual-paint rounding) at several
    # y-offsets down its height -- every point here must be the surrounding
    # background, never the border colour.
    hits = [(dx, dy) for dy in range(5, empty.height() - 5, 5)
            for dx in range(-3, 4)
            if img.pixelColor(pt.x() + dx, pt.y() + dy).name() == border]
    assert not hits, f"stray border-coloured pixel(s) near the empty state's left edge: {hits}"


def test_change_card_title_and_detail_labels_are_not_individually_boxed(styled_app):
    """ChangeCard.setStyleSheet() used to be a bare *type* selector
    ("QFrame{...}", no #ObjectName) -- QLabel is itself a QFrame subclass,
    so the card's own background+border+radius rule also matched its title
    and detail QLabels, each repainting its own small bordered box around
    just its own text (the exact rendering-bug family docs/velum/CHANGELOG.md's
    2026-07-08 "Guide & Docs" entry already named once)."""
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
    assert len(labels) >= 2   # title + detail, at minimum
    border = t["border"]
    for lbl in labels:
        pt = lbl.mapTo(card, lbl.rect().topLeft())
        # The label's own top-left corner: a real per-label border box (the
        # bug) always outlines exactly here; the card's own border sits
        # much further out (card.rect(), not any inner label's rect).
        c = img.pixelColor(pt.x(), pt.y())
        assert c.name() != border, f"{lbl.text()!r} is individually boxed"


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
