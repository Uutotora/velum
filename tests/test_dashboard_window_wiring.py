"""Wiring test: the shared DashboardWindow singleton, and the "Dashboard"
button each of Predict/Train/Assistant gained.

PyQt6-WebEngine is never installed in this sandbox (it's the separate,
heavier `tracking-ui` extra), so DashboardWindow's embedded-view branch is
never exercised here — only its documented fallback (a message + the
always-available "Open in browser" button), which is the realistic install
state for most users (Aim without the embed extra). Aim itself is also never
installed, so `ensure_dashboard_running` is monkeypatched rather than
launching a real `aim up`.

Skipped in the lightweight CI image (no PyQt6).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")
dashboard_window = pytest.importorskip("napari_app.widgets.dashboard_window")

from PyQt6.QtWidgets import QApplication


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _reset_singleton():
    dashboard_window._instance = None
    yield
    dashboard_window._instance = None


def test_get_dashboard_window_is_a_singleton(app):
    a = dashboard_window.get_dashboard_window()
    b = dashboard_window.get_dashboard_window()
    assert a is b


def test_no_webengine_falls_back_to_a_message_not_a_crash(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    assert w._view is None


# Shared fakes/helpers for the sections below. Real QWebEngineView
# construction needs Qt.AA_ShareOpenGLContexts set *before* the very first
# QApplication in the process (see napari_app/main.py's fix, and
# docs/CHANGELOG.md for how this was found) — a constraint this shared test
# process can't reliably guarantee relative to whichever other test file's
# `app` fixture happened to run first, so these tests drive
# DashboardWindow's own logic against a fake view/patched builder instead of
# ever constructing a real one.

class _FakeView:
    def __init__(self):
        self.urls = []

    def setUrl(self, url):
        self.urls.append(url)


def _fire_timers_immediately(monkeypatch):
    from PyQt6.QtCore import QTimer
    monkeypatch.setattr(QTimer, "singleShot", staticmethod(lambda ms, fn: fn()))


def _stub_build_embedded_view(monkeypatch, fake_view=None):
    """Replace DashboardWindow._build_embedded_view with one that assigns a
    fake view instead of constructing a real QWebEngineView."""
    view = fake_view if fake_view is not None else _FakeView()
    monkeypatch.setattr(
        dashboard_window.DashboardWindow, "_build_embedded_view",
        lambda self, layout: setattr(self, "_view", view))
    return view


# ── dynamic upgrade: PyQt6-WebEngine installed after this (singleton)
#    window's first construction must not require an app restart ──────────

def test_upgrades_from_fallback_to_embedded_view_once_available(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    assert w._view is None
    assert w._fallback is not None

    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: True)
    fake_view = _stub_build_embedded_view(monkeypatch)
    w._upgrade_to_embedded_view_if_possible()

    assert w._view is fake_view
    assert w._fallback is None


def test_open_dashboard_triggers_the_upgrade_automatically(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    assert w._view is None

    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: True)
    fake_view = _stub_build_embedded_view(monkeypatch)
    from napari_app.core import experiment_tracking as tracking
    monkeypatch.setattr(tracking, "ensure_dashboard_running", lambda: "http://127.0.0.1:1")
    w.open_dashboard()

    assert w._view is fake_view


def test_upgrade_is_a_noop_once_already_embedded(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: True)
    first_view = _stub_build_embedded_view(monkeypatch)
    w = dashboard_window.DashboardWindow()
    assert w._view is first_view

    build_calls = []
    monkeypatch.setattr(
        dashboard_window.DashboardWindow, "_build_embedded_view",
        lambda self, layout: build_calls.append(1))
    w._upgrade_to_embedded_view_if_possible()

    assert w._view is first_view   # unchanged
    assert build_calls == []       # _build_embedded_view not called again


# ── embedded-view load retry ─────────────────────────────────────────────────
#
# `aim up` spawns a real web server in a subprocess; the very first load
# attempt can race it still starting (confirmed against a real install: the
# first QWebEngineView.loadFinished fired False, a retry ~1s later fired
# True with the real Aim page's DOM actually populated). These tests drive
# the retry state machine directly against a fake view (no real
# QWebEngineView/Chromium involved) so they stay fast and deterministic;
# QTimer.singleShot is monkeypatched to fire immediately rather than really
# waiting `_LOAD_RETRY_MS`.


def test_embedded_view_retries_once_after_a_failed_load(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    _fire_timers_immediately(monkeypatch)
    fake_view = _FakeView()
    w._view = fake_view
    w._url = "http://127.0.0.1:12345"
    w._load_attempt = 0

    w._on_embedded_load_finished(False)

    assert w._load_attempt == 1
    assert len(fake_view.urls) == 1   # the retry re-set the URL
    assert fake_view.urls[0].toString() == "http://127.0.0.1:12345"


def test_embedded_view_does_not_retry_after_a_successful_load(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    _fire_timers_immediately(monkeypatch)
    fake_view = _FakeView()
    w._view = fake_view
    w._url = "http://127.0.0.1:12345"
    w._load_attempt = 0

    w._on_embedded_load_finished(True)

    assert w._load_attempt == 0
    assert fake_view.urls == []


def test_embedded_view_gives_up_after_max_attempts_with_a_status_message(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    _fire_timers_immediately(monkeypatch)
    w._view = _FakeView()
    w._url = "http://127.0.0.1:12345"
    w._load_attempt = 0

    for _ in range(dashboard_window._LOAD_MAX_ATTEMPTS):
        w._on_embedded_load_finished(False)

    assert w._load_attempt == dashboard_window._LOAD_MAX_ATTEMPTS
    assert "embedded load failed" in w._status.text()
    assert "Open in browser" in w._status.text()


def test_open_dashboard_resets_the_retry_counter(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    w._view = _FakeView()   # pretend webengine had been available
    w._load_attempt = 5

    from napari_app.core import experiment_tracking as tracking
    monkeypatch.setattr(tracking, "ensure_dashboard_running", lambda: "http://127.0.0.1:1")
    w.open_dashboard()

    assert w._load_attempt == 0


def test_open_dashboard_shows_the_url_in_the_status_label(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    from napari_app.core import experiment_tracking as tracking
    monkeypatch.setattr(tracking, "ensure_dashboard_running", lambda: "http://127.0.0.1:12345")
    w.open_dashboard()
    assert "12345" in w._status.text()
    assert w._url == "http://127.0.0.1:12345"


def test_browser_button_starts_disabled_and_enables_once_a_url_exists(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    assert w._browser_btn.isEnabled() is False   # nothing to open yet

    from napari_app.core import experiment_tracking as tracking
    monkeypatch.setattr(tracking, "ensure_dashboard_running", lambda: "http://127.0.0.1:12345")
    w.open_dashboard()
    assert w._browser_btn.isEnabled() is True


def test_open_dashboard_shows_the_error_when_aim_is_not_installed(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    from napari_app.core import experiment_tracking as tracking

    def boom():
        raise RuntimeError("Aim is not installed — run: pip install aim")

    monkeypatch.setattr(tracking, "ensure_dashboard_running", boom)
    w.open_dashboard()
    assert "not installed" in w._status.text()
    assert w._url is None
    assert w._browser_btn.isEnabled() is False   # stays disabled, not just silently inert


def test_browser_button_disables_again_after_a_failed_retry(app, monkeypatch):
    """A successful open followed by a failed retry (e.g. the dashboard
    process died) must re-disable the button rather than leaving it
    enabled with a now-stale URL."""
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    from napari_app.core import experiment_tracking as tracking

    monkeypatch.setattr(tracking, "ensure_dashboard_running", lambda: "http://127.0.0.1:1")
    w.open_dashboard()
    assert w._browser_btn.isEnabled() is True

    def boom():
        raise RuntimeError("Aim is not installed — run: pip install aim")

    monkeypatch.setattr(tracking, "ensure_dashboard_running", boom)
    w.open_dashboard()
    assert w._browser_btn.isEnabled() is False
    assert w._url is None


def test_open_in_browser_button_launches_the_system_browser(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    from napari_app.core import experiment_tracking as tracking
    monkeypatch.setattr(tracking, "ensure_dashboard_running", lambda: "http://127.0.0.1:9",)
    w.open_dashboard()

    opened = []
    monkeypatch.setattr(dashboard_window.webbrowser, "open", lambda url: opened.append(url))
    w._browser_btn.click()
    assert opened == ["http://127.0.0.1:9"]


def test_open_in_browser_before_any_url_does_nothing(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    opened = []
    monkeypatch.setattr(dashboard_window.webbrowser, "open", lambda url: opened.append(url))
    w._browser_btn.click()
    assert opened == []


def test_show_and_raise_calls_open_dashboard(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    called = []
    monkeypatch.setattr(w, "open_dashboard", lambda: called.append(True))
    w.show_and_raise()
    assert called == [True]
    w.close()


# ── each widget's own "Dashboard" button calls the shared singleton ─────────

def test_predict_widget_dashboard_button_opens_the_shared_window(app, monkeypatch):
    pw = pytest.importorskip("napari_app.widgets.predict_widget")
    w = pw.PredictWidget(MagicMock())
    monkeypatch.setattr(w, "_append_log", lambda *a, **k: None)
    called = []
    monkeypatch.setattr(dashboard_window, "get_dashboard_window",
                        lambda: MagicMock(show_and_raise=lambda: called.append(True)))
    w._open_dashboard()
    assert called == [True]
    w.close()


def test_train_widget_dashboard_button_opens_the_shared_window(app, monkeypatch):
    tw = pytest.importorskip("napari_app.widgets.train_widget")
    w = tw.TrainWidget(MagicMock())
    called = []
    monkeypatch.setattr(dashboard_window, "get_dashboard_window",
                        lambda: MagicMock(show_and_raise=lambda: called.append(True)))
    w._open_dashboard()
    assert called == [True]
    w.close()


def test_assistant_widget_dashboard_button_opens_the_shared_window(app, monkeypatch):
    aw = pytest.importorskip("napari_app.widgets.assistant_widget")
    predict_stub = MagicMock()
    w = aw.AssistantWidget(viewer=None, predict_widget=predict_stub)
    called = []
    monkeypatch.setattr(dashboard_window, "get_dashboard_window",
                        lambda: MagicMock(show_and_raise=lambda: called.append(True)))
    w._open_dashboard()
    assert called == [True]
    w.close()


def test_train_widget_replaced_the_old_local_history_box_with_the_dashboard(app):
    """The plain-text "Training history" box (STATE_MANAGER.load_history(),
    this session only) was replaced by a "Run history" card pointing at the
    Dashboard (every run, real comparison) -- guards against it silently
    coming back."""
    tw = pytest.importorskip("napari_app.widgets.train_widget")
    w = tw.TrainWidget(MagicMock())
    assert not hasattr(w, "history_box")
    assert not hasattr(w, "_refresh_history")
    w.close()
