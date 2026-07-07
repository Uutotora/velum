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


def test_open_dashboard_shows_the_url_in_the_status_label(app, monkeypatch):
    monkeypatch.setattr(dashboard_window, "_has_webengine", lambda: False)
    w = dashboard_window.DashboardWindow()
    from napari_app.core import experiment_tracking as tracking
    monkeypatch.setattr(tracking, "ensure_dashboard_running", lambda: "http://127.0.0.1:12345")
    w.open_dashboard()
    assert "12345" in w._status.text()
    assert w._url == "http://127.0.0.1:12345"


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
