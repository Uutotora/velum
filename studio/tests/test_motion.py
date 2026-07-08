"""Headless tests for studio/motion.py — fade_in and install_hover_lift.

Offscreen Qt, no napari/torch. The regression tests here reproduce a real
crash: a real user's app aborted (SIGABRT, confirmed from a macOS crash
report) after PyQt6 escalated an unhandled RuntimeError — raised when a
hover callback touched a QGraphicsDropShadowEffect/QPropertyAnimation whose
underlying C++ object had already been deleted — straight to abort(). See
motion.py's module docstring for the full mechanism.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
motion = pytest.importorskip("studio.motion")

from PyQt6 import sip
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QEnterEvent
from PyQt6.QtWidgets import QApplication, QFrame, QGraphicsDropShadowEffect, QGraphicsOpacityEffect


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _enter_event() -> QEnterEvent:
    return QEnterEvent(QPointF(0, 0), QPointF(0, 0), QPointF(0, 0))


# ── fade_in ──────────────────────────────────────────────────────────────────
def test_fade_in_installs_an_opacity_effect_and_returns_an_animation(app):
    w = QFrame()
    anim = motion.fade_in(w, duration=10)
    assert anim is not None
    assert isinstance(w.graphicsEffect(), QGraphicsOpacityEffect)


def test_safe_clear_effect_is_a_noop_on_a_deleted_widget(app):
    w = QFrame()
    motion.fade_in(w, duration=10)
    sip.delete(w)
    motion._safe_clear_effect(w)  # must not raise RuntimeError


# ── install_hover_lift ───────────────────────────────────────────────────────
def test_install_hover_lift_installs_a_shadow_effect(app):
    w = QFrame()
    effect = motion.install_hover_lift(w)
    assert isinstance(effect, QGraphicsDropShadowEffect)
    assert w.graphicsEffect() is effect


def test_hover_enter_and_leave_animate_without_raising(app):
    w = QFrame()
    motion.install_hover_lift(w, base=(14, 22, 3), hover=(22, 34, 6))
    w.enterEvent(_enter_event())
    w.leaveEvent(_enter_event())


def test_enter_event_after_widget_deleted_does_not_crash_the_process(app):
    """Regression test for the real crash: a stale hover closure firing
    after its widget (and effect) were torn down used to raise
    ``RuntimeError: wrapped C/C++ object ... has been deleted`` from inside
    an ``enterEvent`` override — fatal for the whole process under PyQt6,
    not just this call. Capture the bound method before deletion (mirroring
    how Qt's own C++ side would still hold a reference to the override) and
    confirm calling it now degrades quietly instead of raising.
    """
    w = QFrame()
    motion.install_hover_lift(w, base=(14, 22, 3), hover=(22, 34, 6))
    enter = w.enterEvent
    leave = w.leaveEvent
    sip.delete(w)
    enter(_enter_event())  # must not raise
    leave(_enter_event())  # must not raise


def test_leave_event_after_widget_deleted_does_not_crash_the_process(app):
    w = QFrame()
    motion.install_hover_lift(w)
    leave = w.leaveEvent
    sip.delete(w)
    leave(_enter_event())  # must not raise
