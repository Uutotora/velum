"""Tests for the newly-interactive atoms in studio/components.py (Slider,
Stepper) — the rest of the UI kit stays presentational or was already
covered indirectly by screen-wiring tests. Offscreen Qt, no napari/torch.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
components = pytest.importorskip("studio.components")

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from studio import theme


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _press(widget, x, y=7):
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(x, y), Qt.MouseButton.LeftButton,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    widget.mousePressEvent(ev)


def _drag(widget, x, y=7):
    ev = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(x, y), Qt.MouseButton.NoButton,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    widget.mouseMoveEvent(ev)


# ── Slider ───────────────────────────────────────────────────────────────────
def test_slider_default_value_and_clamping(app):
    s = components.Slider(theme.DARK, 0.5)
    assert s.value() == 0.5
    s.set_value(1.5)
    assert s.value() == 1.0
    s.set_value(-1.0)
    assert s.value() == 0.0


def test_slider_click_sets_value_and_emits(app):
    s = components.Slider(theme.DARK, 0.0)
    s.resize(100, 14)
    seen = []
    s.changed.connect(seen.append)
    _press(s, 25)  # 25/100 -> 0.25
    assert s.value() == pytest.approx(0.25, abs=0.02)
    assert seen and seen[-1] == pytest.approx(0.25, abs=0.02)


def test_slider_drag_updates_value(app):
    s = components.Slider(theme.DARK, 0.0)
    s.resize(100, 14)
    _press(s, 10)
    _drag(s, 90)
    assert s.value() == pytest.approx(0.9, abs=0.02)


def test_slider_set_value_without_emit_stays_silent(app):
    s = components.Slider(theme.DARK, 0.2)
    seen = []
    s.changed.connect(seen.append)
    s.set_value(0.8)
    assert s.value() == 0.8
    assert seen == []


# ── Stepper ──────────────────────────────────────────────────────────────────
def test_stepper_default_value_and_display(app):
    st = components.Stepper(32, theme.DARK)
    assert st.value() == 32
    assert st._val_label.text() == "32"


def test_stepper_plus_minus_buttons_change_value_and_emit(app):
    st = components.Stepper(10, theme.DARK, step=2, minimum=0, maximum=100)
    seen = []
    st.changed.connect(seen.append)
    st._plus.click()
    assert st.value() == 12
    st._minus.click()
    st._minus.click()
    assert st.value() == 8
    assert seen == [12, 10, 8]


def test_stepper_clamps_to_bounds(app):
    st = components.Stepper(1, theme.DARK, step=1, minimum=0, maximum=2)
    st._plus.click()
    st._plus.click()
    st._plus.click()
    assert st.value() == 2  # clamped at maximum
    st._minus.click()
    st._minus.click()
    st._minus.click()
    assert st.value() == 0  # clamped at minimum


def test_stepper_decimals_and_suffix_format_display(app):
    st = components.Stepper(0.8, theme.DARK, step=0.05, minimum=0, maximum=1,
                            decimals=2, suffix=" iou")
    assert st._val_label.text() == "0.80 iou"
    st._plus.click()
    assert st._val_label.text() == "0.85 iou"


def test_stepper_set_value_without_emit_stays_silent(app):
    st = components.Stepper(5, theme.DARK)
    seen = []
    st.changed.connect(seen.append)
    st.set_value(9)
    assert st.value() == 9
    assert seen == []
