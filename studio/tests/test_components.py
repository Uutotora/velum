"""Tests for the newly-interactive atoms in studio/components.py (Slider,
Stepper) — the rest of the UI kit stays presentational or was already
covered indirectly by screen-wiring tests. Offscreen Qt, no napari/torch.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
components = pytest.importorskip("studio.components")

from PyQt6.QtCore import QPoint, QPointF, QPropertyAnimation, QVariantAnimation, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

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


# ── label() ──────────────────────────────────────────────────────────────────
def test_label_background_is_explicitly_transparent_inside_a_styled_frame(app):
    """Regression test, from a real bug reported against the actual running
    app (not offscreen): label()'s own per-instance stylesheet (color/size/
    weight) didn't mention `background`, relying on the app-wide QLabel{
    background:transparent} cascade rule (theme.build_qss) -- reliable for
    a label with no styled ancestor, but a label nested inside a QFrame
    that has its OWN qualified background-setting stylesheet (any
    #ObjectName-styled card, any scrim dialog -- overlays.Toast is exactly
    this shape) resolved its own background via the app-wide QWidget{
    background:<bg>} rule instead of the more-specific QLabel one, painting
    an opaque, wrong-coloured box instead of staying transparent over the
    card's own surface. Reproduced in a minimal isolated repro before this
    fix (a bare QLabel directly in a styled QWidget was fine; the identical
    label nested inside a QFrame with its own qualified stylesheet was not)
    to confirm this is about the *styled-ancestor* context, not label()
    itself in isolation.
    """
    import time
    from PyQt6.QtWidgets import QFrame, QHBoxLayout, QWidget

    t = theme.DARK
    app.setStyleSheet(theme.build_qss(t))
    try:
        parent = QWidget()
        parent.resize(400, 200)
        parent.setStyleSheet(f"background:{t['bg']};")
        parent.show()
        card = QFrame(parent)
        card.setObjectName("RegressionTestCard")
        card.setStyleSheet(
            f"QFrame#RegressionTestCard{{background:{t['surface']}; border-radius:8px;}}")
        row = QHBoxLayout(card)
        lbl = components.label("Hello", 13, t["text"], 600)
        row.addWidget(lbl)
        card.move(10, 10)
        card.adjustSize()
        card.show()
        for _ in range(15):
            app.processEvents()
            time.sleep(0.01)

        img = parent.grab().toImage()
        pt = lbl.mapTo(parent, lbl.rect().topLeft())
        sample = img.pixelColor(pt.x(), pt.y())
        assert sample.name() == t["surface"], (
            f"label sampled {sample.name()!r} at its own top-left corner, expected "
            f"the card's own surface {t['surface']!r} to show through it")
    finally:
        app.setStyleSheet("")  # process-wide QApplication singleton -- don't leak


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


# ── SmoothScrollArea ─────────────────────────────────────────────────────────
def _wheel(widget, angle_y: int, pixel_y: int = 0) -> None:
    ev = QWheelEvent(
        QPointF(10, 10), QPointF(10, 10), QPoint(0, pixel_y), QPoint(0, angle_y),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False)
    widget.wheelEvent(ev)


def _tall_scroll_area():
    inner = QWidget()
    lay = QVBoxLayout(inner)
    for i in range(200):
        lay.addWidget(QLabel(f"row {i}"))
    sa = components.SmoothScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(inner)
    sa.resize(300, 200)
    sa.show()
    return sa


def test_smooth_scroll_area_animates_a_discrete_wheel_notch(app):
    """A traditional notched mouse wheel (angleDelta only, no pixelDelta)
    must not jump the scrollbar instantly -- docs/velum/BACKLOG.md's
    "Projects tab v2" entry. Right after the event the bar must still be at
    its starting value (an instant jump would already be at the target the
    moment wheelEvent() returns), and the animation actually queued for it
    must be configured to land on the right target.

    Deliberately does not wait for the animation to actually finish and
    assert on the bar's *settled* value: Qt's shared animation-driver timer
    is process-wide, and another test module (test_motion.py) starts several
    short-lived QPropertyAnimations of its own without ever pumping the
    event loop afterwards to let them complete -- confirmed by bisection
    (`pytest test_motion.py test_components.py` reproduces a stuck-at-0
    failure that neither file alone does) to leave the shared timer unable
    to advance *new* animations for the rest of the process, no matter how
    long a test then waits. Asserting on the configured animation object
    itself (mirroring test_motion.py's own test_slide_in_returns_an_
    animation_with_the_right_start_and_end) verifies the same thing --the
    right eased step was set up, not an instant jump-- without depending on
    that shared, apparently-stateful timer ever actually firing again.
    """
    sa = _tall_scroll_area()
    app.processEvents()
    bar = sa.verticalScrollBar()
    assert bar.maximum() > 0
    start = bar.value()

    _wheel(sa, angle_y=-120)

    assert bar.value() == start, "must not jump instantly -- it should still be animating"
    anim = bar._smooth_scroll_anim
    assert isinstance(anim, QPropertyAnimation)
    assert anim.propertyName() == b"value"
    assert anim.startValue() == start
    assert anim.endValue() == start + components.SmoothScrollArea._STEP_PX
    assert anim.state() == QPropertyAnimation.State.Running


def test_smooth_scroll_area_leaves_trackpad_pixel_delta_untouched(app):
    """A trackpad's pixelDelta is already smooth -- Qt's own default handling
    must run unmodified (synchronously, no animation), not the eased step for
    a discrete wheel -- double-applying both would make trackpad scrolling
    feel wrong, not right."""
    sa = _tall_scroll_area()
    app.processEvents()
    bar = sa.verticalScrollBar()
    start = bar.value()

    _wheel(sa, angle_y=-120, pixel_y=-40)
    assert bar.value() != start  # Qt's own default pixelDelta handling took it, synchronously
    assert not hasattr(bar, "_smooth_scroll_anim")


def test_smooth_scroll_area_noop_when_nothing_to_scroll(app):
    """No scrollable range (content shorter than the viewport) must fall
    through to Qt's default (a no-op / propagate-to-parent), not spin up an
    animation to nowhere."""
    sa = components.SmoothScrollArea()
    inner = QWidget()
    QVBoxLayout(inner).addWidget(QLabel("one short row"))
    sa.setWidgetResizable(True)
    sa.setWidget(inner)
    sa.resize(400, 400)
    sa.show()
    app.processEvents()
    bar = sa.verticalScrollBar()
    assert bar.maximum() <= bar.minimum()

    _wheel(sa, angle_y=-120)
    assert not hasattr(bar, "_smooth_scroll_anim")


# ── WavingEmoji ──────────────────────────────────────────────────────────────
def test_waving_emoji_starts_at_rest(app):
    w = components.WavingEmoji(theme.DARK)
    assert w._angle == 0.0


def test_waving_emoji_play_starts_a_running_animation_back_to_zero(app):
    """Assert on the configured animation object right after play(), not on
    a settled end value -- same reasoning as this file's own
    SmoothScrollArea tests just above: the shared QApplication's
    animation-driver timer can be silently wedged by an unrelated,
    earlier-alphabetical test module's own unpumped animations."""
    w = components.WavingEmoji(theme.DARK)
    w.play()
    assert w._anim.state() == QVariantAnimation.State.Running
    assert w._anim.keyValueAt(0.0) == 0.0
    assert w._anim.keyValueAt(1.0) == 0.0


def test_waving_emoji_play_is_safe_to_call_again_mid_wave(app):
    """play() must restart cleanly, not layer a second animation on top --
    e.g. revisiting Home again before the previous wave finished."""
    w = components.WavingEmoji(theme.DARK)
    w.play()
    w.play()  # must not raise, must still be a single running animation
    assert w._anim.state() == QVariantAnimation.State.Running


def test_waving_emoji_paint_event_does_not_raise(app):
    """A basic render smoke test -- the custom paintEvent (font + rotation
    transform) must not crash at rest or mid-wave."""
    w = components.WavingEmoji(theme.DARK)
    w.resize(40, 43)
    w.show()
    w.grab()  # renders at rest (angle 0)
    w._set_angle(15.0)
    w.grab()  # renders mid-wave (a non-zero rotation)


# ── SwipeRow (swipe-left-to-delete) ───────────────────────────────────────────
def _swipe_release(widget, x, y=7):
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(x, y), Qt.MouseButton.LeftButton,
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    widget.mouseReleaseEvent(ev)


def _swipe_row(app):
    from studio.components import SwipeRow
    seen = {"click": 0, "delete": 0}
    content = QLabel("row")
    row = SwipeRow(content, theme.DARK,
                   on_click=lambda: seen.__setitem__("click", seen["click"] + 1),
                   on_delete=lambda: seen.__setitem__("delete", seen["delete"] + 1),
                   reveal=80)
    row.resize(240, 46)
    return row, seen


def test_swiperow_tap_selects_not_deletes(app):
    row, seen = _swipe_row(app)
    _press(row, 100)
    _swipe_release(row, 100)  # no movement -> a tap
    assert seen == {"click": 1, "delete": 0}


def test_swiperow_small_wobble_still_taps(app):
    """A tap that drifts a few px left -- normal for a trackpad/mouse -- must
    still select, not get swallowed as a swipe. Regression for the reported
    "the image row is hard to select / won't select": the old 4px threshold
    turned any small drift into a swipe and dropped the click."""
    row, seen = _swipe_row(app)
    _press(row, 100)
    _drag(row, 94)          # 6px of leftward drift, under the tap slop
    _swipe_release(row, 94)
    assert seen == {"click": 1, "delete": 0}


def test_swiperow_full_swipe_left_deletes(app):
    row, seen = _swipe_row(app)
    _press(row, 200)
    _drag(row, 130)      # dragged 70px left, past the ~44px commit threshold
    _swipe_release(row, 130)
    assert seen["delete"] == 1
    assert seen["click"] == 0


def test_swiperow_short_swipe_springs_back_no_delete(app):
    row, seen = _swipe_row(app)
    _press(row, 200)
    _drag(row, 185)      # only 15px -- past the tap wobble but under commit
    _swipe_release(row, 185)
    assert seen["delete"] == 0
    assert seen["click"] == 0  # it was a swipe, just not far enough


def test_swiperow_offset_is_clamped_to_reveal_width(app):
    row, _ = _swipe_row(app)
    _press(row, 200)
    _drag(row, -400)     # yank way past the reveal width
    assert row._offset == -80  # clamped to reveal, not unbounded


# ── FieldRow / GroupLabel labels must be transparent (no bg-box border) ────────
def test_fieldrow_label_background_is_transparent(app):
    """A FieldRow's name label sits on a panel toned `inset`/`surface`; without
    an explicit transparent background it paints an opaque `bg`-coloured box
    whose edges read as a faint border ruled around the text (real report:
    "линии... как будто границы расчерчены")."""
    from PyQt6.QtWidgets import QLabel
    fr = components.FieldRow("opacity", QLabel("ctrl"), theme.DARK)
    name_lbl = next(l for l in fr.findChildren(QLabel) if l.text() == "opacity")
    assert "background:transparent" in name_lbl.styleSheet().replace(" ", "")


def test_grouplabel_background_is_transparent(app):
    gl = components.GroupLabel("engine", theme.DARK)
    assert "background:transparent" in gl.styleSheet().replace(" ", "")


# ── _NavItem ampersand mnemonic ───────────────────────────────────────────────
def test_navitem_ampersand_label_is_not_swallowed_as_mnemonic(app):
    """QToolButton treats a single '&' as a keyboard-mnemonic marker and hides
    it (underlining the next char), so "Models & Train" rendered as
    "Models_Train". _NavItem must escape it so the literal ampersand shows."""
    item = components._NavItem("train", "models", "Models & Train", theme.DARK)
    assert item.text() == "   Models && Train"   # && is Qt's escaped literal '&'
    # displayed form drops the escaping and keeps the real ampersand
    assert item.text().replace("&&", "&").strip() == "Models & Train"
