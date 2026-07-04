"""Small Qt UX helpers shared across the app."""
from __future__ import annotations

from PyQt6.QtCore import QObject, QEvent
from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox


class WheelGuard(QObject):
    """Stop mouse-wheel scrolling from silently changing spin boxes / combos.

    Scrolling the panel should never nudge a value the user isn't editing.
    A spin box or combo only reacts to the wheel once it has keyboard focus
    (i.e. the user deliberately clicked into it).
    """

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and isinstance(
                obj, (QAbstractSpinBox, QComboBox)):
            if not obj.hasFocus():
                event.ignore()
                return True
        return super().eventFilter(obj, event)


def install_wheel_guard(app) -> WheelGuard:
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    return guard
