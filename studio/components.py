"""Velum — the static UI kit (design skeleton).

Presentational Qt widgets that reproduce the north-star mockup one-to-one:
chips, buttons, selects, toggles, sliders, steppers, segmented controls, stat
tiles, collapsible sections, and the navigation sidebar. **No business logic** —
this branch is a pure design skeleton; controls render state but don't act.
Interactivity that's part of the *design* (nav, theme toggle, opening the
command palette, expanding a section, flipping a toggle's look) is kept; real
functionality is wired later, tab by tab (see ``docs/velum/``).

Every widget takes a token dict from :mod:`studio.theme`, so it
renders in either theme. All constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import (
    Qt, QSize, QPoint, QPointF, QEasingCurve, QPropertyAnimation, QVariantAnimation, pyqtSignal,
)
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QToolButton,
    QPushButton, QSizePolicy, QGraphicsDropShadowEffect, QMenu, QScrollArea,
)
from PyQt6.QtGui import QColor, QAction, QFont, QPainter

from studio import icons
from studio import theme
from studio.motion import smooth_scroll_by


# ── primitives ───────────────────────────────────────────────────────────────
def hline(t: dict) -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background:{t['border']}; border:none;")
    return f


def bare_widget(layout=None) -> QWidget:
    """A plain ``QWidget``, explicitly transparent, for grouping a layout.

    A bare ``QWidget()`` with no stylesheet of its own still inherits the
    app-wide ``QWidget { background: <bg> }`` rule (``theme.build_qss``) and
    paints an *opaque* bg-coloured rectangle wherever it sits — invisible
    when it's directly on the page canvas, but a stray dark patch cut into
    any lighter card it's nested inside (confirmed by a pixel-level render
    test — this is what produced the banded rows in the Guide screen's
    engine-comparison table and shortcut list). Every plain grouping widget
    should go through this instead of a raw ``QWidget()`` so the mistake
    can't recur silently.
    """
    w = QWidget()
    w.setStyleSheet("background: transparent;")
    if layout is not None:
        w.setLayout(layout)
    return w


def soft_shadow(w: QWidget, blur: int = 16, alpha: int = 26, dy: int = 3) -> None:
    eff = QGraphicsDropShadowEffect(w)
    eff.setBlurRadius(blur)
    eff.setOffset(0, dy)
    eff.setColor(QColor(28, 42, 120, alpha))
    w.setGraphicsEffect(eff)


class SmoothScrollArea(QScrollArea):
    """A ``QScrollArea`` whose mouse-wheel scrolling is eased, not an instant
    per-notch jump.

    A trackpad already delivers smooth, high-resolution ``pixelDelta`` wheel
    events -- Qt's default handling of those is left completely untouched.
    A traditional notched mouse wheel only ever reports ``angleDelta``, and
    Qt's default handling of *that* steps the scrollbar straight to its new
    value with no easing, which is the one case that actually reads as a
    jarring jump rather than a continuous scroll -- this overrides
    ``wheelEvent`` for exactly that case and animates it instead (see
    ``motion.smooth_scroll_by``). Every screen built through this project's
    shared ``scroll()``/``_scroll()`` helpers gets this for free; construct
    it directly for a one-off scroll area.
    """

    _STEP_PX = 120  # ~3 row-heights per notch -- matches Qt's own default "3 lines" feel

    def wheelEvent(self, event) -> None:
        bar = self.verticalScrollBar()
        angle_y = event.angleDelta().y()
        if (not event.pixelDelta().isNull()) or angle_y == 0 \
                or bar is None or bar.maximum() <= bar.minimum():
            super().wheelEvent(event)
            return
        smooth_scroll_by(bar, -round(angle_y / 120 * self._STEP_PX))
        event.accept()


def label(text: str, size: float, color: str, weight: int = 400,
          spacing: float = 0.0) -> QLabel:
    lb = QLabel(text)
    ls = f"letter-spacing:{spacing}px;" if spacing else ""
    # background:transparent is set explicitly, not left to the app-wide
    # QLabel{background:transparent} cascade rule (theme.build_qss) -- that
    # rule is reliable for a QLabel with no instance stylesheet of its own,
    # but this helper always gives every label an instance-level
    # setStyleSheet() call (for color/size/weight), and a QLabel nested
    # inside a QFrame that has its own qualified background-setting
    # stylesheet (any scrim dialog, Toast, any #ObjectName-styled card) can
    # then resolve its *own* background to the app-wide QWidget{background:
    # <bg>} rule instead of the more-specific QLabel one -- confirmed by
    # reproducing a real user-reported bug (Toast's title/subtitle painting
    # an opaque `bg`-coloured box instead of staying transparent over the
    # card's own `surface`) in a minimal, isolated repro, and confirming
    # this exact one-line fix resolves it there before applying it here.
    lb.setStyleSheet(f"color:{color}; font-size:{size}px; font-weight:{weight}; "
                     f"background:transparent; {ls}")
    return lb


class _ElidingLabel(QLabel):
    """A ``QLabel`` that elides long text with "…" instead of forcing its
    parent layout wider — for any label showing unbounded-length *dynamic*
    text (a LoRA checkpoint's filename, a discovered ground-truth mask's
    name, an engine's registry label) inside a fixed-width panel. A plain
    ``QLabel``'s ``sizeHint``/``minimumSizeHint`` always reflect its *full*
    text, so one long value can silently blow out the whole 320px inspector
    panel's width — with no horizontal scrollbar to reach the overflow —
    confirmed by an offscreen width audit (``_val.sizeHint()`` exceeding the
    panel's available budget), not just by eye. ``text()``/``setText()``
    still round-trip the *real*, un-elided string.
    """

    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        super().setText(text)

    def text(self) -> str:
        return self._full_text

    def setText(self, text: str) -> None:
        self._full_text = text
        self._reelide()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._reelide()

    def _reelide(self) -> None:
        elided = self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideRight, max(self.width(), 1))
        super().setText(elided)

    def sizeHint(self) -> QSize:
        natural = super().sizeHint()
        return QSize(min(140, natural.width()), natural.height())

    def minimumSizeHint(self) -> QSize:
        return QSize(20, super().minimumSizeHint().height())


class Chip(QLabel):
    """Rounded pill/badge. ``kind`` selects the colour family."""

    def __init__(self, text: str, t: dict, kind: str = "default"):
        super().__init__(text)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        fg, bg, bd = {
            "default": (t["text_subtle"], t["surface2"], t["border"]),
            "primary": (t["primary"], t["primary_weak"], t["primary_line"]),
            "signal":  (t["signal"], t["signal_weak"], t["signal_line"]),
            "success": (t["success"], t["success_weak"], t["success_weak"]),
            "warning": (t["warning"], t["warning_weak"], t["warning_weak"]),
            "muted":   (t["text_muted"], "transparent", t["border"]),
        }.get(kind, (t["text_subtle"], t["surface2"], t["border"]))
        self.setStyleSheet(
            f"color:{fg}; background:{bg}; border:1px solid {bd};"
            f"border-radius:999px; padding:3px 10px; font-size:11px; font-weight:600;")


class Badge(QLabel):
    """Compact monospaced value badge."""

    def __init__(self, text: str, t: dict):
        super().__init__(text)
        self.setStyleSheet(
            f"color:{t['text_subtle']}; background:{t['surface2']}; border-radius:6px;"
            f"padding:2px 7px; font-size:11px; font-weight:600; font-family:{theme.MONO};")


class EngineChip(QFrame):
    """A pill chip with a leading colour dot — the mockup's engine badge on a
    project-card cover (``.chip.vchip`` + a ``.cd`` swatch dot), one dot hue
    per engine so CellSeg1/Cellpose/SAM2 stay visually distinct at a glance.
    A plain ``Chip`` has no way to embed this second colour, hence its own
    small atom rather than an option bolted onto ``Chip``.
    """

    def __init__(self, text: str, dot_color: str, bg: str, fg: str, border: str):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # Qualified by objectName, not a bare "QFrame{...}" type selector --
        # QLabel (this chip's own text label) and the swatch dot below are
        # both QFrame subclasses too, so an unscoped rule here would also
        # match them and leak this chip's border onto each (neither sets
        # its own `border`, only background) -- the same rendering-bug
        # family docs/velum/CHANGELOG.md's 2026-07-08 "Guide & Docs" entry
        # and the 2026-07-18 Assistant entries already found repeatedly.
        self.setObjectName("EngineChip")
        self.setStyleSheet(
            f"QFrame#EngineChip{{background:{bg}; border:1px solid {border}; border-radius:999px;}}")
        row = QHBoxLayout(self)
        row.setContentsMargins(9, 3, 9, 3)
        row.setSpacing(6)
        dot = QFrame()
        dot.setFixedSize(7, 7)
        dot.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dot.setStyleSheet(f"background:{dot_color}; border-radius:3px;")
        row.addWidget(dot)
        lb = QLabel(text)
        lb.setStyleSheet(f"color:{fg}; font-size:11px; font-weight:600; background:transparent;")
        row.addWidget(lb)


class PillButton(QPushButton):
    """Button in a design ``kind`` (primary | ghost | success | danger), optional icon."""

    def __init__(self, text: str, t: dict, kind: str = "primary",
                 icon_name: Optional[str] = None, small: bool = False):
        super().__init__(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        qss = theme.button_qss(t, kind)
        if small:
            qss += "QPushButton{padding:7px 11px; font-size:12.5px;}"
        self.setStyleSheet(qss)
        if icon_name:
            col = "#ffffff" if kind in ("primary", "danger") else t["text_subtle"]
            self.setIcon(icons.icon(icon_name, col, 15))
            self.setIconSize(QSize(15, 15))


class IconButton(QToolButton):
    """A square icon button with hover surface."""

    def __init__(self, icon_name: str, t: dict, size: int = 30, tip: str = "",
                 on_click: Optional[Callable[[], None]] = None):
        super().__init__()
        self._t = t
        self._icon = icon_name
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(size, size)
        self.setIcon(icons.icon(icon_name, t["text_muted"], int(size * 0.55)))
        self.setIconSize(QSize(int(size * 0.55), int(size * 0.55)))
        if tip:
            self.setToolTip(tip)
        self.setStyleSheet(
            f"QToolButton{{background:transparent; border:1px solid transparent; border-radius:8px;}}"
            f"QToolButton:hover{{background:{t['surface2']}; border-color:{t['border']};}}")
        if on_click:
            self.clicked.connect(lambda: on_click())


class SelectBox(QFrame):
    """A combo-box look-alike (value + chevron).

    Non-interactive by default (the original design-skeleton behaviour, still
    used as-is by not-yet-wired screens). Pass ``options`` + ``on_select`` to
    make it a real, working control: clicking pops up a ``QMenu`` of
    ``options`` beneath the box; choosing one updates the displayed text and
    calls ``on_select(choice)``. Pass ``on_click`` instead (e.g. to open a
    file picker, when the value isn't a small fixed choice set) for a plain
    click action with no menu. Either way it's the same visual, now backed by
    a real value, so a tab can be wired without inventing a new widget.
    """

    def __init__(self, text: str, t: dict, lead_icon: Optional[str] = None,
                 lead_color: Optional[str] = None,
                 options: Optional[list[str]] = None,
                 on_select: Optional[Callable[[str], None]] = None,
                 on_click: Optional[Callable[[], None]] = None):
        super().__init__()
        self._t = t
        self._options = options
        self._on_select = on_select
        self._on_click = on_click
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified -- see EngineChip's comment. This box's value/chevron
        # labels don't set their own `border`, so an unscoped QFrame{...}
        # rule here leaks this box's own border onto them too.
        self.setObjectName("SelectBox")
        self.setStyleSheet(
            f"QFrame#SelectBox{{background:{t['inset']}; border:1px solid {t['border']};"
            f"border-radius:8px;}} QFrame#SelectBox:hover{{border-color:{t['border_strong']};}}")
        row = QHBoxLayout(self)
        row.setContentsMargins(11, 7, 10, 7)
        row.setSpacing(7)
        if lead_icon:
            ic = QLabel()
            ic.setPixmap(icons.pixmap(lead_icon, lead_color or t["primary"], 14))
            row.addWidget(ic)
        self._val = _ElidingLabel(text)
        self._val.setStyleSheet(f"color:{t['text']}; font-size:12.5px; font-weight:600; background:transparent;")
        row.addWidget(self._val, 1)
        chev = QLabel()
        chev.setPixmap(icons.pixmap("chevron_down", t["text_muted"], 13))
        row.addWidget(chev)
        if (options and on_select) or on_click:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def text(self) -> str:
        return self._val.text()

    def set_text(self, text: str) -> None:
        self._val.setText(text)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            if self._options and self._on_select:
                self._open_menu()
            elif self._on_click:
                self._on_click()
        super().mouseReleaseEvent(e)

    def _open_menu(self) -> None:
        t = self._t
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu{{background:{t['surface']}; border:1px solid {t['border_strong']};"
            f"border-radius:8px; padding:4px;}}"
            f"QMenu::item{{color:{t['text']}; padding:6px 14px; border-radius:6px;}}"
            f"QMenu::item:selected{{background:{t['surface2']};}}")
        for choice in self._options:
            action = QAction(choice, menu)
            action.triggered.connect(lambda _=False, c=choice: self._choose(c))
            menu.addAction(action)
        menu.exec(self.mapToGlobal(self.rect().bottomLeft()))

    def _choose(self, choice: str) -> None:
        self.set_text(choice)
        if self._on_select:
            self._on_select(choice)


class Toggle(QFrame):
    """A pill toggle whose look reflects on/off. Clicking flips the look only."""

    toggled = pyqtSignal(bool)

    def __init__(self, t: dict, on: bool = False):
        super().__init__()
        self._t = t
        self._on = on
        self.setFixedSize(36, 21)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._knob = QFrame(self)
        self._knob.setFixedSize(17, 17)
        self._render()

    def _render(self):
        t = self._t
        self.setStyleSheet(
            f"background:{t['signal'] if self._on else t['border_strong']}; border-radius:10px;")
        self._knob.setStyleSheet("background:#ffffff; border-radius:8px;")
        self._knob.move(17 if self._on else 2, 2)

    def is_on(self) -> bool:
        return self._on

    def set_on(self, on: bool):
        self._on = on
        self._render()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._on = not self._on
            self._render()
            self.toggled.emit(self._on)
        super().mouseReleaseEvent(e)


class Slider(QFrame):
    """A draggable value slider (0..1): a track with a fill + knob at
    ``value``. Click or drag anywhere on it to set the value — emits
    ``changed(float)``. ``set_value(v)`` updates the visual without
    emitting, for a caller applying an external change (e.g. a quality
    preset) without re-triggering its own handler.
    """

    changed = pyqtSignal(float)

    def __init__(self, t: dict, value: float = 0.5, color: Optional[str] = None):
        super().__init__()
        self._t = t
        self._v = max(0.0, min(1.0, value))
        self._color = color or t["primary"]
        self.setFixedHeight(14)
        self.setMinimumWidth(80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._track = QFrame(self)
        self._fill = QFrame(self)
        self._knob = QFrame(self)
        self._track.setStyleSheet(f"background:{t['border']}; border-radius:2px;")
        self._fill.setStyleSheet(f"background:{self._color}; border-radius:2px;")
        self._knob.setStyleSheet(
            f"background:#ffffff; border:2px solid {self._color}; border-radius:7px;")

    def value(self) -> float:
        return self._v

    def set_value(self, v: float, emit: bool = False) -> None:
        self._v = max(0.0, min(1.0, v))
        self._reposition()
        if emit:
            self.changed.emit(self._v)

    def _reposition(self) -> None:
        w, h = self.width(), self.height()
        self._track.setGeometry(0, h // 2 - 2, w, 4)
        fw = int(w * self._v)
        self._fill.setGeometry(0, h // 2 - 2, fw, 4)
        self._knob.setGeometry(max(0, min(w - 14, fw - 7)), h // 2 - 7, 14, 14)

    def resizeEvent(self, e):
        self._reposition()
        super().resizeEvent(e)

    def _set_from_x(self, x: float) -> None:
        self.set_value(x / max(1, self.width()), emit=True)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._set_from_x(e.position().x())
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._set_from_x(e.position().x())
        super().mouseMoveEvent(e)


class Stepper(QFrame):
    """− value + control: click −/+ to nudge a numeric value; emits
    ``changed(float)``. ``decimals=0`` (default) displays/steps whole
    numbers; ``suffix`` appends a display-only unit (e.g. ``" px"``).
    """

    changed = pyqtSignal(float)

    def __init__(self, value: float, t: dict, *, step: float = 1, minimum: float = 0,
                maximum: float = 1_000_000, decimals: int = 0, suffix: str = ""):
        super().__init__()
        self._value = float(value)
        self._step = step
        self._min = minimum
        self._max = maximum
        self._decimals = decimals
        self._suffix = suffix
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified -- see EngineChip's comment. The value QLabel doesn't
        # set its own `border`.
        self.setObjectName("Stepper")
        self.setStyleSheet(
            f"QFrame#Stepper{{background:{t['inset']}; border:1px solid {t['border']}; border-radius:8px;}}")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self._minus = self._make_button("−", t)
        self._minus.clicked.connect(lambda: self._nudge(-self._step))
        row.addWidget(self._minus)
        self._val_label = QLabel()
        self._val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._val_label.setMinimumWidth(42)
        self._val_label.setStyleSheet(
            f"color:{t['text']}; font-family:{theme.MONO}; font-size:12.5px; font-weight:600;")
        row.addWidget(self._val_label)
        self._plus = self._make_button("+", t)
        self._plus.clicked.connect(lambda: self._nudge(self._step))
        row.addWidget(self._plus)
        self._render()

    @staticmethod
    def _make_button(glyph: str, t: dict) -> QToolButton:
        b = QToolButton()
        b.setText(glyph)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFixedSize(26, 28)
        b.setStyleSheet(
            f"QToolButton{{background:transparent; border:none; color:{t['text_muted']};"
            f"font-size:15px;}} QToolButton:hover{{background:{t['surface2']}; color:{t['text']};}}")
        return b

    def value(self) -> float:
        return self._value

    def set_value(self, value: float, emit: bool = False) -> None:
        self._value = max(self._min, min(self._max, value))
        self._render()
        if emit:
            self.changed.emit(self._value)

    def _nudge(self, delta: float) -> None:
        self.set_value(self._value + delta, emit=True)

    def _render(self) -> None:
        text = f"{self._value:.{self._decimals}f}" if self._decimals else f"{int(round(self._value))}"
        self._val_label.setText(text + self._suffix)


class SegControl(QFrame):
    """Segmented buttons; one is 'on'. Clicking moves the selection (visual).

    Pass ``icons_`` (parallel to ``options``, entries optional) for icon-only
    segments — e.g. a grid/list view switch — instead of text labels; the icon
    is recoloured to match the on/off state, same as text.
    """

    changed = pyqtSignal(int)

    def __init__(self, options: list[str], t: dict, active: int = 0, compact: bool = False,
                 icons_: Optional[list[Optional[str]]] = None):
        super().__init__()
        self._t = t
        self._btns: list[QToolButton] = []
        self._icon_names: list[Optional[str]] = list(icons_) if icons_ else [None] * len(options)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified -- see EngineChip's comment. No QLabel/QFrame children
        # today (every segment is a self-styled QToolButton), so this isn't
        # currently visibly broken, but a bare QFrame{...} type selector is
        # a silent trap for whoever adds one later -- fixed the same way
        # regardless of current visible impact.
        self.setObjectName("SegControl")
        self.setStyleSheet(
            f"QFrame#SegControl{{background:{t['inset'] if compact else t['surface2']};"
            f"border:1px solid {t['border']}; border-radius:8px;}}")
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(2)
        for i, opt in enumerate(options):
            b = QToolButton()
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setCheckable(True)
            b.setChecked(i == active)
            b.setMinimumHeight(26)
            if self._icon_names[i]:
                b.setIconSize(QSize(14, 14))
                b.setFixedWidth(32)
            else:
                b.setText(opt)
                b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.clicked.connect(lambda _=False, idx=i: self._select(idx))
            self._btns.append(b)
            row.addWidget(b, 0 if self._icon_names[i] else 1)
        self._restyle()

    def _select(self, idx: int):
        for i, b in enumerate(self._btns):
            b.setChecked(i == idx)
        self._restyle()
        self.changed.emit(idx)

    def _restyle(self):
        t = self._t
        for name, b in zip(self._icon_names, self._btns):
            on = b.isChecked()
            if name:
                b.setIcon(icons.icon(name, t["text"] if on else t["text_muted"], 14))
            if on:
                b.setStyleSheet(
                    f"QToolButton{{background:{t['surface']}; color:{t['text']};"
                    f"border:none; border-radius:6px; font-size:12px; font-weight:600; padding:5px 10px;}}")
            else:
                b.setStyleSheet(
                    f"QToolButton{{background:transparent; color:{t['text_muted']};"
                    f"border:none; border-radius:6px; font-size:12px; font-weight:600; padding:5px 10px;}}"
                    f"QToolButton:hover{{color:{t['text_subtle']};}}")


class StatTile(QFrame):
    """A value + label tile (results stats)."""

    def __init__(self, value: str, unit: str, caption: str, t: dict, ok: bool = False):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Qualified -- see EngineChip's comment. Both the value and caption
        # QLabels below set only `color`/font properties, never their own
        # `border`, so an unscoped QFrame{...} rule here leaked this tile's
        # border onto each of them individually -- likely visible whenever
        # StatTile has actually been on screen (Segment results, Dashboard),
        # not just theoretical.
        self.setObjectName("StatTile")
        self.setStyleSheet(
            f"QFrame#StatTile{{background:{t['inset']}; border:1px solid {t['border']}; border-radius:10px;}}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 10, 8, 10)
        lay.setSpacing(2)
        v = QLabel(f"{value}<span style='font-size:11px;color:{t['text_muted']}'> {unit}</span>" if unit else value)
        col = t["success"] if ok else t["text"]
        v.setStyleSheet(f"color:{col}; font-family:{theme.MONO}; font-size:17px; font-weight:600; letter-spacing:-0.5px; background:transparent;")
        c = _ElidingLabel(caption)
        c.setStyleSheet(f"color:{t['text_muted']}; font-size:10px; font-weight:600; letter-spacing:0.1px; background:transparent;")
        lay.addWidget(v)
        lay.addWidget(c)


class FieldRow(QWidget):
    """A label ⟷ control row."""

    def __init__(self, name: str, control: QWidget, t: dict):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        lb = QLabel(name)
        # background:transparent is mandatory, not decorative -- a QLabel with
        # its own instance stylesheet nested inside a styled ancestor otherwise
        # resolves its background via the app-wide QWidget{background:bg} rule
        # and paints an opaque bg-coloured box over the panel's own `inset`
        # tone, whose edges read as a faint border ruled around the text (a
        # real user report: "линии... как будто границы расчерчены"). See the
        # label() helper's comment for the full mechanism.
        lb.setStyleSheet(f"color:{t['text_subtle']}; font-size:12.5px; font-weight:500; background:transparent;")
        row.addWidget(lb)
        row.addStretch(1)
        row.addWidget(control)


class GroupLabel(QLabel):
    """Uppercase group heading."""

    def __init__(self, text: str, t: dict):
        super().__init__(text.upper())
        # background:transparent -- see FieldRow's comment (same opaque-box bug).
        self.setStyleSheet(
            f"color:{t['text_muted']}; font-size:10.5px; font-weight:600; letter-spacing:0.6px;"
            f" background:transparent;")


class WavingEmoji(QWidget):
    """An emoji glyph that plays a one-shot hand-wave rotation on ``play()``
    -- e.g. Home's "Welcome back" greeting, matching the small delight-on-open
    cue common to Slack/Notion-style greetings rather than a plain static
    glyph sitting in the title string. Self-contained: builds and owns its
    own ``QVariantAnimation`` (a plain float angle fed straight into
    ``update()``, not a ``pyqtProperty`` -- nothing outside this class binds
    to the angle by name, so the extra property-declaration ceremony isn't
    earning its keep here) and repaints itself with a rotation transform,
    since QSS has no ``transform``/``transition`` to animate.
    """

    def __init__(self, t: dict, glyph: str = "\U0001F44B", size: float = 26):
        super().__init__()
        self._glyph = glyph
        self._size = size
        self._angle = 0.0
        self.setFixedSize(int(size * 1.5), int(size * 1.6))
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(900)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        # A decaying back-and-forth rotation around the wrist, not a spin --
        # keyframes approximate a real wave: two big swings, then two
        # smaller ones settling back to rest. Float values (not int), so
        # QVariantAnimation's interpolation lands on smooth sub-degree
        # steps instead of choppy whole-degree ones.
        for frac, deg in ((0.0, 0.0), (0.14, 18.0), (0.30, -13.0), (0.46, 16.0),
                          (0.62, -8.0), (0.78, 6.0), (0.92, -2.0), (1.0, 0.0)):
            self._anim.setKeyValueAt(frac, deg)
        self._anim.valueChanged.connect(self._set_angle)

    def _set_angle(self, deg) -> None:
        self._angle = float(deg)
        self.update()

    def play(self) -> None:
        """(Re)start the wave from rest -- safe to call repeatedly (e.g. once
        per Home visit); restarts rather than layering a second animation."""
        self._anim.stop()
        self._angle = 0.0
        self._anim.start()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont()
        font.setPointSizeF(self._size)
        p.setFont(font)
        # Rotate around the glyph's own base (its "wrist"), not the widget
        # centre -- a wave pivots at the wrist; a centred rotation reads as
        # a spin instead.
        pivot = QPointF(self.width() / 2, self.height() * 0.72)
        p.translate(pivot)
        p.rotate(self._angle)
        p.translate(-pivot)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._glyph)


class Accordion(QFrame):
    """A collapsible section with a leading icon, title, chevron, and body.

    ``caps=True`` (the default) is the original micro-label treatment for
    short section headers ("GROUND TRUTH", "BATCH PREDICTION"). Pass
    ``caps=False`` for a full-sentence title — e.g. an FAQ question — which
    reads badly shouted in all-caps at 11.5px.

    ``fill`` picks the background token — default ``"inset"`` (the existing,
    unchanged look every current call site was designed and screenshotted
    against: a *recessed* well, right when the accordion sits inside an
    already-differently-toned card, e.g. the Segment inspector). Pass
    ``fill="surface2"`` ("elevated fill") when the accordion sits more or
    less directly on a plain page/panel background instead — `inset` there
    reads as barely more than a hollow outline (too little contrast against
    its surroundings), the same family of mistake as the token lesson in
    `docs/velum/CHANGELOG.md`'s 2026-07-08 entry, just one token over.
    """

    def __init__(self, title: str, t: dict, lead: str = "check", open_: bool = False,
                 caps: bool = True, fill: str = "inset"):
        super().__init__()
        self._t = t
        self._open = open_
        self._caps = caps
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QFrame#Acc{{background:{t[fill]}; border:1px solid {t['border']}; border-radius:10px;}}")
        self.setObjectName("Acc")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)

        self._head = QToolButton()
        self._head.setCursor(Qt.CursorShape.PointingHandCursor)
        self._head.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._head_row = QWidget()
        hr = QHBoxLayout(self._head_row)
        hr.setContentsMargins(13, 12, 13, 12)
        hr.setSpacing(9)
        licon = QLabel()
        licon.setPixmap(icons.pixmap(lead, t["signal"], 15))
        # caps=True titles are meant to be short, non-wrapping section headers
        # (the docstring's contract) — but a *dynamic* one (e.g. "Engine
        # settings · <engine label>") can still run long, so it gets the same
        # eliding treatment as SelectBox rather than silently overflowing the
        # panel. caps=False already wraps instead (a deliberately different,
        # already-working pattern for full-sentence titles) and is untouched.
        title_lb = _ElidingLabel(title.upper()) if caps else QLabel(title)
        self._title_lb = title_lb
        if not caps:
            title_lb.setWordWrap(True)
        if caps:
            title_lb.setStyleSheet(
                f"color:{t['text_subtle']}; font-size:11.5px; font-weight:600; letter-spacing:0.5px;")
        else:
            title_lb.setStyleSheet(f"color:{t['text']}; font-size:13px; font-weight:600; background:transparent;")
        self._chev = QLabel()
        self._chev.setPixmap(icons.pixmap("chevron" if not open_ else "chevron_down", t["text_muted"], 14))
        hr.addWidget(licon)
        hr.addWidget(title_lb, 1 if caps else 0)
        if caps:
            hr.addWidget(self._chev)
        else:
            hr.addStretch(1)
            hr.addWidget(self._chev)
        self._head_row.setStyleSheet("background:transparent;")
        self._head_row.mouseReleaseEvent = lambda e: self.toggle()
        self._lay.addWidget(self._head_row)

        self._body = QWidget()
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(13, 0, 13, 14)
        self._body_lay.setSpacing(10)
        self._body.setVisible(open_)
        self._lay.addWidget(self._body)

    def add(self, w: QWidget):
        self._body_lay.addWidget(w)

    def add_layout(self, l):
        self._body_lay.addLayout(l)

    def set_title(self, title: str) -> None:
        """Update the header title live (e.g. "Model · Ollama" as the
        backend changes) without rebuilding the whole accordion."""
        self._title_lb.setText(title.upper() if self._caps else title)

    def toggle(self):
        self._open = not self._open
        self._body.setVisible(self._open)
        if self._open:
            # Reveal is animated (a soft fade-in — the same primitive screen
            # switches already use); dismissal stays instant, matching most
            # real accordions (Linear, macOS System Settings, ...): opening
            # invites a look, closing should just get out of the way.
            from studio.motion import fade_in
            fade_in(self._body, 170)
        self._chev.setPixmap(icons.pixmap(
            "chevron_down" if self._open else "chevron", self._t["text_muted"], 14))


# ── Sidebar navigation ───────────────────────────────────────────────────────
class _NavItem(QToolButton):
    def __init__(self, key: str, icon_name: str, text: str, t: dict, pip: bool = False):
        super().__init__()
        self.key = key
        self.icon_name = icon_name
        self._t = t
        self._pip = pip
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        # Escape '&' -> '&&': QToolButton (like every QAbstractButton) treats a
        # single '&' as a keyboard-mnemonic marker and swallows it, underlining
        # the next character -- so "Models & Train" / "Guide & Docs" rendered as
        # "Models_Train" / "Guide_Docs" with an underlined space. Escaping keeps
        # the literal ampersand.
        self.setText("   " + text.replace("&", "&&"))
        self.setIcon(icons.icon(icon_name, t["text_muted"], 18))
        self.setIconSize(QSize(18, 18))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(38)
        self._apply(False)

    def _apply(self, active: bool):
        t = self._t
        if active:
            self.setIcon(icons.icon(self.icon_name, t["signal"], 18))
            self.setStyleSheet(
                f"QToolButton{{background:{t['surface']}; color:{t['text']};"
                f"border:1px solid {t['border']}; border-radius:9px; padding:8px 10px;"
                f"font-size:13px; font-weight:600; text-align:left;}}")
        else:
            self.setIcon(icons.icon(self.icon_name, t["text_muted"], 18))
            self.setStyleSheet(
                f"QToolButton{{background:transparent; color:{t['text_subtle']};"
                f"border:1px solid transparent; border-radius:9px; padding:8px 10px;"
                f"font-size:13px; font-weight:600; text-align:left;}}"
                f"QToolButton:hover{{background:{t['surface2']}; color:{t['text']};}}")

    def set_active(self, active: bool):
        self.setChecked(active)
        self._apply(active)


class Sidebar(QFrame):
    """Navigation rail: wordmark, sectioned nav items, footer (guide, appearance)."""

    navigate = pyqtSignal(str)
    toggle_theme = pyqtSignal()
    open_guide = pyqtSignal()

    WIDTH = 232

    def __init__(self, items: list[tuple[str, str, str, str]], t: dict):
        super().__init__()
        self._t = t
        self._items: dict[str, _NavItem] = {}
        self.setObjectName("Sidebar")
        self.setFixedWidth(self.WIDTH)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"#Sidebar{{background:{t['inset']}; border-right:1px solid {t['border']};}}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(3)

        brand = QLabel(
            f"<span style='font-size:21px;font-weight:600;letter-spacing:-0.4px;color:{t['text']}'>"
            f"Velum<span style='color:{t['primary']}'>.</span></span>")
        brand.setContentsMargins(6, 2, 0, 12)
        lay.addWidget(brand)

        section = None
        for key, icon_name, text, sec in items:
            if sec and sec != section:
                section = sec
                lay.addWidget(self._section(sec))
            item = _NavItem(key, icon_name, text, t)
            item.clicked.connect(lambda _=False, k=key: self.navigate.emit(k))
            self._items[key] = item
            lay.addWidget(item)

        lay.addStretch(1)
        lay.addWidget(hline(t))

        guide = _NavItem("__guide__", "guide", "Guide & Docs", t)
        guide.setCheckable(False)
        guide.clicked.connect(lambda: self.open_guide.emit())
        lay.addWidget(guide)

        appearance = _NavItem("__theme__", "moon", "Appearance", t)
        appearance.setCheckable(False)
        appearance.clicked.connect(lambda: self.toggle_theme.emit())
        lay.addWidget(appearance)

        ver = QLabel("v0.1 · Velum")
        ver.setStyleSheet(f"color:{t['text_muted']}; font-size:10.5px; padding:6px 10px 0; font-family:{theme.MONO}; background:transparent;")
        lay.addWidget(ver)

    def _section(self, text: str) -> QLabel:
        lb = QLabel(text.upper())
        lb.setStyleSheet(
            f"color:{self._t['text_muted']}; font-size:10px; font-weight:600;"
            f"letter-spacing:0.7px; padding:12px 10px 4px;")
        return lb

    def set_active(self, key: str):
        for k, item in self._items.items():
            if item.isCheckable():
                item.set_active(k == key)


class SwipeRow(QFrame):
    """A list row that reveals a Delete action when swiped left, iOS-style.

    A plain click calls ``on_click``; dragging the row left past a commit
    threshold and releasing calls ``on_delete`` (deferred by the caller, since
    deleting rebuilds the very list this row lives in). Below the threshold the
    row springs back. ``content`` is the foreground widget shown at rest — it
    must be opaque (it slides over the red delete backdrop), and it's made
    mouse-transparent so every drag lands on this row, not on the labels inside.

    Deliberately no fancy fly-off animation on delete: the list rebuild that
    ``on_delete`` triggers replaces the row anyway, and animating a widget
    that's about to be torn down is the exact deleted-object hazard motion.py
    documents. The spring-back (the non-destructive case) *is* animated.
    """

    _SWIPE_SLOP = 12  # px of leftward drift tolerated before a tap becomes a swipe

    def __init__(self, content: QWidget, t: dict, on_click, on_delete,
                 height: int = 46, reveal: int = 86):
        super().__init__()
        self._t = t
        self._on_click = on_click
        self._on_delete = on_delete
        self._reveal = reveal
        self._commit = reveal * 0.55  # drag this far left -> release deletes
        # A tap almost never lands perfectly still: a trackpad/mouse click
        # routinely drifts a few px. The old 4px threshold treated that drift
        # as a swipe and swallowed the click, so rows were "hard to select"
        # (a real report). Only past this larger slop do we commit to swipe
        # mode; below it a release is a tap. iOS uses ~10px; 12 is a hair more
        # forgiving without eating a deliberate swipe.
        self._drag_start: Optional[float] = None
        self._swiping = False
        self._offset = 0.0
        self.setFixedHeight(height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._back = QFrame(self)
        self._back.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._back.setObjectName("SwipeDel")
        self._back.setStyleSheet(
            f"QFrame#SwipeDel{{background:{t['danger']}; border-radius:8px;}}")
        bl = QHBoxLayout(self._back)
        bl.setContentsMargins(0, 0, 18, 0)
        bl.addStretch(1)
        trash = QLabel()
        trash.setPixmap(icons.pixmap("trash", "#fff", 16))
        bl.addWidget(trash)

        self._fg = content
        self._fg.setParent(self)
        self._fg.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def resizeEvent(self, e):
        self._back.setGeometry(0, 0, self.width(), self.height())
        self._fg.setGeometry(int(self._offset), 0, self.width(), self.height())

    def _set_offset(self, x: float) -> None:
        self._offset = max(-self._reveal, min(0.0, x))
        self._fg.move(int(self._offset), 0)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_start = e.position().x()
            self._swiping = False

    def mouseMoveEvent(self, e):
        if self._drag_start is None:
            return
        dx = e.position().x() - self._drag_start
        if not self._swiping and dx < -self._SWIPE_SLOP:
            self._swiping = True
        # Only move the row once we've committed to a swipe, and start the
        # reveal from the slop point so it doesn't jump; below the slop a tap
        # leaves the row perfectly still and still selects on release.
        if self._swiping:
            self._set_offset(dx + self._SWIPE_SLOP)

    def mouseReleaseEvent(self, e):
        if self._drag_start is None:
            return
        was_swipe, off = self._swiping, self._offset
        self._drag_start = None
        self._swiping = False
        if was_swipe:
            if off <= -self._commit:
                self._on_delete()  # caller defers the actual list rebuild
            else:
                self._spring_back()
        else:
            self._on_click()

    def _spring_back(self) -> None:
        try:
            anim = QPropertyAnimation(self._fg, b"pos", self)
            anim.setDuration(170)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.setStartValue(QPoint(int(self._offset), 0))
            anim.setEndValue(QPoint(0, 0))
            anim.finished.connect(lambda: setattr(self, "_offset", 0.0))
            anim.start()
            self._spring_anim = anim  # keep a ref alive
        except Exception:
            self._set_offset(0.0)
