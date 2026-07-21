"""Velum — an image-region picker (Steam-avatar-style crop).

When a user picks a cover image, this lets them choose *which part* of it to
use: a draggable, resizable crop rectangle over the image with a live preview
of how the selection reads as a wide banner and as a square avatar. On apply it
hands back the cropped :class:`QPixmap`, which the caller saves as the project's
cover (see ``project_dialogs.ProjectSettingsDialog._pick_cover_image``).

Reusable for any "pick a region of an image" need, not just covers. The
coordinate maths (``clamp_crop`` / ``crop_to_source`` / ``apply_drag``) are pure
functions at module top so they're unit-tested without a display; the Qt
canvas/dialog just drive them from mouse events.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt, QRectF, QRect, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPixmap, QColor, QPen
from PyQt6.QtWidgets import QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout

from studio import theme
from studio.components import PillButton, GroupLabel, hline, label, soft_shadow

MIN_CROP = 28.0          # smallest crop side, in display pixels
HANDLE_HIT = 13.0        # how close (px) a click must be to grab a handle


def clamp_crop(x: float, y: float, w: float, h: float, W: float, H: float,
               min_size: float = MIN_CROP) -> tuple[float, float, float, float]:
    """Keep a crop rect inside ``0..W`` × ``0..H`` and no smaller than
    ``min_size`` on either side."""
    w = max(min_size, min(w, W))
    h = max(min_size, min(h, H))
    x = max(0.0, min(x, W - w))
    y = max(0.0, min(y, H - h))
    return x, y, w, h


def apply_drag(rect: tuple[float, float, float, float], handle: str,
               dx: float, dy: float, W: float, H: float,
               min_size: float = MIN_CROP) -> tuple[float, float, float, float]:
    """Return ``rect`` after dragging ``handle`` by ``(dx, dy)``.

    ``handle`` is ``"move"`` (whole rect) or a compass edge/corner
    (``"n"``/``"e"``/``"s"``/``"w"``/``"nw"``/``"ne"``/``"sw"``/``"se"``). Each
    edge letter moves that side; a corner moves both its sides. Result is
    clamped to bounds and min size.
    """
    x, y, w, h = rect
    if handle == "move":
        return clamp_crop(x + dx, y + dy, w, h, W, H, min_size)
    x0, y0, x1, y1 = x, y, x + w, y + h
    if "w" in handle:
        x0 = min(x0 + dx, x1 - min_size)
    if "e" in handle:
        x1 = max(x1 + dx, x0 + min_size)
    if "n" in handle:
        y0 = min(y0 + dy, y1 - min_size)
    if "s" in handle:
        y1 = max(y1 + dy, y0 + min_size)
    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)
    return x0, y0, max(min_size, x1 - x0), max(min_size, y1 - y0)


def crop_to_source(cx: float, cy: float, cw: float, ch: float,
                   disp_w: float, disp_h: float, src_w: int, src_h: int
                   ) -> tuple[int, int, int, int]:
    """Map a crop rect from display-space to source-pixel coords, clamped to
    the source image so the resulting rect is always valid to ``copy()``."""
    sx = src_w / disp_w if disp_w else 1.0
    sy = src_h / disp_h if disp_h else 1.0
    x = max(0, min(int(round(cx * sx)), src_w - 1))
    y = max(0, min(int(round(cy * sy)), src_h - 1))
    w = max(1, min(int(round(cw * sx)), src_w - x))
    h = max(1, min(int(round(ch * sy)), src_h - y))
    return x, y, w, h


def cover_fit(src: QPixmap, w: int, h: int, radius: int = 8) -> QPixmap:
    """Centre-crop ``src`` to fill ``w``×``h`` with rounded corners — the same
    fit the cover uses, for the crop dialog's live previews."""
    out = QPixmap(w, h)
    out.fill(Qt.GlobalColor.transparent)
    if src.isNull():
        return out
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    from PyQt6.QtGui import QPainterPath
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, w, h), radius, radius)
    p.setClipPath(path)
    sw, sh = src.width(), src.height()
    scale = max(w / sw, h / sh)
    dw, dh = sw * scale, sh * scale
    p.drawPixmap(QRectF((w - dw) / 2, (h - dh) / 2, dw, dh), src, QRectF(0, 0, sw, sh))
    p.end()
    return out


class _CropCanvas(QWidget):
    """The interactive image + crop rectangle. Drag inside to reposition, pull a
    corner/edge to resize; emits ``changed`` so the dialog can refresh previews."""

    changed = pyqtSignal()

    def __init__(self, image: QPixmap, t: dict, max_w: int = 460, max_h: int = 300):
        super().__init__()
        self._t = t
        self._orig = image
        sw, sh = max(image.width(), 1), max(image.height(), 1)
        scale = min(max_w / sw, max_h / sh, 1.0)
        self._dw, self._dh = max(1.0, sw * scale), max(1.0, sh * scale)
        self._disp = image.scaled(int(self._dw), int(self._dh),
                                  Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
        self.setFixedSize(int(self._dw), int(self._dh))
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        # default: a generous centred region
        self._crop = clamp_crop(self._dw * 0.08, self._dh * 0.12,
                                self._dw * 0.84, self._dh * 0.76, self._dw, self._dh)
        self._drag: Optional[str] = None
        self._last: Optional[QPointF] = None

    # ── geometry ─────────────────────────────────────────────────────────────
    def _handle_at(self, pos: QPointF) -> Optional[str]:
        x, y, w, h = self._crop
        edges_x = {"w": x, "e": x + w}
        edges_y = {"n": y, "s": y + h}
        near_x = {k: abs(pos.x() - v) <= HANDLE_HIT for k, v in edges_x.items()}
        near_y = {k: abs(pos.y() - v) <= HANDLE_HIT for k, v in edges_y.items()}
        within_x = x - HANDLE_HIT <= pos.x() <= x + w + HANDLE_HIT
        within_y = y - HANDLE_HIT <= pos.y() <= y + h + HANDLE_HIT
        v = next((k for k, n in near_y.items() if n and within_x), "")
        hge = next((k for k, n in near_x.items() if n and within_y), "")
        if v or hge:
            return v + hge
        if x <= pos.x() <= x + w and y <= pos.y() <= y + h:
            return "move"
        return None

    def _cursor_for(self, handle: Optional[str]) -> Qt.CursorShape:
        return {
            "move": Qt.CursorShape.SizeAllCursor,
            "n": Qt.CursorShape.SizeVerCursor, "s": Qt.CursorShape.SizeVerCursor,
            "e": Qt.CursorShape.SizeHorCursor, "w": Qt.CursorShape.SizeHorCursor,
            "nw": Qt.CursorShape.SizeFDiagCursor, "se": Qt.CursorShape.SizeFDiagCursor,
            "ne": Qt.CursorShape.SizeBDiagCursor, "sw": Qt.CursorShape.SizeBDiagCursor,
        }.get(handle or "", Qt.CursorShape.OpenHandCursor)

    # ── interaction ──────────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        self._drag = self._handle_at(e.position())
        self._last = e.position()

    def mouseMoveEvent(self, e):
        if self._drag is None:
            self.setCursor(self._cursor_for(self._handle_at(e.position())))
            return
        dx = e.position().x() - self._last.x()
        dy = e.position().y() - self._last.y()
        self._last = e.position()
        self._crop = apply_drag(self._crop, self._drag, dx, dy, self._dw, self._dh)
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, e):
        self._drag = None

    # ── output ───────────────────────────────────────────────────────────────
    def cropped_pixmap(self) -> QPixmap:
        x, y, w, h = crop_to_source(*self._crop, self._dw, self._dh,
                                    self._orig.width(), self._orig.height())
        return self._orig.copy(QRect(x, y, w, h))

    # ── paint ────────────────────────────────────────────────────────────────
    def paintEvent(self, e):
        t = self._t
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.drawPixmap(0, 0, self._disp)
        x, y, w, h = self._crop
        scrim = QColor(0, 0, 0, 128)
        p.fillRect(QRectF(0, 0, self._dw, y), scrim)                       # top
        p.fillRect(QRectF(0, y + h, self._dw, self._dh - y - h), scrim)    # bottom
        p.fillRect(QRectF(0, y, x, h), scrim)                             # left
        p.fillRect(QRectF(x + w, y, self._dw - x - w, h), scrim)          # right
        # rule-of-thirds guides
        p.setPen(QPen(QColor(255, 255, 255, 60), 1))
        for i in (1, 2):
            p.drawLine(QPointF(x + w * i / 3, y), QPointF(x + w * i / 3, y + h))
            p.drawLine(QPointF(x, y + h * i / 3), QPointF(x + w, y + h * i / 3))
        # frame
        p.setPen(QPen(QColor(t["primary"]), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(x, y, w, h))
        # handles
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(t["primary"]))
        for hx, hy in [(x, y), (x + w / 2, y), (x + w, y),
                       (x, y + h / 2), (x + w, y + h / 2),
                       (x, y + h), (x + w / 2, y + h), (x + w, y + h)]:
            p.drawEllipse(QPointF(hx, hy), 4.5, 4.5)
        p.end()


class CropDialog(QWidget):
    """Scrim-backed crop dialog (same construction/lifecycle as
    ``project_dialogs.ConfirmDialog``). ``on_apply`` receives the cropped
    :class:`QPixmap`."""

    def __init__(self, parent: QWidget, t: dict, image_path: str,
                 on_apply: Callable[[QPixmap], None]):
        super().__init__(parent)
        self._t = t
        self._on_apply = on_apply
        self._image = QPixmap(image_path)
        self.setStyleSheet(f"background:{theme.SCRIM};")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 70, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self._build_panel())
        self.hide()

    def _build_panel(self) -> QFrame:
        t = self._t
        panel = QFrame()
        panel.setObjectName("CropPanel")
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setStyleSheet(
            f"QFrame#CropPanel{{background:{t['surface']}; border:1px solid {t['border_strong']};"
            f" border-radius:14px;}}")
        soft_shadow(panel, 28, 40, 10)
        v = QVBoxLayout(panel)
        v.setContentsMargins(22, 20, 22, 20)
        v.setSpacing(14)
        v.addWidget(label("Adjust cover image", 15, t["text"], 600))
        v.addWidget(label("Drag to reposition · pull a handle to resize.", 12, t["text_muted"]))

        self._canvas = _CropCanvas(self._image, t)
        self._canvas.changed.connect(self._sync_preview)
        canvas_wrap = QHBoxLayout()
        canvas_wrap.addStretch(1)
        canvas_wrap.addWidget(self._canvas)
        canvas_wrap.addStretch(1)
        v.addLayout(canvas_wrap)

        v.addWidget(hline(t))
        prev = QHBoxLayout()
        prev.setSpacing(16)
        pcol = QVBoxLayout(); pcol.setSpacing(6)
        pcol.addWidget(GroupLabel("Card banner", t))
        self._banner_prev = QLabel(); self._banner_prev.setFixedSize(220, 56)
        pcol.addWidget(self._banner_prev)
        prev.addLayout(pcol)
        acol = QVBoxLayout(); acol.setSpacing(6)
        acol.addWidget(GroupLabel("Avatar", t))
        self._avatar_prev = QLabel(); self._avatar_prev.setFixedSize(56, 56)
        acol.addWidget(self._avatar_prev)
        prev.addLayout(acol)
        prev.addStretch(1)
        v.addLayout(prev)

        row = QHBoxLayout()
        row.addStretch(1)
        cancel = PillButton("Cancel", t, "ghost", small=True)
        cancel.clicked.connect(self.hide)
        row.addWidget(cancel)
        use = PillButton("Use image", t, "primary", small=True)
        use.clicked.connect(self._apply)
        row.addWidget(use)
        v.addLayout(row)

        self._sync_preview()
        return panel

    def _sync_preview(self) -> None:
        cropped = self._canvas.cropped_pixmap()
        self._banner_prev.setPixmap(cover_fit(cropped, 220, 56))
        self._avatar_prev.setPixmap(cover_fit(cropped, 56, 56))

    def _apply(self) -> None:
        cropped = self._canvas.cropped_pixmap()
        self.hide()
        if self._on_apply:
            self._on_apply(cropped)

    def place(self) -> None:
        p = self.parentWidget()
        if p:
            self.setGeometry(0, 0, p.width(), p.height())

    def open(self) -> None:
        self.place()
        self.show()
        self.raise_()

    def mousePressEvent(self, e) -> None:
        if self.childAt(e.position().toPoint()) is None:
            self.hide()
        super().mousePressEvent(e)

    def hideEvent(self, e) -> None:
        super().hideEvent(e)
        self.deleteLater()
