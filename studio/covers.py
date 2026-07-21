"""Velum — project cover art (colour · image · auto tint).

A project's *cover* is its visual identity on library cards and the Home
screen — the thing that lets a microscopist tell "H&E Tissue" from "DAPI
Nuclei" at a glance without reading. Three kinds (see
``studio.project.ProjectCover``):

* **auto**  — a deterministic hue derived from the project id, painted as a
  soft, heavily-blurred "aurora" of glowing blobs (the generated look the
  product owner liked, dialled up on blur). Every project is distinct for free.
* **colour** — the same aurora in a hue the user pinned from a small palette.
* **image** — a user-picked image, centre-cropped to the banner (Notion-style).

Rendering mirrors ``paint.NucleiView``: procedural art is expensive to redraw,
so ``CoverView`` caches to a pixmap and only rebuilds when its size / DPI /
cover actually change — cheap ``drawPixmap`` on every scroll frame otherwise.
Pure-logic helpers (``auto_color``/``resolve_color``) live at module top so the
data layer's tests can import them without a display.
"""
from __future__ import annotations

import zlib
from typing import Optional

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QPixmap, QColor, QRadialGradient, QLinearGradient
from PyQt6.QtWidgets import QWidget

from studio.paint import _rng, _round_rect_path

# The cover palette — a small, curated set so a library stays cohesive (the
# consistency the Notion-cover guides stress) while still differentiating.
# Hues echo the data-viz family (theme.VIZ) plus a rose and a sky.
COVER_COLORS: list[tuple[str, str]] = [
    ("Iris", "#6d87f1"), ("Teal", "#2bd4c0"), ("Kiwi", "#6fae53"),
    ("Amber", "#e0982f"), ("Coral", "#ee6a52"), ("Fig", "#a878cf"),
    ("Rose", "#e05f8e"), ("Sky", "#4aa3df"),
]


def auto_color(project_id: str) -> str:
    """Deterministic palette hue for a project — stable across relaunches,
    stored nowhere (derived from the id, like ``cover_seed``)."""
    return COVER_COLORS[zlib.crc32(project_id.encode("utf-8")) % len(COVER_COLORS)][1]


def resolve_color(kind: str, color: str, project_id: str) -> str:
    """The hex the aurora should paint in: the pinned ``color`` for a "color"
    cover, otherwise the id-derived auto hue."""
    if kind == "color" and color:
        return color
    return auto_color(project_id)


def _shift(hex_color: str, dh: float, ds: float = 0.0, dv: float = 0.0) -> QColor:
    """A hue/sat/val-shifted variant of ``hex_color`` (for the aurora's
    related-but-not-identical blob tints)."""
    c = QColor(hex_color)
    h, s, v, _ = c.getHsvF()
    h = (h + dh) % 1.0
    return QColor.fromHsvF(h, min(1.0, max(0.0, s + ds)), min(1.0, max(0.0, v + dv)), 1.0)


def paint_aurora(p: QPainter, w: int, h: int, hex_color: str, seed: int) -> None:
    """A soft, blurred field of glowing blobs in ``hex_color`` on a deep tint
    of the same hue — the auto/colour cover. Deterministic per ``seed``."""
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    # deep tinted ground (a dark, desaturated version of the hue → reads on both themes)
    bg = QLinearGradient(0, 0, w, h)
    bg.setColorAt(0.0, _shift(hex_color, -0.02, -0.15, -0.60))
    bg.setColorAt(1.0, _shift(hex_color, 0.05, -0.08, -0.50))
    p.fillRect(0, 0, int(w), int(h), bg)

    r = _rng(seed)
    hues = [0.0, 0.07, -0.06, 0.12, -0.03]
    for i in range(5):
        cx, cy = r() * w, r() * h
        rad = max(w, h) * (0.55 + r() * 0.55)           # large radius → very soft
        col = _shift(hex_color, hues[i % len(hues)], 0.02, -0.02 + r() * 0.18)
        g = QRadialGradient(QPointF(cx, cy), rad)
        near, far = QColor(col), QColor(col)
        near.setAlpha(140)                               # low alpha + big falloff = heavy blur
        far.setAlpha(0)
        g.setColorAt(0.0, near)
        g.setColorAt(1.0, far)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawRect(QRectF(0, 0, w, h))
    # a faint top sheen for depth
    sheen = QLinearGradient(0, 0, 0, h)
    top = QColor(255, 255, 255, 16)
    sheen.setColorAt(0.0, top)
    sheen.setColorAt(0.4, QColor(255, 255, 255, 0))
    p.fillRect(0, 0, int(w), int(h), sheen)


def _paint_image(p: QPainter, w: int, h: int, src: QPixmap) -> bool:
    """Centre-crop ``src`` to fill ``w``×``h`` (cover-fit). Returns False if the
    source is null (caller falls back to the aurora)."""
    if src.isNull():
        return False
    sw, sh = src.width(), src.height()
    scale = max(w / sw, h / sh)
    dw, dh = sw * scale, sh * scale
    p.drawPixmap(QRectF((w - dw) / 2, (h - dh) / 2, dw, dh), src, QRectF(0, 0, sw, sh))
    return True


def cover_pixmap(w: int, h: int, *, kind: str, color: str, image: Optional[QPixmap],
                 project_id: str, dpr: float = 2.0, radius: float = 0.0,
                 top_only: bool = False) -> QPixmap:
    """Render a cover to a fixed pixmap. ``image`` is the pre-loaded source for
    an image cover (None for auto/colour). Clips to rounded corners when
    ``radius`` is set, exactly like ``paint.nuclei_pixmap``."""
    px = QPixmap(int(w * dpr), int(h * dpr))
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    if radius:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setClipPath(_round_rect_path(w, h, radius, top_only))
    painted = False
    if kind == "image" and image is not None:
        painted = _paint_image(p, w, h, image)
    if not painted:
        seed = zlib.crc32(project_id.encode("utf-8")) % 1000
        paint_aurora(p, w, h, resolve_color(kind, color, project_id), seed)
    p.end()
    return px


class CoverView(QWidget):
    """A project cover that fills its widget and caches its render (mirrors
    ``paint.NucleiView`` — see that class for the caching rationale). Rebuilds
    only when size / DPI / the cover's identity changes."""

    def __init__(self, *, kind: str = "auto", color: str = "", image_path: str = "",
                 project_id: str = "", radius: float = 0.0, top_only: bool = False,
                 min_size: tuple[int, int] = (120, 64)):
        super().__init__()
        self._kind = kind
        self._color = color
        self._image_path = image_path
        self._project_id = project_id
        self._radius = radius
        self._top_only = top_only
        self._src: Optional[QPixmap] = None
        if kind == "image" and image_path:
            pm = QPixmap(image_path)
            self._src = pm if not pm.isNull() else None
        self.setMinimumSize(*min_size)
        self.setStyleSheet("background:transparent;")
        self._cache: Optional[QPixmap] = None
        self._cache_key: Optional[tuple] = None

    def set_cover(self, *, kind: str, color: str = "", image_path: str = "") -> None:
        """Swap the cover in place (used by the settings picker's live preview)."""
        self._kind, self._color, self._image_path = kind, color, image_path
        self._src = None
        if kind == "image" and image_path:
            pm = QPixmap(image_path)
            self._src = pm if not pm.isNull() else None
        self._cache = None
        self.update()

    def paintEvent(self, e):
        key = (self.width(), self.height(), self.devicePixelRatioF(),
               self._kind, self._color, self._image_path, self._src is not None)
        if self._cache is None or self._cache_key != key:
            w, h = max(self.width(), 1), max(self.height(), 1)
            self._cache = cover_pixmap(
                w, h, kind=self._kind, color=self._color, image=self._src,
                project_id=self._project_id, dpr=max(self.devicePixelRatioF(), 1.0),
                radius=self._radius, top_only=self._top_only)
            self._cache_key = key
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)
        p.end()
