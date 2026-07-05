"""CellSeg1 — a single coherent stroke icon set, rendered to QIcon/QPixmap.

Replaces the old mix of emoji + stray unicode glyphs. Every icon is a 24×24
stroke path, recoloured on demand so it can pick up the accent on hover/active
states. Purely presentational.
"""
from __future__ import annotations

from functools import lru_cache

from PyQt6.QtCore import Qt, QByteArray, QRectF, QSize
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor

try:
    from PyQt6.QtSvg import QSvgRenderer
    _HAVE_SVG = True
except Exception:  # pragma: no cover - QtSvg missing
    _HAVE_SVG = False

from napari_app.theme import TEXT

# 24×24 stroke paths (no fill). Keep names stable — widgets reference them.
PATHS: dict[str, str] = {
    "predict":   '<path d="M7 5l12 7-12 7z"/>',
    "annotate":  '<path d="M4 20l4-1L19 8a2 2 0 0 0-3-3L5 16z"/>',
    "assistant": '<path d="M12 3l2 5 5 2-5 2-2 5-2-5-5-2 5-2z"/>',
    "train":     '<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16"/>',
    "guide":     '<circle cx="12" cy="12" r="9"/><path d="M12 16v.01M12 8a2.4 2.4 0 0 1 1 4.6c-.7.4-1 .8-1 1.4"/>',
    "run":       '<path d="M7 5l12 7-12 7z"/>',
    "download":  '<path d="M12 3v12M7 10l5 5 5-5M5 21h14"/>',
    "refine":    '<path d="M12 3l2 5 5 2-5 2-2 5-2-5-5-2 5-2z"/>',
    "measure":   '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 10h16M10 10v10"/>',
    "save":      '<path d="M5 3h11l3 3v15H5z"/><path d="M8 3v6h7"/>',
    "csv":       '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8 9v6M12 9v6M16 9v6"/>',
    "settings":  '<path d="M4 8h10M18 8h2M4 16h2M10 16h10"/><circle cx="15" cy="8" r="2.4"/><circle cx="7" cy="16" r="2.4"/>',
    "folder":    '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "image":     '<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="11" r="2"/><path d="M3 17l5-4 4 3 3-2 6 4"/>',
    "chevron":   '<path d="M9 6l6 6-6 6"/>',
    "chevron_down": '<path d="M6 9l6 6 6-6"/>',
    "browse":    '<circle cx="6" cy="12" r="1.4"/><circle cx="12" cy="12" r="1.4"/><circle cx="18" cy="12" r="1.4"/>',
    "diagnose":  '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
    "check":     '<path d="M20 6L9 17l-5-5"/>',
    "warning":   '<path d="M12 3l9 16H3z"/><path d="M12 10v4M12 17v.01"/>',
    "batch":     '<rect x="3" y="4" width="7" height="7" rx="1"/><rect x="14" y="4" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/>',
    "chart":     '<path d="M6 20V10M12 20V4M18 20v-7"/>',
    "layers":    '<path d="M4 7h16M4 12h16M4 17h10"/>',
    "target":    '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/>',
    "pin":       '<path d="M9 4h6l-1 6 3 3v2h-4v5l-1 1-1-1v-5H6v-2l3-3z"/>',
    "log":       '<path d="M4 5h16M4 10h16M4 15h10"/>',
    "send":      '<path d="M4 12l16-8-6 16-3-6-7-2z"/>',
    "spark":     '<path d="M12 3l2 5 5 2-5 2-2 5-2-5-5-2 5-2z"/>',
    "undo":      '<path d="M9 7L4 12l5 5M4 12h11a5 5 0 0 1 0 10"/>',
    "ring":      '<circle cx="12" cy="12" r="8.5"/>',
    "refresh":   '<path d="M20 11a8 8 0 1 0-2.3 5.6"/><path d="M20 5v6h-6"/>',
}


def _svg(name: str, color: str, stroke: float) -> bytes:
    body = PATHS.get(name, PATHS["chevron"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    ).encode("utf-8")


@lru_cache(maxsize=512)
def pixmap(name: str, color: str = TEXT, size: int = 18, stroke: float = 1.6,
           dpr: float = 2.0) -> QPixmap:
    """Render an icon to a crisp QPixmap (cached)."""
    px = QPixmap(int(size * dpr), int(size * dpr))
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.GlobalColor.transparent)
    if not _HAVE_SVG:
        return px
    renderer = QSvgRenderer(QByteArray(_svg(name, color, stroke)))
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(p, QRectF(0, 0, size, size))
    p.end()
    return px


def icon(name: str, color: str = TEXT, size: int = 18, stroke: float = 1.6) -> QIcon:
    """Return a QIcon for a button/nav item."""
    return QIcon(pixmap(name, color, size, stroke))


def icon_size(px: int = 18) -> QSize:
    return QSize(px, px)


# ── Brand mark ──────────────────────────────────────────────────────────────
_BRAND_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" fill="none">'
    '<rect x="2.5" y="2.5" width="35" height="35" rx="11" fill="#4d8fff"/>'
    '<polygon points="30,20 25,28.66 15,28.66 10,20 15,11.34 25,11.34" '
    'fill="none" stroke="#ffffff" stroke-width="2.1" stroke-linejoin="round"/>'
    '<circle cx="20" cy="20" r="3.1" fill="#2fe0a0"/></svg>'
)


@lru_cache(maxsize=8)
def brand_pixmap(size: int = 26, dpr: float = 2.0) -> QPixmap:
    """The CellSeg1 concentric-cell logo mark."""
    px = QPixmap(int(size * dpr), int(size * dpr))
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.GlobalColor.transparent)
    if not _HAVE_SVG:
        return px
    r = QSvgRenderer(QByteArray(_BRAND_SVG.encode("utf-8")))
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    r.render(p, QRectF(0, 0, size, size))
    p.end()
    return px
