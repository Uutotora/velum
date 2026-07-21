"""Velum — the Segment workspace's own image canvas.

Not embedded napari (see ``docs/velum/ARCHITECTURE.md``) — a self-contained
``QWidget`` that renders a :class:`studio.layer_model.LayerList` (image +
labels + points + shapes) with pan/zoom, and turns mouse input into real
edits on the selected ``LabelsLayer`` (paint/erase/fill/pick/polygon) exactly
like napari's own canvas does for a ``Labels`` layer. Compositing is plain
numpy (contrast/gamma/colormap tint for images; per-instance colour + opacity
+ contour for labels; translucent/additive/opaque blending) into one RGB
``QImage`` per repaint-worthy change, which ``QPainter`` then scales/pans —
cheap enough at the image sizes this app works with, and it keeps every pixel
rule in one auditable place instead of spread across GPU shaders.

Coordinate convention throughout: ``(row, col)`` in *image* pixel space
(row=y, col=x), matching numpy indexing and napari's own convention.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor, QCursor, QImage, QMouseEvent, QPainter, QPainterPath, QPen, QPolygonF,
    QTransform, QWheelEvent,
)
from PyQt6.QtWidgets import QWidget

from studio.layer_model import (
    ERASE, FILL, ImageLayer, IMAGE_COLORMAPS, LabelsLayer, LayerList, PAINT, PAN_ZOOM,
    PICK, POLYGON, PointsLayer, ShapesLayer, TRANSFORM,
)

_MIN_ZOOM, _MAX_ZOOM = 0.05, 40.0
_ZOOM_STEP = 1.15
_MAX_ROT = 1.4  # radians, ~80° — short of the projection degenerating
_ROT_SENSITIVITY = 0.01  # radians per pixel dragged


class Canvas(QWidget):
    """The viewport: image + layer compositing, pan/zoom, and edit tools.

    ``mode`` is the single source of truth for what a click does (mirrors
    into the edited ``LabelsLayer.mode`` too, so anything reading the layer
    directly stays consistent). Edits always target :meth:`edit_target` — the
    selected layer if it's a ``LabelsLayer``, else the first one in the list,
    else ``None`` (a no-op with a status hint, rather than silently guessing
    a target or crashing).
    """

    def __init__(self, t: dict, layers: LayerList, *,
                 on_status: Optional[Callable[[str], None]] = None,
                 on_label_picked: Optional[Callable[[int], None]] = None,
                 on_mode_change: Optional[Callable[[str], None]] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._t = t
        self.layers = layers
        self._on_status = on_status
        self._on_label_picked = on_label_picked
        self._on_mode_change = on_mode_change

        self.mode = PAN_ZOOM
        self.grid = False
        self.mip = False          # "3D" toggle: a real volume MIPs; a flat image tilts (below)
        self.transposed = False   # view-only row/col swap — never mutates layer data

        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._fitted = False

        # Pseudo-3D orbit angles (radians) for a flat image under the "3D"
        # toggle — a non-zero starting pitch so it reads as tilted the
        # instant 3D turns on, not flat until you first drag it.
        self._rot_x = 0.4
        self._rot_y = 0.0

        self._panning = False
        self._pan_anchor: Optional[QPointF] = None
        self._pan_anchor_pan: Optional[QPointF] = None
        self._rotating = False
        self._rot_anchor: Optional[QPointF] = None
        self._rot_anchor_angles: Optional[tuple[float, float]] = None
        self._painting = False
        self._paint_last: Optional[tuple[float, float]] = None
        self._polygon_pts: list[tuple[float, float]] = []
        self._hover_img: Optional[tuple[float, float]] = None

        self._composite_cache: Optional[QImage] = None
        self._composite_dirty = True

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(200, 160)
        layers.on_change(self._on_layers_changed)

    # ── layer-model plumbing ─────────────────────────────────────────────────
    def _on_layers_changed(self) -> None:
        self._composite_dirty = True
        self.update()

    def edit_target(self) -> Optional[LabelsLayer]:
        sel = self.layers.selected
        if isinstance(sel, LabelsLayer):
            return sel
        labels = self.layers.by_kind("labels")
        return labels[0] if labels else None

    def undo(self) -> bool:
        """Undo the last edit on the target Labels layer; repaint + report.
        Returns False (no-op) with nothing to undo, so a button can gate on it."""
        target = self.edit_target()
        if target is None or not target.undo():
            return False
        self.layers.notify()
        self.update()
        self._status("Undo")
        return True

    def redo(self) -> bool:
        target = self.edit_target()
        if target is None or not target.redo():
            return False
        self.layers.notify()
        self.update()
        self._status("Redo")
        return True

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._polygon_pts = []
        target = self.edit_target()
        if target is not None:
            target.mode = mode
        cursors = {
            PAINT: Qt.CursorShape.CrossCursor, ERASE: Qt.CursorShape.CrossCursor,
            FILL: Qt.CursorShape.PointingHandCursor, PICK: Qt.CursorShape.PointingHandCursor,
            POLYGON: Qt.CursorShape.CrossCursor,
        }
        self.setCursor(cursors.get(mode, Qt.CursorShape.ArrowCursor))
        self.update()

    # ── image-space geometry ─────────────────────────────────────────────────
    def _base_shape(self) -> Optional[tuple[int, int]]:
        """(H, W) of the current plane, from the first image/labels layer."""
        for layer in self.layers:
            if layer.kind in ("image", "labels") and layer.data.size:
                plane = self._plane_of(layer)
                if self.transposed:
                    return (plane.shape[1], plane.shape[0])
                return (plane.shape[0], plane.shape[1])
        return None

    def _plane_of(self, layer) -> np.ndarray:
        data = layer.data
        ndim_2d = 2 if layer.kind == "labels" else 3
        if data.ndim <= ndim_2d:
            return data
        if self.mip:
            return data.max(axis=0)
        z = min(self.layers.current_z, data.shape[0] - 1)
        return data[z]

    def home(self) -> None:
        """Fit the image to the viewport, centre it, and reset the camera —
        matching real napari's ``reset_view()``, which resets orientation as
        well as pan/zoom, not just the latter.
        """
        shape = self._base_shape()
        if shape is None:
            return
        h, w = shape
        vw, vh = max(1, self.width() - 24), max(1, self.height() - 24)
        self._zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, min(vw / w, vh / h)))
        self._pan = QPointF((self.width() - w * self._zoom) / 2,
                            (self.height() - h * self._zoom) / 2)
        self._fitted = True
        self._rot_x = 0.4
        self._rot_y = 0.0
        self._clamp_pan()
        self.update()

    def _clamp_pan(self) -> None:
        """Keep at least a bit of the image reachable near every edge — pan
        (drag or the wheel's zoom-to-cursor) can no longer push the whole
        image out of the viewport with no way back short of hitting Home.
        A margin capped at half the viewport (and at the image's own scaled
        size, so a tiny zoomed-out image can still be nudged off-centre
        rather than being force-glued to the middle) always keeps the two
        bounds the right way round — see the derivation in this method's
        test coverage rather than re-deriving it by eye every read.
        """
        shape = self._base_shape()
        if shape is None:
            return
        h, w = shape
        scaled_w, scaled_h = w * self._zoom, h * self._zoom
        margin = min(80.0, scaled_w, scaled_h, self.width() * 0.5, self.height() * 0.5)
        lo_x, hi_x = margin - scaled_w, self.width() - margin
        lo_y, hi_y = margin - scaled_h, self.height() - margin
        if lo_x > hi_x:
            lo_x, hi_x = hi_x, lo_x
        if lo_y > hi_y:
            lo_y, hi_y = hi_y, lo_y
        self._pan.setX(max(lo_x, min(hi_x, self._pan.x())))
        self._pan.setY(max(lo_y, min(hi_y, self._pan.y())))

    def widget_to_image(self, pt: QPointF) -> tuple[float, float]:
        col = (pt.x() - self._pan.x()) / self._zoom
        row = (pt.y() - self._pan.y()) / self._zoom
        return (col, row) if not self.transposed else (row, col)

    def image_to_widget(self, row: float, col: float) -> QPointF:
        x, y = (row, col) if self.transposed else (col, row)
        return QPointF(x * self._zoom + self._pan.x(), y * self._zoom + self._pan.y())

    # ── view actions (viewer bar) ────────────────────────────────────────────
    def toggle_grid(self) -> None:
        self.grid = not self.grid
        self.update()

    def toggle_mip(self) -> bool:
        """Toggle the "2D/3D" view state — unconditionally, matching real
        napari's own ``ndisplay`` toggle exactly: it is a camera/view-mode
        switch with no dimensionality guard at all (confirmed against the
        installed napari source — ``ViewerModel.dims.ndisplay`` just flips
        2<->3; a flat 2-D layer still renders, tilted, in the resulting 3-D
        camera). A loaded volume (``n_planes>1``) projects via max-intensity
        (``_plane_of``); a single 2-D image instead gets a perspective tilt
        in ``_draw_pseudo_3d`` — "3-D" always does *something* visible,
        never a silent no-op on flat data the way an earlier version of
        this method did.
        """
        self.mip = not self.mip
        self._composite_dirty = True
        self.update()
        return self.mip

    def toggle_transpose(self) -> None:
        self.transposed = not self.transposed
        self._composite_dirty = True
        self.update()

    def roll_channel(self) -> bool:
        """"Roll dimensions": cycle which single image-kind layer is visible,
        the practical equivalent of rolling through an extra axis when that
        axis is really "one more channel" rather than a literal spatial dim.
        Returns whether it actually rolled — False (a no-op) with only one
        image layer, so the caller can tell the user why nothing happened
        instead of a silent, seemingly-broken click."""
        images = [l for l in self.layers if l.kind == "image"]
        if len(images) < 2:
            return False
        visible = [l for l in images if l.visible]
        current = visible[0] if visible else images[0]
        idx = images.index(current)
        for l in images:
            l.visible = False
        images[(idx + 1) % len(images)].visible = True
        self.layers.notify()
        return True

    def step_z(self, delta: int) -> None:
        if self.layers.n_planes > 1 and not self.mip:
            self.layers.set_current_z(self.layers.current_z + delta)

    # ── compositing ──────────────────────────────────────────────────────────
    def _composited_image(self) -> Optional[QImage]:
        if not self._composite_dirty and self._composite_cache is not None:
            return self._composite_cache
        shape = self._base_shape()
        if shape is None:
            self._composite_cache = None
            self._composite_dirty = False
            return None
        h, w = shape
        scope = QColor(self._t.get("scope", "#0a0c10"))
        canvas = np.empty((h, w, 3), dtype=np.float32)
        canvas[:] = (scope.red(), scope.green(), scope.blue())

        for layer in self.layers:
            if not layer.visible or layer.kind not in ("image", "labels"):
                continue
            plane = self._plane_of(layer)
            if self.transposed:
                plane = plane.T if plane.ndim == 2 else plane.transpose(1, 0, 2)
            if plane.shape[:2] != (h, w):
                continue  # a mismatched layer (rare) is skipped, not stretched
            if layer.kind == "image":
                rgb, alpha = _render_image(layer, plane)
            else:
                rgb, alpha = _render_labels(layer, plane)
            canvas = _blend(canvas, rgb, alpha, layer.blending)

        arr = np.ascontiguousarray(np.clip(canvas, 0, 255).astype(np.uint8))
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        self._composite_cache = qimg
        self._composite_dirty = False
        return qimg

    # ── painting ─────────────────────────────────────────────────────────────
    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        t = self._t
        p.fillRect(self.rect(), QColor(t.get("scope", "#0a0c10")))

        if not self._fitted:
            self.home()

        image = self._composited_image()
        if image is None:
            p.setPen(QColor(t.get("text_muted", "#6c7480")))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
            p.end()
            return

        if self.grid:
            self._paint_grid(p, image)
        else:
            self._paint_overlay(p, image)
        p.end()

    def _paint_overlay(self, p: QPainter, image: QImage) -> None:
        p.save()
        p.translate(self._pan)
        p.scale(self._zoom, self._zoom)
        if self.mip and self.layers.n_planes <= 1:
            self._draw_pseudo_3d(p, image)
        else:
            p.drawImage(0, 0, image)
        self._draw_vectors(p)
        self._draw_cursor(p)
        p.restore()

    def _draw_pseudo_3d(self, p: QPainter, image: QImage) -> None:
        """"3-D" on a flat 2-D image (no z-stack to max-project): real
        napari's ndisplay=3 still changes the camera unconditionally — a
        flat layer renders as a tilted plane viewed at an angle, not
        nothing. This fakes that same idea with a real (if simplified)
        rotate-then-perspective-project of the image rectangle's corners
        around ``self._rot_x``/``self._rot_y`` — genuinely interactive via
        :meth:`mousePressEvent`/:meth:`mouseMoveEvent` drag-to-orbit, not a
        fixed trapezoid. Real 3-D rendering would need a GPU/OpenGL camera
        this QPainter canvas doesn't have — this is a deliberately simple,
        honest substitute that always does *something* visible, never a
        silent no-op, and responds to input the way an orbit camera would.
        """
        w, h = image.width(), image.height()
        if w <= 0 or h <= 0:
            p.drawImage(0, 0, image)
            return
        # Kept well above any reachable |z2| (bounded by ~max(w,h)/2, since
        # |sin|<=1) so focal+z2 never approaches 0 and the quad never flips
        # inside out across the whole rotation range.
        focal = max(w + h, 200.0)
        cos_rx, sin_rx = math.cos(self._rot_x), math.sin(self._rot_x)
        cos_ry, sin_ry = math.cos(self._rot_y), math.sin(self._rot_y)

        def project(x: float, y: float) -> QPointF:
            cx, cy = x - w / 2, y - h / 2
            # Yaw (around the vertical axis) then pitch (around the
            # horizontal axis) of the point, in camera space with z=0 at
            # the image plane; then a perspective divide by depth.
            x1 = cx * cos_ry
            z1 = cx * sin_ry
            y2 = cy * cos_rx + z1 * sin_rx
            z2 = cy * sin_rx - z1 * cos_rx
            scale = focal / max(focal + z2, 1e-3)
            return QPointF(x1 * scale + w / 2, y2 * scale + h / 2)

        src = QPolygonF(QRectF(0, 0, w, h))
        dst = QPolygonF([project(0, 0), project(w, 0), project(w, h), project(0, h)])
        transform = QTransform()
        if QTransform.quadToQuad(src, dst, transform):
            p.save()
            p.setTransform(transform, True)
            p.drawImage(0, 0, image)
            p.restore()
        else:
            p.drawImage(0, 0, image)  # degenerate/self-intersecting quad at an extreme angle — draw flat rather than nothing

    def _paint_grid(self, p: QPainter, image: QImage) -> None:
        """Grid mode: one tile per visible layer, each rendered alone — a
        real, simplified reading of napari's grid mode (which tiles multiple
        *images*; here there is always exactly one image, so tiling by
        *layer* is the useful analogue — e.g. Segmentation next to Ground
        truth). Falls back to the normal overlay when there's only one
        visible layer (nothing to compare side by side).

        Each tile's own auto-fit scale is multiplied by ``self._zoom``, so
        the mouse wheel still zooms while grid mode is on (previously a
        no-op — the wheel silently did nothing, which read as "grid mode
        doesn't work"). Panning independently per tile isn't supported (pan
        is calibrated for one full-canvas image, not a grid of small tiles);
        each tile still zooms from its own centre. A tile is clipped to its
        own cell so zooming in doesn't bleed into its neighbours.
        """
        visible = [l for l in self.layers if l.visible and l.kind in ("image", "labels")]
        if len(visible) <= 1:
            self._paint_overlay(p, image)
            return
        cols = int(np.ceil(np.sqrt(len(visible))))
        rows = int(np.ceil(len(visible) / cols))
        cw, ch = self.width() / cols, self.height() / rows
        shape = self._base_shape()
        h, w = shape
        for i, layer in enumerate(visible):
            plane = self._plane_of(layer)
            if self.transposed:
                plane = plane.T if plane.ndim == 2 else plane.transpose(1, 0, 2)
            if layer.kind == "image":
                rgb, alpha = _render_image(layer, plane)
            else:
                rgb, alpha = _render_labels(layer, plane)
            scope = QColor(self._t.get("scope", "#0a0c10"))
            base = np.empty((h, w, 3), dtype=np.float32)
            base[:] = (scope.red(), scope.green(), scope.blue())
            tile = _blend(base, rgb, alpha, "translucent")
            arr = np.ascontiguousarray(np.clip(tile, 0, 255).astype(np.uint8))
            qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
            r, c = i // cols, i % cols
            cell = QRectF(c * cw + 2, r * ch + 2, cw - 4, ch - 4)
            fit_scale = min(cell.width() / w, cell.height() / h)
            scale = max(0.01, fit_scale * self._zoom)
            p.save()
            p.setClipRect(cell)
            p.translate(cell.x() + (cell.width() - w * scale) / 2,
                       cell.y() + (cell.height() - h * scale) / 2)
            p.scale(scale, scale)
            p.drawImage(0, 0, qimg)
            p.restore()
            p.setPen(QColor(self._t.get("text_muted", "#6c7480")))
            p.drawText(QRectF(cell.x(), cell.y(), cell.width(), 16),
                      Qt.AlignmentFlag.AlignLeft, f"  {layer.name}")

    def _draw_vectors(self, p: QPainter) -> None:
        w = 1.0 / max(self._zoom, 0.001)
        for layer in self.layers:
            if not layer.visible:
                continue
            if isinstance(layer, PointsLayer):
                p.setBrush(QColor(layer.face_color))
                p.setPen(QPen(QColor("#ffffff"), w * 0.6))
                for row, col in layer.points:
                    x, y = (row, col) if self.transposed else (col, row)
                    p.drawEllipse(QPointF(x, y), layer.size / 2, layer.size / 2)
            elif isinstance(layer, ShapesLayer):
                pen = QPen(QColor(layer.edge_color), max(layer.edge_width * w, w))
                p.setPen(pen)
                face = layer.face_color
                p.setBrush(QColor(0, 0, 0, 0) if face == "transparent" else QColor(face))
                for shape in layer.shapes:
                    path = QPainterPath()
                    pts = shape["points"]
                    if not pts:
                        continue
                    for i, (row, col) in enumerate(pts):
                        x, y = (row, col) if self.transposed else (col, row)
                        path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
                    path.closeSubpath()
                    p.drawPath(path)

        if self.mode == POLYGON and self._polygon_pts:
            pen = QPen(QColor(self._t.get("primary", "#4f5bd5")), w * 1.5, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(QColor(0, 0, 0, 0))
            path = QPainterPath()
            for i, (row, col) in enumerate(self._polygon_pts):
                x, y = (row, col) if self.transposed else (col, row)
                path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
            p.drawPath(path)
            for row, col in self._polygon_pts:
                x, y = (row, col) if self.transposed else (col, row)
                p.drawEllipse(QPointF(x, y), w * 3, w * 3)

    def _draw_cursor(self, p: QPainter) -> None:
        if self.mode not in (PAINT, ERASE) or self._hover_img is None:
            return
        target = self.edit_target()
        if target is None:
            return
        row, col = self._hover_img
        x, y = (row, col) if self.transposed else (col, row)
        w = 1.0 / max(self._zoom, 0.001)
        p.setPen(QPen(QColor("#ffffff"), w))
        p.setBrush(QColor(0, 0, 0, 0))
        r = target.brush_size / 2.0
        p.drawEllipse(QPointF(x, y), r, r)

    # ── mouse / keyboard input ───────────────────────────────────────────────
    def wheelEvent(self, e: QWheelEvent) -> None:
        before = self.widget_to_image(e.position())
        factor = _ZOOM_STEP if e.angleDelta().y() > 0 else 1.0 / _ZOOM_STEP
        self._zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, self._zoom * factor))
        after_widget = self.image_to_widget(*before[::-1]) if self.transposed else \
            self.image_to_widget(before[1], before[0])
        self._pan += e.position() - after_widget
        self._clamp_pan()
        self.update()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = e.position()
        if e.button() != Qt.MouseButton.MiddleButton and self.mip and self.layers.n_planes <= 1 \
                and (self.mode == PAN_ZOOM or self.mode == TRANSFORM):
            # "3-D" on a flat image: the drag that would otherwise pan now
            # orbits the pseudo-3-D tilt instead — matching real napari's
            # left-drag-to-rotate camera in a 3-D view. Middle-button still
            # always pans (below), same as it does in 2-D.
            self._rotating = True
            self._rot_anchor = pos
            self._rot_anchor_angles = (self._rot_x, self._rot_y)
            return
        if e.button() == Qt.MouseButton.MiddleButton or self.mode == PAN_ZOOM or self.mode == TRANSFORM:
            self._panning = True
            self._pan_anchor = pos
            self._pan_anchor_pan = QPointF(self._pan)
            return

        col, row = self.widget_to_image(pos)
        sel = self.layers.selected
        z = self.layers.current_z if self.layers.n_planes > 1 else None

        # Points/Shapes take over the click while selected and *some tool
        # other than pan_zoom/transform is active* (those two are always
        # reserved for navigation, checked above, so panning still works
        # with a Points/Shapes layer selected) — no separate "points mode"/
        # "shapes mode": left adds, right removes the nearest point; a
        # Shapes layer reuses the Labels POLYGON tool's click-vertices /
        # double-click-closes flow but appends a real shape instead of
        # rasterising into a label mask.
        if isinstance(sel, PointsLayer):
            if e.button() == Qt.MouseButton.LeftButton:
                sel.add(row, col)
                self.layers.notify()
            elif e.button() == Qt.MouseButton.RightButton:
                idx = sel.nearest(row, col, max_dist=max(sel.size, 10))
                if idx is not None:
                    sel.remove_at(idx)
                    self.layers.notify()
            return

        if isinstance(sel, ShapesLayer) and self.mode == POLYGON:
            if e.button() == Qt.MouseButton.LeftButton:
                self._polygon_pts.append((row, col))
            elif e.button() == Qt.MouseButton.RightButton and self._polygon_pts:
                self._polygon_pts.pop()
            self.update()
            return

        target = self.edit_target()

        if self.mode == POLYGON and e.button() == Qt.MouseButton.LeftButton:
            self._polygon_pts.append((row, col))
            self.update()
            return
        if self.mode == POLYGON and e.button() == Qt.MouseButton.RightButton:
            if self._polygon_pts:
                self._polygon_pts.pop()
            self.update()
            return

        if target is None:
            self._status(f"({int(col)}, {int(row)}) — select a Labels layer to edit")
            return

        if self.mode == PAINT:
            target.begin_edit()  # snapshot once at stroke start (drag paints many dabs)
            self._painting = True
            target.paint(row, col, z)
            self._paint_last = (row, col)
            self.layers.notify()
        elif self.mode == ERASE:
            target.begin_edit()
            self._painting = True
            target.erase(row, col, z)
            self._paint_last = (row, col)
            self.layers.notify()
        elif self.mode == FILL:
            target.begin_edit()
            target.fill(row, col, z)
            self.layers.notify()
        elif self.mode == PICK:
            picked = target.pick(row, col, z)
            self.layers.notify()
            if picked is not None and self._on_label_picked:
                self._on_label_picked(picked)

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        if len(self._polygon_pts) < 3:
            return
        sel = self.layers.selected
        if isinstance(sel, ShapesLayer) and self.mode == POLYGON:
            sel.add("polygon", list(self._polygon_pts))
            self.layers.notify()
            self._polygon_pts = []
            self.update()
            return
        if self.mode == POLYGON:
            target = self.edit_target()
            if target is not None:
                z = self.layers.current_z if self.layers.n_planes > 1 else None
                target.begin_edit()  # one undo step for the whole polygon commit
                target.polygon_fill(self._polygon_pts, z)
                self.layers.notify()
            self._polygon_pts = []
            self.update()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        pos = e.position()

        if self._rotating and self._rot_anchor is not None and self._rot_anchor_angles is not None:
            delta = pos - self._rot_anchor
            rx0, ry0 = self._rot_anchor_angles
            self._rot_x = max(-_MAX_ROT, min(_MAX_ROT, rx0 + delta.y() * _ROT_SENSITIVITY))
            self._rot_y = max(-_MAX_ROT, min(_MAX_ROT, ry0 + delta.x() * _ROT_SENSITIVITY))
            self.update()
            return

        col, row = self.widget_to_image(pos)
        self._hover_img = (row, col)

        if self._panning and self._pan_anchor is not None:
            self._pan = self._pan_anchor_pan + (pos - self._pan_anchor)
            self._clamp_pan()
            self.update()
            return

        if self._painting and self.mode in (PAINT, ERASE):
            target = self.edit_target()
            if target is not None:
                z = self.layers.current_z if self.layers.n_planes > 1 else None
                for r, c in _interpolate(self._paint_last, (row, col),
                                         max(2.0, target.brush_size / 3.0)):
                    (target.paint if self.mode == PAINT else target.erase)(r, c, z)
                self._paint_last = (row, col)
                self.layers.notify()

        shape = self._base_shape()
        label_txt = ""
        target = self.edit_target()
        if target is not None and shape is not None:
            r_i, c_i = int(round(row)), int(round(col))
            if 0 <= r_i < shape[0] and 0 <= c_i < shape[1]:
                plane = target.data if target.data.ndim == 2 else \
                    target.data[min(self.layers.current_z, target.data.shape[0] - 1)]
                lbl = int(plane[r_i, c_i])
                label_txt = f"  ·  label {lbl}" if lbl else ""
        self._status(f"({int(col)}, {int(row)}){label_txt}")
        if not self._painting:
            self.update()  # keep the brush-size cursor preview live

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if self._rotating:
            self._rotating = False
            self._rot_anchor = None
            self._rot_anchor_angles = None
        if self._panning and e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = False
            self._pan_anchor = None
        if self._painting:
            self._painting = False
            self._paint_last = None

    # Single-key tool switches, napari-style, active only while the canvas has
    # focus (so they never fight a text field elsewhere in the screen). The
    # workspace wires on_mode_change to refresh its toolbars/labels-tool
    # highlight after the switch.
    _KEY_MODES = {
        Qt.Key.Key_V: PAN_ZOOM, Qt.Key.Key_B: PAINT, Qt.Key.Key_E: ERASE,
        Qt.Key.Key_F: FILL, Qt.Key.Key_G: POLYGON, Qt.Key.Key_K: PICK,
    }

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key.Key_Up, Qt.Key.Key_PageUp):
            self.step_z(-1)
        elif e.key() in (Qt.Key.Key_Down, Qt.Key.Key_PageDown):
            self.step_z(1)
        elif e.key() in self._KEY_MODES and e.modifiers() == Qt.KeyboardModifier.NoModifier:
            mode = self._KEY_MODES[e.key()]
            self.set_mode(mode)
            if self._on_mode_change:
                self._on_mode_change(mode)
        else:
            super().keyPressEvent(e)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if not self._fitted:
            self.home()

    def _status(self, text: str) -> None:
        if self._on_status:
            suffix = f"  ·  z {self.layers.current_z + 1}/{self.layers.n_planes}" \
                if self.layers.n_planes > 1 and not self.mip else ""
            self._on_status(text + suffix)


def _interpolate(a: Optional[tuple[float, float]], b: tuple[float, float],
                 step: float) -> list[tuple[float, float]]:
    """Sample points between ``a`` and ``b`` no more than ``step`` apart, so a
    fast mouse drag still paints a continuous stroke instead of dots."""
    if a is None:
        return [b]
    dist = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    n = max(1, int(dist / max(step, 0.5)))
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(1, n + 1)]


# ── pixel compositing (module functions: easy to unit-test without Qt) ──────
def _render_image(layer: ImageLayer, plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rgb float32 0..255, alpha float32 0..1) for one ImageLayer plane."""
    lo, hi = layer.contrast_limits
    a = plane.astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3]
    rng = (hi - lo) or 1.0
    norm = np.clip((a - lo) / rng, 0.0, 1.0)
    if layer.gamma != 1.0:
        norm = norm ** (1.0 / max(layer.gamma, 1e-3))
    if norm.ndim == 2:
        tint = np.array(IMAGE_COLORMAPS.get(layer.colormap, (1.0, 1.0, 1.0)), dtype=np.float32)
        rgb = norm[..., None] * tint[None, None, :] * 255.0
    else:
        rgb = norm * 255.0
    alpha = np.full(plane.shape[:2], layer.opacity, dtype=np.float32)
    return rgb, alpha


def _render_labels(layer: LabelsLayer, plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rgb float32 0..255, alpha float32 0..1) for one LabelsLayer plane.

    A filled interior at ``layer.fill_opacity`` (soft, default 0.35) with a
    crisper outline band at ``layer.opacity`` (default 0.7) drawn on top
    when ``contour>0`` — one per-pixel alpha mask reproducing the classic
    app's "fill + border, one colour" convention
    (``predict_widget.py``'s ``_add_filled_labels``, which gets there via
    two *stacked* layers since real napari's own ``contour`` is a fill-XOR-
    outline toggle, not additive, in a single layer). ``contour=0`` still
    means "fill only, no outline band" — a real, selectable state.
    """
    h, w = plane.shape
    ids = np.unique(plane)
    lut = np.zeros((int(ids.max()) + 1 if ids.size else 1, 3), dtype=np.float32)
    for i in ids:
        if i > 0:
            lut[i] = layer.get_color(int(i))
    rgb = lut[np.clip(plane, 0, lut.shape[0] - 1)]

    label_mask = plane > 0
    if layer.show_selected_label:
        label_mask = label_mask & (plane == layer.selected_label)

    alpha = np.where(label_mask, layer.fill_opacity, 0.0).astype(np.float32)
    if layer.contour > 0:
        outline_mask = label_mask & _contour_mask(plane, layer.contour)
        alpha = np.where(outline_mask, layer.opacity, alpha)
    return rgb, alpha


def _contour_mask(plane: np.ndarray, thickness: int) -> np.ndarray:
    boundary = np.zeros(plane.shape, dtype=bool)
    boundary[:-1, :] |= plane[:-1, :] != plane[1:, :]
    boundary[1:, :] |= plane[:-1, :] != plane[1:, :]
    boundary[:, :-1] |= plane[:, :-1] != plane[:, 1:]
    boundary[:, 1:] |= plane[:, :-1] != plane[:, 1:]
    if thickness > 1:
        from scipy import ndimage as ndi
        boundary = ndi.binary_dilation(boundary, iterations=thickness - 1)
    return boundary


def _blend(canvas: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, mode: str) -> np.ndarray:
    a = alpha[..., None]
    if mode == "opaque":
        return np.where(a > 0, rgb, canvas)
    if mode == "additive":
        return np.clip(canvas + rgb * a, 0, 255)
    return canvas * (1 - a) + rgb * a  # translucent (default)
