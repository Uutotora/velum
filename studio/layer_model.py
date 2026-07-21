"""Velum — the Segment workspace's own layer model.

Studio does not embed napari (see ``docs/velum/ARCHITECTURE.md``'s "Segment tab
specifically" section) — this is our **own** evented layer list, built to
reproduce napari's *interaction model* for a ``Labels`` layer 1:1 (same mode
names, same properties, same defaults — verified against the installed
``napari.layers.labels.labels.Labels``/``_labels_constants.Mode`` source, per
the product owner's "identical settings, take it from napari's open code"
instruction) without any dependency on napari itself. Everything here is
plain Python + numpy/scipy/skimage (all already in the light CI ``test``
dependency-group), evented via plain callback lists rather than Qt signals or
napari's ``psygnal``, so it is importable and unit-testable with nothing
installed beyond the app's own light dependencies.

``LayerList`` is the shared timeline + z-order + selection; ``Canvas``
(``studio/canvas.py``) renders it, the Layers panel (``studio/workspace.py``)
drives it. Neither owns the data — both react to ``LayerList.on_change``.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np

# napari-style Labels edit modes — mirrors napari.layers.labels._labels_constants.Mode
# so the tool strip's behaviour matches napari's muscle memory exactly.
PAN_ZOOM = "pan_zoom"
TRANSFORM = "transform"
PAINT = "paint"
ERASE = "erase"
FILL = "fill"
PICK = "pick"
POLYGON = "polygon"

BLENDING_MODES = ("translucent", "additive", "opaque")

_GOLDEN = 0.6180339887498949  # golden-angle conjugate: max hue spread per +1 id


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    tt = v * (1.0 - s * (1.0 - f))
    r, g, b = ((v, tt, p), (q, v, p), (p, v, tt), (p, q, v), (tt, p, v), (v, p, q))[i]
    return (round(r * 255), round(g * 255), round(b * 255))


def label_color(label_id: int, seed: float = 0.0) -> tuple[int, int, int]:
    """Deterministic, well-separated RGB for an instance id (0 = unused).

    Same goal as napari's hash-based random label colormap (adjacent ids must
    look nothing alike, and a given id always gets the same colour), reached
    with a golden-angle hue rotation instead of a hash table: consecutive ids
    land far apart on the colour wheel rather than drifting slowly. ``seed``
    (0..1) rotates every id at once — the mode row's "shuffle colours" action.
    """
    if label_id <= 0:
        return (0, 0, 0)
    hue = ((label_id * _GOLDEN) + seed) % 1.0
    return _hsv_to_rgb(hue, 0.55, 0.95)


# Single-hue tints for an additive multi-channel composite (DAPI + membrane,
# etc.) — deliberately small and named, not a full colormap engine.
IMAGE_COLORMAPS: dict[str, tuple[float, float, float]] = {
    "gray":    (1.0, 1.0, 1.0),
    "red":     (1.0, 0.0, 0.0),
    "green":   (0.0, 1.0, 0.0),
    "blue":    (0.0, 0.3, 1.0),
    "cyan":    (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow":  (1.0, 1.0, 0.0),
}


class Layer:
    """Shared shape: name, visibility, opacity, blending. Nothing renders here
    — rendering is the Canvas's job, so this stays Qt-free."""

    kind = "layer"

    def __init__(self, name: str, *, visible: bool = True, opacity: float = 1.0,
                 blending: str = "translucent"):
        self.name = name
        self.visible = visible
        self.opacity = opacity
        self.blending = blending

    def to_summary(self) -> tuple[str, str, bool]:
        """``(name, count-string, visible)`` for the Layers panel row."""
        raise NotImplementedError


class ImageLayer(Layer):
    kind = "image"

    def __init__(self, name: str, data: np.ndarray, **kw):
        super().__init__(name, **kw)
        self.data = data
        self.contrast_limits: tuple[float, float] = _default_contrast(data)
        self.gamma = 1.0
        self.colormap = "gray"

    def to_summary(self) -> tuple[str, str, bool]:
        return (self.name, "image", self.visible)


def _default_contrast(data: np.ndarray) -> tuple[float, float]:
    if data.size == 0:
        return (0.0, 255.0)
    lo, hi = float(data.min()), float(data.max())
    return (lo, hi) if hi > lo else (0.0, max(hi, 1.0))


class LabelsLayer(Layer):
    """A napari-``Labels``-equivalent instance mask layer.

    Defaults match ``napari.layers.Labels`` exactly: ``opacity=0.7``,
    ``blending="translucent"``, ``brush_size=10``, ``contiguous=True``,
    ``preserve_labels=False``, ``show_selected_label=False``,
    ``n_edit_dimensions=2``, ``contour=0``, ``selected_label=1``.
    """

    kind = "labels"

    def __init__(self, name: str, data: np.ndarray, **kw):
        kw.setdefault("opacity", 0.7)
        super().__init__(name, **kw)
        self.data = np.ascontiguousarray(data).astype(np.int32)
        # Bounded edit history for undo/redo (napari's Labels has this; the
        # Studio canvas edited masks in place with no way back). Snapshots are
        # full-plane copies -- simple and correct; the cap keeps memory bounded
        # for large volumes. See begin_edit()/undo()/redo() below.
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []
        self._history_limit = 24
        self.selected_label = 1
        self.brush_size = 10
        self.contiguous = True
        self.preserve_labels = False
        self.show_selected_label = False
        self.n_edit_dimensions = 2
        # contour>0 draws an outline (at `opacity`, the crisper of the two)
        # ON TOP OF a translucent fill (at `fill_opacity`) rather than
        # replacing it — real napari's own Labels layer can't do this in one
        # layer (contour is a filled-XOR-outline toggle there), so the
        # classic app's predict_widget.py adds the *same* mask as two
        # stacked layers to get "fill + border" — see _add_filled_labels and
        # its 0.35/0.7 defaults, matched here as fill_opacity/opacity so our
        # single layer reproduces that exact look without the two-layer
        # trick. contour=0 still means "fill only, no outline" — a real,
        # selectable state, not removed. 2px (not velum_core's 1px): ours is
        # a per-pixel boundary mask, not a GPU-rendered line, and a 1px-wide
        # mask is easy to lose entirely once the canvas is zoomed out below
        # 1:1 — confirmed by direct pixel sampling of a real render, not
        # just by eye (see the studio-subproject memory on why that matters).
        self.contour = 2
        self.fill_opacity = 0.35
        self.mode = PAN_ZOOM
        self.color_seed = 0.0
        # Per-id colour override — how "colour cells by <measurement>" recolours
        # the layer without disturbing the default per-instance identity colours.
        self.color_overrides: dict[int, tuple[int, int, int]] = {}

    @property
    def max_label(self) -> int:
        return int(self.data.max()) if self.data.size else 0

    @property
    def n_labels(self) -> int:
        """Count of *distinct* non-zero instances -- the real "cells detected".
        Unlike ``max_label`` (the highest id), this stays correct when ids are
        non-contiguous, e.g. after erasing a cell in the middle of the range,
        so the canvas legend can't disagree with the Results panel's count."""
        if not self.data.size:
            return 0
        return int((np.unique(self.data) > 0).sum())

    def get_color(self, label_id: int) -> tuple[int, int, int]:
        if label_id <= 0:
            return (0, 0, 0)
        if label_id in self.color_overrides:
            return self.color_overrides[label_id]
        return label_color(label_id, self.color_seed)

    def shuffle_colors(self) -> None:
        self.color_seed = random.random()
        self.color_overrides.clear()

    def set_color_overrides(self, overrides: dict[int, tuple]) -> None:
        """Recolour by measurement: ``overrides`` maps id -> 0..1 float RGB(A)
        tuples (``analysis.label_colormap_from_measurement``'s contract) or
        0..255 int RGB — both normalised to 0..255 ints here."""
        out: dict[int, tuple[int, int, int]] = {}
        for k, v in overrides.items():
            r, g, b = v[0], v[1], v[2]
            if max(r, g, b) <= 1.0:
                r, g, b = r * 255, g * 255, b * 255
            out[int(k)] = (round(r), round(g), round(b))
        self.color_overrides = out

    def clear_color_overrides(self) -> None:
        self.color_overrides = {}

    def set_uniform_color(self, rgb: tuple[int, int, int]) -> None:
        """Colour every id currently in ``data`` the same flat colour —
        matches the classic app's ground-truth convention (a fixed colour,
        not per-instance random hues, since GT and prediction are meant to
        read as visually distinct roles rather than compete for the same
        rainbow)."""
        ids = np.unique(self.data)
        self.color_overrides = {int(i): rgb for i in ids if i > 0}

    # ── undo / redo history ──────────────────────────────────────────────────
    def begin_edit(self) -> None:
        """Snapshot ``data`` before a mutation so it can be undone. Call once
        per *stroke* -- a whole paint/erase drag, or one fill/polygon action --
        not per brush dab, or the history fills with near-identical frames.
        Starting a fresh edit invalidates any redo branch, exactly like napari
        and every other editor."""
        self._undo_stack.append(self.data.copy())
        if len(self._undo_stack) > self._history_limit:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> bool:
        """Revert to the previous snapshot. Returns False (a no-op) when there
        is nothing to undo, so callers can gate a toolbar button on it."""
        if not self._undo_stack:
            return False
        self._redo_stack.append(self.data.copy())
        self.data = self._undo_stack.pop()
        return True

    def redo(self) -> bool:
        """Re-apply the last undone snapshot. False when the redo branch is
        empty (nothing undone, or a new edit cleared it)."""
        if not self._redo_stack:
            return False
        self._undo_stack.append(self.data.copy())
        self.data = self._redo_stack.pop()
        return True

    # ── editing ──────────────────────────────────────────────────────────────
    def _edit_planes(self, z: Optional[int]) -> list[np.ndarray]:
        """Planes an edit touches — honours ``n_edit_dimensions`` for a volume:
        2 -> only the current plane; 3 -> every plane (matches napari)."""
        if self.data.ndim == 2 or z is None:
            return [self.data]
        if self.n_edit_dimensions <= 2:
            return [self.data[z]]
        return [self.data[zz] for zz in range(self.data.shape[0])]

    def _plane(self, z: Optional[int]) -> np.ndarray:
        return self.data if (self.data.ndim == 2 or z is None) else self.data[z]

    @staticmethod
    def _brush_mask(plane: np.ndarray, row: float, col: float, brush_size: float) -> np.ndarray:
        r = max(1.0, brush_size) / 2.0
        rr, cc = np.ogrid[: plane.shape[0], : plane.shape[1]]
        return (rr - row) ** 2 + (cc - col) ** 2 <= r * r

    def paint(self, row: float, col: float, z: Optional[int] = None,
              label: Optional[int] = None) -> None:
        """Stamp a circular brush of ``label`` (default: ``selected_label``)."""
        lbl = self.selected_label if label is None else label
        for plane in self._edit_planes(z):
            m = self._brush_mask(plane, row, col, self.brush_size)
            if self.preserve_labels:
                m = m & (plane == 0)
            plane[m] = lbl

    def erase(self, row: float, col: float, z: Optional[int] = None) -> None:
        """Paint with the background label — napari's Erase mode."""
        for plane in self._edit_planes(z):
            m = self._brush_mask(plane, row, col, self.brush_size)
            plane[m] = 0

    def fill(self, row: float, col: float, z: Optional[int] = None) -> None:
        """Bucket-fill the clicked label with ``selected_label``.

        ``contiguous`` restricts the fill to the 4-connected region touching
        the clicked pixel (a flood fill); off, it replaces *every* pixel of
        that label anywhere in the plane — exactly napari's Fill semantics.
        """
        row_i, col_i = int(round(row)), int(round(col))
        for plane in self._edit_planes(z):
            if not (0 <= row_i < plane.shape[0] and 0 <= col_i < plane.shape[1]):
                continue
            target = int(plane[row_i, col_i])
            if target == self.selected_label:
                continue
            if self.contiguous:
                mask = _flood_fill_mask(plane, row_i, col_i)
            else:
                mask = plane == target
            plane[mask] = self.selected_label

    def pick(self, row: float, col: float, z: Optional[int] = None) -> Optional[int]:
        """Set ``selected_label`` to whatever is under (row, col); returns it."""
        plane = self._plane(z)
        row_i, col_i = int(round(row)), int(round(col))
        if not (0 <= row_i < plane.shape[0] and 0 <= col_i < plane.shape[1]):
            return None
        self.selected_label = int(plane[row_i, col_i])
        return self.selected_label

    def polygon_fill(self, vertices: list[tuple[float, float]], z: Optional[int] = None) -> None:
        """Rasterise a closed polygon (row, col vertices) as ``selected_label``
        — napari's Polygon mode (2-D only there; honours ``n_edit_dimensions``
        here too, for consistency with the other tools)."""
        if len(vertices) < 3:
            return
        for plane in self._edit_planes(z):
            mask = _polygon_mask(plane.shape, vertices)
            if self.preserve_labels:
                mask = mask & (plane == 0)
            plane[mask] = self.selected_label

    def to_summary(self) -> tuple[str, str, bool]:
        return (self.name, str(self.max_label), self.visible)


def _flood_fill_mask(plane: np.ndarray, row: int, col: int) -> np.ndarray:
    from scipy import ndimage as ndi

    target = plane[row, col]
    same = plane == target
    structure = ndi.generate_binary_structure(2, 1)  # 4-connectivity
    labeled, _ = ndi.label(same, structure=structure)
    return labeled == labeled[row, col]


def _polygon_mask(shape: tuple[int, int], vertices: list[tuple[float, float]]) -> np.ndarray:
    from skimage.draw import polygon as sk_polygon

    rows = [v[0] for v in vertices]
    cols = [v[1] for v in vertices]
    rr, cc = sk_polygon(rows, cols, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


class PointsLayer(Layer):
    """Prompt/annotation points — napari's ``Points`` layer, 2-D only here
    (a per-plane concept is enough for prompts/corrections; scope-limited
    deliberately rather than reimplementing N-d points)."""

    kind = "points"

    def __init__(self, name: str, points: Optional[list[tuple[float, float]]] = None, **kw):
        super().__init__(name, **kw)
        self.points: list[tuple[float, float]] = list(points or [])
        self.size = 10.0
        self.face_color = "#e0982f"
        self.selected: set[int] = set()

    def add(self, row: float, col: float) -> int:
        self.points.append((row, col))
        return len(self.points) - 1

    def remove_at(self, index: int) -> None:
        if 0 <= index < len(self.points):
            del self.points[index]
            self.selected = {i for i in self.selected if i != index}

    def nearest(self, row: float, col: float, max_dist: float) -> Optional[int]:
        best, best_d = None, max_dist
        for i, (r, c) in enumerate(self.points):
            d = ((r - row) ** 2 + (c - col) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = i, d
        return best

    def to_summary(self) -> tuple[str, str, bool]:
        return (self.name, str(len(self.points)), self.visible)


class ShapesLayer(Layer):
    """Correction/ROI shapes — napari's ``Shapes`` layer, simplified to
    polygons/rectangles/lines/ellipses stored as a point list each."""

    kind = "shapes"

    def __init__(self, name: str, shapes: Optional[list[dict]] = None, **kw):
        super().__init__(name, **kw)
        self.shapes: list[dict] = list(shapes or [])
        self.edge_color = "#4d8fff"
        self.face_color = "transparent"
        self.edge_width = 2.0

    def add(self, shape_type: str, points: list[tuple[float, float]]) -> int:
        self.shapes.append({"type": shape_type, "points": list(points)})
        return len(self.shapes) - 1

    def remove_at(self, index: int) -> None:
        if 0 <= index < len(self.shapes):
            del self.shapes[index]

    def to_summary(self) -> tuple[str, str, bool]:
        return (self.name, str(len(self.shapes)), self.visible)


class LayerList:
    """An ordered, evented collection of layers — ``viewer.layers``,
    reimplemented small and dependency-free: plain-callback events (no Qt, no
    napari's ``psygnal``), so both the Canvas and the Layers panel can react
    to the same mutations without either owning the data.
    """

    def __init__(self):
        self._layers: list[Layer] = []
        self.selected_index: Optional[int] = None
        self.current_z: int = 0
        self._listeners: list[Callable[[], None]] = []

    # ── events ───────────────────────────────────────────────────────────────
    def on_change(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def notify(self) -> None:
        """Public hook for a caller that mutated a layer's data array in place
        (e.g. one brush stroke) and needs listeners (the Canvas, the panel) to
        redraw — mutating a numpy array in place has no event of its own."""
        for cb in list(self._listeners):
            cb()

    # ── container protocol ──────────────────────────────────────────────────
    def __iter__(self):
        return iter(self._layers)

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, i: int) -> Layer:
        return self._layers[i]

    def index_of(self, layer: Layer) -> Optional[int]:
        try:
            return self._layers.index(layer)
        except ValueError:
            return None

    # ── selection ────────────────────────────────────────────────────────────
    @property
    def selected(self) -> Optional[Layer]:
        if self.selected_index is None or not (0 <= self.selected_index < len(self._layers)):
            return None
        return self._layers[self.selected_index]

    def select(self, index: Optional[int]) -> None:
        self.selected_index = index
        self.notify()

    # ── mutation ─────────────────────────────────────────────────────────────
    def add(self, layer: Layer, *, select: bool = True) -> int:
        self._layers.append(layer)
        idx = len(self._layers) - 1
        if select:
            self.selected_index = idx
        self.notify()
        return idx

    def remove(self, index: int) -> None:
        if not (0 <= index < len(self._layers)):
            return
        del self._layers[index]
        if self.selected_index is not None:
            if self.selected_index == index:
                self.selected_index = None
            elif self.selected_index > index:
                self.selected_index -= 1
        self.notify()

    def remove_selected(self) -> None:
        if self.selected_index is not None:
            self.remove(self.selected_index)

    def clear(self) -> None:
        self._layers.clear()
        self.selected_index = None
        self.current_z = 0
        self.notify()

    def toggle_visible(self, index: int) -> None:
        if 0 <= index < len(self._layers):
            self._layers[index].visible = not self._layers[index].visible
            self.notify()

    def move(self, src: int, dst: int) -> None:
        """Reorder a layer (drag in the panel); z-order = list order, last on top."""
        if not (0 <= src < len(self._layers)) or not (0 <= dst < len(self._layers)):
            return
        layer = self._layers.pop(src)
        self._layers.insert(dst, layer)
        if self.selected_index == src:
            self.selected_index = dst
        self.notify()

    # ── queries ──────────────────────────────────────────────────────────────
    def by_kind(self, kind: str) -> list[Layer]:
        return [l for l in self._layers if l.kind == kind]

    def find(self, name: str) -> Optional[Layer]:
        for l in self._layers:
            if l.name == name:
                return l
        return None

    @property
    def n_planes(self) -> int:
        """Frames along the shared z/t axis — 1 unless an image/labels layer
        holds a (Z,H,W[,C]) volume, in which case every layer navigates the
        same index (matches napari's shared ``dims.current_step``).

        A trailing dim of 3/4 on an ``image`` layer is RGB(A) channels of a
        single 2-D plane, not a z-axis — so images need ``ndim >= 4`` to
        count as a volume, while ``labels`` (never has a channel axis) needs
        only ``ndim >= 3``. Mirrors ``Canvas._plane_of``'s identical rule.
        """
        planes = []
        for l in self._layers:
            if l.kind == "labels" and l.data.ndim >= 3:
                planes.append(l.data.shape[0])
            elif l.kind == "image" and l.data.ndim >= 4:
                planes.append(l.data.shape[0])
        return max(planes) if planes else 1

    def set_current_z(self, z: int) -> None:
        n = self.n_planes
        self.current_z = max(0, min(z, n - 1))
        self.notify()

    def unique_name(self, base: str) -> str:
        """``base`` if free, else ``"base [2]"``, ``"base [3]"``, … — mirrors
        napari's own auto-rename for a repeated "New labels layer" click."""
        existing = {l.name for l in self._layers}
        if base not in existing:
            return base
        i = 2
        while f"{base} [{i}]" in existing:
            i += 1
        return f"{base} [{i}]"
