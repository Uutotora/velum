"""
Quantitative morphometry for instance-segmentation masks.

Turns a labelled mask (0 = background, 1..N = cell ids) into per-cell
measurements that microscopists actually report: area, perimeter,
equivalent diameter, circularity, elongation, convexity and — when an
intensity image is supplied — per-cell intensity statistics.

The module deliberately relies only on stable, long-lived scikit-image
region attributes and computes derived features (circularity, aspect
ratio, equivalent diameter) with plain NumPy so results stay identical
across scikit-image versions. All physical quantities are reported in
pixels by default and in micrometres when a pixel size is provided.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# Column order shown in the table and written to CSV. Each entry is
# (key, human label, kind) where kind drives unit-scaling and formatting:
#   "area"    → scales with pixel_size²
#   "length"  → scales with pixel_size
#   "ratio"   → dimensionless [0..1]-ish
#   "angle"   → degrees
#   "coord"   → pixel coordinate (never scaled)
#   "intens"  → intensity value (never scaled)
_SCHEMA: list[tuple[str, str, str]] = [
    ("cell_id",        "Cell",         "id"),
    ("area",           "Area",         "area"),
    ("perimeter",      "Perimeter",    "length"),
    ("diameter",       "Eq. diameter", "length"),
    ("major_axis",     "Major axis",   "length"),
    ("minor_axis",     "Minor axis",   "length"),
    ("circularity",    "Circularity",  "ratio"),
    ("aspect_ratio",   "Aspect ratio", "ratio"),
    ("eccentricity",   "Eccentricity", "ratio"),
    ("solidity",       "Solidity",     "ratio"),
    ("extent",         "Extent",       "ratio"),
    ("orientation",    "Orientation",  "angle"),
    ("centroid_x",     "Centroid X",   "coord"),
    ("centroid_y",     "Centroid Y",   "coord"),
]

_INTENSITY_SCHEMA: list[tuple[str, str, str]] = [
    ("mean_intensity", "Mean int.",  "intens"),
    ("max_intensity",  "Max int.",   "intens"),
    ("min_intensity",  "Min int.",   "intens"),
    ("integrated_intensity", "Integ. int.", "intens"),
]

# 3-D counterpart of _SCHEMA. Deliberately smaller: several 2-D-only
# scikit-image regionprops properties have no 3-D equivalent at all
# (perimeter, eccentricity, a single orientation angle) and are dropped
# rather than faked. "volume" replaces "area" (regionprops' own "area"
# property is already a voxel count for an ndim=3 label image — no new
# computation needed, just a µm³ instead of µm² unit).
_SCHEMA_3D: list[tuple[str, str, str]] = [
    ("cell_id",        "Cell",         "id"),
    ("volume",         "Volume",       "volume"),
    ("diameter",       "Eq. diameter", "length"),
    ("major_axis",     "Major axis",   "length"),
    ("minor_axis",     "Minor axis",   "length"),
    ("aspect_ratio",   "Aspect ratio", "ratio"),
    ("solidity",       "Solidity",     "ratio"),
    ("extent",         "Extent",       "ratio"),
    ("centroid_x",     "Centroid X",   "coord"),
    ("centroid_y",     "Centroid Y",   "coord"),
    ("centroid_z",     "Centroid Z",   "coord"),
]


def _to_gray(intensity_image: np.ndarray | None) -> np.ndarray | None:
    if intensity_image is None:
        return None
    arr = np.asarray(intensity_image)
    if arr.ndim == 3:
        # Luminosity-weighted grayscale; handles RGB / RGBA.
        arr = arr[..., :3].astype(np.float64)
        arr = arr @ np.array([0.299, 0.587, 0.114])
    return arr.astype(np.float64)


def _to_gray_volume(intensity_image: np.ndarray | None,
                    mask_shape: tuple) -> np.ndarray | None:
    """3-D counterpart of :func:`_to_gray`: collapses a ``(Z,H,W,3)`` colour
    volume (or an already-gray ``(Z,H,W)`` one) to a ``(Z,H,W)`` float64
    volume aligned with a 3-D mask. ``None`` when absent or misaligned."""
    if intensity_image is None:
        return None
    arr = np.asarray(intensity_image)
    if arr.ndim == 4:
        arr = arr[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    arr = arr.astype(np.float64)
    if arr.shape != mask_shape:
        return None
    return arr


def _channel_columns(mask, channel_intensities, channel_names):
    """Per-channel mean-intensity schema + per-label means for a raw stack.

    Returns ``(schema_entries, means)`` where ``schema_entries`` is a list of
    ``(key, label, "intens")`` and ``means`` is a parallel list of arrays
    indexed by label (element ``k`` = mean intensity of cell ``k``). Both are
    empty / ``None`` when no aligned multi-channel stack is supplied, so the
    single-image path is unaffected.
    """
    if channel_intensities is None:
        return [], None
    stack = np.asarray(channel_intensities)
    if stack.ndim == 2:
        stack = stack[:, :, None]
    if stack.ndim != 3 or stack.shape[:2] != mask.shape:
        # Misaligned / malformed → ignore rather than crash.
        return [], None

    c = stack.shape[2]
    names = list(channel_names) if channel_names else [f"Channel {i}" for i in range(c)]
    if len(names) != c:
        names = [f"Channel {i}" for i in range(c)]

    labels = mask.ravel().astype(np.int64)
    n = int(labels.max()) + 1 if labels.size else 1
    counts = np.bincount(labels, minlength=n).astype(np.float64)
    counts[counts == 0] = 1.0

    schema, means = [], []
    for i in range(c):
        sums = np.bincount(labels, weights=stack[:, :, i].ravel().astype(np.float64),
                           minlength=n)
        means.append(sums / counts)
        schema.append((f"ch{i}_mean", f"{names[i]} mean", "intens"))
    return schema, means


def compute_measurements(
    mask: np.ndarray,
    intensity_image: np.ndarray | None = None,
    pixel_size_um: float = 0.0,
    channel_intensities: np.ndarray | None = None,
    channel_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Measure every labelled object in ``mask``.

    Parameters
    ----------
    mask : 2-D ``(H,W)`` or 3-D ``(Z,H,W)`` int array, 0 = background,
        positive ints = cell ids. A 3-D mask (a z-stack/time-lapse result)
        dispatches to a smaller, 3-D-specific schema — see
        :func:`_compute_measurements_3d`; ``channel_intensities`` isn't wired
        up for volumes yet, only ``intensity_image``.
    intensity_image : optional H×W / H×W×C image (or, for a 3-D mask,
        Z×H×W / Z×H×W×C) aligned to ``mask``.
    pixel_size_um : micrometres per pixel; 0 keeps everything in pixels.
    channel_intensities : optional raw ``H×W×C`` stack aligned to ``mask``.
        When given, a per-channel mean-intensity column is appended for every
        channel (values are the *raw* stack intensities, not the normalised
        display image). This is the multi-channel path; leave it ``None`` for
        the ordinary single-image behaviour.
    channel_names : optional labels for those channels (defaults to
        ``Channel 0..C-1``).

    Returns a dict with:
      columns  : list of (key, label, unit_string)
      rows     : list of lists (one per cell, aligned with columns)
      summary  : {key: {"mean","median","std","min","max"}} for numeric cols
      n_cells  : int
      pixel_size_um : echo of the input
    """
    mask = np.ascontiguousarray(mask).astype(np.int32)
    if mask.ndim == 3:
        return _compute_measurements_3d(mask, intensity_image, pixel_size_um)

    from skimage import measure

    gray = _to_gray(intensity_image)
    if gray is not None and gray.shape != mask.shape:
        # Misaligned intensity image → ignore rather than crash.
        gray = None

    ch_schema, ch_means = _channel_columns(mask, channel_intensities, channel_names)

    has_intens = gray is not None
    schema = _SCHEMA + (_INTENSITY_SCHEMA if has_intens else []) + ch_schema

    px = float(pixel_size_um) if pixel_size_um and pixel_size_um > 0 else 0.0
    columns = [(key, label, _unit_for(kind, px)) for key, label, kind in schema]

    if mask.max() <= 0:
        return {
            "columns": columns,
            "rows": [],
            "summary": {},
            "n_cells": 0,
            "pixel_size_um": px,
        }

    props = measure.regionprops(mask, intensity_image=gray)
    rows: list[list[float]] = []
    for p in props:
        area = float(p.area)
        perim = float(_safe(p, ("perimeter",), 0.0))
        major = float(_safe(p, ("axis_major_length", "major_axis_length"), 0.0))
        minor = float(_safe(p, ("axis_minor_length", "minor_axis_length"), 0.0))
        eq_diam = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0
        circ = (4.0 * math.pi * area / (perim * perim)) if perim > 0 else 0.0
        circ = min(circ, 1.0)
        aspect = (major / minor) if minor > 0 else 0.0
        cy, cx = p.centroid  # (row, col) = (y, x)
        orient_deg = math.degrees(float(_safe(p, ("orientation",), 0.0)))

        row = [
            int(p.label),
            _scale(area, "area", px),
            _scale(perim, "length", px),
            _scale(eq_diam, "length", px),
            _scale(major, "length", px),
            _scale(minor, "length", px),
            round(circ, 4),
            round(aspect, 3),
            round(float(_safe(p, ("eccentricity",), 0.0)), 4),
            round(float(_safe(p, ("solidity",), 0.0)), 4),
            round(float(_safe(p, ("extent",), 0.0)), 4),
            round(orient_deg, 1),
            round(float(cx), 1),
            round(float(cy), 1),
        ]
        if has_intens:
            mean_i = float(_safe(p, ("intensity_mean", "mean_intensity"), 0.0))
            max_i = float(_safe(p, ("intensity_max", "max_intensity"), 0.0))
            min_i = float(_safe(p, ("intensity_min", "min_intensity"), 0.0))
            row += [
                round(mean_i, 2),
                round(max_i, 2),
                round(min_i, 2),
                round(mean_i * area, 1),
            ]
        if ch_means is not None:
            lbl = int(p.label)
            row += [round(float(means[lbl]), 2) for means in ch_means]
        rows.append(row)

    rows.sort(key=lambda r: r[0])
    summary = _summarize(schema, rows)
    return {
        "columns": columns,
        "rows": rows,
        "summary": summary,
        "n_cells": len(rows),
        "pixel_size_um": px,
    }


def _compute_measurements_3d(
    mask: np.ndarray,
    intensity_image: np.ndarray | None,
    pixel_size_um: float,
) -> dict[str, Any]:
    """3-D counterpart of :func:`compute_measurements` for a ``(Z,H,W)``
    label volume (a z-stack/time-lapse prediction). Same return shape, a
    smaller schema (see :data:`_SCHEMA_3D`)."""
    from skimage import measure

    gray = _to_gray_volume(intensity_image, mask.shape)

    px = float(pixel_size_um) if pixel_size_um and pixel_size_um > 0 else 0.0
    has_intens = gray is not None
    schema = _SCHEMA_3D + (_INTENSITY_SCHEMA if has_intens else [])
    columns = [(key, label, _unit_for(kind, px)) for key, label, kind in schema]

    if mask.max() <= 0:
        return {
            "columns": columns,
            "rows": [],
            "summary": {},
            "n_cells": 0,
            "pixel_size_um": px,
        }

    props = measure.regionprops(mask, intensity_image=gray)
    rows: list[list[float]] = []
    for p in props:
        volume = float(p.area)   # regionprops' "area" is a voxel count for ndim=3
        major = float(_safe(p, ("axis_major_length", "major_axis_length"), 0.0))
        minor = float(_safe(p, ("axis_minor_length", "minor_axis_length"), 0.0))
        eq_diam = (6.0 * volume / math.pi) ** (1.0 / 3.0) if volume > 0 else 0.0
        aspect = (major / minor) if minor > 0 else 0.0
        cz, cy, cx = p.centroid  # (plane, row, col) = (z, y, x)

        row = [
            int(p.label),
            _scale(volume, "volume", px),
            _scale(eq_diam, "length", px),
            _scale(major, "length", px),
            _scale(minor, "length", px),
            round(aspect, 3),
            round(float(_safe(p, ("solidity",), 0.0)), 4),
            round(float(_safe(p, ("extent",), 0.0)), 4),
            round(float(cx), 1),
            round(float(cy), 1),
            round(float(cz), 1),
        ]
        if has_intens:
            mean_i = float(_safe(p, ("intensity_mean", "mean_intensity"), 0.0))
            max_i = float(_safe(p, ("intensity_max", "max_intensity"), 0.0))
            min_i = float(_safe(p, ("intensity_min", "min_intensity"), 0.0))
            row += [
                round(mean_i, 2),
                round(max_i, 2),
                round(min_i, 2),
                round(mean_i * volume, 1),
            ]
        rows.append(row)

    rows.sort(key=lambda r: r[0])
    summary = _summarize(schema, rows)
    return {
        "columns": columns,
        "rows": rows,
        "summary": summary,
        "n_cells": len(rows),
        "pixel_size_um": px,
    }


def _safe(prop, names, default: float) -> float:
    """Read the first available regionprops attribute among ``names``.

    ``names`` is a tuple of candidate attribute names, newest first, so we
    prefer the current scikit-image spelling (``axis_major_length``) and fall
    back to the legacy one (``major_axis_length``) without triggering
    deprecation warnings on modern versions.
    """
    import warnings
    for name in names:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val = getattr(prop, name)
        except Exception:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            return default
        return default if math.isnan(f) or math.isinf(f) else f
    return default


def _scale(value: float, kind: str, px: float) -> float:
    if px > 0 and kind == "area":
        return round(value * px * px, 3)
    if px > 0 and kind == "volume":
        return round(value * px * px * px, 3)
    if px > 0 and kind == "length":
        return round(value * px, 3)
    return round(value, 3)


def _unit_for(kind: str, px: float) -> str:
    if kind == "area":
        return "µm²" if px > 0 else "px²"
    if kind == "volume":
        return "µm³" if px > 0 else "px³"
    if kind == "length":
        return "µm" if px > 0 else "px"
    if kind == "angle":
        return "°"
    if kind in ("ratio", "id", "coord", "intens"):
        return {"coord": "px", "intens": "a.u."}.get(kind, "")
    return ""


def _summarize(schema, rows) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    arr = np.asarray(rows, dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    for idx, (key, _label, kind) in enumerate(schema):
        if kind == "id":
            continue
        col = arr[:, idx]
        out[key] = {
            "mean": float(np.mean(col)),
            "median": float(np.median(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return out


def summary_line(result: dict[str, Any]) -> str:
    """One-line headline used in the Predict panel."""
    n = result["n_cells"]
    if n == 0:
        return "No cells detected"
    s = result["summary"]
    diam = s.get("diameter", {})
    lu = next((u for k, _l, u in result["columns"] if k == "diameter"), "px")
    if "volume" in s:
        # 3-D result (_compute_measurements_3d) — no area/circularity columns.
        vol = s["volume"]
        vu = next((u for k, _l, u in result["columns"] if k == "volume"), "px³")
        return (
            f"Ø {diam.get('median', 0):.1f} {lu} (median)  ·  "
            f"volume {vol.get('mean', 0):.1f} {vu} (mean)  ·  "
            f"{n} cells across the volume"
        )
    area = s.get("area", {})
    circ = s.get("circularity", {})
    au = next((u for k, _l, u in result["columns"] if k == "area"), "px²")
    return (
        f"Ø {diam.get('median', 0):.1f} {lu} (median)  ·  "
        f"area {area.get('mean', 0):.1f} {au} (mean)  ·  "
        f"circularity {circ.get('median', 0):.2f}"
    )


def rows_as_csv(result: dict[str, Any]) -> str:
    """Serialise the full measurement table to CSV text (header + rows)."""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    header = [f"{label} ({unit})" if unit else label
              for _key, label, unit in result["columns"]]
    w.writerow(header)
    for row in result["rows"]:
        w.writerow(row)
    return buf.getvalue()


def label_colormap_from_measurement(result: dict[str, Any], key: str,
                                    cmap_name: str = "viridis") -> dict[int, tuple]:
    """Colour each cell by one of its measured columns instead of a random
    per-instance colour — a "classify by measurement" heatmap over the
    population, the way QuPath/CellProfiler let you colour detections by a
    computed value rather than just identity.

    Values are min-max normalised across the *current* population and mapped
    through a perceptually-uniform, colourblind-safe colormap — viridis by
    default, matplotlib's own modern default (it replaced jet specifically
    because jet's perceptual non-uniformity distorts how differences in the
    underlying data are perceived; unsuitable for quantitative figures).

    Returns ``{cell_id: (r, g, b, a)}`` in 0..1 floats (matching the format
    napari's ``DirectLabelColormap`` expects), or ``{}`` if ``key`` isn't one
    of ``result["columns"]``, there are no rows, or the column isn't numeric
    (e.g. asking to colour by "cell_id" itself makes no sense — the caller is
    expected to only offer the non-id columns as choices).
    """
    cols = [k for k, _label, _unit in result["columns"]]
    if key not in cols or key == "cell_id" or not result["rows"]:
        return {}
    idx = cols.index(key)
    idx_id = cols.index("cell_id")

    try:
        values = [float(row[idx]) for row in result["rows"]]
    except (TypeError, ValueError):
        return {}
    ids = [int(row[idx_id]) for row in result["rows"]]

    from matplotlib import colormaps
    cmap = colormaps[cmap_name]

    lo, hi = min(values), max(values)
    if hi <= lo:
        # No spread in this population (e.g. a single cell, or all-identical
        # values) — colour every cell the same rather than dividing by zero.
        rgba = tuple(cmap(0.5))
        return {i: rgba for i in ids}
    return {i: tuple(cmap((v - lo) / (hi - lo))) for i, v in zip(ids, values)}


def measurement_range(result: dict[str, Any], key: str) -> tuple[float, float] | None:
    """The (min, max) of one measured column across the current population —
    the range a "color cells by" legend needs to label its gradient.

    A separate function rather than folding into
    ``label_colormap_from_measurement``'s return value so that function's
    contract (an id -> rgba dict, nothing else) stays simple for its
    existing callers. ``None`` under the same conditions that function
    itself treats as "not a real choice" (unknown/id column, no rows,
    non-numeric).
    """
    cols = [k for k, _label, _unit in result["columns"]]
    if key not in cols or key == "cell_id" or not result["rows"]:
        return None
    idx = cols.index(key)
    try:
        values = [float(row[idx]) for row in result["rows"]]
    except (TypeError, ValueError):
        return None
    return (min(values), max(values))
