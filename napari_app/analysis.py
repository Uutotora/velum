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


def _to_gray(intensity_image: np.ndarray | None) -> np.ndarray | None:
    if intensity_image is None:
        return None
    arr = np.asarray(intensity_image)
    if arr.ndim == 3:
        # Luminosity-weighted grayscale; handles RGB / RGBA.
        arr = arr[..., :3].astype(np.float64)
        arr = arr @ np.array([0.299, 0.587, 0.114])
    return arr.astype(np.float64)


def compute_measurements(
    mask: np.ndarray,
    intensity_image: np.ndarray | None = None,
    pixel_size_um: float = 0.0,
) -> dict[str, Any]:
    """
    Measure every labelled object in ``mask``.

    Parameters
    ----------
    mask : 2-D int array, 0 = background, positive ints = cell ids.
    intensity_image : optional H×W or H×W×C image aligned to ``mask``.
    pixel_size_um : micrometres per pixel; 0 keeps everything in pixels.

    Returns a dict with:
      columns  : list of (key, label, unit_string)
      rows     : list of lists (one per cell, aligned with columns)
      summary  : {key: {"mean","median","std","min","max"}} for numeric cols
      n_cells  : int
      pixel_size_um : echo of the input
    """
    from skimage import measure

    mask = np.ascontiguousarray(mask).astype(np.int32)
    gray = _to_gray(intensity_image)
    if gray is not None and gray.shape != mask.shape:
        # Misaligned intensity image → ignore rather than crash.
        gray = None

    has_intens = gray is not None
    schema = _SCHEMA + (_INTENSITY_SCHEMA if has_intens else [])

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
    if px > 0 and kind == "length":
        return round(value * px, 3)
    return round(value, 3)


def _unit_for(kind: str, px: float) -> str:
    if kind == "area":
        return "µm²" if px > 0 else "px²"
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
    area = s.get("area", {})
    diam = s.get("diameter", {})
    circ = s.get("circularity", {})
    au = next((u for k, _l, u in result["columns"] if k == "area"), "px²")
    lu = next((u for k, _l, u in result["columns"] if k == "diameter"), "px")
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
