"""
Cohort-level aggregation for batch runs.

Scientists rarely analyse one image — they process a plate/condition and report
population statistics. This module turns a batch of per-image measurement
results (from :mod:`velum_core.analysis`) into:

  • a long per-cell table (every cell, tagged with its source image),
  • a per-image summary table (counts + central tendencies),
  • pooled population statistics across the whole cohort.

All outputs are plain columns/rows so they serialise straight to CSV and render
in a QTableWidget without pandas.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# (image_name, measurement_result, coverage_percent)
Record = tuple


def per_image_summary(records: list[Record]) -> tuple[list[str], list[list]]:
    cols = ["image", "n_cells", "median_diam", "mean_area",
            "median_area", "median_circularity", "coverage_%"]
    rows: list[list] = []
    for name, res, cov in records:
        s = res.get("summary", {})
        rows.append([
            name,
            res.get("n_cells", 0),
            round(s.get("diameter", {}).get("median", 0.0), 2),
            round(s.get("area", {}).get("mean", 0.0), 2),
            round(s.get("area", {}).get("median", 0.0), 2),
            round(s.get("circularity", {}).get("median", 0.0), 3),
            round(cov, 2),
        ])
    return cols, rows


def per_cell_long(records: list[Record]) -> tuple[list[str], list[list]]:
    """Every cell across the cohort, one row each, tagged with its image."""
    header: list[str] = []
    rows: list[list] = []
    for name, res, _cov in records:
        if not res.get("rows"):
            continue
        if not header:
            header = ["image"] + [f"{label} ({unit})" if unit else label
                                  for _k, label, unit in res["columns"]]
        for r in res["rows"]:
            rows.append([name] + list(r))
    if not header:
        header = ["image"]
    return header, rows


def population_stats(records: list[Record]) -> dict[str, Any]:
    """Pool selected features across all cells in the cohort."""
    keys = ("area", "diameter", "circularity", "eccentricity", "solidity")
    pooled: dict[str, list[float]] = {k: [] for k in keys}
    n_images = 0
    total_cells = 0
    for _name, res, _cov in records:
        n_images += 1
        total_cells += res.get("n_cells", 0)
        cols = [k for k, _l, _u in res["columns"]]
        for k in keys:
            if k in cols and res.get("rows"):
                idx = cols.index(k)
                pooled[k].extend(float(r[idx]) for r in res["rows"])
    out: dict[str, Any] = {"n_images": n_images, "total_cells": total_cells,
                           "features": {}}
    for k, vals in pooled.items():
        if not vals:
            continue
        a = np.asarray(vals, dtype=np.float64)
        out["features"][k] = {
            "mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std()), "n": int(a.size),
        }
    return out


def pooled_values(records: list[Record], feature: str) -> list[float]:
    vals: list[float] = []
    for _name, res, _cov in records:
        cols = [k for k, _l, _u in res["columns"]]
        if feature in cols and res.get("rows"):
            idx = cols.index(feature)
            vals.extend(float(r[idx]) for r in res["rows"])
    return vals


def _write_csv(path, header, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_cohort_csvs(out_dir, records: list[Record]) -> tuple[str, str]:
    """Write per-cell and per-image CSVs; return their paths."""
    from pathlib import Path
    out = Path(out_dir)
    ph, pr = per_cell_long(records)
    sh, sr = per_image_summary(records)
    cell_csv = out / "cohort_measurements.csv"
    summ_csv = out / "cohort_summary.csv"
    _write_csv(cell_csv, ph, pr)
    _write_csv(summ_csv, sh, sr)
    return str(cell_csv), str(summ_csv)
