"""Velum — export a segmented project as a reusable, re-trainable dataset.

This closes the loop the product is built around: *collect real images →
segment → proofread → **export a dataset** → train on it*. A Velum project is
already a named cohort of images plus their (predicted, and increasingly
hand-corrected) instance masks; this module turns that into a standard,
self-describing folder anyone — or Velum's own Train tab — can consume:

    <dataset>/
      images/<stem><ext>        the source image, copied verbatim (bytes
                                preserved, so ND2/CZI/OME-TIFF survive intact)
      masks/<stem>.png          the instance mask as a uint16 PNG, one integer
                                id per cell, 0 = background (the exact encoding
                                the rest of Velum reads/writes — see
                                ``segment_controller.save_mask``)
      measurements/<stem>.csv   per-cell morphometry (optional)
      dataset.json              the manifest / dataset card (provenance below)
      README.md                 a human-readable version of the same

The layout is deliberately the one Velum's training already discovers:
``train_controller.find_mask_for_image`` looks for ``masks/<stem>.*`` beside an
image folder, so pointing the Train tab's *images* at ``images/`` and *masks*
at ``masks/`` trains on the exported dataset with zero conversion.

Everything here is **Qt-free and torch-free** — only ``numpy`` + ``cv2`` +
stdlib (and, when measurements are requested, ``velum_core.analysis``, itself
numpy-only) — so it imports and unit-tests in CI's light group without a
display or a GPU.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

# Bump when the on-disk shape changes in a way a reader must branch on.
DATASET_FORMAT = "velum-instance-seg"
DATASET_VERSION = 1

# Masks are written as uint16 PNGs; ids above this can't be represented and are
# recorded as a caveat in the manifest rather than silently wrapping.
_MAX_LABEL = 65535

# A (image_path, instance_mask) pair. The mask is an integer label array
# (0 = background, each distinct positive id = one cell), any integer dtype.
ExportItem = "tuple[str | Path, np.ndarray]"


def count_instances(mask: np.ndarray) -> int:
    """Number of distinct non-zero instance ids in ``mask``."""
    vals = np.unique(np.asarray(mask))
    return int(np.count_nonzero(vals != 0))


def dedupe_stems(paths: Sequence[str | Path]) -> list[str]:
    """Filesystem-safe, collision-free stems aligned to ``paths`` order.

    Two source images with the same filename in different folders would map to
    the same ``masks/<stem>.png`` and clobber each other; append ``-1``,
    ``-2``… to later duplicates so image and mask stems stay unique *and*
    aligned to each other (training pairs them by stem)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for p in paths:
        base = Path(p).stem or "image"
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}-{seen[base]}")
    return out


def assign_splits(n: int, val_fraction: float, seed: int = 0) -> list[str]:
    """Deterministic per-item ``"train"``/``"val"`` labels for ``n`` items.

    ``val_fraction <= 0`` puts everything in ``train`` (the default — a dataset
    you'll split later, or train one-shot on). A positive fraction holds out a
    reproducible random subset for validation; the same (n, fraction, seed)
    always yields the same assignment so a re-export is stable."""
    labels = ["train"] * n
    if n <= 0 or val_fraction <= 0:
        return labels
    n_val = min(n, max(1, round(n * float(val_fraction))))
    rng = np.random.default_rng(seed)
    for idx in rng.permutation(n)[:n_val]:
        labels[int(idx)] = "val"
    return labels


@dataclass
class ImageRecord:
    """One image's entry in the manifest (paths are dataset-relative)."""
    image: str
    mask: str
    stem: str
    cells: int
    height: int
    width: int
    split: str
    source: str = ""                      # original absolute path (provenance)
    measurements: Optional[str] = None    # relative csv path, if written

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["measurements"] is None:
            d.pop("measurements")
        return d


@dataclass
class DatasetManifest:
    name: str
    project_id: str = ""
    engine: str = ""
    model_name: str = ""
    vit_name: str = ""
    pixel_size_um: Optional[float] = None
    params: dict[str, Any] = field(default_factory=dict)
    images: list[ImageRecord] = field(default_factory=list)
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        n_train = sum(1 for r in self.images if r.split == "train")
        n_val = sum(1 for r in self.images if r.split == "val")
        d: dict[str, Any] = {
            "format": DATASET_FORMAT,
            "version": DATASET_VERSION,
            "name": self.name,
            "created": self.created or _now_iso(),
            "task": "instance-segmentation",
            "labels": ["cell"],
            "mask_encoding": (
                "uint16 PNG, one integer id per instance, 0 = background; "
                f"ids above {_MAX_LABEL} are not representable"
            ),
            "source": {
                "project_id": self.project_id,
                "engine": self.engine,
                "model_name": self.model_name,
                "vit_name": self.vit_name,
                "pixel_size_um": self.pixel_size_um,
            },
            "params": self.params,
            "counts": {
                "n_images": len(self.images),
                "n_cells": sum(r.cells for r in self.images),
                "n_train": n_train,
                "n_val": n_val,
            },
            "images": [r.to_dict() for r in self.images],
        }
        return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_mask_png(mask: np.ndarray, out_path: Path) -> None:
    """Write an instance mask as a uint16 PNG (Velum's canonical encoding)."""
    import cv2
    arr = np.asarray(mask)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), arr.astype(np.uint16))


def readme_text(manifest: dict[str, Any]) -> str:
    """A human-readable dataset card mirroring ``dataset.json``."""
    c = manifest["counts"]
    s = manifest["source"]
    px = s.get("pixel_size_um")
    px_line = f"{px} µm/pixel" if px else "not calibrated"
    lines = [
        f"# {manifest['name']}",
        "",
        f"Instance-segmentation dataset exported from Velum "
        f"({manifest['created']}).",
        "",
        f"- **Images:** {c['n_images']}  ·  **Cells:** {c['n_cells']}",
        f"- **Split:** {c['n_train']} train / {c['n_val']} val",
        f"- **Segmented with:** {s.get('engine') or 'unknown'}"
        + (f" · model `{s['model_name']}`" if s.get("model_name") else ""),
        f"- **Pixel size:** {px_line}",
        "",
        "## Layout",
        "",
        "```",
        "images/<stem><ext>       source image, copied verbatim",
        "masks/<stem>.png         uint16 instance mask, one id per cell, 0 = background",
    ]
    if any("measurements" in r for r in manifest["images"]):
        lines.append("measurements/<stem>.csv  per-cell morphometry")
    lines += [
        "dataset.json             machine-readable manifest",
        "```",
        "",
        "## Re-training on this dataset",
        "",
        "The layout is the one Velum's training already understands: point the "
        "Train tab's *images* folder at `images/` and its *masks* folder at "
        "`masks/` — each `masks/<stem>.png` pairs with `images/<stem>` by name. "
        "Masks are standard integer-label images, so any instance-segmentation "
        "trainer that reads label PNGs can consume them too.",
        "",
    ]
    return "\n".join(lines)


def export_dataset(
    output_dir: str | Path,
    items: Sequence[tuple[str | Path, np.ndarray]],
    *,
    name: str,
    project_id: str = "",
    engine: str = "",
    model_name: str = "",
    vit_name: str = "",
    pixel_size_um: Optional[float] = None,
    params: Optional[dict[str, Any]] = None,
    include_measurements: bool = False,
    val_fraction: float = 0.0,
    seed: int = 0,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    """Write ``items`` (image path + instance mask) as a dataset under
    ``output_dir`` and return the manifest dict.

    Raises ``ValueError`` if ``items`` is empty (nothing segmented to export) —
    the caller surfaces that to the user instead of writing an empty dataset.
    """
    items = list(items)
    if not items:
        raise ValueError("No segmented images to export.")

    out = Path(output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    if include_measurements:
        (out / "measurements").mkdir(parents=True, exist_ok=True)

    stems = dedupe_stems([p for p, _ in items])
    splits = assign_splits(len(items), val_fraction, seed)
    total = len(items)
    records: list[ImageRecord] = []

    for i, ((image_path, mask), stem, split) in enumerate(zip(items, stems, splits)):
        src = Path(image_path)
        ext = src.suffix or ".png"
        img_rel = f"images/{stem}{ext}"
        mask_rel = f"masks/{stem}.png"

        # Copy the source image verbatim — no re-encode, so multi-channel /
        # z-stack microscopy formats (ND2/CZI/OME-TIFF) are preserved exactly.
        if src.exists():
            shutil.copy2(src, out / img_rel)
        arr = np.asarray(mask)
        _write_mask_png(arr, out / mask_rel)

        h, w = (int(arr.shape[0]), int(arr.shape[1])) if arr.ndim >= 2 else (0, 0)
        rec = ImageRecord(
            image=img_rel, mask=mask_rel, stem=stem,
            cells=count_instances(arr), height=h, width=w,
            split=split, source=str(src),
        )
        if include_measurements:
            from velum_core import analysis
            result = analysis.compute_measurements(arr, pixel_size_um=pixel_size_um)
            csv_rel = f"measurements/{stem}.csv"
            (out / csv_rel).write_text(analysis.rows_as_csv(result), encoding="utf-8")
            rec.measurements = csv_rel
        records.append(rec)

        if on_progress is not None:
            on_progress(i + 1, total)

    manifest = DatasetManifest(
        name=name, project_id=project_id, engine=engine, model_name=model_name,
        vit_name=vit_name, pixel_size_um=pixel_size_um, params=params or {},
        images=records, created=_now_iso(),
    ).to_dict()

    (out / "dataset.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "README.md").write_text(readme_text(manifest), encoding="utf-8")
    return manifest
