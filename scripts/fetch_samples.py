#!/usr/bin/env python
"""Drop a few sample images into Velum and create a ready-to-segment project.

Run this once to have something to test the app on immediately — no dataset of
your own required:

    python scripts/fetch_samples.py

It writes a handful of small sample images into
``data_store/test_images/samples/`` and creates a "Sample — cells & objects"
project pointed at them, using the **Cellpose-SAM** engine (zero-shot — it
needs no trained model or downloaded backbone, so you can open the project and
hit *Run* in the Segment tab right away).

The images are generated from scikit-image's *bundled* sample data (no network
download), and deliberately mix a microscopy image (human mitosis) with a
plainly non-cell one (coins) — the quickest way to show the product segments
*any* objects, not only cells. Re-running is safe (it overwrites the samples
and skips creating a duplicate project).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as ``python scripts/fetch_samples.py``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _write_samples(out_dir: Path) -> list[Path]:
    import numpy as np
    from skimage import data
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save(name: str, arr: np.ndarray) -> None:
        a = np.asarray(arr)
        if a.dtype != np.uint8:
            a = a.astype(np.float64)
            lo, hi = float(a.min()), float(a.max())
            a = np.zeros_like(a, dtype=np.uint8) if hi <= lo else \
                ((a - lo) / (hi - lo) * 255).astype(np.uint8)
        path = out_dir / name
        cv2.imwrite(str(path), a)
        written.append(path)

    # Microscopy: fluorescence-style nuclei (bundled, no download).
    _save("cells_human_mitosis.png", data.human_mitosis())
    # Non-cell objects — the "it segments anything, not just cells" proof.
    _save("objects_coins.png", data.coins())
    # A second, denser synthetic field so Run has more to find (no download).
    blobs = data.binary_blobs(length=512, blob_size_fraction=0.06,
                              volume_fraction=0.25, rng=0)
    _save("synthetic_blobs.png", blobs.astype(np.uint8) * 255)
    return written


def main() -> None:
    from project_root import STORAGE_DIR
    from studio.project import ProjectStore, ProjectSettings

    samples_dir = Path(STORAGE_DIR) / "test_images" / "samples"
    images = _write_samples(samples_dir)
    print(f"✓ wrote {len(images)} sample images to {samples_dir}")
    for p in images:
        print(f"    · {p.name}")

    store = ProjectStore(Path(STORAGE_DIR) / "projects")
    existing = {p.name for p in store.list()}
    name = "Sample — cells & objects"
    if name in existing:
        print(f"• project “{name}” already exists — leaving it as is.")
        return

    settings = ProjectSettings()
    settings.engine = "cellpose"  # zero-shot: no weights/training needed to Run
    project = store.create(
        name=name,
        description="Bundled samples (mitosis nuclei + coins + synthetic blobs) "
                    "to try segmentation on. Uses the zero-shot Cellpose-SAM "
                    "engine — open it and press Run.",
        tags=["sample"],
        image_paths=[str(p) for p in images],
        settings=settings,
    )
    print(f"✓ created project “{project.name}” ({len(images)} images, "
          f"engine=cellpose)")
    print("\nOpen Velum → Projects → “Sample — cells & objects” → Segment → Run.")


if __name__ == "__main__":
    main()
