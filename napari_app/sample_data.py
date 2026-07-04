"""
Sample microscopy images for trying the app without hunting for data.

Prefers real, bundled scikit-image microscopy samples (no network needed):
``human_mitosis`` (fluorescence, many dividing nuclei) and ``cell`` (a single
cell). Falls back to a synthetic fluorescence-like field of blobs so the button
always produces something usable, even offline with a minimal install.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[..., :3]
    elif img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float64)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = img.max() or 1.0
        lo = img.min()
    img = np.clip((img - lo) / (hi - lo), 0, 1) * 255.0
    img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def _synthetic_field(h: int = 512, w: int = 512, n: int = 60, seed: int = 0) -> np.ndarray:
    """A synthetic fluorescence-like field of soft elliptical blobs."""
    rng = np.random.default_rng(seed)
    field = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n):
        cy, cx = rng.uniform(20, h - 20), rng.uniform(20, w - 20)
        ry, rx = rng.uniform(8, 18), rng.uniform(8, 18)
        amp = rng.uniform(0.5, 1.0)
        field += amp * np.exp(-(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2))
    field += rng.normal(0, 0.02, field.shape)
    field = np.clip(field, 0, None)
    return _to_uint8_rgb(field)


def _synthetic_labeled(h: int = 512, w: int = 512, n: int = 48, seed: int = 1):
    """A synthetic field of well-separated cells with an exact label mask.

    Returns (image_rgb uint8, label_mask int32). Because we place every cell
    ourselves, the mask is a perfect ground truth — useful for validating the
    Evaluate-vs-GT metrics.
    """
    rng = np.random.default_rng(seed)
    field = np.zeros((h, w), dtype=np.float64)
    labels = np.zeros((h, w), dtype=np.int32)
    yy, xx = np.mgrid[0:h, 0:w]
    placed: list[tuple[float, float, float]] = []
    lab = 0
    attempts = 0
    while lab < n and attempts < n * 40:
        attempts += 1
        r = rng.uniform(9, 16)
        cy, cx = rng.uniform(r + 4, h - r - 4), rng.uniform(r + 4, w - r - 4)
        if any((cy - py) ** 2 + (cx - px) ** 2 < (r + pr + 5) ** 2 for py, px, pr in placed):
            continue
        placed.append((cy, cx, r))
        lab += 1
        d2 = ((yy - cy) / r) ** 2 + ((xx - cx) / r) ** 2
        field += rng.uniform(0.6, 1.0) * np.exp(-d2)
        labels[d2 <= 1.0] = lab
    field += rng.normal(0, 0.015, field.shape)
    return _to_uint8_rgb(np.clip(field, 0, None)), labels


# Broad Bioimage Benchmark Collection 039 — nuclei of U2OS cells (fluorescence),
# 200 fields with expert ground truth. Directly downloadable, no login.
BBBC039_IMAGES = "https://data.broadinstitute.org/bbbc/BBBC039/images.zip"
BBBC039_MASKS = "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip"


def _bbbc039_instances(mask_rgba: np.ndarray) -> np.ndarray:
    """Turn a BBBC039 3-class mask into an instance label image.

    The class channel encodes 0=background, 1=interior, 2=boundary. Connected
    components of the interior (boundaries keep touching nuclei apart) give one
    label per nucleus.
    """
    from scipy.ndimage import label as cclabel
    ch = mask_rgba[..., 2] if mask_rgba.ndim == 3 else mask_rgba
    inst, _n = cclabel(ch == 1)
    return inst.astype(np.uint16)


def download_bbbc039(dest_dir, limit: int = 20, progress=None) -> list[str]:
    """Download a subset of BBBC039 with instance ground truth.

    Writes ``<stem>.png`` images and matching ``<stem>_gt.png`` masks into
    ``dest_dir/BBBC039`` so ground truth auto-fills and benchmarking works.
    Returns the image paths written.
    """
    import io
    import urllib.request
    import zipfile
    import cv2

    dest = Path(dest_dir) / "BBBC039"
    dest.mkdir(parents=True, exist_ok=True)

    def _fetch(url, what):
        if progress:
            progress(f"Downloading {what}…")
        with urllib.request.urlopen(url, timeout=300) as r:
            return r.read()

    imgs = zipfile.ZipFile(io.BytesIO(_fetch(BBBC039_IMAGES, "images")))
    masks = zipfile.ZipFile(io.BytesIO(_fetch(BBBC039_MASKS, "masks")))

    def _clean(names, ext):
        return sorted(n for n in names
                      if n.lower().endswith(ext) and not n.startswith("__MACOSX"))

    img_members = _clean(imgs.namelist(), (".tif", ".tiff"))
    mask_by_stem = {Path(n).stem: n for n in _clean(masks.namelist(), ".png")}

    saved: list[str] = []
    for member in img_members:
        if len(saved) >= limit:
            break
        stem = Path(member).stem
        if stem not in mask_by_stem:
            continue
        try:
            raw = np.frombuffer(imgs.read(member), np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if img is None:
                import tifffile
                img = tifffile.imread(io.BytesIO(imgs.read(member)))
            img8 = _to_uint8_rgb(img)
            mraw = np.frombuffer(masks.read(mask_by_stem[stem]), np.uint8)
            mrgba = cv2.imdecode(mraw, cv2.IMREAD_UNCHANGED)
            inst = _bbbc039_instances(mrgba)

            short = stem.split("_")[1] if "_" in stem else stem[:8]
            img_p = dest / f"bbbc039_{short}.png"
            gt_p = dest / f"bbbc039_{short}_gt.png"
            cv2.imwrite(str(img_p), cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(gt_p), inst)
            saved.append(str(img_p))
            if progress:
                progress(f"Extracted {len(saved)}/{limit}  ({int(inst.max())} nuclei)")
        except Exception:
            continue
    return saved


def fetch_samples(dest_dir) -> list[str]:
    """Write sample images into ``dest_dir``; return the paths written.

    Also writes a labelled phantom image alongside its exact ground-truth mask
    (``sample_phantom.png`` + ``sample_phantom_gt.png``) so the Ground-truth /
    Evaluate feature has something to auto-fill and score against.
    """
    import cv2

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # Labelled phantom with matching ground truth (deterministic).
    try:
        img, gt = _synthetic_labeled()
        img_p = dest / "sample_phantom.png"
        gt_p = dest / "sample_phantom_gt.png"
        cv2.imwrite(str(img_p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(gt_p), gt.astype(np.uint16))
        saved.append(str(img_p))
    except Exception:
        pass

    # Real microscopy images bundled with / fetched by scikit-image (via pooch).
    # Curated for cell/nucleus segmentation: fluorescence nuclei, a single cell,
    # an H&E tissue section, and a 2-D slice of a 3-D cell volume.
    candidates: list = []
    try:
        from skimage import data
        candidates = [
            ("nuclei_fluorescence", data.human_mitosis),   # many dividing nuclei
            ("single_cell",         data.cell),            # one cell, brightfield
            ("tissue_he",           data.immunohistochemistry),  # stained tissue
        ]

        def _cells3d_slice():
            vol = data.cells3d()          # (z, c, y, x)
            return vol[vol.shape[0] // 2, 1]  # mid-z, nuclei channel
        candidates.append(("cell_nuclei_slice", _cells3d_slice))
    except Exception:
        candidates = []

    for name, fn in candidates:
        try:
            img = _to_uint8_rgb(fn())
            path = dest / f"sample_{name}.png"
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            saved.append(str(path))
        except Exception:
            continue

    if not saved:
        img = _synthetic_field()
        path = dest / "sample_synthetic.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        saved.append(str(path))

    return saved
