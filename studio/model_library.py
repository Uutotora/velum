"""Velum — the Model Library catalog + on-disk logic (Qt-free, unit-tested).

The "find any model, download it, or bring your own" surface, one layer below
the Qt screen (``studio/model_library_screen.py``) and its controller
(``studio/model_library_controller.py``). This module is pure logic — a
curated catalog of real, publicly-downloadable segmentation models plus the
file-level operations to install, import and locate them — so it unit-tests
without Qt, torch, or a network (downloads are injected/monkeypatched in
tests, exactly like the optional-reader pattern elsewhere in this app).

Storage conventions are the ones the engines already read from, so a model
the Library installs is immediately usable in the Segment tab with no extra
wiring (see ``velum_core.predict_controller.resolve_sam`` /
``velum_core.engines_sam2.resolve_sam2`` / ``studio.train_controller``):

  * ``sam-backbone``  → ``<storage>/sam_backbone/<canonical .pth>``   (CellSeg1)
  * ``sam2``          → ``<storage>/sam2_checkpoints/<name .pt>``      (SAM 2)
  * ``cellseg1``      → ``<storage>/loras/<name>.pth`` (+ JSON sidecar) (LoRA)
  * ``cellpose``      → fetched by Cellpose itself on first use (built-in), or
                        a user ``.pth`` copied into repo ``checkpoints/``
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ── families ────────────────────────────────────────────────────────────────
# A model's "family" decides which engine consumes it and where it lands on
# disk. Kept as plain strings (not an enum) to match the rest of this package's
# style and stay trivially JSON/round-trippable.
FAMILY_SAM_BACKBONE = "sam-backbone"
FAMILY_SAM2 = "sam2"
FAMILY_LORA = "cellseg1"
FAMILY_CELLPOSE = "cellpose"

FAMILY_LABELS = {
    FAMILY_SAM_BACKBONE: "SAM backbone",
    FAMILY_SAM2: "SAM 2",
    FAMILY_LORA: "CellSeg1 · LoRA",
    FAMILY_CELLPOSE: "Cellpose",
}

# Which Segment engine key actually runs each family (so the screen can show a
# real EngineChip and the "Use in Segment" action knows what to switch to).
FAMILY_ENGINE = {
    FAMILY_SAM_BACKBONE: "cellseg1",
    FAMILY_SAM2: "sam2",
    FAMILY_LORA: "cellseg1",
    FAMILY_CELLPOSE: "cellpose",
}

# dest-dir kind → subdir under the storage root (checkpoints is repo-relative,
# handled in LibraryRoots.dir_for).
_DEST_SUBDIR = {
    FAMILY_SAM_BACKBONE: "sam_backbone",
    FAMILY_SAM2: "sam2_checkpoints",
    FAMILY_LORA: "loras",
}


@dataclass(frozen=True)
class CatalogModel:
    """One discoverable model in the built-in catalog. ``url``/``filename``
    empty means it isn't a file we download (Cellpose fetches its own weights
    on first use); those entries are marked ``builtin=True``."""

    id: str
    name: str
    family: str
    domain: str
    description: str
    size_mb: int
    license: str
    homepage: str
    url: str = ""
    filename: str = ""
    builtin: bool = False
    # Family-specific extras: sam-backbone → {"vit_name": "vit_h"}; sam2 →
    # {"model_type": "large"}; cellpose → {"cp_model": "cpsam"}.
    extra: dict = field(default_factory=dict)

    @property
    def family_label(self) -> str:
        return FAMILY_LABELS.get(self.family, self.family)

    @property
    def engine(self) -> str:
        return FAMILY_ENGINE.get(self.family, self.family)

    @property
    def size_label(self) -> str:
        if self.builtin or self.size_mb <= 0:
            return "—"
        if self.size_mb >= 1024:
            return f"{self.size_mb / 1024:.1f} GB"
        return f"{self.size_mb} MB"


# ── the curated catalog ─────────────────────────────────────────────────────
# Real, publicly downloadable weights. URLs/sizes are the official ones as of
# 2026-07; a wrong size never breaks a download (it only makes the progress
# bar's total an estimate — the stream copies whatever the server sends), and
# a moved URL degrades to a friendly DownloadError, never a crash.
CATALOG: list[CatalogModel] = [
    # SAM backbones (Meta) — the CellSeg1 engine's frozen encoder. vit_b is the
    # light default worth starting with; vit_h is the flagship the paper used.
    CatalogModel(
        id="sam_vit_b", name="SAM ViT-B", family=FAMILY_SAM_BACKBONE,
        domain="Generalist backbone", size_mb=375, license="Apache-2.0",
        description="Lightest SAM encoder — fastest one-shot LoRA fine-tuning, "
                    "a good first backbone to try.",
        homepage="https://github.com/facebookresearch/segment-anything",
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        filename="sam_vit_b_01ec64.pth", extra={"vit_name": "vit_b"}),
    CatalogModel(
        id="sam_vit_l", name="SAM ViT-L", family=FAMILY_SAM_BACKBONE,
        domain="Generalist backbone", size_mb=1250, license="Apache-2.0",
        description="Mid-size SAM encoder — more capacity than ViT-B for "
                    "unusual morphology, still trains on a laptop GPU.",
        homepage="https://github.com/facebookresearch/segment-anything",
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        filename="sam_vit_l_0b3195.pth", extra={"vit_name": "vit_l"}),
    CatalogModel(
        id="sam_vit_h", name="SAM ViT-H", family=FAMILY_SAM_BACKBONE,
        domain="Generalist backbone", size_mb=2560, license="Apache-2.0",
        description="Flagship SAM encoder — highest accuracy, the backbone the "
                    "CellSeg1 paper fine-tuned; needs a real GPU.",
        homepage="https://github.com/facebookresearch/segment-anything",
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        filename="sam_vit_h_4b8939.pth", extra={"vit_name": "vit_h"}),

    # SAM 2 (Meta) — the z-stack / time-lapse engine. tiny is a sane default to
    # download first; large is the accuracy pick.
    CatalogModel(
        id="sam2_tiny", name="SAM 2.1 Hiera-Tiny", family=FAMILY_SAM2,
        domain="3D · z-stack · video", size_mb=149, license="Apache-2.0",
        description="Smallest SAM 2 — zero-shot volumetric/time-lapse "
                    "segmentation with the lightest download.",
        homepage="https://github.com/facebookresearch/sam2",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        filename="sam2.1_hiera_tiny.pt", extra={"model_type": "tiny"}),
    CatalogModel(
        id="sam2_small", name="SAM 2.1 Hiera-Small", family=FAMILY_SAM2,
        domain="3D · z-stack · video", size_mb=176, license="Apache-2.0",
        description="Small SAM 2 — a step up in accuracy over Tiny at a similar "
                    "size.",
        homepage="https://github.com/facebookresearch/sam2",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        filename="sam2.1_hiera_small.pt", extra={"model_type": "small"}),
    CatalogModel(
        id="sam2_base_plus", name="SAM 2.1 Hiera-Base+", family=FAMILY_SAM2,
        domain="3D · z-stack · video", size_mb=309, license="Apache-2.0",
        description="Mid-size SAM 2 — balanced accuracy/speed for volumes.",
        homepage="https://github.com/facebookresearch/sam2",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        filename="sam2.1_hiera_base_plus.pt", extra={"model_type": "base_plus"}),
    CatalogModel(
        id="sam2_large", name="SAM 2.1 Hiera-Large", family=FAMILY_SAM2,
        domain="3D · z-stack · video", size_mb=856, license="Apache-2.0",
        description="Flagship SAM 2 — best zero-shot volumetric/time-lapse "
                    "consistency; needs a real GPU.",
        homepage="https://github.com/facebookresearch/sam2",
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        filename="sam2.1_hiera_large.pt", extra={"model_type": "large"}),

    # Cellpose (built-in) — Cellpose downloads these itself on first use, so the
    # Library lists them for discovery but "installed" just means Cellpose is
    # importable. No file we manage.
    CatalogModel(
        id="cp_cpsam", name="Cellpose-SAM (cpsam)", family=FAMILY_CELLPOSE,
        domain="Generalist cells", size_mb=0, license="BSD-3-Clause",
        description="Zero-shot generalist — no training needed. Cellpose "
                    "fetches the weights automatically the first time you run it.",
        homepage="https://github.com/MouseLand/cellpose", builtin=True,
        extra={"cp_model": "cpsam"}),
    CatalogModel(
        id="cp_cyto3", name="Cellpose cyto3", family=FAMILY_CELLPOSE,
        domain="Cytoplasm", size_mb=0, license="BSD-3-Clause",
        description="Cellpose's cytoplasm generalist. Auto-fetched on first use.",
        homepage="https://github.com/MouseLand/cellpose", builtin=True,
        extra={"cp_model": "cyto3"}),
    CatalogModel(
        id="cp_nuclei", name="Cellpose nuclei", family=FAMILY_CELLPOSE,
        domain="Nuclei", size_mb=0, license="BSD-3-Clause",
        description="Cellpose's nucleus model. Auto-fetched on first use.",
        homepage="https://github.com/MouseLand/cellpose", builtin=True,
        extra={"cp_model": "nuclei"}),
]


def catalog() -> list[CatalogModel]:
    return list(CATALOG)


def get_catalog_model(model_id: str) -> Optional[CatalogModel]:
    for m in CATALOG:
        if m.id == model_id:
            return m
    return None


# ── on-disk roots ───────────────────────────────────────────────────────────
@dataclass
class LibraryRoots:
    """Where each family lives on disk. ``storage_dir`` is the shared
    ``data_store`` (per-family subdirs), ``checkpoints_dir`` is the repo's
    bundled ``checkpoints/`` (Cellpose custom ``.pth`` files)."""

    storage_dir: Path
    checkpoints_dir: Path

    @classmethod
    def default(cls) -> "LibraryRoots":
        from project_root import STORAGE_DIR, PROJECT_ROOT
        return cls(storage_dir=Path(STORAGE_DIR),
                   checkpoints_dir=Path(PROJECT_ROOT) / "checkpoints")

    def dir_for(self, family: str) -> Path:
        if family == FAMILY_CELLPOSE:
            return self.checkpoints_dir
        sub = _DEST_SUBDIR.get(family)
        if sub is None:
            raise ValueError(f"No install directory for family {family!r}")
        return self.storage_dir / sub


def dest_path(model: CatalogModel, roots: LibraryRoots) -> Optional[Path]:
    """Where ``model`` is (or would be) installed, or ``None`` for a built-in
    (Cellpose-managed) model that has no file we own."""
    if model.builtin or not model.filename:
        return None
    return roots.dir_for(model.family) / model.filename


def is_installed(model: CatalogModel, roots: LibraryRoots,
                 cellpose_available: bool = False) -> bool:
    """A downloadable model is installed iff its file exists; a built-in
    Cellpose entry is "installed" iff Cellpose itself is importable."""
    if model.builtin:
        return bool(cellpose_available)
    p = dest_path(model, roots)
    return bool(p and p.exists() and p.stat().st_size > 0)


def installed_size_mb(model: CatalogModel, roots: LibraryRoots) -> Optional[int]:
    p = dest_path(model, roots)
    if p and p.exists():
        return round(p.stat().st_size / (1024 * 1024))
    return None


# ── download ────────────────────────────────────────────────────────────────
class DownloadError(RuntimeError):
    """A friendly, user-facing download failure (offline, 404, disk full)."""


# Injection seam: tests replace this with a fake that writes bytes locally
# instead of hitting the network (mirrors how the optional-reader tests fake
# ``nd2``/``czifile`` modules). Signature: (url, dest, on_progress) -> None.
def _urlretrieve(url: str, dest: Path,
                 on_progress: Optional[Callable[[int, int], None]] = None) -> None:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Velum-ModelLibrary"})
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted catalog URLs)
            total = int(resp.headers.get("Content-Length", 0) or 0)
            done = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if on_progress:
                        on_progress(done, total)
    except urllib.error.HTTPError as e:
        raise DownloadError(f"Server returned {e.code} for {url}") from e
    except urllib.error.URLError as e:
        raise DownloadError(
            f"Couldn't reach {url} — check your internet connection.") from e
    except OSError as e:
        raise DownloadError(f"Couldn't write the file: {e}") from e


def download(model: CatalogModel, roots: LibraryRoots, *,
             on_progress: Optional[Callable[[int, int], None]] = None,
             should_cancel: Optional[Callable[[], bool]] = None) -> Path:
    """Download ``model`` into its canonical location and return the path.

    Streams to a ``.part`` file and renames on success, so an interrupted or
    cancelled download never leaves a truncated file masquerading as installed.
    Raises ``DownloadError`` (friendly) or ``ValueError`` (not downloadable).
    """
    if model.builtin or not model.url:
        raise ValueError(f"{model.name} is not a downloadable file "
                         "(it's fetched by its engine on first use).")
    dest = dest_path(model, roots)
    assert dest is not None
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    def _progress(done: int, total: int) -> None:
        if should_cancel and should_cancel():
            raise DownloadError("Download cancelled.")
        if on_progress:
            on_progress(done, total)

    try:
        _urlretrieve(model.url, part, _progress)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return dest


# ── import your own ─────────────────────────────────────────────────────────
def import_checkpoint(src: str | Path, family: str, roots: LibraryRoots, *,
                      name: Optional[str] = None) -> Path:
    """Copy a user-supplied checkpoint into the right directory for ``family``
    so it shows up alongside the catalog models and is usable in Segment.

    For a LoRA (``cellseg1``) a matching JSON sidecar is copied if present, or
    a minimal one written, so ``train_controller.list_trained_models`` picks it
    up. For a SAM backbone the file is renamed to the canonical
    ``sam_vit_*`` name its ``vit_name`` implies (guessed from the filename),
    since the engine looks the backbone up by that fixed name.
    """
    src = Path(src)
    if not src.exists():
        raise ValueError(f"No such file: {src}")
    dest_dir = roots.dir_for(family)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if family == FAMILY_SAM_BACKBONE:
        from studio.train_controller import BACKBONE_FILES, guess_vit_name
        vit = guess_vit_name(src)
        dest = dest_dir / BACKBONE_FILES[vit]
        shutil.copy2(src, dest)
        return dest

    if family == FAMILY_LORA:
        stem = name or src.stem
        dest = dest_dir / f"{stem}{src.suffix}"
        shutil.copy2(src, dest)
        sidecar_src = src.with_suffix(".json")
        sidecar_dest = dest.with_suffix(".json")
        if sidecar_src.exists():
            shutil.copy2(sidecar_src, sidecar_dest)
        elif not sidecar_dest.exists():
            # Minimal sidecar so the model lists with a name/date instead of
            # being skipped by list_trained_models (which requires a sidecar).
            from datetime import datetime
            sidecar_dest.write_text(json.dumps({
                "result_pth_path": str(dest),
                "vit_name": "vit_h",
                "image_encoder_lora_rank": 8,
                "train_id": [],
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "imported": True,
            }, indent=2), encoding="utf-8")
        return dest

    # sam2 / cellpose custom: straight copy, keep the filename.
    dest = dest_dir / (f"{name}{src.suffix}" if name else src.name)
    shutil.copy2(src, dest)
    return dest


def remove(model: CatalogModel, roots: LibraryRoots) -> bool:
    """Delete a downloaded model's file (and a LoRA's sidecar). Built-ins have
    no file to remove. Returns True if something was deleted."""
    p = dest_path(model, roots)
    if not p or not p.exists():
        return False
    p.unlink()
    if model.family == FAMILY_LORA:
        p.with_suffix(".json").unlink(missing_ok=True)
    return True
