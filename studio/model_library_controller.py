"""Velum — the Model Library tab controller (Qt-free, unit-tested).

Plain glue between ``studio/model_library.py`` (catalog + file ops) and the Qt
screen: it resolves storage roots once, reports each catalog model's install
status, lists the user's own trained/imported LoRAs, and runs downloads on a
background thread with progress + cancellation — the same plain-callbacks,
no-Qt shape as ``segment_controller`` / ``train_controller``.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from studio import model_library as ml
from studio.model_library import CatalogModel, LibraryRoots


@dataclass
class CatalogEntry:
    """A catalog model plus its live install status — one card on the screen."""

    model: CatalogModel
    installed: bool
    installed_size_mb: Optional[int]


@dataclass
class LocalModel:
    """A user-owned model (trained here or imported) — the 'Your models'
    section, distinct from the discoverable catalog above."""

    name: str
    family: str
    path: Path
    meta: str
    detail: str = ""


class ModelLibraryController:
    def __init__(self, storage_dir: Optional[Path | str] = None,
                 checkpoints_dir: Optional[Path | str] = None):
        if storage_dir is None or checkpoints_dir is None:
            defaults = LibraryRoots.default()
        self.roots = LibraryRoots(
            storage_dir=Path(storage_dir) if storage_dir is not None else defaults.storage_dir,
            checkpoints_dir=Path(checkpoints_dir) if checkpoints_dir is not None
            else defaults.checkpoints_dir)
        self._cancel = threading.Event()

    # ── availability ─────────────────────────────────────────────────────────
    @staticmethod
    def cellpose_available() -> bool:
        try:
            from velum_core.engines import cellpose_available
            return bool(cellpose_available())
        except Exception:
            return False

    # ── catalog ──────────────────────────────────────────────────────────────
    def catalog_entries(self, family: Optional[str] = None) -> list[CatalogEntry]:
        cp = self.cellpose_available()
        out: list[CatalogEntry] = []
        for m in ml.catalog():
            if family and m.family != family:
                continue
            out.append(CatalogEntry(
                model=m,
                installed=ml.is_installed(m, self.roots, cellpose_available=cp),
                installed_size_mb=ml.installed_size_mb(m, self.roots)))
        return out

    def families(self) -> list[tuple[str, str]]:
        """``[(family_key, label)]`` in catalog order, for the filter chips."""
        seen: list[str] = []
        for m in ml.catalog():
            if m.family not in seen:
                seen.append(m.family)
        return [(f, ml.FAMILY_LABELS.get(f, f)) for f in seen]

    def get(self, model_id: str) -> Optional[CatalogModel]:
        return ml.get_catalog_model(model_id)

    def dest_path(self, model_id: str) -> Optional[Path]:
        m = ml.get_catalog_model(model_id)
        return ml.dest_path(m, self.roots) if m else None

    # ── your own models ──────────────────────────────────────────────────────
    def local_models(self) -> list[LocalModel]:
        """Trained + imported LoRAs (via ``train_controller.list_trained_models``)
        and any custom Cellpose ``.pth`` bundled in ``checkpoints/``."""
        out: list[LocalModel] = []
        from studio.train_controller import list_trained_models
        for tm in list_trained_models(self.roots.storage_dir / "loras"):
            out.append(LocalModel(
                name=tm.name, family=ml.FAMILY_LORA, path=Path(tm.checkpoint),
                meta=tm.meta, detail=(f"F1 {tm.f1}" if tm.f1 else "") or ""))
        ckpt_dir = self.roots.checkpoints_dir
        if ckpt_dir.is_dir():
            for pth in sorted(ckpt_dir.glob("*.pth")):
                out.append(LocalModel(
                    name=pth.stem, family=ml.FAMILY_CELLPOSE, path=pth,
                    meta="Cellpose checkpoint", detail=""))
        return out

    # ── download (background) ────────────────────────────────────────────────
    def download_async(self, model_id: str, *,
                       on_progress: Optional[Callable[[int, int], None]] = None,
                       on_done: Optional[Callable[[Path], None]] = None,
                       on_error: Optional[Callable[[str], None]] = None,
                       ) -> threading.Thread:
        model = ml.get_catalog_model(model_id)
        if model is None:
            raise ValueError(f"Unknown model id: {model_id}")
        self._cancel.clear()

        def _run() -> None:
            try:
                path = ml.download(
                    model, self.roots, on_progress=on_progress,
                    should_cancel=self._cancel.is_set)
            except ml.DownloadError as e:
                if on_error:
                    on_error(str(e))
                return
            except Exception as e:  # pragma: no cover - defensive
                if on_error:
                    on_error(f"Download failed: {e}")
                return
            if on_done:
                on_done(path)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def cancel_download(self) -> None:
        self._cancel.set()

    # ── import / remove ──────────────────────────────────────────────────────
    def import_file(self, src: str | Path, family: str,
                    name: Optional[str] = None) -> Path:
        return ml.import_checkpoint(src, family, self.roots, name=name)

    def remove(self, model_id: str) -> bool:
        model = ml.get_catalog_model(model_id)
        if model is None:
            return False
        return ml.remove(model, self.roots)
