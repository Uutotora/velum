"""Velum — the Datasets tab's logic (Qt-free, unit-tested).

Turns "collect your own dataset" from a single export button into a real
workflow:

  * **list** the datasets you've built (`list_datasets`),
  * **inspect** what would go into a new one, per image, before committing
    (`build_candidates` — which images are segmented, how many cells, whether
    they have ground truth),
  * **build** one from a curated selection of a project's images
    (`build_from_project`),
  * **import** a dataset back into a fresh project so you can keep proofreading
    it (`import_to_project` — this is "datasets → project"),
  * point training at one (`train_target`).

Takes plain objects (a `SegmentController`, a `ProjectStore`) and plain values,
never Qt widgets — the screen wires callbacks to it. Heavy image I/O (cv2) is
lazily imported inside the methods that read masks.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from studio import dataset_export
from studio.dataset import DatasetInfo, DatasetStore, default_datasets_root
from studio.project import Project, ProjectStore
from studio.segment_controller import SegmentController

# Image extensions Studio can import from disk (mirrors the project importer +
# the optional microscopy readers).
IMPORT_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy",
                     ".nd2", ".czi", ".lif"}


@dataclass
class BuildCandidate:
    """One project image as a possible dataset member — the row the build
    dialog shows with a checkbox."""
    image_path: str
    name: str
    segmented: bool
    cells: int
    has_gt: bool


@dataclass
class ImportCandidate:
    """One image found on disk during an import scan, with its matched mask
    (if any) — the row the import preview shows."""
    image_path: str
    name: str
    mask_path: Optional[str]
    cells: int


@dataclass
class ImportScan:
    """The result of scanning a folder / file selection for import."""
    candidates: list[ImportCandidate] = field(default_factory=list)
    is_velum_dataset: bool = False
    source_label: str = ""
    source_dir: Optional[str] = None

    @property
    def n_images(self) -> int:
        return len(self.candidates)

    @property
    def n_with_mask(self) -> int:
        return sum(1 for c in self.candidates if c.mask_path)


class DatasetController:
    def __init__(
        self,
        segment: Optional[SegmentController] = None,
        store: Optional[DatasetStore] = None,
        datasets_root: Optional[Path | str] = None,
    ):
        self._segment = segment or SegmentController()
        self.store = store or DatasetStore(datasets_root or default_datasets_root())

    # ── list / inspect ───────────────────────────────────────────────────────
    def list_datasets(self) -> list[DatasetInfo]:
        return self.store.list()

    def get(self, dataset_id: str) -> Optional[DatasetInfo]:
        return self.store.get(dataset_id)

    def build_candidates(self, project: Project) -> list[BuildCandidate]:
        """Every image in ``project`` as a build candidate, annotated with
        whether it's been segmented, its cell count, and whether it has a
        ground-truth mask — so the user curates from real status, not guesses."""
        out: list[BuildCandidate] = []
        for p in project.image_paths:
            mask = self._segment.load_result_mask(project, p)
            segmented = mask is not None
            cells = dataset_export.count_instances(mask) if segmented else 0
            has_gt = SegmentController.find_gt_for_image(p) is not None
            out.append(BuildCandidate(
                image_path=str(p), name=Path(p).name,
                segmented=segmented, cells=cells, has_gt=has_gt))
        return out

    def segmented_count(self, project: Project) -> int:
        """How many of the project's images are segmented (i.e. exportable)."""
        return sum(1 for c in self.build_candidates(project) if c.segmented)

    # ── build ────────────────────────────────────────────────────────────────
    def build_from_project(
        self, project: Project, image_paths: list[str], *,
        name: str, include_measurements: bool = True,
        val_fraction: float = 0.0,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> DatasetInfo:
        """Build a dataset from the segmented images among ``image_paths``.

        Raises ``ValueError`` if none of the selected images have a mask yet."""
        items: list[tuple[str, Any]] = []
        for p in image_paths:
            mask = self._segment.load_result_mask(project, p)
            if mask is not None:
                items.append((str(p), mask))
        if not items:
            raise ValueError("None of the selected images have been segmented yet.")

        dataset_id, out_dir = self.store.allocate(name)
        s = project.settings
        try:
            dataset_export.export_dataset(
                out_dir, items, name=name, project_id=project.id,
                engine=s.engine, model_name=s.model_name, vit_name=s.vit_name,
                pixel_size_um=(s.pixel_size_um or None),
                include_measurements=include_measurements,
                val_fraction=val_fraction, on_progress=on_progress,
            )
        except Exception:
            self.store.delete(dataset_id)   # don't leave a half-written folder
            raise
        info = self.store.get(dataset_id)
        assert info is not None             # just written
        return info

    # ── import from disk (bring your own dataset) ────────────────────────────
    @staticmethod
    def _gather_images(folder: Path) -> list[Path]:
        """Top-level image files in ``folder`` (or its ``images/`` subdir if it
        has one), excluding obvious mask files (``*_mask``/``*_masks``)."""
        base = folder / "images" if (folder / "images").is_dir() else folder
        out: list[Path] = []
        for p in sorted(base.iterdir()):
            if (p.is_file() and p.suffix.lower() in IMPORT_IMAGE_EXTS
                    and not p.stem.endswith(("_mask", "_masks"))):
                out.append(p)
        return out

    @staticmethod
    def _count_mask_cells(mask_path: str | Path) -> int:
        import cv2
        m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        return dataset_export.count_instances(m) if m is not None else 0

    def scan_import(self, paths: list[str]) -> ImportScan:
        """Scan a dropped/selected folder or file list and report what would be
        imported — before committing — the "we parsed your data" preview. A
        folder holding a Velum ``dataset.json`` is recognised as a whole
        dataset; otherwise each image is paired with a mask via the same
        ``<stem>_mask.*`` / ``masks/<stem>.*`` convention training already uses."""
        from studio.train_controller import find_mask_for_image
        ps = [Path(p) for p in paths]

        if len(ps) == 1 and ps[0].is_dir():
            folder = ps[0]
            velum = DatasetInfo.from_dir(folder)
            if velum is not None:
                cands = []
                for rec in velum.images:
                    mask = folder / str(rec.get("mask", ""))
                    cands.append(ImportCandidate(
                        image_path=str(folder / str(rec.get("image", ""))),
                        name=Path(str(rec.get("image", "image"))).name,
                        mask_path=str(mask) if mask.is_file() else None,
                        cells=int(rec.get("cells", 0))))
                return ImportScan(cands, True, f"Velum dataset · {folder.name}",
                                  str(folder))
            image_files = self._gather_images(folder)
            source_label = folder.name
        else:
            image_files = [p for p in ps
                           if p.is_file() and p.suffix.lower() in IMPORT_IMAGE_EXTS]
            source_label = (f"{len(image_files)} file"
                            f"{'' if len(image_files) == 1 else 's'}")

        cands = []
        for img in image_files:
            mp = find_mask_for_image(img)
            cands.append(ImportCandidate(
                image_path=str(img), name=img.name,
                mask_path=str(mp) if mp else None,
                cells=self._count_mask_cells(mp) if mp else 0))
        return ImportScan(cands, False, source_label)

    def import_as_dataset(
        self, scan: ImportScan, *, name: str,
        include_measurements: bool = True, val_fraction: float = 0.0,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> DatasetInfo:
        """Create a dataset from an ``ImportScan``. A recognised Velum dataset
        folder is copied in verbatim (lossless — keeps its manifest/provenance);
        a generic folder/file selection builds a dataset from every image that
        has a matched mask. Raises ``ValueError`` if nothing pairs up."""
        if scan.is_velum_dataset and scan.source_dir:
            dataset_id, out_dir = self.store.allocate(name)
            try:
                shutil.copytree(scan.source_dir, out_dir, dirs_exist_ok=True)
                mf = out_dir / "dataset.json"
                data = json.loads(mf.read_text(encoding="utf-8"))
                data["name"] = name
                mf.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                              encoding="utf-8")
            except Exception:
                self.store.delete(dataset_id)
                raise
            info = self.store.get(dataset_id)
            assert info is not None
            return info

        import cv2
        import numpy as np
        items: list[tuple[str, Any]] = []
        for c in scan.candidates:
            if not c.mask_path:
                continue
            m = cv2.imread(str(c.mask_path), cv2.IMREAD_UNCHANGED)
            if m is not None:
                items.append((c.image_path, np.ascontiguousarray(m).astype(np.int32)))
        if not items:
            raise ValueError("No image + mask pairs found to import.")

        dataset_id, out_dir = self.store.allocate(name)
        try:
            dataset_export.export_dataset(
                out_dir, items, name=name, project_id="", engine="imported",
                include_measurements=include_measurements,
                val_fraction=val_fraction, on_progress=on_progress)
        except Exception:
            self.store.delete(dataset_id)
            raise
        info = self.store.get(dataset_id)
        assert info is not None
        return info

    # ── datasets → project (import for further proofreading) ─────────────────
    def import_to_project(
        self, dataset: DatasetInfo, project_store: ProjectStore, *,
        name: Optional[str] = None,
    ) -> Project:
        """Create a fresh project from ``dataset``: copy its images in and seed
        each one's segmentation so opening the project shows the masks straight
        away, ready to keep proofreading. This is the "datasets → project" loop.
        """
        import cv2
        import numpy as np

        from studio.project import ProjectSettings
        settings = ProjectSettings(
            engine=dataset.engine or "cellseg1",
            pixel_size_um=float(dataset.pixel_size_um or 0.0),
        )
        project = project_store.create(name or dataset.name, settings=settings)

        stored_paths: list[str] = []
        for rec in dataset.images:
            src_img = dataset.path / rec.get("image", "")
            if not src_img.is_file():
                continue
            stored = project_store.import_image(project.id, src_img)
            stored_paths.append(str(stored))
            mask_file = dataset.path / rec.get("mask", "")
            if mask_file.is_file():
                m = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
                if m is not None:
                    self._segment.save_result_mask(
                        project, stored, np.ascontiguousarray(m).astype(np.int32))

        project.image_paths = stored_paths
        project.stats.n_images = len(stored_paths)
        project.stats.progress = 100 if stored_paths else 0
        project_store.save(project)
        return project

    # ── training ─────────────────────────────────────────────────────────────
    @staticmethod
    def train_target(dataset: DatasetInfo) -> tuple[Path, Path]:
        """The (images_dir, masks_dir) pair to point training at — the exact
        layout ``train_controller.find_mask_for_image`` discovers (masks are
        ``masks/<stem>.png`` beside ``images/<stem>``)."""
        return dataset.images_dir, dataset.masks_dir

    # ── delete ───────────────────────────────────────────────────────────────
    def delete(self, dataset_id: str) -> None:
        self.store.delete(dataset_id)
