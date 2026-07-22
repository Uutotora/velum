"""Velum — datasets as first-class, on-disk artifacts.

A **Project** is a live working cohort (images you're actively segmenting and
proofreading). A **Dataset** is what you *curate out* of that work: a named,
frozen, self-describing folder of images + instance masks you can share, keep,
re-import, or train on. This module is the persistence layer for datasets — the
registry the "Datasets" tab lists and the store the dataset controller builds
into. The actual folder is written by ``studio.dataset_export`` (images/ +
masks/ + dataset.json); here we just discover, describe, and manage those
folders.

Qt-free and torch-free (numpy isn't even needed) so it imports and unit-tests
in CI's light group.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from studio.dataset_export import DATASET_FORMAT


def default_datasets_root() -> Path:
    """The conventional datasets root: ``<STORAGE_DIR>/datasets``.

    Shares the app's storage dir with projects/models/segment-runs; imported
    lazily so this module has no hard import-time dependency on ``project_root``
    (and stays trivially testable with an explicit temp root)."""
    from studio.train_controller import default_storage_dir
    return default_storage_dir() / "datasets"


def _slugify(name: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "dataset"


@dataclass
class DatasetInfo:
    """A lightweight, read-only view over one built dataset folder — its id
    (folder name), path, and parsed ``dataset.json`` manifest, plus the derived
    numbers the UI shows without re-parsing the manifest each time."""
    id: str
    path: Path
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, path: str | Path) -> Optional["DatasetInfo"]:
        """Read ``<path>/dataset.json`` into a ``DatasetInfo``; ``None`` if the
        folder has no readable, correctly-formatted manifest (so a stray dir in
        the datasets root is skipped, not crashed on)."""
        path = Path(path)
        manifest_file = path / "dataset.json"
        if not manifest_file.is_file():
            return None
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(manifest, dict) or manifest.get("format") != DATASET_FORMAT:
            return None
        return cls(id=path.name, path=path, manifest=manifest)

    # ── Derived, display-friendly accessors ─────────────────────────────────
    @property
    def name(self) -> str:
        return str(self.manifest.get("name") or self.id)

    @property
    def created(self) -> str:
        return str(self.manifest.get("created") or "")

    @property
    def _counts(self) -> dict[str, int]:
        c = self.manifest.get("counts")
        return c if isinstance(c, dict) else {}

    @property
    def n_images(self) -> int:
        return int(self._counts.get("n_images", 0))

    @property
    def n_cells(self) -> int:
        return int(self._counts.get("n_cells", 0))

    @property
    def n_train(self) -> int:
        return int(self._counts.get("n_train", 0))

    @property
    def n_val(self) -> int:
        return int(self._counts.get("n_val", 0))

    @property
    def engine(self) -> str:
        src = self.manifest.get("source")
        return str(src.get("engine", "")) if isinstance(src, dict) else ""

    @property
    def source_project_id(self) -> str:
        src = self.manifest.get("source")
        return str(src.get("project_id", "")) if isinstance(src, dict) else ""

    @property
    def pixel_size_um(self) -> Optional[float]:
        src = self.manifest.get("source")
        return (src or {}).get("pixel_size_um") if isinstance(src, dict) else None

    @property
    def images(self) -> list[dict[str, Any]]:
        imgs = self.manifest.get("images")
        return imgs if isinstance(imgs, list) else []

    @property
    def images_dir(self) -> Path:
        return self.path / "images"

    @property
    def masks_dir(self) -> Path:
        return self.path / "masks"


class DatasetStore:
    """Filesystem-backed collection of datasets under one root directory —
    one folder per dataset (``<root>/<id>/dataset.json`` + images/ + masks/).

    The store discovers and manages folders; the actual write is delegated to
    ``dataset_export.export_dataset`` (see ``allocate``), so there is exactly
    one place that knows the on-disk layout.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, dataset_id: str) -> Path:
        return self.root / dataset_id

    def unique_id(self, name: str) -> str:
        """A slug for ``name`` that doesn't collide with an existing dataset."""
        base = _slugify(name)
        candidate, n = base, 2
        while self._dir(candidate).exists():
            candidate = f"{base}-{n}"
            n += 1
        return candidate

    def allocate(self, name: str) -> tuple[str, Path]:
        """Reserve a unique id + (fresh) folder for a new dataset named
        ``name`` and return ``(id, dir)``. The caller (dataset controller)
        writes the dataset into ``dir`` via ``dataset_export.export_dataset``.
        """
        dataset_id = self.unique_id(name)
        d = self._dir(dataset_id)
        d.mkdir(parents=True, exist_ok=True)
        return dataset_id, d

    def list(self) -> list[DatasetInfo]:
        """Every valid dataset under the root, newest first (by ``created``)."""
        out: list[DatasetInfo] = []
        if not self.root.is_dir():
            return out
        for child in self.root.iterdir():
            if child.is_dir():
                info = DatasetInfo.from_dir(child)
                if info is not None:
                    out.append(info)
        out.sort(key=lambda d: d.created, reverse=True)
        return out

    def get(self, dataset_id: str) -> Optional[DatasetInfo]:
        return DatasetInfo.from_dir(self._dir(dataset_id))

    def exists(self, dataset_id: str) -> bool:
        return (self._dir(dataset_id) / "dataset.json").is_file()

    def delete(self, dataset_id: str) -> None:
        """Permanently remove a dataset folder. No-op if already gone."""
        d = self._dir(dataset_id)
        if d.exists():
            shutil.rmtree(d)
