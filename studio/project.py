"""Velum — the Project data model and on-disk store.

A **Project** is the unit of work the whole product is organised around (the
concept Label Studio gets right and the current app lacks entirely): a named
collection of images, the engine + settings used to segment them, plus
metadata (tags, favourite, timestamps) and cached result stats for the
library cards.

This module is the data layer and is deliberately dependency-free — only the
standard library. No Qt, no torch, no napari, and crucially no
``data.utils`` (which hard-imports ``nibabel``) — so the unit tests run under
the light CI ``test`` dependency-group without pulling the world in.

Persistence layout (under ``<store_root>/<project-id>/``)::

    projects/
      fluorescence-nuclei-dapi/
        project.json          # everything in this module, serialised
        (masks/, exports/, …  # future: per-project artefacts live here)

``ProjectSettings`` mirrors **every** knob the predict/train pipeline reads
(see ``velum_core/predict_controller.py``'s ``build_config`` /
``sam_config``). Keeping them here, per-project and versioned on disk, is what
lets a microscopist reopen a cohort weeks later and get byte-identical
segmentation — and it is the single source of truth for "don't lose any
existing setting" as the UI is rebuilt around it.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Schema version for the on-disk JSON. Bump when a migration is needed; the
# loader tolerates older/newer payloads (unknown keys ignored, missing keys
# defaulted), so this is for explicit migrations, not routine compatibility.
SCHEMA_VERSION = 1

# The three interchangeable engines (keys match velum_core.engine_registry).
ENGINES = ("cellseg1", "cellpose", "sam2")

# Display name and design-token "kind" (see theme.py / components.Chip) for
# each engine — the single source screens draw on so a project's engine reads
# the same way on a card, a chip, or a workspace breadcrumb.
ENGINE_LABELS = {"cellseg1": "CellSeg1 · LoRA", "cellpose": "Cellpose-SAM", "sam2": "SAM 2"}
ENGINE_KIND = {"cellseg1": "primary", "cellpose": "signal", "sam2": "primary"}

# Display colouring modes for the "Colour cells by" control. "instance" is the
# default categorical map; the rest drive a per-cell heatmap over a morphometry.
COLOR_BY = ("instance", "area", "diameter", "solidity", "intensity")


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (stable, sortable, tz-aware)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(name: str) -> str:
    """Turn a human project name into a filesystem-safe id slug.

    Lowercased, non-alphanumerics collapsed to single hyphens, trimmed. Empty
    or all-symbol names fall back to ``"project"`` so a slug is always valid.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "project"


def _coerce(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Filter ``data`` to the known fields of dataclass ``cls``.

    This is what makes loading forward- and backward-compatible: a JSON file
    written by a newer build (extra keys) or an older one (missing keys) still
    loads — unknown keys are dropped, absent keys take the dataclass default.
    """
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


@dataclass
class ProjectSettings:
    """Every segmentation knob, per project — the persisted config contract.

    Field names are kept aligned with the pipeline's config keys so mapping to
    ``PredictController.build_config`` / ``sam_config`` stays mechanical. Adding
    a new engine parameter means adding it here (and defaulting it) so old
    project files keep loading.
    """

    # ── Engine + model ──────────────────────────────────────────────────────
    engine: str = "cellseg1"            # cellseg1 | cellpose | sam2
    model_name: str = ""                # LoRA adapter name/path (cellseg1)
    vit_name: str = "vit_h"             # SAM backbone: vit_h | vit_l | vit_b
    lora_rank: int = 8

    # ── Quality preset + SAM AMG detection thresholds ───────────────────────
    quality_preset: str = "Balanced"    # Fast | Balanced | Accurate | Custom
    resize_size: int = 512
    points_per_side: int = 32
    pred_iou_thresh: float = 0.80
    stability_score_thresh: float = 0.60
    box_nms_thresh: float = 0.05
    min_mask_area: int = 20

    # ── Cellpose-SAM ────────────────────────────────────────────────────────
    cp_diameter: float = 0.0            # 0 = auto
    cp_flow_threshold: float = 0.40
    cp_cellprob_threshold: float = 0.0

    # ── SAM 2 (z-stack / time-lapse) ────────────────────────────────────────
    sam2_model: str = "large"           # large | base_plus | small | tiny
    sam2_tracking_mode: str = "independent"  # independent | propagate
    stitch_iou: float = 0.25            # adjacent-slice instance linking

    # ── Image / channels / large-image ──────────────────────────────────────
    pixel_size_um: float = 0.0          # 0 = measure in pixels
    channels: list[int] = field(default_factory=list)  # selected channel idxs
    channel_low: float = 1.0            # per-channel percentile stretch (low)
    channel_high: float = 99.0          # …(high)
    clahe: bool = False
    tiled: bool = False                 # native-resolution tiled inference
    tile_size: int = 1024
    tile_overlap: int = 0               # 0 = auto from cell diameter

    # ── Display ─────────────────────────────────────────────────────────────
    color_by: str = "instance"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectSettings":
        return cls(**_coerce(cls, data or {}))


@dataclass
class ProjectStats:
    """Cached result summary shown on library cards (recomputed after a run)."""

    n_images: int = 0
    n_cells: int = 0
    last_f1: Optional[float] = None     # None until benchmarked against GT
    progress: int = 0                   # 0–100, images segmented in the project

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectStats":
        return cls(**_coerce(cls, data or {}))


@dataclass
class Project:
    """A named cohort: images + engine + settings + metadata + cached stats."""

    id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    tags: list[str] = field(default_factory=list)
    favorite: bool = False
    image_paths: list[str] = field(default_factory=list)
    settings: ProjectSettings = field(default_factory=ProjectSettings)
    stats: ProjectStats = field(default_factory=ProjectStats)
    schema_version: int = SCHEMA_VERSION

    # ── Convenience ─────────────────────────────────────────────────────────
    @property
    def engine(self) -> str:
        """Engine key, surfaced from settings for card display / filtering."""
        return self.settings.engine

    def touch(self) -> None:
        """Mark as just-modified (drives 'recent' ordering)."""
        self.updated_at = _now_iso()

    # ── Serialisation ───────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)                       # recurses into settings/stats
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        data = dict(data or {})
        kw = _coerce(cls, data)
        kw["settings"] = ProjectSettings.from_dict(data.get("settings", {}))
        kw["stats"] = ProjectStats.from_dict(data.get("stats", {}))
        # id/name are required in practice; guard against a corrupt file.
        kw.setdefault("id", slugify(kw.get("name", "")))
        kw.setdefault("name", kw["id"])
        return cls(**kw)


class ProjectStore:
    """Filesystem-backed collection of projects under a single root directory.

    One folder per project (``<root>/<id>/project.json``), so a project can
    later own sibling artefacts (masks, exports, an embedding cache) without
    schema churn. All mutating methods write through immediately — there is no
    in-memory "dirty" state to lose.
    """

    PROJECT_FILE = "project.json"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Paths ───────────────────────────────────────────────────────────────
    def _dir(self, project_id: str) -> Path:
        return self.root / project_id

    def _file(self, project_id: str) -> Path:
        return self._dir(project_id) / self.PROJECT_FILE

    def _unique_id(self, name: str) -> str:
        """A slug for ``name`` that doesn't collide with an existing project."""
        base = slugify(name)
        candidate = base
        n = 2
        while self._dir(candidate).exists():
            candidate = f"{base}-{n}"
            n += 1
        return candidate

    # ── CRUD ────────────────────────────────────────────────────────────────
    def create(
        self,
        name: str,
        description: str = "",
        tags: Optional[list[str]] = None,
        settings: Optional[ProjectSettings] = None,
        image_paths: Optional[list[str]] = None,
    ) -> Project:
        """Create, persist and return a new project with a unique id."""
        project = Project(
            id=self._unique_id(name),
            name=name.strip() or "Untitled project",
            description=description,
            tags=list(tags or []),
            settings=settings or ProjectSettings(),
            image_paths=list(image_paths or []),
        )
        project.stats.n_images = len(project.image_paths)
        self.save(project)
        return project

    def save(self, project: Project, *, touch: bool = True) -> None:
        """Write a project to disk atomically (temp file + replace).

        ``touch=False`` keeps the project's own ``updated_at``/``created_at``
        instead of stamping the current time — for callers that need
        deterministic, explicit timestamps rather than "now" (``touch=True``,
        the default, is right for every ordinary edit).
        """
        if touch:
            project.touch()
        self._dir(project.id).mkdir(parents=True, exist_ok=True)
        target = self._file(project.id)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(project.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(target)  # atomic on POSIX; never leaves a half-written file

    def load(self, project_id: str) -> Project:
        """Load a single project by id (raises FileNotFoundError if absent)."""
        data = json.loads(self._file(project_id).read_text(encoding="utf-8"))
        return Project.from_dict(data)

    def exists(self, project_id: str) -> bool:
        return self._file(project_id).exists()

    def delete(self, project_id: str) -> None:
        """Permanently remove a project and all its artefacts. Irreversible --
        the Projects tab's Danger Zone gates this behind its own confirmation
        (see ``studio/project_dialogs.py``'s ``confirm_delete_project``).
        No-op if already gone.
        """
        import shutil
        d = self._dir(project_id)
        if d.exists():
            shutil.rmtree(d)

    # ── Imported image files ─────────────────────────────────────────────────
    def image_dir(self, project_id: str) -> Path:
        """Where a project's *copied-in* image files live (``<project>/images``).

        Created on demand. Keeping imported images inside the project's own
        folder -- rather than only referencing wherever the user dragged them
        from -- is what makes them keep opening after the source moves, and
        (the reason this exists) survives macOS's per-folder privacy gate:
        ``~/Downloads``, ``~/Desktop`` and ``~/Documents`` are TCC-protected,
        so a file merely *referenced* there fails to read later with
        ``Operation not permitted`` even though it plainly exists -- exactly
        the "can't open/read file" storm reported against real projects. A
        copy under the app's own store is always readable.
        """
        d = self._dir(project_id) / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def import_image(self, project_id: str, src_path: str | Path) -> Path:
        """Copy ``src_path`` into the project's ``images/`` dir and return the
        stored path. Name collisions get a numeric suffix so two different
        sources with the same filename never clobber each other. Raises (does
        not silently swallow) if the source can't be read -- the caller
        decides whether to fall back to referencing the original path.
        """
        import shutil
        src = Path(src_path)
        dest_dir = self.image_dir(project_id)
        dest = dest_dir / src.name
        n = 2
        while dest.exists():
            dest = dest_dir / f"{src.stem}-{n}{src.suffix}"
            n += 1
        shutil.copy2(src, dest)
        return dest

    def import_images(self, project_id: str, paths) -> list[str]:
        """Copy each of ``paths`` into the project (see ``import_image``),
        returning the list of stored paths. A source that can't be copied
        right now (already gone, or a transient read failure) falls back to
        its original path rather than being dropped -- the reference is kept
        so the UI can still show a clear "can't read" state for it instead of
        silently losing the image. An already-inside-the-store path is left
        as-is (idempotent: re-importing doesn't copy a copy)."""
        store_root = str(self.root.resolve())
        out: list[str] = []
        for p in paths:
            try:
                if str(Path(p).resolve()).startswith(store_root):
                    out.append(str(p))  # already ours -- don't re-copy
                else:
                    out.append(str(self.import_image(project_id, p)))
            except Exception:
                out.append(str(p))  # keep the reference; UI surfaces the failure
        return out

    # ── Queries ─────────────────────────────────────────────────────────────
    def list(self) -> list[Project]:
        """All projects, newest-modified first. Corrupt files are skipped."""
        out: list[Project] = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            f = child / self.PROJECT_FILE
            if not f.exists():
                continue
            try:
                out.append(Project.from_dict(json.loads(f.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError, TypeError):
                continue  # tolerate a bad file rather than crash the library
        out.sort(key=lambda p: p.updated_at, reverse=True)
        return out

    def recent(self, limit: int = 6) -> list[Project]:
        """The ``limit`` most-recently-modified projects."""
        return self.list()[:limit]

    def favorites(self) -> list[Project]:
        """Favourited projects, newest-modified first."""
        return [p for p in self.list() if p.favorite]

    def set_favorite(self, project_id: str, value: bool) -> Project:
        """Toggle a project's favourite flag and persist it."""
        project = self.load(project_id)
        project.favorite = bool(value)
        self.save(project)
        return project

    def rename(self, project_id: str, new_name: str) -> Project:
        """Rename a project in place -- the id (and its on-disk folder) is
        unaffected, so anything already referencing the id keeps working."""
        project = self.load(project_id)
        project.name = new_name.strip() or project.name
        self.save(project)
        return project

    def duplicate(self, project_id: str) -> Project:
        """Copy a project's settings/tags/image references into a new
        project. Mirrors Label Studio's own "Duplicate": configuration
        carries over, results don't -- stats/favourite/timestamps reset,
        since nothing has run against the copy yet. Image paths are copied
        as references (images live on the filesystem, not owned by the
        project), so both projects point at the same source files with
        independent settings/results from here on.
        """
        source = self.load(project_id)
        return self.create(
            f"{source.name} copy",
            description=source.description,
            tags=list(source.tags),
            settings=ProjectSettings.from_dict(source.settings.to_dict()),
            image_paths=list(source.image_paths),
        )


def default_store_root() -> Path:
    """The conventional projects root: ``<STORAGE_DIR>/projects``.

    Imported lazily so this module has no hard dependency on ``project_root``
    (and stays trivially testable with an explicit temp root).
    """
    try:
        from project_root import STORAGE_DIR
        return Path(STORAGE_DIR) / "projects"
    except Exception:
        # Fallback for odd launch contexts; never blocks project creation.
        return Path.home() / ".cellseg1" / "projects"
