"""Velum — the Projects tab controller.

Qt-free glue between the ``Project``/``ProjectStore`` data model
(``studio/project.py``) and the Home/Projects screens: search + filter,
favourites, the "active project" shared with the Workspace tab, and seeding a
fresh install with the same sample projects the mockup has always shown (now
real, persisted ``Project`` records instead of hard-coded ``demo`` content).

Mirrors ``velum_core/predict_controller.py``'s shape — plain data in,
plain callbacks out, no Qt — so it is unit-tested without PyQt6.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from studio.project import (
    ENGINE_LABELS,
    Project,
    ProjectSettings,
    ProjectStats,
    ProjectStore,
    default_store_root,
    slugify,
)

# (name, description, engine, tags, favorite, n_images, n_cells, progress, last_f1)
_SEED_PROJECTS: list[tuple] = [
    ("Fluorescence Nuclei — DAPI",
     "384-well DAPI screen, one-shot LoRA fine-tuned on a single field.",
     "cellseg1", ["fluorescence", "nuclei"], True, 128, 31400, 96, 0.94),
    ("H&E Tissue Cohort",
     "Whole-slide H&E biopsies, tiled at native resolution across 12 patients.",
     "cellpose", ["histology", "H&E"], False, 342, 188000, 41, None),
    ("Live-cell Mitosis",
     "Confocal z-stacks tracked across time with SAM 2 propagation.",
     "sam2", ["time-lapse", "3D"], True, 24, 9700, 70, 0.90),
    ("BBBC039 Nuclei Benchmark",
     "Public benchmark for regression-testing engine accuracy.",
     "cellseg1", ["benchmark"], False, 200, 52000, 100, 0.91),
    ("Organoid Membranes",
     "Brightfield organoid sections, membrane-channel segmentation.",
     "cellpose", ["membrane", "brightfield"], False, 88, 14200, 33, None),
    ("Phantom QC",
     "Synthetic phantoms for daily pipeline quality control.",
     "cellseg1", ["QC", "synthetic"], False, 12, 1100, 100, 0.98),
]


def format_count(n: int) -> str:
    """Compact card stat: ``128`` -> ``"128"``, ``31400`` -> ``"31.4k"``."""
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k".replace(".0k", "k")


def cover_seed(project_id: str) -> int:
    """Deterministic seed for a project's procedural cover art (paint.py).

    Stable across runs/relaunches for a given id, without storing a rendering
    detail in the domain model.
    """
    return zlib.crc32(project_id.encode("utf-8")) % 1000


def relative_time(iso_ts: str) -> str:
    """Human "N units ago" string for an ISO-8601 timestamp, e.g. ``updated_at``."""
    try:
        then = datetime.fromisoformat(iso_ts)
    except (TypeError, ValueError):
        return ""
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - then).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        n = int(seconds // 60)
        return f"{n} minute{'s' if n != 1 else ''} ago"
    if seconds < 86400:
        n = int(seconds // 3600)
        return f"{n} hour{'s' if n != 1 else ''} ago"
    if seconds < 172800:
        return "yesterday"
    if seconds < 604800:
        n = int(seconds // 86400)
        return f"{n} days ago"
    n = int(seconds // 604800)
    return f"{n} week{'s' if n != 1 else ''} ago"


@dataclass
class ProjectCard:
    """Display-ready view of a ``Project`` for the Home/Projects card renderers."""

    id: str
    name: str
    description: str
    engine_key: str
    engine_label: str
    n_images: int
    n_cells: str
    progress: int
    f1: Optional[str]
    tags: list[str]
    seed: int
    favorite: bool
    when: str


def to_card(project: Project) -> ProjectCard:
    """Adapt a ``Project`` into the pre-formatted fields the screens render."""
    f1 = project.stats.last_f1
    return ProjectCard(
        id=project.id,
        name=project.name,
        description=project.description,
        engine_key=project.engine,
        engine_label=ENGINE_LABELS.get(project.engine, project.engine),
        n_images=project.stats.n_images,
        n_cells=format_count(project.stats.n_cells),
        progress=project.stats.progress,
        f1=f"{f1:.2f}" if f1 is not None else None,
        tags=list(project.tags),
        seed=cover_seed(project.id),
        favorite=project.favorite,
        when=relative_time(project.updated_at),
    )


class ProjectController:
    """Owns the ``ProjectStore``: search/filter, favourites, the active project."""

    def __init__(self, store: Optional[ProjectStore] = None,
                 store_root: Optional[Path | str] = None,
                 seed_if_empty: bool = True):
        self.store = store or ProjectStore(store_root or default_store_root())
        self._active_id: Optional[str] = None
        if seed_if_empty and not self.store.list():
            self._seed_sample_projects()

    def _seed_sample_projects(self) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        for i, (name, desc, engine, tags, favorite, n_images, n_cells, progress, f1) \
                in enumerate(_SEED_PROJECTS):
            # Stagger timestamps (newest = first in the list) so the seeded
            # grid has a well-defined order — ProjectStore.list() sorts by
            # updated_at, and same-second "now" stamps would otherwise tie.
            ts = (now - timedelta(seconds=i)).isoformat()
            project = Project(
                id=slugify(name),
                name=name,
                description=desc,
                created_at=ts,
                updated_at=ts,
                tags=list(tags),
                favorite=favorite,
                settings=ProjectSettings(engine=engine),
                stats=ProjectStats(n_images=n_images, n_cells=n_cells,
                                    last_f1=f1, progress=progress),
            )
            self.store.save(project, touch=False)

    # Display label -> sort key, each already in the right order (name is
    # the one ascending case; everything else is "biggest/newest first").
    # A dict (not a bare set of strings) so the toolbar's SelectBox options
    # and the controller's accepted sort= values can never drift apart --
    # SORT_OPTIONS is the single source both read from.
    SORT_OPTIONS: dict[str, str] = {
        "Last modified": "modified",
        "Name (A–Z)": "name",
        "Date created": "created",
        "Most cells": "cells",
    }

    # ── queries ──────────────────────────────────────────────────────────────
    def list_projects(self, query: str = "", favorites_only: bool = False,
                       engines: Optional[set[str]] = None,
                       sort: str = "modified") -> list[Project]:
        """Projects matching ``query`` (name/description/tags/engine).

        ``engines``, when non-empty, restricts to projects whose engine key is
        in the set (the Projects tab's "Filter" popover) — composes with
        ``query``/``favorites_only``. ``sort`` is one of ``SORT_OPTIONS``'s
        values (default ``"modified"``, matching the store's own newest-first
        order — the only one that needs no re-sort).
        """
        projects = self.store.list()
        if favorites_only:
            projects = [p for p in projects if p.favorite]
        if engines:
            projects = [p for p in projects if p.engine in engines]
        q = query.strip().lower()
        if q:
            def matches(p: Project) -> bool:
                haystack = [p.name, p.description, p.engine,
                            ENGINE_LABELS.get(p.engine, ""), *p.tags]
                return any(q in h.lower() for h in haystack)
            projects = [p for p in projects if matches(p)]
        if sort == "name":
            projects = sorted(projects, key=lambda p: p.name.lower())
        elif sort == "created":
            projects = sorted(projects, key=lambda p: p.created_at, reverse=True)
        elif sort == "cells":
            projects = sorted(projects, key=lambda p: p.stats.n_cells, reverse=True)
        return projects

    def recent(self, limit: int = 4) -> list[Project]:
        return self.store.recent(limit=limit)

    def summary(self) -> tuple[int, int, int]:
        """``(n_projects, n_images_total, n_distinct_engines)`` for the page header."""
        projects = self.store.list()
        n_images = sum(p.stats.n_images for p in projects)
        n_engines = len({p.engine for p in projects})
        return len(projects), n_images, n_engines

    # ── mutation ─────────────────────────────────────────────────────────────
    def toggle_favorite(self, project_id: str) -> Project:
        project = self.store.load(project_id)
        return self.store.set_favorite(project_id, not project.favorite)

    def rename_project(self, project_id: str, new_name: str) -> Project:
        return self.store.rename(project_id, new_name)

    def duplicate_project(self, project_id: str) -> Project:
        return self.store.duplicate(project_id)

    def delete_project(self, project_id: str) -> None:
        """Permanently delete a project. Irreversible -- gated behind the
        Project Settings screen's Danger Zone confirmation, same as Label
        Studio's own Danger Zone. Clears it as the active project (shared
        with the Workspace tab) if it was the one open."""
        self.store.delete(project_id)
        if self._active_id == project_id:
            self._active_id = None

    # ── active project (shared with the Workspace tab) ────────────────────────
    def set_active(self, project_id: str) -> Project:
        project = self.store.load(project_id)
        self._active_id = project_id
        return project

    def get_active(self) -> Optional[Project]:
        if self._active_id is None:
            return None
        try:
            return self.store.load(self._active_id)
        except FileNotFoundError:
            return None
