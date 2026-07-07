"""CellSeg1 Studio — the standalone desktop application layer.

This package turns CellSeg1 from a napari *plugin* (a dock inside napari's
window) into a self-contained desktop *product* that owns its own window:
a Home screen, a Projects library, and a Workspace that embeds napari's
canvas as one component among many (layers, segment settings, results,
Assistant, Logs, Dashboard).

It is deliberately additive. The classic entry point
(:func:`napari_app.main.main`, ``run_napari.sh``, the ``cellseg1`` console
script) is left byte-for-byte unchanged; Studio ships behind its own entry
point (``cellseg1-studio`` / ``run_studio.sh``) so the proven app keeps
working and you can revert instantly by launching the old one.

Layering (import direction is one-way, leaf → shell):

- :mod:`napari_app.studio.project` — the **data layer**. A ``Project`` and a
  ``ProjectStore``, carrying every predict/train setting, persisted as JSON.
  Pure Python: no Qt, no torch, no napari. Fully unit-tested.
- :mod:`napari_app.studio.theme` — the **design layer**. The Studio identity
  (light + dark token palettes, QSS builders). Pure strings, unit-tested.
- :mod:`napari_app.studio.app` — the **shell** (Qt). Imported lazily by the
  entry point so this package stays importable without PyQt6.
"""

from napari_app.studio.project import (  # noqa: F401
    Project,
    ProjectSettings,
    ProjectStore,
)

__all__ = ["Project", "ProjectSettings", "ProjectStore"]
