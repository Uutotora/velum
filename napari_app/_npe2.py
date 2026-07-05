"""npe2 entry points backing the napari plugin manifest (``napari.yaml``).

These thin factories let CellSeg1's dock widgets be discovered through napari's
plugin system (Plugins ▸ CellSeg1) in addition to the bundled ``cellseg1`` app
launcher (see ``napari_app/main.py``). The heavy widget import is deferred to
call time so merely loading the manifest stays cheap and torch/PyQt are only
imported when the widget is actually opened.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import napari

    from napari_app.widgets.predict_widget import PredictWidget


def make_predict_widget(viewer: "napari.Viewer") -> "PredictWidget":
    """Return the CellSeg1 Predict dock widget bound to ``viewer``.

    napari injects the active ``Viewer`` via the annotated parameter.
    """
    from napari_app.widgets.predict_widget import PredictWidget

    return PredictWidget(viewer)
