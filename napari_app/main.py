import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import napari
from PyQt6.QtCore import QLocale, Qt
from napari_app.widgets.train_widget import TrainWidget
from napari_app.widgets.predict_widget import PredictWidget
from napari_app.widgets.assistant_widget import AssistantWidget
from napari_app.widgets.annotate_widget import AnnotateWidget
from napari_app.widgets.guide_widget import GuideWidget
from napari_app.widgets.shell import Shell
from napari_app.theme import WIDGET_SS


def main():
    # Force en-US locale so spinboxes always use dot decimal, never comma (§01)
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))

    viewer = napari.Viewer(title="CellSeg1 — Cell Instance Segmentation")

    # Stop wheel-scroll from accidentally changing spin boxes / combos.
    from PyQt6.QtWidgets import QApplication
    from napari_app.ui_utils import install_wheel_guard
    app = QApplication.instance()
    if app is not None:
        install_wheel_guard(app)

    predict_widget   = PredictWidget(viewer)
    train_widget     = TrainWidget(viewer)
    assistant_widget = AssistantWidget(viewer, predict_widget)
    annotate_widget  = AnnotateWidget(viewer, predict_widget)
    guide_widget     = GuideWidget(viewer)

    # Left icon-rail shell (brand header + status footer). Same widgets, same
    # wiring — only the navigation chrome changed from top tabs to a rail.
    shell = Shell([
        ("predict",   "Predict",   predict_widget),
        ("annotate",  "Annotate",  annotate_widget),
        ("assistant", "Assistant", assistant_widget),
        ("train",     "Train",     train_widget),
        ("guide",     "Guide",     guide_widget),
    ])
    shell.setStyleSheet(WIDGET_SS)

    dock = viewer.window.add_dock_widget(shell, name="CellSeg1", area="right")
    dock.setMinimumWidth(340)
    # Open the dock comfortably wide so content never squeezes behind the rail.
    try:
        win = viewer.window._qt_window
        win.resizeDocks([dock], [500], Qt.Orientation.Horizontal)
    except Exception:
        pass

    napari.run()


if __name__ == "__main__":
    main()
