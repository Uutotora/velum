import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import napari
from PyQt6.QtWidgets import QTabWidget, QWidget, QVBoxLayout
from PyQt6.QtCore import QLocale
from napari_app.widgets.train_widget import TrainWidget
from napari_app.widgets.predict_widget import PredictWidget
from napari_app.widgets.assistant_widget import AssistantWidget

TAB_SS = """
QTabWidget::pane {
    border: none;
    background: #14192a;
}
QTabBar::tab {
    background: #0e1220;
    color: #5e6d88;
    padding: 8px 24px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: 12px;
    font-weight: 500;
}
QTabBar::tab:selected {
    color: #dce4f0;
    border-bottom: 2px solid #4d8fff;
    background: #14192a;
}
QTabBar::tab:hover:!selected {
    color: #8a9bbe;
    background: #181f30;
}
"""


def main():
    # Force en-US locale so spinboxes always use dot decimal, never comma (§01)
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))

    viewer = napari.Viewer(title="CellSeg1 — Cell Instance Segmentation")

    tabs = QTabWidget()
    tabs.setStyleSheet(TAB_SS)

    predict_widget   = PredictWidget(viewer)
    train_widget     = TrainWidget(viewer)
    assistant_widget = AssistantWidget(viewer, predict_widget)

    tabs.addTab(predict_widget,   "Predict")
    tabs.addTab(assistant_widget, "Assistant")
    tabs.addTab(train_widget,     "Train")

    dock = viewer.window.add_dock_widget(tabs, name="CellSeg1", area="right")
    dock.setMinimumWidth(340)

    napari.run()


if __name__ == "__main__":
    main()
