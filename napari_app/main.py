import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import napari
from PyQt6.QtWidgets import QTabWidget, QWidget, QVBoxLayout
from napari_app.widgets.train_widget import TrainWidget
from napari_app.widgets.predict_widget import PredictWidget

TAB_SS = """
QTabWidget::pane {
    border: none;
    background: #262930;
}
QTabBar::tab {
    background: #1e2128;
    color: #868e93;
    padding: 8px 24px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: 12px;
    font-weight: 500;
}
QTabBar::tab:selected {
    color: #f0f1f2;
    border-bottom: 2px solid #007acc;
    background: #262930;
}
QTabBar::tab:hover:!selected {
    color: #c0c4cc;
    background: #2a2f38;
}
"""


def main():
    viewer = napari.Viewer(title="CellSeg1 — Cell Instance Segmentation")

    tabs = QTabWidget()
    tabs.setStyleSheet(TAB_SS)

    predict_widget = PredictWidget(viewer)
    train_widget   = TrainWidget(viewer)

    tabs.addTab(predict_widget, "Predict")
    tabs.addTab(train_widget,   "Train")

    dock = viewer.window.add_dock_widget(tabs, name="CellSeg1", area="right")
    dock.setMinimumWidth(340)

    napari.run()


if __name__ == "__main__":
    main()
