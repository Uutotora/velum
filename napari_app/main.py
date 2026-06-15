import sys
import os

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import napari
from napari_app.widgets.train_widget import TrainWidget
from napari_app.widgets.predict_widget import PredictWidget


def main():
    viewer = napari.Viewer(title="CellSeg1 — Cell Instance Segmentation")

    train_widget = TrainWidget(viewer)
    predict_widget = PredictWidget(viewer)

    viewer.window.add_dock_widget(train_widget, name="Train", area="right")
    viewer.window.add_dock_widget(predict_widget, name="Predict", area="right")

    napari.run()


if __name__ == "__main__":
    main()
