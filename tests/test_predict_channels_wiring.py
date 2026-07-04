"""Wiring test: the Predict read path handles multi-channel stacks (opt-in).

Exercises napari_app.widgets.predict_widget._read_for_predict / _predict_cached
with a fake engine. Skipped in the lightweight CI image (no PyQt6/napari).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pw = pytest.importorskip("napari_app.widgets.predict_widget")


def test_read_for_predict_default_path_no_stack(tmp_path):
    import cv2
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    img[5:15, 5:25] = (10, 20, 30)
    path = tmp_path / "rgb.png"
    cv2.imwrite(str(path), img)

    rgb, stack = pw._read_for_predict({"image_path": str(path)})
    assert stack is None                          # ordinary path → no channel stack
    assert rgb.shape == (20, 30, 3)
    # cv2 BGR→RGB conversion preserved exactly as the legacy path did.
    assert tuple(rgb[10, 10]) == (30, 20, 10)


def test_read_for_predict_channel_path_projects_and_returns_stack(tmp_path):
    import tifffile
    arr = np.stack([np.full((24, 32), i * 40, dtype=np.uint16) for i in range(5)])  # C,H,W
    path = tmp_path / "five.tif"
    tifffile.imwrite(path, arr)

    rgb, stack = pw._read_for_predict(
        {"image_path": str(path), "channels": [0, 2]})
    assert rgb.shape == (24, 32, 3)
    assert rgb[..., 2].max() == 0                 # 2 channels → blue slot empty
    assert stack is not None and stack.n_channels == 5


def test_predict_cached_populates_sink_with_stack(tmp_path, monkeypatch):
    import tifffile
    from scipy import ndimage
    import napari_app.engines as engines

    def _cc(tile_img):
        fg = tile_img[..., 0] > 0 if tile_img.ndim == 3 else tile_img > 0
        return ndimage.label(fg)[0].astype(np.int32)

    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    arr = np.zeros((4, 40, 50), dtype=np.uint16)  # C,H,W
    arr[1, 10:30, 10:40] = 500                     # signal in channel 1
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, arr)

    cfg = {"engine": "cellpose", "image_path": str(path), "channels": [1],
           "resize_size": [40, 50], "clahe": False, "tile_size": 40}
    sink = {}
    rgb, mask = pw._predict_cached(cfg, sink=sink)
    assert sink["stack"] is not None
    assert sink["stack"].n_channels == 4
    assert mask.shape == (40, 50)


def test_predict_cached_default_path_sink_stack_none(tmp_path, monkeypatch):
    import cv2
    from scipy import ndimage
    import napari_app.inference_cache as ic
    monkeypatch.setattr(ic, "predict_cached",
                        lambda cfg, t: ndimage.label(t[..., 0] > 0)[0].astype(np.int32))

    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[10:30, 10:30] = 200
    path = tmp_path / "small.png"
    cv2.imwrite(str(path), img)
    cfg = {"engine": "cellseg1", "image_path": str(path),
           "resize_size": [40, 40], "clahe": False, "tile_size": 1024}
    sink = {}
    _, mask = pw._predict_cached(cfg, sink=sink)
    assert sink["stack"] is None                  # no channels → default read
    assert mask.shape == (40, 40)
