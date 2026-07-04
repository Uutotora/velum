"""Wiring test: the Predict widget's large-image path routes through tiling.

Exercises napari_app.widgets.predict_widget._predict_tiled with a fake engine
(connected components) so no model/GPU is needed. Skipped in the lightweight CI
image, which has no PyQt6/napari installed.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from scipy import ndimage

pytest.importorskip("PyQt6")  # GUI stack absent in the fast CI job → skip there
pw = pytest.importorskip("napari_app.widgets.predict_widget")


def _cc(tile_img: np.ndarray) -> np.ndarray:
    fg = tile_img[..., 0] > 0 if tile_img.ndim == 3 else tile_img > 0
    lab, _ = ndimage.label(fg)
    return lab.astype(np.int32)


def test_predict_tiled_cellpose_branch_stitches(monkeypatch):
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img[60:140, 40:360] = 255                     # one object across many tiles
    cfg = {"engine": "cellpose", "tile_size": 128, "tile_overlap": 64,
           "clahe": False}
    out = pw._predict_tiled(cfg, img)
    assert out.shape == (200, 400)
    assert out.max() == 1                          # reassembled into one cell
    assert (out > 0).sum() == int((img[..., 0] > 0).sum())


def test_predict_tiled_sam_branch_stitches(monkeypatch):
    import napari_app.inference_cache as ic
    monkeypatch.setattr(ic, "predict_cached", lambda cfg, t: _cc(t))

    img = np.zeros((160, 320, 3), dtype=np.uint8)
    img[40:120, 20:300] = 200
    cfg = {"engine": "cellseg1", "tile_size": 100, "tile_overlap": 40,
           "clahe": False}
    out = pw._predict_tiled(cfg, img)
    assert out.max() == 1


def test_predict_tiled_counts_separate_cells(monkeypatch):
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    img = np.zeros((160, 480, 3), dtype=np.uint8)
    for cx in (40, 230, 420):                      # spacing > overlap → distinct
        img[70:90, cx:cx + 20] = 255
    cfg = {"engine": "cellpose", "tile_size": 128, "tile_overlap": 48,
           "clahe": False}
    out = pw._predict_tiled(cfg, img)
    assert out.max() == 3


def test_predict_tiled_threads_on_tile_progress(monkeypatch):
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    img = np.zeros((160, 480, 3), dtype=np.uint8)
    img[70:90, 40:60] = 255
    cfg = {"engine": "cellpose", "tile_size": 128, "tile_overlap": 48,
           "clahe": False}

    calls = []
    pw._predict_tiled(cfg, img, on_tile=lambda d, t: calls.append((d, t)))

    from napari_app.tiling import plan_tiles
    n = len(plan_tiles(*img.shape[:2], tile=128, overlap=48))
    assert n > 1                                   # actually tiled
    assert calls == [(i + 1, n) for i in range(n)]  # 1/n .. n/n, monotonic


def test_predict_cached_forwards_on_tile_to_tiler(monkeypatch, tmp_path):
    import cv2
    import napari_app.engines as engines
    monkeypatch.setattr(engines, "predict_cellpose", lambda t, **k: _cc(t))

    img = np.zeros((160, 480, 3), dtype=np.uint8)
    img[70:90, 40:60] = 255
    path = tmp_path / "big.png"
    cv2.imwrite(str(path), img)
    cfg = {"engine": "cellpose", "tile_size": 128, "tile_overlap": 48,
           "clahe": False, "tiled": True, "image_path": str(path),
           "resize_size": 256}

    calls = []
    _, mask = pw._predict_cached(cfg, on_tile=lambda d, t: calls.append((d, t)))
    assert mask.shape == (160, 480)
    assert calls and calls[-1][0] == calls[-1][1]  # ends at total/total


def test_predict_cached_non_tiled_ignores_on_tile(monkeypatch, tmp_path):
    import cv2
    import napari_app.inference_cache as ic
    monkeypatch.setattr(ic, "predict_cached", lambda cfg, t: _cc(t))

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[20:40, 20:40] = 255
    path = tmp_path / "small.png"
    cv2.imwrite(str(path), img)
    cfg = {"engine": "cellseg1", "tiled": True, "image_path": str(path),
           "resize_size": [64, 64], "clahe": False, "tile_size": 1024}

    calls = []
    _, mask = pw._predict_cached(cfg, on_tile=lambda d, t: calls.append((d, t)))
    assert calls == []                             # small image → no tiling, no progress
    assert mask.shape == (64, 64)
