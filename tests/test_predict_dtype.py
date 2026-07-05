"""Regression: the SAM/CellSeg1 read path must hand engines uint8.

A 16-bit PNG (e.g. a uint16 phantom/GT image) previously reached SAM as uint16
and crashed in torchvision's ``to_pil_image`` ("Input type uint16 is not
supported"). ``_to_display_uint8`` converts non-uint8 input while leaving
ordinary uint8 images byte-for-byte unchanged.
"""
import numpy as np
import pytest

pytest.importorskip("PyQt6")  # predict_widget imports Qt at module import time

from napari_app.widgets.predict_widget import _to_display_uint8


def test_uint8_passthrough_is_identical():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    out = _to_display_uint8(img)
    assert out.dtype == np.uint8
    assert out is img  # unchanged object → default path stays byte-for-byte


def test_uint16_is_converted_to_uint8():
    img = (np.arange(16 * 16, dtype=np.uint16).reshape(16, 16) * 100)  # up to ~25k
    img = np.stack([img, img, img], axis=-1)
    out = _to_display_uint8(img)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert int(out.max()) <= 255 and int(out.min()) >= 0
    assert out.max() > out.min()  # real contrast preserved


def test_float_image_is_converted():
    img = np.linspace(0.0, 5.0, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    out = _to_display_uint8(img)
    assert out.dtype == np.uint8
    assert out.max() > 0


def test_flat_image_does_not_crash():
    img = np.full((8, 8, 3), 1234, dtype=np.uint16)
    out = _to_display_uint8(img)
    assert out.dtype == np.uint8
    assert out.shape == (8, 8, 3)
