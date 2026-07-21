"""Pure-logic tests for microscopy format reading (velum_core.channels).

Covers physical pixel-size extraction (OME-TIFF + baseline TIFF resolution
tags), unit conversion, the shared axes→stack transform, and graceful
degradation when an optional reader (nd2/czi/lif) is not installed. No
torch/napari/GPU; ND2/CZI readers are exercised through fake modules injected
into ``sys.modules`` so we never need the real (unavailable) libraries.
"""
import sys
import types

import numpy as np
import pytest
import tifffile

from velum_core import channels as ch


# ── unit conversion ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,unit,expected", [
    (0.5, "µm", 0.5),
    (0.5, "um", 0.5),
    (0.5, "micron", 0.5),
    (500, "nm", 0.5),
    (0.0005, "mm", 0.5),
    (1, None, 1.0),            # missing unit defaults to microns
    (2, "microns", 2.0),       # trailing plural stripped
])
def test_physical_size_to_um(value, unit, expected):
    assert ch.physical_size_to_um(value, unit) == pytest.approx(expected)


def test_physical_size_inch_is_micron_scaled():
    assert ch.physical_size_to_um(1, "inch") == pytest.approx(25400.0)


def test_physical_size_rejects_bad_input():
    assert ch.physical_size_to_um(0, "um") is None       # non-positive
    assert ch.physical_size_to_um(-1, "um") is None
    assert ch.physical_size_to_um(1, "furlong") is None  # unknown unit
    assert ch.physical_size_to_um("nope", "um") is None


# ── shared axes→stack transform ─────────────────────────────────────────────

def test_stack_from_axes_reduces_z_and_finds_channel():
    arr = np.zeros((3, 4, 20, 30), dtype=np.uint16)   # Z,C,Y,X
    arr[0, 2] = 7
    stack = ch.stack_from_axes_array(arr, "ZCYX", pixel_size_um=0.25)
    assert stack.shape == (20, 30)
    assert stack.n_channels == 4
    assert stack.pixel_size_um == 0.25
    assert stack.channel(2).max() == 7                # first Z plane kept


def test_stack_from_axes_mismatched_axes_falls_back():
    arr = np.zeros((4, 20, 30), dtype=np.uint16)      # C,H,W but wrong axes len
    stack = ch.stack_from_axes_array(arr, "XY", pixel_size_um=1.0)
    assert stack.n_channels == 4                       # heuristic still works
    assert stack.pixel_size_um == 1.0


# ── pixel size from real TIFF files ──────────────────────────────────────────

def test_ome_tiff_pixel_size_and_dims(tmp_path):
    arr = np.zeros((4, 24, 24), dtype=np.uint16)      # C,Y,X
    path = tmp_path / "cal.ome.tif"
    tifffile.imwrite(
        path, arr, photometric="minisblack",
        metadata={"axes": "CYX",
                  "PhysicalSizeX": 0.32, "PhysicalSizeXUnit": "µm",
                  "PhysicalSizeY": 0.32, "PhysicalSizeYUnit": "µm"})
    stack = ch.read_channel_stack(path)
    assert stack.n_channels == 4
    assert stack.pixel_size_um == pytest.approx(0.32)
    assert ch.read_pixel_size_um(path) == pytest.approx(0.32)


def test_ome_tiff_nanometer_unit_converted(tmp_path):
    path = tmp_path / "nm.ome.tif"
    tifffile.imwrite(
        path, np.zeros((16, 16), dtype=np.uint16), photometric="minisblack",
        metadata={"axes": "YX",
                  "PhysicalSizeX": 320.0, "PhysicalSizeXUnit": "nm"})
    assert ch.read_pixel_size_um(path) == pytest.approx(0.32)


def test_plain_tiff_resolution_tags(tmp_path):
    # 20000 pixels/cm → 0.5 µm/pixel.
    path = tmp_path / "res.tif"
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint8),
                     resolution=(20000, 20000), resolutionunit="CENTIMETER")
    assert ch.read_pixel_size_um(path) == pytest.approx(0.5, rel=1e-3)


def test_plain_png_has_no_pixel_size(tmp_path):
    import cv2
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), np.zeros((8, 8, 3), dtype=np.uint8))
    assert ch.read_pixel_size_um(p) is None


# ── graceful degradation for optional readers ───────────────────────────────

def test_unknown_format_degrades_gracefully(tmp_path):
    p = tmp_path / "mystery.xyz"
    p.write_bytes(b"not an image")
    # read_pixel_size never raises; it just reports "unknown".
    assert ch.read_pixel_size_um(p) is None


def test_nd2_without_reader_raises_friendly_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "nd2", None)  # force ImportError on import
    p = tmp_path / "cells.nd2"
    p.write_bytes(b"")
    with pytest.raises(ch.MissingReaderError) as exc:
        ch.read_channel_stack(p)
    assert "nd2" in str(exc.value)
    assert "pip install nd2" in str(exc.value)


def test_czi_without_reader_raises_friendly_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "czifile", None)
    p = tmp_path / "cells.czi"
    p.write_bytes(b"")
    with pytest.raises(ch.MissingReaderError) as exc:
        ch.read_channel_stack(p)
    assert "czifile" in str(exc.value)


# ── ND2 reading via a fake nd2 module ────────────────────────────────────────

class _FakeND2File:
    def __init__(self, arr, sizes, channel_names, vx):
        self._arr, self.sizes = arr, sizes
        self._names, self._vx = channel_names, vx

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def asarray(self):
        return self._arr

    def voxel_size(self):
        return types.SimpleNamespace(x=self._vx, y=self._vx, z=1.0)

    @property
    def metadata(self):
        chans = [types.SimpleNamespace(channel=types.SimpleNamespace(name=n))
                 for n in self._names]
        return types.SimpleNamespace(channels=chans)


def _install_fake_nd2(monkeypatch, arr, sizes, names, vx):
    mod = types.ModuleType("nd2")
    mod.ND2File = lambda path: _FakeND2File(arr, sizes, names, vx)
    monkeypatch.setitem(sys.modules, "nd2", mod)


def test_nd2_reader_dims_names_and_pixel_size(tmp_path, monkeypatch):
    arr = np.zeros((3, 32, 40), dtype=np.uint16)       # C,Y,X
    arr[1] = 100
    _install_fake_nd2(monkeypatch, arr, {"C": 3, "Y": 32, "X": 40},
                      ["DAPI", "GFP", "RFP"], vx=0.216)
    p = tmp_path / "cells.nd2"
    p.write_bytes(b"")
    stack = ch.read_channel_stack(p)
    assert stack.shape == (32, 40)
    assert stack.n_channels == 3
    assert stack.names == ["DAPI", "GFP", "RFP"]
    assert stack.pixel_size_um == pytest.approx(0.216)
    assert stack.channel(1).max() == 100


def test_nd2_reader_reduces_z_axis(tmp_path, monkeypatch):
    arr = np.zeros((5, 2, 16, 24), dtype=np.uint16)    # Z,C,Y,X
    arr[0, 0] = 9
    _install_fake_nd2(monkeypatch, arr, {"Z": 5, "C": 2, "Y": 16, "X": 24},
                      ["a", "b"], vx=0.5)
    p = tmp_path / "zstack.nd2"
    p.write_bytes(b"")
    stack = ch.read_channel_stack(p)
    assert stack.shape == (16, 24)
    assert stack.n_channels == 2
    assert stack.channel(0).max() == 9                 # first z kept
