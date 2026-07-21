"""Pure-logic tests for multi-channel microscopy support (velum_core.channels).

No torch/napari/GPU — runs in the lightweight CI job. Multi-page and OME-TIFF
reading is exercised through tifffile round-trips on synthetic stacks.
"""
import sys
import types

import numpy as np
import pytest

from velum_core import channels as ch


# ── canonicalisation ────────────────────────────────────────────────────────

def test_to_channel_stack_2d_becomes_single_channel():
    stack = ch.to_channel_stack(np.zeros((10, 12)))
    assert stack.data.shape == (10, 12, 1)
    assert stack.n_channels == 1
    assert stack.names == ["Channel 0"]


def test_to_channel_stack_infers_channel_first():
    arr = np.zeros((5, 40, 60))              # C,H,W with small C
    stack = ch.to_channel_stack(arr)
    assert stack.shape == (40, 60)
    assert stack.n_channels == 5


def test_to_channel_stack_infers_channel_last():
    arr = np.zeros((40, 60, 3))              # H,W,C RGB-like
    stack = ch.to_channel_stack(arr)
    assert stack.shape == (40, 60)
    assert stack.n_channels == 3


def test_to_channel_stack_explicit_axis_wins():
    arr = np.zeros((4, 4, 7))                # ambiguous-ish; force axis 2
    stack = ch.to_channel_stack(arr, channel_axis=2)
    assert stack.n_channels == 7


def test_to_channel_stack_squeezes_leading_singleton():
    arr = np.zeros((1, 6, 30, 40))           # 1,C,H,W OME page
    stack = ch.to_channel_stack(arr)
    assert stack.shape == (30, 40)
    assert stack.n_channels == 6


def test_to_channel_stack_rejects_higher_dim_without_axis():
    with pytest.raises(ValueError):
        ch.to_channel_stack(np.zeros((5, 6, 30, 40)))


def test_to_channel_stack_names_length_checked():
    with pytest.raises(ValueError):
        ch.to_channel_stack(np.zeros((30, 40, 3)), names=["a", "b"])


# ── percentile normalisation ────────────────────────────────────────────────

def test_percentile_normalize_stretches_to_full_range():
    chan = np.linspace(0, 1000, 100).reshape(10, 10)
    out = ch.percentile_normalize(chan, low=0, high=100)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_percentile_normalize_flat_channel_is_zero():
    out = ch.percentile_normalize(np.full((8, 8), 7.0))
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def test_percentile_normalize_clips_outliers():
    chan = np.zeros((10, 10), dtype=np.float32)
    chan[0, 0] = 10_000                       # single hot pixel
    chan[5:, :] = 100
    out = ch.percentile_normalize(chan, low=1, high=99)
    assert out.max() == 255                    # hot pixel clipped, not scaling everything down
    assert out[5, 0] > 0


def test_percentile_normalize_validates_bounds():
    with pytest.raises(ValueError):
        ch.percentile_normalize(np.zeros((4, 4)), low=90, high=10)


# ── RGB projection ──────────────────────────────────────────────────────────

def _stack(n, h=12, w=15):
    # Each channel a distinct spatial ramp so percentile-normalisation has
    # real dynamic range (a constant channel correctly normalises to zeros).
    base = np.linspace(0, 1, h * w).reshape(h, w)
    data = np.stack([base * (i + 1) * 100.0 for i in range(n)], axis=-1)
    return ch.to_channel_stack(data, channel_axis=2)


def test_project_single_channel_is_gray():
    rgb = ch.project_to_rgb(_stack(4), channels=[1])
    assert rgb.shape[-1] == 3
    assert np.array_equal(rgb[..., 0], rgb[..., 1])
    assert np.array_equal(rgb[..., 1], rgb[..., 2])


def test_project_two_channels_fills_red_green_only():
    s = _stack(4)
    rgb = ch.project_to_rgb(s, channels=[0, 2])
    assert rgb[..., 2].max() == 0              # blue slot empty
    assert rgb[..., 0].max() > 0 and rgb[..., 1].max() > 0


def test_project_three_channels_into_rgb():
    rgb = ch.project_to_rgb(_stack(5), channels=[0, 1, 2])
    assert all(rgb[..., k].max() > 0 for k in range(3))


def test_project_default_uses_first_three():
    rgb = ch.project_to_rgb(_stack(6))
    assert rgb.shape[-1] == 3


def test_project_rejects_out_of_range_channel():
    with pytest.raises(ValueError):
        ch.project_to_rgb(_stack(3), channels=[9])


def test_project_rejects_empty_selection():
    with pytest.raises(ValueError):
        ch.project_to_rgb(_stack(3), channels=[])


# ── per-channel intensity under a mask ──────────────────────────────────────

def test_channel_means_reports_raw_intensity_per_label():
    stack = ch.to_channel_stack(
        np.stack([np.full((6, 6), 10.0), np.full((6, 6), 50.0)], axis=-1),
        channel_axis=2)
    mask = np.zeros((6, 6), dtype=np.int32)
    mask[:3, :] = 1
    mask[3:, :] = 2
    means = ch.channel_means(mask, stack)
    assert means[0][1] == pytest.approx(10.0)  # channel 0, cell 1
    assert means[1][2] == pytest.approx(50.0)  # channel 1, cell 2


def test_channel_means_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        ch.channel_means(np.zeros((4, 4), np.int32), _stack(2, h=8, w=8))


# ── file round-trips ────────────────────────────────────────────────────────

def test_read_multipage_tiff_over_three_channels(tmp_path):
    import tifffile
    arr = np.stack([np.full((20, 30), i, dtype=np.uint16) for i in range(6)])  # C,H,W
    path = tmp_path / "six.tif"
    tifffile.imwrite(path, arr)
    stack = ch.read_channel_stack(path)
    assert stack.shape == (20, 30)
    assert stack.n_channels == 6


def test_probe_channels_multipage(tmp_path):
    import tifffile
    arr = np.stack([np.zeros((16, 16), dtype=np.uint16) for _ in range(5)])
    path = tmp_path / "five.tif"
    tifffile.imwrite(path, arr)
    n, names = ch.probe_channels(path)
    assert n == 5
    assert len(names) == 5


def test_read_ome_tiff_uses_channel_names(tmp_path):
    import tifffile
    arr = np.zeros((4, 24, 24), dtype=np.uint16)   # C,H,W
    for i in range(4):
        arr[i] = i + 1
    path = tmp_path / "cells.ome.tif"
    tifffile.imwrite(
        path, arr, photometric="minisblack",
        metadata={"axes": "CYX",
                  "Channel": {"Name": ["DAPI", "GFP", "RFP", "Cy5"]}})
    stack = ch.read_channel_stack(path)
    assert stack.n_channels == 4
    # tifffile writes the names into OME-XML; our reader should surface them.
    assert stack.names == ["DAPI", "GFP", "RFP", "Cy5"]


def test_probe_channels_plain_rgb(tmp_path):
    import tifffile
    tifffile.imwrite(tmp_path / "rgb.tif",
                     np.zeros((16, 16, 3), dtype=np.uint8), photometric="rgb")
    n, _ = ch.probe_channels(tmp_path / "rgb.tif")
    assert n == 3                              # a normal RGB image → picker stays hidden


# ── z-stack / time-lapse (VolumeStack) ──────────────────────────────────────

def test_stack_from_axes_zstack_keeps_z_as_leading_plane_axis():
    arr = np.zeros((3, 4, 20, 30), dtype=np.uint16)   # Z,C,Y,X
    for z in range(3):
        arr[z, 2] = z + 1                             # distinguish planes
    vol = ch.stack_from_axes_array_zstack(arr, "ZCYX", pixel_size_um=0.25)
    assert vol.n_planes == 3
    assert vol.n_channels == 4
    assert vol.shape == (20, 30)
    assert vol.pixel_size_um == 0.25
    for z in range(3):
        assert vol.plane(z).channel(2).max() == z + 1


def test_stack_from_axes_zstack_prefers_z_over_t_and_reduces_t():
    # T,Z,C,Y,X — a real timepoint x z-stack x channel acquisition.
    arr = np.zeros((2, 3, 2, 8, 10), dtype=np.uint16)
    for z in range(3):
        for c in range(2):
            arr[0, z, c] = z * 100 + c * 10 + 1   # t=0 plane: distinguishable, non-zero
            arr[1, z, c] = 9999                    # t=1: must never surface

    vol = ch.stack_from_axes_array_zstack(arr, "TZCYX")
    assert vol.n_planes == 3
    assert vol.n_channels == 2
    assert vol.shape == (8, 10)
    assert 9999 not in vol.data                    # T reduced to its first plane
    for z in range(3):
        for c in range(2):
            assert vol.plane(z).channel(c).max() == z * 100 + c * 10 + 1


def test_stack_from_axes_zstack_no_stack_axis_gives_single_plane():
    arr = np.zeros((4, 20, 30), dtype=np.uint16)      # C,Y,X — no Z/T at all
    vol = ch.stack_from_axes_array_zstack(arr, "CYX", pixel_size_um=1.0)
    assert vol.n_planes == 1
    assert vol.n_channels == 4
    assert vol.pixel_size_um == 1.0


def test_stack_from_axes_zstack_mismatched_axes_falls_back_single_plane():
    arr = np.zeros((4, 20, 30), dtype=np.uint16)
    vol = ch.stack_from_axes_array_zstack(arr, "XY")   # wrong length for axes
    assert vol.n_planes == 1
    assert vol.n_channels == 4                          # heuristic still finds channels


def test_read_volume_stack_real_zstack_tiff(tmp_path):
    import tifffile
    arr = np.zeros((5, 16, 24), dtype=np.uint16)        # Z,Y,X grayscale z-stack
    for z in range(5):
        arr[z] = (z + 1) * 10
    path = tmp_path / "zstack.tif"
    tifffile.imwrite(path, arr, photometric="minisblack", metadata={"axes": "ZYX"})

    vol = ch.read_volume_stack(path)
    assert vol.n_planes == 5
    assert vol.shape == (16, 24)
    for z in range(5):
        assert vol.plane(z).channel(0).max() == (z + 1) * 10


def test_read_volume_stack_propagates_pixel_size(tmp_path):
    import tifffile
    arr = np.zeros((3, 16, 16), dtype=np.uint16)
    path = tmp_path / "zcal.ome.tif"
    tifffile.imwrite(path, arr, photometric="minisblack",
                     metadata={"axes": "ZYX", "PhysicalSizeX": 0.5, "PhysicalSizeXUnit": "µm"})
    vol = ch.read_volume_stack(path)
    assert vol.pixel_size_um == pytest.approx(0.5)


def test_read_volume_stack_no_z_axis_is_one_plane(tmp_path):
    import tifffile
    arr = np.zeros((4, 20, 30), dtype=np.uint16)        # C,Y,X — a plain multi-channel image
    path = tmp_path / "flat.tif"
    tifffile.imwrite(path, arr, metadata={"axes": "CYX"})
    vol = ch.read_volume_stack(path)
    assert vol.n_planes == 1
    assert vol.n_channels == 4


def test_read_volume_stack_non_tiff_is_one_plane(tmp_path):
    import cv2
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    path = tmp_path / "rgb.png"
    cv2.imwrite(str(path), img)
    vol = ch.read_volume_stack(path)
    assert vol.n_planes == 1
    assert vol.shape == (20, 30)


# ── has_z_stack ──────────────────────────────────────────────────────────────

def test_has_z_stack_true_for_multiplane_z(tmp_path):
    import tifffile
    arr = np.zeros((5, 16, 16), dtype=np.uint16)
    path = tmp_path / "z.tif"
    tifffile.imwrite(path, arr, metadata={"axes": "ZYX"})
    assert ch.has_z_stack(path) is True


def test_has_z_stack_true_for_multiplane_t(tmp_path):
    import tifffile
    arr = np.zeros((4, 16, 16), dtype=np.uint16)
    path = tmp_path / "t.tif"
    tifffile.imwrite(path, arr, metadata={"axes": "TYX"})
    assert ch.has_z_stack(path) is True


def test_has_z_stack_false_for_single_plane(tmp_path):
    import tifffile
    path = tmp_path / "flat.tif"
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint8))
    assert ch.has_z_stack(path) is False


def test_has_z_stack_false_for_length_one_z_axis(tmp_path):
    import tifffile
    arr = np.zeros((1, 16, 16), dtype=np.uint16)
    path = tmp_path / "z1.tif"
    tifffile.imwrite(path, arr, metadata={"axes": "ZYX"})
    assert ch.has_z_stack(path) is False


def test_has_z_stack_false_for_multichannel_single_plane(tmp_path):
    import tifffile
    arr = np.zeros((4, 20, 30), dtype=np.uint16)   # C,Y,X — channels, not a stack
    path = tmp_path / "chan.tif"
    tifffile.imwrite(path, arr, metadata={"axes": "CYX"})
    assert ch.has_z_stack(path) is False


def test_has_z_stack_false_for_non_tiff(tmp_path):
    import cv2
    path = tmp_path / "rgb.png"
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
    assert ch.has_z_stack(path) is False


def test_has_z_stack_false_for_unreadable_file(tmp_path):
    path = tmp_path / "mystery.tif"
    path.write_bytes(b"not a real tiff")
    assert ch.has_z_stack(path) is False


# ── ND2/CZI/LIF volume (z-stack) reading via fake modules ───────────────────
# No real nd2/czifile/readlif packages available here, so every test below
# injects a minimal fake module into sys.modules, mirroring test_formats.py's
# existing _install_fake_nd2 pattern.

class _FakeND2FileVol:
    def __init__(self, arr, sizes):
        self._arr, self.sizes = arr, sizes

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def asarray(self):
        return self._arr


def _install_fake_nd2_vol(monkeypatch, arr, sizes):
    mod = types.ModuleType("nd2")
    mod.ND2File = lambda path: _FakeND2FileVol(arr, sizes)
    monkeypatch.setitem(sys.modules, "nd2", mod)


def test_read_volume_stack_nd2_keeps_z_axis(tmp_path, monkeypatch):
    arr = np.zeros((4, 2, 16, 24), dtype=np.uint16)   # Z,C,Y,X
    for z in range(4):
        arr[z, 1] = z + 1
    _install_fake_nd2_vol(monkeypatch, arr, {"Z": 4, "C": 2, "Y": 16, "X": 24})
    p = tmp_path / "stack.nd2"
    p.write_bytes(b"")
    vol = ch.read_volume_stack(p)
    assert vol.n_planes == 4
    assert vol.n_channels == 2
    for z in range(4):
        assert vol.plane(z).channel(1).max() == z + 1


def test_has_z_stack_true_for_nd2_zstack(tmp_path, monkeypatch):
    _install_fake_nd2_vol(monkeypatch, np.zeros((4, 2, 16, 24), dtype=np.uint16),
                          {"Z": 4, "C": 2, "Y": 16, "X": 24})
    p = tmp_path / "stack.nd2"
    p.write_bytes(b"")
    assert ch.has_z_stack(p) is True


def test_has_z_stack_false_for_nd2_single_plane(tmp_path, monkeypatch):
    _install_fake_nd2_vol(monkeypatch, np.zeros((3, 16, 24), dtype=np.uint16),
                          {"C": 3, "Y": 16, "X": 24})
    p = tmp_path / "flat.nd2"
    p.write_bytes(b"")
    assert ch.has_z_stack(p) is False


class _FakeCziFile:
    def __init__(self, arr, axes):
        self._arr, self.axes, self.shape = arr, axes, arr.shape

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def asarray(self):
        return self._arr

    def metadata(self):
        return "<Metadata/>"


def _install_fake_czi(monkeypatch, arr, axes):
    mod = types.ModuleType("czifile")
    mod.CziFile = lambda path: _FakeCziFile(arr, axes)
    monkeypatch.setitem(sys.modules, "czifile", mod)


def test_read_volume_stack_czi_keeps_z_axis(tmp_path, monkeypatch):
    arr = np.zeros((3, 2, 20, 30), dtype=np.uint16)   # Z,C,Y,X
    for z in range(3):
        arr[z, 0] = z + 5
    _install_fake_czi(monkeypatch, arr, "ZCYX")
    p = tmp_path / "stack.czi"
    p.write_bytes(b"")
    vol = ch.read_volume_stack(p)
    assert vol.n_planes == 3
    assert vol.n_channels == 2
    for z in range(3):
        assert vol.plane(z).channel(0).max() == z + 5


def test_read_volume_stack_czi_drops_singleton_acquisition_axes(tmp_path, monkeypatch):
    # czifile pads with acquisition axes (S here); a length-1 S must be
    # dropped while the genuine multi-plane Z survives, in both the channel
    # and volume paths.
    arr = np.zeros((1, 3, 2, 20, 30), dtype=np.uint16)  # S,Z,C,Y,X
    _install_fake_czi(monkeypatch, arr, "SZCYX")
    p = tmp_path / "padded.czi"
    p.write_bytes(b"")
    vol = ch.read_volume_stack(p)
    assert vol.n_planes == 3
    assert vol.n_channels == 2


def test_has_z_stack_true_for_czi_zstack(tmp_path, monkeypatch):
    _install_fake_czi(monkeypatch, np.zeros((3, 2, 20, 30), dtype=np.uint16), "ZCYX")
    p = tmp_path / "stack.czi"
    p.write_bytes(b"")
    assert ch.has_z_stack(p) is True


def test_has_z_stack_false_for_czi_single_plane(tmp_path, monkeypatch):
    _install_fake_czi(monkeypatch, np.zeros((2, 20, 30), dtype=np.uint16), "CYX")
    p = tmp_path / "flat.czi"
    p.write_bytes(b"")
    assert ch.has_z_stack(p) is False


class _FakeLifImage:
    """``channel_frames`` back get_iter_c() (channel-only fallback path);
    ``z_frames[(z, c)]`` back get_frame(z=, c=) (the z-stack path)."""

    def __init__(self, channel_frames, z_frames=None, dims_z=1, n_channels=None, scale_x=2.0):
        self._channel_frames = channel_frames
        self._z_frames = z_frames or {}
        self.dims = types.SimpleNamespace(z=dims_z)
        # n_channels defaults from channel_frames but is independently
        # settable, since a z-stack-only fake may not populate channel_frames
        # (get_iter_c() is only ever called on the no-z fallback path).
        self.channels = n_channels if n_channels is not None else len(channel_frames)
        self.scale = (scale_x, scale_x, 1.0)

    def get_iter_c(self):
        return iter(self._channel_frames)

    def get_frame(self, z=0, c=0):
        return self._z_frames[(z, c)]


def _install_fake_readlif(monkeypatch, img):
    # __import__("readlif.reader") returns the *top* package ("readlif"), so
    # LifFile must live there, not on the "readlif.reader" submodule object —
    # matches velum_core.channels._require's exact import mechanism.
    parent = types.ModuleType("readlif")
    parent.LifFile = lambda path: types.SimpleNamespace(get_image=lambda idx: img)
    submod = types.ModuleType("readlif.reader")
    monkeypatch.setitem(sys.modules, "readlif", parent)
    monkeypatch.setitem(sys.modules, "readlif.reader", submod)


def test_read_volume_stack_lif_keeps_z_axis(tmp_path, monkeypatch):
    z_frames = {(z, c): np.full((16, 20), z * 10 + c, dtype=np.uint8)
               for z in range(3) for c in range(2)}
    img = _FakeLifImage(channel_frames=[], z_frames=z_frames, dims_z=3, n_channels=2)
    _install_fake_readlif(monkeypatch, img)
    p = tmp_path / "stack.lif"
    p.write_bytes(b"")
    vol = ch.read_volume_stack(p)
    assert vol.n_planes == 3
    assert vol.n_channels == 2
    for z in range(3):
        for c in range(2):
            assert vol.plane(z).channel(c).max() == z * 10 + c


def test_read_volume_stack_lif_falls_back_when_no_z(tmp_path, monkeypatch):
    channel_frames = [np.full((16, 20), c, dtype=np.uint8) for c in range(3)]
    img = _FakeLifImage(channel_frames=channel_frames, dims_z=1)
    _install_fake_readlif(monkeypatch, img)
    p = tmp_path / "flat.lif"
    p.write_bytes(b"")
    vol = ch.read_volume_stack(p)
    assert vol.n_planes == 1
    assert vol.n_channels == 3


def test_read_volume_stack_lif_falls_back_when_dims_unavailable(tmp_path, monkeypatch):
    """A readlif version whose LifImage doesn't expose the attributes this
    module guesses at (dims.z / channels / get_frame) must degrade to the
    channel-only path instead of raising."""
    channel_frames = [np.full((16, 20), c, dtype=np.uint8) for c in range(2)]

    class _MinimalLifImage:
        def get_iter_c(self):
            return iter(channel_frames)
        # deliberately no .dims / .channels / .get_frame / .scale

    _install_fake_readlif(monkeypatch, _MinimalLifImage())
    p = tmp_path / "minimal.lif"
    p.write_bytes(b"")
    vol = ch.read_volume_stack(p)
    assert vol.n_planes == 1
    assert vol.n_channels == 2


def test_has_z_stack_true_for_lif_zstack(tmp_path, monkeypatch):
    img = _FakeLifImage(channel_frames=[np.zeros((8, 8), dtype=np.uint8)], dims_z=5)
    _install_fake_readlif(monkeypatch, img)
    p = tmp_path / "stack.lif"
    p.write_bytes(b"")
    assert ch.has_z_stack(p) is True


def test_has_z_stack_false_for_lif_single_plane(tmp_path, monkeypatch):
    img = _FakeLifImage(channel_frames=[np.zeros((8, 8), dtype=np.uint8)], dims_z=1)
    _install_fake_readlif(monkeypatch, img)
    p = tmp_path / "flat.lif"
    p.write_bytes(b"")
    assert ch.has_z_stack(p) is False
