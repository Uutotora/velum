"""Pure-logic tests for multi-channel microscopy support (napari_app.channels).

No torch/napari/GPU — runs in the lightweight CI job. Multi-page and OME-TIFF
reading is exercised through tifffile round-trips on synthetic stacks.
"""
import numpy as np
import pytest

from napari_app import channels as ch


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
