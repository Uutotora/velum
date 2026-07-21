"""Per-channel intensity columns in analysis.compute_measurements (opt-in).

Pure-logic; no torch/napari. Verifies the multi-channel path adds one mean
column per channel and that the default (single-image) behaviour is unchanged.
"""
import numpy as np

from velum_core import analysis


def _two_cell_mask():
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[:4, :4] = 1
    mask[4:, 4:] = 2
    return mask


def test_default_path_has_no_channel_columns():
    mask = _two_cell_mask()
    res = analysis.compute_measurements(mask)
    keys = [k for k, _l, _u in res["columns"]]
    assert not any(k.startswith("ch") and k.endswith("_mean") for k in keys)


def test_channel_columns_added_and_labelled():
    mask = _two_cell_mask()
    stack = np.stack([np.full((8, 8), 10.0), np.full((8, 8), 200.0)], axis=-1)
    res = analysis.compute_measurements(
        mask, channel_intensities=stack, channel_names=["DAPI", "GFP"])
    cols = {k: label for k, label, _u in res["columns"]}
    assert cols.get("ch0_mean") == "DAPI mean"
    assert cols.get("ch1_mean") == "GFP mean"

    idx0 = [k for k, _l, _u in res["columns"]].index("ch0_mean")
    idx1 = [k for k, _l, _u in res["columns"]].index("ch1_mean")
    # Every cell sees channel 0 ≈ 10 and channel 1 ≈ 200 (constant channels).
    for row in res["rows"]:
        assert abs(row[idx0] - 10.0) < 1e-6
        assert abs(row[idx1] - 200.0) < 1e-6


def test_channel_means_reflect_spatial_variation():
    mask = _two_cell_mask()
    chan = np.zeros((8, 8), dtype=np.float32)
    chan[:4, :4] = 30.0     # under cell 1
    chan[4:, 4:] = 70.0     # under cell 2
    res = analysis.compute_measurements(
        mask, channel_intensities=chan[..., None], channel_names=["marker"])
    idx = [k for k, _l, _u in res["columns"]].index("ch0_mean")
    by_cell = {row[0]: row[idx] for row in res["rows"]}
    assert by_cell[1] == 30.0
    assert by_cell[2] == 70.0


def test_channel_summary_present():
    mask = _two_cell_mask()
    stack = np.stack([np.full((8, 8), 5.0)], axis=-1)
    res = analysis.compute_measurements(mask, channel_intensities=stack)
    assert "ch0_mean" in res["summary"]
    assert res["summary"]["ch0_mean"]["mean"] == 5.0


def test_misaligned_channel_stack_ignored():
    mask = _two_cell_mask()
    bad = np.zeros((3, 3, 2), dtype=np.float32)   # wrong H,W
    res = analysis.compute_measurements(mask, channel_intensities=bad)
    keys = [k for k, _l, _u in res["columns"]]
    assert not any(k.startswith("ch") for k in keys)


def test_channels_combine_with_grayscale_intensity():
    mask = _two_cell_mask()
    gray = np.full((8, 8), 100, dtype=np.uint8)
    stack = np.stack([np.full((8, 8), 42.0)], axis=-1)
    res = analysis.compute_measurements(
        mask, intensity_image=gray, channel_intensities=stack)
    keys = [k for k, _l, _u in res["columns"]]
    assert "mean_intensity" in keys      # grayscale block still present
    assert "ch0_mean" in keys            # plus per-channel block
