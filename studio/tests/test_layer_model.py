"""Tests for the Studio Segment workspace's own layer model (studio/layer_model.py).

Pure-logic, no Qt/torch/napari — numpy/scipy/skimage only (all in the light CI
`test` group already).
"""
import numpy as np
import pytest

from studio.layer_model import (
    ERASE,
    FILL,
    ImageLayer,
    LabelsLayer,
    LayerList,
    PAINT,
    PICK,
    PointsLayer,
    ShapesLayer,
    label_color,
)


# ── label_color ──────────────────────────────────────────────────────────────
def test_label_color_background_is_black():
    assert label_color(0) == (0, 0, 0)
    assert label_color(-3) == (0, 0, 0)


def test_label_color_deterministic_and_distinct():
    assert label_color(5) == label_color(5)
    assert label_color(5) != label_color(6)


def test_label_color_seed_shuffles_without_changing_bg():
    assert label_color(5, seed=0.0) != label_color(5, seed=0.37)
    assert label_color(0, seed=0.37) == (0, 0, 0)


# ── ImageLayer ───────────────────────────────────────────────────────────────
def test_image_layer_default_contrast_from_data():
    data = np.array([[10, 20], [30, 200]], dtype=np.uint8)
    layer = ImageLayer("DAPI", data)
    assert layer.contrast_limits == (10.0, 200.0)
    assert layer.to_summary() == ("DAPI", "image", True)


def test_image_layer_flat_data_gets_nonzero_range():
    data = np.zeros((4, 4), dtype=np.uint8)
    layer = ImageLayer("blank", data)
    lo, hi = layer.contrast_limits
    assert hi > lo


# ── LabelsLayer defaults (must match napari.layers.Labels exactly) ───────────
def test_labels_layer_defaults_match_napari():
    layer = LabelsLayer("Segmentation", np.zeros((10, 10), dtype=np.int32))
    assert layer.opacity == 0.7           # matches napari's own Labels default
    assert layer.blending == "translucent"
    assert layer.brush_size == 10
    assert layer.contiguous is True
    assert layer.preserve_labels is False
    assert layer.show_selected_label is False
    assert layer.n_edit_dimensions == 2
    assert layer.selected_label == 1
    assert layer.mode == "pan_zoom"


def test_labels_layer_defaults_match_classic_app_fill_plus_outline():
    """Real napari's own contour is fill-XOR-outline (0 or an int, never
    both); the classic app's predict_widget.py works around that by adding
    the same mask as two stacked layers (fill_opacity=0.35, outline
    opacity=0.7, outline.contour=1) to get "fill + border" in one look.
    Our own canvas renders both from a single LabelsLayer instead — these
    are the values that reproduce that exact look by default."""
    layer = LabelsLayer("Segmentation", np.zeros((10, 10), dtype=np.int32))
    assert layer.contour == 2  # thicker than velum_core's 1px — see the field's own comment
    assert layer.fill_opacity == 0.35


def test_labels_layer_paint_stamps_a_circle():
    layer = LabelsLayer("L", np.zeros((20, 20), dtype=np.int32))
    layer.selected_label = 3
    layer.brush_size = 6
    layer.paint(10, 10)
    assert layer.data[10, 10] == 3
    assert layer.data[0, 0] == 0
    assert layer.max_label == 3


def test_labels_layer_n_labels_counts_distinct_not_max_id():
    """n_labels is the real "cells detected" -- distinct non-zero instances --
    and must stay correct when ids are non-contiguous (a cell erased out of the
    middle of the range), unlike max_label which tracks the highest id."""
    data = np.zeros((10, 10), dtype=np.int32)
    data[0:2, 0:2] = 1
    data[0:2, 4:6] = 2
    data[4:6, 0:2] = 5      # gap: ids present are {1, 2, 5}
    layer = LabelsLayer("L", data)
    assert layer.max_label == 5
    assert layer.n_labels == 3
    empty = LabelsLayer("E", np.zeros((4, 4), dtype=np.int32))
    assert empty.n_labels == 0


def test_labels_layer_erase_paints_background():
    layer = LabelsLayer("L", np.full((20, 20), 5, dtype=np.int32))
    layer.brush_size = 4
    layer.erase(10, 10)
    assert layer.data[10, 10] == 0


def test_labels_layer_preserve_labels_skips_existing_nonzero():
    data = np.zeros((20, 20), dtype=np.int32)
    data[8:12, 8:12] = 7   # an existing cell
    layer = LabelsLayer("L", data)
    layer.selected_label = 9
    layer.preserve_labels = True
    layer.brush_size = 20  # radius 10: covers the existing cell + surrounding bg
    layer.paint(10, 10)
    assert (layer.data[8:12, 8:12] == 7).all()   # untouched
    assert layer.data[10, 1] == 9                # background within the brush got painted


def test_labels_layer_fill_contiguous_vs_global():
    # two disconnected blobs of the same label
    data = np.zeros((20, 20), dtype=np.int32)
    data[1:4, 1:4] = 2
    data[15:18, 15:18] = 2
    contiguous = LabelsLayer("L", data.copy())
    contiguous.selected_label = 9
    contiguous.fill(2, 2)  # inside the first blob
    assert (contiguous.data[1:4, 1:4] == 9).all()
    assert (contiguous.data[15:18, 15:18] == 2).all()  # untouched

    global_fill = LabelsLayer("L", data.copy())
    global_fill.selected_label = 9
    global_fill.contiguous = False
    global_fill.fill(2, 2)
    assert (global_fill.data[1:4, 1:4] == 9).all()
    assert (global_fill.data[15:18, 15:18] == 9).all()  # also replaced


def test_labels_layer_fill_out_of_bounds_is_a_noop():
    layer = LabelsLayer("L", np.zeros((5, 5), dtype=np.int32))
    layer.fill(-1, -1)  # should not raise
    assert layer.data.sum() == 0


def test_labels_layer_pick_sets_selected_label():
    data = np.zeros((10, 10), dtype=np.int32)
    data[5, 5] = 42
    layer = LabelsLayer("L", data)
    result = layer.pick(5, 5)
    assert result == 42
    assert layer.selected_label == 42


def test_labels_layer_pick_out_of_bounds_returns_none_and_keeps_state():
    layer = LabelsLayer("L", np.zeros((10, 10), dtype=np.int32))
    assert layer.pick(-5, -5) is None
    assert layer.selected_label == 1


def test_labels_layer_polygon_fill_rasterises_region():
    layer = LabelsLayer("L", np.zeros((20, 20), dtype=np.int32))
    layer.selected_label = 4
    layer.polygon_fill([(2, 2), (2, 10), (10, 10), (10, 2)])
    assert layer.data[6, 6] == 4
    assert layer.data[0, 0] == 0


def test_labels_layer_polygon_fill_needs_at_least_three_vertices():
    layer = LabelsLayer("L", np.zeros((10, 10), dtype=np.int32))
    layer.polygon_fill([(1, 1), (5, 5)])
    assert layer.data.sum() == 0


def test_labels_layer_n_edit_dimensions_2_touches_only_current_plane():
    vol = np.zeros((3, 20, 20), dtype=np.int32)
    layer = LabelsLayer("L", vol)
    layer.selected_label = 1
    layer.brush_size = 6
    layer.paint(10, 10, z=1)
    assert layer.data[1, 10, 10] == 1
    assert layer.data[0, 10, 10] == 0
    assert layer.data[2, 10, 10] == 0


def test_labels_layer_n_edit_dimensions_3_touches_every_plane():
    vol = np.zeros((3, 20, 20), dtype=np.int32)
    layer = LabelsLayer("L", vol)
    layer.n_edit_dimensions = 3
    layer.brush_size = 6
    layer.paint(10, 10, z=1)
    assert (layer.data[:, 10, 10] == 1).all()


def test_labels_layer_get_color_background_and_overrides():
    layer = LabelsLayer("L", np.zeros((5, 5), dtype=np.int32))
    assert layer.get_color(0) == (0, 0, 0)
    before = layer.get_color(3)
    layer.set_color_overrides({3: (1.0, 0.0, 0.0)})
    assert layer.get_color(3) == (255, 0, 0)
    assert layer.get_color(4) == label_color(4, layer.color_seed)
    layer.clear_color_overrides()
    assert layer.get_color(3) == before


def test_labels_layer_shuffle_colors_changes_seed_and_clears_overrides():
    layer = LabelsLayer("L", np.zeros((5, 5), dtype=np.int32))
    layer.set_color_overrides({1: (0, 1, 0)})
    seed_before = layer.color_seed
    layer.shuffle_colors()
    assert layer.color_seed != seed_before
    assert layer.color_overrides == {}


def test_labels_layer_set_uniform_color_covers_every_present_id():
    data = np.zeros((10, 10), dtype=np.int32)
    data[0, 0] = 3
    data[5, 5] = 9
    layer = LabelsLayer("Ground truth", data)
    layer.set_uniform_color((0, 255, 90))
    assert layer.get_color(3) == (0, 255, 90)
    assert layer.get_color(9) == (0, 255, 90)
    assert layer.get_color(0) == (0, 0, 0)  # background is never overridden


def test_labels_layer_to_summary_reports_max_label():
    data = np.zeros((5, 5), dtype=np.int32)
    data[0, 0] = 7
    layer = LabelsLayer("Segmentation", data)
    assert layer.to_summary() == ("Segmentation", "7", True)


# ── PointsLayer / ShapesLayer ─────────────────────────────────────────────────
def test_points_layer_add_remove_nearest():
    layer = PointsLayer("Prompts")
    i0 = layer.add(10, 10)
    i1 = layer.add(50, 50)
    assert layer.to_summary() == ("Prompts", "2", True)
    assert layer.nearest(11, 11, max_dist=5) == i0
    assert layer.nearest(200, 200, max_dist=5) is None
    layer.remove_at(i0)
    assert len(layer.points) == 1
    assert layer.points[0] == (50, 50)


def test_shapes_layer_add_remove():
    layer = ShapesLayer("Corrections")
    idx = layer.add("rectangle", [(0, 0), (0, 5), (5, 5), (5, 0)])
    assert layer.to_summary() == ("Corrections", "1", True)
    layer.remove_at(idx)
    assert len(layer.shapes) == 0


# ── LayerList ────────────────────────────────────────────────────────────────
@pytest.fixture
def events():
    calls = []
    return calls


def test_layer_list_add_selects_and_notifies(events):
    layers = LayerList()
    layers.on_change(lambda: events.append(1))
    idx = layers.add(ImageLayer("DAPI", np.zeros((4, 4), dtype=np.uint8)))
    assert idx == 0
    assert layers.selected_index == 0
    assert layers.selected.name == "DAPI"
    assert len(events) == 1


def test_layer_list_remove_updates_selection():
    layers = LayerList()
    layers.add(ImageLayer("a", np.zeros((2, 2))))
    layers.add(ImageLayer("b", np.zeros((2, 2))))
    layers.add(ImageLayer("c", np.zeros((2, 2))))
    layers.select(2)
    layers.remove(0)  # removes "a"; selected index 2 -> 1, still "c"
    assert layers.selected.name == "c"
    assert len(layers) == 2

    layers.select(0)
    layers.remove(0)  # removes the currently-selected layer
    assert layers.selected_index is None


def test_layer_list_toggle_visible():
    layers = LayerList()
    layers.add(ImageLayer("a", np.zeros((2, 2))))
    assert layers[0].visible is True
    layers.toggle_visible(0)
    assert layers[0].visible is False


def test_layer_list_move_reorders_and_tracks_selection():
    layers = LayerList()
    layers.add(ImageLayer("a", np.zeros((2, 2))))
    layers.add(ImageLayer("b", np.zeros((2, 2))))
    layers.select(0)
    layers.move(0, 1)
    assert [l.name for l in layers] == ["b", "a"]
    assert layers.selected.name == "a"


def test_layer_list_by_kind_and_find():
    layers = LayerList()
    layers.add(ImageLayer("DAPI", np.zeros((2, 2))))
    layers.add(LabelsLayer("Segmentation", np.zeros((2, 2), dtype=np.int32)))
    assert [l.name for l in layers.by_kind("labels")] == ["Segmentation"]
    assert layers.find("DAPI").kind == "image"
    assert layers.find("nope") is None


def test_layer_list_unique_name():
    layers = LayerList()
    layers.add(ShapesLayer("Shapes"))
    assert layers.unique_name("Shapes") == "Shapes [2]"
    layers.add(ShapesLayer("Shapes [2]"))
    assert layers.unique_name("Shapes") == "Shapes [3]"
    assert layers.unique_name("Points") == "Points"


def test_layer_list_n_planes_and_current_z():
    layers = LayerList()
    layers.add(ImageLayer("vol", np.zeros((5, 8, 8, 3), dtype=np.uint8)))
    assert layers.n_planes == 5
    layers.set_current_z(3)
    assert layers.current_z == 3
    layers.set_current_z(99)
    assert layers.current_z == 4  # clamped to n_planes - 1


def test_layer_list_clear_resets_everything():
    layers = LayerList()
    layers.add(ImageLayer("a", np.zeros((2, 2))))
    layers.set_current_z(0)
    layers.clear()
    assert len(layers) == 0
    assert layers.selected_index is None
    assert layers.current_z == 0


# ── LabelsLayer undo / redo history ───────────────────────────────────────────
def test_labels_undo_restores_pre_edit_data():
    layer = LabelsLayer("L", np.zeros((20, 20), dtype=np.int32))
    assert not layer.can_undo and not layer.can_redo
    layer.begin_edit()
    layer.paint(10, 10, label=3)
    assert layer.data.max() == 3
    assert layer.can_undo
    assert layer.undo() is True
    assert layer.data.max() == 0  # back to the pre-paint snapshot
    assert layer.can_redo


def test_labels_redo_reapplies_undone_edit():
    layer = LabelsLayer("L", np.zeros((20, 20), dtype=np.int32))
    layer.begin_edit()
    layer.paint(10, 10, label=7)
    layer.undo()
    assert layer.data.max() == 0
    assert layer.redo() is True
    assert layer.data.max() == 7
    assert not layer.can_redo


def test_labels_undo_redo_are_noops_when_empty():
    layer = LabelsLayer("L", np.zeros((5, 5), dtype=np.int32))
    assert layer.undo() is False
    assert layer.redo() is False


def test_labels_new_edit_clears_the_redo_branch():
    layer = LabelsLayer("L", np.zeros((10, 10), dtype=np.int32))
    layer.begin_edit(); layer.paint(2, 2, label=1)
    layer.undo()
    assert layer.can_redo
    layer.begin_edit(); layer.paint(8, 8, label=2)  # a fresh edit
    assert not layer.can_redo  # redo branch invalidated, like every editor


def test_labels_undo_history_is_bounded():
    layer = LabelsLayer("L", np.zeros((6, 6), dtype=np.int32))
    for i in range(layer._history_limit + 10):
        layer.begin_edit()
        layer.paint(3, 3, label=i + 1)
    assert len(layer._undo_stack) == layer._history_limit  # capped, not unbounded


def test_labels_multi_dab_stroke_is_one_undo_step():
    """A paint drag calls paint() many times but begin_edit() once at stroke
    start -- so a single undo reverts the whole stroke, not one dab."""
    layer = LabelsLayer("L", np.zeros((30, 30), dtype=np.int32))
    layer.begin_edit()
    for c in range(5, 25):
        layer.paint(15, c, label=4)  # many dabs, no further begin_edit
    assert layer.data.max() == 4
    layer.undo()
    assert layer.data.max() == 0  # the entire stroke gone in one undo
