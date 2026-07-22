"""Tests for the Segment workspace screen, wired for real (studio/workspace.py).

Offscreen Qt. Every predict/batch/benchmark test monkeypatches
velum_core.inference_cache.predict_cached (same seam as
test_segment_controller.py) so the real UI -> controller -> engine chain
runs end to end without a GPU/SAM weights/torch actually loading anything.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")
workspace = pytest.importorskip("studio.workspace")

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QFileDialog, QFrame, QLabel

from studio import demo, theme
from studio.layer_model import ImageLayer, LabelsLayer, PAINT, PAN_ZOOM, PointsLayer, ShapesLayer
from studio.project import Project, ProjectSettings, ProjectStore
from studio.project_controller import ProjectController
from studio.segment_controller import SegmentController
from studio.workspace import WorkspaceScreen


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _write_image(path: Path, size=48, seed=0) -> None:
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), (rng.random((size, size, 3)) * 255).astype(np.uint8))


def _write_mask(path: Path, *labels_and_boxes, size=48) -> None:
    mask = np.zeros((size, size), dtype=np.uint16)
    for lbl, (y0, y1, x0, x1) in labels_and_boxes:
        mask[y0:y1, x0:x1] = lbl
    cv2.imwrite(str(path), mask)


def _fake_predict_cached(config, image_rgb):
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[h // 8: h // 2, w // 8: w // 2] = 1
    mask[h // 2: h - h // 8, w // 2: w - w // 8] = 2
    return mask


@pytest.fixture(autouse=True)
def _fake_engine(monkeypatch):
    monkeypatch.setattr("velum_core.inference_cache.predict_cached", _fake_predict_cached)
    # cellpose may be genuinely installed in whatever env runs these tests —
    # without this, the benchmark test sees it as available and actually
    # runs real Cellpose inference instead of treating it as absent (see the
    # identical note in test_segment_controller.py's own _fake_engine).
    monkeypatch.setattr("velum_core.engines.cellpose_available", lambda: False)


@pytest.fixture
def storage(tmp_path):
    d = tmp_path / "storage"
    (d / "sam_backbone").mkdir(parents=True)
    (d / "sam_backbone" / "sam_vit_h_4b8939.pth").write_bytes(b"fake")
    (d / "loras").mkdir(parents=True)
    (d / "loras" / "nuclei-dapi-r8.pth").write_bytes(b"fake-lora")
    return d


@pytest.fixture
def segment(storage):
    return SegmentController(storage_dir=storage)


@pytest.fixture
def projects(tmp_path):
    return ProjectController(ProjectStore(tmp_path / "projects"), seed_if_empty=False)


@pytest.fixture
def toasts():
    return []


def _ws(app, segment, projects, toasts, on_toggle_logs=None):
    return WorkspaceScreen(theme.DARK, segment, projects,
                           lambda title, sub: toasts.append((title, sub)), on_toggle_logs)


def _make_project(tmp_path, projects, storage, *, n_images=1, with_gt=False, size=48):
    imgs = []
    for i in range(n_images):
        p = tmp_path / f"img_{i}.png"
        _write_image(p, size=size, seed=i)
        imgs.append(str(p))
    if with_gt:
        _write_mask(tmp_path / "img_0_mask.png", (1, (2, 10, 2, 10)), (2, (20, 30, 20, 30)), size=size)
    lora = storage / "loras" / "nuclei-dapi-r8.pth"
    project = projects.store.create(
        "Test Project", "x", image_paths=imgs,
        settings=ProjectSettings(engine="cellseg1", model_name=str(lora)))
    return project


def _pump(app, ws, timeout=10):
    t0 = time.monotonic()
    while (ws._predicting or ws._batching or ws._benching) and time.monotonic() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(5):
        app.processEvents()


# ── construction / empty state ────────────────────────────────────────────────
def test_construct_with_no_project_shows_empty_states(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    assert ws._project is None
    assert ws._canvas._base_shape() is None


def test_refresh_is_a_noop_when_no_active_project(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    ws.refresh()  # should not raise
    assert ws._project is None


def test_construct_with_no_project_shows_the_no_project_view_not_the_three_panel_body(
        app, segment, projects, toasts):
    """Regression test for "empty canvas looks sad" -- and, more pointedly,
    for the first fix at this being wrong: landing on Segment with no
    project used to show the full three-panel IDE layout (Images/Layers ·
    canvas + its floating tool strip/viewer bar · Segment/Results) with
    every panel empty, plus (round one) just a friendly message layered
    into the canvas's corner on top of all that -- still broken chrome
    everywhere else, reported directly against a screenshot ("канвас
    боковые панели... убрать все" -- remove the canvas and side panels, all
    of it). The three-panel body and the no-project view are now two full
    alternatives in one QStackedWidget; with no project, the three-panel
    body (index 0) must not be the visible page at all."""
    ws = _ws(app, segment, projects, toasts)
    assert ws._project is None
    assert ws._body_stack.currentIndex() == 1


def test_loading_a_project_switches_to_the_three_panel_body(app, segment, projects, toasts, tmp_path, storage):
    ws = _ws(app, segment, projects, toasts)
    assert ws._body_stack.currentIndex() == 1
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws._load_project(project)
    assert ws._body_stack.currentIndex() == 0


def test_clearing_the_active_project_switches_back_to_the_no_project_view(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert ws._body_stack.currentIndex() == 0
    ws._load_project(None)
    assert ws._body_stack.currentIndex() == 1


def test_no_project_view_open_a_project_button_navigates_to_projects(app, segment, projects, toasts):
    seen = []
    ws = WorkspaceScreen(theme.DARK, segment, projects,
                         lambda title, sub: toasts.append((title, sub)),
                         on_navigate=lambda key: seen.append(key))
    ws._no_project_open_btn.click()
    assert seen == ["projects"]


def test_no_project_view_new_project_button_calls_on_new_project(app, segment, projects, toasts):
    seen = []
    ws = WorkspaceScreen(theme.DARK, segment, projects,
                         lambda title, sub: toasts.append((title, sub)),
                         on_new_project=lambda: seen.append(True))
    ws._no_project_new_btn.click()
    assert seen == [True]


def test_topbar_hidden_without_a_project_shown_once_one_is_open(
        app, segment, projects, toasts, tmp_path, storage):
    """Regression test, round 2: the first fix merely disabled Export/Run
    inside the topbar while leaving the whole bar (breadcrumb saying "No
    project selected", an engine chip saying "No project") visible with no
    project open -- direct feedback ("почему верхняя панель осталась? если
    проект не выбран то она не нужна" -- why did the top panel stay? if no
    project is selected it isn't needed) said the greyed-out buttons weren't
    enough, the whole bar is redundant once the no-project view has its own
    "Open a Project" action. The bar must be hidden outright, not just
    disabled inside; _start_predict/_export_csv's own internal
    precondition guards (a toast, not a crash -- see app.py's identical
    command-palette comment) are what actually makes hiding safe rather
    than merely disabling."""
    ws = _ws(app, segment, projects, toasts)
    assert ws._topbar_widget.isHidden()
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws._load_project(project)
    assert not ws._topbar_widget.isHidden()


def test_topbar_engine_badge_is_a_centred_pill_reflecting_the_engine(
        app, segment, projects, toasts, tmp_path, storage):
    """The engine now reads as a standalone rounded badge (EngineChip: pill +
    engine-hued dot) centred in the topbar, not a square chip beside the name.
    Rebuilt in place on each project switch, in its own centre holder."""
    from studio.components import EngineChip
    project = _make_project(tmp_path, projects, storage, n_images=1)  # cellseg1
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert isinstance(ws._engine_badge, EngineChip)
    # It lives in the dedicated centre holder, flanked by stretches (so it's
    # centred), not inline with the breadcrumb.
    assert ws._engine_badge_layout.indexOf(ws._engine_badge) != -1
    labels = [lb.text() for lb in ws._engine_badge.findChildren(QLabel)]
    assert any("CellSeg1" in txt for txt in labels)


def test_topbar_breadcrumb_shows_only_the_project_name(app, segment, projects, toasts, tmp_path, storage):
    """The name label carries just the name now -- the "/" separator is its own
    label between the Projects link and the name, not baked into the name's
    rich text (which is what let the engine move out to the centre)."""
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert ws._crumb_name.text() == project.name
    assert ws._crumb_sep.text() == "/"


def test_topbar_projects_crumb_navigates_back(app, segment, projects, toasts, tmp_path, storage):
    nav = []
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = WorkspaceScreen(theme.DARK, segment, projects,
                         lambda a, b: toasts.append((a, b)), on_navigate=nav.append)
    ws._load_project(project)
    ws._crumb_projects.mouseReleaseEvent(None)
    assert nav == ["projects"]


# ── project / image loading ──────────────────────────────────────────────────
def test_load_project_selects_first_image_and_builds_layers(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=2)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert ws._current_image_path == project.image_paths[0]
    kinds = [l.kind for l in ws._layers]
    assert kinds == ["image", "labels"]
    assert ws._layers.selected.kind == "labels"  # Segmentation layer selected by default


def test_load_project_auto_discovers_ground_truth(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1, with_gt=True)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    gt = ws._layers.find("Ground truth")
    assert gt is not None
    assert gt.visible is False  # auto-discovered GT starts hidden


def test_ground_truth_layers_get_the_fixed_gt_colour_not_random_hues(
        app, segment, projects, toasts, tmp_path, storage):
    """Matches the classic app's own GT convention: a fixed colour (not
    per-instance random), so GT and predictions read as distinct roles."""
    from studio.workspace import _GT_COLOR

    project = _make_project(tmp_path, projects, storage, n_images=1, with_gt=True)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    gt = ws._layers.find("Ground truth")
    assert gt.opacity == 0.9
    ids = [i for i in set(gt.data.ravel().tolist()) if i > 0]
    assert ids  # the fixture's GT mask actually has labelled cells
    assert all(gt.get_color(i) == _GT_COLOR for i in ids)


def test_refresh_reloads_only_on_project_switch(app, segment, projects, toasts, tmp_path, storage):
    p1 = _make_project(tmp_path, projects, storage, n_images=1)
    p2_dir = tmp_path / "p2"
    p2_dir.mkdir()
    p2 = _make_project(p2_dir, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)

    projects.set_active(p1.id)
    ws.refresh()
    assert ws._project.id == p1.id
    seg = ws._layers.find("Segmentation")
    seg.data[0, 0] = 77  # simulate in-progress unsaved work

    ws.refresh()  # same project still active -> must NOT reset layers
    assert ws._layers.find("Segmentation").data[0, 0] == 77

    projects.set_active(p2.id)
    ws.refresh()
    assert ws._project.id == p2.id
    assert ws._layers.find("Segmentation").data[0, 0] == 0  # fresh layers for the new project


def test_select_image_switches_layers(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=2)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._select_image(project.image_paths[1])
    assert ws._current_image_path == project.image_paths[1]
    assert ws._last_result is None


# ── adding images to an existing project ──────────────────────────────────────
def test_add_image_paths_copies_into_project_and_persists(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    new_img = tmp_path / "extra.png"
    _write_image(new_img, seed=99)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._add_image_paths([str(new_img)])
    assert len(project.image_paths) == 2
    # The image is copied into the project (survives moves + macOS's Downloads
    # privacy block), so the stored path is the copy, not the original source.
    added = project.image_paths[1]
    assert Path(added).parent == projects.store.image_dir(project.id)
    assert Path(added).name == "extra.png"
    assert Path(added).read_bytes() == new_img.read_bytes()
    reloaded = projects.store.load(project.id)
    assert added in reloaded.image_paths
    assert reloaded.stats.n_images == 2
    assert any("Images added" in t[0] for t in toasts)


def test_add_image_paths_dedupes_and_filters_unsupported(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    existing = project.image_paths[0]
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("hello")
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._add_image_paths([existing, str(not_an_image)])
    assert len(project.image_paths) == 1  # nothing new actually added
    assert any("No new images" in t[0] for t in toasts)


@pytest.mark.parametrize("suffix", [".nd2", ".czi", ".lif"])
def test_add_image_paths_accepts_native_microscopy_formats(
        app, segment, projects, toasts, tmp_path, storage, suffix):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    source = tmp_path / f"field{suffix}"
    source.write_bytes(b"microscopy fixture")
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._add_image_paths([str(source)])
    assert len(project.image_paths) == 2
    assert Path(project.image_paths[-1]).suffix == suffix


def test_add_image_paths_auto_selects_first_image_for_empty_project(
        app, segment, projects, toasts, tmp_path, storage):
    project = projects.store.create(
        "Empty", "", settings=ProjectSettings(
            engine="cellseg1", model_name=str(storage / "loras" / "nuclei-dapi-r8.pth")))
    new_img = tmp_path / "first.png"
    _write_image(new_img)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert ws._current_image_path is None
    ws._add_image_paths([str(new_img)])
    # Auto-selects the freshly-added image -- now the in-project copy, not the
    # original source path (copy-on-import).
    assert ws._current_image_path == project.image_paths[0]
    assert Path(ws._current_image_path).parent == projects.store.image_dir(project.id)


def test_add_images_without_a_project_toasts(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    ws._add_images()
    assert toasts and "No project open" in toasts[-1][0]


# ── image-read robustness: thumbnail cache + actionable error hint ────────────
def test_thumbnail_is_cached_per_path(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    path = project.image_paths[0]
    first = ws._thumbnail(path)
    second = ws._thumbnail(path)
    assert first is second  # second call returns the cached pixmap, no re-decode
    assert path in ws._thumb_cache


def test_thumbnail_caches_the_fallback_for_an_unreadable_file(app, segment, projects, toasts, tmp_path, storage):
    # A missing/unreadable file must be attempted once, not re-decoded on every
    # images-pane rebuild (the "can't open/read file" log storm).
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    gone = str(tmp_path / "vanished.png")
    pm = ws._thumbnail(gone)
    assert gone in ws._thumb_cache
    assert ws._thumbnail(gone) is pm


def test_read_error_hint_flags_a_missing_file(tmp_path):
    hint = WorkspaceScreen._read_error_hint(str(tmp_path / "nope.png"))
    assert hint is not None and "no longer exists" in hint


def test_read_error_hint_is_none_for_a_readable_file(tmp_path):
    p = tmp_path / "ok.png"
    _write_image(p)
    assert WorkspaceScreen._read_error_hint(str(p)) is None


def test_images_drop_adds_only_local_image_files(app, segment, projects, toasts, tmp_path, storage):
    from PyQt6.QtCore import QUrl

    project = _make_project(tmp_path, projects, storage, n_images=1)
    new_img = tmp_path / "dropped.png"
    _write_image(new_img, seed=7)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)

    class _FakeMime:
        def hasUrls(self_inner):
            return True

        def urls(self_inner):
            return [QUrl.fromLocalFile(str(new_img)), QUrl("https://example.com/not-local")]

    class _FakeDropEvent:
        def mimeData(self_inner):
            return _FakeMime()

        def acceptProposedAction(self_inner):
            pass

    ws._images_drop(_FakeDropEvent())
    # The local image is copied into the project (copy-on-import); the non-local
    # URL is ignored. So we gained exactly one image, stored under the project.
    assert len(project.image_paths) == 2
    added = project.image_paths[1]
    assert Path(added).parent == projects.store.image_dir(project.id)
    assert Path(added).name == Path(new_img).name


# ── undo / redo wiring ────────────────────────────────────────────────────────
def test_workspace_undo_redo_revert_and_reapply_a_mask_edit(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    target = ws._canvas.edit_target()
    target.begin_edit()
    target.paint(10, 10, label=5)
    assert target.max_label == 5
    ws._undo()
    assert target.max_label == 0
    ws._redo()
    assert target.max_label == 5


def test_workspace_undo_is_a_safe_noop_with_empty_history(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._undo()  # nothing painted yet -- must not raise
    ws._redo()


def test_canvas_bar_has_undo_and_redo_buttons(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert "undo" in ws._vbar_buttons
    assert "redo" in ws._vbar_buttons


# ── layer drag-reorder ────────────────────────────────────────────────────────
def test_move_layer_reorders_and_keeps_selection(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)  # [image, Segmentation]
    names_before = [l.name for l in ws._layers]
    seg_idx = ws._layers.index_of(ws._layers.find("Segmentation"))
    ws._layers.select(seg_idx)
    ws._move_layer(seg_idx, 0)  # send Segmentation to the bottom of the list
    assert [l.name for l in ws._layers] == list(reversed(names_before))
    assert ws._layers.selected.name == "Segmentation"  # selection follows the move


def test_layer_row_drag_down_reorders_not_selects(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    first_before = list(ws._layers)[0].name
    row = ws._layers_list_layout.itemAt(0).widget()  # topmost row (index 0)
    _dispatch_press(row, pos=(5, 4))
    _dispatch_release(row, pos=(5, 80))  # a clear downward drag
    _pump_events(app)
    # index-0 layer moved down (no longer first); a drag must not be read as a tap
    assert list(ws._layers)[0].name != first_before


# ── image contrast limits ─────────────────────────────────────────────────────
def test_auto_contrast_sets_percentile_limits(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    img = ws._layers.by_kind("image")[0]
    img.contrast_limits = (0.0, 255.0)
    ws._auto_contrast(img)
    lo, hi = img.contrast_limits
    assert hi > lo
    # a 1–99 percentile stretch sits strictly inside the full 0..255 range for
    # a random image (never both exactly 0 and 255)
    assert not (lo == 0.0 and hi == 255.0)


def test_set_contrast_moves_a_limit_without_crossing(app, segment, projects, toasts, tmp_path, storage):
    from studio.components import Badge
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    img = ws._layers.by_kind("image")[0]
    img.contrast_limits = (0.0, 255.0)
    # drive the low limit above the current high -> it must clamp just below hi
    ws._set_contrast(img, "lo", 1.0, 0.0, 255.0, Badge("", theme.DARK))
    lo, hi = img.contrast_limits
    assert lo < hi


# ── Layers-pane empty state ───────────────────────────────────────────────────
def test_layers_pane_shows_empty_state_when_no_image_loaded(app, segment, projects, toasts, tmp_path, storage):
    project = projects.store.create("Empty", settings=ProjectSettings(engine="cellseg1"))
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)  # a project with no images -> no layers
    assert len(list(ws._layers)) == 0
    assert ws._layers_stack.currentIndex() == 1  # the empty-state, not the toolbar+list


def test_layers_pane_shows_content_once_an_image_is_loaded(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)  # auto-selects the first image -> layers exist
    assert len(list(ws._layers)) > 0
    assert ws._layers_stack.currentIndex() == 0


def test_removing_the_last_image_returns_to_the_empty_state(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert ws._layers_stack.currentIndex() == 0
    ws._remove_image(project.image_paths[0])
    assert ws._layers_stack.currentIndex() == 1  # back to empty-state


# ── swipe-to-delete image rows ────────────────────────────────────────────────
def test_remove_image_drops_it_from_the_project_and_persists(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=2)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    victim = project.image_paths[1]  # not the currently-open one
    ws._remove_image(victim)
    assert victim not in project.image_paths
    assert projects.store.load(project.id).image_paths == project.image_paths
    assert projects.store.load(project.id).stats.n_images == 1
    assert any("Image removed" in t[0] for t in toasts)


def test_remove_current_image_switches_to_another(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=2)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    current = ws._current_image_path
    assert current == project.image_paths[0]
    ws._remove_image(current)
    assert ws._current_image_path == project.image_paths[0]  # moved to the survivor
    assert current not in project.image_paths


def test_remove_last_image_clears_the_canvas(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._remove_image(project.image_paths[0])
    assert ws._current_image_path is None
    assert project.image_paths == []
    assert len(list(ws._layers)) == 0


def test_remove_image_deletes_the_in_project_copy_on_disk(app, segment, projects, toasts, tmp_path, storage):
    project = projects.store.create("Copy Del", settings=ProjectSettings(
        engine="cellseg1", model_name=str(storage / "loras" / "nuclei-dapi-r8.pth")))
    src = tmp_path / "orig.png"
    _write_image(src)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._add_image_paths([str(src)])           # copied into the project store
    copied = project.image_paths[0]
    assert Path(copied).exists()
    ws._remove_image(copied)
    assert not Path(copied).exists()          # our copy is cleaned up


# ── resizable / collapsible panels ────────────────────────────────────────────
def test_body_is_a_resizable_splitter_with_three_panes(app, segment, projects, toasts, tmp_path, storage):
    from PyQt6.QtWidgets import QSplitter
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert isinstance(ws._body_splitter, QSplitter)
    assert ws._body_splitter.count() == 3  # left | canvas | inspector
    assert not ws._body_splitter.isCollapsible(1)  # canvas never collapses


def test_left_panel_toggle_hides_and_restores_it(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert not ws._left_panel_w.isHidden()
    ws._toggle_left_panel()
    assert ws._left_panel_w.isHidden()
    ws._toggle_left_panel()
    assert not ws._left_panel_w.isHidden()


def test_inspector_toggle_hides_and_restores_it(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert not ws._inspector_w.isHidden()
    ws._toggle_inspector()
    assert ws._inspector_w.isHidden()
    ws._toggle_inspector()
    assert not ws._inspector_w.isHidden()


# ── sidebar declutter / dedupe ────────────────────────────────────────────────
def test_transpose_button_has_its_own_key_not_shuffle(app, segment, projects, toasts, tmp_path, storage):
    """Transpose used the "shuffle" glyph, which also means "shuffle label
    colours" in the Labels tools -- a duplicated icon. It now has its own
    'transpose' icon/key."""
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert "transpose" in ws._vbar_buttons
    assert "shuffle" not in ws._vbar_buttons


def test_new_label_selects_max_plus_one(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=1)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    layer = ws._layers.find("Segmentation")
    layer.data[5:8, 5:8] = 4  # highest existing id
    ws._new_label(layer)
    assert layer.selected_label == 5  # max(4) + 1


def test_transform_tool_removed_from_labels_mode_icons():
    """The Transform tool did nothing the Pan/zoom tool didn't (the canvas
    treated both identically), so it's gone from the Labels tool row."""
    from studio.workspace import MODE_ICONS
    modes = [m for _icon, _tip, m in MODE_ICONS]
    assert "transform" not in modes
    assert "pan_zoom" in modes


# ── layer panel actions ───────────────────────────────────────────────────────
def test_add_points_and_shapes_layers(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._add_points_layer()
    ws._add_shapes_layer()
    kinds = [l.kind for l in ws._layers]
    assert kinds.count("points") == 1
    assert kinds.count("shapes") == 1
    assert ws._layers.selected.kind == "shapes"  # most-recently-added is selected


def test_add_labels_layer_needs_a_loaded_image(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    ws._add_labels_layer()
    assert not any(l.kind == "labels" for l in ws._layers)
    assert toasts and "No image loaded" in toasts[-1][0]


def test_delete_selected_layer(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    n_before = len(ws._layers)
    ws._delete_selected_layer()
    assert len(ws._layers) == n_before - 1


def test_toggle_layer_visible(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    assert seg.visible is True
    ws._toggle_layer_visible(ws._layers.index_of(seg))
    assert seg.visible is False


def test_select_layer_rebuilds_controls_for_the_right_kind(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    image_layer = ws._layers.find(Path(project.image_paths[0]).stem)
    ws._select_layer(ws._layers.index_of(image_layer))
    assert ws._layers.selected is image_layer


def test_set_canvas_mode_updates_target_layer_mode(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._set_canvas_mode(PAINT)
    assert ws._canvas.mode == PAINT
    assert ws._layers.find("Segmentation").mode == PAINT


def test_shuffle_colors_changes_seed(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    before = seg.color_seed
    ws._shuffle_colors()
    assert seg.color_seed != before


# ── Segment settings pane ─────────────────────────────────────────────────────
def test_engine_select_switches_engine_and_rebuilds_model_field(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    # this file's _fake_engine fixture reports cellpose as unavailable (see
    # its docstring), so its option text carries the "(not installed)" note
    ws._on_engine_select("Cellpose-SAM (zero-shot, generalist)  (not installed)")
    assert project.settings.engine == "cellpose"


def test_quality_preset_updates_thresholds(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._on_quality_preset(2)  # Accurate
    assert project.settings.quality_preset == "Accurate"
    assert project.settings.points_per_side == 48


# ── Command palette integration aliases ─────────────────────────────────────
def test_switch_engine_alias_sets_the_engine_by_key(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws.switch_engine("cellpose")
    assert project.settings.engine == "cellpose"


def test_switch_engine_alias_is_a_noop_without_a_project(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    ws.switch_engine("cellpose")  # must not raise


def test_apply_preset_alias_applies_by_name(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws.apply_preset("Accurate")
    assert project.settings.quality_preset == "Accurate"
    assert project.settings.points_per_side == 48


def test_apply_preset_alias_ignores_an_unknown_name(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    before = project.settings.quality_preset
    ws.apply_preset("Nonsense")
    assert project.settings.quality_preset == before


def test_run_batch_alias_delegates_to_start_batch(app, segment, projects, toasts, monkeypatch):
    ws = _ws(app, segment, projects, toasts)
    calls = []
    monkeypatch.setattr(ws, "_start_batch", lambda: calls.append(True))
    ws.run_batch()
    assert calls == [True]


def test_run_benchmark_alias_delegates_to_start_benchmark(app, segment, projects, toasts, monkeypatch):
    ws = _ws(app, segment, projects, toasts)
    calls = []
    monkeypatch.setattr(ws, "_start_benchmark", lambda: calls.append(True))
    ws.run_benchmark()
    assert calls == [True]


def test_save_masks_alias_delegates_to_save_masks_private(app, segment, projects, toasts, monkeypatch):
    ws = _ws(app, segment, projects, toasts)
    calls = []
    monkeypatch.setattr(ws, "_save_masks", lambda: calls.append(True))
    ws.save_masks()
    assert calls == [True]


def test_export_measurements_alias_delegates_to_export_csv(app, segment, projects, toasts, monkeypatch):
    ws = _ws(app, segment, projects, toasts)
    calls = []
    monkeypatch.setattr(ws, "_export_csv", lambda: calls.append(True))
    ws.export_measurements()
    assert calls == [True]


def test_manual_threshold_change_marks_preset_custom(app, segment, projects, toasts, tmp_path, storage):
    from studio.components import Badge
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    badge = Badge("0.80", theme.DARK)
    ws._on_threshold_change("pred_iou_thresh", 0.91, badge)
    assert project.settings.pred_iou_thresh == pytest.approx(0.91)
    assert project.settings.quality_preset == "Custom"
    assert badge.text() == "0.91"


def test_pixel_size_edit_recomputes_results(app, segment, projects, toasts, tmp_path, storage):
    from PyQt6.QtWidgets import QLineEdit
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._last_result = segment.compute_measurements(np.zeros((10, 10), dtype=np.int32))
    edit = QLineEdit("0.5")
    ws._on_pixel_size_edited(edit)
    assert project.settings.pixel_size_um == pytest.approx(0.5)


# ── predict flow ─────────────────────────────────────────────────────────────
def test_start_predict_without_project_toasts(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    ws._start_predict()
    assert toasts and "No project open" in toasts[-1][0]


def test_start_predict_runs_and_populates_results(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    assert ws._predicting is True
    _pump(app, ws)
    assert ws._predicting is False
    assert ws._last_result is not None
    assert ws._last_result["n_cells"] == 2
    seg = ws._layers.find("Segmentation")
    assert seg.max_label == 2
    reloaded = projects.store.load(project.id)
    assert reloaded.stats.n_cells == 2
    assert any("Segmentation complete" in t[0] for t in toasts)


def test_predict_result_survives_switching_images_and_back(app, segment, projects, toasts, tmp_path, storage):
    """The core "reopen and see your result" requirement, exercised via an
    in-session round trip: run predict on image A, switch to B (a fresh,
    unsegmented image), switch back to A — A's mask and stats must come
    back exactly, not an empty layer."""
    project = _make_project(tmp_path, projects, storage, n_images=2)
    img_a, img_b = project.image_paths
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert ws._current_image_path == img_a
    ws._start_predict()
    _pump(app, ws)
    assert ws._layers.find("Segmentation").max_label == 2

    ws._select_image(img_b)
    assert ws._layers.find("Segmentation").max_label == 0  # a fresh image starts empty
    assert ws._last_result is None

    ws._select_image(img_a)
    assert ws._layers.find("Segmentation").max_label == 2
    assert ws._last_result is not None
    assert ws._last_result["n_cells"] == 2


def test_rebuilding_results_pane_does_not_accumulate_widgets(app, segment, projects, toasts, tmp_path, storage):
    """Regression: `_rebuild_results_pane` lays its hero number, stat tiles and
    action buttons into *nested* QHBox/QGrid layouts added via addLayout. The
    old `_clear_layout` only removed direct child *widgets*, never recursing
    into those nested layouts -- so every rebuild (a pixel-calibration edit, a
    ground-truth load, a colour-by change...) orphaned the previous copies,
    which kept the container as parent and stayed visible, stacking up. The
    visible symptom was the "Refine…"/"Measurements" buttons overlapping and a
    ghost "Measure" label bleeding over the calibration hint. Rebuilding many
    times must leave exactly one copy of each action button."""
    from studio.components import PillButton
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    assert ws._last_result is not None

    def action_buttons():
        wanted = {"Save masks", "Export CSV", "Refine…", "Analytics",
                  "Explore cell population"}
        return [b.text() for b in ws._results_container.findChildren(PillButton)
                if b.text() in wanted]

    expected = ["Analytics", "Explore cell population", "Export CSV",
                "Refine…", "Save masks"]
    assert sorted(action_buttons()) == expected
    for _ in range(5):
        ws._rebuild_results_pane()
        app.processEvents()
    # still exactly one of each -- not 6 stacked copies
    assert sorted(action_buttons()) == expected


def test_editing_the_mask_keeps_results_and_card_count_in_sync(app, segment, projects, toasts, tmp_path, storage):
    """Regression for the reported "project card says 122 cells, Results says
    45" drift. Erasing cells used to update only the canvas legend (max id),
    leaving the Results panel stats and the persisted project stats stale.
    After an edit the debounced sync must recompute the result AND persist the
    new distinct-cell count, so the Results panel, the stored stats, and the
    legend all agree."""
    import time
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    seg = ws._layers.find("Segmentation")
    assert ws._last_result["n_cells"] == 2
    assert ws._project.stats.n_cells == 2

    # erase one of the two cells -> one distinct instance remains
    data = seg.data.copy()
    data[data == 2] = 0
    seg.data = data
    ws._layers.notify()
    # let the debounced results-sync timer fire
    t0 = time.monotonic()
    while ws._results_sync_timer.isActive() and time.monotonic() - t0 < 2:
        app.processEvents(); time.sleep(0.01)
    for _ in range(5):
        app.processEvents()

    assert ws._last_result["n_cells"] == 1                      # Results panel
    assert ws._project.stats.n_cells == 1                       # in-memory stats
    assert projects.store.load(project.id).stats.n_cells == 1   # persisted -> the card
    assert seg.n_labels == 1
    assert "1 detected" in ws._legend_detected.text()           # canvas legend


def test_clear_layout_recurses_into_nested_layouts(app):
    """Unit-level guard for the same bug: `_clear_layout` must empty widgets
    held inside nested layouts, not just direct child widgets."""
    from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget
    from studio.workspace import _clear_layout
    host = QWidget()
    outer = QVBoxLayout(host)
    outer.addWidget(QLabel("direct"))
    inner = QHBoxLayout()
    nested_a, nested_b = QLabel("a"), QLabel("b")
    inner.addWidget(nested_a)
    inner.addWidget(nested_b)
    outer.addLayout(inner)
    assert len(host.findChildren(QLabel)) == 3
    _clear_layout(outer)
    app.processEvents()  # let deleteLater() run
    assert outer.count() == 0
    assert host.findChildren(QLabel) == []


def test_predict_result_survives_a_full_project_reload(app, segment, projects, toasts, tmp_path, storage):
    """The literal "close and reopen the project" case: a brand new
    WorkspaceScreen (as if the app were restarted) must still show the
    previous session's result for an image that was segmented."""
    project = _make_project(tmp_path, projects, storage)
    ws1 = _ws(app, segment, projects, toasts)
    ws1._load_project(project)
    ws1._start_predict()
    _pump(app, ws1)
    assert ws1._layers.find("Segmentation").max_label == 2

    # a second, independent screen instance re-loading the same project —
    # nothing shared except what's actually on disk
    ws2 = _ws(app, segment, projects, [])
    ws2._load_project(projects.store.load(project.id))
    seg2 = ws2._layers.find("Segmentation")
    assert seg2.max_label == 2
    assert ws2._last_result is not None
    assert ws2._last_result["n_cells"] == 2


def test_images_pane_shows_predicted_status_for_a_saved_but_unselected_image(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=2)
    img_a, img_b = project.image_paths
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    ws._select_image(img_b)  # img_a is no longer the selected image...
    assert ws._current_image_path != img_a
    assert segment.has_result_mask(project, img_a) is True  # ...but its result is still on disk


def test_reopening_an_image_with_gt_also_shows_evaluation_immediately(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, with_gt=True)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    assert ws._gt_metrics is not None
    first_f1 = ws._gt_metrics["f1"]

    ws2 = _ws(app, segment, projects, [])
    ws2._load_project(projects.store.load(project.id))
    assert ws2._gt_metrics is not None
    assert ws2._gt_metrics["f1"] == first_f1


def test_start_predict_bad_config_toasts_synchronously(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    project.settings.model_name = ""  # no LoRA -> build_config raises
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    assert ws._predicting is False
    assert any("Can't run segmentation" in t[0] for t in toasts)


# ── ground truth evaluation ───────────────────────────────────────────────────
def test_load_gt_and_evaluate_updates_metrics_and_dashboard_stats(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, with_gt=False)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)

    gt_path = tmp_path / "gt.png"
    _write_mask(gt_path, (1, (6, 24, 6, 24)), (2, (24, 42, 24, 42)))
    ws._load_gt(str(gt_path))
    assert ws._gt_metrics is not None
    reloaded = projects.store.load(project.id)
    assert reloaded.stats.last_f1 == ws._gt_metrics["f1"]


def test_color_by_recolors_segmentation(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    seg = ws._layers.find("Segmentation")
    ws._on_color_by("Area (heatmap)")
    assert seg.color_overrides
    ws._on_color_by("Instance ID (default)")
    assert seg.color_overrides == {}


# ── save / export (QFileDialog monkeypatched) ────────────────────────────────
def test_save_masks_without_a_result_toasts(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._save_masks()
    assert toasts and "Nothing to save" in toasts[-1][0]


def test_save_masks_writes_a_file(app, segment, projects, toasts, tmp_path, storage, monkeypatch):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    out = tmp_path / "out_mask.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    ws._save_masks()
    assert out.exists()
    assert toasts[-1][0] == "Masks saved"


def test_export_csv_writes_a_file(app, segment, projects, toasts, tmp_path, storage, monkeypatch):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    out = tmp_path / "out.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    ws._export_csv()
    assert out.exists()
    assert "Area" in out.read_text()


# ── batch ─────────────────────────────────────────────────────────────────────
def test_start_batch_runs_and_updates_dashboard_stats(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=3)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_batch()
    assert ws._batching is True
    _pump(app, ws)
    assert ws._batching is False
    reloaded = projects.store.load(project.id)
    assert reloaded.stats.n_cells == 3 * 2
    assert reloaded.stats.progress == 100
    assert any("Batch complete" in t[0] for t in toasts)


def test_start_batch_empty_project_toasts(app, segment, projects, toasts, tmp_path, storage):
    project = projects.store.create("Empty", "", settings=ProjectSettings(
        engine="cellseg1", model_name=str(storage / "loras" / "nuclei-dapi-r8.pth")))
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_batch()
    assert toasts and "Can't start batch" in toasts[-1][0]


# ── benchmark ─────────────────────────────────────────────────────────────────
def test_start_benchmark_without_gt_toasts(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, with_gt=False)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_benchmark()
    assert toasts and "Can't run benchmark" in toasts[-1][0]


def test_start_benchmark_scores_engines(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, with_gt=True)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_benchmark()
    assert ws._benching is True
    _pump(app, ws)
    assert ws._benching is False
    assert ws._bench_rows
    assert any("Benchmark complete" in t[0] for t in toasts)


# ── viewer bar actions ─────────────────────────────────────────────────────────
def test_toggle_mip_works_without_a_volume_like_real_napari(app, segment, projects, toasts, tmp_path, storage):
    """Real napari's ndisplay toggle has no dimensionality guard — it must
    not silently refuse (or toast an excuse) on a plain 2-D image; the
    Canvas gets the pseudo-3D tilt instead of a real MIP projection, but the
    *toggle itself* always works."""
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._toggle_mip()
    assert ws._canvas.mip is True


def test_roll_channel_with_one_channel_toasts_instead_of_silent_noop(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._roll_channel()
    assert toasts and "Only one channel loaded" in toasts[-1][0]


def test_toggle_grid_and_transpose(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._toggle_grid()
    assert ws._canvas.grid is True
    ws._toggle_transpose()
    assert ws._canvas.transposed is True


# ── toolbar active-state sync ─────────────────────────────────────────────────
def _is_toolbar_button_on(btn) -> bool:
    """True if _style_toolbar_button styled this button as "on" (signal_weak
    background) rather than the default transparent/off look."""
    return "background:transparent" not in btn.styleSheet()


def test_grid_button_highlights_when_grid_is_on(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    grid_btn = ws._vbar_buttons["grid"]
    assert not _is_toolbar_button_on(grid_btn)
    ws._toggle_grid()
    assert _is_toolbar_button_on(grid_btn)
    ws._toggle_grid()
    assert not _is_toolbar_button_on(grid_btn)


def test_mip_button_highlights_with_or_without_a_volume(app, segment, projects, toasts, tmp_path, storage):
    """Matches real napari's own ndisplay toggle: works — and so highlights
    — regardless of whether the loaded data has a volume to project."""
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    cube_btn = ws._vbar_buttons["cube3d"]
    ws._toggle_mip()  # no volume loaded -> still toggles (pseudo-3D tilt), still highlights
    assert ws._canvas.mip is True
    assert _is_toolbar_button_on(cube_btn)
    ws._toggle_mip()
    assert not _is_toolbar_button_on(cube_btn)

    volume = LabelsLayer("Segmentation", np.zeros((3, 20, 20), dtype=np.int32))
    ws._layers.remove(ws._layers.index_of(ws._layers.find("Segmentation")))
    ws._layers.add(volume, select=False)
    ws._toggle_mip()
    assert ws._canvas.mip is True
    assert _is_toolbar_button_on(cube_btn)


def test_floating_tool_strip_highlights_active_mode(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    pan_btn = ws._floating_tool_buttons[0][0]  # (target, PAN_ZOOM)
    point_btn = ws._floating_tool_buttons[1][0]  # (points, __add_point__ -- an action, not a mode)
    assert _is_toolbar_button_on(pan_btn)  # PAN_ZOOM is the default mode
    assert not _is_toolbar_button_on(point_btn)
    ws._set_canvas_mode(PAINT)
    assert not _is_toolbar_button_on(pan_btn)  # pan no longer the active mode
    assert not _is_toolbar_button_on(point_btn)  # an action never shows "active"


def test_selecting_a_new_image_resets_mip(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=2)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._layers.remove(ws._layers.index_of(ws._layers.find("Segmentation")))
    ws._layers.add(LabelsLayer("Segmentation", np.zeros((3, 20, 20), dtype=np.int32)), select=False)
    ws._toggle_mip()
    assert ws._canvas.mip is True
    ws._current_image_path = None  # force _select_image to treat it as a real switch
    ws._select_image(project.image_paths[1])
    assert ws._canvas.mip is False


def test_toggle_logs_console_calls_the_injected_callback(app, segment, projects, toasts):
    calls = []
    ws = _ws(app, segment, projects, toasts, on_toggle_logs=lambda: calls.append(True))
    ws._toggle_logs_console()
    assert calls == [True]


# ── breadcrumb navigation ─────────────────────────────────────────────────────
def test_breadcrumb_projects_click_navigates_back(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    seen = []
    ws = WorkspaceScreen(theme.DARK, segment, projects,
                         lambda title, sub: toasts.append((title, sub)),
                         on_navigate=lambda key: seen.append(key))
    ws._load_project(project)
    ws._go_to_projects()
    assert seen == ["projects"]


def test_breadcrumb_shows_project_name_not_in_the_projects_link(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert ws._crumb_projects.text() == "Projects"
    assert project.name in ws._crumb_name.text()


def test_floating_add_point_selects_or_creates_a_points_layer(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._on_floating_tool("__add_point__")
    assert isinstance(ws._layers.selected, PointsLayer)
    assert ws._canvas.mode == PAINT

    ws._on_floating_tool(PAN_ZOOM)
    assert ws._canvas.mode == PAN_ZOOM


# ── labels-layer settings (contour/dims/checkboxes/colour) ───────────────────
def test_set_selected_label_updates_layer_and_rebuilds(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    ws._set_selected_label(seg, 7)
    assert seg.selected_label == 7


def test_pick_label_color_then_auto_clears_overrides(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    seg.selected_label = 3
    ws._pick_label_color(seg, "#ff8800")
    assert seg.color_overrides.get(3) == (0xFF, 0x88, 0x00)
    ws._set_color_mode(seg, "auto")
    assert seg.color_overrides == {}


def test_set_layer_int_attr_updates_contour_and_edit_dims(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    ws._set_layer_int_attr(seg, "contour", 5)
    assert seg.contour == 5
    ws._set_layer_int_attr(seg, "n_edit_dimensions", 3)
    assert seg.n_edit_dimensions == 3


def test_toggle_layer_bool_flips_each_labels_checkbox(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    for attr in ("contiguous", "preserve_labels", "show_selected_label"):
        before = getattr(seg, attr)
        ws._toggle_layer_bool(seg, attr)
        assert getattr(seg, attr) is not before


def test_set_brush_size_updates_layer_and_badge(app, segment, projects, toasts, tmp_path, storage):
    from studio.components import Badge
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    badge = Badge("1", theme.DARK)
    ws._set_brush_size(seg, 0.5, badge)
    assert seg.brush_size == 50
    assert badge.text() == "50"


# ── image-layer settings (gamma/colormap) ────────────────────────────────────
def test_set_image_gamma_updates_layer_and_badge(app, segment, projects, toasts, tmp_path, storage):
    from studio.components import Badge
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    img_layer = ws._layers.find(Path(project.image_paths[0]).stem)
    badge = Badge("1.00", theme.DARK)
    ws._set_image_gamma(img_layer, 0.5, badge)
    assert img_layer.gamma == pytest.approx(1.5)
    assert badge.text() == "1.50"


def test_set_image_colormap_updates_layer(app, segment, projects, toasts, tmp_path, storage):
    from studio.layer_model import IMAGE_COLORMAPS
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    img_layer = ws._layers.find(Path(project.image_paths[0]).stem)
    other = next(c for c in IMAGE_COLORMAPS if c != img_layer.colormap)
    ws._set_image_colormap(img_layer, other)
    assert img_layer.colormap == other


# ── points/shapes settings ────────────────────────────────────────────────────
def test_set_point_size_and_clear_points(app, segment, projects, toasts, tmp_path, storage):
    from studio.components import Badge
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._add_points_layer()
    pts = ws._layers.selected
    pts.add(5, 5)
    pts.add(10, 10)
    badge = Badge("10", theme.DARK)
    ws._set_point_size(pts, 0.5, badge)
    assert pts.size == 20
    assert badge.text() == "20"
    ws._clear_points(pts)
    assert pts.points == []


def test_set_edge_width_and_clear_shapes(app, segment, projects, toasts, tmp_path, storage):
    from studio.components import Badge
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._add_shapes_layer()
    shapes = ws._layers.selected
    shapes.add("polygon", [(1, 1), (1, 5), (5, 5)])
    badge = Badge("1.0", theme.DARK)
    ws._set_edge_width(shapes, 0.5, badge)
    assert shapes.edge_width == pytest.approx(5.0)
    assert badge.text() == "5.0"
    ws._clear_shapes(shapes)
    assert shapes.shapes == []


# ── overlay visibility toggles ────────────────────────────────────────────────
def test_toggle_show_predictions_hides_segmentation_not_gt(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, with_gt=True)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    gt_path = tmp_path / "img_0_mask.png"
    ws._load_gt(str(gt_path))
    ws._toggle_show_predictions(False)
    assert ws._layers.find("Segmentation").visible is False
    assert ws._layers.find("Ground truth").visible is True
    ws._toggle_show_predictions(True)
    assert ws._layers.find("Segmentation").visible is True


def test_toggle_show_gt_without_gt_toasts(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._toggle_show_gt(True)
    assert toasts and "No ground truth loaded" in toasts[-1][0]


def test_toggle_show_gt_with_gt_flips_visibility(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, with_gt=True)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._load_gt(str(tmp_path / "img_0_mask.png"))
    assert ws._layers.find("Ground truth").visible is True
    ws._toggle_show_gt(False)
    assert ws._layers.find("Ground truth").visible is False


# ── engine-specific settings (SAM2 / backbone) ────────────────────────────────
def test_backbone_select_updates_vit_name(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._on_backbone_select("ViT-B")
    assert project.settings.vit_name == "vit_b"


def test_sam2_model_and_tracking_mode_selects(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    project.settings.engine = "sam2"
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._on_sam2_model_select("Small")
    assert project.settings.sam2_model == "small"
    ws._on_sam2_tracking_mode(1)
    assert project.settings.sam2_tracking_mode == "propagate"
    ws._on_sam2_tracking_mode(0)
    assert project.settings.sam2_tracking_mode == "independent"


def test_generic_set_setting_roundtrips_a_plain_toggle(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    assert project.settings.clahe is False
    ws._set_setting("clahe", True)
    assert project.settings.clahe is True
    assert project.settings.quality_preset == "Balanced"  # plain _set_setting never marks Custom


# ── refine / measurements ─────────────────────────────────────────────────────
def test_refine_button_toasts_honest_coming_soon(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._refine_coming_soon()
    assert toasts and "coming soon" in toasts[-1][0].lower()


def test_show_measurements_without_a_result_toasts(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._show_measurements()
    assert toasts and "No measurements yet" in toasts[-1][0]


def test_show_measurements_after_predict_opens_the_analytics_dialog(app, segment, projects, toasts, tmp_path, storage):
    from studio.cell_analytics import CellAnalyticsDialog, CellAnalyticsPanel
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws.show()  # the screen is always visible when a user opens the dialog
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    ws._show_measurements()
    app.processEvents()
    dlgs = ws.findChildren(CellAnalyticsDialog)
    assert dlgs and dlgs[-1].isVisible()
    # the dialog is fed the real per-cell result and builds its panel
    assert dlgs[-1].findChildren(CellAnalyticsPanel)


def test_export_csv_without_a_result_toasts(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._export_csv()
    assert toasts and "Nothing to export" in toasts[-1][0]


# ── Assistant integration ───────────────────────────────────────────────────
def test_assistant_context_with_no_project_is_empty(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    image, mask, params = ws.assistant_context()
    assert image is None and mask is None and params == {}


def test_assistant_context_before_any_predict_has_image_and_params_but_no_mask(
        app, segment, projects, toasts, tmp_path, storage):
    """An image is selected (so a placeholder all-zero "Segmentation" layer
    already exists) but nothing has actually been predicted yet — the
    advisor must see mask=None (-> "run a prediction first"), not an
    all-zero mask (-> "no cells detected", the wrong diagnosis)."""
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    image, mask, params = ws.assistant_context()
    assert image is not None
    assert mask is None
    assert params["points_per_side"] == project.settings.points_per_side


def test_assistant_context_after_predict_returns_real_mask(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws._start_predict()
    _pump(app, ws)
    image, mask, params = ws.assistant_context()
    assert image is not None
    assert mask is not None and int(mask.max()) == 2
    assert params["engine"] == "cellseg1"


def test_apply_assistant_changes_without_a_project_returns_none(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    assert ws.apply_assistant_changes({"pred_iou_thresh": 0.5}) is None


def test_apply_assistant_changes_updates_settings_marks_custom_and_persists(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    applied = ws.apply_assistant_changes({"pred_iou_thresh": 0.55, "min_mask_area": 40})
    assert set(applied) == {"pred_iou_thresh", "min_mask_area"}
    assert ws._project.settings.pred_iou_thresh == 0.55
    assert ws._project.settings.min_mask_area == 40
    assert ws._project.settings.quality_preset == "Custom"
    reloaded = projects.store.load(project.id)
    assert reloaded.settings.pred_iou_thresh == 0.55


def test_apply_assistant_changes_ignores_unknown_keys(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    applied = ws.apply_assistant_changes({"not_a_real_setting": 1})
    assert applied == []
    assert ws._project.settings.quality_preset == "Balanced"  # untouched — nothing real was applied


def test_apply_assistant_changes_cannot_shadow_a_settings_method(
        app, segment, projects, toasts, tmp_path, storage):
    """A changes dict is always advisor-sourced in practice (a fixed,
    known-safe key set) — but apply_assistant_changes whitelists against
    real dataclass fields regardless, so a stray key can never setattr over
    e.g. ProjectSettings.to_dict via this path."""
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    applied = ws.apply_assistant_changes({"to_dict": "clobbered"})
    assert applied == []
    assert callable(ws._project.settings.to_dict)


def test_rerun_predict_without_project_toasts_like_start_predict(app, segment, projects, toasts):
    ws = _ws(app, segment, projects, toasts)
    ws.rerun_predict()
    assert toasts and "No project open" in toasts[-1][0]


def test_rerun_predict_runs_a_real_prediction(app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    ws.rerun_predict()
    assert ws._predicting is True
    _pump(app, ws)
    assert ws._last_result is not None
    assert ws._last_result["n_cells"] == 2


# ── real-click regressions: raw mouseReleaseEvent widgets must not crash ────
# Found from an actual running session: clicking "preserve labels" / "show
# selected" raised "TypeError: invalid argument to sipBadCatcherResult()" —
# a hard process abort (SIGABRT), not a catchable Python exception. Root
# cause: several rows/swatches override mouseReleaseEvent directly (not a
# real Qt signal) and, from inside that same call, trigger a container
# rebuild (_clear_layout -> setParent(None)) that reparents the very widget
# whose own mouseReleaseEvent is still executing — a PyQt/SIP
# reentrant-virtual-call hazard. Every prior test called the handler method
# directly (ws._toggle_layer_bool(...), ws._select_layer(...), etc.) or
# even called widget.mouseReleaseEvent(event) directly as a plain Python
# method call — neither actually dispatches through Qt's C++ virtual-call
# machinery the way a real click does, so neither ever reproduced this.
# Only routing the event through QApplication.sendEvent() (the same path
# QWidget::event() takes for a real click) triggers it — confirmed by
# reproducing the exact SIGABRT against the unfixed code before applying
# the fix below. Each dispatch is followed by pumping the event loop so a
# deferred (QTimer.singleShot(0, ...)) handler gets a chance to run.
def _dispatch_release(widget, pos=(5, 5)):
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(*pos),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, ev)


def _dispatch_press(widget, pos=(5, 5)):
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(*pos),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, ev)


def _pump_events(app, n=10):
    for _ in range(n):
        app.processEvents()


def _find_check_row(ws, name: str) -> QFrame:
    from PyQt6.QtWidgets import QLabel
    for lbl in ws._layer_controls_container.findChildren(QLabel):
        if lbl.text() == name:
            return lbl.parentWidget()
    raise AssertionError(f"no checkbox row named {name!r} found")


def test_clicking_preserve_labels_checkbox_does_not_crash(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    before = seg.preserve_labels
    row = _find_check_row(ws, "preserve labels")
    _dispatch_release(row)
    _pump_events(app)
    assert seg.preserve_labels is not before


def test_clicking_show_selected_checkbox_does_not_crash(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    before = seg.show_selected_label
    row = _find_check_row(ws, "show selected")
    _dispatch_release(row)
    _pump_events(app)
    assert seg.show_selected_label is not before


def test_clicking_contiguous_checkbox_does_not_crash(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    before = seg.contiguous
    row = _find_check_row(ws, "contiguous")
    _dispatch_release(row)
    _pump_events(app)
    assert seg.contiguous is not before


def test_clicking_a_label_colour_swatch_does_not_crash(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    seg = ws._layers.find("Segmentation")
    seg.selected_label = 1
    first_color = demo.LABEL_COLORS[0]
    sw = next(f for f in ws._layer_controls_container.findChildren(QFrame)
              if f"background:{first_color}" in f.styleSheet())
    _dispatch_release(sw)
    _pump_events(app)
    assert seg.color_overrides.get(1) is not None


def test_clicking_a_layer_row_does_not_crash_and_selects_it(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    gt_idx = ws._layers.index_of(ws._layers.find("Segmentation"))
    other_idx = 0 if gt_idx != 0 else 1
    row = ws._layers_list_layout.itemAt(other_idx).widget()
    _dispatch_release(row)
    _pump_events(app)
    assert ws._layers.selected_index == other_idx


def test_clicking_an_image_row_does_not_crash_and_switches_image(
        app, segment, projects, toasts, tmp_path, storage):
    project = _make_project(tmp_path, projects, storage, n_images=2)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    target_path = project.image_paths[1]
    row = ws._images_list_layout.itemAt(1).widget()
    # Image rows are SwipeRows now: a tap is a press then release at the same
    # spot (no drag). Release-only would be ignored (no drag in progress).
    _dispatch_press(row)
    _dispatch_release(row)
    _pump_events(app)
    assert ws._current_image_path == target_path


# ── regression: floating tool-strip icons must not fall back to "chevron" ───
def test_floating_tool_strip_shows_real_icons_not_a_fallback_chevron(
        app, segment, projects, toasts, tmp_path, storage):
    """_sync_toolbars used to re-derive each button's icon name from its
    raw mode/action string (e.g. "pan_zoom", "paint") instead of the
    semantic icon name used at construction ("target", "brush"). Neither
    is a real key in icons.PATHS, so icons.py's own fallback silently drew
    the generic "chevron" glyph for both — on every restyle, which fires on
    nearly every interaction, visibly replacing the real icons almost
    immediately. Existing coverage only checked the on/off *highlight*
    style, never which glyph was actually drawn, so this shipped unnoticed
    until a real screenshot showed two tool buttons as plain chevrons."""
    from studio import icons
    project = _make_project(tmp_path, projects, storage)
    ws = _ws(app, segment, projects, toasts)
    ws._load_project(project)
    t = theme.DARK

    def _img(name, color):
        return icons.icon(name, color, 16).pixmap(16, 16).toImage()

    pan_btn = ws._floating_tool_buttons[0][0]   # (target, PAN_ZOOM)
    point_btn = ws._floating_tool_buttons[1][0]  # (points, __add_point__)
    assert pan_btn.icon().pixmap(16, 16).toImage() == _img("target", t["signal"])
    assert point_btn.icon().pixmap(16, 16).toImage() == _img("points", t["text_muted"])

    ws._set_canvas_mode(PAINT)  # forces another _sync_toolbars() restyle pass
    assert pan_btn.icon().pixmap(16, 16).toImage() == _img("target", t["text_muted"])


# ── real logging (studio/log_bus.py) ────────────────────────────────────────
def test_predict_log_forwards_to_the_shared_log_bus_and_toasts_are_unchanged(
        app, segment, projects, toasts, monkeypatch):
    """_on_predict_log is shared by predict/batch/benchmark and used to throw
    away every line except [ERROR]/[HINT] (skimmed only for a toast). It must
    now also reach the real Logs console's LogBus, with the existing toast
    behaviour byte-for-byte unchanged."""
    import studio.workspace as ws_mod
    from studio import log_bus
    from studio.log_bus import LogBus

    bus = LogBus()
    monkeypatch.setattr(ws_mod, "get_log_bus", lambda: bus)
    ws = _ws(app, segment, projects, toasts)

    ws._on_predict_log("[ERROR] boom")
    ws._on_predict_log("✓ 12 cells")
    ws._on_predict_log("[HINT] try a smaller tile size")

    recs = bus.snapshot()
    assert [(r.level, r.message, r.source) for r in recs] == [
        (log_bus.ERROR, "boom", "studio.segment"),
        (log_bus.INFO, "✓ 12 cells", "studio.segment"),
        (log_bus.INFO, "try a smaller tile size", "studio.segment"),
    ]
    assert toasts == [("Segmentation failed", "boom"), ("Hint", "try a smaller tile size")]
