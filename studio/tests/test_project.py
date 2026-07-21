"""Tests for the Studio Project data model + store (studio/project.py).

Pure-logic, no Qt/torch/napari — runs under the light CI `test` group.
"""
import json
from pathlib import Path

import pytest

from studio.project import (
    ENGINE_KIND,
    ENGINE_LABELS,
    ENGINES,
    Project,
    ProjectSettings,
    ProjectStats,
    ProjectStore,
    slugify,
    SCHEMA_VERSION,
)


# ── slugify ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name,expected", [
    ("Fluorescence Nuclei — DAPI", "fluorescence-nuclei-dapi"),
    ("  H&E Tissue  ", "h-e-tissue"),
    ("BBBC039", "bbbc039"),
    ("!!!", "project"),          # all-symbol falls back
    ("", "project"),            # empty falls back
    ("Live/Cell #2", "live-cell-2"),
])
def test_slugify(name, expected):
    assert slugify(name) == expected


# ── engine display constants ─────────────────────────────────────────────────
def test_engine_labels_and_kinds_cover_every_engine():
    for engine in ENGINES:
        assert engine in ENGINE_LABELS
        assert engine in ENGINE_KIND


# ── settings round-trip ──────────────────────────────────────────────────────
def test_settings_round_trip_is_identity():
    s = ProjectSettings(engine="sam2", points_per_side=48, pred_iou_thresh=0.77,
                        channels=[0, 2], clahe=True, tiled=True, pixel_size_um=0.65)
    assert ProjectSettings.from_dict(s.to_dict()) == s


def test_settings_from_dict_ignores_unknown_and_defaults_missing():
    # forward/backward compatible: extra key dropped, absent keys defaulted
    s = ProjectSettings.from_dict({"engine": "cellpose", "totally_new_key": 123})
    assert s.engine == "cellpose"
    assert s.points_per_side == 32          # default preserved
    assert not hasattr(s, "totally_new_key")


def test_settings_cover_every_engine_family():
    s = ProjectSettings()
    d = s.to_dict()
    # a guard so nobody silently drops a knob the pipeline needs
    for key in ("engine", "vit_name", "lora_rank",
                "points_per_side", "pred_iou_thresh", "stability_score_thresh",
                "box_nms_thresh", "min_mask_area",
                "cp_diameter", "cp_flow_threshold", "cp_cellprob_threshold",
                "sam2_model", "sam2_tracking_mode", "stitch_iou",
                "pixel_size_um", "channels", "channel_low", "channel_high",
                "clahe", "tiled", "tile_size", "tile_overlap", "color_by"):
        assert key in d, f"settings lost the '{key}' knob"


# ── create / load / list ─────────────────────────────────────────────────────
def test_create_persists_and_reloads(tmp_path):
    store = ProjectStore(tmp_path)
    p = store.create("Fluorescence Nuclei — DAPI", description="DAPI screen",
                     tags=["fluorescence", "nuclei"])
    assert p.id == "fluorescence-nuclei-dapi"
    assert store.exists(p.id)

    reloaded = store.load(p.id)
    assert reloaded.name == p.name
    assert reloaded.description == "DAPI screen"
    assert reloaded.tags == ["fluorescence", "nuclei"]
    assert reloaded.schema_version == SCHEMA_VERSION


def test_create_generates_unique_ids_for_same_name(tmp_path):
    store = ProjectStore(tmp_path)
    a = store.create("Nuclei")
    b = store.create("Nuclei")
    c = store.create("Nuclei")
    assert a.id == "nuclei"
    assert b.id == "nuclei-2"
    assert c.id == "nuclei-3"
    assert len(store.list()) == 3


def test_settings_survive_project_round_trip(tmp_path):
    store = ProjectStore(tmp_path)
    s = ProjectSettings(engine="cellpose", cp_diameter=42.0,
                        cp_flow_threshold=0.5, resize_size=1024, color_by="area")
    p = store.create("Organoids", settings=s)
    assert store.load(p.id).settings == s


# ── recent ordering ──────────────────────────────────────────────────────────
def test_list_is_ordered_newest_modified_first(tmp_path):
    store = ProjectStore(tmp_path)
    a = store.create("Alpha")
    b = store.create("Beta")
    # bump Alpha's updated_at past Beta by re-saving with a later timestamp
    a.updated_at = "2999-01-01T00:00:00+00:00"
    store._file(a.id).write_text(json.dumps(a.to_dict()), encoding="utf-8")
    ids = [p.id for p in store.list()]
    assert ids[0] == a.id
    assert set(ids) == {a.id, b.id}


def test_recent_respects_limit(tmp_path):
    store = ProjectStore(tmp_path)
    for i in range(8):
        store.create(f"P{i}")
    assert len(store.recent(limit=3)) == 3
    assert len(store.recent(limit=99)) == 8


# ── favorites ────────────────────────────────────────────────────────────────
def test_favorites_toggle_and_filter(tmp_path):
    store = ProjectStore(tmp_path)
    a = store.create("A")
    store.create("B")
    assert store.favorites() == []
    store.set_favorite(a.id, True)
    favs = store.favorites()
    assert [p.id for p in favs] == [a.id]
    store.set_favorite(a.id, False)
    assert store.favorites() == []


# ── rename / duplicate ──────────────────────────────────────────────────────
def test_rename_updates_name_keeps_id(tmp_path):
    store = ProjectStore(tmp_path)
    p = store.create("Old Name")
    renamed = store.rename(p.id, "New Name")
    assert renamed.id == p.id
    assert renamed.name == "New Name"
    assert store.load(p.id).name == "New Name"


def test_rename_blank_keeps_the_existing_name(tmp_path):
    store = ProjectStore(tmp_path)
    p = store.create("Keep Me")
    renamed = store.rename(p.id, "   ")
    assert renamed.name == "Keep Me"


def test_rename_strips_whitespace(tmp_path):
    store = ProjectStore(tmp_path)
    p = store.create("A")
    renamed = store.rename(p.id, "  Padded  ")
    assert renamed.name == "Padded"


def test_duplicate_copies_settings_tags_and_images_resets_stats(tmp_path):
    store = ProjectStore(tmp_path)
    settings = ProjectSettings(engine="sam2", points_per_side=48)
    source = store.create("Source", description="desc", tags=["a", "b"],
                          settings=settings, image_paths=["/x/1.tif", "/x/2.tif"])
    store.set_favorite(source.id, True)
    source = store.load(source.id)
    source.stats = ProjectStats(n_images=2, n_cells=500, last_f1=0.9, progress=100)
    store.save(source, touch=False)

    dup = store.duplicate(source.id)

    assert dup.id != source.id
    assert dup.name == "Source copy"
    assert dup.description == "desc"
    assert dup.tags == ["a", "b"]
    assert dup.settings == settings
    assert dup.image_paths == ["/x/1.tif", "/x/2.tif"]
    assert dup.favorite is False  # not carried over
    # n_images is a derived count of the copied image list (create() sets it
    # for every new project, duplicate included) -- everything that reflects
    # an actual *run*, nothing has, resets.
    assert dup.stats == ProjectStats(n_images=2)


def test_duplicate_settings_are_independent_of_the_source(tmp_path):
    store = ProjectStore(tmp_path)
    source = store.create("Source", settings=ProjectSettings(points_per_side=32))
    dup = store.duplicate(source.id)
    dup.settings.points_per_side = 64
    store.save(dup)
    assert store.load(source.id).settings.points_per_side == 32  # untouched


def test_duplicate_twice_does_not_collide_on_id(tmp_path):
    store = ProjectStore(tmp_path)
    source = store.create("Source")
    dup1 = store.duplicate(source.id)
    dup2 = store.duplicate(source.id)
    assert dup1.id != dup2.id
    assert len(store.list()) == 3


# ── delete ───────────────────────────────────────────────────────────────────
def test_delete_removes_project_and_is_idempotent(tmp_path):
    store = ProjectStore(tmp_path)
    p = store.create("Doomed")
    assert store.exists(p.id)
    store.delete(p.id)
    assert not store.exists(p.id)
    store.delete(p.id)  # no error second time
    assert store.list() == []


# ── robustness ───────────────────────────────────────────────────────────────
def test_list_skips_corrupt_and_stray_entries(tmp_path):
    store = ProjectStore(tmp_path)
    good = store.create("Good")
    # a corrupt project.json
    bad_dir = tmp_path / "broken"
    bad_dir.mkdir()
    (bad_dir / "project.json").write_text("{ not json", encoding="utf-8")
    # a stray directory with no project.json
    (tmp_path / "empty").mkdir()
    # a stray loose file at the root
    (tmp_path / "README.txt").write_text("hi", encoding="utf-8")

    listed = store.list()
    assert [p.id for p in listed] == [good.id]


def test_save_is_atomic_no_tmp_left_behind(tmp_path):
    store = ProjectStore(tmp_path)
    p = store.create("Atomic")
    leftovers = list(store._dir(p.id).glob("*.tmp"))
    assert leftovers == []


def test_project_from_dict_tolerates_missing_settings_and_stats(tmp_path):
    # a minimal hand-written file (older schema) still loads with defaults
    p = Project.from_dict({"id": "x", "name": "X"})
    assert p.settings == ProjectSettings()
    assert p.stats.n_images == 0
    assert p.engine == "cellseg1"


# ── save(touch=False) ─────────────────────────────────────────────────────────
def test_save_with_touch_false_preserves_explicit_timestamps(tmp_path):
    store = ProjectStore(tmp_path)
    p = Project(id="frozen", name="Frozen", updated_at="2020-01-01T00:00:00+00:00",
               created_at="2020-01-01T00:00:00+00:00")
    store.save(p, touch=False)
    reloaded = store.load("frozen")
    assert reloaded.updated_at == "2020-01-01T00:00:00+00:00"


def test_save_default_still_touches(tmp_path):
    store = ProjectStore(tmp_path)
    p = Project(id="live", name="Live", updated_at="2020-01-01T00:00:00+00:00")
    store.save(p)  # touch=True by default
    assert store.load("live").updated_at != "2020-01-01T00:00:00+00:00"


# ── import_image / import_images: copy-into-project (survives moves + TCC) ─────
# import_image only copies bytes (shutil.copy2, no decode), so a plain byte
# payload exercises it fully without pulling cv2/numpy into this light-group
# test module.
def _png(path, payload=b"\x89PNG\r\n\x1a\n-fake-bytes"):
    Path(path).write_bytes(payload)


def test_import_image_copies_into_project_images_dir(tmp_path):
    store = ProjectStore(tmp_path / "store")
    proj = store.create("P")
    src = tmp_path / "download.png"
    _png(src)
    dest = store.import_image(proj.id, src)
    assert dest.exists()
    assert dest.parent == store.image_dir(proj.id)
    assert dest.read_bytes() == src.read_bytes()  # a real copy, not a reference


def test_import_image_dedupes_same_filename_from_different_sources(tmp_path):
    store = ProjectStore(tmp_path / "store")
    proj = store.create("P")
    a = tmp_path / "a" / "download.png"
    b = tmp_path / "b" / "download.png"
    a.parent.mkdir(); b.parent.mkdir()
    _png(a); _png(b)
    d1 = store.import_image(proj.id, a)
    d2 = store.import_image(proj.id, b)
    assert d1 != d2  # same basename, two distinct stored files
    assert d1.exists() and d2.exists()


def test_import_images_falls_back_to_original_path_when_source_unreadable(tmp_path):
    store = ProjectStore(tmp_path / "store")
    proj = store.create("P")
    good = tmp_path / "good.png"
    _png(good)
    missing = str(tmp_path / "gone.png")  # never created
    out = store.import_images(proj.id, [str(good), missing])
    assert Path(out[0]).parent == store.image_dir(proj.id)  # copied
    assert out[1] == missing  # reference kept, not dropped
    assert len(out) == 2


def test_import_images_is_idempotent_for_already_stored_paths(tmp_path):
    store = ProjectStore(tmp_path / "store")
    proj = store.create("P")
    src = tmp_path / "download.png"
    _png(src)
    once = store.import_images(proj.id, [str(src)])
    twice = store.import_images(proj.id, once)  # re-importing our own copy
    assert twice == once  # no copy-of-a-copy
    assert len(list(store.image_dir(proj.id).glob("*.png"))) == 1


# ── ProjectCover ──────────────────────────────────────────────────────────────
def test_project_cover_defaults_to_auto():
    from studio.project import ProjectCover
    c = ProjectCover()
    assert c.kind == "auto" and c.color == "" and c.image_path == ""


def test_project_cover_round_trips_through_project(tmp_path):
    from studio.project import ProjectCover
    store = ProjectStore(tmp_path)
    p = store.create("Covered")
    p.cover = ProjectCover(kind="color", color="#6d87f1")
    store.save(p)
    reloaded = store.load(p.id)
    assert reloaded.cover.kind == "color"
    assert reloaded.cover.color == "#6d87f1"


def test_project_cover_image_round_trips(tmp_path):
    from studio.project import ProjectCover
    store = ProjectStore(tmp_path)
    p = store.create("Img")
    p.cover = ProjectCover(kind="image", image_path="/tmp/cover.png")
    store.save(p)
    assert store.load(p.id).cover.image_path == "/tmp/cover.png"


def test_project_without_cover_key_loads_as_auto():
    """Backward compatibility: a project file written before covers existed
    (no ``cover`` key) must load with the default auto cover, not crash."""
    data = {"id": "old", "name": "Old", "settings": {}, "stats": {}}
    p = Project.from_dict(data)
    assert p.cover.kind == "auto"


def test_project_to_dict_includes_cover():
    p = Project(id="x", name="X")
    assert "cover" in p.to_dict()
    assert p.to_dict()["cover"]["kind"] == "auto"
