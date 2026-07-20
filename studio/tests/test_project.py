"""Tests for the Studio Project data model + store (studio/project.py).

Pure-logic, no Qt/torch/napari — runs under the light CI `test` group.
"""
import json

import pytest

from studio.project import (
    ENGINE_KIND,
    ENGINE_LABELS,
    ENGINES,
    Project,
    ProjectSettings,
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


# ── trash ────────────────────────────────────────────────────────────────────
def test_trash_excludes_from_list_recent_and_favorites(tmp_path):
    store = ProjectStore(tmp_path)
    a = store.create("A")
    store.create("B")
    store.set_favorite(a.id, True)

    store.trash(a.id)

    assert [p.id for p in store.list()] == ["b"]
    assert store.favorites() == []
    assert [p.id for p in store.recent(limit=10)] == ["b"]
    assert store.exists(a.id)  # files stay on disk -- this is a soft delete


def test_trash_is_reversible_via_restore(tmp_path):
    store = ProjectStore(tmp_path)
    a = store.create("A")
    store.trash(a.id)
    assert a.id not in [p.id for p in store.list()]

    store.restore(a.id)

    assert a.id in [p.id for p in store.list()]
    assert store.load(a.id).trashed_at is None
    assert not store.load(a.id).is_trashed


def test_trashed_lists_only_trashed_newest_trashed_first(tmp_path):
    store = ProjectStore(tmp_path)
    a = store.create("A")
    b = store.create("B")
    store.create("C")
    store.trash(a.id)
    store.trash(b.id)
    # trash() stamps "now" at second precision -- both calls can tie within
    # the same wall-clock second, so pin b's trashed_at explicitly later
    # than a's instead of relying on real elapsed time between the two
    # calls (same technique test_list_is_ordered_newest_modified_first uses
    # for updated_at, for the identical reason).
    proj_b = store.load(b.id)
    proj_b.trashed_at = "2999-01-01T00:00:00+00:00"
    store.save(proj_b, touch=False)

    trashed_ids = [p.id for p in store.trashed()]
    assert set(trashed_ids) == {a.id, b.id}
    assert trashed_ids[0] == b.id  # pinned later trashed_at sorts first
    assert "c" not in trashed_ids


def test_trash_does_not_bump_updated_at(tmp_path):
    store = ProjectStore(tmp_path)
    p = store.create("A")
    before = store.load(p.id).updated_at
    store.trash(p.id)
    assert store.load(p.id).updated_at == before


def test_list_include_trashed_returns_everything(tmp_path):
    store = ProjectStore(tmp_path)
    a = store.create("A")
    store.create("B")
    store.trash(a.id)
    assert len(store.list()) == 1
    assert len(store.list(include_trashed=True)) == 2


def test_is_trashed_property_reflects_trashed_at(tmp_path):
    store = ProjectStore(tmp_path)
    p = store.create("A")
    assert not p.is_trashed
    trashed = store.trash(p.id)
    assert trashed.is_trashed


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
