"""Tests for the Studio Projects tab controller (studio/project_controller.py).

Pure-logic, no Qt/torch/napari — runs under the light CI `test` group.
"""
import pytest

from studio.project import Project, ProjectSettings, ProjectStats, ProjectStore
from studio.project_controller import (
    ProjectController,
    cover_seed,
    format_count,
    relative_time,
    to_card,
)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path)


# ── format_count ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n,expected", [
    (0, "0"), (999, "999"), (1000, "1k"), (1100, "1.1k"),
    (31400, "31.4k"), (188000, "188k"), (52000, "52k"), (9700, "9.7k"),
])
def test_format_count(n, expected):
    assert format_count(n) == expected


# ── cover_seed ───────────────────────────────────────────────────────────────
def test_cover_seed_is_deterministic_per_id():
    assert cover_seed("nuclei-dapi") == cover_seed("nuclei-dapi")


def test_cover_seed_differs_across_ids():
    seeds = {cover_seed(f"project-{i}") for i in range(20)}
    assert len(seeds) > 1  # not all colliding


# ── relative_time ──────────────────────────────────────────────────────────────
def test_relative_time_just_now():
    from datetime import datetime, timezone
    assert relative_time(datetime.now(timezone.utc).isoformat()) == "just now"


def test_relative_time_hours_and_days():
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    assert relative_time((now - timedelta(hours=3)).isoformat()) == "3 hours ago"
    assert relative_time((now - timedelta(days=1, hours=1)).isoformat()) == "yesterday"
    assert relative_time((now - timedelta(days=5)).isoformat()) == "5 days ago"


def test_relative_time_handles_bad_input():
    assert relative_time("") == ""
    assert relative_time("not-a-timestamp") == ""


# ── to_card ──────────────────────────────────────────────────────────────────
def test_to_card_formats_display_fields():
    p = Project(id="x", name="X", description="desc", tags=["a", "b"],
               favorite=True, settings=ProjectSettings(engine="cellpose"),
               stats=ProjectStats(n_images=88, n_cells=14200, last_f1=0.91, progress=33))
    card = to_card(p)
    assert card.id == "x"
    assert card.engine_key == "cellpose"
    assert card.engine_label == "Cellpose-SAM"
    assert card.n_images == 88
    assert card.n_cells == "14.2k"
    assert card.f1 == "0.91"
    assert card.favorite is True
    assert card.tags == ["a", "b"]


def test_to_card_f1_is_none_when_unbenchmarked():
    p = Project(id="x", name="X")
    assert to_card(p).f1 is None


# ── seeding ──────────────────────────────────────────────────────────────────
def test_fresh_store_is_seeded_with_six_sample_projects(store):
    ctrl = ProjectController(store)
    projects = ctrl.list_projects()
    assert len(projects) == 6
    assert projects[0].name == "Fluorescence Nuclei — DAPI"  # newest of the seed set


def test_seed_if_empty_false_leaves_store_empty(store):
    ctrl = ProjectController(store, seed_if_empty=False)
    assert ctrl.list_projects() == []


def test_non_empty_store_is_not_reseeded(store):
    store.create("Existing")
    ctrl = ProjectController(store)
    assert len(ctrl.list_projects()) == 1


# ── search / filter ────────────────────────────────────────────────────────────
def test_search_matches_name_description_tags_and_engine(store):
    ctrl = ProjectController(store)
    assert {p.name for p in ctrl.list_projects(query="mitosis")} == {"Live-cell Mitosis"}
    assert {p.name for p in ctrl.list_projects(query="benchmark")} == {"BBBC039 Nuclei Benchmark"}
    assert {p.name for p in ctrl.list_projects(query="sam 2")} == {"Live-cell Mitosis"}
    assert ctrl.list_projects(query="nonexistent-xyz") == []


def test_search_is_case_insensitive(store):
    ctrl = ProjectController(store)
    assert len(ctrl.list_projects(query="DAPI")) == len(ctrl.list_projects(query="dapi"))


def test_favorites_only_filter(store):
    ctrl = ProjectController(store)
    favs = ctrl.list_projects(favorites_only=True)
    assert favs and all(p.favorite for p in favs)
    assert len(favs) < len(ctrl.list_projects())


def test_query_and_favorites_only_combine(store):
    ctrl = ProjectController(store)
    result = ctrl.list_projects(query="mitosis", favorites_only=True)
    assert {p.name for p in result} == {"Live-cell Mitosis"}
    result = ctrl.list_projects(query="organoid", favorites_only=True)
    assert result == []  # Organoid Membranes isn't a favourite


def test_engines_filter(store):
    ctrl = ProjectController(store)
    result = ctrl.list_projects(engines={"sam2"})
    assert {p.name for p in result} == {"Live-cell Mitosis"}
    result = ctrl.list_projects(engines={"cellseg1", "sam2"})
    assert {p.engine for p in result} == {"cellseg1", "sam2"}


def test_engines_filter_empty_or_none_means_no_filter(store):
    ctrl = ProjectController(store)
    assert ctrl.list_projects(engines=None) == ctrl.list_projects()
    assert ctrl.list_projects(engines=set()) == ctrl.list_projects()


def test_engines_filter_combines_with_query_and_favorites(store):
    ctrl = ProjectController(store)
    result = ctrl.list_projects(query="nuclei", engines={"cellseg1"}, favorites_only=True)
    assert {p.name for p in result} == {"Fluorescence Nuclei — DAPI"}


# ── recent / summary ───────────────────────────────────────────────────────────
def test_recent_respects_limit(store):
    ctrl = ProjectController(store)
    assert len(ctrl.recent(limit=2)) == 2
    assert len(ctrl.recent(limit=99)) == 6


def test_summary_counts_projects_images_and_engines(store):
    ctrl = ProjectController(store)
    n_projects, n_images, n_engines = ctrl.summary()
    assert n_projects == 6
    assert n_images == 128 + 342 + 24 + 200 + 88 + 12
    assert n_engines == 3  # cellseg1, cellpose, sam2


# ── favorites mutation ─────────────────────────────────────────────────────────
def test_toggle_favorite_flips_and_persists(store):
    ctrl = ProjectController(store)
    p = ctrl.list_projects()[0]
    was = p.favorite
    ctrl.toggle_favorite(p.id)
    assert store.load(p.id).favorite != was
    ctrl.toggle_favorite(p.id)
    assert store.load(p.id).favorite == was


# ── active project ─────────────────────────────────────────────────────────────
def test_active_project_starts_unset(store):
    ctrl = ProjectController(store)
    assert ctrl.get_active() is None


def test_set_active_and_get_active_round_trip(store):
    ctrl = ProjectController(store)
    p = ctrl.list_projects()[0]
    returned = ctrl.set_active(p.id)
    assert returned.id == p.id
    assert ctrl.get_active().id == p.id


def test_get_active_returns_none_if_project_was_deleted(store):
    ctrl = ProjectController(store)
    p = ctrl.list_projects()[0]
    ctrl.set_active(p.id)
    store.delete(p.id)
    assert ctrl.get_active() is None


# ── trash ────────────────────────────────────────────────────────────────────
def test_trash_project_removes_it_from_list_projects(store):
    ctrl = ProjectController(store)
    p = ctrl.list_projects()[0]
    ctrl.trash_project(p.id)
    assert p.id not in [x.id for x in ctrl.list_projects()]
    assert p.id in [x.id for x in ctrl.list_trashed()]


def test_trash_project_clears_active_id_if_it_was_active(store):
    ctrl = ProjectController(store)
    p = ctrl.list_projects()[0]
    ctrl.set_active(p.id)
    assert ctrl.get_active() is not None

    ctrl.trash_project(p.id)

    assert ctrl.get_active() is None


def test_trash_project_leaves_other_active_project_untouched(store):
    ctrl = ProjectController(store)
    a, b = ctrl.list_projects()[:2]
    ctrl.set_active(a.id)
    ctrl.trash_project(b.id)
    assert ctrl.get_active().id == a.id


def test_restore_project_makes_it_reappear(store):
    ctrl = ProjectController(store)
    p = ctrl.list_projects()[0]
    ctrl.trash_project(p.id)
    ctrl.restore_project(p.id)
    assert p.id in [x.id for x in ctrl.list_projects()]
    assert ctrl.list_trashed() == []


def test_delete_project_permanently_removes_files(store):
    ctrl = ProjectController(store)
    p = ctrl.list_projects()[0]
    ctrl.trash_project(p.id)
    ctrl.delete_project_permanently(p.id)
    assert not store.exists(p.id)
    assert p.id not in [x.id for x in ctrl.list_trashed()]


def test_delete_project_permanently_clears_active_id(store):
    ctrl = ProjectController(store)
    p = ctrl.list_projects()[0]
    ctrl.set_active(p.id)
    ctrl.delete_project_permanently(p.id)
    assert ctrl.get_active() is None
