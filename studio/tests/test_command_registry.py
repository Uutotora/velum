"""Pure-logic tests for studio/command_registry.py -- the ⌘K palette's
fuzzy search + section grouping. No Qt import: runs in CI's light `test`
group.
"""
from studio.command_registry import Command, group_by_section, search


def _cmd(id, label, section="Actions", **kw):
    return Command(id=id, label=label, section=section, **kw)


# ── search(): empty query ────────────────────────────────────────────────────
def test_empty_query_returns_commands_unchanged_in_order():
    cmds = [_cmd("a", "Run segmentation"), _cmd("b", "Switch engine → SAM 2")]
    assert search(cmds, "") == cmds
    assert search(cmds, "   ") == cmds  # whitespace-only counts as empty


# ── search(): contiguous substring always outranks a scattered match ───────
def test_contiguous_substring_outranks_a_scattered_subsequence_match():
    run_seg = _cmd("run", "Run segmentation")
    switch_engine = _cmd("switch", "Switch engine → SAM 2")
    # "seg" is a literal substring of "Run segmentation" but only a
    # scattered subsequence of "Switch engine" (s...e...g across 3 words) --
    # the real, more relevant match must win regardless of how many
    # word-boundary starts the scattered one happens to hit.
    ranked = search([switch_engine, run_seg], "seg")
    assert ranked[0] is run_seg


def test_prefix_match_ranks_above_a_mid_label_substring_match():
    go_segment = _cmd("go", "Go to Segment")   # "seg" mid-label
    run_seg = _cmd("run", "Run segmentation")   # "seg" also mid-label, but shorter/tighter label
    switch = _cmd("sw", "Segment settings")     # "seg" is a genuine prefix
    ranked = search([go_segment, run_seg, switch], "seg")
    assert ranked[0] is switch


def test_scattered_subsequence_still_matches_as_a_fallback():
    # "rseg" has no contiguous run in "Run segmentation" (there's a space
    # between "run" and "seg") but is still a real in-order subsequence.
    cmd = _cmd("run", "Run segmentation")
    assert search([cmd], "rseg") == [cmd]


def test_non_matching_query_excludes_the_command():
    cmd = _cmd("run", "Run segmentation")
    assert search([cmd], "xyz") == []


def test_case_insensitive():
    cmd = _cmd("run", "Run Segmentation")
    assert search([cmd], "RUN") == [cmd]


# ── search(): keywords ───────────────────────────────────────────────────────
def test_keywords_are_searchable_but_rank_below_an_equal_label_match():
    by_keyword = _cmd("sam2", "Switch engine → SAM 2", keywords="zstack timelapse")
    by_label = _cmd("zoo", "Zstack viewer")   # contrived label match for the same term
    ranked = search([by_keyword, by_label], "zstack")
    assert by_label in ranked and by_keyword in ranked
    assert ranked.index(by_label) < ranked.index(by_keyword)


def test_keywords_alone_are_enough_to_surface_a_command():
    cmd = _cmd("sam2", "Switch engine → SAM 2", keywords="cellpose zstack")
    assert search([cmd], "cellpose") == [cmd]


# ── group_by_section() ──────────────────────────────────────────────────────
def test_group_by_section_preserves_first_seen_section_order():
    cmds = [
        _cmd("a", "Run segmentation", section="Segment"),
        _cmd("b", "Go home", section="Navigate"),
        _cmd("c", "Export CSV", section="Segment"),
        _cmd("d", "Go to Logs", section="Navigate"),
    ]
    grouped = group_by_section(cmds)
    assert [section for section, _ in grouped] == ["Segment", "Navigate"]
    assert [c.id for c in grouped[0][1]] == ["a", "c"]
    assert [c.id for c in grouped[1][1]] == ["b", "d"]


def test_group_by_section_on_empty_list():
    assert group_by_section([]) == []


# ── Command defaults ─────────────────────────────────────────────────────────
def test_command_defaults_to_enabled_and_a_no_op_handler():
    cmd = Command(id="x", label="X", section="Actions")
    assert cmd.enabled is True
    cmd.handler()  # must not raise
