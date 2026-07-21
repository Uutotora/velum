"""Pure-logic tests for the Guide & Docs content (studio/guide_content.py).

No Qt import here on purpose — this module is content data, so it runs in
CI's light `test` dependency-group same as any other pure-logic test.
"""
from studio import guide_content as gc

_KNOWN_NAV_KEYS = {"home", "projects", "workspace", "train", "dashboard", "guide", "assistant"}
_KNOWN_SPECIAL_ACTIONS = {"new_project", "open_sample"}


def test_every_article_has_a_unique_id():
    ids = [a.id for a in gc.ARTICLES]
    assert len(ids) == len(set(ids))


def test_default_article_id_exists():
    assert gc.DEFAULT_ARTICLE_ID in gc.ARTICLES_BY_ID


def test_articles_by_id_matches_the_articles_list():
    assert gc.ARTICLES_BY_ID == {a.id: a for a in gc.ARTICLES}


def test_every_article_category_is_declared_in_categories():
    used = {a.category for a in gc.ARTICLES}
    assert used <= set(gc.CATEGORIES)


def test_every_article_has_a_summary_and_at_least_one_block():
    for article in gc.ARTICLES:
        assert article.summary.strip()
        assert article.blocks


def test_getting_started_has_several_actionable_steps():
    article = gc.ARTICLES_BY_ID["getting-started"]
    step_blocks = [b for b in article.blocks if b[0] == "steps"]
    assert len(step_blocks) == 1
    steps = step_blocks[0][1]
    assert len(steps) >= 4
    assert all(isinstance(s, gc.Step) for s in steps)


def test_step_actions_only_reference_real_nav_keys_special_actions_or_real_articles():
    for article in gc.ARTICLES:
        for block in article.blocks:
            if block[0] != "steps":
                continue
            for step in block[1]:
                if step.action is None:
                    continue
                if step.action.startswith("article:"):
                    target = step.action.split(":", 1)[1]
                    assert target in gc.ARTICLES_BY_ID, f"{article.id}: dangling article link {step.action!r}"
                else:
                    assert step.action in _KNOWN_NAV_KEYS | _KNOWN_SPECIAL_ACTIONS, (
                        f"{article.id}: unknown step action {step.action!r}")


def test_shortcuts_cover_the_four_real_key_bindings():
    # studio/app.py wires exactly: Ctrl+K/Meta+K -> palette, Ctrl+T/Meta+T ->
    # Assistant, Ctrl+L/Meta+L -> Logs, Escape -> close. Every addition there
    # should get a matching row here.
    all_keys = [k for sc in gc.SHORTCUTS for k in sc.keys]
    assert any("K" in k for k in all_keys)
    assert any("T" in k for k in all_keys)
    assert any("L" in k for k in all_keys)
    assert any("Esc" in k for k in all_keys)
    assert len(gc.SHORTCUTS) == 4


def test_faq_has_several_entries_with_real_content():
    assert len(gc.FAQ) >= 4
    for item in gc.FAQ:
        assert item.q.strip().endswith("?")
        assert item.a.strip()


def test_assistant_article_exists_and_names_all_three_real_backends():
    # The Assistant is wired now (docs/velum/BACKLOG.md, 2026-07-18) — it gets
    # a real article, and that article must actually match the three
    # backends studio/assistant_controller.py implements (BACKENDS), not an
    # aspirational subset.
    from studio.assistant_controller import BACKENDS
    assert "assistant" in gc.ARTICLES_BY_ID
    article = gc.ARTICLES_BY_ID["assistant"]
    text = " ".join(
        b[1] if isinstance(b[1], str) else " ".join(b[1]) if b[0] == "ul" else ""
        for b in article.blocks
    ).lower()
    assert "offline" in BACKENDS and "ollama" in BACKENDS and "custom" in BACKENDS
    assert "offline" in text
    assert "ollama" in text
    assert "custom api" in text


def test_faq_data_privacy_answer_carves_out_the_custom_api_exception():
    # FAQ used to say data never leaves the device, full stop — now that
    # Custom API can point at a remote endpoint, that claim must be
    # qualified, not silently wrong.
    item = next(f for f in gc.FAQ if "leave this computer" in f.q.lower())
    assert "custom api" in item.a.lower()


def test_engines_article_only_names_the_three_real_engine_keys():
    article = gc.ARTICLES_BY_ID["engines"]
    table_blocks = [b for b in article.blocks if b[0] == "table"]
    assert table_blocks
    _, headers, rows = table_blocks[0]
    engine_names = {row[0] for row in rows}
    assert engine_names == {"CellSeg1 · LoRA", "Cellpose-SAM", "SAM 2"}
