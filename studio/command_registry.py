"""CellSeg1 Studio — the ⌘K command palette's action registry + fuzzy search.

Qt-free (pure stdlib) so it stays importable in CI's light `test` group and
is unit-testable without a display: a `Command` is just a label/section/icon/
hint plus a plain callable, and `search()` is ordinary string scoring. The
palette's *content* (which commands exist, whether each is enabled right
now) is built by ``studio.app.StudioWindow._build_commands()``, which has
the real controller/screen references this module deliberately has no
knowledge of — the same "Qt-free logic, Qt-only shell" split every other
controller in this package already follows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class Command:
    id: str
    label: str
    section: str
    icon: str = "run"
    emoji: str = ""                 # a Raycast-style leading glyph; falls back to `icon` if empty
    hint: str = ""
    keywords: str = ""              # extra searchable text, never displayed
    handler: Callable[[], None] = field(default=lambda: None)
    enabled: bool = True


_WORD_BOUNDARY = " -/→_·|()&"


def _match_score(query: str, text: str) -> Optional[float]:
    """Fuzzy match, Sublime-Text/VS-Code style: a real contiguous substring
    always outranks a scattered subsequence, which is itself still a real
    (weaker) match — not a flat subsequence-only score, which would let a
    query like "seg" rank "Switch engine" (s...e...g scattered across three
    words) *above* "Run segmentation" (where "seg" is a literal substring)
    just because the former happens to hit more word-boundary starts. Two
    separate bands fix that: contiguous matches score in the thousands,
    scattered ones in the tens, so band alone always decides ties across
    match kinds.

    Returns ``None`` when ``query`` doesn't even subsequence-match ``text``;
    otherwise a score where higher is better.
    """
    if not query:
        return 0.0
    q = query.lower()
    t = text.lower()

    idx = t.find(q)
    if idx != -1:
        # A plain contiguous substring -- the common case for a short query
        # ("seg", "csv", "run") and always the most relevant kind of match.
        score = 1000.0
        if idx == 0:
            score += 200.0                    # a prefix of the whole label
        elif t[idx - 1] in _WORD_BOUNDARY:
            score += 100.0                     # starts a word mid-label
        score -= idx * 0.5                     # earlier in the label is better
        score -= len(t) * 0.1                  # a tighter/shorter label is better
        return score

    # No contiguous run at all -- fall back to a scattered subsequence match
    # (every character of query still appears in order): rewards internal
    # contiguous runs and word-boundary starts, lightly penalises gaps.
    search_from = 0
    score = 0.0
    consecutive = 0
    for ch in q:
        pos = t.find(ch, search_from)
        if pos == -1:
            return None
        if pos == search_from:
            consecutive += 1
            score += 5 + consecutive * 2
        else:
            consecutive = 0
            gap = pos - search_from
            score += 3 - min(gap, 8) * 0.3
        if pos == 0 or t[pos - 1] in _WORD_BOUNDARY:
            score += 4
        search_from = pos + 1
    score -= len(t) * 0.05
    return score


def search(commands: Sequence[Command], query: str) -> list[Command]:
    """Filter+rank ``commands`` against ``query``.

    Empty query: return ``commands`` as given (the caller already ordered
    them section-by-section for browsing). Non-empty query: a flat,
    score-ranked list with no section grouping — real command palettes
    (VS Code, Spotlight) drop section headers the moment you start typing,
    since ranking across sections is exactly the point of searching.
    A command matches via its visible ``label`` *or* its hidden
    ``keywords`` — a keywords-only match ranks slightly below an equal
    label match, so "Cellpose" (a keyword on the SAM-2 switch-engine
    command, say) still surfaces it, just behind anything matching by name.
    """
    query = query.strip()
    if not query:
        return list(commands)
    scored: list[tuple[float, int, Command]] = []
    for order, cmd in enumerate(commands):
        best = _match_score(query, cmd.label)
        if cmd.keywords:
            kw_score = _match_score(query, cmd.keywords)
            if kw_score is not None:
                kw_score -= 5
                best = kw_score if best is None else max(best, kw_score)
        if best is not None:
            scored.append((best, order, cmd))
    scored.sort(key=lambda triple: (-triple[0], triple[1]))
    return [cmd for _, _, cmd in scored]


def group_by_section(commands: Sequence[Command]) -> list[tuple[str, list[Command]]]:
    """Group ``commands`` into ``(section, [commands...])`` pairs, in
    first-seen section order — the shape the palette's empty-query browsing
    view renders (a heading per section, matching the mockup's "ACTIONS" /
    "EXPORT" caps labels)."""
    order: list[str] = []
    buckets: dict[str, list[Command]] = {}
    for cmd in commands:
        if cmd.section not in buckets:
            buckets[cmd.section] = []
            order.append(cmd.section)
        buckets[cmd.section].append(cmd)
    return [(section, buckets[section]) for section in order]
