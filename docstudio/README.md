<p align="center">
  <img src="../studio/assets/icon.png" width="120" alt="CellSeg1 Studio app icon">
</p>

# CellSeg1 Studio — documentation

Studio is the standalone desktop app for CellSeg1: a self-contained product
that owns its window (Home · Projects · Segment · Models & Train · Dashboard),
with its **own** image canvas (Segment's `studio/canvas.py` — explicitly
*not* embedded napari; see `ARCHITECTURE.md`). It replaces the "napari
plugin in a dock" experience.

This folder is Studio's own doc set — separate from the repo-wide `docs/`
(which covers the whole CellSeg1 project). Read in this order:

1. **[OVERVIEW.md](OVERVIEW.md)** — what Studio is, where it stands (Phase 0 +
   1 done, Phase 2 in progress — see `ROADMAP.md`), how to run it, and the
   ground rules.
2. **[DESIGN.md](DESIGN.md)** — the visual identity: palette, type, tokens,
   components. The single source of truth for how Studio should look.
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** — the module map and, crucially, **how
   to wire a tab** (turn a static screen into a real, functional one).
4. **[BACKLOG.md](BACKLOG.md)** — the tab-by-tab plan. Each tab is its own mini
   backlog with a task list. Pick a tab, do it end to end.
5. **[ROADMAP.md](ROADMAP.md)** — the phases from skeleton → full product.
6. **[CHANGELOG.md](CHANGELOG.md)** — what actually shipped, dated.
7. **[AGENT_PROMPT.md](AGENT_PROMPT.md)** — paste this to start a fresh agent
   session on Studio.

## TL;DR for a new contributor

- **Every screen from the mockup is reproduced in native Qt, and all of it is
  now real**: Home, Projects, Models & Train, Dashboard, Segment (own
  canvas + layer model, real predict/GT/batch/benchmark), Assistant (a
  real chat — offline diagnostics, Ollama, or any OpenAI-compatible Custom
  API — that can act on the Segment tab), Logs (a real, live stream from
  every tab's actual log lines — see `studio/log_bus.py`), and the ⌘K
  command palette (a real Spotlight-style action registry spanning every
  tab, fuzzy search, full keyboard navigation — see
  `studio/command_registry.py`) all run on live data/logic, not `demo.py`.
  P1 is fully done — check `BACKLOG.md`'s P2 ("polish & platform") for
  what's next.
- The goal now is **polish & platform** (P2): theme persistence, onboarding,
  a Settings screen, native rounded corners, packaging — tracked in
  `BACKLOG.md`.
- The **classic app is untouched** (`napari_app/main.py`, `run_napari.sh`, the
  `cellseg1` command) — launch it any time to use the fully-functional (if
  less polished) product.
- **Studio lives on its own branch and is never merged into `main`** (which
  holds the classic app + other critical work). All Studio work happens on the
  Studio branch.

## Run it

```bash
bash run_studio.sh          # or:  python -m studio.app
```

No GPU, no weights, no napari needed — it's design only.
