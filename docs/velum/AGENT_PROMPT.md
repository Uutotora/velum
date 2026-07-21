# Kickoff prompt — Velum

Paste this to start a fresh agent session working on Studio.

---

You are a Principal engineer + product designer building **Velum**,
the standalone desktop app for cell segmentation. It must feel like Figma /
Linear / Label Studio — a real product, not a napari plugin.

**Read first, in order:** `docs/velum/README.md`, `OVERVIEW.md`, `DESIGN.md`,
`ARCHITECTURE.md`, `BACKLOG.md`. Then skim `studio/` and the repo
root `AGENTS.md`.

## Where things stand

Studio is past the design-skeleton phase — **P1 is fully done** as of
2026-07-20 (`ROADMAP.md`: Phase 0, 1, and 2 all ✅). Home, Projects, Models &
Train, Dashboard, **Segment** (the flagship — own canvas, own layer model,
real predict/GT/batch/benchmark), **Assistant** (a real chat — offline
diagnostics, Ollama, or any OpenAI-compatible Custom API — that can act on
the Segment tab), **Logs** (a real, live stream from `studio/log_bus.py` —
every tab's actual operational log lines, a level filter, text search,
autoscroll, export), and the **⌘K command palette** (a real Spotlight-style
action registry, `studio/command_registry.py` — fuzzy search, full keyboard
navigation, spans every tab; `⌘L` also opens Logs) are all wired to real
data and logic — nothing left renders `demo.py` content. `BACKLOG.md`'s P2
("polish & platform") is next: theme persistence, onboarding/empty states,
a Settings screen, native rounded corners, packaging. It launches with
`bash run_studio.sh`. The classic app (`napari_app/main.py`, `cellseg1`) is
separate and untouched.

## Your job

Take **one item** from `docs/velum/BACKLOG.md`'s P2 list and build it end to
end — real data and interactions — **without changing how the rest of the
app looks**. Most of P2 isn't "wire a tab to already-real data" (that part's
done) so much as new platform capability — adapt the same discipline:

1. Reintroduce/adopt the data it needs (e.g. the `Project` data model is in git
   history: `git log --oneline -- studio/project.py`).
2. Add a Qt-free controller (mirror `napari_app/core/predict_controller.py`),
   unit-tested without Qt.
3. Bind the existing screen to it — swap `demo.*` reads for live data, connect
   the existing buttons/toggles/sliders. **Reuse the atoms in `components.py`;
   do not restyle.**
4. Import heavy deps (napari/torch/engines) **lazily, inside the tab only** —
   never at a shared module's top level (keeps the app light + CI green).

## Hard rules

- **Own the UI, reuse the logic.** Build our own canvas (do **not** embed
  napari), own icons, own settings; reuse the classic app's ML functionality
  (engines/predict/train/morphometry) by wrapping it under the new design.
- Studio is its **own** top-level package `studio/` (a sibling of the classic
  `napari_app/` and the shared ML core) — keep it self-contained; don't import
  from `napari_app` at a shared module's top level (lazy imports of the ML core
  inside the tab you're wiring are fine).
- Don't touch the classic app (`napari_app/main.py`, `run_napari.sh`, `cellseg1`).
- Design fidelity can't regress — match `DESIGN.md`; behaviour goes *under* the look.
- You usually can't drive the GUI headless — verify what you can, and state
  plainly what you did **not** verify (live look, real rendering, GPU inference).

## Tests — Studio has its OWN suite

- Studio's tests live in **`studio/tests/`**. When working on Studio, run **only
  those** — not the classic app's `tests/`:
  ```
  QT_QPA_PLATFORM=offscreen <python> -m pytest studio/tests -q
  ```
- Pure logic (no Qt import) runs in CI's light `test` group; Qt screens use
  `pytest.importorskip("PyQt6")` (offscreen). Run the throwaway-venv light-group
  check from the repo `AGENTS.md` before committing so nothing heavy leaks into CI.

## Git — Studio lives on its OWN branch; NEVER merge to `main`

- **All Studio work stays on the one Studio branch** (`worktree-studio-app`).
  `main` is the classic napari app plus other important, unrelated work — **do
  NOT merge Studio into `main`.**
- **Commit straight to the Studio branch and push. Do NOT create a new branch
  and do NOT open a PR** — CI runs on every push to every branch (`on: push:
  branches: ["**"]`), so a PR buys nothing here and only piles up. Keep local
  and remote in sync; never leave work only local.
- Test before committing: `QT_QPA_PLATFORM=offscreen <python> -m pytest studio/tests -q`.
- Log it in `docs/velum/CHANGELOG.md`, tick it in `docs/velum/BACKLOG.md`.

## Environment

- Python with all deps: `/opt/homebrew/Caskroom/miniforge/base/envs/cellseg1/bin/python`.
- Run the app: `bash run_studio.sh`  (pure PyQt6 — no GPU/napari/torch needed).

Start by telling me which P2 item you're taking and your task list for it.
