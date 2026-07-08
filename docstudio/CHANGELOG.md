# Changelog — CellSeg1 Studio

What actually shipped in Studio, dated, newest first. (The repo-wide log is
`docs/CHANGELOG.md`; this one is Studio-specific.)

---

## 2026-07-08 — Projects tab wired end to end (first real tab, skeleton → functional)

The Projects tab is no longer demo cards — the first tab taken from
`BACKLOG.md`'s "how to wire a tab" recipe, end to end:

- **`studio/project.py` reintroduced** (the `Project`/`ProjectSettings`/
  `ProjectStats`/`ProjectStore` data model, pure stdlib, previously removed in
  the design-skeleton reset and preserved in git history) — adapted with
  `ENGINE_LABELS`/`ENGINE_KIND` display-mapping constants (now the single
  source every screen draws engine colour/label from) and a `touch=` escape
  hatch on `ProjectStore.save` for callers that need explicit, deterministic
  timestamps instead of "now".
- **New `studio/project_controller.py`** — a Qt-free `ProjectController`
  (mirrors `napari_app/core/predict_controller.py`'s shape): search/filter,
  favourites, the "active project" shared with the Workspace tab, and
  first-run sample seeding so a fresh install still shows the same 6 sample
  projects the mockup always had — now real, persisted `Project` records
  instead of hard-coded `demo` content. Small pure formatting helpers
  (`to_card`, `format_count`, `relative_time`, `cover_seed`) keep screens.py
  free of formatting/date logic.
- **Home + Projects screens bound to the controller**: `demo.PROJECTS` /
  `demo.RECENT_WHEN` reads replaced with live data; the search box and a new
  "favourites only" toggle on the existing "Filter" button live-filter the
  grid; a favourite star (new, on each card — the data model always had
  `favorite`, the static skeleton just never rendered an affordance for it)
  toggles and persists through the store; open callbacks switched from
  list-index to project-id (index broke once filtering could reorder/drop
  cards); the page header's counts are now computed from real data.
- **"Active project" shared to the Workspace tab**: `WorkspaceScreen.
  set_active_project()` updates the top-bar breadcrumb + engine chip — no
  longer hardcoded to "Fluorescence Nuclei — DAPI" / "CellSeg1 · LoRA"
  regardless of what you actually opened, and shows a neutral "No project
  selected" state before any project is opened. Survives the theme-toggle
  rebuild. The rest of the Workspace (layers, canvas, predict) is still the
  Segment tab's own, separate, not-yet-started backlog item.
- Cover art seeds are now derived deterministically from each project's id
  (`zlib.crc32`) rather than the arbitrary integers `demo.py` used, since a
  real project has no "seed" field to persist — same procedural nuclei-art
  look, stable per project across relaunches, just no longer pinned to the
  exact noise pattern the static mockup happened to show.
- **Not wired here** (separate BACKLOG items): the "+ New Project" creation
  dialog itself — cards/ghost-card still just navigate to a blank workspace,
  no create-through-the-store flow yet.

Verified: `studio/tests` green (87 tests: reintroduced + extended the historic
pure-logic `project.py` suite, added a pure-logic `project_controller.py`
suite, extended `test_app_wiring.py`'s screen/window tests to inject a
`tmp_path`-backed controller — real `data_store/projects` is never touched by
tests); the repo-root throwaway-venv check (`pip install --group test` only,
Python 3.11, no torch/napari/PyQt6) collects and passes both `project.py` and
`project_controller.py`'s suites for real (64 passed, `test_app_wiring.py`
correctly skips as one unit via `importorskip("PyQt6")`); an offscreen
end-to-end smoke run against the real default store (`data_store/projects`,
which already has 4 real local projects from earlier manual testing) —
construct, list, navigate to Projects, open a project, confirm the workspace
breadcrumb and active-project state — all passed with neither napari nor
torch imported. Not verified here (no display): the live look/animations of
the new favourite star and filter-toggle states.

## 2026-07-08 — Studio is now its own top-level project + docs pivot to "own canvas"

Structural + directional clarity, no behaviour change:

- **Studio promoted to a top-level `studio/` package** (`git mv` from
  `napari_app/studio/`, history preserved), a **sibling** of the classic
  `napari_app/` (old app) and the shared ML core — the standard monorepo
  "old app + new app + shared core" shape, so the branch reads as its own
  project. Studio is now **self-contained**: its own `icons.py` (the mockup's
  icons, not the classic app's) and `motion.py`; it imports nothing from
  `napari_app`. The classic `napari_app/icons.py` was reverted to pristine.
- **Studio has its own test suite** in `studio/tests/` (run `pytest
  studio/tests`); `pytest.ini` includes it; packaging/entry point updated
  (`cellseg1-studio = studio.app:main`, `studio/` packaged, tests excluded).
- **Docs pivot — we are NOT embedding napari.** The Segment tab will get our
  **own** canvas (like Label Studio's / napari's viewers, but ours: own tool
  strip, own layer model, own interactions), reusing only the **ML logic**
  (engines/predict/morphometry). New guiding principle across the docs: *own
  the UI, the icons, the canvas, the settings; reuse the logic.* Label Studio
  reaffirmed as the primary **structure** reference (not look). AGENT_PROMPT
  gained explicit git-sync (keep local↔remote in sync) and "run only
  `studio/tests`" guidance.

Verified: full suite 473 passed; Studio's suite green from its new location;
the app imports and boots from the top-level `studio` package offscreen,
importing neither napari nor torch.

## 2026-07-07 — Design skeleton: the mockup, reproduced natively (no logic)

Reset Studio to a pure **design skeleton** — a faithful, static, native-Qt
reproduction of the north-star mockup with **all business logic removed** — so
there's a clean, consistent target to wire functionality against, tab by tab.

- **Stripped all logic** from the running app: no napari, no torch, no model,
  no project/file IO. `import napari` / `import torch` never runs; the app
  launches on PyQt6 alone. Removed the wired-in `PredictWidget`/`TrainWidget`
  hosting and the `project.py` data model (preserved in git history; returns
  when the Projects tab is wired).
- **Native reproduction of every mockup screen** with static demo content
  (`demo.py`): Home, Projects, the Segment workspace (adapted-napari
  **Images|Layers** panel with full layer controls · nuclei canvas · **Segment|
  Results** inspector), Models & Train, Dashboard — plus overlays: Assistant
  drawer, Logs console, ⌘K command palette, toast.
- **Design system** as reusable modules: `theme.py` (tokens), `components.py`
  (the UI-kit atoms + sidebar), `paint.py` (a QPainter nuclei stand-in for the
  canvas / card covers / thumbnails).
- **Rounded window corners** (12px rounded mask) on the frameless window.
- **`docstudio/`** — this doc set (OVERVIEW, DESIGN, ARCHITECTURE, BACKLOG,
  ROADMAP, CHANGELOG, AGENT_PROMPT) driving the tab-by-tab plan.

Verified: full pure-logic suite green; the app boots offscreen and navigates
every screen, opens all overlays, toggles theme and resizes cleanly, importing
**neither napari nor torch**. Not verified here (no display): the live look,
the rounded corners (offscreen can't set window masks — real macOS can), fades.

### Earlier (foundation, superseded by the reset above)
- Frameless window + own dark title bar (own traffic lights, native
  move/resize via `startSystemMove` + grips) replacing the grey OS title bar.
- First shell: sidebar + Home/Projects backed by a `ProjectStore`, embedding
  the classic `PredictWidget`. Reset to a logic-free skeleton on the same day.
