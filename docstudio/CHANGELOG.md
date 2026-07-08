# Changelog — CellSeg1 Studio

What actually shipped in Studio, dated, newest first. (The repo-wide log is
`docs/CHANGELOG.md`; this one is Studio-specific.)

---

## 2026-07-08 — Fix: a newly created project didn't show up until restart

User-reported: creating a project via the New Project modal didn't add it to
Home's "Recent projects" or the Projects grid until the whole app was
relaunched. Root cause: `HomeScreen`/`ProjectsScreen` are built once and kept
alive across navigation (`StudioWindow.navigate()` just swaps the visible
`QStackedWidget` page) — so their content reflected whatever the store looked
like at construction time, and nothing ever told them to rebuild afterwards.

- `HomeScreen.refresh()` / `ProjectsScreen.refresh()` (new) rebuild the
  recent-projects list / the grid + header counts from the store's current
  state. `StudioWindow.navigate()` now calls a screen's `refresh()` (if it
  has one) every time it becomes the active page — so switching to Home or
  Projects always shows current data, not just right after a create.
  `ProjectsScreen.refresh()` preserves whatever search/favourites-only
  filter was already active.
- Also fixed while verifying this: `Toast`'s subtitle could get clipped
  instead of wrapping for a long project name + engine combination
  (`setWordWrap` + a max width, rather than relying on `adjustSize()` timing
  after a dynamic `setText()`).

Verified: reproduced the exact bug first (constructed the window, created a
project, counted rendered project-card/row widgets before and after — stayed
at 0 after create+navigate without the fix, confirming the root cause), then
confirmed the fix with the same reproduction. Three new regression tests
(`studio/tests`, now 118 total, all green): `HomeScreen`/`ProjectsScreen`
picking up a project created directly through the store, and a full
app-level end-to-end test creating a project through the real dialog and
navigating to both Home and Projects. Repo-root throwaway-venv check (`pip
install --group test` only) still green. Offscreen-screenshot-reconfirmed
(`QWidget.grab()`): the new project now appears immediately in Home's recent
list after creation, and the toast wraps correctly for a long name.

## 2026-07-08 — Home screen: every element real, + the New Project modal

Follow-up pass focused entirely on Home (Projects tab intentionally left
alone this round):

- **New `studio/new_project_dialog.py`** — the "+ New Project" flow ticked
  off `BACKLOG.md`'s own item: a scrim-backed modal (identical construction
  to `overlays.CommandPalette` — no native `QDialog` frame, stays consistent
  with the app owning its own chrome) with the 3-step Label Studio pattern:
  name + description → import images (a real drag-and-drop zone plus a
  native file picker, both funnelling into the same add/remove-file state) →
  engine (`SegControl` over the same three engines everywhere else in the
  app). "Create Project" writes through the real `ProjectStore.create()` and
  opens straight into the workspace, reusing the Projects tab's existing
  active-project flow.
- **Every Home element is now a real action**, not just the Projects grid:
  the "New Project" CTA and quick card, and the "Import Images" quick card,
  open the new dialog; "Train a Model" navigates to Models & Train; "Open
  Sample" opens an existing project if one exists, or opens the dialog when
  the store is empty; "Ask the Assistant" opens the Assistant drawer;
  "Documentation"/"Getting started guide" open real local docs
  (`README.md` / `docstudio/OVERVIEW.md`) and "GitHub" opens the real origin
  remote (read from `git remote get-url origin` at runtime, converted to an
  `https://` URL — never a hard-coded/guessed link, and it degrades to a
  no-op if there's no remote).
- **`Toast.announce()`** — the bottom-right success toast existed since the
  design-skeleton phase but nothing had ever called `.show()` on it; project
  creation is its first real trigger ("Project created · <name> · N images ·
  engine"), auto-hiding on a timer.
- **Hover "lift"** on Home's quick cards and recent-project rows, matching
  the north-star mockup's `.qcard:hover`/`.rrow:hover` CSS
  (`transform:translateY()` + a deeper shadow, ~160ms). QSS has no
  `transform`/`transition`, so `motion.install_hover_lift()` animates a
  `QGraphicsDropShadowEffect`'s blur/offset instead — same "the card is
  rising toward you" read, without fighting Qt's layout engine.

Verified: `studio/tests` green, 115 tests (28 new: a `test_new_project_dialog.py`
covering the full step flow — validation, back/forward, persistence across
steps, file add/remove, and a real end-to-end create-through-the-store; a
`test_home_wiring.py` covering every quick-card/resource-link callback,
`QDesktopServices.openUrl` mocked rather than actually invoked so tests never
really open a browser). Two real bugs the new tests caught before shipping:
`_go_next()` relied entirely on the Next button being disabled to block an
empty project name (fixed with its own guard); a test asserting
`isVisible()` on a dialog button needed the test's own parent widget shown
first (`isHidden()` is the explicit per-widget flag; `isVisible()` needs the
whole ancestor chain actually shown — same distinction already called out in
`test_app_wiring.py`). Repo-root throwaway-venv check (`pip install --group
test` only) still passes, 380 passed / 14 skipped, confirming nothing in
this round leaked a heavy dependency into the light CI group.

Offscreen screenshot verification this round (`QWidget.grab()` under
`QT_QPA_PLATFORM=offscreen`, both themes, all 3 dialog steps, plus a hover
state settled via `QTest.qWait`): layout, spacing, data and the dialog flow
all matched the design intent. One rendering artifact showed up (thin outline
boxes around label text inside scrim-backed panels) — traced to a pre-existing
offscreen-QPA quirk by reproducing it identically in the untouched
`CommandPalette`, so it isn't a real bug and isn't expected on a real display;
not independently re-verified with a physical display. The Projects tab
(left untouched this round) also rendered correctly in these screenshots.

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
