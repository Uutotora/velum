# Architecture — Velum

## Module map (`studio/`)

```
app.py            StudioWindow(QMainWindow) + main(). Frameless rounded window,
                  title bar, sidebar, screen stack, overlays, ⌘K, theme toggle.
                  THE entry point. Pure design — imports no napari/torch.
window_chrome.py  TitleBar (own traffic lights, native move) + corner grips.
theme.py          Design tokens (light+dark) + QSS builders + viridis ramp. Pure.
components.py     The static UI kit (atoms) + the navigation Sidebar.
paint.py          QPainter "nuclei" stand-in art (canvas, card covers, thumbs).
demo.py           Static demo content for tabs not yet wired. No logic.
project.py        Project/ProjectSettings/ProjectStats/ProjectStore — the real,
                  persisted data model (pure stdlib). The Projects tab's data layer.
project_controller.py  ProjectController (Qt-free): search/filter, favourites,
                  the active project shared with the Workspace tab, sample
                  seeding. Home/Projects screens are bound to it.
hardware.py       detect() -> DeviceInfo for Home's "This device" card: real
                  CUDA/MPS/CPU, not a hard-coded platform string. Lazy
                  `import torch` inside the function; the actual "is this
                  GPU usable" check delegates to the repo-root
                  device_utils.py (shared with the classic app).
screens.py        HomeScreen, ProjectsScreen — bound to project_controller
                  (+ shared page_header/scroll helpers, the latter now
                  components.SmoothScrollArea — see components.py). Live
                  data, not demo. Home's quick cards/resource links are all
                  real actions now (New Project dialog, navigate, external
                  open, "Open Sample"). Cards/rows are plain text — no
                  cover-art thumbnail, matching Label Studio's own reference
                  cards — with a real ⋯ overflow menu (Open/Duplicate/
                  Settings — see project_dialogs.py) and a Sort control
                  (project_controller.ProjectController.SORT_OPTIONS).
new_project_dialog.py  NewProjectDialog — the "+ New Project" modal (scrim +
                  centred panel, same construction as overlays.CommandPalette):
                  name+description → import (drag-drop or a file picker) →
                  engine, writing through ProjectStore.create() on finish.
project_dialogs.py  ConfirmDialog (generic scrim+panel confirm — used for
                  "Delete Project?" via the confirm_delete_project builder)
                  + ProjectSettingsDialog (General — editable name/
                  description — and a Danger Zone card, Delete Project ->
                  its own nested ConfirmDialog; mirrors Label Studio's own
                  Settings/Danger Zone) — the Projects tab's small modals,
                  split out once ProjectsScreen itself got too large for
                  them to live inline. An earlier RenameDialog + TrashDialog
                  (soft-delete with Restore/Undo) were built, then reverted
                  the same day after real (non-offscreen) usage found both a
                  rendering bug and that the product didn't want the extra
                  machinery — see docs/velum/CHANGELOG.md's dated entry.
layer_model.py    Our own evented layer model (Layer/ImageLayer/LabelsLayer/
                  PointsLayer/ShapesLayer/LayerList) — napari-Labels-faithful
                  properties/defaults, plain-callback events (no Qt/psygnal),
                  Qt-free so it stays in the light CI test group.
canvas.py         Canvas(QWidget) — the Segment workspace's own image
                  viewport (NOT embedded napari): numpy compositing of a
                  LayerList (image contrast/gamma/colormap, labels colour/
                  opacity/contour, translucent/additive/opaque blending)
                  under a QPainter pan/zoom transform; paint/erase/fill/
                  pick/polygon editing; Points/Shapes click-to-add/draw;
                  grid mode, a 2D/3D (max-intensity-projection) toggle,
                  channel-roll, view-only transpose.
segment_controller.py  SegmentController (Qt-free): maps ProjectSettings to
                  napari_app.core.predict_controller's params/config and
                  reuses PredictController.run_prediction_async/
                  run_batch_async/run_benchmark_async unmodified, plus
                  analysis/benchmark/cohort wrappers. record_run() mutates
                  project.stats (caller saves) — the Dashboard-visibility hook.
workspace.py      WorkspaceScreen — the signature Segment screen (Images|
                  Layers panel · canvas · Segment|Results inspector). Real:
                  bound to segment_controller + project_controller — Images/
                  Layers/layer-controls/Segment-settings/Run/Results (incl.
                  GT & evaluation/batch/benchmark) all live, not demo. With
                  no active project, the three-panel body is swapped out
                  entirely for a full-screen "no project" view (_body_stack,
                  a QStackedWidget) rather than layered under it — see
                  BACKLOG.md's "Home motion polish + Segment's 'no project'
                  state" entry.
extra_screens.py  ModelsScreen (train), DashboardScreen (charts + runs table).
assistant_controller.py  AssistantController (Qt-free): settings
                  (AssistantSettings/AssistantSettingsStore, one small JSON
                  file) + backend-dispatching send_async() across three
                  interchangeable chat backends — offline (the deterministic
                  diagnostic engine, napari_app.advisor.diagnose, reused
                  read-only), Ollama (napari_app.advisor's existing bridge,
                  reused verbatim), and Custom API (this module's OWN
                  OpenAI-compatible urllib+SSE bridge — new capability, not
                  reused from napari_app, since the classic app has no
                  bring-your-own-endpoint story).
assistant_panel.py  Studio's own chat UI (own Qt, no import of
                  napari_app.widgets.chat/assistant_widget): ChatView
                  (bubbles/streaming/typing-indicator/empty-state),
                  ChangeCard (Apply/Apply & re-run), and AssistantDrawer
                  itself — header + a collapsed "Model" settings accordion
                  (backend picker/per-backend fields/live status/Ollama
                  catalogue) + the chat + Diagnose/input/Send. Talks to the
                  Segment WorkspaceScreen only through its narrow
                  assistant_context()/apply_assistant_changes()/
                  rerun_predict() hook (see workspace.py's "Assistant
                  integration" section) — the cross-tab wiring.
log_bus.py        LogBus (a bounded, thread-safe ring buffer of LogRecord)
                  + StudioLogHandler, a real stdlib logging.Handler bridge
                  — the Studio-wide log stream every tab's real operational
                  log lines (segment/train/assistant/app) feed and
                  overlays.LogsConsole reads live. Qt-free (stdlib only).
command_registry.py  Command (label/section/icon/hint/keywords/handler/
                  enabled) + a real Sublime/VS-Code-style fuzzy matcher
                  (a contiguous substring always outranks a scattered
                  subsequence match — two score bands, not one flat
                  heuristic) + group_by_section for the empty-query
                  browsing view. Qt-free (stdlib only) — the palette's
                  *content* (which commands exist, whether each is enabled
                  right now) is built by app.py's StudioWindow.
                  _build_commands(), which has the real controller/screen
                  references this module deliberately doesn't.
overlays.py       LogsConsole (real, live — see log_bus.py above: level
                  filter, text search, autoscroll, clear, export),
                  CommandPalette (real, live — see command_registry.py
                  above: fuzzy search, full keyboard navigation, disabled/
                  dim rows, click-to-run, a bounded scrollable results
                  list), Toast (a real announce() used by project creation,
                  not just static). The Assistant drawer used to live here
                  too — it outgrew this file and moved to
                  assistant_panel.py above.
icons.py          Studio's OWN icon set (from the mockup) — self-contained.
motion.py         Small motion helpers: fade_in (screen switches),
                  install_hover_lift (animated shadow "elevation" on hover —
                  QSS has no transform/transition, so this animates a
                  QGraphicsDropShadowEffect instead). Self-contained.
fonts/            Figtree (SIL OFL), registered at startup.
assets/           icon.png — the app icon (Dock tile), set via
                  QApplication.setWindowIcon() in app.py's main().
tests/            Studio's own test suite (run `pytest studio/tests`).
```

`studio/` is a **self-contained** top-level package — a sibling of the classic
`napari_app/` (old app) and the shared ML-core modules. It has its own icons and
motion and imports nothing from `napari_app` (the ML core is pulled in lazily,
only inside a tab being wired). Import direction is one-way, leaf → shell:
`theme`/`icons` ← `components`/`paint` ← `demo`/screens ← `app`.

## Entry points

- **New:** `run_studio.sh` / `cellseg1-studio` → `studio.app:main`.
  Runs the file directly and self-bootstraps `sys.path` (works from any cwd —
  `python -m` would prepend the caller's cwd and import the wrong `napari_app`).
- **Classic (untouched):** `run_napari.sh` / `cellseg1` →
  `napari_app.main:main`. The proven, fully-functional app.

## Why the skeleton is logic-free

`app.py` and every shared module import only PyQt6 + our own leaves. That keeps
the app light (launches with no torch/napari/GPU), keeps the pure-logic tests
runnable in CI's light `test` group, and keeps the design a stable target.
Real dependencies get imported **lazily, inside the tab being wired** — never
at a shared module's top level.

## How to wire a tab (the core workflow)

Each tab goes from *static* to *functional* without changing how it looks.
General recipe:

1. **Re-introduce the data it needs.** If it was removed in the skeleton reset,
   it's in git history (`git log -- studio/<name>.py`) — reintroduce and adapt.
   Worked example: `studio/project.py` (the `Project`/`ProjectStore` model),
   restored for the Projects tab.
2. **Add a controller**, Qt-free where possible, that owns the logic and takes
   plain callbacks — mirror `napari_app/core/predict_controller.py`. Unit-test
   it without Qt. Worked example: `studio/project_controller.py`.
3. **Bind the screen to the controller.** Replace the screen's `demo.*` reads
   with live data; connect its buttons/toggles/sliders to controller calls;
   feed results back into the same widgets. **Do not restyle** — reuse the
   existing atoms.
4. **Lazily import heavy deps** (torch, engines, ML-core modules — **not**
   napari; we build our own canvas) inside the controller / handlers, never at
   a shared module's top level.
5. **Test:** pure controller logic in the light group; screen wiring
   headless (`pytest.importorskip("PyQt6")`, offscreen). Note GUI/GPU parts as
   not-verified-here.
6. **Ship** per the repo's branch → PR → green-CI workflow; log it in this
   folder's `CHANGELOG.md` and tick the tab in `BACKLOG.md`.

### The Segment tab specifically — our OWN canvas, not embedded napari

**We are not embedding napari.** Studio gets its **own** image canvas — like
Label Studio's and napari's viewers, but ours — so we own the look, the tool
strip, the layer model and every interaction (that's why the mockup's canvas
toolbar was redrawn from scratch). We reuse the *interaction patterns*
(pan/zoom, layers, brush/polygon/point editing, 2D↔3D, grid) and, above all,
the **segmentation logic** (engines, predict, morphometry) — we do not
reimplement the ML, and we do not reimplement the UI napari-style either.

**Done (2026-07-09)** — see `BACKLOG.md`'s Segment tab entry and this same
date's `CHANGELOG.md` entry for the full detail:

- `Canvas` (`studio/canvas.py`) is a plain `QWidget` + `QPainter` (not
  `QGraphicsView`/GPU — that door is still open if performance ever demands
  it) that composites image + labels/shapes/points layers with pan/zoom,
  replacing the `NucleiView` stand-in. Owns the viewer bar (2D↔3D as a
  max-intensity projection, grid as one tile per visible layer, home) + the
  floating tool strip, both now re-styling themselves to match live state.
- `studio/layer_model.py` is our **own layer model** (an evented
  `LayerList` of image/labels/shapes/points layers) that the Layers panel
  drives — visibility, opacity, new-layer, delete, colours, plus real
  paint/erase/fill/pick/polygon editing on `LabelsLayer`. Interaction model
  *faithful to* napari's `Labels` layer (verified against its installed
  source); code entirely ours.
- Segment settings + Run are wired to `SegmentController`
  (`studio/segment_controller.py`), which **reuses the ML core** —
  `napari_app/core/predict_controller.py`, `napari_app/engines*`,
  morphometry in `napari_app/analysis.py` — imported lazily, unmodified.
  Results (stats, calibration, save/export, colour-by heatmap, GT & eval,
  batch, benchmark) and the toast render into the existing widgets.

The principle for every tab: **own the UI, the icons, the canvas, the settings;
reuse the logic.** We wrap the classic app's proven functionality under the new
design instead of rewriting it — and we build our own viewer instead of
embedding napari's.

## Testing conventions

Studio has its **own** suite in `studio/tests/`. When working on Studio, run
just those (not the classic app's `tests/`):

```
QT_QPA_PLATFORM=offscreen <python> -m pytest studio/tests -q
```

- Pure logic → no Qt import, runs in CI's light `test` group.
- Qt screens → offscreen construct/smoke with `pytest.importorskip("PyQt6")`.
- Before committing, run the throwaway-venv light-group check from the repo
  `AGENTS.md` so nothing heavy leaks into CI.
