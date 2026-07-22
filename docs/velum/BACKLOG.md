# Backlog — Velum (tab by tab)

The plan to take Studio from **design skeleton** to **full product**, one tab
at a time. Each tab below is its own mini-backlog: a goal, the work, and a
task list. Do a tab **end to end** (data + interactions + tests) before moving
on, keeping the look pixel-stable. Legend: size S (hours) · M (day) · L (multi-day).

When you finish a tab: log it in `CHANGELOG.md`, tick it here, update
`ROADMAP.md` if a phase closed.

---

## ✅ Done (skeleton phase)

- [x] **Window chrome** — frameless, rounded corners, own dark title bar (own
  traffic lights, native move/resize), screen fade transitions.
- [x] **Design system** — tokens (light+dark), the `components.py` UI kit,
  `paint.py` nuclei art, `demo.py` static content.
- [x] **All screens reproduced natively (static)** — Home, Projects, Segment
  workspace (Images|Layers · canvas · Segment|Results), Models & Train,
  Dashboard, + overlays (Assistant drawer, Logs console, ⌘K palette, toast).

---

## P0 — make the shell truly usable

### Projects tab · M · ✅ done (2026-07-08, fidelity pass same day)
- **Goal:** real projects, not demo cards — and matching the mockup exactly.
- **Work:** reintroduced the `Project`/`ProjectStore` data model (in git history,
  `studio/project.py`); backed Home "recent" + Projects grid with it;
  wired search/favourite; card click opens that project in the workspace.
  Follow-up same day: the toolbar's "All · Favorites · Shared" scope control
  and the grid/list view toggle were rebuilt for real (see `CHANGELOG.md`) —
  the first pass had repurposed "Filter" as a favourites toggle instead of
  building the mockup's actual scope control, and the view toggle was
  decorative; both the Projects tab's own "+ New Project" CTA and ghost card
  now open the real dialog too (previously only Home's did).
- **Tasks:** ☑ restore + adapt data model · ☑ store→screens binding ·
  ☑ live search/favourite/scope(All·Favorites·Shared)/engine-filter ·
  ☑ real grid↔list views · ☑ "active project" state shared to workspace ·
  ☑ tests (store pure-logic + screen wiring + 2 rendering-bug regressions).
- Still not done: the workspace's own layers/canvas/predict wiring (Segment
  tab, unstarted — see below, the flagship item).

### New-project dialog · S · ✅ done (2026-07-08)
- **Goal:** the "+ New Project" flow (name · description · import · engine),
  the 3-step Label-Studio pattern, writing through the store.
- **Work:** `studio/new_project_dialog.py` — a scrim-backed modal (same
  construction as `overlays.CommandPalette`) with 3 steps: name+description →
  import (drag-and-drop + a native file picker, reusing existing atoms) →
  engine (`SegControl`). Reachable from Home's "New Project"/"Import Images"
  quick cards and top CTA; "Open Sample" opens an existing project or falls
  back to this dialog when the store is empty. Creating a project writes
  through `ProjectStore.create()` and opens straight into the workspace
  (reusing the Projects tab's active-project flow), with a real toast
  confirmation (`Toast.announce()` — previously built but never triggered
  anywhere in the app).
- **Tasks:** ☑ modal/stepper UI (reuse atoms) · ☑ file/drag import picker ·
  ☑ persist + open · ☑ tests.

### Segment (Workspace) tab · L · ✅ done (2026-07-09) ← the flagship
- **Goal:** the real segmentation surface on **our own canvas** — NOT embedded
  napari. Own viewer + layer model + tools; reuse only the ML logic.
- **Work:** `studio/layer_model.py` (new) — our own evented layer model
  (Layer/ImageLayer/LabelsLayer/PointsLayer/ShapesLayer/LayerList), Labels
  properties/defaults verified 1:1 against the installed napari source
  (opacity 0.7, brush_size 10, contiguous True, n_edit_dimensions 2, the
  PAN_ZOOM/TRANSFORM/PAINT/ERASE/FILL/PICK/POLYGON mode set). `studio/
  canvas.py` (new) — a plain QWidget/QPainter viewport (image+labels
  compositing with contrast/gamma/colormap/contour/opacity/blending all
  real) with pan/zoom/home, paint/erase/fill/pick/polygon editing, Points/
  Shapes click-to-add/draw, grid mode (one tile per visible layer), a 2D/3D
  toggle (max-intensity projection across a loaded z-stack; a genuinely
  drag-to-orbit rotate-then-perspective-project tilt for a flat 2-D image —
  not GPU volumetric rendering, noted as the simplification it is), channel-roll
  and a non-destructive view-only transpose. `studio/segment_controller.py`
  (new) — maps `ProjectSettings` to `PredictController`'s params/config and
  reuses `run_prediction_async`/`run_batch_async`/`run_benchmark_async`
  unmodified, plus `analysis`/`benchmark`/`cohort` wrappers; `record_run()`
  is the Dashboard-visibility hook (mutates `project.stats`, caller saves).
  `studio/workspace.py` rewired end to end onto all of the above — Images
  pane (real paths + real thumbnails), Layers pane + per-kind controls,
  Segment settings (engine/model/preset/thresholds/image/overlays/per-engine
  accordion), Run (real progress + elapsed-time status/toast), Results
  (stats, editable pixel calibration, save/export/measurements, colour-by,
  GT & evaluation, batch, benchmark) — plus toolbar active-state sync for
  the viewer bar and floating tool strip. `components.py`'s `Slider`/
  `Stepper` gained real interactivity (were presentational-only) since the
  whole pane needed them to work. `DashboardController.runs_table()`
  loosened so a plain segmented-but-unbenchmarked project shows up too
  (F1 "—"), not just a GT-scored one.
- **Known, deliberate gaps** (called out rather than faked): TRANSFORM mode
  is aliased to pan/zoom (no real affine transform UI); z-stack/time-lapse
  *predict* isn't wired (a project's images are individual files — the
  canvas can *display* an already-loaded volume via MIP, nothing triggers
  `_predict_volume` on one yet); "Refine…" is an explicit "coming soon"
  toast, not interactive point-prompt refinement; cross-tab settings sync
  (e.g. selecting a model in Models & Train while the same project's
  Workspace session is already open) needs reopening the project.
- **Follow-up from real-usage feedback (same day):** a real content-overflow
  bug (long dynamic text silently blowing out the fixed-width inspector
  panel), no way to add images to an already-created project, default
  rendering was fully-solid instead of fill+outline, the 2D/3D toggle
  no-opped on plain 2-D images, and grid mode ignored the mouse wheel — all
  fixed; see `CHANGELOG.md`'s same-dated follow-up entries for the detail
  and what was checked against which reference (the classic app's own
  `_add_filled_labels`, real napari's viewer-button source) before fixing.
- **Follow-up, `/goal`-driven polish pass (same day):** results now persist
  per (project, image) and reload on reopen; the canvas can no longer be
  panned/zoomed the image fully out of view; the 2D/3D toggle on a flat
  image is genuinely drag-to-orbit interactive instead of a fixed static
  tilt; a comprehensive control-by-control re-verification pass (19 new
  workspace tests + 1 cross-screen Segment→Dashboard integration test) and
  a load-speed check found no other gaps. See `CHANGELOG.md`'s same-dated
  entry.
- **Tasks:** ☑ Canvas widget (image + pan/zoom) · ☑ own layer model ↔ Layers
  panel · ☑ label/shape/point rendering + editing (brush/eraser/fill/polygon/
  point) · ☑ viewer bar (2D↔3D/grid/home) real · ☑ engine/threshold controls →
  config · ☑ Run + progress + results (reuse predict core) · ☑ GT overlay +
  eval metrics · ☑ colour-by heatmap · ☑ batch + benchmark · ☑ tests
  (controller pure + canvas/wiring — 135 new cases across 5 new test files
  plus updates to 2 existing ones, offscreen, real StudioWindow smoke-tested
  with a real image through the exact production predict chain).

---

## P1 — differentiation

### Models & Train tab · M · ✅ done (2026-07-09)
- **Goal:** real one-shot LoRA training + model management.
- **Work:** `studio/train_controller.py` — wires the train form to
  `napari_app/core/train_model.py` / `train_state_manager` (background
  thread, live progress via a guarded cross-thread signal); trained-models
  list + recent-runs history from real on-disk JSON (`loras/*.json`
  sidecars, `training_history.json`); "select into workspace" writes the
  chosen model into the active project's settings; "Import model" copies an
  external checkpoint in. Each run trains on an isolated, copied-in
  image+mask pair rather than the classic app's shared, accumulating
  folders — see `CHANGELOG.md` for why.
- **Tasks:** ☑ train form → training entry · ☑ progress/run state · ☑ model
  registry list · ☑ select-into-workspace · ☑ tests.

### Assistant tab · M · ✅ done (2026-07-18)
- **Goal:** the diagnostic Assistant as a real chat.
- **Work:** `studio/assistant_controller.py` (new, Qt-free) — settings
  (`AssistantSettings`/`AssistantSettingsStore`, one small JSON file under
  the shared storage dir) + `AssistantController`, dispatching one chat turn
  to whichever of three backends is configured: **offline** (the
  deterministic engine, `napari_app.advisor.diagnose`, reused read-only,
  always available), **Ollama** (`napari_app.advisor`'s existing bridge,
  reused verbatim — model discovery/pull/"bake an agent"/streaming chat),
  and **Custom API** (new — any OpenAI-compatible `/chat/completions`
  endpoint, local or remote, with or without a key; Studio's own bridge,
  stdlib `urllib` + SSE, since bring-your-own-model is capability the
  classic app doesn't have). `studio/workspace.py` gained the narrow
  `assistant_context()`/`apply_assistant_changes()`/`rerun_predict()` hook
  the drawer reads/acts through (mirrors the classic app's
  `PredictWidget.last_context()`/`apply_params()`/`rerun()` contract) — the
  actual cross-tab wiring: a diagnosis/chat suggestion applied in the
  Assistant writes straight into the active project's `ProjectSettings`,
  marks the quality preset "Custom", persists, and can trigger a real
  re-run, all reflected back in the Segment tab's own inspector immediately.
  `studio/assistant_panel.py` (new) — Studio's own chat UI (`ChatView`:
  bubbles/streaming/typing-indicator/empty-state; `ChangeCard`: Apply/Apply
  & re-run) and `AssistantDrawer` itself (moved out of `overlays.py`
  entirely): header + a collapsed-by-default "Model" accordion (backend
  picker, per-backend fields, live status, Ollama's download catalogue +
  "Tune for CellSeg1") + the chat (the hero surface) + Diagnose/input/Send.
  Every model/network call runs on a guarded background thread (the
  established `_safe_emit_*` + `sip.delete()`-tested pattern) so a slow or
  unreachable backend never freezes the UI.
- **Known, deliberate gap:** the auto-tune predict→score→adjust loop the
  classic app's Assistant also has (trajectory chart/table/CSV export/
  parameter importance) is *not* wired here — a large, separate sub-feature
  on top of an already-large change; left as a follow-up rather than
  half-built (`napari_app/core/tuning_loop.py` is Qt-free and reusable
  as-is whenever this is picked up).
- **Tasks:** ☑ advisor bridge (offline + Ollama, reused; Custom API, new) ·
  ☑ streaming replies · ☑ apply-suggestion actions (incl. real cross-tab
  re-run) · ☑ local model selection (Ollama discovery/pull/bake-agent,
  Custom API base-url/key/model + test-connection) · ☑ tests.

### Dashboard tab · M · ✅ done (2026-07-09)
- **Goal:** real experiment tracking.
- **Work:** `studio/dashboard_controller.py` — the loss chart, F1-across-runs
  chart and Runs table now read real on-disk data (training history +
  checkpoint sidecars + benchmarked project stats), *not* Aim's storage
  directly (`Repo.get_run()`/`query_runs()` proved unreliable outside
  Aim's own server — see `CHANGELOG.md`); "Open in Aim" still opens the
  real Aim server in the system browser.
- **Tasks:** ☑ data source · ☑ charts from real metrics · ☑ runs table · ☑ open-in-Aim.

### Logs tab · S · ✅ done (2026-07-19)
- **Goal:** real app log stream in the console (reuse `widgets/log_window.py`
  logic / a log handler), autoscroll, level filter.
- **Work:** `studio/log_bus.py` (new, Qt-free) — a bounded, thread-safe
  `LogBus` (a ring buffer of `LogRecord`: seq/timestamp/level/source/
  message) plus `StudioLogHandler`, a real stdlib `logging.Handler` bridge,
  so an ordinary `logging.getLogger(__name__).info(...)` call anywhere in
  the process reaches the console — not just hand-picked call sites.
  `install_handler()` attaches it to the root logger (idempotent, called
  from `StudioWindow.__init__` and `main()`), raises the effective level to
  INFO if it was less verbose, and keeps Studio's own `"studio"` namespace
  at DEBUG regardless, so third-party noise stays out but Studio's own
  breadcrumbs always get through. `studio/overlays.py`'s `LogsConsole` is
  rebuilt on top: backfills the bus's history on open, then stays live for
  as long as it exists (a `pyqtSignal` + the established guarded
  `_safe_emit_*`/`sip.delete()`-tested pattern marshals a record emitted
  from any thread onto the Qt main thread) — level filter (`SelectBox`,
  All/Debug/Info/Warn/Error, a minimum-severity threshold, default "Info"),
  a text search box (filters by message or source), an autoscroll toggle
  (on by default), Clear (empties the console *and* the bus), and Export
  (saves the currently-filtered lines to a `.txt` file) — a `QTextEdit`
  rather than one `QLabel` per line (the original static version's
  approach), the professional choice once the stream is unbounded, and
  matches the classic app's own `widgets/log_window.py` widget choice.
  Real emitters: `workspace.py`'s `_on_predict_log` (shared by predict/
  batch/benchmark — the reused `PredictController`'s real operational log,
  previously skimmed only for a `[ERROR]`/`[HINT]` toast, the rest thrown
  away) and `extra_screens.py`'s training `_on_log` now both also forward
  every line to the bus (`log_bus.emit_prefixed`, parsing the existing
  `[ERROR]`/`[WARN]`/`[HINT]`/`[INFO]` prefix convention onto real
  `logging` severities) — existing toast behaviour is unchanged, additive
  only. The Assistant (`assistant_panel.py`) logs backend switches, chat
  errors, model pull/tuned-agent-create results (INFO/WARNING), and
  connection-status checks (DEBUG, since those fire automatically rather
  than from a deliberate action). `app.py` logs startup, project creation,
  theme toggles (DEBUG), and now also routes uncaught exceptions through
  the bus as a real CRITICAL entry (in addition to the existing
  `traceback.print_exception`) — a crash is no longer only visible to
  whoever had a terminal open behind the app. See `CHANGELOG.md`'s
  same-dated entry for the full detail, incl. a real `SelectBox` layout bug
  found and fixed while building the new toolbar.
- **Tasks:** ☑ log handler/bus (Qt-free, real `logging` bridge) · ☑ live
  console (backfill + live updates) · ☑ level filter · ☑ search filter ·
  ☑ autoscroll · ☑ clear · ☑ export · ☑ wire real emitters (segment/train/
  assistant/app) · ☑ tests (bus/handler pure-logic + console Qt-wiring,
  incl. a cross-thread emit and a `sip.delete()` unsubscribe regression).

### Command palette (⌘K) · M · ✅ done (2026-07-20) ← the last P1 item
- **Goal:** every action reachable — run, switch engine, apply preset, export,
  navigate. Fuzzy search over a real action registry.
- **Work:** `studio/command_registry.py` (new, Qt-free) — a `Command`
  dataclass (label/section/icon/hint/keywords/handler/enabled) and a real
  Sublime-Text/VS-Code-style fuzzy matcher: a literal contiguous substring
  always outranks a scattered subsequence match (two separate score bands,
  not one flat heuristic — a flat one ranked "Switch engine" above "Run
  segmentation" for the query "seg" purely because it hit more word-boundary
  starts, caught by a test before it shipped). Empty query returns commands
  grouped by section (the mockup's "ACTIONS"/"EXPORT" caps-label browsing
  view); a real query returns a flat, score-ranked list with no section
  headers, matching how every real command palette actually behaves once
  you start typing. `studio/overlays.py`'s `CommandPalette` is rebuilt on
  top: a bounded, scrollable results list (`_PaletteRow`, cheap to
  reselect-in-place so arrow keys never rebuild or lose scroll position),
  full keyboard navigation (Up/Down/Enter via an event filter on the search
  box, wrapping top↔bottom), click-to-run, disabled rows shown dimmed
  rather than hidden (discoverability — "this is possible, but not right
  now"), and running a command is deferred one event-loop tick
  (`QTimer.singleShot(0, ...)`) before hiding the palette and calling the
  handler — the same established sipBadCatcherResult-safe pattern
  `workspace.py`'s 2026-07-10 fix already uses, since a handler can itself
  rebuild the very screen the palette sits over. `get_commands` is called
  fresh every time the palette opens (`studio/app.py`'s new
  `StudioWindow._build_commands()`), so availability always reflects the
  live project/theme/backend/running state, never a stale snapshot.
- **The registry itself** spans every tab, each command wired through the
  same narrow, testable public-alias convention the Assistant integration
  already established (`workspace.py`/`extra_screens.py`/
  `assistant_panel.py`'s own "Command palette integration" sections) —
  nothing invented, every command is a real, already-existing action:
  **Navigate** (derived straight from the sidebar's own `_NAV` list, so it
  can never drift out of sync, plus Guide & Docs; shortcut hints shown for
  Assistant/Logs); **Segment** (Run/Batch/Benchmark/Save/Export, "Switch
  engine → X" and "Apply preset → X" generated per project — only the
  *other* available engines/presets are offered, using the short
  `ENGINE_LABELS` display name, not `list_available_engines()`'s long
  descriptive combo-box text); **Models & Train** (Start/Stop — mutually
  gated on `is_training()` — Import); **Dashboard** (Open in Aim);
  **Assistant** (Diagnose, "Switch backend → X" — both open the drawer
  first so the effect is visible immediately); **Appearance** (names the
  concrete destination theme, not a generic toggle); **Projects** (New
  Project…, Open Sample); **Help** (mirrors Home's own Resources links
  exactly, GitHub included).
- **Also added: ⌘L / Ctrl+L opens (or closes) Logs** — the shortcut Logs
  itself never got when it shipped (2026-07-19), mirroring the exact
  ⌘K/⌘T dual-binding pattern. Documented in the Guide's keyboard-shortcuts
  article and the in-app shortcuts list (now 4 real bindings, was 3).
- **A real, pre-existing rendering bug found and fixed along the way**
  (not introduced by this work, just never caught): `CommandPalette`'s
  input-row and footer wrappers were plain `QWidget()`s, which inherit the
  app-wide `QWidget{background:<bg>}` rule and paint an opaque
  `<bg>`-coloured rectangle over their own children — invisible against the
  near-identical dark tones of the dark theme but a glaring flat-grey patch
  in light theme, the same "bare `QWidget()` wrapper" family
  `CHANGELOG.md`'s 2026-07-09 entry already found and fixed elsewhere
  (Guide screen). `CommandPalette` was still 100% static content at the
  time and never got a real screenshot pass, so this instance went
  undiscovered until the palette actually rendered live content — caught by
  an actual light-theme screenshot, not by any test passing; a pixel-level
  regression test now pins it (confirmed to fail against the reverted code
  first).
- **Tasks:** ☑ action registry + fuzzy search (Qt-free) · ☑ live, scrollable
  results list · ☑ keyboard navigation (Up/Down/Enter, wrapping) · ☑
  click-to-run · ☑ disabled/dim rows · ☑ every tab's real actions wired ·
  ☑ ⌘L for Logs · ☑ tests (registry pure-logic + palette Qt-wiring, incl.
  a deferred-execution regression and the bare-widget pixel regression).

**P1 is now fully done** — every P1 backlog item (Projects, New-project
dialog, Segment, Models & Train, Assistant, Dashboard, Logs, Command
palette) is real. See `ROADMAP.md`.

---

## P2 — polish & platform

- [x] **Datasets tab — first-class dataset building** · L · done (2026-07-22).
  A dedicated **Datasets** sidebar tab (`studio/dataset_screen.py`) backed by a
  Qt-free model/store (`dataset.py`) + controller (`dataset_controller.py`):
  browse built datasets (empty state → cards → detail), an **interactive build
  dialog** (pick a source project, tick which segmented images to include with
  per-image cell-count/GT status, name + val split + measurements), and
  **re-import** a dataset back into a fresh project with masks pre-seeded for
  more proofreading. Promotes the one-button export below into a real
  collect→segment→proofread→curate→re-use loop. 15 tests + both-theme
  screenshots. Follow-up: fold real folder-training on a dataset into the Train
  tab (today it points training at the dataset's images/ + masks/ dirs via a
  toast; `train_target()` already returns them).
  **Update (2026-07-22):** the New-dataset modal was redesigned (fixed a panel
  stretch/spacing bug; compact `SegControl` Val split) and gained an **"Import
  from disk"** mode — drag-drop / pick a folder or files, auto-detect image+mask
  pairs or an exported Velum dataset, preview, create. Researched against
  Roboflow / Label Studio upload UX. See `CHANGELOG.md`. Remaining: `.zip`
  import and cloud-storage sources (S3/GCS) are still open.
- [x] **Export a project as a re-trainable dataset** · M · done (2026-07-22).
  `studio/dataset_export.py` (Qt/torch-free) writes `images/` + `masks/<stem>.png`
  (uint16 instance labels) + optional `measurements/` + a `dataset.json`
  manifest/README with full provenance (engine, model, calibration, counts,
  optional train/val split). Layout matches what `train_controller` discovers,
  so the export re-trains with zero conversion — closing the *collect → segment
  → proofread → export → train* loop. Wired via `SegmentController.
  export_project_dataset()`, a Results-pane "Export dataset…" button, and a ⌘K
  command. 17 tests. Follow-ups worth doing: a real end-to-end "train on an
  exported dataset" path in the Train tab, and a COCO/CSV export variant for
  non-Velum trainers.
- [ ] Live theme repaint without a full rebuild; persist the choice.
- [x] **Projects tab v2 — deletion, trash, rename/duplicate, real scroll
  performance** · L · done (2026-07-20). The Projects tab was marked
  done in the skeleton-to-real pass (2026-07-08), but that pass only covered
  browse/search/favourite/grid-list — a real product also needs to
  delete/rename/duplicate/organise projects, and the grid's scroll needs to
  stop stuttering. Triggered by a full review against Label Studio (the
  project's own design reference, see `OVERVIEW.md`) and general
  product-dashboard practice.
  - **Root cause found for "scroll is bad, not smooth"** (read from the code,
    not guessed — cross-checked against Qt performance literature):
    `ProjectsScreen._card()`'s cover is a *live-painting* `NucleiView`
    (`paint.py`), whose `paintEvent` regenerates the entire procedural
    nuclei field (gradient blobs + antialiased polygons, `paint_nuclei()`)
    from scratch on **every single repaint** — nothing is cached. Every
    card also carries an **always-on** `QGraphicsDropShadowEffect`
    (`install_hover_lift(card, base=(14, 22, 3), ...)` — `base` alpha is 22,
    not 0, so the blur composites live even at rest, not only on hover).
    `QGraphicsDropShadowEffect` blur is known-expensive to composite
    (uncached ~30fps vs cached ~60fps in Qt-forum benchmarks — it rasterises
    the whole source and Gaussian-blurs it, unbounded by the changed area).
    Scrolling the grid forces a repaint of every visible card each frame, so
    N simultaneously-visible cards × (full procedural repaint + live blur
    composite) run on *every* scroll tick. Projects is the one screen in the
    app that combines "many cards," "expensive live-painted cover," and
    "always-on blur" inside a `QScrollArea` — Home's quick-cards use the same
    `install_hover_lift` base, but there are only 4 of them and the page
    rarely scrolls, which is why the identical mechanism doesn't manifest
    there. Secondary, smaller contributor: `screens.py`'s `scroll()` and
    `workspace.py`'s separate, duplicated `_scroll()` are both bare
    `QScrollArea`s with zero wheel-step tuning — fine on a trackpad
    (pixel-precise deltas), a blunt fixed jump on a physical mouse wheel.
  - **Work:**
    1. *Perf:* cache `NucleiView`'s painted output to an internal `QPixmap`,
       regenerated only when size/seed/density actually change, not on every
       `paintEvent` — keeps its whole reason to exist (tracking the card's
       live responsive width, unlike the pre-baked `nuclei_pixmap()` used
       elsewhere) while making repeat repaints (scroll, hover, sibling
       updates) a cheap `drawPixmap`. A wall-clock timing test proves the
       drop, run against the pre-fix code first to confirm it actually
       fails there (per the project's own established regression-test
       discipline).
    2. *Perf:* consolidate `screens.py`'s `scroll()` and `workspace.py`'s
       `_scroll()` (currently duplicated) into one shared helper and give it
       a smaller, less jarring wheel `singleStep` — a real fix, not a mask
       for #1, but part of "make scrolling itself better" as separately
       asked for.
    3. *Feature — deletion:* `ProjectStore.delete()` (`project.py`) already
       exists but is called from **nowhere** — dead code. **Revised same day,
       after real (non-offscreen) usage:** the first version wired this
       behind a trash/soft-delete layer (`trashed_at` on `Project`, a Trash
       view with Restore/Delete Forever, a Toast "Undo" action) reasoning
       that "reversible by default" beats interrogation for a local
       single-user app. Direct feedback from actually running the app —
       plus Label Studio reference screenshots showing its own Settings >
       Danger Zone pattern — said this was more machinery than the product
       needs, and a rendering bug in the Trash dialog (found live, not
       offscreen) reinforced the point. **Reverted to a direct, permanent
       delete** gated behind a `ConfirmDialog` (`confirm_delete_project`) —
       the one truly irreversible action in the flow, one click of friction,
       no undo layer to maintain. `Project.trashed_at`/`ProjectStore.trash()`
       /`restore()` were removed entirely (unused once nothing called them),
       not left as dead code.
    4. *Feature — organise:* a **⋯ kebab menu** on every grid card and list
       row. **Revised same day**, matching Label Studio's reference
       screenshots exactly: their own card overflow menu is just two items
       (Settings / Label) — landed as **Open · Duplicate · Settings**, not
       the longer Open/Rename/Duplicate/Move-to-Trash first shipped.
    5. *Feature — a real Settings screen:* **new same day**, replacing both
       the standalone `RenameDialog` and the Trash view. `project_dialogs.
       ProjectSettingsDialog` — General (editable Project Name/Description,
       mirroring Label Studio's own General Settings fields exactly) and a
       visually distinct Danger Zone card (red-tinted, `Delete Project` ->
       its own nested `ConfirmDialog`) in one compact panel rather than a
       separate navigated screen with a sidebar, since two sections don't
       need one. Reached from the kebab menu's "Settings" item.
    6. *Visual — the biggest single change:* **removed all decorative cover
       art.** The original card had a live-painted "nuclei art" thumbnail
       (the whole reason `NucleiView` got a caching pass earlier the same
       day) with star/kebab/engine-chip/progress floating on top of it as an
       overlay. Direct feedback: "why do I need project logos" against Label
       Studio's own reference cards, which are plain text (name, a stat
       row, a footer) with zero imagery anywhere. Cards (grid + list) and
       Home's recent-projects row all redesigned the same way: a plain
       header row (engine chip, star, kebab) then name/description/stats/
       tags/footer, no cover, no overlay positioning. This removes the
       single most expensive thing the grid used to repaint on every scroll
       frame — the more important half of the same-day scroll-perf story,
       on top of the NucleiView caching + eased-wheel work. `NucleiView`
       itself is untouched (still used nowhere in Projects/Home now, but
       left as tested, available infrastructure rather than deleted, since
       `workspace.py` still uses its sibling `nuclei_pixmap()` for its own,
       legitimately different, canvas-placeholder context).
    7. *Feature — findability:* a **Sort** control (Name / Last modified /
       Created / Most cells) — today the grid has no user-facing sort at
       all, only the store's implicit `updated_at` ordering. Present in
       essentially every comparable product (Label Studio, Linear, Notion).
       Unaffected by the same-day revision above.
    8. *Visual:* an audit of every margin/spacing/radius literal in
       `ProjectsScreen` against `DESIGN.md`'s rhythm (2·4·8·14·16·24·34) and
       radii (7/10/14/18) tokens — fix any that drifted off it. Unaffected
       by the same-day revision above.
  - **Tasks:** ☑ NucleiView pixmap cache + call-count regression test ☑
    consolidated scroll helper + eased wheel step ☑ delete gated behind a
    single `ConfirmDialog`, no trash layer ☑ kebab menu (Open · Duplicate ·
    Settings) ☑ `ProjectSettingsDialog` (General + Danger Zone) ☑ duplicate
    (controller + kebab wiring) ☑ Sort control ☑ spacing audit ☑ cover-art
    removal (cards, list rows, Home's recent row) ☑ offscreen screenshots,
    both themes, real QSS applied (per this file's own hard-learned
    verification rule — caught 2 real rendering bugs and 2 layout/overflow
    bugs across the first two rounds, not by tests; a **third** round found
    the Toast border/overflow bug had survived round two's fix — the earlier
    fix removed a widget that correlated with the symptom without being its
    cause — and root-caused it to two separate bugs: `Toast._subtitle`'s
    `setMaximumWidth` vs `setFixedWidth`, and `components.label()` never
    setting its own `background`, the latter a systemic gap affecting every
    label nested inside any styled `QFrame` app-wide, not just Toast. Both
    now covered by regression tests, each individually confirmed to fail
    against the pre-fix code — see `CHANGELOG.md`'s dated entry) ☑
    `CHANGELOG.md` entries.
  - **Known, deliberate gaps** (see the same-dated `CHANGELOG.md` entries for
    the full list): delete/rename from inside an open project (Workspace's
    own breadcrumb — only the grid/list kebab has it today), bulk
    multi-select, pagination/virtualisation past the ~6-project seed scale,
    a real multi-user "workspaces" concept (deliberately out of scope, see
    `OVERVIEW.md`), no undo for a deleted project (a deliberate simplification
    this time, not an oversight — see the revision note above).
- [x] **Guide & Docs screen** — done (2026-07-08). A real, in-app documentation
  surface (`studio/guide_content.py` + `guide_screen.py`): searchable article
  nav (5 topics, 10 articles) + the selected article, reached from the
  sidebar's "Guide & Docs" row (previously wired to a no-op) and from Home's
  Documentation / Getting started guide resource links (previously
  `QDesktopServices.openUrl`'d raw `.md` files — `README.md` /
  `docs/velum/OVERVIEW.md`, an internal dev doc that had no business being
  user-facing). Getting Started's steps trigger the same real callbacks
  Home's quick cards use (New Project, Open a sample, navigate a tab) rather
  than being purely descriptive. Assistant is deliberately not documented —
  it isn't wired yet (see below).
- [x] **Home motion polish + Segment tab's "no project" empty state** — done
  (2026-07-20). Two related product-feel gaps, reported directly against the
  real running app (not offscreen): Home's own "Welcome back" felt lifeless
  and its recent-projects block re-animated jarringly on every single
  revisit, and Segment (Workspace) with no project open showed a sad,
  literally-empty dark canvas -- just tiny "No image loaded" text, with a
  full three-panel IDE layout (Images/Layers · canvas · Segment/Results)
  around it, every panel empty. See `CHANGELOG.md`'s dated entry for the
  full technical story, including a first attempt at the Segment fix that
  only patched the canvas's own corner with a message and left the (now
  entirely non-functional) side panels and floating canvas toolbars on
  screen — corrected the same day after direct feedback ("канвас боковые
  панели... убрать все") into the real fix: the whole three-panel body and
  a new full-screen "no project" view are two complete alternatives in one
  `QStackedWidget`, not a message layered into a still-broken layout.
  **Revised three more times the same day, all from real usage again:** (1)
  the topbar (breadcrumb + Export/Run) stayed visible above the new
  no-project view — hidden outright now, not just its two buttons disabled,
  since the view's own "Open a Project" action already covers what the
  breadcrumb was for. (2) `NewProjectDialog`'s scrim turned out to be a
  pre-existing, app-wide bug (present in `ConfirmDialog`/
  `ProjectSettingsDialog`/`CommandPalette` too, not new to this work) —
  `rgba(8,10,20,0.34)` was tuned for light theme and nearly invisible
  against dark theme's own `bg`; fixed everywhere at once with a new
  `theme.SCRIM` constant. (3) `NewProjectDialog`'s panel also centred
  against the *whole window*, sidebar included, instead of just the content
  area — the one dialog that couldn't be parented to a single screen (it's
  shared across several), now parented to `StudioWindow._stack` (the
  content area exactly) instead. See `CHANGELOG.md`'s dated entries for all
  three.
  **Known gap, deliberately not audited this pass:** whether Models & Train
  / Dashboard have equally bad "nothing here yet" states — only Home and
  Segment were reported and fixed.
- [ ] A first-run / fresh-install onboarding pass proper (a guided first
  launch, not just a per-screen empty state) — broader than the item above,
  genuinely unstarted.
- [ ] Native macOS rounded corners + shadow via pyobjc (drop the mask) — optional.
- [ ] Settings screen (device, storage, paths, defaults).
- [ ] Packaging: a real `.app` bundle.

---

## House rules

- Keep the classic app (`napari_app/main.py`) untouched.
- Don't restyle when wiring — behaviour goes *under* the existing design.
- Heavy deps (napari/torch) imported lazily inside the tab, never in shared modules.
- Tests mandatory for new logic; note GUI/GPU as not-verified-here.
