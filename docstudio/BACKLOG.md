# Backlog — CellSeg1 Studio (tab by tab)

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
  toggle (max-intensity projection across a loaded z-stack — not GPU
  volumetric rendering, noted as the simplification it is), channel-roll and
  a non-destructive view-only transpose. `studio/segment_controller.py`
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

### Assistant tab · M
- **Goal:** the diagnostic Assistant as a real chat.
- **Work:** back the drawer with `napari_app/advisor.py` (offline diagnostics +
  optional Ollama); make "apply" chips actually change settings & re-run.
- **Tasks:** ☐ advisor bridge · ☐ streaming replies · ☐ apply-suggestion
  actions · ☐ tests.

### Dashboard tab · M · ✅ done (2026-07-09)
- **Goal:** real experiment tracking.
- **Work:** `studio/dashboard_controller.py` — the loss chart, F1-across-runs
  chart and Runs table now read real on-disk data (training history +
  checkpoint sidecars + benchmarked project stats), *not* Aim's storage
  directly (`Repo.get_run()`/`query_runs()` proved unreliable outside
  Aim's own server — see `CHANGELOG.md`); "Open in Aim" still opens the
  real Aim server in the system browser.
- **Tasks:** ☑ data source · ☑ charts from real metrics · ☑ runs table · ☑ open-in-Aim.

### Logs tab · S
- **Goal:** real app log stream in the console (reuse `widgets/log_window.py`
  logic / a log handler), autoscroll, level filter.

### Command palette (⌘K) · M
- **Goal:** every action reachable — run, switch engine, apply preset, export,
  navigate. Fuzzy search over a real action registry.

---

## P2 — polish & platform

- [ ] Live theme repaint without a full rebuild; persist the choice.
- [x] **Guide & Docs screen** — done (2026-07-08). A real, in-app documentation
  surface (`studio/guide_content.py` + `guide_screen.py`): searchable article
  nav (5 topics, 10 articles) + the selected article, reached from the
  sidebar's "Guide & Docs" row (previously wired to a no-op) and from Home's
  Documentation / Getting started guide resource links (previously
  `QDesktopServices.openUrl`'d raw `.md` files — `README.md` /
  `docstudio/OVERVIEW.md`, an internal dev doc that had no business being
  user-facing). Getting Started's steps trigger the same real callbacks
  Home's quick cards use (New Project, Open a sample, navigate a tab) rather
  than being purely descriptive. Assistant is deliberately not documented —
  it isn't wired yet (see below).
- [ ] Onboarding / empty states for a fresh install.
- [ ] Native macOS rounded corners + shadow via pyobjc (drop the mask) — optional.
- [ ] Settings screen (device, storage, paths, defaults).
- [ ] Packaging: a real `.app` bundle.

---

## House rules

- Keep the classic app (`napari_app/main.py`) untouched.
- Don't restyle when wiring — behaviour goes *under* the existing design.
- Heavy deps (napari/torch) imported lazily inside the tab, never in shared modules.
- Tests mandatory for new logic; note GUI/GPU as not-verified-here.
