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

### Segment (Workspace) tab · L  ← the flagship
- **Goal:** the real segmentation surface on **our own canvas** — NOT embedded
  napari. Own viewer + layer model + tools; reuse only the ML logic.
- **Work:** build a `Canvas` widget (QGraphicsView / QPainter; GPU later) that
  renders image + label/shape/point layers with pan/zoom, replacing
  `NucleiView`; build our own evented **layer model** the existing custom
  Layers panel drives (visibility, opacity, new labels/shapes/points, delete,
  grid, 2D↔3D — real effects); wire Segment settings + Run to a predict
  controller that **reuses the ML core** (`napari_app/core/predict_controller.py`,
  `engines*`, `analysis.py`), imported lazily; populate Results (stats,
  calibration, save/export/refine/measurements, colour-by heatmap, GT & eval,
  batch, benchmark); toast on completion.
- **Tasks:** ☐ Canvas widget (image + pan/zoom) · ☐ own layer model ↔ Layers
  panel · ☐ label/shape/point rendering + editing (brush/eraser/fill/polygon/
  point) · ☐ viewer bar (2D↔3D/grid/home) real · ☐ engine/threshold controls →
  config · ☐ Run + progress + results (reuse predict core) · ☐ GT overlay +
  eval metrics · ☐ colour-by heatmap · ☐ batch + benchmark · ☐ tests
  (controller pure + canvas/wiring).

---

## P1 — differentiation

### Models & Train tab · M
- **Goal:** real one-shot LoRA training + model management.
- **Work:** wire the train form to `napari_app/core/train_model.py` /
  `train_state_manager`; live run list + progress; model list from disk;
  import/select a model for the workspace.
- **Tasks:** ☐ train form → training entry · ☐ progress/run state · ☐ model
  registry list · ☐ select-into-workspace · ☐ tests.

### Assistant tab · M
- **Goal:** the diagnostic Assistant as a real chat.
- **Work:** back the drawer with `napari_app/advisor.py` (offline diagnostics +
  optional Ollama); make "apply" chips actually change settings & re-run.
- **Tasks:** ☐ advisor bridge · ☐ streaming replies · ☐ apply-suggestion
  actions · ☐ tests.

### Dashboard tab · M
- **Goal:** real experiment tracking.
- **Work:** replace static charts with the Aim integration
  (`napari_app/core/experiment_tracking.py`) — embedded view or live data;
  runs table from real runs.
- **Tasks:** ☐ data source · ☐ charts from real metrics · ☐ runs table · ☐ open-in-Aim.

### Logs tab · S
- **Goal:** real app log stream in the console (reuse `widgets/log_window.py`
  logic / a log handler), autoscroll, level filter.

### Command palette (⌘K) · M
- **Goal:** every action reachable — run, switch engine, apply preset, export,
  navigate. Fuzzy search over a real action registry.

---

## P2 — polish & platform

- [ ] Live theme repaint without a full rebuild; persist the choice.
- [ ] Guide & Docs screen (currently a no-op sidebar item).
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
