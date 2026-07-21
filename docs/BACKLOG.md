# CellSeg1 — Engineering Backlog (agent-actionable)

The machine-readable companion to [`docs/AUDIT_2026.md`](AUDIT_2026.md)
(strategic rationale — read sections by reference, don't duplicate them here)
and [`docs/CHANGELOG.md`](CHANGELOG.md) (dated record of what actually shipped,
including work that never came through this file — see its intro for why that
matters). Each task has a **goal**, **why**, **acceptance criteria** (how you
know it's done), **touch** (files), and a rough **size**. Work top-down within
a priority band: take the top unchecked P0, satisfy its acceptance criteria,
add tests, commit, push, tick it off. Keep this file honest — check items only
when their criteria are met.

**Before picking a task**, spend two minutes sanity-checking this file against
`git log --oneline -20`: has something below already shipped (a stale prompt
or a resumed session can point at a done task)? Has something shipped that
*isn't* below at all (the "Lab" design system UI redesign, 2026-07-05, landed
as ~17 commits with zero backlog entry — see `docs/CHANGELOG.md`)? If docs and
code have diverged, reconcile first — it's cheap now and expensive later.

Legend — size: S (hours) · M (day) · L (multi-day). Priority: P0 ship-blocker
for a credible product · P1 differentiation · P2 later.

---

## ✅ Done

- [x] **Due-diligence audit** → `docs/AUDIT_2026.md`
- [x] **Test + CI foundation** — pytest suite (analysis/benchmark/cohort/
      advisor/tiling), `pytest.ini`, GitHub Actions on py3.11/3.12.
- [x] **Tiled inference core** — `napari_app/tiling.py` (plan/stitch/
      tiled_predict) + 13 tests.
- [x] **Tiled inference wired into Predict** — "Large image" toggle,
      `_predict_tiled`, off by default.
- [x] **Remove Streamlit GUI** — shared logic moved to `napari_app/core/`.
- [x] **Remove paper artifacts** — figures/, experiment_in_paper/, video/,
      visualize_cell.py.
- [x] **UI redesign — the "Lab" design system** (2026-07-05) — *shipped
      outside this backlog; logged here retroactively, see `docs/CHANGELOG.md`
      and `docs/AUDIT_2026.md` §4.4.* Icon-rail navigation replacing top tabs
      (`widgets/shell.py`), card system v2 (`widgets/common.py`), a custom
      `Combo` dropdown, the Assistant rebuilt as a chat surface
      (`widgets/chat.py`), an icon set + micro-animations (`icons.py`,
      `motion.py`). Raises the UX/UI audit score (6.5 → 7.5) but **does not**
      close any concrete UX task below — those gaps are interaction-model,
      not visual, and are listed fresh in P1.
- [x] **Docs reconciliation** (2026-07-06) — `AUDIT_2026.md` moved to `docs/`
      with dated addenda (not rewritten); `docs/CHANGELOG.md` added;
      `docs/AGENT_KICKOFF_PROMPT.md` added; this file reconciled against both.

---

## P0 — ship-blockers

### [x] Tiling progress in the UI  · S
- **Goal:** the progress bar reflects per-tile progress during a tiled run.
- **Why:** a whole-slide run is minutes long; an indeterminate spinner reads
  as "hung" and users kill it.
- **Acceptance:** `tiled_predict`'s `on_tile(done, total)` is passed from the
  predict worker and drives a determinate `QProgressBar` ("tile 7/48"); normal
  (non-tiled) runs unchanged.
- **Touch:** `napari_app/widgets/predict_widget.py` (worker + `_predict_tiled`).
- **Verify:** unit-test the callback is threaded through; note GUI not driven.

### [x] Real multi-channel support  · M
- **Goal:** stop collapsing fluorescence to RGB; let the user map channels
  (e.g. nucleus/cytoplasm) and normalise per channel.
- **Why:** real microscopy is N-channel (DAPI + membrane + markers);
  `data/utils.read_image_to_numpy` throws that information away.
- **Acceptance:** OME-TIFF/multi-page TIFF with >3 channels loads; a channel
  picker chooses the segmentation channel(s); per-channel percentile
  normalisation; measurements can report intensity per selected channel.
- **Touch:** `data/utils.py`, `napari_app/widgets/predict_widget.py`,
  `napari_app/analysis.py`.
- **Verify:** unit tests on channel parsing/normalisation with synthetic
  multi-channel arrays.

### [x] Microscopy formats (OME-TIFF / ND2 / CZI / LIF)  · M
- **Goal:** open the formats microscopes actually produce, with pixel size
  read from metadata.
- **Why:** users can't get their data in today; PNG/plain-TIFF only.
- **Acceptance:** at least OME-TIFF + ND2 open with correct dims and
  auto-filled µm/pixel; unknown formats degrade gracefully.
- **Touch:** `napari_app/channels.py` (format router + pixel-size readers),
  `napari_app/widgets/predict_widget.py` (auto-fill µm/pixel, file filter).
- **Done:** `read_channel_stack` routes `.nd2/.czi/.lif` to optional readers
  (`nd2`/`czifile`/`readlif`) sharing one `stack_from_axes_array` transform;
  `read_pixel_size_um` reads OME-XML `PhysicalSizeX` + baseline TIFF resolution
  tags (unit-converted to µm) and native ND2/CZI/LIF voxel metadata; the
  Predict widget pre-fills its µm/pixel field from metadata (never clobbering a
  manual value). A missing reader raises a friendly `MissingReaderError`
  ("pip install nd2") and unknown formats report no calibration instead of
  crashing. 21 new pure-logic tests (`tests/test_formats.py`) cover unit
  conversion, OME/TIFF pixel size, the shared transform, ND2 via a fake module,
  and graceful degradation. **Not verified here:** real ND2/CZI/LIF files (libs
  not installed) and live GUI auto-fill.

### [x] Packaging + dependency lock  · S
- **Goal:** `pip install -e .` works; deps are pinned; napari entry point.
- **Why:** install is a bespoke shell script; no versioning, no reproducible
  env; `requirements.txt` is stale (Streamlit/ray leftovers).
- **Acceptance:** a real `pyproject.toml` with deps + a `napari.manifest`
  entry point; `requirements.txt` removed or regenerated; CI installs from it.
- **Done:** PEP 621 `pyproject.toml` (setuptools backend) declares the runtime
  deps, a `formats` extra (nd2/czifile/readlif), a `console_scripts` launcher
  `cellseg1 = napari_app.main:main`, and a `napari.manifest` entry point →
  `napari_app/napari.yaml` (validated with `npe2 validate`; backed by
  `napari_app/_npe2.py`). Flat layout handled via `py-modules` + namespace
  `packages.find` so vendored `peft`/`data` (no `__init__.py`) still ship — a
  built wheel and `pip install -e .` both resolve every module from a foreign
  cwd. The pure-logic test deps moved to a PEP 735 `[dependency-groups].test`
  so CI runs `pip install --group test` (installs *from* pyproject, no
  torch/napari). Stale Streamlit/ray `requirements.txt` regenerated as a pinned
  lock; `requirements-napari.txt` removed; `setup_napari.sh` now does
  `pip install -e .`. 8 stdlib-only tests (`tests/test_packaging.py`) guard the
  metadata (entry points resolve, py-modules exist, test group stays light).
  **Not verified here:** live napari plugin discovery/loading in the GUI (needs
  a display) and a full-dependency `pip install -e .` against PyPI (heavy deps
  already present locally, so tested with `--no-deps`).

### [x] Split the `predict_widget` god-object  · M
- **Goal:** separate prediction logic from the Qt view.
- **Why:** ~1.8k lines mixing UI + threading + IO + eval + batch is untestable
  and every change is risky.
- **Acceptance:** a `PredictController` (pure-ish, unit-tested) owns config
  build + predict/batch/benchmark orchestration; the widget only wires UI to
  it; behaviour unchanged; new controller tests added.
- **Touch:** `napari_app/widgets/predict_widget.py`, new `napari_app/core/`.
- **Done:** new `napari_app/core/predict_controller.py` holds the engine-level
  functions that used to live at the bottom of `predict_widget.py`
  (`_predict_cached`, `_predict_tiled`, `_read_for_predict`,
  `_to_display_uint8`, `_apply_clahe`) plus a new `PredictController` class:
  `build_config`/`sam_config`/`resolve_lora`/`resolve_sam` take a plain params
  dict (no Qt) and return the engine config exactly as the old
  `_build_config`/`_sam_config` widget methods did; `run_prediction_async`/
  `run_batch_async`/`run_benchmark_async` reproduce the old threaded
  `_run_prediction`/`_run_batch`/`_run_benchmark` closures 1:1 but report
  through plain callbacks (`on_log`/`on_progress`/`on_tile`/`on_result`/
  `on_done`/`on_finish`) instead of emitting Qt signals directly — the widget
  now just builds a config/params snapshot, calls the controller, and wires
  its callbacks to the same Qt signals it always had, so behaviour (including
  the stop-mid-batch `for/else` semantics) is unchanged. The whole controller
  module is free of PyQt6/torch/napari at import time (heavy deps are lazy
  imports inside the functions that need them, as they always were), so 22
  new tests in `tests/test_predict_controller.py` cover config building and
  threaded orchestration (success, per-item error, stop-early, tiling
  progress) with fake engines and run in the fast CI job — verified by
  simulating that environment (PyQt6/torch/napari/cellpose import-blocked).
  All 111 pre-existing tests plus the 22 new ones pass (133 total); the three
  existing `predict_widget` wiring test files needed zero edits since the
  moved functions are re-exported under their old names.
  **Not verified here:** the real GUI (Predict tab still looks and behaves
  the same by code inspection, but wasn't click-tested with a display) and
  real-model inference (SAM/Cellpose weights).

### [x] Rename `streamlit_storage/` → `data_store/` (or similar)  · S
- **Goal:** the misleading Streamlit-era name is gone.
- **Why:** Streamlit is removed; the dir now holds weights + samples.
- **Acceptance:** dir renamed; every path reference updated (grep
  `streamlit_storage`); `setup_napari.sh` updated; app still finds weights.
- **Touch:** repo-wide grep; `setup_napari.sh`, widgets, configs.
- **Done:** the directory is gitignored and created locally by
  `setup_napari.sh` (SAM backbone download + `test_images/`), never tracked
  in git, so there was no on-disk content to move — this was a path-string
  rename. `project_root.py`'s `STORAGE_DIR` is the single source of truth
  (every widget derives subpaths like `STORAGE_DIR / "loras"` from it), so
  the rename touched exactly `project_root.py`, `setup_napari.sh`,
  `.gitignore`, and the `AGENTS.md` repo map — confirmed by grepping the
  whole repo for `streamlit_storage`/`STORAGE_DIR` before and after. No
  behavioural change, so no new tests; full pre-existing suite stays green.

---

## P1 — differentiation

### [ ] Flaky PyQt6/sip segfault on Linux — hits both `pytest` and the real app  · M
- **Goal:** the full `pytest` run (`tests/` + `studio/tests/`) exits 0 every
  time on Linux with the full runtime dependency set installed (torch +
  PyQt6 + napari together); the real classic app (`run_napari.sh`) exits 0
  every time it's closed, not just while it's running.
- **Why:** originally found on a Linux box with an NVIDIA GPU (GTX 1070,
  compute capability 6.1, CUDA 13 wheel) and written up as a probable
  CUDA-capability-mismatch issue (`device_utils.py`'s own documented
  scenario). **2026-07-18, a second Linux box (Arch, Hyprland/Wayland,
  Intel UHD 620 — no NVIDIA GPU, no CUDA at all,
  `torch.cuda.is_available()` is `False`) reproduced the exact same class of
  crash**, which rules the GPU theory out as the root cause — this is a
  Linux/PyQt6/pyqtgraph issue independent of any GPU. Two distinct crash
  sites confirmed via `coredumpctl`/`gdb` native backtraces (the
  Python-side `faulthandler` trace alone, as used in the original writeup,
  only shows half the picture):
  1. **Mid-test** (`python -m pytest tests/ studio/tests/ -q`, 2/5 runs on
     the Arch box): `SIGSEGV` inside PyQt6's own
     `qpycore_qobject_getattr` — a `hasattr()` call on a `QObject` — reached
     from pyqtgraph's `WidgetGroup.acceptsType`/`autoAdd` (recursive
     widget-tree walk) while constructing a **new** `PlotItem`/`PlotWidget`.
     Seen via both `train_widget.py`'s `LossChart` (`tests/
     test_dashboard_window_wiring.py`) and `assistant_widget.py`'s chart
     (`tests/test_assistant_widget_wiring.py`) — not specific to one widget.
     Consistent with a stale/dangling sip wrapper reachable via Qt's own
     widget-children tree after hundreds of prior tests' widgets were
     garbage-collected without an explicit `deleteLater()`/event-loop spin.
  2. **New finding — the real app crashes on exit too, not just pytest:**
     launching the actual classic app end-to-end (a script mirroring
     `napari_app/main.py` exactly — real `napari.Viewer()`, all 5 widgets,
     real Wayland display, stepping through every tab) and closing it
     cleanly (`napari.run()`'s event loop already returned) still
     segfaults on **2 of 3** launches, natively inside PyQt6/sip's own
     `cleanup_on_exit` atexit handler (`Py_FinalizeEx` →
     `sip_api_visit_wrappers` → `cleanup_qobject` → `sip_api_get_address`
     on a bogus pointer). Every launch renders and functions correctly
     through the entire session first — this only hits during interpreter
     shutdown, after the user is already done, so no work is lost, but it's
     a real, frequent (not rare) native crash on this platform.
  **Tried and rejected as a fix:** forcing `QT_QPA_PLATFORM=xcb` (routing
  through XWayland instead of PyQt6's native Wayland platform plugin) does
  avoid the exit-time crash, but trades it for a worse, *live* regression:
  continuous vispy/OpenGL "Attempt to retrieve context when no valid
  context" errors, surfaced to the user as a real napari error toast, during
  normal use. **Do not** set `QT_QPA_PLATFORM=xcb` as a workaround — it
  breaks canvas rendering. Native Wayland (the default, no override) is the
  better tradeoff: correct rendering throughout use, occasional noisy exit.
  **Invisible to CI** either way: the `test` dependency-group never installs
  PyQt6 + torch together, so every `*_wiring.py` test just `importorskip`s
  there instead of building real widgets.
- **Acceptance:** full `pytest` run is reliably exit-0 on Linux (GPU or not)
  across e.g. 20 consecutive runs, *and* the real app exits 0 on close
  across e.g. 10 consecutive launch-and-quit cycles; or, if this is an
  upstream PyQt6/sip/Qt6-Wayland-client-thread bug outside this repo's
  control (increasingly looks like it, given two unrelated GPUs reproduce it
  and the crash site is inside sip's own exit handler, not this repo's
  code), a documented, verified-not-to-regress-rendering workaround.
- **Touch:** likely nothing fixable purely in this repo — `napari_app/
  widgets/train_widget.py` and `assistant_widget.py` (the two confirmed
  `PlotWidget` construction sites) are where the mid-test crash surfaces,
  but the actual defect is in PyQt6/sip/pyqtgraph's own object-lifecycle
  handling under Wayland. Worth checking for a newer PyQt6 point release
  before investing more here.
- **Verify:** repeat `python -m pytest tests/ studio/tests/ -q` many times
  and confirm 0 crashes; for the app, launch+quit many times and check the
  exit code, not just that the window appeared. `coredumpctl list` +
  `coredumpctl debug <PID> --debugger-arguments="-batch -ex \"thread apply
  all bt\" -ex quit"` gets the native backtrace directly (no manual gdb
  attach needed) if it reproduces again.

### CellSeg1 Studio — standalone desktop app  · XL (epic, multi-PR)
The headline UX bet: stop being "a napari plugin in a dock" and become a
self-contained product that owns its window (Home · Projects · Workspace),
with its **own** canvas (**not** embedded napari — own viewer, tools, layer
model), reusing only the ML *logic*. Structure derived from Label Studio,
retuned for microscopy. **Additive + opt-in**: classic `napari_app/main.py` /
`run_napari.sh` / `cellseg1` stays byte-for-byte unchanged; Studio ships behind
`run_studio.sh` / `cellseg1-studio`. Now a **top-level `studio/` package** with
its own docs + tab-by-tab plan in **`docs/velum/`** (that's the live source of
truth; the bullets below are a historical summary).

- [x] **Own window chrome + transitions** (2026-07-07) — frameless window with
  a custom dark title bar (own traffic lights, native move/resize via
  startSystemMove/grips), replacing the grey OS title bar; soft fade on screen
  navigation. `window_chrome.py`, 5 tests.
- [x] **Foundation** (2026-07-07) — `project.py` (Project/ProjectStore/
  ProjectSettings carrying every predict/train knob, JSON-persisted, 20 tests);
  `theme.py` (light+dark tokens, QSS, viridis, 14 tests); shell `app.py`/
  `components.py`/`screens.py` (StudioWindow + sidebar + live Home/Projects,
  reusing PredictWidget/TrainWidget; 12 headless tests); Figtree bundled; new
  entry point. See `docs/CHANGELOG.md`.
- [ ] **Workspace layer panel** · M — a custom, Studio-styled layer list +
  layer controls driven by `viewer.layers` events (napari-fidelity: new
  labels/shapes/points, visibility, opacity/blending/contour/brush/eraser,
  contiguous/preserve/show-selected, bigger colour palette, 2D↔3D + grid +
  home) on our **own** canvas + layer model (not embedded napari), so the
  workspace is ours end to end. Ground-truth overlay toggle lives here.
- [ ] **Results panel parity** · M — port the full Results surface into the
  Studio inspector (cells detected, median Ø/mean area/coverage, pixel
  calibration, Save masks/Export CSV/Refine/Measurements, "Colour cells by"
  heatmap, Ground-truth & evaluation, Batch, Benchmark). Reuse existing
  controllers; don't re-implement.
- [ ] **New-project dialog** · S — replace the auto-named stub in
  `StudioWindow._new_project` with a real Name/Description/Import/Engine flow
  (the 3-step Label Studio pattern), writing through `ProjectStore`.
- [ ] **Assistant drawer + Logs console + Dashboard screen** · M — host the
  existing `AssistantWidget` as a right drawer, `log_window` as a bottom
  console, `dashboard_window` as a screen, wired into the sidebar Tools/nav.
- [ ] **Live theme repaint** · S — currently a theme toggle rebuilds the
  Home/Projects chrome only; make embedded/reused widgets re-theme too (or
  restyle them through Studio tokens), and persist the choice.
- [ ] **⌘K command palette** · M — folds in the existing P1 "Command palette"
  item below; the Studio shell is the natural host.
- **Verify:** pure logic unit-tested; Qt screens headless-import + offscreen
  construct-tested (`pytest.importorskip("PyQt6")`); GUI/model behaviour noted
  as not-verified-here each PR.

### [x] SAM 2 engine (3D / video)  · L
- **Goal:** add SAM 2 as an engine for z-stacks and time-lapse.
- **Why:** confocal/lightsheet/organoids are 3D; the current pipeline is 2D.
- **Acceptance:** an `Engine` entry that segments a z-stack and stitches
  instances across z; napari shows the n-D labels.
- **Done:** new `napari_app/engines_sam2.py` registers a `sam2` `EngineSpec`
  the same lazy-import/`available()`-gated way `cellpose` already works —
  `sam2`'s package + a checkpoint are optional (new `sam2` extra in
  `pyproject.toml`), never a hard dependency, so the app (and CI's pure-logic
  suite) is unaffected when neither is installed. Its `predict()` is an
  ordinary single-plane `EngineSpec` (SAM2's `SAM2AutomaticMaskGenerator`,
  the same "segment everything" contract as SAM1's generator — reuses
  `predict.sam_output_to_mask` unchanged) — the z-stack capability itself is
  a new **engine-agnostic** layer above the registry, not special-cased to
  SAM2: `napari_app/volume_stitch.py` links independently-segmented 2-D
  slices into one consistent instance volume by adjacent-slice IoU (the same
  idea Cellpose's own `stitch3D` uses for its 3-D mode — a well-precedented,
  fully pure-logic algorithm, unit-tested without any GPU/model), and
  `napari_app/channels.py` gained `VolumeStack`/`read_volume_stack`/
  `has_z_stack` to read a multi-plane TIFF/OME-TIFF keeping the Z/T axis
  (previously always reduced to its first plane). `predict_controller.py`'s
  new `_predict_volume`/`PredictController.run_volume_prediction_async` read
  a stack, run *any* registered engine per-plane, and stitch — so Cellpose or
  CellSeg1 can drive a z-stack too, though SAM2 is the flagship (it was
  trained for video/volumetric consistency, unlike the other two). The
  Predict tab gained an off-by-default "Segment as z-stack" checkbox (shown
  only when `has_z_stack()` says the loaded file genuinely has one) and a
  SAM2 settings card, both following the exact `tiled`-toggle pattern this
  file's house rules already establish. napari's Labels/Image layers are
  n-D natively, so the result path just adds the volume arrays directly
  (`_show_volume_results`, a scoped-down sibling of `_show_results`).
  a mock-viewer construction that — a first for this repo — actually
  constructs a real `PredictWidget` under offscreen Qt (a real
  `napari.Viewer()` segfaults in this sandbox's offscreen platform); that
  same technique caught a real bug pre-commit (a test file's own
  module-level import order could flip which engine registers first and
  silently change the Predict tab's default engine, fixed by importing
  `predict_controller` first).

  **Follow-up pass (same day)** closed every gap the first cut had
  deliberately left open, once it was clear the acceptance criteria above
  didn't require them but a genuinely complete engine would benefit from
  them:
  - **z-stack + tiling composed**: `_predict_volume` now tiles a plane large
    enough that `should_tile` recommends it (same `tiled`/`tile_size` config
    already used for a single 2-D image), instead of unconditionally
    shrinking every plane to `resize_size`.
  - **ND2/CZI/LIF volume reading**: `read_volume_stack`/`has_z_stack` now
    keep the real Z/T axis for these three formats too (previously
    TIFF/OME-TIFF only) — `_nd2_raw`/`_czi_raw` share their array+axes
    extraction between the channel-only and volume-keeping paths; LIF's
    per-plane API (`LifImage.dims.z`/`.get_frame`) is this module's
    best-effort read of readlif's docs (no real `.lif` file anywhere in this
    codebase to check against), so it falls back to the existing
    channel-only behaviour on any attribute/shape surprise rather than
    risking a wrong guess crashing the read.
  - **3-D per-cell measurements**: `analysis.compute_measurements` now
    dispatches on `mask.ndim`, with a real 3-D schema (volume in voxels/µm³,
    3-D centroid, equivalent sphere diameter, major/minor axis, solidity,
    extent) instead of faking 2-D-only regionprops properties that have no
    3-D equivalent (perimeter, circularity, eccentricity, a single
    orientation angle — all correctly absent from the 3-D schema, not
    zero-filled). `_show_volume_results` now populates `_last_measure` for
    real, so "Open Measurements" and "Export CSV" both work on a volume
    result exactly like the 2-D path; the compact hero-chip row stays hidden
    since its captions ("Area") are hardcoded 2-D wording, not schema-driven.
  - **SAM2 video-predictor propagation mode**: a second, opt-in tracking
    mode (Predict tab → SAM2 settings → "Tracking mode") that seeds objects
    with the automatic mask generator on the first plane, then tracks each
    one across every other plane via SAM2's actual video predictor
    (`add_new_mask` + `propagate_in_video` on a temp directory of JPEG
    frames) instead of independent-per-plane detection + IoU stitching. This
    is the single most speculative piece in the whole feature: unlike the
    automatic-mask-generator path (which mirrors SAM1's well-known API
    closely), the video predictor's exact method signatures are this
    module's best-effort understanding of the public API, entirely unverified
    against a real install — capped at `sam2_max_objects` (default 40)
    tracked objects since a video predictor's memory bank holds every
    tracked object for every frame, a real cost with no hardware here to
    measure it on.

  107 new tests total across both passes (163 pre-existing → 270): the
  z-stitching algorithm, z-stack reading for all four formats (fake
  nd2/czifile/readlif modules, mirroring `test_formats.py`'s existing
  pattern), SAM2 registration/config/propagation (fake `sam2` video/mask
  predictors — no torch needed to test the temp-directory and label-volume
  bookkeeping), 3-D measurements (synthetic label volumes, hand-computed
  expected volumes/diameters/centroids), and widget wiring (checkbox
  visibility, engine-card switching, tracking-mode combo, `_show_volume_results`
  populating real measurements).

  **Not verified anywhere in this work** (no GPU, no real `sam2`
  package/checkpoint, no display in this sandbox): actual SAM2 inference of
  any kind (automatic *or* propagate mode), the exact checkpoint/Hydra-config
  filenames guessed in `engines_sam2.py` (overridable in the SAM2 settings
  card if wrong), the video predictor's exact method signatures, real
  ND2/CZI/LIF files (only fake-module round-trips), and the widget rendered
  on an actual screen. **Still explicitly out of scope:** GT overlay for
  volume results (`_show_volume_results` doesn't touch the 2-D-only GT
  evaluate path), and OME-Zarr/dask lazy loading for a whole-slide-sized
  z-stack (loads every plane into memory).

### [x] Engine registry + plugins  · M
- **Goal:** turn the two hard-coded engines into a registry so StarDist/
  InstanSeg/Micro-SAM/DeepCell can be added.
- **Acceptance:** engines register via a small interface (`predict(image,
  params) -> label mask`); the UI lists whatever is registered.
- **Touch:** `napari_app/engines.py`, `predict_widget` engine selector.
- **Done:** new `napari_app/engine_registry.py` — a Qt/torch-free
  `EngineSpec` (key, label, `predict(image, config) -> mask`, `available()`,
  optional `status_line()`, `bench_label`/`result_label`) plus
  `register`/`get`/`all_engines`/`is_registered`. `napari_app/engines.py`
  registers the two built-ins at import time (closures still lazily import
  torch/cellpose/inference_cache inside the function body, exactly as
  before — the module stays cheap to import). `predict_controller.py`'s
  `_predict_cached`/`_predict_tiled` dispatch (previously a hardcoded
  `if engine == "cellpose": ... else: ...`, duplicated in both functions) now
  do a single `engine_registry.get(config["engine"]).predict(image, config)`
  — which also collapsed `_predict_tiled`'s two near-identical per-tile
  closures into one. `ENGINE_LABELS` (used by the benchmark results table) is
  now derived from the registry instead of a hand-maintained dict.
  `predict_widget.py`'s engine combo and the benchmark checklist are both
  populated by looping `engine_registry.all_engines()` instead of two
  hardcoded entries each; adding a third engine now means one `register()`
  call plus its own settings-card/config-building code — not edits to the
  combo, the checklist, or the dispatch branches.
  **Deliberately out of scope** (still genuinely engine-specific, not
  genericized): config *building* (`build_config`/`sam_config` — SAM+LoRA and
  Cellpose need entirely different parameter shapes), the manifest's
  per-engine extra fields, and `_on_engine_changed`'s settings-card
  visibility/hint text. A future engine needs its own branch for those, same
  as Cellpose already does. 15 new tests in `tests/test_engine_registry.py`
  (registry mechanics + label defaulting + the built-ins' exact label
  strings); all 133 pre-existing tests pass unmodified (148 total) — one
  pre-existing test caught a real bug during this change (a captured-vs-live
  function reference for `available()`, same class of pitfall the dispatch
  closures already had to avoid for `predict_cellpose`).
  **Not verified here:** the real GUI (combo/benchmark-checklist population
  inspected by code, not click-tested with a display).

### [x] fp16 + `torch.compile` inference  · S
- **Goal:** 2–4× faster inference with no accuracy change of note.
- **Acceptance:** autocast/half where supported (CUDA), optional
  `torch.compile` on the decoder, behind a setting; benchmark shows speedup;
  MPS path documented (currently falls back to CPU).
- **Touch:** `predict.py`, `inference_cache.py`, `engines.py`.
- **Done:** two independent, off-by-default checkboxes in the Predict tab's
  Model settings card — "Half precision (fp16 autocast)" and "Compile mask
  decoder (experimental)" — thread `half_precision`/`compile_decoder` through
  `PredictController.sam_config()` into the config. Both are gated CUDA-only
  by new pure predicates `use_amp()`/`use_compile()` in `inference_cache.py`
  (`selected_device` not in `("cpu", "mps")`, the same device-string
  convention `_load_model` already used), so flipping either box on a
  CPU/MPS machine is a proven no-op rather than a silent behaviour change.
  `use_amp` wraps `predict_cached`'s `mg.generate()` in
  `torch.autocast(device_type="cuda", dtype=torch.float16)`; `use_compile`
  `torch.compile()`s the cached model's `mask_decoder` once at load time
  inside a bare `try/except` (a compile failure leaves the decoder eager
  instead of crashing prediction) and is folded into `_mk_model_key` — using
  the *gated* value, not the raw flag, so toggling it never forces a reload
  on non-CUDA devices. `cache_status()` appends "· compiled" when it stuck.
  Actual touch was `inference_cache.py` + `predict_controller.py` +
  `predict_widget.py`, not `predict.py`/`engines.py` as originally scoped:
  `predict.py`'s `predict_images`/`predict_config` have no live caller
  anywhere in the app (confirmed by a repo-wide grep — an unwired legacy
  path) and Cellpose has no SAM decoder to compile, so `inference_cache.py`
  (the single choke point per this file's own house rules and `AGENTS.md`)
  was the entire real surface. 11 new tests: 9 pure-logic
  (`tests/test_inference_cache.py` — CUDA-only gating, `_mk_model_key`
  differentiation, the compiled flag through `cache_status`/
  `invalidate_model`) plus 2 in `tests/test_predict_controller.py`
  (`sam_config` threads both flags through, default off). 163 total (152
  pre-existing + 11), green both in the full conda env and in a throwaway
  venv with only the `test` dependency-group installed.
  **Not verified here:** the actual speedup ("benchmark shows speedup") —
  this sandbox has no CUDA (`torch.cuda.is_available()` is False, MPS only),
  so there's no hardware to run autocast/`torch.compile` on at all, only to
  confirm they never fire and never change output on this machine. MPS stays
  excluded by design, not as a follow-up gap: it already runs with
  `PYTORCH_ENABLE_MPS_FALLBACK=1` elsewhere in this module, so any op these
  paths introduce that MPS can't execute would silently fall back to the CPU
  per-op — slower than plain eager MPS, not faster.

### [x] Agentic tuning loop  · L
- **Goal:** the Assistant can itself run predict → score → adjust until AP
  plateaus, showing the trajectory.
- **Why:** today `advisor.diagnose` proposes changes but a human clicks Apply.
- **Acceptance:** a tool-calling loop with `run_predict`/`score`/`apply`
  tools; stops on plateau; every step visible and undoable.
- **Touch:** `advisor.py`, `widgets/assistant_widget.py`.
- **Done:** built as automation of the cycle a user already runs by hand from
  the Assistant tab — Diagnose, Apply & re-run, Evaluate against ground
  truth, look, repeat — rather than a literal LLM tool-calling loop: the
  three "tools" (`run_predict`/`score`/`apply`) are `_predict_cached`,
  `benchmark.evaluate`, and `advisor.diagnose`'s existing rule-based
  `changes` dicts, called by a plain Python loop instead of by a model
  deciding what to invoke. This was a deliberate interpretation, not a
  shortcut: the acceptance criterion is literally "until **AP** plateaus" —
  AP is only defined against ground truth, and a deterministic loop over the
  advisor's own (already-shipped, already-tested) suggestions is fully
  unit-testable without an LLM in the loop, whereas Ollama is optional and
  its output non-deterministic — matching this file's own house rule of
  reusing what already exists over introducing a parallel mechanism.
  New `napari_app/core/tuning_loop.py` (Qt/torch-free): `run_tuning_loop`
  repeatedly calls a `predict_fn`, scores the mask with a `score_fn`
  (`default_score_fn` wraps `benchmark.evaluate`, scoring on the mean of
  AP@0.5/0.75/0.9 — the same "mAP" the engine-benchmark table already
  reports), and asks a `propose_fn` (`default_propose_fn` wraps
  `advisor.diagnose`, merging every finding's `changes` and dropping any
  that wouldn't actually move a parameter) what to change next. It stops on
  the first of: `patience` consecutive rounds that fail to beat the best
  score by more than `min_delta` (the plateau); `max_steps`; the advisor
  running out of suggestions; a change repeating one already tried (guards
  an oscillation looping forever); or cooperative cancellation. Every round
  is recorded with its own full parameter snapshot (a `TuningStep`), not
  just the winner, so a caller can jump back to *any* step — that is what
  makes it "undoable" rather than a black box reporting only a final
  answer. `PredictController.run_tuning_loop_async`/`stop_tuning` (new,
  alongside the three existing `run_*_async` orchestration methods) thread
  this through the real `build_config` + `_predict_cached` on a background
  daemon thread, reporting each `TuningStep` via `on_step` and errors via
  `on_log`, exactly the callback shape every other controller method
  already uses. `PredictWidget` gained `has_ground_truth`/`start_auto_tune`/
  `stop_auto_tune`/`restore_tuning_step` — the last two exactly mirror the
  existing `apply_params`+`rerun` pair the Assistant's manual "Apply &
  re-run" already calls, so restoring a step re-runs the real pipeline
  instead of just repainting a cached array. The Assistant tab gained an
  "Auto-tune" icon button (next to the existing "Diagnose" one) that starts/
  stops the loop and streams each `TuningStep` into the chat as a
  `AutoTuneStepCard` (score, delta from the previous round, cell count, the
  changes that produced it, and a "Use these params" button) as it arrives
  — deliberately never touching the viewer mid-loop (the loop's own
  `predict_fn` calls `_predict_cached` directly, bypassing the widget's
  normal result-display path entirely) so a multi-round background search
  can't flicker or corrupt the live prediction the user is looking at;
  only an explicit "Use these params" click calls `restore_tuning_step`,
  which does go through the normal `apply_params` + `rerun` path and so
  does update the viewer. Finishing posts a summary naming the best step's
  score/cell count.
  38 new tests (331 → 369): `tests/test_tuning_loop.py` (18 — the pure loop's
  plateau/budget/dedup/cancellation logic against scripted fakes, plus
  `default_score_fn`/`default_propose_fn` against the same synthetic
  label-array style `test_benchmark.py`/`test_advisor.py` already use, plus
  one real-defaults-end-to-end case), 3 more in `tests/test_predict_controller.py`
  (threaded orchestration through the real fake-connected-components
  cellpose path already established there — success/step-sequencing, error
  handling, `stop_tuning` cancellation), `tests/test_assistant_widget_wiring.py`
  (7, new — a first for this widget, which had zero test coverage before;
  drives a real `AssistantWidget` against a small fake predict-widget stand-
  in rather than a full `PredictWidget`, since the Assistant only ever calls
  its documented small API), and `tests/test_predict_widget_autotune_wiring.py`
  (10, new — a real `PredictWidget` under offscreen Qt, the
  `test_predict_labels_display_wiring.py` pattern, covering the new hooks'
  own precondition/GT-loading/resize/dispatch glue with the controller call
  itself monkeypatched out).
  Caught one real gap while verifying against the *true* light CI
  dependency-group (not the full conda env — see the `AGENTS.md` fix in the
  same change): two new controller tests exercised `build_config`'s
  `cellpose` branch for the first time with a real (not hand-built) params
  dict, which checks `engine_registry`'s `cellpose.available()` — false in
  the light group (no real `cellpose` package there) — fixed the same way
  `test_build_config_cellpose_shape` already does, by monkeypatching
  `engines.cellpose_available`.
  **Not verified anywhere in this work:** the real GUI (button/card layout
  inspected by code, not seen rendered with a display attached) and real
  model inference — every test drives the loop against the fake
  connected-components engine already used throughout
  `test_predict_controller.py`, never real SAM/Cellpose/SAM2 weights, so the
  loop's *plumbing* (threading, stopping rules, undo, rendering) is proven
  but a real advisor-guided AP improvement on a real model has not been
  observed.

  **Follow-up pass (same session), on direct instruction to go further:**
  the first pass explicitly deferred both a user-facing settings panel and
  an LLM-driven strategy as "a natural follow-up" — the user then asked for
  exactly that, plus asked to research how comparable products instrument
  this kind of loop and bring the best of it in ("больше полезной
  информации — я смогу её убрать", i.e. bias toward more instrumentation,
  not less). Researched: AutoML/sweep dashboards (Optuna's
  `plot_optimization_history`/`plot_param_importances`; Weights & Biases'
  sweep parameter-importance panel), the ReAct tool-calling-loop pattern and
  its stop-condition/infinite-loop failure modes, agentic-coding-tool UX
  (Cursor/Devin's one-click checkpoints), and this app's own segmentation
  neighbourhood (CellSeg3D's grid-search threshold tuner; the
  "the-segmentation-game" napari plugin's sortable Highscore table). Four
  concrete features came out of that, all real, none cosmetic:
  - **A genuine LLM tool-calling strategy.** `tuning_loop.llm_propose_fn`
    hands the "what to change next" decision to a connected local Ollama
    model each round instead of the fixed rule table — a real ReAct round
    (reason, then act): `advisor.build_tuning_prompt` shows the model the
    full score trajectory so far and asks for one `SUGGEST:` line or a
    `STOP: <reason>` line (parsed by the new `advisor.parse_stop`), reusing
    the exact `SUGGEST:` protocol the chat already speaks. Falls back to
    the rule-based advisor whenever the model errors, is unreachable, or
    replies with nothing usable — a flaky/slow local model degrades the
    loop, never crashes it. `run_tuning_loop`/`PredictController.
    run_tuning_loop_async` gained a `strategy`/`model` parameter (default
    `"advisor"`, unchanged behaviour); the Assistant's new settings row
    lets the user pick "Local model" only once one is actually connected.
  - **A settings panel, exposed rather than hardcoded.** Reversing the
    first pass's own "deliberately out of scope" call: `max_steps`/
    `patience`/`min_delta` are now three spinboxes in the new "Auto-tune"
    card, not fixed constants — same reasoning the rest of this app's
    Inference-parameters card already uses for every other numeric knob.
  - **Product-grade trajectory instrumentation**, replacing the first
    pass's per-round chat cards: a live score-vs-round chart (`TuningChart`
    — pyqtgraph if present, else matplotlib, the exact fallback
    `train_widget.LossChart`/`measurements_window.Histogram` already use —
    an Optuna/W&B-style optimization-history plot with the best round
    marked); a sortable `QTableWidget` leaderboard (round,
    score, Δ, cell count, reason — the "the-segmentation-game" Highscore-
    table idea) that a click-then-"Use selected round" restores from,
    exactly like a Cursor/Devin checkpoint; a CSV export
    (`tuning_loop.write_trajectory_csv`) for taking the raw run elsewhere;
    and a plain-language "what mattered most" panel
    (`tuning_loop.parameter_importance` — Pearson correlation between each
    varying numeric parameter and the score across the run, W&B's
    parameter-importance panel without needing enough trials for its
    random-forest model). The chat still gets one short line per round (the
    conversational, "what just happened" narrative) instead of a growing
    pile of cards.
  - **Every stop is explained, not just coded.** `TuningResult` carries
    both a machine `stop_reason` (`STOP_REASONS`/`describe_stop_reason`:
    plateau, max_steps, no_more_suggestions, repeated_change, cancelled,
    error) *and* a free-text `stop_detail` — the advisor's or the local
    model's own words for why it stopped, not discarded once the loop
    exits, surfaced verbatim in the finish banner.

  This pass touches every file the first one did (`tuning_loop.py`,
  `predict_controller.py`, `predict_widget.py`, `assistant_widget.py`) plus
  `advisor.py` (new `build_tuning_prompt`/`parse_stop`); `AutoTuneStepCard`
  (the first pass's chat-card widget) was removed in favour of the table.
  24 more net-new tests (369 → 393, 62 total across both passes; the
  pure-loop and controller suites were also rewritten in place for the new
  `Proposal`/`TuningResult`/4-arg-`ProposeFn` contracts, so more than 24
  individual test functions changed even though 24 is the net count):
  `llm_propose_fn`'s success/stop/no-usable-
  suggestion/model-error paths (a monkeypatched `advisor.ollama_chat`, never
  a real network call), `parameter_importance`/`write_trajectory_csv`/
  `describe_stop_reason` against synthetic trajectories, the controller's
  `strategy="llm"` wiring (confirms the fake Ollama call actually happens,
  confirms a missing model name degrades to the advisor instead of
  crashing), and the Assistant's settings-row/chart/table/CSV-export/
  parameter-importance wiring end to end via a fake predict-widget. Full
  suite green in the full conda env (393 passed) and in a from-scratch venv
  with only `pip install --group test` (the corrected command — 288 passed,
  9 skipped for PyQt6-gated files).
  **Not verified in this pass either:** the real GUI (chart/table
  rendering, in particular whether pyqtgraph or the matplotlib fallback
  actually engages on a real screen, and whether the two coexist visually
  with the rest of the card) and a real local model's actual tuning
  judgement (the LLM strategy's tests all use a scripted fake
  `ollama_chat` — no real Ollama install in this sandbox to verify a real
  model gives *useful* suggestions, only that the plumbing around whatever
  it says is correct). **Still deliberately out of scope:** a beam-search/
  multi-candidate-per-round strategy (CellSeg3D's grid search tries many
  combinations per pass; this loop still tries exactly one per round) and a
  parallel-coordinates-style multi-parameter plot (W&B's other headline
  sweep visualization) — the single-parameter importance ranking captures
  most of the same insight without a custom multi-axis renderer neither
  pyqtgraph nor matplotlib gives for free in a ~100px-tall embedded widget.

### [x] Unified experiment tracking (Aim)  · L
- **Goal:** every predict/batch/benchmark/train/auto-tune run logged to one
  real, open-source experiment-tracking dashboard instead of three separate
  custom charts (the auto-tune trajectory chart above, `train_widget`'s
  `LossChart`, and no history at all for ordinary predicts).
- **Why:** direct request, not a pre-existing backlog item — asked in the
  same conversation as the auto-tune follow-up, after researching open
  self-hosted tools that could be embedded "for free" (no custom chart code)
  rather than reinventing history/comparison views a third time.
- **Acceptance:** predict/batch/benchmark/train/auto-tune each write to a
  shared local run history; a "Dashboard" button opens it, embedded in-app
  when possible and always available in the system browser as a fallback.
- **Touch:** new `napari_app/core/experiment_tracking.py`,
  new `napari_app/widgets/dashboard_window.py`, `predict_controller.py`,
  `core/train_model.py`, `pyproject.toml`, three widgets' footers.
- **Done:** chose [Aim](https://aimstack.io/) over TensorBoard/MLflow/
  ClearML after comparing them directly: fully local/self-hosted (no
  account, matching this app's existing "nothing leaves your machine"
  stance — unlike W&B's cloud-first model), a modern UI explicitly built to
  out-do TensorBoard, and its run/hparam-comparison shape is the closest
  match to what this app's three producers (one-shot predicts, multi-epoch
  training, multi-round auto-tune) all generate. New
  `napari_app/core/experiment_tracking.py` is the only file that imports
  `aim` (lazily, `available()`-gated exactly like sam2/cellpose/Ollama
  elsewhere in this package — a new `tracking` extra, not a hard
  dependency): `start_run(experiment, hparams)` returns either a real
  `aim.Run`-backed handle or a `_NullRun` no-op, and **every single call on
  the real handle is individually wrapped in its own try/except** — a
  tracking hiccup (disk full, a future Aim API change, a corrupt repo)
  degrades to one dropped data point, never an interrupted predict/train/
  auto-tune run. `ensure_dashboard_running()`/`stop_dashboard()` manage
  Aim's own `aim up` dashboard server as a background subprocess, shared
  across every "Dashboard" button the same way `get_log_window()`/
  `get_measurements_window()` already share one floating window each.
  `predict_controller.py` logs one run per single predict, one per batch
  image, and one per (engine × image) benchmark pair (with the real AP/F1
  metrics already computed there — Aim becomes a second, comparable view
  onto the same numbers the benchmark results table shows); the auto-tune
  loop logs one run for the whole loop, `.track()`-ing score/cell-count per
  round exactly like the existing chart/table do, plus the final
  `stop_reason`. `core/train_model.py`'s per-epoch loop now also tracks
  `loss` per epoch, a first for that file (`LossChart` itself is untouched
  — Aim is an *addition*, not a replacement, since the in-app chart is still
  the faster at-a-glance view during a live run).
  New `napari_app/widgets/dashboard_window.py` is a floating singleton
  window (the exact `log_window.py`/`measurements_window.py` pattern):
  embeds Aim's real web UI via `QWebEngineView` when the separate,
  heavier `tracking-ui` extra (`PyQt6-WebEngine` — it bundles Chromium,
  roughly +100MB, a real cost for a desktop app) is installed, and always
  offers "Open in browser" (`webbrowser.open`) as a fallback that needs
  only the base `tracking` extra — deliberately two separate extras so
  installing Aim itself stays light. A "Dashboard" button was added to
  Predict's and Train's existing log-footer row and to the Assistant's
  Auto-tune card, all three opening the *same* shared window/repo.
  40 new tests (393 → 433): `tests/test_experiment_tracking.py` (22 — the
  `tests/test_engines_sam2.py` pattern of injecting a bare
  `types.ModuleType("aim")` into `sys.modules` rather than installing the
  real package, covering the no-op path, the real-module path via a
  recording fake `aim.Run`, every guarded-failure path (`track`/`__setitem__`/
  `close`/constructor each raising), `_sanitize`, and the dashboard
  subprocess singleton/reuse/restart-after-death logic with a fake
  `subprocess.Popen`), 4 more in `tests/test_predict_controller.py` (one
  per orchestration method, monkeypatching `experiment_tracking.start_run`
  to a spy), `tests/test_train_model_tracking.py` (4, new — a first for
  `train_model.py`, which had zero test coverage before this; every heavy
  dependency it touches — `cellseg1_train`'s functions, `set_environment.
  set_env`, dataset loading — is monkeypatched to a scripted fake so the
  "training loop" runs in milliseconds), and `tests/test_dashboard_window_wiring.py`
  (10, new — the singleton, both the embedded and (since PyQt6-WebEngine
  isn't installed here) fallback-message paths, and each of the three
  widgets' own "Dashboard" button). Full suite green in the full conda env
  (433 passed) and in a from-scratch venv with only `pip install --group
  test` (314 passed, 11 skipped — the 2 new skips are exactly the
  torch-gated and PyQt6-gated new files, as expected).
  **Not verified anywhere in this work:** a real Aim install of any kind (no
  network access to actually `pip install aim` and confirm `aim up` really
  serves a working dashboard, or that its UI renders sensibly for this
  app's specific hparam shapes) and the embedded `QWebEngineView` path (no
  `PyQt6-WebEngine` installed either) — every test here proves the
  *wrapping* code (guards, singleton reuse, correct CLI args, correct
  `track()`/hparams calls) is correct assuming Aim's own documented API
  surface, not that a real Aim server behaves as documented.

  **Follow-up (same day), on request — closed the real-Aim gap above and
  fixed a real bug it found:** the user reported not seeing the "Dashboard"
  button in Predict/Train. Investigated by literally rendering both widgets
  under offscreen Qt (`QT_QPA_PLATFORM=offscreen`) and grabbing a real
  `QPixmap` screenshot rather than only import-checking — the button *was*
  present and correctly placed in both footers (screenshots showed it next
  to "Log"), so this wasn't a code defect; the far more likely explanation
  is a stale already-running app process from before PR #23 merged (this
  screenshot technique itself is worth keeping in mind as a stronger
  verification tool than a plain headless import for future UI-layout
  questions in this sandbox). Installed the real `aim` package into a
  throwaway venv (not this repo's shared conda env) and drove
  `experiment_tracking.py` for real end-to-end: a real `aim.Run` created a
  real `.aim` repo with real hparams/tracked values, and —
  critically — **`ensure_dashboard_running()` failed** with
  `FileNotFoundError: 'aim'`, a genuine bug this real install caught that no
  amount of fake-module-injection testing could have: `subprocess.Popen(["aim",
  ...])` resolves the bare command through `PATH`, which a pip-installed
  console-script's directory is not guaranteed to be on (it wasn't, in this
  throwaway venv, invoked by its full interpreter path rather than an
  activated shell — plausibly also how a packaged desktop app would launch
  Aim). Fixed with a new `_aim_cli_path()`: prefer the sibling of
  `sys.executable` (where pip actually installs the console-script,
  regardless of `PATH`), falling back to the bare name only if that sibling
  doesn't exist. Re-ran the same real-install script after the fix:
  `ensure_dashboard_running()` now launches a real `aim up` that answers
  `HTTP 200` with real Aim UI HTML — the dashboard genuinely works
  end-to-end, not just against fakes. 2 new tests for `_aim_cli_path()`'s
  two branches (sibling present / absent, via a monkeypatched
  `sys.executable`); the existing Popen-args test's assertion loosened from
  an exact `"aim"` match to `endswith("aim")` so it stays valid under both
  branches. Separately, replaced Train's old plain-text "Training history"
  card (`self.history_box`, `STATE_MANAGER.load_history()` — this session
  only, no comparison) with a "Run history" card pointing at the same
  Dashboard — removing a second, now-redundant history UI instead of
  leaving two competing ones. 3 new tests in this follow-up (433 → 436; 393
  → 436 across the whole Aim feature including the earlier pass); full
  suite green in the full conda env (436 passed) and the light venv (316
  passed, 11 skipped). The embedded `QWebEngineView` path is
  still unverified (`PyQt6-WebEngine` was not installed for this pass
  either — it's the separate, heavier `tracking-ui` extra).

### [ ] Vision-grounded QC in the Assistant  · L
- **Goal:** the agent inspects the actual mask (not just scalar stats) and
  highlights specific wrong cells ("45 and 46 are merged").
- **Acceptance:** per-instance error candidates surfaced as selectable labels
  in the viewer with a natural-language explanation.
- **Touch:** `advisor.py`, `analysis.py`, viewer integration.

### [ ] Reproducibility capsule  · M
- **Goal:** one click exports model + params + input hash + versions so a
  result can be reproduced.
- **Why (partially done already):** `PredictWidget._write_manifest` already
  writes a JSON sidecar on every "Save masks" — engine, full params, python/
  torch versions, a 16-char sha256 of the input image. See
  `docs/AUDIT_2026.md` §5.4. What's missing is the other half: reading a
  manifest back and re-running from it.
- **Acceptance:** a "reproduce" path — load a `*.json` manifest, rebuild the
  exact config, re-run, and confirm the resulting mask matches (or report a
  diff if the environment can't reproduce it bit-for-bit, e.g. different
  torch/CUDA version).
- **Touch:** `napari_app/widgets/predict_widget.py` (`_write_manifest` already
  there), new `napari_app/core/provenance.py` for the read/replay half.

### [ ] Built-in statistics + auto-report  · M
- **Goal:** compare conditions (t-test/Mann-Whitney) and emit a figure-ready
  report from a cohort.
- **Touch:** `cohort.py`, `widgets/cohort_window.py`, new report module.

### [ ] Command palette (⌘K)  · M
- **Goal:** one keyboard-driven surface for predict/switch-engine/apply-
  suggestion/export/save, instead of hunting across five rail panels.
- **Why:** `docs/AUDIT_2026.md` §4.2/§4.3 — the "Lab" redesign (2026-07-05)
  improved every panel's visual hierarchy but didn't touch the click-heavy,
  multi-tab interaction model itself; this is still fully open.
- **Acceptance:** a fuzzy-searchable command list bound to a shortcut (⌘K/
  Ctrl+K), covering at minimum: run prediction, switch engine, switch tab,
  apply an Assistant suggestion, export CSV/masks.
- **Touch:** new `napari_app/widgets/command_palette.py`, `main.py` (global
  shortcut), each widget exposes its actions to a shared registry.

### [ ] Undo/redo for mask *proofreading* edits  · M
- **Goal:** a real history stack (merge/split/delete cells on a Labels layer)
  with hotkeys, distinct from Annotate's existing single-step "Undo last"
  point-click button (`annotate_widget.py:170` — that undoes one annotation
  click, not a mask edit).
- **Why:** `docs/AUDIT_2026.md` §4.2 — proofreading a predicted mask today has
  no undo beyond napari's generic layer history.
- **Acceptance:** a visible undo/redo stack for label-merge/split/delete
  actions with keyboard shortcuts; multi-step; survives at least one Save.
- **Touch:** `napari_app/widgets/predict_widget.py` or `annotate_widget.py`
  (wherever proofreading actions live), viewer Labels-layer integration.

### [ ] Persist floating-window geometry across sessions  · S
- **Goal:** the Log/Measurements/Cohort windows (`widgets/log_window.py`,
  `measurements_window.py`, `cohort_window.py`) reopen where you left them.
- **Why:** these are already independent, multi-monitor-friendly `QWidget`
  windows (`docs/AUDIT_2026.md` §4.2's "detachable panels" ask is more done
  than the audit assumed) — the one real gap is that position/size resets to
  default every launch.
- **Acceptance:** each window's geometry is saved on close (`QSettings` or a
  small JSON sidecar) and restored on next open; falls back to the current
  default position if nothing saved yet.
- **Touch:** `napari_app/widgets/log_window.py`, `measurements_window.py`,
  `cohort_window.py`.

### [ ] Interactive onboarding tour  · S
- **Goal:** a first-run guided path on sample data — "load → predict → N
  cells → export" — instead of a static Guide tab.
- **Why:** `docs/AUDIT_2026.md` §4.2; the Guide tab got new visuals in the
  2026-07-05 redesign (`design: Train, Guide and floating windows on the new
  language`) but is still static content, not an interactive walkthrough.
- **Acceptance:** a first-launch (or Guide-triggered) sequence that drives the
  user through one real prediction on bundled sample data in under 60s.
- **Touch:** `napari_app/widgets/guide_widget.py`, `predict_widget.py`.

### [x] Warn before silently downsampling a large image  · S
- **Goal:** when an image would be shrunk significantly for inference (and
  "Large image" tiling is off), tell the user before they lose small cells.
- **Why:** `docs/AUDIT_2026.md` §4.2 — tiling (done) is opt-in; a user who
  doesn't know to enable it gets silently downsampled with no warning.
- **Acceptance:** if `should_tile(img.shape, ...)` would return true but the
  `tiled` toggle is off, show a one-line hint suggesting "Large image" mode
  before/after running.
- **Touch:** `napari_app/widgets/predict_widget.py`, `napari_app/tiling.py`
  (`should_tile` already exists).
- **Done:** new `tiling.should_warn_no_tiling(shape, tiled, tile, margin)` —
  `not tiled and should_tile(shape, tile, margin)`, i.e. exactly the
  acceptance criteria's own condition. The actual touch point turned out to
  be `predict_controller.py`, not `predict_widget.py` directly: a
  `"[HINT] Large image — inference resized it, which can lose small cells.
  Enable \"Large image: tile at native resolution\" for full detail."` line is
  emitted through `run_prediction_async`'s existing `on_log` callback, right
  after the "✓ N cells" line — `on_log` was already wired straight to the
  widget's log window, so no widget code needed to change at all. Scoped to
  the single-prediction path only (not batch/benchmark, which don't have an
  equivalent interactive per-run log line to hang this off of). 4 new tests
  (2 in `tests/test_tiling.py` for the pure condition, 2 more in
  `tests/test_predict_controller.py` verifying the wiring: hints when large +
  untiled, silent when tiled or small); 148 pre-existing tests unmodified
  (152 total).
  **Not verified here:** the real GUI (log line inspected by code, not seen
  rendered in the actual log window with a display attached).

### [ ] OME-Zarr multiscale viewing for whole-slide images  · L
- **Goal:** open a whole-slide image as a lazy pyramid so napari's viewer
  stays responsive without loading the full-resolution array.
- **Why:** `docs/AUDIT_2026.md` §3.2/§10 — tiled *inference* is done, but
  *viewing* a 100k×100k image still means loading it whole; this is the
  remaining half of the original "no tiling" finding.
- **Acceptance:** an OME-Zarr (or a zarr/dask-backed pyramid built on the fly)
  path that napari renders at multiple zoom levels without a full-res load;
  degrades gracefully for non-pyramidal formats.
- **Touch:** `napari_app/channels.py` or a new `napari_app/pyramids.py`,
  `predict_widget.py` image-loading path.

---

## P2 — platform / enterprise (own product surface; see `docs/AUDIT_2026.md` §8)

**Foundation landed 2026-07-21** — a new, additive top-level `server/` package
(accounts, organizations, RBAC, immutable audit log, the Label-Studio-shaped
Project→Task→Annotation→Review model) on a scale-ready DB (stdlib SQLite/WAL by
default, Postgres-portable). Pure stdlib, 74 tests, no HTTP tier yet. See
`server/README.md` and `docs/CHANGELOG.md`. This starts several items below;
the acceptance criteria are narrowed to the real remaining gap.

- [ ] Service core (REST/gRPC) + task queue + object storage  · L —
      *data/service layer done in `server/`; remaining: the HTTP/REST tier over
      `server.service` (a FastAPI `server` extra), a task queue for the ML, and
      S3/MinIO object storage for images/masks.*
- [ ] SSO / RBAC / immutable audit log  · L —
      *RBAC (6 roles + permission matrix + escalation guard) and the immutable
      audit log are done in `server/`; remaining: SSO (OIDC/SAML) + SCIM, which
      need the HTTP tier first.*
- [ ] Dataset + model versioning & lineage  · L
- [ ] Collaborative annotation + review workflow  · L —
      *the review workflow (annotator→reviewer approve/reject with task-status
      transitions) + the multi-user data model are done in `server/`; remaining:
      the UI/API surface (assignment queues, comments, inter-annotator
      agreement) and wiring Studio to it as a client.*
- [ ] Docker / Helm / K8s deploy (on-prem + cloud)  · L
- [ ] 21 CFR Part 11 / GxP compliance contour  · L

---

## House rules for editing this file

- Add a task before you start non-trivial work; tick it only when its
  **acceptance criteria** are met and tests are green.
- Keep priorities honest — if something becomes a ship-blocker, move it to P0.
- Link deeper rationale to `docs/AUDIT_2026.md` sections rather than
  duplicating it.
- **Shipped something that wasn't a backlog item** (a design pass, a quick
  fix, anything non-trivial)? Add a line to `docs/CHANGELOG.md` regardless —
  that file's whole job is to catch exactly this, since this backlog can't
  retroactively track work it never knew about.
- If a chunk of work here turns out to already be partially done by the time
  you read it (check the code, not just the checkbox), narrow the acceptance
  criteria to the real remaining gap instead of redoing it — see the
  "Reproducibility capsule" item for the pattern.
