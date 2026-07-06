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

### [ ] SAM 2 engine (3D / video)  · L
- **Goal:** add SAM 2 as an engine for z-stacks and time-lapse.
- **Why:** confocal/lightsheet/organoids are 3D; the current pipeline is 2D.
- **Acceptance:** an `Engine` entry that segments a z-stack and stitches
  instances across z; napari shows the n-D labels.
- **Touch:** `napari_app/engine_registry.py` (`register()` the new
  `EngineSpec` — the registry itself is done, see below), 3D stitching logic,
  a SAM2 weights/dependency story. Still sizeable: 3D instance stitching is
  real algorithm work and needs GPU + real SAM2 checkpoints to verify, neither
  available in the agent sandbox.

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

### [ ] fp16 + `torch.compile` inference  · S
- **Goal:** 2–4× faster inference with no accuracy change of note.
- **Acceptance:** autocast/half where supported (CUDA), optional
  `torch.compile` on the decoder, behind a setting; benchmark shows speedup;
  MPS path documented (currently falls back to CPU).
- **Touch:** `predict.py`, `inference_cache.py`, `engines.py`.

### [ ] Agentic tuning loop  · L
- **Goal:** the Assistant can itself run predict → score → adjust until AP
  plateaus, showing the trajectory.
- **Why:** today `advisor.diagnose` proposes changes but a human clicks Apply.
- **Acceptance:** a tool-calling loop with `run_predict`/`score`/`apply`
  tools; stops on plateau; every step visible and undoable.
- **Touch:** `advisor.py`, `widgets/assistant_widget.py`.

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

- [ ] Service core (REST/gRPC) + task queue + object storage  · L
- [ ] SSO / RBAC / immutable audit log  · L
- [ ] Dataset + model versioning & lineage  · L
- [ ] Collaborative annotation + review workflow  · L
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
