# Changelog

What actually shipped, dated, in plain language — as opposed to
`docs/BACKLOG.md` (what's planned) or `docs/AUDIT_2026.md` (a point-in-time
strategic assessment). Newest first.

**Why this file exists:** on 2026-07-05 a full UI redesign (~17 commits, see
below) landed with no corresponding `docs/BACKLOG.md` entry — it wasn't
planned as a task, so nothing recorded that it happened, and
`docs/AUDIT_2026.md`'s UX section quietly went stale. This file is the fix:
**every meaningful change gets a line here, planned or not** (see the
"Working agreement" in `AGENTS.md`). A line here costs one sentence; not
having one cost a confused reconciliation pass across three documents.

Full detail always lives in `git log` — this is the curated, product-level
narrative, not a mirror of it. Don't transcribe every commit; one bullet per
*change a user or the next agent would want to know about*.

---

## 2026-07-07 (even later) — requested UI polish, not a backlog item

- **Predicted/ground-truth masks now show a translucent fill under the
  outline**, instead of an outline with nothing behind it — the user
  reported outlines sometimes disappearing against a busy image and asked
  for the "filled + bordered, same colour" look common tools (QuPath's
  "Fill detections", CellProfiler's OverlayOutlines) already default to.
  napari's own Labels layer can't blend a translucent fill and a contour in
  one layer — `contour` is a 0/N *toggle*, not additive (confirmed against
  napari's own docs) — so the fix is the standard two-layer overlay: a new
  `PredictWidget._add_filled_labels` adds a low-opacity filled layer
  *underneath* the existing contour=1 outline layer, both showing the same
  label data, so a cell's fill and border always match (napari's per-label
  colour is a deterministic function of the label id, confirmed with
  `get_color()`). Applied everywhere a result mask is shown — `_show_results`,
  `_show_volume_results`, and the ground-truth overlay (which keeps its
  existing uniform-green colouring, now on both layers) — but not
  Annotate's interactive paint layer, a live editing canvas with different
  tradeoffs left for a separate look if wanted. 7 new tests exercising real
  `napari.layers.Labels`/`Image` construction (not just recorded mock
  calls): layer names/contour/opacity, matching fill/outline colour per
  cell, and no layer accumulation on repeated runs.

## 2026-07-07 (later) — user-reported bug, not a backlog item

- **Fixed the Predict panel forcing horizontal scroll when SAM2 is
  selected.** `QComboBox`/`QCheckBox` text never wraps (unlike `QLabel`), so
  a widget's on-screen width is set by its widest item's/label's text —
  three strings added for SAM2 were meaningfully longer than every sibling
  they sit next to: the engine combo's SAM2 label (51 chars vs. 37-39 for
  the other two), the "segment as z-stack" checkbox (57 chars vs. 39-44 for
  the other checkboxes), and the "propagate" tracking-mode combo item (43
  chars). Shortened all three to the same range as their siblings (moving
  the dropped detail into tooltips, which were already there); confirmed
  with `QFontMetrics` that each is now narrower than its own reference
  sibling, not just "shorter than before."

## 2026-07-07

- **SAM 2 / z-stack follow-up: closed every gap the previous entry left
  open.** Requested directly rather than picked from the backlog, after
  landing the SAM2 engine — four pieces of real, additional work:
  - `_predict_volume` now composes z-stacks with tiling (a plane large
    enough that `should_tile` recommends it goes through `_predict_tiled`
    instead of always shrinking to `resize_size`).
  - `read_volume_stack`/`has_z_stack` now keep the real Z/T axis for
    ND2/CZI/LIF too, not just TIFF/OME-TIFF (`_nd2_raw`/`_czi_raw` share
    their array+axes extraction between the channel-only and
    volume-keeping read paths; LIF falls back to its existing
    channel-only behaviour if its per-plane API doesn't look like
    expected, since there's no real `.lif` file anywhere to confirm the
    guess against).
  - `analysis.compute_measurements` now dispatches on `mask.ndim` to a
    real 3-D schema (volume, 3-D centroid, equivalent diameter, ...) —
    2-D-only regionprops properties with no 3-D equivalent (perimeter,
    circularity, eccentricity, orientation) are correctly absent rather
    than faked, and `_show_volume_results` now populates real
    measurements so "Open Measurements"/"Export CSV" work on a volume
    result exactly like the 2-D path.
  - A second, opt-in SAM2 tracking mode — **propagate** — seeds objects
    with the automatic mask generator on the first plane, then tracks
    each one across the rest of the stack with SAM2's actual video
    predictor (memory-bank propagation) instead of independent-per-plane
    detection + IoU stitching. The single most speculative piece here:
    the video predictor's exact API is this module's best-effort reading
    of the public interface, not confirmed against a real install.

  107 new tests total across both SAM2 passes (163 pre-existing → 270).
  **Not verified anywhere in this work:** actual SAM2 inference (either
  tracking mode), the guessed checkpoint/Hydra-config names, the video
  predictor's exact method signatures, real ND2/CZI/LIF files (fake-module
  round-trips only), the widget on a real screen. Full writeup in
  `docs/BACKLOG.md`.

## 2026-07-06 (night)

- **SAM 2 engine for z-stacks / time-lapse** (top open P1 backlog item —
  previously the single largest ML-pipeline gap per `docs/AUDIT_2026.md`).
  New `napari_app/engines_sam2.py` registers SAM2 as a third `EngineSpec`,
  lazy-imported and `available()`-gated exactly like Cellpose, so the app
  (and CI) are unaffected without the optional `sam2` package/checkpoint
  installed (`pip install -e ".[sam2]"`). The z-stack/time-lapse capability
  itself is a new engine-agnostic layer, not SAM2-specific: new
  `napari_app/volume_stitch.py` links independently-segmented 2-D planes
  into one consistent instance volume by adjacent-slice IoU (the same idea
  Cellpose's own `stitch3D` uses — fully pure-logic, unit-tested without any
  GPU); `napari_app/channels.py` gained a `VolumeStack`/`read_volume_stack`/
  `has_z_stack` path that keeps a TIFF/OME-TIFF's Z/T axis instead of always
  reducing it to the first plane; `predict_controller.py`'s new
  `_predict_volume`/`run_volume_prediction_async` read a stack, run *any*
  registered engine per-plane, and stitch — so this also works with Cellpose
  or CellSeg1, not only SAM2 (which remains the flagship choice, being the
  only one of the three actually trained for video/volumetric consistency).
  The Predict tab gained an off-by-default "Segment as z-stack" checkbox
  (shown only for a file that genuinely has one) and a SAM2 settings card,
  both following the existing `tiled`-toggle pattern; napari's Image/Labels
  layers are n-D natively, so results just add the volume arrays directly —
  3-D per-cell measurements are explicitly not wired up yet (`analysis.py`
  is 2-D-only), so the results card is hidden rather than showing stale
  numbers. 66 new tests (163 → 229), including a `PredictWidget`
  constructed under offscreen Qt with a mocked napari viewer (a real
  `napari.Viewer()` segfaults in this sandbox) — a first for this repo's
  test suite, and it caught a real bug pre-commit: a test file's own import
  order could flip which engine registers first and silently change the
  Predict tab's default engine. **Not verified:** any real SAM2 inference
  (no GPU/package/checkpoint here), the guessed checkpoint/Hydra-config
  filenames (overridable in the UI), or the widget on an actual screen.
  **Deliberately out of scope:** SAM2's video-predictor propagation mode
  (stronger but a different, prompt-driven workflow), ND2/CZI/LIF volumes,
  and z-stack + tiling composed together. See `docs/BACKLOG.md` for the full
  writeup.

## 2026-07-06 (evening)

- **fp16 + `torch.compile` inference** (top open P1 backlog item). Two
  off-by-default checkboxes in the Predict tab's Model settings card —
  half precision (CUDA autocast fp16) and mask-decoder `torch.compile` —
  gated CUDA-only by new `use_amp()`/`use_compile()` predicates in
  `inference_cache.py`, so they're a proven no-op on CPU/MPS rather than an
  unverifiable behaviour change. `torch.compile` failures fall back to eager
  silently; toggling it is folded into the model-cache key so it forces a
  reload+recompile only where it actually matters (a CUDA device). Scoped to
  `inference_cache.py` (the app's one prediction choke point) after
  confirming `predict.py`'s `predict_images`/`predict_config` — the
  originally-scoped touch point — have no live caller anywhere in the app.
  11 new tests (163 total), green in the full conda env and in a throwaway
  venv with only the declared `test` dependency-group installed. **Not
  verified:** the actual CUDA speedup — no CUDA in this sandbox (MPS only)
  to benchmark on.

## 2026-07-06 (later still) — user-reported bugs, not backlog items

- **Fixed a Refine crash (`KeyError: 'deterministic'`)** — `_run_refine` built
  its config through the engine-selector-dependent `_build_config()`, so
  refining a LoRA checkpoint while the Predict tab's engine selector was on
  Cellpose-SAM produced Cellpose's short config, missing every SAM/LoRA
  training key. Refine always trains via `cellseg1_train.py` regardless of
  the selector, so it now calls `_sam_config()` directly, same as the
  interactive Annotate session already did.
- **Fixed a duplicated download icon** on "Load sample microscopy images" —
  the button already had a proper `QIcon`; the post-fetch label text also
  prepended its own "⬇" emoji on top of it.
- **Fixed leftover native step-button chrome on the µm/pixel calibration
  field** — it was the one spinbox in `predict_widget.py` missing
  `setButtonSymbols(NoButtons)`, which every sibling spinbox already sets.
- **Clarified the "Sample" switcher and GT autofill** — not a bug: GT autofill
  only fires when a matching `_gt`/`_mask` sidecar file actually exists, and
  of the 6 bundled quick samples only the synthetic `sample_phantom.png` ships
  with one (real ground truth for the others requires "Download BBBC039...").
  The "Sample" dropdown now labels entries "· has GT" so this is visible
  where the user is actually looking, instead of only in the GT card's status
  line further down.

---

## 2026-07-06 (yet even later)

- **Warn before silently downsampling a large image** (P1 backlog item).
  `tiling.should_warn_no_tiling()` flags exactly the case the backlog
  described: an image large enough that "Large image" tiling would help, but
  the toggle is off. `predict_controller.run_prediction_async` now emits a
  `[HINT]` log line for that case through its existing `on_log` callback —
  already wired to the widget's log window, so no widget changes were needed.
  Scoped to the single-prediction path (not batch/benchmark). 4 new tests;
  148 pre-existing tests unmodified (152 total).

---

## 2026-07-06 (even later)

- **Engine registry** (top open P1 backlog item). New `napari_app/engine_registry.py`
  (`EngineSpec` + `register`/`get`/`all_engines`) replaces the hardcoded
  `if engine == "cellpose": ... else: ...` dispatch in `predict_controller.py`
  and the two hardcoded entries in the Predict tab's engine combo + benchmark
  checklist. Adding a future engine (StarDist/InstanSeg/Micro-SAM/DeepCell/the
  still-open SAM2 item) is now one `register()` call instead of edits across
  three files. Config-building and per-engine settings UI stay bespoke per
  engine deliberately — see `docs/BACKLOG.md` for the exact boundary. 15 new
  tests; 133 pre-existing tests unmodified (148 total).

---

## 2026-07-06 (later)

- **Added `CLAUDE.md`** — a one-line file that imports `AGENTS.md` via `@AGENTS.md`.
  Researched how Claude Code actually loads project instructions: it reads
  `CLAUDE.md` at session start, not `AGENTS.md` — confirmed directly, since
  this repo's own `AGENTS.md` was never auto-loaded before this, only read
  because the kickoff prompt explicitly said to. This closes that gap for
  good without giving up `AGENTS.md`'s cross-tool portability.
- **Renamed `streamlit_storage/` → `data_store/`** (last open P0 backlog
  item). The dir is gitignored and only ever created locally by
  `setup_napari.sh`, so this was a path-string rename, not a file move:
  `project_root.py`'s `STORAGE_DIR` constant, `setup_napari.sh`, and
  `.gitignore` updated; every other reference already derived from
  `STORAGE_DIR` so needed no change. No behaviour change.

---

## 2026-07-06

- **`predict_widget.py` god-object split.** New `napari_app/core/predict_controller.py`
  — a Qt-free `PredictController` owns config-building and predict/batch/
  benchmark orchestration; the widget just wires UI to it. 22 new tests,
  behaviour unchanged. (PR #5)
- **CI fix:** added `nibabel` to the pure-logic test dependency group —
  `data/utils.py` needs it and CI didn't have it, which the fuller test
  coverage above finally exposed.
- **Docs overhaul** (this change): `AUDIT_2026.md` moved to `docs/`, annotated
  with dated addenda instead of silently rewritten; this changelog added;
  `docs/BACKLOG.md` reconciled against the audit and against undocumented
  work; `AGENTS.md` updated with the auto-merge git workflow and a
  before-you-start reality-check step; `docs/AGENT_KICKOFF_PROMPT.md` and a
  root `README.md` added.

## 2026-07-05

- **The "Lab" design system — a full, unplanned UI redesign** (~17 commits,
  `dd4596c`..`cd6b283`). Not tracked as a backlog task at the time; this is
  the retroactive record:
  - Navigation: top tabs → a permanent icon-only left rail (`widgets/shell.py`).
  - Component system v2: `SectionCard`/`CollapsibleCard`/`CollapsibleSection`
    (`widgets/common.py`), a custom `Combo` dropdown (`widgets/controls.py`).
  - Assistant rebuilt as a real chat surface (`widgets/chat.py`): message
    bubbles, streaming.
  - New `icons.py` (icon set) and `motion.py` (micro-animations — count-up
    counters, status-dot pulse).
  - Predict panel: hero cell-count KPI, stat chips, 2×2 result-action grid.
  - Assessment: `docs/AUDIT_2026.md` §4.4.
- **Microscopy formats**: OME-TIFF/ND2/CZI/LIF readers + auto-filled µm/pixel
  from metadata (`napari_app/channels.py`). (PR #3)
- **Packaging**: real `pyproject.toml`, `pip install -e .`, `cellseg1`
  console script, napari plugin manifest, pinned `requirements.txt`. (PR #4)
- **Real multi-channel support**: channel picker + per-channel percentile
  normalisation, replacing the old collapse-to-RGB read path.
- fix(predict): coerce 16-bit / float images to uint8 before handing them to SAM.

## 2026-07-04

- **`docs/AUDIT_2026.md` written** — the due-diligence audit this changelog
  now keeps honest.
- **Streamlit GUI removed**; shared logic moved into `napari_app/core/`.
  `AGENTS.md` and `docs/BACKLOG.md` added as the agent orientation + task
  queue.
- **Test + CI foundation**: first pytest suite (analysis/benchmark/cohort/
  advisor/tiling), GitHub Actions matrix on py3.11/3.12.
- **Tiled inference** (`napari_app/tiling.py`): native-resolution tiling with
  overlap + instance stitching for large images; wired into Predict behind an
  opt-in "Large image" toggle; per-tile progress in the UI. (PR #1)
- Cellpose-SAM zero-shot engine added alongside CellSeg1/LoRA; real sample
  data; local Assistant (heuristic diagnostics + optional Ollama chat).

## Earlier (2024-12-01 – 2026-07-03)

Project origin through the initial napari desktop app: SAM+LoRA one-shot
training pipeline (`cellseg1_train.py`, `peft/`), the original Streamlit GUI
(later removed, see above), then the napari `PredictWidget`/`TrainWidget`
rewrite, ground-truth evaluation, cohort/batch analysis, engine benchmarking,
and several rounds of UI polish predating the "Lab" design system. See
`git log` for the itemized history — not reconstructed here since it predates
this file and the backlog/audit process it supports.
