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
