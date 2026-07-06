# AGENTS.md — orientation for AI coding agents

Read this first. It tells you what this repo is, how to work in it, and where
the task queue lives. If you change how the project is built, run, or tested,
update this file in the same commit.

## What this is

**CellSeg1** is a desktop application for **cell instance segmentation** in
microscopy images, built as a **napari** plugin/app. Two interchangeable
engines produce instance masks, and everything downstream (morphometry,
cohort stats, the Assistant, export) is engine-agnostic:

- **CellSeg1** — SAM ViT backbone + **LoRA**, one-shot fine-tuning from a
  single annotated image (`cellseg1_train.py`, `peft/`, `predict.py`).
- **Cellpose-SAM** — zero-shot generalist, no training required
  (`napari_app/engines.py`).

Target users are microscopists and cell biologists, **not** ML engineers.
The commercial goal is a world-class, enterprise-grade segmentation platform;
see `AUDIT_2026.md` for the strategic gap analysis and `docs/BACKLOG.md` for
the prioritised, actionable task queue.

## Repo map

```
napari_app/            THE PRODUCT (napari desktop app)
  main.py              entry point: builds the tabbed dock (Predict/Annotate/
                       Assistant/Train/Guide)
  widgets/             Qt widgets, one per tab. predict_widget.py wires the
                       Predict tab's UI to a PredictController instance —
                       widget-only code left: Qt construction, view-update
                       slots, drag/drop, refine, GT/evaluate, measurements
  core/                product-side logic, Qt-free and unit-tested:
                       predict_controller.py (config build + predict/batch/
                       benchmark orchestration — see below), training/predict
                       state managers, training entry — moved out of the old
                       Streamlit GUI
  engines.py           Cellpose-SAM engine + engine device mapping
  inference_cache.py   model + ViT-embedding cache (smart: changing only
                       thresholds skips the encoder)
  tiling.py            native-resolution tiled inference for large images
  advisor.py           the Assistant's offline diagnostic engine + Ollama bridge
  analysis.py          per-cell morphometry (skimage regionprops)
  benchmark.py         instance F1/AP vs ground truth
  cohort.py            batch/population aggregation

ML core (shared, imported by the app — do not delete):
  segment_anything/    vendored SAM fork (incl. the mask-NMS generators)
  peft/                LoRA implementation for SAM
  data/                dataset + image IO (data/utils.py has read/resize)
  predict.py cellseg1_train.py metrics.py sampler.py mask_nms.py
  cell_loss.py set_environment.py gpu_memory_tracker.py project_root.py

tests/                 pytest suite (pure-logic, no GPU/GUI)
.github/workflows/     CI (runs the pure-logic suite on py3.11/3.12)
checkpoints/ streamlit_storage/   bundled weights + sample data (misnamed
                       dir — see backlog; do not delete, paths reference it)
```

## Environment & how to run things

Packaging is a real `pyproject.toml` (setuptools): `pip install -e .` installs
the app + a `cellseg1` launcher + a `napari.manifest` plugin entry point. Runtime
deps live in `[project.dependencies]`; exact known-good pins are in
`requirements.txt` (the lock). The pure-logic **test** deps are a PEP 735
dependency-group (`pip install --group test` — no torch/napari). Use the existing
conda env for actual work:

- **Python with all deps:** `/opt/homebrew/Caskroom/miniforge/base/envs/cellseg1/bin/python`
  (Python 3.11; numpy/torch/skimage/cv2/napari/tifffile/pytest present).
- **Run the tests:** `<that python> -m pytest`  (fast, < 1 s, no GPU).
- **Install from source:** `pip install -e .`  (or `bash setup_napari.sh` to
  also create the env + fetch SAM weights).
- **Run the app:** `bash run_napari.sh` or `cellseg1`  (needs a real display +
  SAM weights; **cannot be driven headless** in CI or an agent sandbox).

### Verifying changes without a display

You usually cannot launch the napari GUI. Verify what you can:

1. `python -m py_compile <file>` for syntax.
2. Headless **import** check (works for Qt modules — import doesn't need a
   display, only instantiating a `QApplication`/`Viewer` does):
   ```
   QT_QPA_PLATFORM=offscreen PYTHONPATH=. <python> -c "import napari_app.widgets.predict_widget"
   ```
3. Extract new logic into **pure, importable functions** and unit-test them
   with a fake engine/predictor (see `tests/test_tiling.py` and
   `tests/test_predict_tiled_wiring.py` for the pattern — the latter uses
   `pytest.importorskip("PyQt6")` so it skips in the GUI-less CI job).
4. Say plainly in your summary what you did **not** verify (e.g. real GUI
   behaviour, real model inference).

**If a new pure-logic test reaches real image I/O** (anything that ends up
calling `data.utils`, e.g. through `_predict_cached`), don't trust a green
suite in the full conda env as proof it'll pass in CI — that env already has
every dependency installed, so it can't reveal a module the light `test`
dependency-group is missing (this bit the predict-controller split: `data/
utils.py` hard-imports `nibabel`, which no `test`-group package pulls in
transitively, and the local suite stayed green while CI failed). Check with a
throwaway venv that installs *only* the declared group:
```
python3.12 -m venv /tmp/civenv && /tmp/civenv/bin/pip install --group test . \
  && /tmp/civenv/bin/python -m pytest -q
```
(needs a `python<3.13` on PATH — `requires-python` caps at 3.12).

## Working agreement (conventions)

- **One meaningful change = one commit, then push.** Keep commits focused.
- **Tests are mandatory for new pure logic.** Full suite must stay green
  before you commit. Prefer adding a golden/regression test over a smoke test.
- **Commit messages**: imperative subject, a short body explaining *why* and
  what was verified. End every commit with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- **Default branch is `main`** — it is the product. Branch for non-trivial work.
- **Don't break the default path.** New behaviour that can't be fully verified
  here (GUI, model) goes behind an **opt-in flag, off by default**, so existing
  behaviour is byte-for-byte unchanged (see the `tiled` toggle for the pattern).
- **The single prediction choke point** is `_predict_cached(config)` in
  `napari_app/core/predict_controller.py` (re-exported by
  `napari_app/widgets/predict_widget.py`, which existing wiring tests still
  import it through). `PredictController` in the same module owns config
  building (`build_config`/`sam_config`) and predict/batch/benchmark
  orchestration; it takes plain dicts and plain callbacks, not Qt widgets/
  signals, so it's unit-tested without PyQt6/torch (`tests/
  test_predict_controller.py`). Wire engine-level changes there.
- Don't add heavy deps to the CI test path — the pure-logic suite must run
  without torch/napari.

## Where to go next

`docs/BACKLOG.md` is the task queue. Pick the **top unchecked P0**, read its
acceptance criteria, implement, test, commit, push, check it off. Update the
backlog and this file when reality changes.
