# AGENTS.md — orientation for AI coding agents

Read this first. It tells you what this repo is, how to work in it, and where
the task queue lives. If you change how the project is built, run, or tested,
update this file in the same commit.

(Claude Code specifically: this file is auto-loaded every session via a
one-line `CLAUDE.md` at repo root that imports it — Claude Code reads
`CLAUDE.md`, not `AGENTS.md`, by default. Keep instructions in this file, not
duplicated into `CLAUDE.md`.)

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
The commercial goal is a world-class, enterprise-grade segmentation platform.
Three docs, three jobs — don't blur them:
- **`docs/BACKLOG.md`** — the task queue (what's next). Start here.
- **`docs/AUDIT_2026.md`** — the strategic gap analysis (why it matters,
  scored). A point-in-time snapshot with dated addenda appended, not rewritten.
- **`docs/CHANGELOG.md`** — what actually shipped, dated (including work that
  was never a backlog item — see its intro for why that's tracked deliberately).

If you're starting a session cold, `docs/AGENT_KICKOFF_PROMPT.md` has the
prompt to paste.

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
docs/                  BACKLOG.md, AUDIT_2026.md, CHANGELOG.md,
                       AGENT_KICKOFF_PROMPT.md — see above, one job each
README.md              human-facing front door (this file is the agent one)
CLAUDE.md              one line, imports this file — see the Claude Code
                       note above; don't put instructions here directly
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
- **Log it in `docs/CHANGELOG.md` too**, one dated bullet, for *any* meaningful
  change — planned or not. This is not optional busywork: the 2026-07-05 UI
  redesign shipped ~17 commits with zero record anywhere that it happened,
  and reconciling that afterwards cost far more than a bullet each would have.
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

### Git workflow: branch → PR → merge, no manual step

**The user has pre-authorized fully automatic merging for this repo.** Don't
stop to ask before merging — that's the point of writing it here. Once you've
pushed a branch and opened a PR:

1. `gh pr create` — title + a body with a summary, test plan, and what's
   explicitly **not** verified (GUI, real model inference, etc.).
2. Wait for CI: `gh pr checks <N>`, or `gh run watch <run-id> --exit-status`
   to block until it finishes rather than polling by hand.
3. **Both `unit tests (py3.11)` and `(py3.12)` green → merge it yourself,
   immediately:** `gh pr merge <N> --merge --delete-branch` (a regular merge
   commit, matching existing history — not squash, not rebase).
4. Red instead? Fix it and push a new commit to the same branch (CI reruns
   automatically). Never merge on red, never `--admin`/force past a failing
   or pending check.
5. **Sync local right after merging** — this is the step that was missing
   before and the reason local `main` kept falling behind: `git checkout main
   && git merge --ff-only origin/main`, then `git branch -d <branch>` (refuses
   unless fully merged — that's the safety check, not busywork).
6. Report the merge commit, not a PR link — there should be nothing left for
   the user to click.

Scope of this authorization: your own branches, your own committed work, in
this repo. Never another person's PR, never with `--force`, never past a
failing check.

## Where to go next

`docs/BACKLOG.md` is the task queue. **Before picking a task**, cross-check it
against `git log --oneline -20` — twice now the docs have drifted from reality
(a resumed session naming an already-finished task; a whole UI redesign that
landed with no backlog entry at all). If something below is already done, or
something's done that isn't tracked anywhere, fix the docs first (small,
separate commit — update checkboxes, add a `docs/CHANGELOG.md` line, adjust
`docs/AUDIT_2026.md` scores if a whole dimension moved) — *then* take the
**top unchecked P0**, read its acceptance criteria, implement, test, commit,
push, merge (see above), check it off. Update the backlog and this file when
reality changes.
