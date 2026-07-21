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
microscopy images, built as a **napari** plugin/app. Three interchangeable
engines produce instance masks, and everything downstream (morphometry,
cohort stats, the Assistant, export) is engine-agnostic:

- **CellSeg1** — SAM ViT backbone + **LoRA**, one-shot fine-tuning from a
  single annotated image (`cellseg1_train.py`, `peft/`, `predict.py`).
- **Cellpose-SAM** — zero-shot generalist, no training required
  (`velum_core/engines.py`).
- **SAM 2** — zero-shot, the flagship choice for z-stacks/time-lapse
  (`velum_core/engines_sam2.py`; optional dependency, degrades gracefully
  when not installed — see `docs/BACKLOG.md`'s "SAM 2 engine" entry).

> **2026-07-21 — the app is now `studio/`, not `napari_app/`.** Studio (its own
> PyQt6 app, its own canvas — never embedded napari) is THE product. The old
> `napari_app/` napari-plugin UI has been **deleted**; its engine-agnostic ML
> core moved to a new Qt-free package **`velum_core/`**. Wherever this doc
> still says `napari_app/…` for a *core* module (engines, analysis, benchmark,
> cohort, advisor, tiling, volume_stitch, inference_cache, engine_registry, or
> anything under `core/`), read `velum_core/…`. The repo map below is updated;
> some prose further down may still lag.

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
studio/                THE PRODUCT (PyQt6 desktop app — NOT napari)
  app.py               entry point: window chrome + sidebar + screen stack
                       (Home · Projects · Segment · Models & Train · Dashboard)
  screens.py extra_screens.py guide_screen.py   the non-Segment screens
  workspace.py         the Segment screen: three-pane IDE (Images/Layers ·
                       canvas · Segment/Results inspector) wired to a
                       SegmentController
  canvas.py            our OWN Qt image canvas (paint/erase/fill/pick/pan-zoom,
                       n-D) — explicitly not embedded napari
  layer_model.py       our OWN evented layer list + LabelsLayer (reproduces
                       napari's Labels interaction model 1:1, no napari dep)
  *_controller.py      Qt-free, unit-tested product logic: segment/project/
                       train/dashboard/assistant controllers (take plain dicts
                       + callbacks, not Qt widgets — see tests/)
  components.py theme.py icons.py motion.py overlays.py   the UI kit
  assets/icon.png      the app icon (see docs/app_icon/, docs/velum/PACKAGING.md)

velum_core/         THE ML CORE (engine-agnostic, Qt-free, no napari) — what
                       studio imports as its backend. Extracted 2026-07-21 from
                       the deleted napari_app/.
  predict_controller.py  THE single prediction choke point: config build +
                       predict/batch/benchmark orchestration. Plain dicts +
                       callbacks; unit-tested without Qt/torch.
  engine_registry.py   EngineSpec + register/get/all_engines — the pluggable
                       engine interface predict_controller dispatches through
  engines.py           Cellpose-SAM engine + registers the built-ins
                       (cellseg1, cellpose)
  engines_sam2.py      SAM2 engine (lazy `sam2` import; degrades to
                       available()=False when absent) — flagship for z-stacks
  inference_cache.py   model + ViT-embedding cache (changing only thresholds
                       skips the encoder)
  tiling.py            native-resolution tiled inference for large 2-D images
  volume_stitch.py     engine-agnostic z-stack/time-lapse instance linking
                       (IoU-based) — stitches per-plane masks into one n-D volume
  advisor.py           the Assistant's offline diagnostic engine + Ollama bridge
  analysis.py          per-cell morphometry (skimage regionprops); 3-D schema
                       for z-stack results
  benchmark.py         instance F1/AP vs ground truth
  cohort.py            batch/population aggregation
  channels.py train_model.py train_state_manager.py experiment_tracking.py
  tuning_loop.py       (multi-channel IO · training entry + state · Aim tracking
                       · auto-tune loop)

repo-root ML libs (shared, imported by velum_core — do not delete):
  segment_anything/    vendored SAM fork (incl. the mask-NMS generators)
  peft/                LoRA implementation for SAM
  data/                dataset + image IO (data/utils.py has read/resize)
  predict.py cellseg1_train.py metrics.py sampler.py mask_nms.py
  cell_loss.py set_environment.py gpu_memory_tracker.py project_root.py
  device_utils.py      shared CUDA-capability check (a GPU can be "available"
                       per torch yet ship no kernels for the installed
                       build's CUDA version — see its docstring); used by
                       set_environment.py and studio/hardware.py

server/                THE MULTI-USER BACKEND (opt-in, additive) — the accounts
                       + shared-database contour the desktop apps never had, for
                       team/collaborative use and a future web deployment. A
                       dependency-free *foundation* (no HTTP tier yet — see
                       server/README.md): stdlib sqlite3 in WAL mode by default,
                       Postgres-portable, stateless-token auth, RBAC + immutable
                       audit log, and the Label-Studio-shaped Organization →
                       Project → Task → Annotation → Review model. Pure stdlib,
                       so server/tests/ runs in CI's light `test` group.
  security.py rbac.py  scrypt passwords + opaque session/API-key tokens; 6 roles
                       + a permission matrix + privilege-escalation guard
  validation.py errors.py  field validation/normalisation; the exception set
  models.py db.py      entity dataclasses; sqlite3 connection factory + schema +
                       migrations (thread-local conns, WAL, FK cascade)
  repository.py service.py  data access (one repo per entity, plain SQL) + the
                       business API (Auth/ApiKey/Org/Project/Task/Annotation/
                       Audit services; ServerApp.create() is the front door)

tests/                 pytest suite (pure-logic, no GPU/GUI)
.github/workflows/     CI (runs the pure-logic suite on py3.11/3.12)
checkpoints/ data_store/   bundled weights + sample data (data_store/ is
                       gitignored, created locally by scripts/setup.sh / first
                       run — do not delete, paths reference it)
docs/                  project-wide docs (BACKLOG · AUDIT_2026 · CHANGELOG ·
                       AGENT_KICKOFF_PROMPT) + docs/velum/ (the Velum app's own
                       doc set: ARCHITECTURE · DESIGN · ROADMAP · OVERVIEW ·
                       PACKAGING · its own BACKLOG/CHANGELOG). See docs/README.md.
scripts/               shell tooling: setup.sh (env + SAM weights),
                       build_bundle.sh + make_app.sh (packaging)
README.md              human-facing front door (this file is the agent one)
CLAUDE.md              one line, imports this file — see the Claude Code
                       note above; don't put instructions here directly
```

## Environment & how to run things

Packaging is a real `pyproject.toml` (setuptools): `pip install -e .` installs
the app + the `velum` / `cellseg1` console launchers. Runtime deps live in
`[project.dependencies]`; exact known-good pins are in `requirements.txt` (the
lock). The pure-logic **test** deps are a PEP 735 dependency-group
(`pip install --group test` — no torch/PyQt6). Use the existing
conda env for actual work:

- **Python with all deps:** `/opt/homebrew/Caskroom/miniforge/base/envs/cellseg1/bin/python`
  (Python 3.11; numpy/torch/skimage/cv2/napari/tifffile/pytest present).
- **Run the tests:** `<that python> -m pytest`  (fast, < 1 s, no GPU).
- **Install from source:** `pip install -e .`  (or `bash scripts/setup.sh` to
  also create the env + fetch SAM weights).
- **Run the app:** `bash run_studio.sh` or `velum` / `cellseg1`  (needs a real
  display + SAM weights; **cannot be driven headless** in CI or an agent sandbox).

**No conda, or a fresh Linux box?** The path above is one session's original
macOS setup and won't exist elsewhere — confirmed 2026-07-18 on an Arch Linux
laptop with no conda/mamba at all and a system Python too new to use directly
(Arch ships a rolling-release Python past this project's `requires-python`
`<3.13` cap, and it's externally-managed with no `pip`). `uv`
(https://docs.astral.sh/uv/) is a solid conda-free substitute:
```
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install "packaging>=24.2" setuptools wheel
uv pip install --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt          # drop both flags above if there's a real
                                  # NVIDIA GPU to install CUDA wheels for
uv pip install --no-build-isolation -e . --no-deps
```
The `packaging`/`setuptools` pre-install works around a real uv build-isolation
gap hit on that box: `setuptools>=77`'s license-expression normalizer needs a
newer `packaging` than uv's isolated build env supplied on its own
(`ImportError: Cannot import packaging.licenses`) — pre-installing both into
the venv and building with `--no-build-isolation` sidesteps it. Still keep
the two-command `requirements.txt` + `-e . --no-deps` split from that file's
own comment (not `pip install -r requirements.txt -e .` in one call) — same
reasoning, plus a second failure mode this session hit doing it as one uv
call: uv can start building the local package before every resolved
dependency (e.g. `napari`'s own `pydantic` via `npe2`) is actually installed.

### Verifying changes without a display

You usually cannot launch the Studio GUI. Verify what you can:

1. `python -m py_compile <file>` for syntax.
2. Headless **import** check (works for Qt modules — import doesn't need a
   display, only instantiating a `QApplication`/`Viewer` does):
   ```
   QT_QPA_PLATFORM=offscreen PYTHONPATH=. <python> -c "import studio.workspace"
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
throwaway venv that installs *only* the declared group — run from the repo
root, **no trailing `.`**: that would also pull in the project's own
`[project.dependencies]` (torch/napari/PyQt6 — everything this check exists
to exclude), silently turning it back into the full-env check it's supposed
to replace (`pytest.ini`'s `pythonpath = .` is what makes `velum_core`/`data`/
etc. importable with nothing installed, so the package itself never needs to
be):
```
python3.12 -m venv /tmp/civenv && /tmp/civenv/bin/pip install --group test \
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
  `velum_core/predict_controller.py`. `PredictController` in the same module owns config
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
