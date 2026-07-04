# CellSeg1 — Engineering Backlog (agent-actionable)

The machine-readable companion to `AUDIT_2026.md`. Each task has a **goal**,
**why**, **acceptance criteria** (how you know it's done), **touch** (files),
and a rough **size**. Work top-down within a priority band: take the top
unchecked P0, satisfy its acceptance criteria, add tests, commit, push, tick
it off. Keep this file honest — check items only when their criteria are met.

Legend — size: S (hours) · M (day) · L (multi-day). Priority: P0 ship-blocker
for a credible product · P1 differentiation · P2 later.

---

## ✅ Done

- [x] **Due-diligence audit** → `AUDIT_2026.md`
- [x] **Test + CI foundation** — pytest suite (analysis/benchmark/cohort/
      advisor/tiling), `pytest.ini`, GitHub Actions on py3.11/3.12.
- [x] **Tiled inference core** — `napari_app/tiling.py` (plan/stitch/
      tiled_predict) + 13 tests.
- [x] **Tiled inference wired into Predict** — "Large image" toggle,
      `_predict_tiled`, off by default.
- [x] **Remove Streamlit GUI** — shared logic moved to `napari_app/core/`.
- [x] **Remove paper artifacts** — figures/, experiment_in_paper/, video/,
      visualize_cell.py.

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

### [ ] Real multi-channel support  · M
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

### [ ] Microscopy formats (OME-TIFF / ND2 / CZI / LIF)  · M
- **Goal:** open the formats microscopes actually produce, with pixel size
  read from metadata.
- **Why:** users can't get their data in today; PNG/plain-TIFF only.
- **Acceptance:** at least OME-TIFF + ND2 open with correct dims and
  auto-filled µm/pixel; unknown formats degrade gracefully.
- **Touch:** `data/utils.py`, image-load path in the widget.
- **Verify:** tests on a tiny fixture per format (or mock the reader).

### [ ] Packaging + dependency lock  · S
- **Goal:** `pip install -e .` works; deps are pinned; napari entry point.
- **Why:** install is a bespoke shell script; no versioning, no reproducible
  env; `requirements.txt` is stale (Streamlit/ray leftovers).
- **Acceptance:** a real `pyproject.toml` with deps + a `napari.manifest`
  entry point; `requirements.txt` removed or regenerated; CI installs from it.
- **Touch:** new `pyproject.toml`, `requirements*.txt`, `.github/workflows/`.

### [ ] Split the `predict_widget` god-object  · M
- **Goal:** separate prediction logic from the Qt view.
- **Why:** ~1.8k lines mixing UI + threading + IO + eval + batch is untestable
  and every change is risky.
- **Acceptance:** a `PredictController` (pure-ish, unit-tested) owns config
  build + predict/batch/benchmark orchestration; the widget only wires UI to
  it; behaviour unchanged; new controller tests added.
- **Touch:** `napari_app/widgets/predict_widget.py`, new `napari_app/core/`.

### [ ] Rename `streamlit_storage/` → `data_store/` (or similar)  · S
- **Goal:** the misleading Streamlit-era name is gone.
- **Why:** Streamlit is removed; the dir now holds weights + samples.
- **Acceptance:** dir renamed; every path reference updated (grep
  `streamlit_storage`); `setup_napari.sh` updated; app still finds weights.
- **Touch:** repo-wide grep; `setup_napari.sh`, widgets, configs.

---

## P1 — differentiation

### [ ] SAM 2 engine (3D / video)  · L
- **Goal:** add SAM 2 as an engine for z-stacks and time-lapse.
- **Why:** confocal/lightsheet/organoids are 3D; the current pipeline is 2D.
- **Acceptance:** an `Engine` entry that segments a z-stack and stitches
  instances across z; napari shows the n-D labels.
- **Touch:** `napari_app/engines.py` (make it a registry), tiling for 3D.

### [ ] Engine registry + plugins  · M
- **Goal:** turn the two hard-coded engines into a registry so StarDist/
  InstanSeg/Micro-SAM/DeepCell can be added.
- **Acceptance:** engines register via a small interface (`predict(image,
  params) -> label mask`); the UI lists whatever is registered.
- **Touch:** `napari_app/engines.py`, `predict_widget` engine selector.

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
- **Acceptance:** a manifest (json) + optional bundle; a "reproduce" path that
  re-runs from it and matches.
- **Touch:** new `napari_app/core/provenance.py`, predict/export paths.

### [ ] Built-in statistics + auto-report  · M
- **Goal:** compare conditions (t-test/Mann-Whitney) and emit a figure-ready
  report from a cohort.
- **Touch:** `cohort.py`, `widgets/cohort_window.py`, new report module.

---

## P2 — platform / enterprise (own product surface; see AUDIT_2026 §8)

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
- Link deeper rationale to `AUDIT_2026.md` sections rather than duplicating it.
