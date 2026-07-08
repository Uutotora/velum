# Overview — CellSeg1 Studio

## What it is

A world-class desktop application for cell instance segmentation, built to
compete on UX with Figma / Linear / Label Studio. Same ML core as the classic
CellSeg1 app (three engines: CellSeg1·LoRA, Cellpose-SAM, SAM 2), but wrapped
in a product that owns its window and is organised around **Projects**.

The design was set by an interactive HTML north-star mockup, agreed with the
product owner, and then reproduced natively.

**Primary design reference: Label Studio.** We borrow its *product structure* —
projects, the workspace, the modern sidebar, spacious clean panels, the visual
hierarchy — **not** its exact look (our palette, type and icons are our own).
See `DESIGN.md`.

**Own the UI, reuse the logic.** We build our **own** everything on the surface
— our own canvas (we are **not** embedding napari), our own icons, our own
settings, our own tools — and reuse the classic app's proven **functionality**
(the ML core: engines, predict, train, morphometry) by wrapping it under the
new design, so we don't rewrite the hard parts from scratch.

## The current phase: **design skeleton**

Right now this branch is deliberately **all design, no logic**:

- Every mockup screen is reproduced in **native PyQt6** — Home, Projects, the
  Segment workspace (our **own** layers panel · canvas · inspector), Models &
  Train, Dashboard — plus the overlays (Assistant drawer, Logs console, ⌘K
  command palette, toast).
- It renders **static demo content** (`studio/demo.py`) and gives
  only light visual feedback. There is **no** napari, torch, model inference,
  file IO, or project persistence. `import napari` / `import torch` never runs.
- The window is **frameless with rounded corners** and our own dark title bar
  (own traffic lights, native move/resize), so it reads as a product, not a
  Qt window.

Why strip the logic? The half-wired earlier version mixed the new shell with
the raw legacy `PredictWidget`, which felt like "napari with a skin". Resetting
to a pure, faithful design skeleton gives a clean, consistent target to build
against — then each tab is wired properly, in isolation, with its own plan.

## Ground rules

- **Studio lives on its own branch — never merged into `main`.** `main` holds
  the classic napari app plus other important, unrelated work; keep all Studio
  work on the Studio branch (`worktree-studio-app`). See AGENT_PROMPT → *Git*.
- **Keep the classic app untouched.** `napari_app/main.py`, `run_napari.sh`,
  the `cellseg1` console script must stay byte-for-byte. Studio ships behind
  `run_studio.sh` / `cellseg1-studio`.
- **Design fidelity first.** Match `DESIGN.md`. When wiring a tab, the look
  must not regress — behaviour is added *under* the existing design.
- **No logic leaks into the skeleton's shared modules.** Keep `theme`,
  `components`, `paint`, `demo`, screens free of torch/napari so the app stays
  light and the pure-logic tests run in CI's light group. Wire real deps only
  inside the tab you're building, lazily imported.
- **One tab at a time.** Take a tab from `BACKLOG.md`, give it its own task
  list, wire it end to end (data + interactions), test what's testable
  headless, note what needs a GUI/GPU, ship. Then the next tab.

## Run & verify

```bash
bash run_studio.sh                                     # launch (pure PyQt6)
QT_QPA_PLATFORM=offscreen <cellseg1-python> -m pytest studio/tests -q   # Studio's own tests
QT_QPA_PLATFORM=offscreen PYTHONPATH=. <python> -c "import studio.app"   # import check
```

GUI behaviour (the live look, animations, real rendering) can't be verified in
a headless sandbox — always say so in a summary.
