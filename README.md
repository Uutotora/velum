<p align="center">
  <img src="studio/assets/icon.png" width="120" alt="CellSeg1 app icon">
</p>

<h1 align="center">CellSeg1</h1>

A desktop app for **cell instance segmentation** in microscopy images, built
as a [napari](https://napari.org) plugin. Two interchangeable engines produce
instance masks; everything downstream (morphometry, cohort statistics, an
offline diagnostic Assistant, export) works the same regardless of which one
ran:

- **CellSeg1** — a SAM ViT backbone fine-tuned with **LoRA** from a single
  annotated image ("one-shot"). No large labeled dataset required.
- **Cellpose-SAM** — a zero-shot generalist engine; strong out-of-the-box
  accuracy with no training or checkpoint.

Target users are microscopists and cell biologists, not ML engineers.

## Status

Early-stage. It runs real segmentation, real quantitative morphometry, and
tiled inference on large images — but it's a single-user desktop app, not yet
an enterprise platform (no auth, audit trail, cloud, or team features). See
[`docs/AUDIT_2026.md`](docs/AUDIT_2026.md) for an unvarnished gap analysis and
[`docs/BACKLOG.md`](docs/BACKLOG.md) for what's being worked on next.

## Install

Requires Python 3.10–3.12.

```bash
pip install -e .
```

or, to also create the environment and fetch SAM weights:

```bash
bash setup_napari.sh
```

## Run

```bash
cellseg1
# or: bash run_napari.sh
```

Needs a display and downloaded SAM weights (`setup_napari.sh` fetches them).
CellSeg1 also registers as a napari plugin — accessible from *Plugins ▸
CellSeg1* if napari is launched some other way.

## Repository layout

See [`AGENTS.md`](AGENTS.md) for the full map — it's written for AI coding
agents working in this repo, but it's the most accurate orientation for a
human contributor too. In short: `napari_app/` is the product, the repo root
holds the shared ML core (SAM fork, LoRA/PEFT, training/prediction scripts),
and `docs/` holds the task queue, strategic audit, and changelog.

## License

[Apache 2.0](LICENSE).
