#!/bin/bash
# One-time setup for the CellSeg1 napari desktop app.
# Creates a conda env "cellseg1", installs dependencies, and downloads the
# SAM vit_h backbone weights the bundled checkpoints need.
#
# Usage:   bash setup_napari.sh
# Then:    bash run_napari.sh
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Pick a conda-like tool (mamba is faster if present).
if command -v mamba >/dev/null 2>&1; then CONDA=mamba
elif command -v conda >/dev/null 2>&1; then CONDA=conda
else
    echo "ERROR: conda/mamba not found. Install Miniforge from https://conda-forge.org/download/"
    exit 1
fi

echo "==> [1/3] Creating env 'cellseg1' (python 3.11)…"
if conda env list | grep -qE '[/ ]cellseg1$'; then
    echo "    env already exists — skipping create"
else
    $CONDA create -n cellseg1 python=3.11 -y
fi

echo "==> [2/3] Installing Python dependencies (this can take a few minutes)…"
$CONDA run -n cellseg1 python -m pip install --upgrade pip
# Editable install from pyproject.toml (single source of truth for deps).
$CONDA run -n cellseg1 python -m pip install -e "$DIR"

echo "==> [3/3] Downloading SAM vit_h backbone weights (~2.5 GB, one time)…"
BACKBONE_DIR="$DIR/data_store/sam_backbone"
mkdir -p "$BACKBONE_DIR"
WEIGHTS="$BACKBONE_DIR/sam_vit_h_4b8939.pth"
if [ -f "$WEIGHTS" ]; then
    echo "    weights already present — skipping"
else
    curl -L --fail -o "$WEIGHTS" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
fi

mkdir -p "$DIR/data_store/test_images"

echo ""
echo "✓ Setup complete. Launch the app with:"
echo "    bash run_napari.sh"
