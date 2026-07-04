#!/bin/bash
# Launch the CellSeg1 napari desktop app.
# Resolution order for the Python interpreter:
#   1. $CELLSEG1_PYTHON if you set it explicitly
#   2. a conda/mamba env named "cellseg1" if one exists
#   3. whatever "python" is on PATH (e.g. an activated venv)
DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$DIR"

if [ -n "$CELLSEG1_PYTHON" ]; then
    exec "$CELLSEG1_PYTHON" "$DIR/napari_app/main.py"
elif command -v conda >/dev/null 2>&1 && conda env list | grep -qE '[/ ]cellseg1$'; then
    exec conda run --no-capture-output -n cellseg1 python "$DIR/napari_app/main.py"
else
    exec python "$DIR/napari_app/main.py"
fi
