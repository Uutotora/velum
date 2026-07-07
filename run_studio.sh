#!/bin/bash
# Launch CellSeg1 Studio — the standalone desktop app (napari embedded inside).
#
# This is the NEW entry point. The classic napari-plugin app is untouched and
# still launches via run_napari.sh / the `cellseg1` command — run that to
# revert instantly.
#
# Resolution order for the Python interpreter (same as run_napari.sh):
#   1. $CELLSEG1_PYTHON if set explicitly
#   2. a conda/mamba env named "cellseg1" if one exists
#   3. whatever "python" is on PATH (e.g. an activated venv)
DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$DIR"

# Run the entry FILE directly (not `python -m`): app.py bootstraps sys.path to
# the repo root itself, so this works from any cwd. `-m` would prepend the
# caller's cwd to sys.path ahead of PYTHONPATH and import the wrong napari_app
# when launched from another checkout. (Same pattern as run_napari.sh.)
APP="$DIR/napari_app/studio/app.py"
if [ -n "$CELLSEG1_PYTHON" ]; then
    exec "$CELLSEG1_PYTHON" "$APP"
elif command -v conda >/dev/null 2>&1 && conda env list | grep -qE '[/ ]cellseg1$'; then
    exec conda run --no-capture-output -n cellseg1 python "$APP"
else
    exec python "$APP"
fi
