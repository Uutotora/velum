#!/bin/bash
PYTHONPATH="$(dirname "$0")" \
  /opt/homebrew/Caskroom/miniconda/base/envs/cellseg1/bin/python \
  "$(dirname "$0")/napari_app/main.py"
