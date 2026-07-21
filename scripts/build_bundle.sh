#!/bin/bash
# Build a SELF-CONTAINED Velum bundle with PyInstaller — the kind you
# ship to someone who has neither the repo nor a Python env. Produces a macOS
# .app (+ .dmg) or a Linux dist folder (+ .tar.gz), auto-detected by OS.
#
# This is the "release" packaging (docs/velum/PACKAGING.md, Mode 2) — heavier and
# slower than the dev-launcher (scripts/make_app.sh). It bundles Python + torch +
# PyQt6 + the app, so the artifact is large (~1.5–2.5 GB, torch dominates). The
# ~2.5 GB SAM ViT-H backbone is NOT bundled — it downloads on first use into the
# app's data store (small LoRA checkpoints in checkpoints/ ARE bundled).
#
# Prereqs: run inside an env that already has the app's runtime deps installed
# (pip install -e . ) plus pyinstaller. On CI this is the release workflow;
# locally: `pip install -e . pyinstaller` then `bash scripts/build_bundle.sh`.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
NAME="Velum"
OS="$(uname -s)"
DIST="$REPO/dist"
BUILD="$REPO/build"
rm -rf "$DIST" "$BUILD"

DATASEP=":"   # PyInstaller --add-data separator is ':' on macOS and Linux

COMMON=(
  --noconfirm --clean --windowed
  --name "$NAME"
  --distpath "$DIST" --workpath "$BUILD" --specpath "$BUILD"
  # heavy packages whose data/ext modules PyInstaller can't fully trace on its own
  --collect-all torch --collect-all torchvision
  --collect-all skimage --collect-all scipy --collect-all sklearn
  --collect-all cellpose --collect-all numpy --collect-all cv2
  --collect-all tifffile --collect-all matplotlib
  # the app's own packages (pure-python, but keep them whole)
  --collect-submodules studio --collect-submodules velum_core
  --collect-submodules segment_anything --collect-submodules peft
  # bundled assets + small LoRA checkpoints (NOT the big SAM backbone).
  # Source paths are ABSOLUTE: --add-data resolves the source relative to
  # --specpath (build/), not the cwd, so a relative "studio/fonts" is looked
  # for under build/ and fails ("Unable to find .../build/studio/fonts").
  --add-data "${REPO}/studio/fonts${DATASEP}studio/fonts"
  --add-data "${REPO}/studio/assets${DATASEP}studio/assets"
  --add-data "${REPO}/checkpoints${DATASEP}checkpoints"
)

if [ "$OS" = "Darwin" ]; then
  echo "==> macOS build"
  pyinstaller "${COMMON[@]}" \
    --icon "${REPO}/docs/app_icon/AppIcon.icns" \
    --osx-bundle-identifier "com.velum.app" \
    studio/app.py
  APP="$DIST/$NAME.app"
  # Ad-hoc codesign so Gatekeeper lets a locally-built app run.
  codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || echo "(codesign skipped)"
  echo "==> DMG (drag-to-Applications layout)"
  DMG="$DIST/Velum.dmg"
  rm -f "$DMG"
  # Stage the app next to an /Applications symlink so opening the DMG shows the
  # standard "drag the app onto Applications" install window, like every normal
  # macOS download — not just a lone .app.
  STAGE="$(mktemp -d)/dmgroot"
  mkdir -p "$STAGE"
  cp -R "$APP" "$STAGE/"
  ln -s /Applications "$STAGE/Applications"
  hdiutil create -volname "$NAME" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
  echo "Built: $APP"
  echo "       $DMG"
else
  echo "==> Linux build"
  pyinstaller "${COMMON[@]}" studio/app.py
  echo "==> tar.gz"
  TARBALL="$DIST/Velum-linux-x86_64.tar.gz"
  ( cd "$DIST" && tar czf "$(basename "$TARBALL")" "$NAME" )
  echo "Built: $DIST/$NAME/"
  echo "       $TARBALL"
fi
