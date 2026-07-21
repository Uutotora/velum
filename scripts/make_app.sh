#!/bin/bash
# Build a thin macOS .app launcher for Velum.
#
# The .app contains ONLY an icon + a launch script — NOT the Python code. It
# runs the live source from this git checkout, so you update the app by editing
# code and relaunching; you never rebuild the .app for a code change. Rebuild
# only when the icon, name, or launcher itself changes. See docs/velum/PACKAGING.md.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"        # repo root (this script lives in scripts/)
APP="$REPO/dist/Velum.app"
CONTENTS="$APP/Contents"

echo "Repo:  $REPO"
echo "App:   $APP"

rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

# ── App icon: pad the full-bleed art to the macOS grid (~0.875) and build .icns.
# A plain .icns is drawn as-is by macOS (no automatic Tahoe margin), so we bake
# the margin in here to match system icons' footprint. Source of truth is the
# single studio/assets/icon.png.
PYBIN="${CELLSEG1_PYTHON:-}"
if [ -z "$PYBIN" ]; then
  if [ -x "/opt/homebrew/Caskroom/miniforge/base/envs/cellseg1/bin/python" ]; then
    PYBIN="/opt/homebrew/Caskroom/miniforge/base/envs/cellseg1/bin/python"
  else
    PYBIN="python3"
  fi
fi
ICONSET="$(mktemp -d)/AppIcon.iconset"; mkdir -p "$ICONSET"
"$PYBIN" - "$REPO/studio/assets/icon.png" "$ICONSET" <<'PY'
import sys
from PIL import Image
src = Image.open(sys.argv[1]).convert("RGBA")
iconset = sys.argv[2]
CANVAS = 1024; RATIO = 0.875
content = round(CANVAS * RATIO)
art = src.resize((content, content), Image.LANCZOS)
base = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
mx = (CANVAS - content) // 2
top = round((CANVAS - content) * 18 / 32)   # slight downward bias, like system icons
base.alpha_composite(art, (mx, top))
sizes = {"icon_16x16.png":16,"icon_16x16@2x.png":32,"icon_32x32.png":32,"icon_32x32@2x.png":64,
         "icon_128x128.png":128,"icon_128x128@2x.png":256,"icon_256x256.png":256,"icon_256x256@2x.png":512,
         "icon_512x512.png":512,"icon_512x512@2x.png":1024}
import os
for name, s in sizes.items():
    base.resize((s, s), Image.LANCZOS).save(os.path.join(iconset, name))
print("iconset built (0.875 macOS grid)")
PY
iconutil -c icns "$ICONSET" -o "$CONTENTS/Resources/AppIcon.icns"

# ── Launcher: resolve the interpreter WITHOUT relying on an interactive shell's
# PATH (a .app starts with a bare environment — conda/python are not on PATH).
cat > "$CONTENTS/MacOS/launch" <<LAUNCH
#!/bin/bash
REPO="$REPO"
LOG="\$HOME/Library/Logs/Velum.log"
mkdir -p "\$(dirname "\$LOG")"
if [ -n "\${CELLSEG1_PYTHON:-}" ] && [ -x "\${CELLSEG1_PYTHON}" ]; then
  PY="\$CELLSEG1_PYTHON"
elif [ -x "/opt/homebrew/Caskroom/miniforge/base/envs/cellseg1/bin/python" ]; then
  PY="/opt/homebrew/Caskroom/miniforge/base/envs/cellseg1/bin/python"
elif command -v conda >/dev/null 2>&1; then
  PY="conda-run"
else
  PY="python3"
fi
export PYTHONPATH="\$REPO"
{
  echo "=== \$(date) launching Velum (PY=\$PY) ==="
  if [ "\$PY" = "conda-run" ]; then
    exec conda run --no-capture-output -n cellseg1 python "\$REPO/studio/app.py"
  else
    exec "\$PY" "\$REPO/studio/app.py"
  fi
} >> "\$LOG" 2>&1
LAUNCH
chmod +x "$CONTENTS/MacOS/launch"

# ── Info.plist
cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Velum</string>
  <key>CFBundleDisplayName</key><string>Velum</string>
  <key>CFBundleIdentifier</key><string>com.velum.app</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleExecutable</key><string>launch</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSUIElement</key><false/>
</dict>
</plist>
PLIST

# ── Ad-hoc codesign (no Apple Developer ID) — good enough for a personal dev
# app, but Gatekeeper still gates the FIRST launch (see the note printed below).
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || echo "(codesign skipped)"

cat <<EOF

Built: $APP

This is the DEV launcher (runs your live source — edit code, relaunch, no rebuild).

Simplest way to run during development (no Gatekeeper, no dialog):
    bash run_studio.sh

To use the .app from the Dock instead:
  1. Drag "$APP" into your Dock (or /Applications).
  2. First launch is blocked by Gatekeeper because it's not signed by Apple
     ("Apple could not verify..."). Allow it ONCE:
     System Settings > Privacy & Security > scroll down > "Open Anyway".
     After that it launches from the Dock like any app.

Logs: ~/Library/Logs/Velum.log
EOF
