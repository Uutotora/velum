# Bundled assets

**`icon.png`** — the Velum app icon, 1024×1024 RGBA (source
resolution for crisp scaling to any size the OS requests). Set as
`QApplication.setWindowIcon()` in `studio/app.py`'s `main()` — this is what
macOS shows as the Dock tile for the running (unbundled — `run_studio.sh`
launches the interpreter directly, no `.app` bundle yet) process. Loaded via
`load_icon()`, which degrades to a null `QIcon` (Qt's own safe default, no
Dock override) if the file is missing, rather than raising.

This is the **Default** variant of the icon design, used **full-bleed** (the
raw Icon Composer export — the rounded square fills the whole canvas). We
deliberately do *not* hand-pad it to the macOS grid: the app currently runs
**unbundled** (`run_studio.sh` launches the `python3.11` interpreter, so the
Dock tile is `python3.11`'s window icon, drawn by Qt as-is with no system
treatment), and faking the Tahoe squircle margin by eye just chased its tail.
The correct margin/masking is applied automatically once Studio is a real
`.app` bundle — see [`docs/velum/PACKAGING.md`](../../docs/velum/PACKAGING.md).

The full source set (every iOS variant + a full-bleed `AppIcon.iconset` and
`AppIcon.icns`) lives in [`docs/app_icon/`](../../docs/app_icon/).
