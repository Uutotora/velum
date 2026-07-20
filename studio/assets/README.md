# Bundled assets

**`icon.png`** — the CellSeg1 Studio app icon, 1024×1024 RGBA (source
resolution for crisp scaling to any size the OS requests). Set as
`QApplication.setWindowIcon()` in `studio/app.py`'s `main()` — this is what
macOS shows as the Dock tile for the running (unbundled — `run_studio.sh`
launches the interpreter directly, no `.app` bundle yet) process. Loaded via
`load_icon()`, which degrades to a null `QIcon` (Qt's own safe default, no
Dock override) if the file is missing, rather than raising.

If Studio is later packaged as a real `.app` bundle (see
`docstudio/BACKLOG.md`'s "Packaging" entry), this same source image is what
an `.icns` iconset would be generated from.
