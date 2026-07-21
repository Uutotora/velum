# Bundled assets

**`icon.png`** — the CellSeg1 Studio app icon, 1024×1024 RGBA (source
resolution for crisp scaling to any size the OS requests). Set as
`QApplication.setWindowIcon()` in `studio/app.py`'s `main()` — this is what
macOS shows as the Dock tile for the running (unbundled — `run_studio.sh`
launches the interpreter directly, no `.app` bundle yet) process. Loaded via
`load_icon()`, which degrades to a null `QIcon` (Qt's own safe default, no
Dock override) if the file is missing, rather than raising.

This is the **Default** variant of the icon design, **padded to the macOS icon
grid** — the raw export is a full-bleed iOS icon (rounded square edge-to-edge),
which looks oversized in the Dock next to system icons, so the art is scaled to
~0.772 of the canvas and centred on transparent padding. The full source set
(every iOS variant + a ready `AppIcon.iconset` and generated `AppIcon.icns`)
and the exact re-padding recipe live in
[`docs/app_icon/`](../../docs/app_icon/). When Studio is packaged as a real
`.app` bundle (see `docstudio/BACKLOG.md`'s "Packaging" entry), use that
`AppIcon.icns`.
