# App icon

Source art for the Velum app icon.

- **`AppIcon.icns`** — the macOS icon, built from `AppIcon.iconset/` with
  `iconutil` (full-bleed Default; see the note below).
- **`AppIcon.iconset/`** — the `iconutil` input (`icon_16x16.png` …
  `icon_512x512@2x.png`), the full-bleed Default variant.
- **`exports/`** — the raw design export from **Apple Icon Composer**: every
  iOS variant (Default / Dark / ClearLight / ClearDark / TintedLight /
  TintedDark) at every size (16 → 1024, @1x/@2x/@3x). The **Default** variant
  is the one shipped.

> **On sizing.** These are **full-bleed** icons — the rounded square fills the
> canvas edge-to-edge (that's what Icon Composer exports; Xcode normally adds
> the macOS margin at build time). While Studio runs **unbundled** (the Dock
> tile is `python3.11`'s Qt window icon, drawn as-is with no system treatment),
> there's no perfect margin to match by hand, so we ship it full-bleed and let
> the OS do the right thing once it's a real `.app`. See
> [`docs/velum/PACKAGING.md`](../../docs/velum/PACKAGING.md).

The running app loads the single 1024×1024
[`studio/assets/icon.png`](../../studio/assets/icon.png) via
`QApplication.setWindowIcon()` — a copy of `exports/…Default-1024@1x.png`.

## Changing the icon

1. Re-export from Icon Composer into `exports/` (keep `Default` as shipped).
2. Copy the new `Default-1024` over `studio/assets/icon.png`, regenerate the
   iconset sizes from it, then rebuild the `.icns`:
   ```sh
   iconutil -c icns docs/app_icon/AppIcon.iconset -o docs/app_icon/AppIcon.icns
   ```
3. For the real macOS-margined icon, prefer Icon Composer's `.icon` bundle at
   `.app` packaging time (see `docs/velum/PACKAGING.md`) rather than hand-padding.
