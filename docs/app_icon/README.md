# App icon

Source art for the CellSeg1 Studio app icon.

- **`AppIcon.icns`** — the packaged macOS icon, ready for a future `.app`
  bundle (built from `AppIcon.iconset/` with `iconutil`).
- **`AppIcon.iconset/`** — the `iconutil` input (`icon_16x16.png` …
  `icon_512x512@2x.png`), the **macOS-padded** Default variant.
- **`exports/`** — the raw design export: every iOS variant
  (Default / Dark / ClearLight / ClearDark / TintedLight / TintedDark) at every
  size (16 → 1024, @1x/@2x/@3x). The **Default** variant is the one shipped.

> **Padding matters.** The raw exports are **iOS** icons — the rounded square
> fills the whole canvas edge-to-edge. macOS Dock icons instead sit on a grid
> with a transparent margin (the artwork is ~0.77 of the canvas, centred);
> dropping a full-bleed iOS icon straight in makes it look oversized next to
> system icons. So `studio/assets/icon.png` and this iconset are the Default
> art **scaled to ~0.772 and centred** on a transparent 1024² canvas, not a
> raw copy of `Default-1024`.

The running app loads the single 1024×1024
[`studio/assets/icon.png`](../../studio/assets/icon.png) via
`QApplication.setWindowIcon()`.

## Changing the icon

1. Re-export the design (keep the `Default` variant as the shipped one) into
   `exports/`.
2. Re-pad it to the macOS grid and regenerate `studio/assets/icon.png` + this
   iconset (0.772 content ratio, centred on a transparent 1024² canvas — see
   the commit that added this folder for the exact snippet).
3. Rebuild the packaging artifact:
   ```sh
   iconutil -c icns docs/app_icon/AppIcon.iconset -o docs/app_icon/AppIcon.icns
   ```
