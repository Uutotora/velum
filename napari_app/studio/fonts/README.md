# Bundled fonts

**Figtree** (`Figtree-Regular.ttf`, `Figtree-SemiBold.ttf`) — the CellSeg1
Studio UI typeface. Figtree is released under the **SIL Open Font License 1.1**
(<https://openfontlicense.org>), which permits bundling and redistribution with
an application. Designed by Erik Kennedy. Upstream:
<https://github.com/erikdkennedy/figtree>.

Loaded at startup via `QFontDatabase.addApplicationFont` in
`napari_app/studio/app.py`; the design tokens reference the family name with a
system fallback chain so the app degrades gracefully if the face fails to load.
