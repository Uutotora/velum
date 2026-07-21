# Changelog — Velum

What actually shipped in Studio, dated, newest first. (The repo-wide log is
`docs/CHANGELOG.md`; this one is Studio-specific.)

---

---

## 2026-07-21 — Renamed the product to **Velum**

CellSeg1 Studio -> **Velum**. The product is no longer positioned as cells-only:
its one-shot LoRA engine learns to segment *any* object from a single annotated
example, so the name shed the "CellSeg". Renamed the user-facing name everywhere
it shows: window title + title bar, the sidebar wordmark ("Velum." with a small
accent dot), version footer, the `.app` bundle (name/identifier `com.velum.app`),
the release artifacts (`Velum.dmg`, `Velum-linux-x86_64.tar.gz`), the log file
(`~/Library/Logs/Velum.log`), README (broadened the pitch to "one-shot instance
segmentation of any object"), and the docs. Added a `velum` console launcher
(`cellseg1`/`cellseg1-studio` kept as aliases). Deliberately NOT renamed: the
Python packages `studio/` and `velum_core/`, and the **CellSeg1 engine** (one
of the three engines — that's its method name, not the product).

---

## 2026-07-21 — Downloadable releases: PyInstaller bundle + DMG/tar.gz CI

Wire up a self-contained distributable (docs/velum/PACKAGING.md Mode 2):
`scripts/build_bundle.sh` builds a PyInstaller bundle — a macOS `.app` + `.dmg`,
or a Linux dist folder + `.tar.gz` — bundling Python/torch/PyQt6/the app + small
LoRA checkpoints (the ~2.5 GB SAM backbone downloads on first use).
`.github/workflows/release.yml` builds both on a `v*` tag and attaches them to
the GitHub Release (macos + ubuntu runners). Not verified end-to-end here
(PyInstaller+torch needs a real build run to shake out hidden imports; DMG is
ad-hoc signed) — documented as such.


## 2026-07-21 — New app icon (v2) + show it on GitHub

Replaced the app icon again with the new design (the **Default**, full-colour
variant — a soft light rounded-square with a teal→blue gradient blob). Removed
the previous `docs/AppIcon.iconset/`. The full design export now lives properly
on the branch under `docs/app_icon/`: the raw iOS exports (all variants, 16→1024)
in `exports/`, plus a ready `AppIcon.iconset/` and a generated `AppIcon.icns`
for a future `.app` bundle. The running app still loads the single
`studio/assets/icon.png`. Added the icon to the top of both `README.md` and
`docs/velum/README.md`, centred GitHub-style, so it shows on the repo's front
page. Verified `load_icon()` picks up the new art.

**Decision: ship it full-bleed for now.** Hand-padding the icon to match the
Dock was a tail-chase *because the app is unbundled* — the Dock tile is the
`python3.11` interpreter's Qt window icon (drawn as-is, no macOS icon
treatment), so there's no stable system margin to match by eye (full-bleed read
"too big", 0.772 "too small", 0.875 "too big" again). Reverted
`studio/assets/icon.png` and the iconset to the **full-bleed Default** export
and stopped hand-padding. The proper margin/masking is the OS's job once Studio
is a real `.app`; wrote **[docs/velum/PACKAGING.md](PACKAGING.md)** covering a
thin dev-launcher `.app` (build once, then edit code + relaunch to update — no
rebuild), the agent prompt to build it, the icon options for a real bundle
(pre-padded `.icns` at ~0.875, or Icon Composer's `.icon` for the full Tahoe
"Liquid Glass" look), and the vibe-coding update loop.

---

## 2026-07-21 — New app icon

Replaced the Dock/app icon with the new light rounded-square design: swapped
`studio/assets/icon.png` (the 1024×1024 source `load_icon()` loads) for the new
art and added the full multi-size source set under `docs/AppIcon.iconset/`
(16→512 @1x/@2x) as the tracked origin for a future `.icns` bundle. Verified
`load_icon()` picks it up and renders across Dock sizes offscreen.

---

## 2026-07-21 — Segment: sync the cell count after edits, and fix "hard to select" image rows

Two more reported bugs, both reproduced with offscreen scripts before fixing.

**1. "Project card says 122 cells, Results say 45."** Editing a mask
(paint/erase/fill/undo/redo) updated only the canvas legend; the Results panel
stats and the *persisted* project stats stayed frozen at the last predict run,
so the project card drifted from what was on screen. Now a debounced sync
(gated on a cheap content fingerprint so pure selection/visibility/reorder
churn is ignored, and off the per-mouse-move paint path) recomputes the result
after an edit settles and persists the new **distinct-cell** count. Added
`LabelsLayer.n_labels` (distinct non-zero instances, unlike `max_label` = the
highest id) so the legend can't disagree with the Results panel after a
mid-range cell is erased; the exact count is computed once on settle, not
`np.unique` on every paint tick (16 ms on a 2k² mask).

**2. Image rows were "hard to select — they don't select."** `SwipeRow`'s
tap-vs-swipe threshold was 4 px, so a click that drifted even a few px left —
routine on a trackpad — was swallowed as a swipe and never selected the row.
Raised the slop to 12 px and only move the row once a swipe is actually
committed, so normal taps select while a deliberate left-swipe still reveals
Delete. (Real event delivery through the mouse-transparent foreground was
verified fine with QTest; the threshold was the whole bug.)

Covered by `test_editing_the_mask_keeps_results_and_card_count_in_sync`,
`test_labels_layer_n_labels_counts_distinct_not_max_id`, and
`test_swiperow_small_wobble_still_taps`; full Studio suite green.

---

## 2026-07-21 — Segment/Results: kill the HiDPI text seams and the rebuild-overlap

Two reported bugs in the Segment screen's right-hand inspector, both fixed and
regression-tested.

**1. "Чёрточки у начала текста" — a hairline down the left of every text block.**
Each side panel painted its divider as a `border-left`/`border-right:1px solid`
on its own `WA_StyledBackground` `QFrame`. That inset the panel's content by the
border, and at HiDPI it left a `border`-coloured 1px seam down the left edge of
every text block inside — the number `100`, each stat tile, the calibration
field, `Instance ID`, each accordion title. Isolated to exactly this border by
bisection (flattening the border colour, then removing it, made the seams
vanish; the letter-spacing / opaque-box / nested-layout theories all rendered
clean). Fixed by moving both dividers onto the **canvas** edges
(`border-left`/`right` on the viewport, a dark image with no text to seam) — the
divider reads identically, the panels no longer seam.

**2. "Refine…/Measurements overlap" + a ghost "Measure" over the calibration
hint.** `_rebuild_results_pane` lays its hero number, stat tiles and action
buttons into *nested* QHBox/QGrid layouts (`addLayout`). `_clear_layout` only
removed direct child *widgets* and never recursed into nested layouts, so every
rebuild (pixel-calibration edit, GT load, colour-by change) orphaned the prior
copies — they kept the container as parent and stayed visible, stacking up.
`_clear_layout` now recurses into nested layouts.

Both verified with offscreen renders (seams gone, no ghost) and covered by
`test_rebuilding_results_pane_does_not_accumulate_widgets` +
`test_clear_layout_recurses_into_nested_layouts`; full Studio suite green.

---

## 2026-07-21 — Fix the faint "ruled border" boxes around panel text/labels

Reported against both Segment side panels: text (section headings, field
labels, the breadcrumb, layer/image names, GT metrics) had a faint rectangle
ruled around it — "линии... как будто границы расчерчены".

Root cause is the exact opaque-box bug the `label()` helper already documents,
just in the labels that *don't* go through it. A `QLabel` given its own
instance stylesheet (colour/size) but no `background` resolves its background
via the app-wide `QWidget{background:bg}` rule instead of `QLabel{background:
transparent}` once it's nested inside a styled ancestor — so it paints an
opaque `bg`-toned box. On a panel toned `inset`/`surface` (both a shade off
`bg`), that box's edges read as a border ruled around the text. `bg`=#0d0f13
vs `inset`=#101318 vs `surface`=#15181e in dark, so it's a real, checkable
tone gap, not a guess.

Fixed by adding `background:transparent` to every such label: the shared
`FieldRow` name label and `GroupLabel`, `SelectBox`'s value, `StatTile`'s
value/caption, the `Accordion` title, the sidebar version, and the Segment
inline labels (breadcrumb name/separator, layer-controls titles, image/layer
row names, GT metric values). Regression tests assert `FieldRow`/`GroupLabel`
carry the transparent background. Verified with an offscreen render (boxes
gone); full suite green.

---

## 2026-07-21 — Segment: drag a layer up/down to reorder it (napari parity)

`LayerList.move` existed but nothing drove it. Layer rows are draggable now:
press and drag a row vertically, release to drop it at the new position
(z-order = list order, so this changes what draws on top); a plain click still
selects, and selection follows the moved layer. Both outcomes are deferred one
tick (the same list-rebuild SIP hazard the other rows guard). Rows also gained
a fixed height + a trailing stretch so the list stays compact and top-aligned
(they used to expand to fill) — which also makes the drag's row-step maths
predictable. Tests cover `_move_layer` (reorder + selection-follows) and a
real drag-down gesture reordering rather than selecting. Full suite green.

That completes the napari-parity batch the redesign feedback asked for
(undo/redo, tool shortcuts, layer reorder, image contrast, tool-strip dedup)
alongside the structural work (resizable/collapsible panes, swipe-to-delete,
empty-state).

---

## 2026-07-21 — Segment: editable image contrast (min/max sliders + Auto) — napari parity

The image layer's contrast was a read-only "lo – hi" badge. It's now editable,
like napari: a **min** and a **max** slider (each mapping 0..1 across the
image's own data range, kept from crossing) plus an **Auto** button that
stretches to the 1–99 percentile (falling back to full min/max on a flat
image). The renderer already honoured `contrast_limits`, so moving a slider
re-renders live. Auto rebuilds the controls so the sliders jump to the new
limits. Tests cover the percentile auto-stretch and the no-cross clamp. Full
suite green.

---

## 2026-07-21 — Segment: single-key tool shortcuts + a de-duplicated canvas tool strip (napari parity)

- **Tool shortcuts.** With the canvas focused: B paint, E erase, F fill, G
  polygon, K pick, V pan/zoom — napari-style. Handled in `Canvas.keyPressEvent`
  (only while the canvas has focus, so they never fight a text field elsewhere)
  and gated on no modifier, so Ctrl/Cmd combos pass through. A new
  `on_mode_change` callback refreshes the labels tool-row highlight + toolbars.
- **De-duplicated floating strip.** The canvas tool strip carried Paint
  (duplicating the Labels tool row, now also the B key) and reused the exact
  "target" glyph for both Pan and Home. It's navigation + prompts only now —
  Pan, Add-prompt-point, Home — each with its own icon.

Verified: tests for every shortcut key (+ modifier is ignored) and the strip's
new icon set/highlighting. Full suite green. Not verified: live keypress focus
behaviour on a real window.

---

## 2026-07-21 — Segment: a real empty-state for the Layers pane (no more stray lines on a blank panel)

Direct feedback on the left panel with nothing loaded: "непонятные линии если
там ничего нету" — a floating add-layer toolbar hovering over a blank area with
a stray scrollbar groove, which read as broken.

The Layers pane is now a two-state QStackedWidget. With no image loaded it
shows a single centred empty-state — a soft image badge, "No image loaded", one
line of guidance, and an "Add images" button — instead of the toolbar + empty
list + controls scroll. Once an image loads (any layer exists) it swaps to the
real content. Driven by `_sync_layers_pane_state()` off every layer-list change
(and once at construction). Verified with an offscreen render of a project with
no images (clean centred card, no stray lines) and tests for the state
switching (empty on no-image, content on load, back to empty when the last
image is removed).

---

## 2026-07-21 — Segment: swipe an image row left to delete it (iOS-style)

Direct feedback: a referenced-but-unreadable image just sat stuck in the list
with no obvious way to remove it — the ask was an iPhone-style swipe-left.

- New `components.SwipeRow`: the foreground content slides left under the
  drag to reveal a red Delete backdrop; releasing past a commit threshold
  fires delete, a shorter swipe springs back (animated), and a plain tap still
  selects. The content is made mouse-transparent so the whole row is one drag
  target, and it's opaque (panel-inset when unselected) so the red only shows
  once you swipe.
- Image rows use it: tap selects (as before), swipe-left removes.
  `WorkspaceScreen._remove_image` drops the image from the project, persists,
  cleans up the in-project *copy* on disk (only ever a file under the
  project's own images dir — never an external source we merely reference),
  and — if it was the open image — moves to the next one or clears the canvas.
  This is finally an easy way to clear out a dead `~/Downloads` reference.

Both row callbacks are deferred one event-loop tick (the same SIP virtual-call
hazard the tap handler already guarded). Full suite green; tests cover SwipeRow
(tap vs commit-swipe vs spring-back vs offset clamp) and `_remove_image`
(drop+persist, switch-to-next, clear-on-last, on-disk copy cleanup). Not
verified: the live drag/spring-back animation on a real pointer.

---

## 2026-07-21 — Segment: the three panes are now resizable and collapsible (product-standard layout)

Direct feedback: the side panels were a locked, un-hideable rail ("нельзя её
растягивать и скрывать, что неправильно" / "нельзя скрывать правую панель") —
not how real tools behave.

- The old fixed 240 / canvas / 320 layout is now a `QSplitter`: drag either
  handle to rebalance (panels clamp to sensible min/max so nothing collapses
  to an unreadable sliver), and the canvas never collapses.
- Two topbar toggles — a `panel_left` button by the breadcrumb and a
  `panel_right` button past Run — hide/show each side pane, handing the
  reclaimed width to the canvas. They tint active while their pane is visible.
  Toggle logic keys off `isHidden()` so it's correct whether or not the window
  is realised.

First step of the larger Segment design overhaul the same feedback asked for
(empty-state/stray-line cleanup, swipe-to-delete image rows, typography, and
the remaining napari-parity items are tracked as follow-ups). Verified with an
offscreen render (both toggles present + active) and tests for the splitter
shape (3 panes, canvas non-collapsible) and each toggle's hide/restore. Not
verified: live handle-drag resizing.

---

## 2026-07-21 — Segment sidebar: declutter the Labels controls, dedupe icons, drop the dead Transform tool

Direct feedback: the left panel was too cramped ("слишком кучно"), some icons
duplicated others, and a tool did nothing.

- **Advanced fold.** The Labels controls were a wall of ten stacked fields.
  The five less-often-touched ones (blending, direct-colour mode, contour
  width, n-edit-dim, and the contiguous/preserve/show-selected flags) move
  into a collapsed "Advanced" accordion; the common ones (tools, opacity,
  label, palette, brush size) stay visible. napari folds the same knobs a
  level deeper too.
- **New label button.** A "+" next to the label stepper selects `max + 1` —
  napari's increment-selected-label, the fast way to start a fresh instance.
- **Dead tool removed.** The "Transform" tool did exactly what Pan/zoom did
  (the canvas treated both modes identically — it never actually transformed),
  so it's gone from the 8-icon Labels tool row, now 7.
- **De-duplicated glyph.** The canvas bar's "Transpose" reused the "shuffle"
  icon, which also means "shuffle label colours" two toolbars over. It has its
  own `transpose` (axis-swap) glyph now.

Verified with an offscreen render (grouped panel, Advanced collapsed, distinct
transpose glyph, "+" on the label row) and tests for the New-label max+1
behaviour, the transpose key rename, and Transform's removal from the tool row.
Not verified: live accordion expand animation.

---

## 2026-07-21 — Segment: mask editing gets undo / redo (napari parity)

The canvas edited masks in place with no way back — paint a wrong stroke and
it was permanent. Direct feedback flagged the missing undo/redo outright.

- `LabelsLayer` now keeps a bounded per-layer edit history (`begin_edit()` +
  `undo()`/`redo()`/`can_undo`/`can_redo`), snapshotting `data` before a
  mutation. The canvas calls `begin_edit()` once at the *start* of a stroke —
  a whole paint/erase drag, or one fill/polygon commit — so a single undo
  reverts the whole stroke, not one brush dab, and a fresh edit invalidates
  the redo branch like every editor. History is capped (24 steps) to stay
  memory-bounded on volumes.
- Canvas `undo()`/`redo()` operate on the active Labels layer, repaint, and
  report; they no-op safely with an empty history or no labels layer.
- Wired to ⌘Z / ⇧⌘Z (Ctrl+Z / Ctrl+Y off macOS, via `QKeySequence.StandardKey`
  so it's per-platform correct) and two new buttons at the head of the canvas
  bar (new `undo`/`redo` icons). Picking a colour makes no history entry.

Verified: offscreen render shows the two buttons in the canvas bar; tests
cover the model history (undo/redo/no-op/redo-invalidation/cap/one-step-stroke),
the canvas wiring through real mouse drags (paint stroke + fill are single undo
steps; pick creates none), and the workspace shortcut delegation. Not verified:
the live ⌘Z keypress against a real focused window (headless can't press keys),
but the same handler the shortcut calls is exercised directly.

---

## 2026-07-21 — Segment topbar: engine moves to a centred rounded badge; breadcrumb ancestor gets a real hover

Direct feedback on the workspace topbar: the engine sat in a square-ish chip
crammed next to the project name, and the "Projects" breadcrumb ancestor
didn't feel interactive.

- **Engine badge, centred.** The engine is now an `EngineChip` (rounded pill
  with an engine-hued dot, same hue the Projects-tab card uses) centred in the
  bar between two equal stretches, in its own holder that's rebuilt in place on
  each project switch. It reads as a standalone status badge for the whole
  workspace rather than part of the title. The old inline `Chip` next to the
  name is gone.
- **Breadcrumb hover.** "Projects" brightens from muted to full text colour on
  hover (Label-Studio-style ancestor link) and keeps the pointer cursor + click
  back to the Projects tab. The "/" separator is now its own label between the
  link and the name, instead of being baked into the name's rich text — which
  is what freed the engine to move to the centre.

Verified with an offscreen render of the Segment screen (badge centred, dot
present, breadcrumb split) and tests covering the badge type/placement, the
name/separator split, and the breadcrumb navigation. Not verified: the live
hover colour transition (needs a real pointer).

---

## 2026-07-21 — Segment: images are copied into the project on import; unreadable files no longer storm the log

Reported against a real project: opening it threw "Can't load image / Cannot
read: …/Downloads/download (2).png" every time, the selected image never
appeared on the canvas, and the terminal filled with hundreds of
`imread_(...): can't open/read file: check file path/integrity` warnings.

Root cause: the file was fine — macOS's per-folder privacy gate (TCC) blocks
reading `~/Downloads` (also `~/Desktop`, `~/Documents`) unless the launching
terminal has Full Disk Access, so `cv2.imread` returns `None` and `open()`
raises `PermissionError: Operation not permitted`. Projects stored the
original external path, so every open re-hit the blocked file. Confirmed by
reproducing the exact `PermissionError` directly on the reported path (the
file exists and is intact; only the folder is gated).

Fixes:
- **Copy-on-import.** `ProjectStore.import_image/import_images` copy chosen
  files into `<project>/images/` and store *that* path. Both entry points use
  it (the New Project dialog and the Images pane's +/drag-drop). Reading at
  import time works even without Full Disk Access because the file dialog /
  Finder drag grants a transient read scope; the copy then lives in the app's
  own store and is always readable afterwards. A source that can't be copied
  right now falls back to referencing the original path (kept, not dropped).
- **No more log storm.** Thumbnails are cached per path (`_thumb_cache`), so
  an unreadable file is attempted once, not re-decoded on every wholesale
  rebuild of the images pane (select / add / project-load each rebuilt it).
- **Actionable error.** When a preview still can't load, the toast now names
  the real cause — a moved/deleted file vs a macOS privacy block — with the
  fix for each (`_read_error_hint`), instead of a raw `Cannot read: …`.

Existing projects that still point at `~/Downloads` need the source re-imported
(now copied in) or the terminal granted Full Disk Access — the app can't grant
itself a TCC permission. Verified: full studio suite green; new tests cover
the copy/dedupe/fallback/idempotence of import, the thumbnail cache (incl. the
unreadable-file case), and the error-hint classification. Not verified: the
real on-screen canvas for a TCC-blocked file (can't reproduce the GUI
headless) — but new imports are copied, so they read normally.

---

## 2026-07-21 — Home: recent-projects entrance animation only plays on a real change, and carries no shadows

Reported again, directly: the recent-projects list on Home animates badly
every time you open the tab ("плохая анимация очень... каждый раз"). The
892ccef pass had already dropped the whole-screen fade for Home and scoped
it to just the recent section, but two things still made it stutter and
replay:

1. **It re-ran on every single visit.** `refresh()` unconditionally rebuilt
   the recent section and faded it in, even when you'd tabbed away and back
   without creating or changing anything — the common case. Now `refresh()`
   fingerprints the list (`HomeScreen._recent_sig`, built from raw store
   fields: id/updated_at/n_images/n_cells/engine/name) and returns early on
   an identical revisit: no rebuild, no fade. Only a genuine change (a new /
   renamed project, new images or cells, a re-order) re-animates. The very
   first `navigate("home")` at launch matches the construction snapshot, so
   it's a calm no-op too.

2. **The fade still composited a drop-shadow per row every frame.** Even
   scoped, fading a `QGraphicsOpacityEffect` over rows that each carry an
   `install_hover_lift` `QGraphicsDropShadowEffect` forces Qt to
   re-rasterise every one of those nested effects on every frame — the same
   expensive mechanism root-caused for the Projects-grid scroll stutter.
   Fixed by building the rebuilt rows *without* their hover shadows
   (`_recent_section(with_hover=False)`), fading the plain rows, then
   installing the shadows once the fade settles (`_install_recent_hover` via
   `fade_in`'s new `on_finished` hook). Same visible result, none of the
   per-frame re-rasterisation.

The waving-hand greeting still plays once per visit (cheap, no nested
effects). `motion.fade_in` gained an optional, guarded `on_finished`
callback; no other caller passes it, so their behaviour is byte-for-byte
unchanged. Verified: full studio suite green (exit 0), plus an offscreen
render of Home to confirm the recent list still lays out correctly and a
repeat `navigate("home")` no-ops without a rebuild. Not verified: real
on-screen frame timing (can't drive the GUI headless) — but the change
removes work per frame and per visit, so it can only be lighter.

---

## 2026-07-20 — App icon: fixed padding to match the macOS icon-grid convention

Same-day follow-up: reported directly against a real Dock screenshot next
to the icon's actual neighbours (Finder, Notes, Safari, App Store, ...) —
the icon read visibly larger than every other Dock icon around it.

Root cause: the previous pass scaled the cropped glyph to fill ~94% of the
1024×1024 canvas, versus the documented macOS Big Sur+ icon-grid convention
of ~80.5% (an ~824px glyph inside the 1024px canvas — the ~10%-per-side
transparent margin every stock macOS icon actually has, which is *why*
icons with completely different glyph shapes still read as the same visual
size sitting in a Dock row). Not a subjective "looks a bit big" call — a
concrete, checkable ratio that was simply wrong. Fixed by rescaling to the
824/1024 target and re-centring on a fresh transparent canvas; verified by
compositing the old and new icon side by side at a realistic 96px Dock tile
size before trusting it, not just re-eyeballing the full-resolution source.

No code changes — `studio/assets/icon.png` swapped again, `load_icon()`
unchanged. Full suite: 736 passed, 0 failed (asset-only change).

---

## 2026-07-20 — App icon replaced with a proper macOS-style rounded-square version

Same-day follow-up: product owner supplied a second, better logo (the same
blob glyph, now presented as a proper macOS-style rounded-square icon —
blue-to-purple gradient background, soft drop shadow, the kind Apple's own
Human Interface Guidelines icon templates produce) and asked for it in place
of the first.

The source PNG wasn't ready to use as-is: it was flat RGB (no alpha
channel), the actual icon content — the rounded square — sat on a plain
light-grey mockup/presentation backdrop that needed removing, not keeping.
Naive colour-keying (treat anything close to the background colour as
transparent) would have been wrong here specifically: the icon's own glyph
is a *white* blob, close enough in colour to the light-grey backdrop that a
flat threshold would have punched a hole through it. Used flood-fill /
connected-component labelling instead (`scipy.ndimage.label`): only
background-coloured pixels *connected to the image's own border* count as
background, so the enclosed white glyph — never touching the border — stays
opaque regardless of its own colour, while the true backdrop, wherever it
reads similarly, is correctly removed.

First attempt at the cutout left a faint light halo around the icon's edge
against a dark backdrop — the anti-aliased transition band between icon and
original backdrop is a real colour *blend*, and treating those pixels as
fully opaque kept their too-light blended colour. Fixed by eroding the
foreground mask a further ~16px past the hard edge (discarding the
ambiguous blended band entirely, which a smooth, large-radius glyph like
this one can easily afford to lose without visibly changing shape) before
a small Gaussian blur for the actual anti-aliasing — checked at 2x zoom on
a dark backdrop specifically (where any residual light fringe would be most
visible) before trusting it. Final result centred onto a clean, transparent
1024×1024 canvas (the standard icon-authoring resolution — a non-square
crop can subtly squish when scaled to fill a square Dock tile slot).

No code changes needed beyond replacing `studio/assets/icon.png` itself —
`load_icon()` from the previous entry already does the rest. Verified
offscreen: loads non-null at 1024×1024, and a small (256px) preview
composited over both a dark and a light backdrop shows no fringing or
artifacts. Full suite: 736 passed, 0 failed (unchanged from the previous
entry — this is an asset swap, not a logic change).

---

## 2026-07-20 — A real app icon: the macOS Dock tile, not embedded in the UI

Product owner supplied a logo (a blue-to-purple glowing organic blob, on
brand with the app's own iris-indigo primary colour) with one explicit
constraint: it's for the **Dock icon** specifically, not for embedding
anywhere inside the app's own UI (no sidebar wordmark replacement, nothing
"sucked into" a screen) — an initial assumption about where it was headed
was corrected before any wiring work started.

Added `studio/assets/icon.png` (1024×1024 RGBA, the source resolution app
icons are conventionally authored at, for crisp scaling to whatever size
the OS actually requests — verified the alpha channel is genuinely
transparent at the corners, not a baked-in grey backdrop, before trusting
it). `studio/app.py` gained `load_icon()` (mirrors `load_fonts()`'s own
missing-asset-degrades-quietly pattern — a null `QIcon`, not a raise, if the
file isn't there) and wired it into `main()`: `QApplication.setWindowIcon()`
right after constructing `app`, which is what macOS actually uses as the
Dock tile for an unbundled running process (`run_studio.sh` launches the
interpreter directly — there's no `.app` bundle yet, see `docs/velum/
BACKLOG.md`'s "Packaging" entry, which this same source image would feed
into whenever that's built). `StudioWindow` also gets it via
`setWindowIcon()` for the window/Cmd+Tab-switcher icon.

One regression test, confirmed to fail when the asset is (temporarily)
missing. Full suite: 736 passed, 0 failed. Verified offscreen: the icon
loads non-null, exposes its full 1024×1024 source, and still reads cleanly
at a realistic 128px Dock-tile size.

---

## 2026-07-20 — NewProjectDialog centred on the sidebar too, not just the content area

Third fresh report the same day against the same "no project" work.
Screenshot: `NewProjectDialog`'s panel sitting noticeably left of where it
should be, with the sidebar still fully bright/undimmed behind the scrim.
Direct feedback: "окно надо центровать не по всему приложению а по центру
области с эмодзи" (the window needs to be centred not against the whole
app, but against the centre of the [content] area).

Root cause: `NewProjectDialog` is one shared instance, triggered from every
screen (Home, Projects, Guide, and now Workspace's own no-project view), so
it was parented directly to the whole `StudioWindow` — its scrim and its
panel's centring both measured against the *full* window width, sidebar
included, rather than just the content area to the sidebar's right.
`ConfirmDialog`/`ProjectSettingsDialog` never had this problem: both are
parented to whichever *screen* opened them, and a screen's own bounds are
already exactly the content area (screens live inside `StudioWindow._stack`,
to the right of the sidebar) — `NewProjectDialog` was the one dialog that
couldn't do the same, since no single screen owns it.

Fixed by parenting it to `StudioWindow._stack` (the `QStackedWidget` all
screens live in) instead of the window itself — required moving `self._stack
= QStackedWidget()`'s construction earlier in `app.py._build_ui()`, ahead of
`NewProjectDialog`, but no change to `NewProjectDialog`'s own `place()`
logic: `self.parentWidget()` now simply *is* the content area, so the
existing `setGeometry(0, 0, p.width(), p.height())` does the right thing
automatically. Verified this doesn't fight `QStackedWidget`'s own page
management (a raw, non-`addWidget`-added child floats and raises normally,
same established pattern as `workspace.py`'s own floating viewport chrome)
and survives `toggle_theme()`'s full `_build_ui()` teardown/rebuild (which
already destroys and recreates `NewProjectDialog` on every toggle, so
reparenting introduced no new lifecycle coupling).

One regression test (`test_app_wiring.py`), confirmed to fail against the
pre-fix parenting. Full suite: 735 passed, 0 failed. Verified offscreen in
both themes and across a theme toggle: sidebar now stays fully undimmed,
panel centres within the content area exactly like every other dialog in
the app already did.

---

## 2026-07-20 — Two more rounds on Segment's "no project" state: the topbar, and every scrim dialog in the app

Same-day follow-up to the "Home motion polish + Segment's 'no project'
state" entry directly below. Two fresh, specific reports against the
just-shipped fix, both real:

**1. The topbar stayed visible with no project open.** The previous fix
replaced the three-panel body with a full-screen empty state but left
`WorkspaceScreen`'s topbar untouched above it -- breadcrumb reading "No
project selected," an engine chip reading "No project," Export/Run
disabled but still sitting there. Reported directly: "почему верхняя
панель осталась? если проект не выбран то она не нужна" (why did the top
panel stay? if no project is selected it isn't needed). Correct once
stated plainly: the no-project view already has its own "Open a Project"
action, making the breadcrumb's "Projects" link redundant, and nothing
else in the bar means anything without a project either. Fixed by hiding
the whole bar (`WorkspaceScreen._topbar_widget.setVisible(project is not
None)`) instead of disabling two buttons inside it -- safe to hide
outright, not just greyed out, because `_start_predict`/`_export_csv`
already guard their own preconditions with a toast regardless of what
triggers them (the command palette lists the identical Run/Export/Save
commands unconditionally for exactly this reason, per `app.py`'s own
existing comment there).

**2. "куда сьехало окно при создании проекта?" (where did the window slide
off to when creating a project?)** -- a screenshot of `NewProjectDialog`
showing its panel oddly placed near the top of the screen with the sidebar
and topbar still fully bright behind it, no visible dimming. Reproduced
offscreen and pixel-sampled before touching anything (a real regression
would be too easy to misdiagnose as "reparent/geometry bug" from the
screenshot alone): the dialog's own geometry was exactly correct (0, 0,
1320, 860 -- the full window), and reproducing the identical click from
Home (a trigger point untouched by any of today's changes) showed the
identical bug, proving this was pre-existing, not a regression from the
topbar/body-stack work, just newly surfaced by testing the "New Project"
button this session added to the no-project view. Root cause: every
scrim-backed modal in the app (`NewProjectDialog`, `ConfirmDialog`,
`ProjectSettingsDialog`, `CommandPalette` -- all four found by grepping for
the literal) used the identical hardcoded `rgba(8,10,20,0.34)`, a value
clearly tuned against light theme's white backdrop (a strong, obvious dim
there) but never re-checked against dark theme, where that RGB nearly
matches dark theme's own `bg` (`#0d0f13`) -- compositing barely moved
anything (confirmed by pixel-sampling a real dialog: sidebar `#101318` ->
`#0e1017`, a few units of drift, not a visible dim). Fixed with a new
theme-independent `theme.SCRIM = "rgba(0,0,0,0.45)"` constant (pure black,
higher alpha -- darkens *any* backdrop by a consistent proportion
regardless of how dark it already is, unlike a colour tuned to one theme's
lightness) referenced from all four call sites, replacing each one's own
copy of the old literal. Re-sampled the same sidebar pixel after the fix:
`#101318` -> `#090b0d`, a real, clearly visible ~45% darkening. Nested
modals (Settings -> Delete confirmation) now compose with a proper visual
depth hierarchy as a side effect, not just the single-dialog case.

Seven new regression tests: one for the topbar-hide (confirmed to fail
against the pre-fix code), and six for the scrim -- a computed-luminance
contrast check (`test_theme.py`) parametrized over both themes' `bg`/
`surface`/`surface2` tokens, asserting the scrim darkens each by at least
30% in relative luminance. Run against the pre-fix literal, exactly the
three dark-theme cases failed (bg 10%, surface 19%, surface2 23% actual
darkening -- all well under the bug's own real-world symptom) while all
three light-theme cases already passed, precisely confirming the
diagnosis: light theme was always fine, only dark theme was broken. Full
suite: 734 passed, 0 failed. Verified offscreen in both themes: Segment
with no project (topbar gone), `NewProjectDialog` from both Segment and
Home, and the nested Settings -> Delete-confirmation flow.

---

## 2026-07-20 — Home motion polish + Segment's "no project" state, corrected same-day

Two product-feel gaps reported directly against the real running app.

**Home: the recent-projects block "shows with a terrible animation every
time."** Root cause: `app.py`'s `navigate()` faded the *entire* HomeScreen
in via a `QGraphicsOpacityEffect` on every single visit, not just the
first. Home's quick-cards, recent rows and aside cards each carry their own
`install_hover_lift`/`soft_shadow` `QGraphicsDropShadowEffect` (up to ~10 at
once); animating an opacity effect on their shared ancestor forces Qt to
re-rasterise every one of those nested effects on every frame of the fade —
the same composited-effects-are-expensive mechanism already root-caused for
the Projects grid's scroll stutter earlier this same day, just triggered by
a repeated opacity animation instead of scrolling, and replaying on *every*
revisit to an already-built, unchanged screen rather than just the first.
Fixed by excluding `"home"` from `app.py`'s generic per-navigation fade
(alongside the pre-existing `"workspace"` exclusion) and replacing it with
two smaller, more deliberate cues in `HomeScreen.refresh()` itself: a light
fade scoped to just the recent-projects list (the part that actually
changed — far fewer nested shadow effects than the whole page), and a new
`components.WavingEmoji` widget playing a one-shot hand-wave rotation next
to "Welcome back" on every visit. `WavingEmoji` is a small self-contained
`QWidget` (`components.py`) with its own `QVariantAnimation` driving a
rotation transform in `paintEvent` around the glyph's own base (not its
centre — a wave pivots at the wrist, a centred rotation reads as a spin);
`page_header()` grew an optional `title_extra` param (unused by every other
caller, so their layout is unchanged) to seat it beside the title text.

**Segment (Workspace) with no project open: "the empty canvas looks sad."**
Landing on Segment fresh showed only Canvas's own plain, tiny "No image
loaded" `paintEvent` text on an otherwise blank dark viewport, with the full
three-panel IDE layout (Images/Layers · canvas + its floating tool strip/
viewer bar · Segment/Results) around it, every panel showing nothing useful.

The first fix attempt only addressed the canvas's own corner: a friendly
emoji + message + "Open a Project" button overlaid on top of the canvas,
leaving the three surrounding panels and the canvas's own floating tool
strip/viewer bar on screen exactly as before — still empty, still
non-functional, still cluttered. Direct feedback caught this immediately
("ты просто добавил текст и эмодзи и все" — you just added text and an
emoji, that's it; "убрать все" — remove all of it) before it shipped.
Corrected same-day into the real fix: the three-panel body and a new
full-screen "no project" view are now two complete alternatives in one
`QStackedWidget` (`WorkspaceScreen._body_stack`), toggled in `_load_project`
— with no project, the three-panel layout isn't just covered by a message,
it *isn't the visible page at all*. The new view (`_no_project_view()`)
uses the app's normal theme tokens (`t['bg']`, `t['text']`, ...), not the
viewport's own always-dark, theme-independent canvas colours — it replaces
the whole body rather than sitting inside that canvas, so it should look
like any other page, light or dark. Two actions: "Open a Project"
(navigates to Projects, reusing the existing breadcrumb callback) and a new
"New Project" (opens `NewProjectDialog` directly — `WorkspaceScreen` grew
an `on_new_project` param, wired in `app.py` alongside the other screens
that already receive it). The topbar's Export/Run buttons — meaningless
with no project, but not part of the swapped body — are now disabled
instead of sitting there clickable-but-useless (`theme.button_qss`'s
existing `:disabled` rules already covered this; they just weren't wired to
anything before).

Ten new regression tests across `test_components.py` (`WavingEmoji`
construction/`play()`/paint), `test_home_wiring.py` (scoped fade + wave
trigger), `test_app_wiring.py` (the `"home"` fade exclusion specifically),
and `test_workspace.py` (`_body_stack` index toggling, both empty-state
buttons, Export/Run enabled-state) — every one individually confirmed to
fail against the pre-fix code (or, for the two brand-new-feature button
tests where "pre-fix" doesn't quite apply, confirmed to fail when the
specific wiring they check was deliberately broken) before being trusted.
Full suite: 728 passed, 0 failed. Verified offscreen in both themes: Home's
header at rest/mid-wave/settled and after a simulated revisit, and
Segment's empty state, its two buttons, the disabled topbar buttons, and
the full three-panel body once a project is actually open.

---

## 2026-07-20 — Toast: the border/overflow bug was never the Undo button

Same-day follow-up to the "Projects tab v2, revised" entry directly below
this one, which had (reasonably, but wrongly) guessed that the Toast's
border-not-fully-enclosing bug was caused by the removed Undo action button
and closed the question by deleting that feature instead of root-causing it.
The product owner then sent a fresh screenshot of a real, running "Project
deleted" toast — border still not fully filled, subtitle text still
overflowing the card — proving the bug lived somewhere else. Two independent
root causes, found this time by reproducing both offscreen with pixel-level
measurement before touching any fix:

**Root cause 1 — `Toast._subtitle` used `setMaximumWidth(280)`, not
`setFixedWidth(280)`.** With nothing else in the widget chain anchoring a
width, a word-wrapping `QLabel`'s natural width for Qt's `heightForWidth`
negotiation settled on an arbitrary, too-narrow value (measured on a real
toast: 137px, not the intended 280px cap) — undersizing the computed height
and letting wrapped text spill past the card's own rounded background. Fixed
in `studio/overlays.py` by switching to `setFixedWidth`.

**Root cause 2 — `components.label()` never set its own `background`,** a
systemic gap in the single most-used text helper in the app. Every label
gets its own instance-level `setStyleSheet()` call (for color/size/weight),
which mattered because a `QLabel` with no instance stylesheet of its own
falls back cleanly to the app-wide `QLabel{background:transparent}` cascade
rule (`theme.build_qss`) — but one with *any* instance stylesheet, nested
inside a `QFrame` that has its own qualified background-setting stylesheet
(any `#ObjectName`-styled card or scrim dialog — `Toast`'s `#Toast` selector
is exactly this shape), instead resolved its background via the more-generic
app-wide `QWidget{background:<bg>}` rule, painting an opaque, wrong-coloured
box instead of staying transparent over the card's own surface. Confirmed by
isolated reproduction (a bare label directly in a styled `QWidget` rendered
fine; the identical label nested inside a styled `QFrame` did not), then by
precise pixel sampling of a real `Toast` showing the exact wrong colour
(`bg` `#0d0f13` where `surface` `#15181e` was expected). Fixed by adding
`background:transparent;` directly into `label()`'s own generated
stylesheet string — one line, but it reaches every label in the app nested
inside a styled frame, including the Project Settings dialog's Danger Zone
card (`project_dialogs.py`'s `#DangerZone` `QFrame`), a second real instance
of the identical shape, re-screenshotted and confirmed clean in both themes
alongside Toast.

Two regression tests added, each individually confirmed to fail against the
pre-fix code before being trusted (house discipline, not skipped this time):
`test_toast_long_subtitle_stays_within_the_cards_own_height` (`studio/tests/
test_overlays.py`, failed pre-fix with `assert 137 == 280`) and
`test_label_background_is_explicitly_transparent_inside_a_styled_frame`
(`studio/tests/test_components.py`, failed pre-fix with `assert '#0d0f13' ==
'#15181e'`). Full suite re-run clean after both fixes: 715 passed, 0 failed.
Re-verified visually offscreen in both themes: the grid, the card kebab
menu, Settings (General + Danger Zone), the nested delete-confirm dialog,
and the post-delete Toast itself with both a short Cyrillic project name
(matching the exact real-world report) and a long English one long enough to
wrap the subtitle across three lines.

The meta-lesson, now twice-confirmed this same day: a plausible-looking fix
that isn't independently reproduced and measured is still a guess, and
offscreen screenshots only catch what you think to screenshot — this bug
survived a "declared done" round specifically because the earlier fix
correlated with the symptom (both touched the same widget) without being
its cause.

---

## 2026-07-20 — Projects tab v2, revised: real usage reverts Trash/Undo, cards lose their cover art

Same-day follow-up to the "Projects tab v2" entry directly below this one.
That entry shipped, was screenshotted offscreen in both themes, and was
believed done — then the product owner actually ran the app (not offscreen)
and sent real screenshots: a rendering bug in the Toast, a rendering bug in
the Trash dialog (window looked cropped after clicking a button inside it),
and, more fundamentally, direct comparison against Label Studio's own
reference screenshots (project cards, the overflow menu, Settings > Danger
Zone) making the case that the whole tab needed to be simpler and closer to
that reference, not just bug-fixed. This is exactly the kind of gap
offscreen-only verification can't close — the project's own `docs/velum/
studio-subproject.md`-equivalent lessons about "always screenshot before
trusting it" cover rendering bugs found *offscreen*; this is the next tier
up, bugs and product-shape problems only surfaced by a real person actually
using the real, rendered app.

**Toast reverted to plain/informational.** The previous entry's "Undo"
action button (`overlays.Toast`'s `action_label`/`on_action` pair) rendered
with its border not fully enclosing the taller, three-line content (title +
wrapped subtitle + the Undo link) — very likely a `QLabel.setWordWrap(True)`
+ single-pass `adjustSize()` interaction (a real, if narrower, Qt quirk
worth remembering, not fully root-caused since the whole feature was about
to have zero callers either way). Since deletion no longer needs an Undo
action at all (see below), the fix was to remove the feature rather than
debug a code path nothing would use afterward — `Toast` is back to exactly
what it was before that entry: title + subtitle, no action button, no
`QToolButton`.

**Deletion simplified to match Label Studio's own Danger Zone exactly.**
Removed entirely: `Project.trashed_at`, `ProjectStore.trash()`/`restore()`/
`trashed()`, `ProjectController.trash_project()`/`restore_project()`/
`list_trashed()`/`delete_project_permanently()`, `TrashDialog`,
`confirm_trash()`, the toolbar's Trash entry point, the standalone
`RenameDialog`. In their place: `ProjectController.delete_project()` (a
thin wrapper straight over the store's own, already-existing hard
`delete()`) and a new `studio/project_dialogs.ProjectSettingsDialog` --
General (editable Project Name/Description, matching Label Studio's own
General Settings fields) and a Danger Zone card (a real, qualified
`#DangerZone` red-tinted `QFrame`, not just red text -- "Deleting this
project removes its images, results, and settings from this device. This
can't be undone." + a `Delete Project` button) inside one compact scrim+
panel modal rather than a separate navigated screen with its own sidebar,
since this product only needs two sections' worth of settings. Deleting
still requires its own nested `ConfirmDialog` -- the one truly irreversible
click in the whole flow -- but there is no Trash to check afterward and
nothing to restore.

**The kebab menu shrank to match Label Studio's own reference exactly:**
their card overflow menu is two items (Settings / Label); ours is now
**Open · Duplicate · Settings**, down from Open/Rename…/Duplicate/Move to
Trash. Rename lives in Settings' own Project Name field now (with the
field's cursor explicitly set to position 0 on open, so a long description
shows its start, not wherever the field's default cursor-at-end scrolled
to) -- Duplicate stayed a direct, undo-free kebab action since it's
additive and trivially reversible by just deleting the copy.

**The single biggest change: every project card, list row, and Home's
recent-projects row lost its decorative cover art entirely.** The original
design had a live-painted "nuclei art" thumbnail (`paint.NucleiView`,
`cover_label()`) with the star/kebab/engine-chip/progress floating on top
of it as an overlay -- direct feedback, with Label Studio's own reference
cards held up for comparison: plain text, a stat row, a footer, zero
imagery anywhere, "почему мне нужны лого проектов" ("why do I need project
logos"). Redesigned to match: a plain header row (engine chip · star ·
kebab, no floating/overlay positioning), then name, description, the
existing stats row (unchanged -- Images/Cells/F1 vs GT was already good),
tags, and a footer (progress % + relative timestamp, replacing the old
dark-overlay corner badges). `cover_label()` (dead after this, zero
remaining callers anywhere) was deleted outright rather than left unused.
`paint.NucleiView` itself was **not** deleted -- it's tested, working
infrastructure with no callers left in Projects/Home specifically, but
`workspace.py` still uses its sibling `nuclei_pixmap()` for the Segment
canvas's own placeholder art, a legitimately different context, and a
future decorative use isn't implausible enough to justify deleting a
tested class over. The Ghost ("+ New Project") card's height was remeasured
and matched to a real card's new (much shorter, no-cover) height so the
grid's trailing cell doesn't stand out as a different size.

**Known, deliberate gaps** (unchanged from the entry below except where
noted): no undo for a deleted project -- a deliberate simplification this
round, not an oversight, matching Label Studio's own Danger Zone having no
undo either; everything else (in-project delete/rename, bulk multi-select,
pagination past the seed scale, multi-user workspaces) as before.

**Tested:** every pure-logic and Qt test touching the removed trash/rename
surface was deleted or rewritten against the new API (not left disabled or
skipped) -- `test_project.py`'s trash section removed outright,
`test_project_controller.py`'s trash tests replaced with `delete_project`
equivalents, `test_project_dialogs.py` substantially rewritten (`ConfirmDialog`
tests kept as-is since that class didn't change; `RenameDialog`/`TrashDialog`
tests replaced with `ProjectSettingsDialog` tests, including its own
scrim-bleed regression test -- this dialog applied the learned
`background:transparent` fix from the start, so this test confirms that
holds rather than finding a new instance of the bug), `test_app_wiring.py`'s
sort-order test fixed for the new card's layout-item chain (verified
directly against a real card before trusting it, not guessed), its trash-
button/rename-dialog tests replaced with settings/delete equivalents. One
genuine test-writing mistake caught and fixed before commit, not after: the
new Settings-dialog scrim-bleed test initially flagged the Danger Zone's own
labels as a false positive (they're supposed to sit on the red tint, not
the plain white panel -- the test's blanket "must equal `surface`" check
needed to exclude the Danger Zone's own descendants and instead assert
they're *not* plain white). Full `pytest studio/tests` green throughout.
Every change screenshotted offscreen in both themes -- including, this
time, the exact interaction sequence the product owner actually hit
(duplicate a project, delete the duplicate, reopen the surfaces involved)
-- but real on-screen feel/click behaviour is still genuinely unverified in
this sandbox; that gap is what caused this whole follow-up entry to exist,
worth remembering rather than re-learning.

---

## 2026-07-20 — Projects tab v2: real deletion, rename/duplicate, sort, a scroll-perf fix

P1 was fully done (see the command-palette entries below, same day) but the
Projects tab itself — marked done back on 2026-07-08 — only ever covered
browse/search/favourite/grid-list. Asked to bring it to "real product" bar
against Label Studio (the project's own design reference) and general
product-dashboard practice; this landed as 6 separate commits, each tested
and screenshotted in both themes before the next one started. Full detail
lives in the commit messages; this is the roll-up. See `docs/velum/
BACKLOG.md`'s "Projects tab v2" entry for the original analysis this was
scoped from.

**Scroll performance, root-caused not guessed.** `ProjectsScreen`'s card
cover (`paint.NucleiView`) regenerated its whole procedural nuclei field
from scratch on *every* repaint, and every card also carried an always-on
`QGraphicsDropShadowEffect` (`install_hover_lift`'s base alpha was 22, not
0) — both fire on every scroll-triggered repaint, for every visible card.
Fixed by caching `NucleiView`'s painted output to an internal pixmap,
rebuilt only on an actual resize — a call-count regression test confirmed
3-vs-0 against the pre-fix code. Secondary polish: the three duplicated
`scroll()`/`_scroll()` helpers (`screens.py`, `workspace.py`) were
consolidated into one shared `components.SmoothScrollArea`, which eases a
discrete mouse-wheel notch via `QPropertyAnimation` instead of an instant
jump (a trackpad's own `pixelDelta` is already smooth and left untouched).

**Deletion — the explicitly-requested feature.** `ProjectStore.delete()`
had existed since the tab shipped but was dead code, called from nowhere.
Rather than wire it straight to the grid, added a soft-delete layer in
front of it: `Project.trashed_at` + `ProjectStore.trash()`/`restore()`
(hard `delete()` stays as the irreversible "Delete Forever" path, reached
only from Trash). Chosen over an immediate hard delete because a project
can represent real segmentation/training work and "reversible by default"
beat interrogation-heavy confirmation for a local single-user app — see
the BACKLOG entry for the full reasoning against Label Studio's own Danger
Zone pattern. UI: a ⋯ overflow menu on every card/row (new `more` icon —
three round dots via the same zero-length-stroke trick `assistant`/`guide`
already use), `studio/project_dialogs.py`'s new `ConfirmDialog` (named
project, red action, reused for both "Move to Trash" and "Delete Forever"),
a Toast "Undo" action (`overlays.Toast` gained an optional `action_label`/
`on_action` pair, purely additive), and `TrashDialog` reached from an
always-visible-but-disabled-when-empty toolbar entry (matches the app's
own "discoverable, not hidden" rule for disabled affordances).

**Rename + Duplicate** round out the kebab menu. `RenameDialog` (a
`QLineEdit`, Cancel/Save, Enter-to-save). Duplicate mirrors Label Studio's
own semantics — settings/tags/image references copy, results don't
(stats/favourite reset, since nothing has run against the copy yet) — and
needs no confirmation, since it's additive and trivially undoable.

**Sort** (Last modified / Name A-Z / Date created / Most cells) — present
in every comparable product and, until now, entirely absent; the grid only
ever had the store's implicit `updated_at` order.

**Four real, pre-existing/newly-introduced bugs found and fixed along the
way, all by actually screenshotting both themes (or measuring real
geometry), not by tests alone:**
1. **`NewProjectDialog` had the same rendering-bug family this file already
   has several entries about** (an unstyled `QWidget()` container inheriting
   the app-wide background rule), glaring in light theme specifically
   because that dialog's scrim is translucent — a scrim-over-`bg` blend is
   visibly darker than scrim-over-`surface`. Pre-existing, never caught
   across "several prior New-Project-dialog sessions" per this file's own
   history, because dark theme (the apparent default review theme) hides
   it almost completely. Fixed at all 8 sites; `TrashDialog` got the
   identical bug in its own header/scroll-content (same fix, matching
   `CommandPalette`'s own `_results_container` precedent).
2. **`ProjectsScreen._confirm_trash`/`TrashDialog._confirm_delete` each
   discarded their `ConfirmDialog`'s return value** — with nothing
   Python-side referencing it, it was fair game for garbage collection
   before a click. Caught by a wiring test, fixed with a kept `self.
   _active_dialog` ref (the same convention `_open_card_menu` already uses
   for its `QMenu`).
3. **The new Sort `SelectBox` rendered as just a bare chevron, no visible
   label text** — its value label (`_ElidingLabel`) uses a horizontally
   `Ignored` size policy (correct for its original fixed-width-panel
   context, the Segment inspector), so `SelectBox`'s own `sizeHint` never
   reserved room for text in a toolbar with room to spare. Fixed with an
   explicit `setMinimumWidth(152)`.
4. **Found in the very last verification pass, not by a test**:
   `TrashDialog`'s row used a plain, non-eliding `label()` for the project
   name, sharing a `QHBoxLayout` with two non-shrinking buttons — a long
   name (a duplicate's own `"<name> copy"` reliably qualifies) forced the
   whole row 469px wide inside a fixed 440px panel, and with the horizontal
   scrollbar explicitly off (`SmoothScrollArea`, matching every other
   scroll area in the app), "Delete Forever" ended up partly outside the
   visible panel — measured directly (`row.width()`, a button's mapped
   geometry), not eyeballed. Fixed by switching the name to
   `components._ElidingLabel` (the same atom `SelectBox`/`Accordion`/
   `StatTile` already use for exactly this shape of problem), which elides
   with "…" instead of forcing the row wider than its container.

**Visual audit against `DESIGN.md`'s rhythm** found the codebase already
disciplined — most "off-rhythm" numbers turned out to be small, deliberate
insets reused identically across sibling call sites, not drift. Found and
fixed exactly one genuine inconsistency: the project card body's padding
was an asymmetric `(15, 14, 15, 15)`, normalised to a uniform 14.

**Known, deliberate gaps** (not built, not silently skipped either):
deleting/renaming from *inside* an open project (Workspace's own breadcrumb
⋯ menu) — only the Projects grid/list has it; bulk multi-select
(trash/tag several projects at once); pagination/virtualisation for a
library much larger than the ~6-project seed set (the grid is a plain
`QGridLayout`, fine at today's scale, untested past it); a literal
"workspaces" concept (Label Studio has one, this app is single-user local,
deliberately out of scope per `OVERVIEW.md`).

**Tested:** ~86 new test cases across 7 commits (pure-logic: trash/restore/
rename/duplicate/sort on the store and controller; Qt: `NucleiView` caching,
`SmoothScrollArea` wheel behaviour, `ConfirmDialog`/`RenameDialog`/
`TrashDialog` in isolation, `Toast`'s new action support, and the full
flows wired through `ProjectsScreen`) plus 3 geometry/pixel-sampling
regression tests for bugs 1/1/4 above (all three confirmed to fail against
the pre-fix/reverted code first, reporting the exact wrong values —
e.g. `469 <= 440` for bug 4). Full `pytest studio/tests` green throughout;
every commit screenshotted offscreen in both themes before moving to the
next. Not verified: real on-screen feel/click behaviour (no display in
this sandbox).

---

## 2026-07-20 — Follow-up #2: a real visual pass, Raycast as the reference

Direct feedback again after the sizing fix: still not right — the search
row read as too wide/undefined, the whole thing didn't read as a *real*
launcher the way Spotlight/Raycast do, and "why is there an ESC button" (a
fair question — no real launcher shows one). Named reference this time:
Raycast specifically, plus "emoji... for quick navigation." A design pass,
not another layout bug.

**Changes, all in `_PaletteRow`/`_build_panel` (`overlays.py`) and
`_build_commands()` (`app.py`):**

- **The ESC chip is gone.** Closing on Escape is a universal enough
  convention that no real launcher spells it out on-screen; Raycast
  doesn't either.
- **Every command now carries an emoji** (`Command.emoji`, new field) —
  🏠 Home, 🔬 Segment, 🧠 Models & Train, ▶️ Run segmentation, 🎯 Benchmark,
  🩺 Diagnose, 🌞/🌙 the *destination* theme, and so on for all ~30 —
  rendered in `_PaletteRow` in place of the existing line-icon set (which
  still renders as a fallback for any command with no emoji, so the field
  is additive, not a breaking change to `Command`'s shape). Emoji are real
  colour glyphs a stylesheet `color:` can't recolour, so a disabled row
  now dims via a real `QGraphicsOpacityEffect` on the whole row instead of
  just muting the text colour — covers the glyph too.
- **Rows read as a real Raycast-style list**: tighter padding (10/8px, was
  17/10), inset 8px from the panel's own edges (was flush), each row's own
  selected-state highlight now a rounded 8px "pill" rather than a flat
  edge-to-edge wash — the shape reads as a distinct rounded rectangle
  instead of a hard-edged bar. Section headers' own margin was tightened
  to match (10px, was 17px) so header text lands under row text instead of
  indented further than it.
- **The footer is now dynamic, not a static legend.** Instead of a fixed
  "↑↓ navigate · ⏎ run · esc close" that never changes and states the
  merely mechanical, the right side now shows the *currently selected
  command's own label* plus a "⏎" hint (blank if the row is disabled) —
  updated live on every arrow-key move and every re-render, telling you
  what Enter actually does rather than just that it does something. The
  left side keeps a small app mark + "CellSeg1 Studio" for context, the
  same "own branding, not a generic legend" idea Raycast's own footer uses
  (its own extension name + primary action, not a fixed instructions bar).

**Verified:** 661 `studio/tests` (up from 655) — 7 new cases (no ESC chip
anywhere in the input row; a command with an emoji renders it; a command
without one falls back to the icon pixmap; the footer tracks the selected
row's label + hint across a move; the hint is blank for a disabled
selection; the footer clears when nothing matches) plus the existing
`_commands()` test fixture extended with one real emoji to exercise both
the emoji and fallback paths side by side. Full repo `tests/` (445)
untouched. The throwaway python3.10 light-group check (592 passed/22
skipped) green. Fresh offscreen screenshots in both themes, with and
without an active project (so both the enabled, full-colour-emoji state
and the dimmed/disabled state were actually checked, not assumed),
confirm the header-alignment fix and the overall look side by side
against the north-star mockup screenshot this whole thread started from.

**Not verified:** how this reads on a real, non-offscreen display, incl.
actual emoji glyph rendering/alignment across different OS font stacks
(offscreen Qt uses whatever emoji font resolution this sandbox has, which
may render individual glyphs slightly differently than a real user's
machine); real Ollama/Custom-API server interaction (unchanged from prior
entries).

---

## 2026-07-20 — Follow-up: the palette read as bloated, and it genuinely was

Direct feedback right after the ship below: the palette looked worse than
the north-star mockup — specifically, badly over-spaced. Right call: a real
bug, not a matter of taste.

**Root cause**: `_results_area` (the scrollable results list) only ever had
`setMaximumHeight(420)` — a ceiling, not an actual size. With few results
(a narrow search, or a fresh install with no project yet), the box still
reserved the *full* 420px, rendering as one or two rows sitting inside a
mostly-empty white rectangle — the "raздуло" (bloated) the mockup, sized
to exactly 6 fixed demo rows and nothing more, never had.

Three separate, compounding issues, found in order by direct measurement
(`.sizeHint()`/`.geometry()` inspection), each confirmed against a real
offscreen screenshot before moving to the next rather than guessed at:

1. **A `QScrollArea`'s `sizeHint()` doesn't track its content's actual
   size** by default — needed `_BoundedScrollArea`, a small subclass whose
   `sizeHint()` reads the content widget's *current* `sizeHint().height()`,
   capped at `_MAX_RESULTS_HEIGHT`, replacing the static
   `setMaximumHeight(420)`. This is what makes the box shrink for a short
   list and only reach the cap (then scroll) for a long one.
2. **A freshly-mutated layout's `sizeHint()` reads back stale (in one case,
   just its own margins — `(0, 12)` for 2 real rows) for one event-loop
   tick** after `_clear_layout` + re-adding rows — confirmed directly:
   correct one tick later, wrong in the same tick. `_rerender()` now defers
   the fit-and-scroll step one tick (`QTimer.singleShot(0, ...)`, guarded
   the same way every other cross-callback hazard in Studio is).
3. **The actual root cause of the panel itself never shrinking**: a plain
   `QFrame`'s default vertical size policy is `Preferred`, which happily
   *grows* to fill whatever space a layout offers — `AlignTop` (already set
   on the outer layout) only governs how a layout **shorter than** its
   available space is positioned, it doesn't stop a `Preferred`-policy sole
   child from being stretched to fill in the first place. Confirmed by
   direct experiment: `panel.sizeHint()` was correctly small the whole
   time; only `outer.addStretch(1)` after the panel — letting the stretch
   absorb the extra space instead of the panel — actually changed the
   on-screen geometry. The original static 6-item version never exposed
   this: its fixed content happened to be large enough, relative to the
   window, that "stretched to fill" and "sized to content" looked the same
   by coincidence.

Also refactored `_apply_selection_styles()`/`_scroll_to_selected()` apart
from the old combined `_restyle_rows()` — the styling half is safe to run
synchronously from arrow-key navigation (existing, already-settled rows),
only the post-*re-render* fit-and-scroll needed the deferred tick.

**Verified**: 655 `studio/tests` (up from 653) — 2 of the previous entry's
own new tests updated for the corrected contract (no more static
`setMaximumHeight` to assert on; no more manual trailing-stretch-inside-
the-results-layout to count), plus 3 new regression tests, including one
that reproduces the *actual* end-to-end symptom (the panel's real
`.height()` across three searches — full → short → back to full — not
just an internal sizeHint) and was confirmed to fail first (804 == 804,
not shrinking) with the stretch fix reverted, before being trusted. Full
repo `tests/` (445) untouched. Fresh offscreen screenshots in both themes
at multiple result-set sizes (2 rows, ~7 rows, the full ~27-command browse
list) confirm the palette now sizes tightly to its actual content, closely
matching the mockup's density, instead of a fixed oversized box.

**Not verified**: how this reads on a real, non-offscreen display, incl.
whether the resize itself is visually smooth frame-to-frame while typing
fast (only the end-state geometry was checked, not intermediate frames).

---

## 2026-07-20 — Command palette (⌘K) wired end to end — P1 is now fully done

The ⌘K palette goes from a hard-coded 6-item `demo.PALETTE` list (no search,
no keyboard navigation, clicking did nothing) to Studio's real Spotlight-style
action registry — the actual last P1 item (the previous entry's own title
called Logs "the last P1 backlog item," which was already inaccurate at the
time — this one genuinely is). Also added: ⌘L for Logs, which shipped
without its own shortcut yesterday.

**New `studio/command_registry.py`** (Qt-free, stdlib only):
- `Command` — label/section/icon/hint/keywords/handler/enabled. Plain data
  plus a callable, no Qt, so the whole registry is unit-testable headless.
- A real fuzzy matcher, Sublime-Text/VS-Code style. The first version
  scored every match on one flat "reward contiguous runs + word-boundary
  starts" heuristic — and a test caught it ranking "Switch engine → SAM 2"
  *above* "Run segmentation" for the query "seg", purely because "seg" as a
  scattered subsequence across "switch"/"engine" happened to hit two
  word-boundary starts, while "seg" as a literal substring of "segmentation"
  only hit one. Fixed with two separate score bands instead: any real
  contiguous substring match (scored in the thousands, with a prefix bonus
  and a tighter-label bonus) always outranks any scattered subsequence match
  (scored in the tens, a genuine but weaker fallback signal for
  abbreviation-style queries like "rseg"). `search()` returns commands
  grouped by section for an empty query (the mockup's own "ACTIONS"/
  "EXPORT" caps-label browsing view) and a flat, score-ranked list with no
  section headers the moment there's a real query — real command palettes
  (VS Code, Spotlight) drop headers exactly at that point, since ranking
  *across* sections is the whole point of searching.

**`studio/overlays.py`'s `CommandPalette`** rebuilt on top: `get_commands`
is called fresh every time the palette opens, so availability always
reflects the live project/theme/backend/running state, never a stale
snapshot from whenever the app launched. A bounded `QScrollArea` (420px
max) replaces the flat list the 6-item static version got away with; each
row is now a small `_PaletteRow` that restyles itself in place
(`set_selected()`) rather than the whole list rebuilding on every arrow
key — instant, and never loses scroll position mid-navigation. Full
keyboard control via an event filter on the search box (Up/Down move the
selection and wrap top↔bottom, Enter activates), plus real click-to-run.
Disabled commands render dimmed rather than being hidden entirely —
discoverability ("this is possible, just not right now") over silence.
Running a command is deferred one event-loop tick
(`QTimer.singleShot(0, ...)`) before hiding the palette and calling the
handler: the same established sipBadCatcherResult-safe pattern
`workspace.py`'s 2026-07-10 fix already uses for the identical shaped
hazard, since a handler can itself rebuild the very screen the palette
sits over (switching tabs, switching engines) while the click/key dispatch
that triggered it is still on the call stack.

**The registry spans every tab**, each command wired through the same
narrow, testable public-alias convention the Assistant integration already
established — nothing invented, every command is a real, already-existing
action reached one hop away:
- `workspace.py` gained `switch_engine(key)`/`apply_preset(name)`/
  `run_batch()`/`run_benchmark()`/`save_masks()`/`export_measurements()` —
  thin aliases over already-self-guarding private methods (mirrors
  `rerun_predict()`'s existing shape exactly).
- `extra_screens.py`'s `ModelsScreen` gained `start_training()`/
  `stop_training()`/`import_model()`; `DashboardScreen` gained
  `open_in_aim()`.
- `assistant_panel.py`'s `AssistantDrawer` gained `run_diagnose()`/
  `switch_backend(idx)` (the latter reuses `_backend_seg._select()` exactly
  as this module's own tests already did, rather than a second copy of
  `_on_backend_changed`'s effect).
- `studio/app.py`'s new `StudioWindow._build_commands()` assembles all of
  it: **Navigate** (derived straight from the sidebar's own `_NAV` list —
  can never drift out of sync — plus Guide & Docs; real shortcut hints for
  Assistant/Logs), **Segment** (Run/Batch/Benchmark/Save/Export always
  listed but greyed out without a project; "Switch engine → X"/"Apply
  preset → X" generated per project, only the *other* available
  engines/presets), **Models & Train** (Start/Stop mutually gated on
  `is_training()`, Import), **Dashboard** (Open in Aim), **Assistant**
  (Diagnose, "Switch backend → X" — both open the drawer first so the
  effect is visible immediately, not silently behind a closed panel),
  **Appearance** (names the concrete destination theme), **Projects** (New
  Project…, Open Sample), **Help** (mirrors Home's own Resources links
  exactly, GitHub included).
- A real bug caught before it shipped: the engine-switch commands
  initially used `list_available_engines()`'s own label — the long,
  descriptive combo-box text ("Cellpose-SAM (zero-shot, generalist)"), not
  the short display name the mockup's own "Switch engine → SAM 2" style
  uses. Fixed to use `ENGINE_LABELS` (`project.py`) instead, caught while
  writing the end-to-end test, not by a test passing.

**⌘L / Ctrl+L opens (or closes) Logs** — the one shortcut Logs itself
didn't get when it shipped yesterday, mirroring the exact ⌘K/⌘T
dual-binding pattern (`QShortcut` on both key sequences). Documented in the
Guide's keyboard-shortcuts article and the in-app shortcuts list (now 4
real bindings, was 3).

**A real, pre-existing rendering bug found and fixed, not introduced by
this work** — caught by an actual light-theme screenshot, not by any test
passing: `CommandPalette`'s input-row and footer wrappers (`inp_wrap`,
`foot`) were plain `QWidget()`s, which inherit the app-wide
`QWidget{background:<bg>}` rule and paint an opaque `<bg>`-coloured
rectangle over their own children. Invisible in dark theme (`bg`/`surface`
are both near-black), a glaring flat-grey patch in light theme (`bg`
`#f4f6f8` vs. this panel's own `surface` `#ffffff`) — the exact "bare
`QWidget()` wrapper" bug family the 2026-07-09 entry below already found
and fixed in the Guide screen's table/shortcut rows. `CommandPalette` was
still 100% static content at the time of that audit and never got a real
screenshot pass, so this instance sat undiscovered until the palette
finally rendered live content here. Fixed with `bare_widget()`, the
existing helper built for exactly this. A pixel-level regression test
pins it — confirmed to *fail* against the reverted code first (sampling a
`QWidget()` wrapper's own margin corner, not its centre, which falls on a
child widget instead and would pass either way — the same trap
`test_guide_screen.py`'s own row-fill test already documents).

**Verified:** 653 `studio/tests` (up from 604), all green, incl. a new
`test_command_registry.py` (11 pure-logic cases: the substring-vs-scattered
ranking fix above, keyword matching, section grouping) and 15 new
`CommandPalette` cases in `test_overlays.py` (grouped-vs-flat rendering,
arrow-key wrap, Enter/click execution, disabled commands no-op, the
deferred-hide regression, `open()` re-fetching commands fresh, the
bare-widget pixel regression) — plus new alias tests in
`test_workspace.py`/`test_extra_screens.py`/`test_assistant_panel.py` and
18 new end-to-end cases in `test_app_wiring.py` covering the real registry
construction (per-section content, enabled/disabled state under every
condition above) and three full real-palette runs: typing a query and
switching tabs, typing and toggling the theme, and — the flagship proof —
typing "run segmentation" and watching a real prediction happen through
the exact production call chain, not a mocked handler standing in for the
palette's own wiring. Full repo `tests/` (445, classic app untouched) also
green. The throwaway-venv light-group check (`python3.10`) green: 592
passed, 22 skipped, no torch/napari/PyQt6 pulled in. Full `studio/tests`
re-run clean after every fix. Real offscreen screenshots in both themes
(the empty-query grouped view with live Navigate/Segment sections and real
⌘T/⌘L hints, a live "switch engine" search narrowing to one ranked
result, and the light-theme shot that caught the bare-widget bug above)
all visually confirmed correct.

**Not verified:** how this reads on a real, non-offscreen display, incl.
felt keyboard-navigation responsiveness; real Ollama/Custom-API server
interaction (unchanged from prior entries); real model/GPU inference (the
flagship palette→predict test uses the same monkeypatched engine seam this
suite always has).

---

## 2026-07-19 — Logs tab wired end to end — the last P1 backlog item, done

The Logs console goes from a hard-coded `demo.LOGS` transcript (7 static
lines, a close button, nothing else) to Studio's real, central log stream —
the last unwired P1 item (`docs/velum/BACKLOG.md`), leaving only the ⌘K
command palette.

**New `studio/log_bus.py`** (Qt-free, stdlib only) — the thing every other
piece of this change hangs off:
- `LogBus`: a bounded (4000), thread-safe ring buffer of `LogRecord`
  (seq/timestamp/level/source/message) with a plain-callback subscription
  (no Qt/psygnal — same convention as `layer_model.LayerList`'s events).
  `subscribe()` returns `(backlog, unsubscribe)` — the backlog snapshot is
  taken atomically under the same lock as registration, so a record can
  never be double-delivered or dropped across the join point a separate
  `snapshot()` + `subscribe()` pair would race.
- `StudioLogHandler(logging.Handler)` bridges the *real* stdlib `logging`
  module onto a `LogBus` — so an ordinary
  `logging.getLogger(__name__).info(...)` call anywhere in the process
  (Studio's own modules, the reused ML core, even a third-party dependency)
  reaches the console, not just hand-picked call sites that remember to
  invoke a bespoke callback. `install_handler()` attaches it to the root
  logger exactly once (idempotent per `(logger, bus)` pair — safe to call
  from every `StudioWindow.__init__`, real app or test), raises the
  logger's effective level to INFO if it was less verbose (root defaults to
  WARNING, which would otherwise silently swallow every `.info()` call
  before it reached a handler) without ever lowering a level already set
  more verbose, and always sets Studio's own `"studio"` namespace to DEBUG
  regardless — third-party dependency noise stays out, Studio's own
  breadcrumbs always get through. Capture is deliberately broad (DEBUG at
  the handler); filtering *for display* is the console's own level filter's
  job, so flipping the filter to "Debug" retroactively reveals debug lines
  already sitting in the buffer instead of requiring a restart.
- `emit_prefixed()` maps the ML core's existing `on_log(msg)` string
  convention (`napari_app.core.predict_controller`/`train_model`'s
  `[ERROR]`/`[WARN]`/`[HINT]`/`[INFO]` prefixes, reused unmodified by
  `segment_controller`/`train_controller`) onto the bus's real `logging`
  severities instead of everything defaulting to INFO, stripping the
  now-redundant bracket (the console renders its own coloured level badge
  per line).
- A real bug caught while writing this module's own tests, before it ever
  reached `overlays.py`: `LogBus` defines `__len__` (for `len(bus)`), which
  means a freshly-constructed *empty* bus is falsy under Python's
  truthiness rules — the common `bus = bus or get_log_bus()` DI idiom used
  throughout this codebase (`self._train = train_controller or
  TrainController()`, etc.) would have silently discarded a real,
  intentionally-passed-in empty test bus in favour of the global singleton
  the instant a test asserted on it before emitting anything. Caught by a
  test that failed for exactly this reason on first write, not by
  inspection. Fixed everywhere it applies (`install_handler`,
  `LogsConsole.__init__`) with an explicit `bus if bus is not None else
  get_log_bus()`; a regression test pins it
  (`test_install_handler_honors_an_explicit_but_still_empty_bus`).

**`studio/overlays.py`'s `LogsConsole`** rebuilt on the same real-time
stream: backfills the bus's full history at construction (opening Logs
after a background run finished still shows it), then stays live — a
`pyqtSignal` + the established guarded `_safe_emit_record`/
`sip.delete()`-tested pattern (a record can arrive from any thread: a
predict/training worker, the Assistant's urllib SSE thread) marshals every
new record onto the Qt main thread, and `self.destroyed.connect
(unsubscribe)` unhooks it from the bus the moment the widget is actually
torn down (a theme toggle rebuilds every overlay), so a stale subscriber
can never pile up across repeated toggles. A `QTextEdit` — not one `QLabel`
per line, the original static version's approach — is the professional
choice once the stream is unbounded instead of 7 fixed demo lines, and
matches the classic app's own `widgets/log_window.py` widget choice
(`docs/velum/BACKLOG.md`'s own instruction to "reuse `widgets/log_window.py`
logic"). New toolbar, still inside the unchanged 210px-tall bottom panel: a
live count badge (`"842 · 3 err · 5 warn"`, omitting a count that's zero),
a text search box (filters by message or source substring), a level filter
(`SelectBox`, All/Debug/Info/Warn/Error — a minimum-severity threshold,
default "Info" so Debug-level breadcrumbs stay hidden unless asked for),
an autoscroll toggle (on by default — snaps to the bottom on every new
line; off leaves the scroll position alone even while lines keep arriving
underneath, exactly like a real terminal), Clear (empties the console *and*
the bus, not just the view), and Export (saves the currently-*filtered*
lines to a `.txt` file via a real save dialog) — real professional-console
table stakes (Console.app, Chrome DevTools, `journalctl`), not just the two
features `BACKLOG.md` named. Level colours are real design tokens, not new
hard-coded hex (`t['danger']`/`t['warning']` for error/warn,
`t['text_subtle']` for plain info, `t['success']` for an `on_log` line that
starts with the ML core's own existing `✓` success convention) — the
console body itself keeps the deliberate always-dark `scope` ground
regardless of the app's light/dark theme (the same token the image
viewport uses), matching this file's own established "instrument, not a
page" precedent.

**A real, if minor, layout bug found and fixed while building the new
toolbar** (`SelectBox` has no stretch factor of its own): packed into a
`QHBoxLayout` next to a stretched search box and an `addStretch()`, Qt
honoured `SelectBox`'s own `sizeHint()` literally — and that sizeHint
under-reports the width its value label actually needs (confirmed by
inspecting `_val`'s allocated geometry: width 0), so "Debug"/"Error"
collapsed to a sliver with only the chevron left visible. Every other place
`SelectBox` is used either gets a stretch factor from its container or is a
vertical layout's sole child (which stretches it regardless of sizeHint),
so this never showed up before. Fixed locally with an explicit
`setMinimumWidth(96)` (measured: the widest option, "Debug"/"Error", is
42px) rather than touching the shared `components.py` atom — a broader fix
there risks regressing every other screen's fidelity without the ability to
re-screenshot all of them, out of scope for this change.

**Real emitters wired, not just a new empty pipe**: `workspace.py`'s
`_on_predict_log` (shared by predict/batch/benchmark — the reused
`PredictController`'s real operational log, previously skimmed only for a
`[ERROR]`/`[HINT]` toast and every other line thrown away) and
`extra_screens.py`'s training `_on_log` now both also forward every line to
the bus via `emit_prefixed`, tagged `studio.segment`/`studio.train` —
existing toast behaviour is unchanged, this is additive. `assistant_panel.py`
logs backend switches, chat errors, model-pull/tuned-agent-create results
(INFO on success, WARNING on failure) via the stdlib bridge, and
connection-status checks at DEBUG (they fire automatically on every backend
switch, not from a deliberate action, so the default filter keeps them out
of the way). `app.py` logs a startup line, project creation, theme toggles
(DEBUG), and — the one genuinely new capability, not just moved plumbing —
routes uncaught exceptions through the bus as a real CRITICAL entry
alongside the existing `traceback.print_exception`, so a crash is visible
in the app itself, not only to whoever had a terminal open behind it.

**Verified:** 604 `studio/tests` (up from 551), all green, incl. two new
files — `test_log_bus.py` (27 pure-logic cases: bus mechanics, the
`__len__`-truthiness regression above, the real `StudioLogHandler` bridge
against actual `logging` calls including `exc_info` formatting,
`install_handler` idempotency/level rules) and `test_overlays.py` (21
offscreen Qt cases: backfill, live updates, level filter applied to both
existing and new records, text search, autoscroll on/off incl. the
just-toggled-back-on snap, Clear, Export incl. a cancelled dialog, a
cross-thread emit from a real `threading.Thread`, and the `sip.delete()`
guard/unsubscribe pair) — plus forwarding tests added to
`test_workspace.py`/`test_extra_screens.py` (existing toast behaviour
pinned unchanged) and two new `test_assistant_panel.py` cases using
`caplog`. Full repo `tests/` (445, classic app untouched) also green. The
throwaway-venv light-group check (`python3.10`, the documented fallback
when no bare 3.11/3.12 is on PATH — `tests/test_packaging.py`'s `tomllib`
gap is the same pre-existing, unrelated py3.11+-only failure this repo's
own `AGENTS.md` already documents) green: 581 passed, 22 skipped, no
torch/napari/PyQt6 pulled in. Full `studio/tests` re-run three times in a
row clean (a real product built on threads/signals earns that scrutiny,
not just one green run). Real offscreen screenshots in both themes
(standalone `LogsConsole` and a full `StudioWindow` with real records
flowing through `get_log_bus()`) confirm the level colours, badge, and the
Debug-hidden-by-default filter all render correctly — including a
full-window integration shot showing Logs correctly anchored bottom-right
of the sidebar with real data.

**Not verified:** how this reads on a real, non-offscreen display; real
Ollama/Custom-API server interaction (unchanged from prior entries); real
model/GPU inference (the segment/train log lines in every test are
monkeypatched seams, per this suite's existing convention).

---

## 2026-07-18 — Project-wide audit: the same border-cascade bug, everywhere else

Asked directly, after the Assistant fixes: go through and fix this same
problem across the whole product, not just the one screen it was reported
on. It was a good instinct — the audit found the *worst* instance of this
bug family yet, in one of the most-used flows in the app, that no prior
session's reactive one-screenshot-at-a-time fixing had ever reached.

**Method**: `grep -n "border:1px solid\|border:1px dashed\|border:2px solid" studio/*.py`
for every literal visible-border declaration (not just background — the
specific property this bug family leaks), then read the surrounding ~10-15
lines of *each* to check two things: is the selector qualified
(`#ObjectName{...}` or `QType#ObjectName{...}`, not a bare `QFrame{...}`
type selector or no selector at all), and does the widget actually have
`QLabel`/`QFrame`-family children that could inherit the leak (a leaf
widget with no children is safe regardless of its own selector). An
earlier automated regex pass was tried first and abandoned after it missed
the single worst finding below — manual review of every real match was
slower but the only reliable way to be sure. 16 real, previously-unscoped
instances found and fixed across 5 files, all following the exact
`#ObjectName{...}`-qualification pattern already established for
`AssistantDrawer`/`ChangeCard`:

- **`studio/new_project_dialog.py`'s `_build_panel()` — the single worst
  instance found anywhere in the app.** The `QFrame` wrapping the *entire*
  New Project dialog body (header, every step's fields, the footer) had a
  fully unqualified `background:...;border:...;` rule — meaning every
  `label()` call anywhere inside the whole 3-step flow inherited the
  panel's own border and repainted its own small box around just its own
  text. Confirmed by an actual offscreen screenshot: "Import images," "Step
  2 of 3," "Drag & drop images here," the format list, the footer hint —
  every single line of text in the dialog individually double-boxed.
  Reduced to a clean, correct render by qualifying one selector.
- **`studio/components.py`**: `EngineChip`, `SelectBox`, `Stepper`,
  `SegControl` (defensive — no affected children today, but the same trap
  for whoever adds one), `StatTile`. `StatTile` in particular is used
  throughout Segment's Results pane and Dashboard's stat rows — this had
  likely been double-boxing every stat value/caption pair since the tab
  shipped.
- **`studio/screens.py`**: `HomeScreen`'s "Tip — press ⌘K anywhere" callout
  and its generic `_card()` helper (used for "Resources" and "This
  device"). **Both are the exact instances this file's own 2026-07-08 entry
  already found and screenshotted** ("the identical double-box on the Tip
  card's text… HomeScreen._card()/its Tip callout still use the unscoped
  form and carry this same latent, currently invisible bug") **and
  deliberately left unfixed as out of scope at the time.** Now in scope,
  now fixed.
- **`studio/extra_screens.py`**: `ModelsScreen._train_card`, its
  `_aside()` "Recent training runs" card, its "One-shot fine-tuning" tip,
  and `DashboardScreen._chart_card` (shared by the loss and F1 charts) and
  `_runs_table`. Between these and `screens.py`, this means Home, Models &
  Train, and Dashboard were *all* silently double-boxing card text.
- **`studio/workspace.py`**: `_image_row` and `_layer_row` (both per-row,
  shared `objectName` across every instance — the same convention
  `screens.py`'s `"PCard"`/`"RRow"` already use, confirmed safe: QSS
  matches by name, not uniqueness) plus the floating tool strip and viewer
  bar panels (defensive — currently-safe children, same reasoning as
  `SegControl`).

**Not touched, checked and confirmed safe**: every bare-type-selector or
unqualified rule on an actual *leaf* widget (a colour swatch, a status
dot, an icon-only button, `Slider`/`ChangeCard`'s drag knobs, `Chip` —
which is itself a `QLabel` with no children) — these have no descendants to
leak onto regardless of selector qualification, so "fix everything blindly"
would have been performative rather than useful. `studio/guide_screen.py`
was already fully qualified from its own 2026-07-08 fix and needed nothing
further.

Verified: 1 new pixel-level regression test for the worst finding
(`test_new_project_dialog.py`), confirmed to *fail* against the reverted
code first (correctly named all five individually-boxed labels) before
being trusted, then confirmed to pass with the fix restored — the same
discipline the previous entry called for. The other 15 fixes were verified
by direct offscreen screenshot comparison (real app-wide QSS applied, real
elapsed settle time) rather than one dedicated pixel-test each, given the
volume — Home, Projects, Models & Train, Dashboard, the New Project dialog,
and a constructed Segment workspace with real image rows were all
screenshotted before and after and visually confirmed clean; a dedicated
static-analysis "no bare type selectors" test was considered and skipped as
its own separate undertaking with real false-positive risk, not a quick
addition. Full `studio/tests` green (551, up from 550), full repo `tests/`
green, the throwaway python3.10-venv light-group check green.

**Not verified:** how any of this reads on a real, non-offscreen display.

---

## 2026-07-18 — Follow-up #2: the fill fix wasn't the actual bug, found it for real

The previous entry's `Accordion` `fill` fix was real but wasn't what the
user was pointing at — a second screenshot, cleaner and uncompressed, still
showed "borders aren't fully filled" plus "a bunch of extra sticks." Root-
caused properly this time, by pixel-sampling offscreen instead of
eyeballing a screenshot, and — critically — after discovering the verification
method itself was broken.

**The verification bug first, because it cost the most time**: every
offscreen check this session (including the previous entry's "confirmed
clean" screenshots) constructed widgets *without* ever calling
`app.setStyleSheet(theme.build_qss(t))` — the real, global stylesheet
`studio.app.main()` always applies. Without it, a container's own
(buggy, unqualified) cascade was the *only* thing giving its children any
background colour at all, so removing that cascade first made the chat
area render Qt's raw default grey, not the intended `t['bg']` — a second,
worse-looking regression that only existed in the test harness, not in the
fix. `docs/velum/CHANGELOG.md`'s 2026-07-08 entry already documented this
exact hazard for a different screen (a `styled_app` fixture, reset in
teardown since `QApplication` is a process-wide singleton) — it just hadn't
been applied to this round's ad hoc verification scripts.

**The two real bugs, found once verification was done properly:**

1. **`AssistantDrawer.setStyleSheet(...)` was an unqualified rule**
   (`"background:...;border-left:...;"`, no selector) — QWidget has an
   app-wide type-selector for `background` (theme.build_qss), so that
   property is always safely overridden lower down, but nothing overrides
   plain `border` for a bare `QWidget`/`QLabel`. The drawer's own
   `border-left` leaked onto *every* such descendant that has its own
   inset x-position: `ChatView`'s empty-state container (the long line)
   *and*, independently, its title/subtitle `QLabel`s (the two shorter
   "sticks" next to the text) — one root cause, multiple visible marks,
   which is why fixing the accordion's fill alone didn't touch it. Fixed
   by scoping the selector to `QFrame#AssistantDrawer{...}`. Applied the
   same fix proactively to three more overlays with the identical
   unqualified-background+border pattern (`LogsConsole`, `Toast`,
   `CommandPalette`'s panel) rather than waiting to find each by
   screenshot once those are exercised more.
2. **`ChangeCard.setStyleSheet(...)` used a bare `QFrame{...}` *type*
   selector**, not scoped to an object name — `QLabel` is itself a
   `QFrame` subclass, so the card's own background+border+radius rule
   *also* matched its own title and detail labels, each repainting its
   own small bordered box around just its own text. This is the exact
   rendering-bug family `docs/velum/CHANGELOG.md`'s 2026-07-08 "Guide &
   Docs" entry already named and root-caused once ("even a bare type
   selector like `QFrame{…}` cascades") — reproduced again in new code
   despite that lesson being on record, caught only by an actual
   offscreen screenshot, not by any test passing. Fixed by scoping to
   `QFrame#ChangeCard{...}`.

Verified properly this time: both new regression tests
(`studio/tests/test_assistant_panel.py`) confirmed to *fail* against the
pre-fix code before trusting them (temporarily reverted both fixes, ran the
tests, watched them fail for the right reason, restored the fixes, watched
them pass) — the same discipline this file's own 2026-07-08 entries already
called out as necessary and not automatic. Full `studio/tests` green (550
tests), full repo `tests/` green, the throwaway python3.10-venv light-group
check green. Fresh offscreen screenshots in both themes, using the corrected
methodology (real app-wide QSS applied, real elapsed time for the fade-in to
settle via `time.sleep`, not just empty `processEvents()` calls) — the empty
state, Diagnose's `ChangeCard`, and the expanded Model panel all confirmed
clean: no stray lines, no double-boxed labels, correct `t['bg']` chat
background (not the accidental leaked colour the earlier, uncorrected-
methodology screenshots happened to show).

**Not verified:** how this reads on a real, non-offscreen display — asking
for confirmation after this pass, same as the standing practice for every
Studio rendering fix in this log.

---

## 2026-07-18 — Follow-up: real-usage feedback on the Assistant, a design pass

Direct user feedback on a real (non-offscreen) screenshot of the just-shipped
Assistant drawer, same day: a rendering issue ("borders of blocks aren't
fully filled"), no keyboard shortcut to open it, and a blunt overall verdict
— it read as a mockup, not a product, because nothing about it moved.

- **Root cause of the "hollow" look, found by pixel-sampling the drawer
  offscreen rather than guessing from the (compressed) screenshot**: the
  "Model" settings `Accordion` used the shared component's default fill —
  `t['inset']`, a *recessed field well* token — sitting close to directly on
  the drawer's own `surface` background. In dark mode that's `#101318` vs.
  `#15181e`: a real fill, confirmed present by direct pixel sampling, just
  with so little contrast against its surroundings that it read as a hollow
  outline rather than a card — the same *category* of token mistake as
  `docs/velum/CHANGELOG.md`'s 2026-07-08 "it all looks like one dark canvas"
  entry (`inset` used where `surface2`, "elevated fill," was called for),
  just a different instance of it. Fixed at the component level:
  `components.Accordion` gained an additive `fill: str = "inset"` parameter
  (every existing call site's exact look is unchanged — confirmed via a new
  regression test asserting the default styleSheet is byte-identical to
  before); the Assistant's "Model" accordion now passes `fill="surface2"`.
  The nested "Download a model" accordion inside it keeps the *default*
  `inset` — now correctly reading as *recessed within* the elevated Model
  card instead of just another flat rectangle, real layered depth instead
  of one flat mistake compounding into a second.
- **Zero motion anywhere was the bigger complaint** — the drawer popped in
  instantly, the accordion snapped open with no transition, and every chat
  message just appeared with no acknowledgement, which reads as unfinished
  regardless of whether every pixel is technically correct. Added, all
  using the existing `motion.py` primitives (or new ones following its
  exact conventions — degrade quietly if animation can't run, guard every
  callback against a torn-down widget):
  - **The Assistant drawer (and Logs console) now slide into place** from
    the edge they're anchored to, instead of popping in — new
    `motion.slide_in()`, wired into `StudioWindow._toggle_drawer()` (shared
    by both overlays). Direction is inferred from which of the widget's own
    dimensions dominates its window dimension (`height/window_height` vs.
    `width/window_width`) rather than "which edge does it touch" — both
    overlays' *right* edge touches the window's right edge (LogsConsole
    spans the full remaining width too, not just AssistantDrawer), so edge-
    touching alone doesn't disambiguate them; caught by a wiring test
    before it shipped; a naive first version really did slide LogsConsole
    in sideways instead of up.
  - **Chat messages fade in** — every bubble/card/note goes through
    `ChatView._insert()` exactly once (streamed tokens mutate an
    already-inserted bubble's text, never re-insert), so one `fade_in()`
    call there covers every message kind uniformly without re-triggering
    on every streamed token.
  - **`Accordion.toggle()` fades its body in on open** (instant on close —
    revealing invites a look, dismissing should just get out of the way),
    applied to the shared component so every accordion in Studio gained
    this, not just the Assistant's.
  - **A new `_StatusDot`** (`studio/assistant_panel.py`) replaces the
    plain "Checking…" text with a small dot that visibly *breathes* (a
    looping opacity pulse) while a backend check is in flight, and settles
    to a solid colour once resolved — reads as "checking right now," not a
    static label next to an inert circle.
- **⌘T / Ctrl+T now opens (or closes) the Assistant** — mirrors the
  existing ⌘K/Ctrl+K dual-binding pattern exactly (`QShortcut` on both key
  sequences, since Qt's Ctrl/Meta swap on macOS isn't something to rely on
  silently). Documented in the Guide's keyboard-shortcuts article and the
  in-app shortcuts list (now 3 real bindings, was 2).

Verified: 11 new tests (548 total in `studio/tests`, up from 537) — the
`Accordion` fill parameter (default unchanged + override works), the
open-fades/close-doesn't behaviour, `motion.slide_in`'s start/end values and
its deleted-widget degrade-quietly path, the `_toggle_drawer` direction
inference for *both* overlays (including the LogsConsole-slides-sideways
bug caught mid-session), the ⌘T/Ctrl+T shortcuts actually toggling the
drawer, and the `_StatusDot`'s checking/resolved/deleted-widget states.
Full `studio/tests` and the repo-wide `tests/` green, plus the throwaway
python3.10-venv light-group check. Fresh offscreen screenshots of the
expanded Model accordion in both themes confirm the fill is now clearly
visible as a distinct, elevated card rather than a hollow outline — directly
compared against the pre-fix screenshot from the same complaint.

**Not verified:** how the slide-in/fade/pulse animations actually feel on a
real display at real frame rates (offscreen screenshots can only prove the
animation objects exist with correct start/end state, not perceived
smoothness); real Ollama/Custom-API server interaction, unchanged from the
previous entry.

---

## 2026-07-18 — Assistant tab wired end to end — the last P1 flagship, done

The Assistant drawer goes from a hard-coded `demo.CHAT` transcript (three
static bubbles, chips that changed nothing) to a real chat: own UI, reuse
the logic, exactly the principle every other tab has already followed —
plus local model selection, an Ollama connection, and a Custom-API
connection for a user's own hosted/self-hosted model, as requested.

**New `studio/assistant_controller.py`** (Qt-free, mirrors `train_controller.py`/
`segment_controller.py`'s shape) — three interchangeable chat backends
behind one `send_async()`:
- **"offline"** — `napari_app.advisor.diagnose`, the existing deterministic
  diagnostic engine, reused read-only (lazy import, never modified — the
  same "reuse the logic" pattern `segment_controller.py` already uses for
  `predict_controller`). Always available, no model, no network call.
- **"ollama"** — `napari_app.advisor`'s existing Ollama bridge, also reused
  verbatim: model discovery, `ollama_pull`, the "bake a task-specialised
  `cellseg1-assistant`" flow, streaming chat.
- **"custom"** — a brand-new bridge (`custom_api_available`/
  `custom_api_models`/`custom_api_chat`), stdlib `urllib` + Server-Sent-Events
  parsing, no new dependency — any OpenAI-compatible `/chat/completions`
  endpoint, local (LM Studio, vLLM, llama.cpp's server, text-generation-webui,
  …) or remote (OpenAI itself, OpenRouter, Groq, …), with or without an API
  key. This is genuinely new capability (the classic app only ever had
  Ollama), so it lives entirely in Studio's own module rather than being
  added to `napari_app/advisor.py` — nothing in `napari_app/` changed.

`AssistantSettings`/`AssistantSettingsStore` persist which backend + model
is selected to one small JSON file under the shared storage dir (a
machine-level choice, not a per-project one — mirrors `ProjectStore.save`'s
atomic temp-file-then-replace write). `send_async()` falls back to the
synchronous offline diagnostic reply whenever the configured backend isn't
actually ready (no Ollama model picked, no Custom-API base-url+model set)
instead of erroring, so a half-configured backend degrades gracefully.
`refresh_status_async()` does the live reachability check (Ollama server up?
Custom API endpoint reachable?) off the UI thread, always — including the
one the drawer fires automatically the moment a non-offline backend is
selected, so a slow or unreachable server never blocks the UI even for a
second.

**`studio/workspace.py`** gained the narrow read/write hook the Assistant
needs into the active Segment session, mirroring the classic app's
`PredictWidget.last_context()`/`current_params()`/`apply_params()`/`rerun()`
contract: `assistant_context()` (image/mask/params — gated on `_last_result`,
not just the "Segmentation" layer's existence, since `_select_image()`
always creates a zero-filled placeholder the moment an image is picked; without
that gate an unpredicted image would misreport to the advisor as "0 cells
found" instead of "no prediction yet"), `apply_assistant_changes()` (same
convention as a manual threshold edit — marks `quality_preset` "Custom",
persists, rebuilds the Segment pane so a cross-tab change is visible without
reopening the project; whitelisted against `ProjectSettings`' real dataclass
fields rather than a bare `setattr`, so a stray key can never shadow an
instance method), and `rerun_predict()` (a thin alias for `_start_predict()`,
which already guards every precondition). This is the actual "connect it to
the other tabs" wiring — a suggestion applied in the Assistant changes real
settings the Segment tab's own inspector reflects immediately, and can
trigger a real re-run.

**New `studio/assistant_panel.py`** — Studio's own chat UI, importing
nothing from `napari_app.widgets` (own tokens, own icons, same as every
other tab):
- `ChatView` — a Studio-native port of the chat idiom (message bubbles, a
  streaming assistant reply with a typing indicator, an empty state) built
  from Studio's own atoms/tokens rather than reusing the classic app's Qt
  widget.
- `ChangeCard` — a recommended parameter change with Apply / Apply &
  re-run, built from `PillButton`/tokens.
- `AssistantDrawer` itself (moved out of `overlays.py` entirely — it had
  outgrown living alongside `LogsConsole`/`CommandPalette`/`Toast`): header
  (unchanged) + a collapsed-by-default "Model" settings accordion (an
  Offline/Ollama/Custom-API `SegControl`, per-backend fields, a live status
  line, Ollama's recommended-models download catalogue + "Tune for
  CellSeg1") + the chat (the hero surface, filling whatever room is left) +
  Diagnose/input/Send. Every model/network call runs on a background
  thread with guarded (`_safe_emit_*`, the established
  `sip.delete()`-tested pattern already used throughout Studio for this
  exact hazard) Qt-signal delivery back to the main thread.

**Wiring**: `studio/app.py` now imports `AssistantDrawer` from
`assistant_panel` (not `overlays`); `StudioWindow` gained an
`assistant_controller` param (mirrors `project_controller`/
`train_controller`/`segment_controller` — constructed once in `__init__`,
*not* rebuilt in `_build_ui()`, so a chosen backend/model/API key survives
`toggle_theme()`'s full UI teardown-and-rebuild) and passes the real Segment
`WorkspaceScreen` through as the drawer's cross-tab context/actions source.
`components.Accordion` gained a small, additive `set_title()` so the "Model
· Ollama" header can update live as the backend picker changes, without
rebuilding the whole accordion.

**A real bug, found only by actually screenshotting the result offscreen —
not by tests passing.** `ChangeCard`'s severity dot used a nested
widget-with-its-own-`addStretch()` layout (`dot_wrap`/`dwl`) to pin the dot
to the title's cap-height — copied faithfully from `napari_app`'s own
`ChangeCard`, which uses the identical construction. Once embedded inside
`ChatView`'s `QScrollArea` (rather than a plain standalone widget), that
nested stretch made Qt balloon the whole card to 3-4x its `sizeHint()`:
confirmed by instrumenting `card.geometry()` vs. `card.sizeHint()` before
and after (75px tall standalone; 313px once actually laid out inside the
chat), and by bisecting the exact construction line by line until removing
just the inner `addStretch()` collapsed it back to 75px. A "Run a
prediction first" card rendered as a ~300px-tall box with a large empty gap
between its title and detail text instead of a compact one. Fixed by adding
the dot directly to its row with `Qt.AlignmentFlag.AlignTop` instead of a
nested-stretch wrapper widget — the identical visual result (the dot stays
pinned to the top if the title wraps to two lines) without the failure
mode, confirmed both by geometry introspection and by fresh offscreen
screenshots. (The identical pattern in `napari_app`'s own `ChangeCard` was
left untouched — out of scope, `napari_app/` wasn't touched at all this
round, and it may or may not manifest the same way inside that app's own
`ChatView` construction; not verified either way.)

**Known, deliberate gap** — called out rather than half-built: the
auto-tune predict→score→adjust loop the classic app's Assistant also has
(a live score chart, a sortable trajectory table, CSV export, a
parameter-importance readout) is *not* wired into Studio's drawer this
round. It's a large, separate sub-feature on top of an already-large
change; `napari_app/core/tuning_loop.py` (the Qt-free loop logic) is
reusable as-is whenever this is picked up — see `BACKLOG.md`.

**Verified:** 76 new pure-logic/Qt-wiring tests across three new files —
47 in `studio/tests/test_assistant_controller.py` (settings round-trip,
`backend_ready()` per backend, the Custom-API bridge against a mocked
`urllib.request.urlopen` covering SSE framing/auth headers/an empty-url
short-circuit/cooperative `stop()`, `send_async`'s per-backend dispatch
including the tuned-agent-model system-prompt-skip path, error propagation,
`chat_busy()`/`model_op_busy()` held open against a real background
thread); 9 in `studio/tests/test_workspace.py` (now 85 total) for the three
new hook methods, including the "unpredicted image must read as no-mask,
not zero-cells" regression; 29 in `studio/tests/test_assistant_panel.py`
for the drawer itself — backend switching + persistence, Diagnose →
`ChangeCard`s, Apply/Apply & re-run against a fake workspace (a `None`
"no active project" return handled without crashing), the offline
backend's synchronous reply, a connected backend's real threaded streaming
(send-while-busy is a no-op, error reporting lands in the chat, a stale
status result from a since-switched backend is discarded), Ollama
refresh/pull/bake-agent and Custom-API test-connection/field-persistence
flows, and a `sip.delete()` regression proving every `_safe_emit_*`
survives a torn-down widget. Two real test bugs caught and fixed before
they shipped, not by luck: a case-sensitivity assertion mismatch against
`Accordion`'s all-caps title rendering, and a fake-Ollama-pull-too-fast
race (the trivial fake completed before the main thread's own
`_rebuild_model_body()` call could observe `model_op_busy()==True` —
fixed by holding the fake open on a `threading.Event` until the assertion
runs, the same deterministic-synchronization pattern already used
elsewhere in this suite for background-thread tests).

Full `studio/tests` green throughout (536 tests, up from 460), both in the
real conda env and in a from-scratch `python3.10` venv with only the
declared `test` dependency-group installed (554 passed / 21 skipped — no
PyQt6/torch/napari pulled in; `test_assistant_panel.py` correctly skips as
one unit via `importorskip("PyQt6")`, and `test_assistant_controller.py`'s
47 cases run for real there since the module has zero PyQt6/torch
dependency). The full repo-wide `tests/` suite (classic app) also green,
confirming the zero `napari_app/` changes this round didn't regress it.
10 real offscreen screenshots across both themes (`QWidget.grab()`, several
`app.processEvents()` pumps) — the empty state, Diagnose on an unpredicted
image, a full simulated chat exchange (a monkeypatched Ollama reply with
`SUGGEST:` lines streaming in, correctly parsed into a working Apply &
re-run card), and the Ollama/Custom-API settings bodies (including a real,
unmocked `custom_api_available()` call against `localhost:1234` failing
fast and surfacing the "could not confirm, may still work" message rather
than hanging) — plus one full-`StudioWindow` integration screenshot
(sidebar → Assistant → the real drawer positioned correctly over a real
opened project) — all visually confirmed correct, including the
`ChangeCard` fix above (found *by* this same screenshot pass).

**Not verified:** real Ollama/Custom-API server interaction beyond the
mocked-`urlopen`/monkeypatched-`advisor` test seams (no live Ollama or
OpenAI-compatible server was reachable in this environment); on-screen
(non-offscreen) look, font rendering, and animation smoothness; real model
inference through either bridge.

---

## 2026-07-10 — Fix a real crash: clicking several Segment-tab controls could abort the app

Found from an actual running session, not a test: clicking "preserve
labels" or "show selected" in the Labels layer panel raised `TypeError:
invalid argument to sipBadCatcherResult()` — and reproducing it directly
confirmed this is a hard process abort (SIGABRT), not a catchable Python
exception.

- **Root cause:** several rows/swatches (checkbox rows, the label
  colour-palette swatches, layer rows, image rows) override
  `mouseReleaseEvent` directly instead of using a real Qt signal, and from
  inside that same call, trigger a container rebuild
  (`_clear_layout`/`setParent(None)`) that reparents the very widget whose
  own `mouseReleaseEvent` is still executing on the call stack — a
  PyQt/SIP reentrant-virtual-call hazard. Every existing test called the
  handler method directly (`ws._toggle_layer_bool(...)`, etc.) or even
  called `widget.mouseReleaseEvent(event)` as a plain Python method call —
  neither actually dispatches through Qt's C++ virtual-call machinery the
  way a real click does, so none of them ever caught this; only routing a
  `QMouseEvent` through `QApplication.sendEvent()` (the same path a real
  click takes) reproduces it, which is how this was confirmed before
  fixing and is now regression-tested.
- **Fix:** defer each of the four affected handlers via
  `QTimer.singleShot(0, ...)`, so the destructive rebuild runs on the next
  event-loop tick, after the widget's own `mouseReleaseEvent` call has
  fully returned — an existing, already-used pattern in this codebase
  (`extra_screens.py`), not a new idiom. Affects: the Labels checkboxes
  (contiguous/preserve labels/show selected), the colour-palette swatches,
  layer-list row selection, and image-list row selection — the last two
  weren't reported yet but shared the exact same hazard and would have
  crashed just as reliably the first time either was clicked.
- **Also fixed, found while investigating (reported in the same message,
  with a screenshot):** the floating tool strip's pan/zoom and brush
  icons had silently become generic chevrons. `_sync_toolbars()` was
  re-deriving each icon's name from its raw mode string ("pan_zoom",
  "paint") instead of the semantic icon name used at construction
  ("target", "brush"); neither raw string is a real key in `icons.PATHS`,
  so `icons.py`'s own unknown-name fallback silently substituted a
  generic chevron glyph — on every restyle, which fires on almost every
  interaction. Fixed by storing each button's real icon name alongside
  its action and always styling from that, never re-deriving it.

Verified: full `studio/tests` green, 453 cases (7 new — one per crash site
plus one proving the icon fix, using `QApplication.sendEvent()` to
faithfully reproduce the crash before fixing, confirmed by first
reproducing the exact SIGABRT standalone against the unfixed code); a
fresh offscreen screenshot of the floating tool strip confirms the
pan/zoom and brush icons now render their real, distinct glyphs instead
of chevrons.

---

## 2026-07-09 — Segment tab: persistence, pan bounds, real 3-D rotation, verification pass

A `/goal`-driven round: work until the Segment tab genuinely satisfies "3-D
rotates like real 3-D, every button/panel is correct, the canvas can't be
flown out of bounds, a saved run survives reopening the project, and it all
logs to Dashboard, quickly." Four pieces of work plus a closing audit:

- **Segmentation results now persist per (project, image) and reload on
  reopen.** Previously a predicted mask lived only in the in-memory
  `LabelsLayer` — closing and reopening the project (or just switching
  images and back) lost it, so "is this image done?" had no durable answer.
  `SegmentController` now saves/loads each result mask to a SHA1-hash-keyed
  path (hashed on the image's resolved absolute path, so two images
  sharing a filename in different folders never collide) under the
  project's own run directory; `_select_image` checks for a saved mask
  before falling back to an empty layer, and the Images pane's per-row
  status reflects it even for an image that's never been the *selected*
  one this session. Batch runs write into the same cache so a batch and a
  single predict agree on "has this been run" for the same image.
- **The canvas can no longer be panned or zoomed the image fully out of
  view.** Dragging or repeated wheel-zooming used to let the whole image
  slide past every edge with no way back short of the Home button.
  `Canvas._clamp_pan()` now keeps a margin (capped at half the viewport and
  at the image's own scaled size, so a tiny zoomed-out image can still be
  nudged off-centre rather than glued to the middle) reachable near every
  edge, called after every pan-drag, wheel-zoom, and `home()`.
- **The "3-D" toggle on a flat 2-D image is now genuinely interactively
  rotatable**, not the fixed static trapezoid tilt from the previous
  round's honest-substitute. Left-drag while `mip` is on over a
  single-plane image now orbits a real rotate-then-perspective-project of
  the image rect's corners around pitch/yaw angles (`Canvas._rot_x`/
  `_rot_y`), clamped to ±80° short of the projection degenerating;
  middle-button still always pans; `home()` resets rotation to the default
  pitch along with pan/zoom, matching real napari's `reset_view()`
  resetting the whole camera, not just pan/zoom. This is still not a GPU
  3-D camera (no real depth/occlusion, no volume orbiting) — an honest,
  interactive substitute for a flat image, same category of simplification
  as the rest of this canvas, just no longer *static*.
- **Comprehensive re-verification pass** across every wired control in the
  Segment tab: the Labels layer's fuller settings (contour, edit-dimension
  count, contiguous/preserve-labels/show-selected checkboxes, direct colour
  picking, colour-mode reset), image gamma/colormap, Points/Shapes size,
  edge-width and clear actions, the Show-predictions/Show-ground-truth
  overlay toggles, SAM 2's model-size/tracking-mode selects and the SAM
  backbone select, the generic per-engine setting path, the Refine
  ("coming soon" — confirmed it still honestly says so rather than
  silently doing nothing) and Measurements buttons, and CSV export's
  no-result guard — all either already covered or found to be correctly
  wired, with 19 new regression tests closing the ones that weren't. Added
  one true cross-screen integration test: running a fake predict in a real
  `WorkspaceScreen` and then navigating a real `StudioWindow` to its
  `DashboardScreen` and reading `runs_table()` back — proving the two
  screens' shared `ProjectController` wiring end to end, not just each
  screen's own controller in isolation.
- **Load-speed check**: project open, image switching (~16 ms/image
  averaged over 5), layer/settings changes, and Dashboard navigation
  (~7 ms) are all effectively instant. The one measured hitch (~0.7–0.8 s)
  is a one-time `import torch` paid by whichever code first calls
  `cellpose_available()` in a given process — inherent to the `cellpose`
  dependency, identical in real napari, paid once per app launch rather
  than once per project — not a Studio inefficiency.

Verified: full `studio/tests` green throughout, 458 → 478 cases (8 new
canvas tests directly driving simulated drag events and asserting on
`_rot_x`/`_rot_y` plus a pixel-diff proving a drag visibly changes the
render, 19 new workspace-control tests, 1 new cross-screen integration
test); offscreen screenshots at the default/dragged/clamped-extreme
rotation angles confirm the projection looks like a real tilt (receding
edge narrows, no inversion) all the way to the clamp boundary. Not
verified: how the rotation drag feels on a real display with a real mouse
(offscreen `QMouseEvent` sequences prove the math and state transitions,
not felt input latency); real model inference speed (all engine calls in
this pass are monkeypatched, per the existing convention).

---

## 2026-07-09 — Follow-up: real-usage feedback on the Segment tab, four fixes

Four issues found from actually running the app (not just offscreen tests),
each investigated against a real reference before fixing rather than guessed:

- **A real content-overflow bug**: `SelectBox`'s value label and
  `Accordion`'s title had no width constraint, so a long dynamic string
  (an engine's full registry label, a LoRA filename, a discovered GT mask's
  name) forced the fixed-320px inspector panel to 356px — with horizontal
  scrolling off, the excess was silently, permanently clipped. Root-caused
  with an offscreen width-audit script (comparing every widget's sizeHint
  against its allocated width), not guessed at. Fixed with a new
  `_ElidingLabel` (components.py) that elides instead of forcing its parent
  wider; StatTile padding tightened too (three tiles' *combined* width was
  independently over budget). Re-verified: zero overflow at both the
  default and minimum window sizes.
- **Breadcrumb didn't navigate.** "Projects / ProjectName" was one static
  label. Split into two — "Projects" is now a real link back to the
  Projects tab, matching Label Studio's pattern (only the ancestor segment
  navigates, not the current page's own name).
- **No way to add images to an already-created project** — the Images pane
  only ever had a filter box; a project's image list was set once, at
  creation, and never revisited. Added a "+" button (multi-select file
  picker) and drag-and-drop onto the Images pane, both funnelling into one
  dedup-and-persist path.
- **Default rendering was fully solid and "too bright"; the 2D/3D toggle
  did nothing on a plain 2-D image; grid mode ignored the mouse wheel
  entirely.** All three were checked against a real reference before
  fixing — `napari_app/widgets/predict_widget.py`'s actual mask-display
  code and the installed napari package's own viewer-button source —
  rather than assumed:
  - Labels now render fill (soft, 0.35 opacity) *and* an outline (crisper,
    0.7 opacity, 2px) simultaneously by default, matching the classic app's
    own two-stacked-layers convention (`_add_filled_labels`) in one layer
    instead of two. Ground truth gets that same convention's *other* half:
    a fixed uniform colour instead of per-instance random hues.
  - The 2D/3D toggle no longer refuses on a flat image — confirmed real
    napari's `ndisplay` toggle has no such guard at all; ours now applies a
    projective "tilted plane" view (`Canvas._draw_pseudo_3d`, an honest
    non-GPU substitute, not real volumetric rendering) instead of a silent
    no-op + toast.
  - Grid mode's per-tile auto-fit scale is now multiplied by the live zoom
    level (real napari's grid shares one camera across every tile) instead
    of being recomputed from scratch every paint regardless of it.

Verified: full studio/tests green throughout (new/updated cases across
layer_model, canvas, components, and workspace covering every fix above,
including two direct before/after pixel-diff tests proving the pseudo-3D
tilt and grid-mode zoom actually change rendered pixels, not just internal
state) + real offscreen screenshots, cropped and pixel-sampled rather than
trusted at a glance (a 1px-wide contour measured technically-present but
visually-too-faint before bumping it to 2px). Not verified: how any of this
reads on a real, non-offscreen display.

## 2026-07-09 — Segment (Workspace) tab wired end to end — the flagship backlog item, done

The last unstarted P0 — the whole reason Studio has a "design skeleton"
phase distinct from a finished product. Segment goes from `NucleiView` +
static `demo.*` reads to a real segmentation surface, on **our own** canvas
(explicitly not embedded napari, per `ARCHITECTURE.md`), reusing only the
classic app's proven ML core underneath — matching the mockup's exact look
throughout (no restyle, behaviour added under the existing design).

**New modules:**
- `studio/layer_model.py` — our own evented layer list (Layer/ImageLayer/
  LabelsLayer/PointsLayer/ShapesLayer/LayerList), plain-callback events (no
  Qt/psygnal) so it stays in the light CI test group. `LabelsLayer`'s
  properties and defaults were checked 1:1 against the installed
  `napari.layers.Labels` source per the product owner's "identical settings,
  it's open source, take it from there" instruction: opacity 0.7, blending
  translucent, brush_size 10, contiguous True, preserve_labels False,
  n_edit_dimensions 2, contour 0, selected_label 1, and the PAN_ZOOM/
  TRANSFORM/PAINT/ERASE/FILL/PICK/POLYGON mode set. Paint/erase/fill/pick/
  polygon operate directly on the backing numpy mask; a golden-angle hue
  rotation gives well-separated, deterministic per-instance colours without
  a hash table (plus a colour-override path for "colour cells by
  measurement"). A bug this module's own tests caught before it shipped:
  `n_planes` was counting an RGB *image* layer's channel axis as a z-axis.
- `studio/canvas.py` — a plain `QWidget`/`QPainter` viewport: numpy
  compositing (contrast/gamma/single-hue colormap tint for images;
  per-instance colour + opacity + contour for labels; translucent/additive/
  opaque blending) cached and redrawn by `QPainter` under a pan/zoom
  transform, so pan/zoom never re-runs the numpy math. Real mouse-driven
  editing (brush-stroke interpolation so a fast drag doesn't leave gaps;
  Points add/remove; Shapes' own click-vertices/double-click-closes
  polygon flow). Grid mode tiles one cell per visible layer (the useful
  reading of napari's grid mode when there's always exactly one image);
  the 2D/3D toggle is a max-intensity projection across a loaded z-stack,
  *not* GPU volumetric rendering — a real, working, but deliberately
  simpler substitute, described as such rather than overclaimed.
- `studio/segment_controller.py` — maps `ProjectSettings` to
  `PredictController`'s params/config shape and calls
  `run_prediction_async`/`run_batch_async`/`run_benchmark_async`
  **unmodified** (the "reuse the logic" principle, literally): every test
  that exercises a predict run monkeypatches
  `napari_app.inference_cache.predict_cached` (the one seam every engine
  path reaches for "cellseg1") so the real call chain runs at real cv2/
  numpy speed with no GPU/SAM weights/torch actually loading. `record_run()`
  mutates `project.stats` (n_cells/last_f1/progress) without saving — same
  convention as `TrainController.select_model_for_project`, which leaves
  `ProjectStore.save` to the caller — and is what makes Segment-tab activity
  show up in the Dashboard.
- `components.py`'s `Slider`/`Stepper` — were purely presentational in the
  design skeleton (no mouse handling, no working +/- buttons at all); now
  real value + `changed` signal, same exact visual, since the Segment
  pane's thresholds/opacity/brush-size/contour controls needed them to work.

**`studio/workspace.py` rewired throughout:** Images pane (real
`project.image_paths`, real cv2 thumbnails falling back to the procedural
nuclei art on an unreadable file); Layers pane (add points/shapes/labels,
delete, select, eye-toggle, all backed by the new layer model) with
per-kind controls (Labels gets the full napari-faithful block; Image gets
contrast/gamma/colormap; Points/Shapes get a smaller real set); Segment
settings (engine/model/quality-preset/thresholds/image settings/overlay
toggles/a per-engine "Engine settings" accordion); Run (real progress,
elapsed-time status pill + toast, synchronous config-error toasts before
any thread starts vs. async `[ERROR]` log lines for a failure *during* the
run — matching the established convention exactly); Results (stats,
editable pixel calibration that recomputes on change, Save masks/Export
CSV/Measurements — "Refine…" is an explicit "coming soon" toast, not faked
— colour-by heatmap, Ground truth & evaluation, Batch prediction, Benchmark
engines vs GT); the viewer bar and floating tool strip now both call real
`Canvas` methods *and* re-style themselves to reflect live state (grid/MIP/
transpose on, or the active edit mode) rather than just acting on clicks
with no visual feedback. `DashboardController.runs_table()` was loosened so
a plain segmented-but-unbenchmarked project shows up too (F1 "—"), not
invisible until someone runs a GT benchmark — the actual "segmentation
activity shows up in the Dashboard" requirement.

**Known, deliberate gaps**, called out rather than papered over: TRANSFORM
mode aliases to pan/zoom (no real affine-transform UI); z-stack/time-lapse
*prediction* isn't wired (the canvas can display an already-loaded volume
via MIP; nothing yet builds one from a project's flat image-file list and
calls `_predict_volume` on it); a cross-tab settings change to an
already-open project's Workspace session (e.g. picking a model in Models &
Train) needs reopening the project to pick up, rather than live-syncing.

**Verified:** 135 new test cases across `studio/tests/test_layer_model.py`,
`test_canvas.py`, `test_segment_controller.py`, `test_components.py`,
`test_workspace.py`, plus updates to `test_dashboard_controller.py` and
`test_app_wiring.py` — full `studio/tests` suite green offscreen (300+
tests, ~47s), and confirmed the same tests pass in a from-scratch venv with
only the declared `test` dependency-group installed (no torch/napari/
PyQt6/cellpose). Beyond unit tests: built a real `StudioWindow` offscreen
with real controllers, created a real project with real image files on
disk, ran a full predict through the exact production call chain end to
end, evaluated it against a real ground-truth file, and screenshotted the
result in both themes — confirmed by direct pixel sampling in one case
rather than trusting the image by eye (a paint-timing artifact on the very
first screenshot attempt needed more `processEvents()` pumping before it
was trustworthy, not an actual bug — the exact lesson from this file's own
2026-07-08 entries, still holding). Also caught and fixed a real test-
isolation bug along the way: cellpose is genuinely installed in the
environment these tests ran in, so the benchmark tests were silently
running real (slow, intermittently timing-out) Cellpose inference instead
of the intended fake seam, until explicitly monkeypatched unavailable.
**Not verified:** real on-screen (non-offscreen) interaction, real model/GPU
inference, a real SAM2 or Aim install, multi-monitor/HiDPI rendering.

## 2026-07-09 — Follow-up: SAM backbone had no manual fallback when nothing was auto-detected

Reported right after the tabs above shipped: "SAM backbone" showed "Not
found" and did nothing when clicked, while "Annotated image" opened a real
file picker — inconsistent, and a dead end for exactly the environment this
was tested in (no SAM weights downloaded at all, so `available_backbones()`
correctly returns empty). The classic Train tab has always had an escape
hatch for this via its separate, always-editable `sam_path` field; Studio's
single "SAM backbone" field didn't carry that over — clicking only ever
opened a menu of *auto-detected* files, with nothing to click when there
were none.

Fixed: the field is now clickable either way. With ≥1 backbone found, its
menu gained a trailing "Browse…" entry; with none found, clicking opens a
file picker directly instead of a menu. `TrainController.build_config()`
gained an optional `backbone_path` that's used as-is instead of resolving
`vit_name` against `sam_backbone_dir` — `vit_name` becomes just a label,
best-effort-guessed from the picked file's name (`guess_vit_name()`,
defaulting to `vit_h` with no hint in the name) if not given, the same
trust-the-user contract the classic widget's separate, never-cross-
validated `sam_path`/`vit_name` pair already has. `_status_text()`'s hint
now says so explicitly ("click SAM backbone to browse for a checkpoint, or
run setup_napari.sh to download one") instead of only mentioning the setup
script.

**Verified:** new pure-logic tests (`guess_vit_name` parametrized over real
SAM filenames + an ambiguous one; `build_config` with `backbone_path` used
directly, with an explicit `vit_name` override winning over the guess, and
raising when the manual path doesn't exist) and new Qt-wiring tests
(browsing when nothing is auto-detected; the menu's "Browse…" entry;
`_start_training` passing the manual path through) — 700 total in
`studio/tests` + the repo suite, still green. Offscreen before/after
screenshots of the exact reported scenario (empty `sam_backbone_dir`):
"Not found" + non-interactive before, a real file picker click updating the
field to the chosen file's name and clearing the warning after.

## 2026-07-09 — Models & Train and Dashboard tabs wired end to end (P1 done)

Both `docs/velum/BACKLOG.md` P1 items in one pass — real one-shot LoRA
training and real experiment tracking, reusing the classic app's proven
pipeline exactly as `ARCHITECTURE.md` prescribes, with no change to the
mockup's look.

**Models & Train** (`studio/train_controller.py`, new): the 4-field train
card (Annotated image · SAM backbone · LoRA rank · Epochs) is now real.
`SelectBox` (`components.py`) gained an optional click-to-choose mode —
`options`/`on_select` pops a `QMenu` and updates its own text, `on_click`
opens a file picker instead — so all four fields work without inventing a
new widget or restyling. Picking an image looks for a same-stem mask (next
to the image, a sibling `masks/` folder, or the classic app's shared
`train_masks/`) and shows its real cell count; missing a mask disables Start
with an inline explanation. Start Training spawns
`napari_app.core.train_model.train_model` on a background thread (the exact
function the classic Train tab already calls), reporting live progress into
the "Recent training runs" aside via a guarded cross-thread signal (see
below). "Trained models" and "Recent training runs" both read real on-disk
state — `loras/*.json` sidecars and `training_history.json` — so a model
trained via the classic app already shows up here too. Clicking a trained
model writes it into the active project's settings (the "select into
workspace" hook the Segment tab will read once it's wired). "Import model"
copies an external checkpoint (+ sidecar) into the shared `loras/` folder.

One deliberate design choice worth recording: each training run's chosen
image+mask is copied into an **isolated** `studio_train_runs/<run_id>/`
folder rather than the classic app's shared, accumulating `train_images/`/
`train_masks/`. The mockup's UI only ever shows *one* image — training on
the shared folder would silently include every image ever picked in past
sessions too, which the UI never says and the user never agreed to. The
checkpoint output and the sidecar/history bookkeeping still land in the
same shared `loras/` folder either app uses.

**Dashboard** (`studio/dashboard_controller.py`, new): the training-loss
line chart, the F1-across-runs bar chart, and the Runs table are now real,
sourced from the same on-disk JSON (training history + per-checkpoint
sidecars + benchmarked project stats), not by querying Aim's storage
directly. That was tried first and abandoned after empirical testing:
`aim.Repo.get_run()` returns `None` for every hash and `Repo(...).
query_runs("")` raises `NotImplementedError` outside of Aim's own `aim up`
server process — confirmed against both a fresh throwaway repo *and* this
repo's real, 484-run `data_store/aim_repo`, so it isn't a fixture-data
fluke. "Open in Aim" still shells out to that real server
(`experiment_tracking.ensure_dashboard_running()`) and opens it in the
system browser — Studio's own charts stay fed by the robust, no-extra-
process path instead of trying to parse Aim's internals. Empty states (no
training yet, nothing benchmarked yet) render a plain "No runs yet" message
rather than crashing — the original static chart widgets call `min()`/
`max()` on their data and would otherwise throw on an empty list.

Also fixed in passing: a latent circular-import hazard in `screens.py`,
which imported `WorkspaceScreen`/`ModelsScreen`/`DashboardScreen` at its own
bottom purely for side effects — nothing in the file used them, and every
real caller already imports them directly from `workspace`/`extra_screens`.
It only "worked" by accident of import order (`app.py` always imports
`screens` first); importing `extra_screens` before `screens` anywhere — as
the new tests here do — hit `ImportError: cannot import name 'ModelsScreen'
from partially initialized module`. Deleted the two dead lines. Also
promoted `guide_screen.py`'s private `_bare()` helper (a plain `QWidget`
with an explicit `background: transparent`, working around the app-wide
QSS's `QWidget{background:<bg>}` rule painting an opaque patch inside a
lighter card — see the 2026-07-08 entries below) to a public
`components.bare_widget()`, since the Dashboard runs table needed the exact
same fix for its own per-row wrapper (`rowwrap = QFrame(); rowwrap.
setLayout(...)`, no stylesheet of its own) — a third file that would have
shipped the same latent bug otherwise.

Cross-thread safety: a training thread's completion callback can outlive
the `ModelsScreen` instance it targets (a theme toggle tears down and
rebuilds every screen). Guarded both signal emits with `except RuntimeError:
pass`, the same pattern `motion.py`'s hover closures already use for the
equivalent hazard, and added a regression test that force-deletes the
widget with `sip.delete()` and confirms the guarded emit doesn't raise —
mirroring how that hazard was originally caught.

**Verified:** `pytest studio/tests -q` (243 tests, incl. two new pure-logic
suites — `test_train_controller.py`, `test_dashboard_controller.py` — and a
new Qt-wiring suite, `test_extra_screens.py`); the full repo suite (688
tests) in the real env; the AGENTS.md throwaway-venv light-group check
(python3.10, since no bare python3.11/3.12 was available locally — the one
pre-existing failure, `tests/test_packaging.py`, is a documented py3.11+-only
file unrelated to this change); real offscreen screenshots of both tabs, in
both themes, in both empty (fresh install) and populated states — no
banded rows, no dark-canvas patches, correct real data throughout.
**Not verified:** the real GUI's live look/animation, and real torch
training (every test monkeypatches `TrainController.start_training` rather
than spawning actual model training — `train_model()` itself is exercised
by the classic app's own suite, not re-tested here).

---

## 2026-07-09 — Home's "This device" card: real hardware, not a hard-coded Mac string

First Linux check of the app (Studio had only ever run on macOS). Home's
"This device" card showed `("Compute", "Apple M-series · MPS")` unconditionally
— demo content nobody had wired, wrong on every non-Mac machine, and on Linux
specifically it said "MPS" (an Apple-only API) on a box with an NVIDIA GPU.

Added `studio/hardware.py` (`detect()` → a `DeviceInfo(kind, label, os_name)`,
lazy `import torch` inside the function so it stays out of Studio's shared-
module import graph): CUDA on Linux/Windows, MPS on Apple Silicon, `"<OS> ·
CPU"` as the honest fallback (e.g. `"Linux · CPU"`). The device-kind label
delegates the actual "is this GPU real" check to the new repo-root
`device_utils.py` (shared with the classic app's device dropdowns/status
label — see `docs/CHANGELOG.md` for that half) rather than trusting
`torch.cuda.is_available()` alone, since that call can be `True` for a GPU
the installed torch build ships no kernels for.

Verified: `studio/tests` (189 tests, +13 new in `test_hardware.py`) green
under real PyQt6 on Linux; confirmed end-to-end on a real Linux box with an
NVIDIA GTX 1070 — the Home screen now renders `"Linux · CPU"` (that card's
GPU is present but its torch build's CUDA 13 wheel doesn't ship kernels for
Pascal, so CPU is the honest answer there) instead of the old Mac string.
Not verified: rendering on an actual CUDA-usable Linux GPU or on Windows (the
capability-check logic is unit-tested with a fake torch module standing in
for both, not run against real matching hardware).

## 2026-07-09 — A third rendering bug in the same family: bare QWidget() wrappers

Third round of direct user feedback, pointing at two specific remaining
spots: the engine-comparison table (banded rows) and the keyboard-shortcuts
list (two-tone rows) still showed a dark patch. Related to, but distinct
from, the previous two fixes (unqualified stylesheets cascading; `inset` vs
`surface2`) — this was a *third* mechanism producing the same visual family
of bug.

Root cause: several plain `QWidget()` instances used purely to host a
sub-layout (`_table_block`'s per-row wrapper, `_shortcuts_block`'s
`keys_wrap`, and others) have no stylesheet of their own, so they inherit
the **app-wide** `QWidget { background: <bg> }` rule
(`theme.build_qss`, applied via `app.setStyleSheet()` at startup) and paint
an *opaque* bg-coloured rectangle wherever they sit. Invisible when a bare
wrapper sits directly on the page canvas (matches its surroundings exactly
— true for most of them); a visible dark patch when it sits inside an
already-lighter `surface2` card (true for these two). Confirmed with a
pixel-level render test before touching any code.

Fix: added `_bare()`, a small helper that returns a `QWidget` with
`background: transparent` set explicitly, and replaced every plain
`QWidget()` grouping wrapper in `guide_screen.py` with it (12 call sites) —
not just the two currently-visible ones, since this is the third time this
general class of mistake has shipped and a systematic fix is cheaper than
finding the next instance by screenshot again.

Also strengthened the test suite significantly: the first two regression
tests written for this (`test_table_block_row_fill_...`,
`test_shortcuts_block_keys_area_...`) *passed against the unfixed code* on
first write — twice, for two different reasons — before being corrected:
(1) they never applied the real app-wide stylesheet (`app.setStyleSheet
(theme.build_qss(...))`, extracted into a new `styled_app` fixture with
teardown, since the plain `app` fixture's `QApplication` is a
process-wide singleton shared with every other test module — the bug is
literally invisible without that stylesheet applied, so a test skipping it
passes regardless of whether the code is fixed); (2) they sampled a pixel
inside the *card's own margin* (or the *row's* own margin, for the
shortcuts case) rather than inside the actual bare-widget-under-test's
bounds — a coordinate immune to the bug either way, since nothing painted
by the wrapper ever reaches it. Fixed by giving the wrappers themselves
object names (`GuideTableRow`, `GuideShortcutKeys`) and sampling from their
own real `.geometry()`, confirmed by running each test against the
pre-fix code and watching it actually fail before trusting it.

Verified: `studio/tests` green. Confirmed all three new regression tests
fail against the pre-fix `guide_screen.py` and pass against the fix (this
took three attempts to get the tests themselves right — see above).
Confirmed visually via real offscreen screenshots across all 10 articles,
both themes. Not verified: how this reads on the user's actual display.

---

## 2026-07-09 — Guide & Docs: dropped the redundant outer "card" entirely

A second round of direct user feedback on the previous day's contrast fix:
still looked like a backing layer behind the content, "as if you'd pasted in
raw HTML." Right call, wrong fix — the token/shadow pass treated the
*symptom* (murky colour), not the actual cause.

The real problem: `GuideScreen` wrapped the **entire** nav rail and the
**entire** content pane each in one big bordered/filled panel — and *within*
that, the individual step/table/shortcut blocks were *also* boxed. Two
nested layers of "this is a boxed region," one of them serving no purpose
except to sit decoratively behind content that was already visually
structured on its own. Home and Projects never do this — every card there
is small and sized to its own content, floating directly on the page
canvas with visible gaps between; there's never an outer card whose only
job is to contain other cards.

Fix: removed the outer panel entirely from both `_build_nav()` and
`_build_content()` — no background, no border, no radius, just a plain
layout column. The search field, nav rows, and each content block (steps,
shortcuts, comparison table, callout, FAQ accordion) keep their own
(correct) styling and now float directly on the page background, the same
way Home's cards and body text already do. Confirmed via fresh offscreen
screenshots in both themes — reads as one cohesive page now, not boxes
inside a box inside a box.

Verified: `studio/tests` green (no test changes needed — nothing asserted
on the removed panels' styling specifically, only on behaviour, which is
unchanged). Confirmed visually via real offscreen screenshots, both themes.
Not verified: how this reads on the user's actual display — asking for
confirmation after this pass.

---

## 2026-07-08 — A real crash fixed, Guide gets a Close button, and a contrast fix

Direct user feedback on a real (non-offscreen) run, the same day Guide & Docs
shipped: the app aborted after a while of use, Guide had no way back to
where you were, and the guide read as one flat dark mass rather than
distinct panels.

- **The crash — root-caused and fixed, not just caught.** A macOS crash
  report showed `SIGABRT` inside `sipQFrame::enterEvent` → PyQt6's
  `pyqt6_err_print()` → `QMessageLogger::fatal()` → `abort()`: an unhandled
  Python exception escaping a Qt-invoked callback takes the whole process
  down, not just that interaction. Reproduced directly: `motion.
  install_hover_lift()`'s `enter`/`leave` closures (installed on every Home/
  Projects card) hold a `QGraphicsDropShadowEffect` + two
  `QPropertyAnimation`s that outlive the widget being torn down (e.g.
  `StudioWindow.toggle_theme()`'s `deleteLater()` rebuild, or any future
  screen teardown) — touching them from a stale hover callback raises
  `RuntimeError: wrapped C/C++ object ... has been deleted`, and that's what
  PyQt6 escalates to `abort()`. `fade_in()`'s `finished` callback had the
  identical hazard. Both now guard narrowly against `RuntimeError` (a
  genuine new bug still surfaces — this doesn't swallow exceptions
  generally); `studio/app.py:main()` also installs a `sys.excepthook` that
  logs instead of the PyQt6 default, as defense in depth for anything not
  yet found. New `studio/tests/test_motion.py` (motion.py had zero coverage
  before this) — confirmed these regression tests actually fail against the
  pre-fix code (not tautological) before confirming they pass against the
  fix.
- **Guide & Docs gets a Close button.** Every other full screen is a
  sidebar-nav peer (nothing to "close"), but Guide is reached the same way
  while conceptually being a utility panel like Assistant/Logs — which do
  have one. Added a ghost "Close" button to its header, navigating home.
- **Fixed the "everything looks like one dark canvas" complaint.** Two
  contributing causes, both fixed: (1) `soft_shadow()` on the two large
  nav/content panels — a soft shadow reads as "elevation" on a small
  floating card, but on a nearly-full-viewport panel it just smears into a
  murky halo against an already-dark page; dropped it, kept the plain
  border (matching how Workspace's own full-height panels already do it).
  (2) The bigger one: `_step_row`/`_shortcuts_block`/`_table_block` used
  `t['inset']` — the *recessed field well* token, meant for input boxes,
  darker than the page background itself — as the fill for large content
  blocks sitting inside an already-dark `surface` card. At that width it
  reads as a hole punched through the card to the canvas behind it, not a
  distinct raised row. Switched to `t['surface2']` ("elevated fill") —
  right token for "this sits *on* the card," confirmed lighter than
  `surface` in both themes.

Verified: `studio/tests` green. Repo-root throwaway-venv light-`test`-group
check passes clean. Confirmed the crash-path regression tests fail against
the pre-fix `motion.py` and pass against the fix (not just green by
construction). Confirmed the contrast fix visually via real offscreen
screenshots, both themes. Not verified here: the exact crash trigger
sequence on a real display (no way to reproduce a live mouse hover mid
theme-toggle-teardown outside a real session) — the fix addresses the
*confirmed* underlying hazard (touching a deleted Qt object from a stale
callback), which is sufficient regardless of the precise timing that
triggered it for the user.

---

## 2026-07-08 — Guide & Docs: real in-app documentation, not a no-op

Took the P2 backlog item "Guide & Docs screen (currently a no-op sidebar
item)" end to end. Home's "Documentation" and "Getting started guide"
resource links used to shell out to `QDesktopServices.openUrl()` on raw
`.md` files — `README.md`, and, worse, `docs/velum/OVERVIEW.md`, an internal
agent-facing dev doc with no business being shown to a microscopist. Neither
that nor the sidebar's "Guide & Docs" row (a literal no-op,
`open_guide.connect(lambda: None)`) held up as a real product surface.

- **`studio/guide_content.py`** — pure content, no Qt (mirrors `demo.py`'s
  spirit but is real, shipping copy, not placeholder data): 10 articles
  across 5 topics (Guide · Working with projects · Segmenting · Training ·
  Analysis), written for the product's actual audience — microscopists, not
  ML engineers (repo-root `AGENTS.md`) — and checked line-by-line against
  what's actually implemented today (exact engine keys/labels from
  `project.py`, the New Project wizard's real 3 step titles, the two real
  key bindings in `app.py`, the Segment workspace's actual panels from
  `workspace.py`) rather than aspirational copy. Assistant isn't documented
  at all — it isn't wired yet, and a diagnostic chat article that doesn't
  diagnose anything would be worse than no article.
- **`studio/guide_screen.py`** — `GuideScreen`: a searchable article nav rail
  + the selected article, composed entirely from existing atoms
  (`components.py`) and plain `QLabel`/`QFrame`, the same idiom every other
  screen already uses, rather than a rich-text engine — keeps typography and
  colour on the same tokens in both themes instead of fighting a second
  rendering paradigm's own defaults. Getting Started's steps are real
  actions, not just prose: "New Project" and "Open a sample" call the exact
  same callbacks `HomeScreen`'s quick cards do; "Go to Segment/Dashboard"
  navigates for real; "Choosing an engine" jumps to that article in place.
  Constructor mirrors `HomeScreen`/`ProjectsScreen` exactly (same 4
  callbacks) so wiring it into `app.py` was a one-line addition to
  `_STACK_KEYS` + the screens dict, not new plumbing.
- **Wiring**: sidebar's `open_guide` signal now navigates to `"guide"`
  instead of a no-op; `StudioWindow.navigate()` gained a `"guide:<id>"`
  prefix so a resource link can deep-link straight to an article (Getting
  started guide → the `getting-started` article) without changing the
  `Callable[[str], None]` signature every screen already takes.
  `_open_local_doc` (the raw-file-opening helper) is gone; GitHub is the one
  resource link still legitimately external.
- **`components.Accordion`** gained an additive `caps: bool = True` parameter
  (default preserves all 4 existing call sites byte-for-byte) — FAQ questions
  needed a full-sentence title, and the existing all-caps 11.5px micro-label
  treatment reads as shouting for a question like "Do I need a GPU?".
- **A real rendering bug, caught only by an actual offscreen screenshot, not
  by tests passing:** every paragraph/bullet/heading in the new screen
  painted with a second, tightly-fitted rounded-rect box around just its own
  text. Root cause: several card frames set their background/border/radius
  via an *unqualified* `setStyleSheet("background:…;border:…")` (no
  selector) — Qt Style Sheets cascade an unqualified rule to every
  descendant widget, and `QLabel` paints border/background natively (it's a
  `QFrame` subclass), so each label re-painted the same rounded box at its
  own small bounds. Invisible when a card's fill is opaque and identical to
  its children's inherited fill (the pre-existing, still-unfixed instances
  of this same pattern in `extra_screens.py`'s cards and
  `HomeScreen._card()`/its "Tip" callout — confirmed by an offscreen
  screenshot of Home showing the identical double-box on the Tip card's
  text, just easy to miss against small single-line labels); glaring here
  because the callout uses a translucent `primary_weak` fill that visibly
  doubles up, and because multi-line prose makes each stray box's rounded
  corners obvious. Reproduced in isolation (a minimal QFrame+QLabel repro,
  confirmed by scanning rendered pixels for the border colour) and fixed the
  same way `HomeScreen._quick_card`'s `#QCard` / `ProjectsScreen`'s `#PCard`
  already do it correctly: scope every card's stylesheet to its own
  `#ObjectName` selector instead of a bare/unqualified one, which stops the
  cascade at that widget. Left the pre-existing Home/Models/Dashboard
  instances alone (invisible in current usage, out of scope for this
  change) rather than drive-by refactoring unrelated screens.
- Also fixed along the way: a `QGraphicsDropShadowEffect` installed on the
  per-article content cards while 9 of the 10 start hidden inside a
  `QStackedWidget` — the exact bug class already diagnosed for the Projects
  list view (stale effect-source cache once later shown) — same fix,
  don't install the shadow there; the border alone still gives definition.

Verified: `studio/tests` green (166 tests, 32 net new — 35 added across the
two new files plus 4 app-wiring/sidebar tests, minus 3 removed
`_open_local_doc` tests that no longer apply: pure-content tests for the
article data — unique ids, step actions only reference real nav
keys/articles, no Assistant content, shortcuts match `app.py`'s actual key
bindings; headless screen tests — nav/search/selection, block renderers,
Getting Started's steps firing the real callbacks; sidebar/app wiring for
`open_guide` and `"guide:<id>"`). The repo-root throwaway-venv light-`test`-
group check passes clean (no PyQt6/torch/napari pulled in — the new pure
`guide_content` tests run for real there, the Qt ones skip via
`importorskip`, same as every other Studio Qt test). The rendering bug above
was caught and confirmed fixed via real offscreen screenshots
(`QT_QPA_PLATFORM=offscreen`, `QWidget.grab()`), in both themes, across
every article. Not verified here: on-screen behaviour with a real display
(font hinting, animation smoothness) and real model/file-system integration
(none of this touches the ML core).

---

## 2026-07-08 — Projects tab: three more real rendering bugs, from a live screenshot

A same-day follow-up after a real (non-offscreen) screenshot of the running
app showed the Projects toolbar/cards weren't actually matching the mockup
the way the previous pass's own offscreen renders had suggested. Three
underlying causes, all fixed at the source rather than patched around:

- **Card cover art wasn't rounded, and the engine label had no colour dot.**
  Root cause: `cover` (the nuclei-art `QLabel`) was a *raw, non-layout* child
  of `cwrap` (`cover.setParent(cwrap)`, never `cwl.addWidget(cover)` —
  needed so the star/engine-chip/progress overlay could stack *on top of*
  it rather than below it in a column) with no code keeping its geometry in
  sync — it kept whatever size an unparented `QLabel` happens to start with
  (640×480 in this environment; nothing to do with the card), stretched
  over the real card via `setScaledContents`. Any `border-radius` on that
  `QLabel`'s stylesheet was always a no-op too — Qt's QSS `border-radius`
  shapes a widget's own background/border, never a child's pixmap. Fixed
  properly, not patched: the cover is now a live-painting `NucleiView`
  (already existed for the workspace canvas) that recomputes a rounded-
  corner clip path from its own *current* size every paint — no distortion
  regardless of actual width — with an explicit `resizeEvent` on `cwrap`
  keeping it in sync (a raw child never gets this for free). Thumbnails
  (`cover_label`, Home's recent rows + the Projects list view) get the
  fixed-size equivalent: the radius baked straight into `nuclei_pixmap`
  (new `radius`/`top_only` params). New `components.EngineChip` adds the
  mockup's missing per-engine colour dot (`.chip` + `.cd`) — reusing
  `theme.VIZ` for the three engine hues rather than inventing new tokens.
- **The toolbar's controls didn't line up.** The search box, the All/
  Favorites/Shared segmented control, Filter, and the grid/list toggle each
  computed their own height from padding + real Figtree font metrics
  instead of a shared explicit one — converging closely enough in an
  offscreen dev render to look fine, but visibly drifting apart under the
  bundled font on a real display (reported directly: "Filter renders
  shorter than its neighbour"). Every control but the page header's primary
  "New Project" CTA (deliberately excluded — it's a bigger, separate call
  to action by design, matching the mockup's own distinct `.btn` vs
  `.btn-sm`) now shares one explicit height (`ProjectsScreen._TOOLBAR_H`).
  Also widened the search box (max-width 420→560, plus a stretch factor so
  it actually grows to use the room) — at this window's real width the
  card grid runs meaningfully wider than the mockup's fixed 1300px hero
  shot ever needed to plan for, and 420px reads as cramped next to it.
- Repo-root throwaway-venv check still exits 0 clean, so none of this pulled
  in a heavy dependency.

Verified: `studio/tests` green (144 tests, 8 new — the rounded-corner clip
path top-only vs. all-four-corners, the raw-child geometry-sync pattern in
isolation, toolbar-height equality, and the engine chip's dot colour);
offscreen-screenshot-reconfirmed end to end (both themes, grid + list +
every scope/filter state) that covers now clip correctly at every card
width tested (305/450/620px, not just the one size a single screenshot
happens to catch), the dot renders in each engine's own hue, and the
toolbar sits flush. Not verified here (no physical display): the exact
Figtree-rendered pixel heights on a real screen — fixing every control to
one explicit height sidesteps needing to reproduce that mismatch exactly,
but the *original* reported drift was only ever visible on a real display,
not this offscreen setup.

---

## 2026-07-08 — Projects tab: full toolbar fidelity + real grid/list views

A design-fidelity + functionality pass on the Projects tab against the
north-star mockup, prompted by a side-by-side review against the mockup
artifact. The previous pass wired data (search/favourites/store); this one
fixes everything the toolbar/cards still got wrong or left dead:

- **The mockup's "All · Favorites · Shared" segmented control was missing
  entirely** — the "Filter" button had been repurposed as a favourites-only
  toggle instead. Restored the real 3-way `SegControl` (matching the mockup
  exactly) and gave "Filter" its own, real job: a checkable engine multi-select
  popover (`QMenu`, one entry per engine) — composes with search and the scope
  tab. "Shared" is a genuine, wired scope (not a dead label): it always yields
  zero projects, honestly, since Studio has no multi-user/sharing backend
  anywhere in the roadmap — with its own empty-state message rather than a
  silent blank grid. `ProjectController.list_projects()` gained an `engines=`
  filter to back this (pure-logic, tested).
- **The grid/list view toggle was decorative** — two text glyphs (`▦`/`☰`)
  that changed nothing when clicked. `SegControl` (`components.py`) now
  supports icon-only segments (an `icons_=` param, backward-compatible with
  every existing text-only caller) so the toggle uses the mockup's actual
  grid/list SVGs; clicking it now really switches between the card grid and a
  new dense list view (row = cover thumb, name/engine/stats meta line, F1,
  favourite star — reusing Home's `.rrow` visual language, since the mockup
  itself never designed a Projects list view to match against). Both views
  stay populated behind the scenes so toggling is instant.
- **The Projects tab's own "+ New Project"** (top-right CTA and the grid's
  ghost card) still just navigated straight to a blank workspace — the New
  Project dialog existed (wired to Home a pass ago) but nothing on this
  screen opened it. Both now open the real dialog, same as Home.
- **Pixel fidelity against the mockup CSS** (fetched and read directly, not
  eyeballed): cover art 120→132px, the ghost "new project" card's
  `min-height` 240→290px plus its missing 44×44 rounded plus-icon box, the
  search icon (was reusing the "diagnose"/magnifier icon — close but not the
  mockup's own path), stats row gap 16→14px, card-body spacing tightened to
  match `padding-top`+`border-top`+`margin-top` (was 19px total, now 25px),
  star/engine-chip/progress-badge overlay margins 8/12→10px, and a
  `install_hover_lift` shadow-elevation on hover (cards had a permanently-on
  static shadow instead of the mockup's rest→hover shadow-sm→shadow-md
  transition; list rows deliberately do *not* get this — see the bug below).
  Grid columns now get explicit equal stretch, so a heavily filtered result
  (1–2 cards) no longer renders one card stretched absurdly wide.
- **Two real rendering bugs caught by actually looking at offscreen
  screenshots, not just construct-without-crashing tests:**
  1. The non-favourited star icon was invisible on every project card. Root
     cause: `icons.py` hands its colour argument straight into an SVG
     `stroke="..."` attribute, and `QSvgRenderer` silently drops CSS
     `rgba(255,255,255,0.65)` syntax there (no error — zero pixels drawn).
     The favourited state used a plain hex (`#f0b357`) and was fine, which is
     exactly why this went unnoticed since the tab was first wired. Confirmed
     by direct pixel-count rendering of the SVG in isolation, fixed with an
     opaque muted grey, and locked down with a new regression test that
     renders the icon and asserts at least one non-transparent pixel — a
     construct-only test would never have caught this.
  2. Switching to list view could render rows with wildly inflated spacing
     and an overlapping ghost row. Root cause: list rows call
     `install_hover_lift()` (a `QGraphicsDropShadowEffect`) while their
     container starts hidden (list isn't the default view) — Qt's effect
     source cache goes stale once the container is later shown, and the
     widgets paint at the wrong extents despite reporting correct
     `.geometry()`. Confirmed by toggling `install_hover_lift` off and
     watching the bug disappear; fixed by not installing it on list rows
     (the existing QSS `:hover` border-color still gives real hover
     feedback). Grid cards are unaffected — they're visible from
     construction, since grid is the default view.

Verified: `studio/tests` green (136 tests, 21 new — engine-filter/scope/view
toggle logic, the two regression tests above, new-project wiring for the
Projects tab's own CTA/ghost/ghost-row); repo-root throwaway-venv check (`pip
install --group test` only, Python 3.11, no torch/napari/PyQt6) exit code 0,
zero failures. Offscreen-screenshot-verified end to end (`QWidget.grab()`,
both themes): grid view, list view, the Favorites/Shared scopes (including
the real button-click path via `SegControl._select()`, not just calling the
handler), the engine filter's active-button restyle, and the shared-scope
empty state — all matched intent, including the two bugs above being
visually confirmed fixed after the code changes, not just asserted by a
passing test. Not verified here (no physical display): real hover-lift
animation smoothness, real QMenu popover interaction (its resulting filter
logic is tested directly; the popup itself is a thin, hard-to-drive-headless
Qt native menu).

## 2026-07-08 — Fix: a newly created project didn't show up until restart

User-reported: creating a project via the New Project modal didn't add it to
Home's "Recent projects" or the Projects grid until the whole app was
relaunched. Root cause: `HomeScreen`/`ProjectsScreen` are built once and kept
alive across navigation (`StudioWindow.navigate()` just swaps the visible
`QStackedWidget` page) — so their content reflected whatever the store looked
like at construction time, and nothing ever told them to rebuild afterwards.

- `HomeScreen.refresh()` / `ProjectsScreen.refresh()` (new) rebuild the
  recent-projects list / the grid + header counts from the store's current
  state. `StudioWindow.navigate()` now calls a screen's `refresh()` (if it
  has one) every time it becomes the active page — so switching to Home or
  Projects always shows current data, not just right after a create.
  `ProjectsScreen.refresh()` preserves whatever search/favourites-only
  filter was already active.
- Also fixed while verifying this: `Toast`'s subtitle could get clipped
  instead of wrapping for a long project name + engine combination
  (`setWordWrap` + a max width, rather than relying on `adjustSize()` timing
  after a dynamic `setText()`).

Verified: reproduced the exact bug first (constructed the window, created a
project, counted rendered project-card/row widgets before and after — stayed
at 0 after create+navigate without the fix, confirming the root cause), then
confirmed the fix with the same reproduction. Three new regression tests
(`studio/tests`, now 118 total, all green): `HomeScreen`/`ProjectsScreen`
picking up a project created directly through the store, and a full
app-level end-to-end test creating a project through the real dialog and
navigating to both Home and Projects. Repo-root throwaway-venv check (`pip
install --group test` only) still green. Offscreen-screenshot-reconfirmed
(`QWidget.grab()`): the new project now appears immediately in Home's recent
list after creation, and the toast wraps correctly for a long name.

## 2026-07-08 — Home screen: every element real, + the New Project modal

Follow-up pass focused entirely on Home (Projects tab intentionally left
alone this round):

- **New `studio/new_project_dialog.py`** — the "+ New Project" flow ticked
  off `BACKLOG.md`'s own item: a scrim-backed modal (identical construction
  to `overlays.CommandPalette` — no native `QDialog` frame, stays consistent
  with the app owning its own chrome) with the 3-step Label Studio pattern:
  name + description → import images (a real drag-and-drop zone plus a
  native file picker, both funnelling into the same add/remove-file state) →
  engine (`SegControl` over the same three engines everywhere else in the
  app). "Create Project" writes through the real `ProjectStore.create()` and
  opens straight into the workspace, reusing the Projects tab's existing
  active-project flow.
- **Every Home element is now a real action**, not just the Projects grid:
  the "New Project" CTA and quick card, and the "Import Images" quick card,
  open the new dialog; "Train a Model" navigates to Models & Train; "Open
  Sample" opens an existing project if one exists, or opens the dialog when
  the store is empty; "Ask the Assistant" opens the Assistant drawer;
  "Documentation"/"Getting started guide" open real local docs
  (`README.md` / `docs/velum/OVERVIEW.md`) and "GitHub" opens the real origin
  remote (read from `git remote get-url origin` at runtime, converted to an
  `https://` URL — never a hard-coded/guessed link, and it degrades to a
  no-op if there's no remote).
- **`Toast.announce()`** — the bottom-right success toast existed since the
  design-skeleton phase but nothing had ever called `.show()` on it; project
  creation is its first real trigger ("Project created · <name> · N images ·
  engine"), auto-hiding on a timer.
- **Hover "lift"** on Home's quick cards and recent-project rows, matching
  the north-star mockup's `.qcard:hover`/`.rrow:hover` CSS
  (`transform:translateY()` + a deeper shadow, ~160ms). QSS has no
  `transform`/`transition`, so `motion.install_hover_lift()` animates a
  `QGraphicsDropShadowEffect`'s blur/offset instead — same "the card is
  rising toward you" read, without fighting Qt's layout engine.

Verified: `studio/tests` green, 115 tests (28 new: a `test_new_project_dialog.py`
covering the full step flow — validation, back/forward, persistence across
steps, file add/remove, and a real end-to-end create-through-the-store; a
`test_home_wiring.py` covering every quick-card/resource-link callback,
`QDesktopServices.openUrl` mocked rather than actually invoked so tests never
really open a browser). Two real bugs the new tests caught before shipping:
`_go_next()` relied entirely on the Next button being disabled to block an
empty project name (fixed with its own guard); a test asserting
`isVisible()` on a dialog button needed the test's own parent widget shown
first (`isHidden()` is the explicit per-widget flag; `isVisible()` needs the
whole ancestor chain actually shown — same distinction already called out in
`test_app_wiring.py`). Repo-root throwaway-venv check (`pip install --group
test` only) still passes, 380 passed / 14 skipped, confirming nothing in
this round leaked a heavy dependency into the light CI group.

Offscreen screenshot verification this round (`QWidget.grab()` under
`QT_QPA_PLATFORM=offscreen`, both themes, all 3 dialog steps, plus a hover
state settled via `QTest.qWait`): layout, spacing, data and the dialog flow
all matched the design intent. One rendering artifact showed up (thin outline
boxes around label text inside scrim-backed panels) — traced to a pre-existing
offscreen-QPA quirk by reproducing it identically in the untouched
`CommandPalette`, so it isn't a real bug and isn't expected on a real display;
not independently re-verified with a physical display. The Projects tab
(left untouched this round) also rendered correctly in these screenshots.

## 2026-07-08 — Projects tab wired end to end (first real tab, skeleton → functional)

The Projects tab is no longer demo cards — the first tab taken from
`BACKLOG.md`'s "how to wire a tab" recipe, end to end:

- **`studio/project.py` reintroduced** (the `Project`/`ProjectSettings`/
  `ProjectStats`/`ProjectStore` data model, pure stdlib, previously removed in
  the design-skeleton reset and preserved in git history) — adapted with
  `ENGINE_LABELS`/`ENGINE_KIND` display-mapping constants (now the single
  source every screen draws engine colour/label from) and a `touch=` escape
  hatch on `ProjectStore.save` for callers that need explicit, deterministic
  timestamps instead of "now".
- **New `studio/project_controller.py`** — a Qt-free `ProjectController`
  (mirrors `napari_app/core/predict_controller.py`'s shape): search/filter,
  favourites, the "active project" shared with the Workspace tab, and
  first-run sample seeding so a fresh install still shows the same 6 sample
  projects the mockup always had — now real, persisted `Project` records
  instead of hard-coded `demo` content. Small pure formatting helpers
  (`to_card`, `format_count`, `relative_time`, `cover_seed`) keep screens.py
  free of formatting/date logic.
- **Home + Projects screens bound to the controller**: `demo.PROJECTS` /
  `demo.RECENT_WHEN` reads replaced with live data; the search box and a new
  "favourites only" toggle on the existing "Filter" button live-filter the
  grid; a favourite star (new, on each card — the data model always had
  `favorite`, the static skeleton just never rendered an affordance for it)
  toggles and persists through the store; open callbacks switched from
  list-index to project-id (index broke once filtering could reorder/drop
  cards); the page header's counts are now computed from real data.
- **"Active project" shared to the Workspace tab**: `WorkspaceScreen.
  set_active_project()` updates the top-bar breadcrumb + engine chip — no
  longer hardcoded to "Fluorescence Nuclei — DAPI" / "CellSeg1 · LoRA"
  regardless of what you actually opened, and shows a neutral "No project
  selected" state before any project is opened. Survives the theme-toggle
  rebuild. The rest of the Workspace (layers, canvas, predict) is still the
  Segment tab's own, separate, not-yet-started backlog item.
- Cover art seeds are now derived deterministically from each project's id
  (`zlib.crc32`) rather than the arbitrary integers `demo.py` used, since a
  real project has no "seed" field to persist — same procedural nuclei-art
  look, stable per project across relaunches, just no longer pinned to the
  exact noise pattern the static mockup happened to show.
- **Not wired here** (separate BACKLOG items): the "+ New Project" creation
  dialog itself — cards/ghost-card still just navigate to a blank workspace,
  no create-through-the-store flow yet.

Verified: `studio/tests` green (87 tests: reintroduced + extended the historic
pure-logic `project.py` suite, added a pure-logic `project_controller.py`
suite, extended `test_app_wiring.py`'s screen/window tests to inject a
`tmp_path`-backed controller — real `data_store/projects` is never touched by
tests); the repo-root throwaway-venv check (`pip install --group test` only,
Python 3.11, no torch/napari/PyQt6) collects and passes both `project.py` and
`project_controller.py`'s suites for real (64 passed, `test_app_wiring.py`
correctly skips as one unit via `importorskip("PyQt6")`); an offscreen
end-to-end smoke run against the real default store (`data_store/projects`,
which already has 4 real local projects from earlier manual testing) —
construct, list, navigate to Projects, open a project, confirm the workspace
breadcrumb and active-project state — all passed with neither napari nor
torch imported. Not verified here (no display): the live look/animations of
the new favourite star and filter-toggle states.

## 2026-07-08 — Studio is now its own top-level project + docs pivot to "own canvas"

Structural + directional clarity, no behaviour change:

- **Studio promoted to a top-level `studio/` package** (`git mv` from
  `napari_app/studio/`, history preserved), a **sibling** of the classic
  `napari_app/` (old app) and the shared ML core — the standard monorepo
  "old app + new app + shared core" shape, so the branch reads as its own
  project. Studio is now **self-contained**: its own `icons.py` (the mockup's
  icons, not the classic app's) and `motion.py`; it imports nothing from
  `napari_app`. The classic `napari_app/icons.py` was reverted to pristine.
- **Studio has its own test suite** in `studio/tests/` (run `pytest
  studio/tests`); `pytest.ini` includes it; packaging/entry point updated
  (`cellseg1-studio = studio.app:main`, `studio/` packaged, tests excluded).
- **Docs pivot — we are NOT embedding napari.** The Segment tab will get our
  **own** canvas (like Label Studio's / napari's viewers, but ours: own tool
  strip, own layer model, own interactions), reusing only the **ML logic**
  (engines/predict/morphometry). New guiding principle across the docs: *own
  the UI, the icons, the canvas, the settings; reuse the logic.* Label Studio
  reaffirmed as the primary **structure** reference (not look). AGENT_PROMPT
  gained explicit git-sync (keep local↔remote in sync) and "run only
  `studio/tests`" guidance.

Verified: full suite 473 passed; Studio's suite green from its new location;
the app imports and boots from the top-level `studio` package offscreen,
importing neither napari nor torch.

## 2026-07-07 — Design skeleton: the mockup, reproduced natively (no logic)

Reset Studio to a pure **design skeleton** — a faithful, static, native-Qt
reproduction of the north-star mockup with **all business logic removed** — so
there's a clean, consistent target to wire functionality against, tab by tab.

- **Stripped all logic** from the running app: no napari, no torch, no model,
  no project/file IO. `import napari` / `import torch` never runs; the app
  launches on PyQt6 alone. Removed the wired-in `PredictWidget`/`TrainWidget`
  hosting and the `project.py` data model (preserved in git history; returns
  when the Projects tab is wired).
- **Native reproduction of every mockup screen** with static demo content
  (`demo.py`): Home, Projects, the Segment workspace (adapted-napari
  **Images|Layers** panel with full layer controls · nuclei canvas · **Segment|
  Results** inspector), Models & Train, Dashboard — plus overlays: Assistant
  drawer, Logs console, ⌘K command palette, toast.
- **Design system** as reusable modules: `theme.py` (tokens), `components.py`
  (the UI-kit atoms + sidebar), `paint.py` (a QPainter nuclei stand-in for the
  canvas / card covers / thumbnails).
- **Rounded window corners** (12px rounded mask) on the frameless window.
- **`docs/velum/`** — this doc set (OVERVIEW, DESIGN, ARCHITECTURE, BACKLOG,
  ROADMAP, CHANGELOG, AGENT_PROMPT) driving the tab-by-tab plan.

Verified: full pure-logic suite green; the app boots offscreen and navigates
every screen, opens all overlays, toggles theme and resizes cleanly, importing
**neither napari nor torch**. Not verified here (no display): the live look,
the rounded corners (offscreen can't set window masks — real macOS can), fades.

### Earlier (foundation, superseded by the reset above)
- Frameless window + own dark title bar (own traffic lights, native
  move/resize via `startSystemMove` + grips) replacing the grey OS title bar.
- First shell: sidebar + Home/Projects backed by a `ProjectStore`, embedding
  the classic `PredictWidget`. Reset to a logic-free skeleton on the same day.
