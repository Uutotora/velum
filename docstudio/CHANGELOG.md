# Changelog — CellSeg1 Studio

What actually shipped in Studio, dated, newest first. (The repo-wide log is
`docs/CHANGELOG.md`; this one is Studio-specific.)

---

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

Both `docstudio/BACKLOG.md` P1 items in one pass — real one-shot LoRA
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
`.md` files — `README.md`, and, worse, `docstudio/OVERVIEW.md`, an internal
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
  (`README.md` / `docstudio/OVERVIEW.md`) and "GitHub" opens the real origin
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
- **`docstudio/`** — this doc set (OVERVIEW, DESIGN, ARCHITECTURE, BACKLOG,
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
