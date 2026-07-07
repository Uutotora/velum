# Changelog

What actually shipped, dated, in plain language — as opposed to
`docs/BACKLOG.md` (what's planned) or `docs/AUDIT_2026.md` (a point-in-time
strategic assessment). Newest first.

**Why this file exists:** on 2026-07-05 a full UI redesign (~17 commits, see
below) landed with no corresponding `docs/BACKLOG.md` entry — it wasn't
planned as a task, so nothing recorded that it happened, and
`docs/AUDIT_2026.md`'s UX section quietly went stale. This file is the fix:
**every meaningful change gets a line here, planned or not** (see the
"Working agreement" in `AGENTS.md`). A line here costs one sentence; not
having one cost a confused reconciliation pass across three documents.

Full detail always lives in `git log` — this is the curated, product-level
narrative, not a mirror of it. Don't transcribe every commit; one bullet per
*change a user or the next agent would want to know about*.

---

## 2026-07-07 (night, later) — the real reason PyQt6-WebEngine kept saying "not installed": a Qt startup-order requirement, not this app's dependency detection

User installed `PyQt6-WebEngine` and still saw "Embedded view needs the
optional PyQt6-WebEngine package" in the Dashboard window. Two compounding
bugs, both real:

1. `DashboardWindow` is a singleton (one shared instance across every
   "Dashboard" button, by design — see the earlier entry on why). Whether
   it builds the embedded view or the fallback message was decided *once*,
   at first construction, and never revisited — so a window opened before
   the package was installed would show the fallback forever, even after
   installing it, until the whole app restarted. Fixed:
   `_upgrade_to_embedded_view_if_possible()` now re-checks on every open
   and swaps the fallback out for a real view the moment it becomes
   available, no restart needed.
2. The deeper one: `PyQt6.QtWebEngineWidgets` itself refuses to import
   unless `Qt.AA_ShareOpenGLContexts` was set *before the process's first
   `QApplication` was constructed* — otherwise it raises `ImportError:
   QtWebEngineWidgets must be imported or Qt.AA_ShareOpenGLContexts must be
   set before a QCoreApplication instance is created`, indistinguishable
   from "not installed" to `_has_webengine()`'s broad except clause.
   `napari_app/main.py` calls `napari.Viewer()` (which constructs the
   QApplication) long before any Dashboard code ever runs, and never set
   this attribute — so the lazy `import PyQt6.QtWebEngineWidgets` inside
   `_has_webengine()`, reached only when a user actually clicks
   "Dashboard", was **structurally unreachable no matter what was
   installed**. This means fix #1 alone would not have been enough — even
   with instant re-checking, the check itself was doomed. Fixed by setting
   `QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)` at the very
   top of `main.py`, before `napari` is even imported.

Verified both fixes against reproductions of the exact failure (a bare
`QApplication()` with no attribute set, then a lazy `QtWebEngineWidgets`
import — reliably fails) and the fix (attribute set first — reliably
succeeds), and confirmed importing the real `napari_app.main` module (the
actual entry point) makes `_has_webengine()` return `True` even when
checked long after its own `QApplication` already exists, matching a real
"click Dashboard deep into a running session" timing.

3 new tests for the dynamic-upgrade logic (against a patched
`_build_embedded_view`, not a real `QWebEngineView` — real construction
needs the same attribute-before-QApplication ordering this shared test
process can't reliably guarantee across files, so the tests exercise the
window's own swap logic instead). Full suite green in the full conda env
(445 passed, with `aim` and `PyQt6-WebEngine` both genuinely installed
there now) and the light venv (316 passed, 11 skipped).

## 2026-07-07 (night) — the embedded dashboard view raced Aim's server startup and loaded blank

The user installed `PyQt6-WebEngine` (the `tracking-ui` extra) and reported
the embedded Dashboard view staying blank while "Open in browser" — now
fixed — worked. Installed the real package here too and reproduced it: the
`QWebEngineView`'s first `loadFinished` fired `False`. Root cause:
`open_dashboard()` calls `ensure_dashboard_running()` (spawns the `aim up`
subprocess) and immediately points the view at its URL — but a freshly
`Popen`'d web server isn't listening yet the instant the subprocess object
exists, so the very first load attempt can race it (confirmed: a *second*
attempt ~1s later succeeded, with the real Aim page's DOM actually
populated — 116KB of HTML, `document.title == "Aim"`).

Fixed with a non-blocking retry in `DashboardWindow` itself (not a blocking
wait in `experiment_tracking.py`, which callers reach from the GUI thread —
sleeping there would freeze the whole app): on `loadFinished(False)`,
`QTimer.singleShot` re-triggers the load after 700ms, up to 8 attempts,
giving up with a "try Open in browser" status message rather than retrying
forever. 4 new tests drive the retry state machine against a fake view
(`QTimer.singleShot` monkeypatched to fire immediately) — fast and
deterministic, no real Chromium involved.

**Sandbox limitation, not re-verified against a real screen:** even after
the DOM genuinely populated in the real-install check above, a `.grab()`
screenshot of the embedded view stayed blank, with the Chromium GPU process
logging `Unable to initialize SkSurface` / `Context lost during
MakeCurrent` / `Failed to make current since context is marked as lost` —
consistent with Chromium's compositor failing to get a working (even
software) rendering surface specifically under `QT_QPA_PLATFORM=offscreen`,
not with anything actually wrong in the page or this app's code. The DOM
content is confirmed correct; the *pixels* on a real display with real GPU
acceleration are not confirmed here and need the user's own look.

## 2026-07-07 (evening, later) — installed Aim for real in the app's actual env; "Open in browser" was silently inert, not broken

The previous entry's real-Aim verification ran in a throwaway venv, kept
deliberately separate from this repo's shared conda env. Once the user
started actually using the feature in the real running app, that
separation stopped being useful and started being the problem: their
Dashboard window correctly reported "Aim is not installed", because in
*that* environment, it genuinely wasn't. Installed `aim` for real into the
`cellseg1` conda env `run_napari.sh`/the `cellseg1` launcher actually use.

The "Open in browser" button "not working" was a direct, correct
consequence of that — `_open_in_browser` already no-ops without a URL — but
the button gave no visual signal it was inert, so it read as broken rather
than "waiting". Fixed: the button now starts disabled, enables only once
`open_dashboard()` gets a real URL, and disables again if a later retry
fails (e.g. the server died) — matching the tooltip to whichever state it's
in. 2 new tests for both transitions. No app restart needed to pick up the
newly-installed package — Python doesn't cache a *failed* import, so the
next click of any "Dashboard" button retries `import aim` fresh.

## 2026-07-07 (evening) — real-Aim verification found and fixed a genuine bug; Train's history card replaced

User report: the "Dashboard" button wasn't visible in Predict/Train.
Investigated by rendering both widgets under offscreen Qt and grabbing a
real screenshot (`QWidget.grab()`), not just an import check — the button
*was* there, correctly placed next to "Log" in both footers, so this was
very likely a stale already-running app process from before the previous
PR merged, not a code defect (worth remembering this screenshot technique
for future UI questions in this sandbox — it catches what a plain headless
import can't).

While already investigating, closed the previous entry's one real
remaining gap: installed the actual `aim` package into a throwaway venv
(not this repo's shared conda env) and drove `experiment_tracking.py` for
real. A real `aim.Run` worked exactly as the wrapper assumed — **except**
`ensure_dashboard_running()` failed with `FileNotFoundError: 'aim'`:
`subprocess.Popen(["aim", ...])` resolves through `PATH`, which a
pip-installed console-script isn't guaranteed to be on when invoked outside
an activated shell (exactly how a packaged desktop app plausibly launches
it). Fixed with `_aim_cli_path()`: prefer the sibling of `sys.executable`
(where pip actually puts the script, `PATH`-independent), falling back to
the bare name otherwise. Re-verified against the same real install
afterward: the dashboard now actually launches and answers `HTTP 200` with
real Aim UI HTML.

Also replaced Train's old plain-text "Training history" card
(session-only, no comparison) with a "Run history" card pointing at the
same Dashboard — one real history UI instead of two competing ones.

3 new tests (433 → 436): `_aim_cli_path()`'s two branches, and a guard that
the old history box/method stay removed. Full suite green in the full
conda env and the light `pip install --group test` venv.

## 2026-07-07 (afternoon, later) — unified experiment tracking (Aim), not a backlog item

Direct request following the auto-tune follow-up: an open-source dashboard
that predict/train/auto-tune could all log to, instead of hand-rolling
another custom chart. Compared Aim, TensorBoard, MLflow and ClearML;
picked **Aim** — fully local/self-hosted (no account, matching this app's
existing "nothing leaves your machine" stance, unlike W&B's cloud-first
model), a modern UI explicitly built to out-do TensorBoard, and its
run/hparam-comparison shape is the closest match to what this app's three
producers (one-shot predicts, multi-epoch training, multi-round auto-tune)
all generate. Full writeup in `docs/BACKLOG.md`'s new "Unified experiment
tracking (Aim)" entry.

- New `napari_app/core/experiment_tracking.py`: `start_run(experiment,
  hparams)` — a real `aim.Run`-backed handle or a `_NullRun` no-op,
  lazy-imported and `available()`-gated exactly like sam2/cellpose/Ollama
  elsewhere (a new `tracking` extra, not a hard dependency). Every call on
  the real handle is individually guarded, so a tracking hiccup degrades to
  one dropped data point, never an interrupted real run.
- `predict_controller.py` now logs one run per single predict, one per
  batch image, and one per benchmark (engine × image) pair with the real
  AP/F1 metrics already computed there; the auto-tune loop logs one run for
  the whole loop, tracking score/cell-count per round plus the final stop
  reason. `core/train_model.py` now also tracks loss per epoch (a first
  for that file) alongside its existing `LossChart` — Aim is an addition,
  not a replacement.
- New `napari_app/widgets/dashboard_window.py`: a floating singleton window
  (the `log_window.py`/`measurements_window.py` pattern) embedding Aim's
  real web UI via `QWebEngineView` when the separate, heavier
  `tracking-ui` extra (`PyQt6-WebEngine` — bundles Chromium, ~+100MB) is
  installed, always with an "Open in browser" fallback that needs only the
  base `tracking` extra. A "Dashboard" button was added to Predict's and
  Train's log-footer row and to the Assistant's Auto-tune card, all three
  sharing the same window/repo.

40 new tests (393 → 433): the `tests/test_engines_sam2.py` pattern of
injecting a bare `types.ModuleType("aim")` into `sys.modules` rather than
installing the real package, covering the no-op path, the real-module path
via a recording fake `aim.Run`, every guarded-failure path, the dashboard
subprocess singleton/reuse/restart logic, and each orchestration method's/
widget's own wiring (`train_model.py`'s tests are a first for that file,
which had zero coverage before). Full suite green in the full conda env
(433 passed) and in a from-scratch venv with only `pip install --group
test` (314 passed, 11 skipped — the 2 new skips are exactly the new
torch-gated and PyQt6-gated files).

**Not verified:** a real Aim install (no network access to confirm `aim up`
actually serves a working dashboard for this app's specific hparam shapes)
and the embedded `QWebEngineView` path (`PyQt6-WebEngine` isn't installed
either) — every test proves the wrapping code is correct assuming Aim's
documented API surface, not that a real Aim server behaves as documented.

## 2026-07-07 (afternoon) — agentic tuning loop follow-up: a real LLM tool-calling strategy, live chart + leaderboard + CSV export, parameter importance

Direct instruction, right after the first pass landed: don't stop at the
minimum that satisfies the acceptance criteria — go all the way, and
research how comparable products instrument this kind of loop before
building further ("больше полезной информации — я смогу её убрать", i.e.
bias toward more instrumentation, not less). Researched AutoML/sweep
dashboards (Optuna, Weights & Biases), the ReAct tool-calling-loop pattern,
agentic-coding-tool UX (Cursor/Devin checkpoints), and this app's own
segmentation neighbourhood (CellSeg3D's grid-search tuner, the
"the-segmentation-game" napari plugin's Highscore table) — see
`docs/BACKLOG.md`'s "Agentic tuning loop" entry for the full writeup and
which of it is grounded in which precedent. Four real additions, not
cosmetic:

- **A genuine LLM tool-calling strategy** (closes the first pass's own
  "deliberately out of scope" note). `tuning_loop.llm_propose_fn` hands the
  "what to change next" decision to a connected local Ollama model each
  round — a real ReAct round (reason, then act), reusing the chat's
  existing `SUGGEST:` protocol plus a new `STOP: <reason>` one
  (`advisor.build_tuning_prompt`/`advisor.parse_stop`). Falls back to the
  rule-based advisor on any model error/timeout/unusable reply. A new
  "Strategy" combo in the Assistant only offers it once a local model is
  actually connected.
- **A settings panel** (closes the first pass's other "deliberately out of
  scope" note): `max_steps`/`patience`/`min_delta` are now three spinboxes
  in the Assistant's new "Auto-tune" card, not fixed constants.
- **Product-grade trajectory instrumentation**, replacing the first pass's
  growing pile of per-round chat cards: a live score-vs-round chart
  (pyqtgraph/matplotlib fallback, the same one `train_widget.LossChart`
  already uses), a sortable leaderboard table (click a round → "Use
  selected round" restores it, a Cursor/Devin-style one-click checkpoint),
  a CSV export of the full trajectory, and a "what mattered most" panel
  (Pearson correlation between each varying parameter and the score —
  W&B's parameter-importance idea without needing enough trials for its
  random-forest model). The chat keeps one short line per round for the
  conversational narrative; the rich data lives in the dedicated card.
- **Every stop is explained, not just coded**: `TuningResult` now carries a
  free-text `stop_detail` (the advisor's or the model's own words for why
  it stopped) alongside the machine-readable `stop_reason`, surfaced
  verbatim in the finish banner instead of discarded once the loop exits.

24 net-new tests (369 → 393; the pure-loop and controller suites were also
rewritten in place for the new `Proposal`/`TuningResult`/4-arg-`ProposeFn`
contracts, so more test functions than that actually changed). Full suite
green in the full conda env (393 passed) and in a from-scratch venv with
only `pip install --group test` (288 passed, 9 skipped for PyQt6-gated
files).

**Not verified:** the real GUI (chart/table rendering on an actual screen)
and a real local model's actual tuning judgement — the LLM strategy's tests
use a scripted fake `ollama_chat`, proving the plumbing around whatever the
model says, not that a real model's suggestions are good ones. **Still out
of scope:** a multi-candidate-per-round (beam search) strategy and a
parallel-coordinates-style multi-parameter plot.

## 2026-07-07 (midday) — agentic tuning loop, plus two documentation fixes

- **Agentic tuning loop** (top open P1 backlog item). The Assistant can now
  run predict → score-against-ground-truth → adjust → repeat on its own
  instead of a human clicking Diagnose/Apply/Evaluate one round at a time —
  new "Auto-tune" button next to the existing "Diagnose" one. Built as
  automation of that exact existing manual cycle (reusing
  `advisor.diagnose`'s rule-based suggestions and `benchmark.evaluate`'s
  AP/F1 scoring verbatim) rather than a literal LLM tool-calling loop: the
  acceptance criterion is "until **AP** plateaus", AP only means anything
  against ground truth, and a deterministic loop is fully unit-testable
  without Ollama in the picture. New `napari_app/core/tuning_loop.py`
  (`run_tuning_loop`, Qt/torch-free) stops on a plateau (`patience` rounds
  without a `min_delta` improvement), a step budget, the advisor running dry,
  or a repeated change (an oscillation guard) — every round keeps its own
  full parameter snapshot, not just the best one, so any step is one click
  ("Use these params") away, restoring parameters *and* re-running for real
  rather than just replaying a cached result. See `docs/BACKLOG.md` for the
  full writeup (touch points, all 56 new tests, what's not verified).
- **Fixed two stale dates in this changelog.** The previous two entries were
  headed "2026-07-08", a full calendar day ahead of `git log`'s actual
  commit dates (both really landed 2026-07-07, 00:18–03:22) — a session
  boundary that flipped the date this file used without anyone flipping the
  actual clock. Relabelled to "(late night)" rather than reordering anything
  (the entries were already in the right relative order — only the calendar
  label was wrong). Caught while sanity-checking the backlog against `git
  log` before picking this session's task, per this file's own house rule.
- **Fixed a second, unrelated documentation bug in `AGENTS.md`** while
  verifying this change against the light CI dependency group: the
  documented throwaway-venv check (`pip install --group test .`) has a
  trailing `.` that installs the whole project — pulling in
  `[project.dependencies]` (torch/napari/PyQt6, exactly what the check
  exists to exclude) on top of the `test` group, silently turning it back
  into a full-env check indistinguishable from the one it's meant to
  replace. This bit the very tests added in this change: two new
  `PredictController` tests only failed once run through the *real* light
  group (no `.`), for an unrelated reason (`cellpose_available()` is false
  without the real optional `cellpose` package) — fixed the same way an
  existing test already does, by monkeypatching it. Corrected the documented
  command to drop the `.` and noted why (`pytest.ini`'s `pythonpath = .`
  already makes the source tree importable with nothing installed).

## 2026-07-07 (late night) — scale bar always on + a canvas info caption

Direct user feedback on the previous entry: "why does the scale bar need a
toggle — just always show it", plus a request to think about what other
information the canvas overlay could usefully carry.

- **Scale bar has no toggle any more — always on.** Removed `scale_bar_cb`
  entirely; `viewer.scale_bar.visible = True` is set once at construction,
  no user action needed. A screenshot/exported figure should be self-
  contained by default, not depend on remembering to switch something on.
- **New one-line canvas caption** (napari's built-in `text_overlay`,
  top-left — the scale bar keeps its own default corner, bottom-right, so
  they don't collide), refreshed alongside every result and every "Colour
  cells by" change: cell count, the headline size stat (median diameter for
  2-D, mean volume for a z-stack), and calibration status (`0.25 µm/px` or
  `uncalibrated (px)` — a figure should never silently imply real units
  when none were actually set). When a colour-by-measurement is active, a
  second line names the metric and the range it spans, so a screenshot
  without the side panel in frame is still self-interpreting, e.g.:
  ```
  2 cells  ·  Ø 10.2 µm  ·  0.25 µm/px
  Coloured by Area (µm²): 56.2 – 112
  ```

5 new tests (326 -> 331) for the overlay's content across the uncalibrated/
calibrated and 2-D/volume/colour-by-active cases, plus one updated test
confirming the scale bar is on immediately at construction with no toggle
to click. Full suite green in the full conda env, stable across repeats.

**Not verified:** the actual rendered look on a real screen (no display in
this sandbox) — in particular whether the two overlays' corners visually
stay clear of each other as intended.

## 2026-07-07 (late night) — more viewer polish, researched and requested directly

Four more features researched against current napari/QuPath conventions and
implemented without asking first, per standing instruction. All in the
Predict tab's "Display" card (now always visible, not just while a result
with cells exists — see below) plus the Measurements window.

- **Real µm scale bar.** A "Show scale bar" toggle drives napari's own
  built-in `viewer.scale_bar` overlay. Backing it: `add_image`/`add_labels`
  now pass `scale=`/`units=` (µm/pixel from the existing calibration field)
  so the scale bar — and anything else reading world coordinates — reflects
  real units instead of raw pixels, only when the user has actually
  calibrated (pixel size at its "off" sentinel keeps everything at the
  previous 1-unit-per-pixel default). Found a real correctness risk while
  wiring this up: napari's own RGB auto-detection requires *both*
  non-channel axes to exceed 30px, so a small crop/thumbnail could silently
  be treated as a 3-plane grayscale stack instead of colour, corrupting the
  scale/units tuple length and crashing `add_image`. Fixed by passing `rgb=`
  explicitly (`_is_rgb_like`), based on shape alone, not size.
- **Colour-by-measurement legend.** The heatmap card added 2026-07-07 had no
  key — a gradient with no way to read a value off it. `analysis.
  measurement_range()` plus a small custom gradient-swatch widget
  (`_ColorLegend`) now show the min/max the current colour choice spans,
  mirroring QuPath's own measurement-map legend.
- **View in 3D.** A toggle (volume results only) flips `viewer.dims.
  ndisplay` between 2 and 3 — an actual rotatable 3D rendering of a
  segmented z-stack/time-lapse, not just a slice scrollbar. napari's Labels/
  Image layers are n-D natively, so this needed nothing beyond the toggle
  itself.
- **Click a cell in the table, see it highlighted on the image.** QuPath's
  own signature "linked views" behaviour. `MeasurementsWindow` gained a
  `row_selected` signal (emits the clicked row's cell_id, or -1 when the
  selection clears) — deliberately not napari-aware itself, so
  `PredictWidget` is the only side that knows about layers, toggling each
  result layer's existing `show_selected_label`/`selected_label` (a real,
  already-shipped napari Labels feature, not something new).

The "Display" card's own visibility is no longer tied to having cells to
colour by — the scale bar and 3D toggles are meaningful even then, so only
the "Colour cells by" combo itself disables in that case.

Caught (and fixed) one real cross-test-contamination bug while writing the
linked-selection test: exercising `_open_measurements()`'s real
`get_measurements_window()` singleton left its underlying Qt object
corrupted for whichever test ran next in the same process — the same class
of shared-singleton fragility already documented against `_recompute_
measurements` in `tests/test_predict_labels_display_wiring.py`, reached
here by a different path that fixture's stub didn't cover. Fixed by
monkeypatching a throwaway `MeasurementsWindow` in for that one test rather
than touching the real singleton.

23 new tests (303 -> 326): `analysis.measurement_range`, `_is_rgb_like`/
`_layer_scale_kwargs`, the scale-bar/legend/3D-toggle/linked-selection
wiring, and — a first for that file — `MeasurementsWindow`'s own
`row_selected` signal. Full suite green in the full conda env (326 passed,
stable across repeats) and in a venv matching CI's exact `pip install
--group test` (251 passed, 7 skipped for PyQt6-gated files).

**Not verified:** the actual rendered look/feel on a real screen (no display
in this sandbox) — in particular, whether napari's scale bar and 3D view
render exactly as expected is inferred from its own API/docs, not seen.

## 2026-07-07 (yet even later) — requested features, not backlog items

- **Colour cells by a measurement** (a "Display" card in the Predict tab).
  A researched-and-requested feature: QuPath/CellProfiler both let you
  colour detections by a computed value (a population heatmap) instead of
  only by identity. New `analysis.label_colormap_from_measurement()` maps a
  chosen column (area, circularity, intensity, ...) through matplotlib's
  viridis colormap (perceptually-uniform, colourblind-safe — the modern
  default that replaced jet for exactly this reason), min-max normalised
  across the current population; the widget applies it to both the fill and
  outline layers via napari's `DirectLabelColormap`, and remembers each
  layer's original colormap (in `.metadata`) to restore instance-ID
  colouring on demand. Works for 2-D and volume (z-stack) results alike,
  independent of the Results card's 2-D-only hero chips. Surfaced a real gap
  while wiring it up: `_apply_color_by` is called unconditionally after every
  prediction, and an older test fixture's plain-list stand-in for
  `viewer.layers` doesn't support name-based lookup the way a real
  `LayerList` does — broadened the lookup's exception handling
  (`KeyError`/`TypeError`) so it degrades to a no-op regardless, instead of
  only working against real napari. matplotlib added to the pure-logic
  `test` dependency-group (pyproject.toml) — the same class of gap as the
  nibabel incident: it's only an optional extra of scikit-image/scipy/
  nibabel, never installed by `pip install --group test` on its own.

- **Interactive box-prompt segmentation** (Annotate tab). Also
  researched-and-requested: napari-segment-anything and similar tools pair
  a Points layer (click prompts) with a Shapes layer (box prompts) — SAM's
  own `SamPredictor.predict()` already accepts a `box=` argument natively,
  so `InteractiveSession.predict()` just needed to thread it through
  (`multimask_output=False` for a box, per SAM's own docs — a box is a far
  less ambiguous prompt than a single click). The Annotate tab now creates a
  second layer, `*_box_prompt` (a Shapes layer, pre-armed with the rectangle
  tool), alongside its existing click-to-segment Points layer: drawing a
  rectangle segments everything inside it as a new object, and a follow-up
  shift/⌘-click still refines it same as a point-started cell (the box's
  low-res mask carries forward as `_last_low` exactly like a click's does).
  The existing point-click callback steps aside while the Shapes layer is
  the active one, so drawing a box is never misread as an accidental pan-
  triggered click. Caught two real bugs before shipping: clearing the
  Shapes layer after each box re-entered the same data-changed handler
  recursively (napari hadn't finished its own bookkeeping by the time the
  re-entrant call read `.data` back, so the "already empty" guard never
  tripped) — fixed by disconnecting the handler for the duration of the
  clear; and the box-to-pixel-bounds clipping let a box drawn entirely
  outside the image through as a valid tiny one (an asymmetric clip range
  meant to avoid a zero-width box instead let a fully-out-of-bounds box
  clip to a valid-looking corner) — fixed with symmetric clipping plus the
  existing degeneracy check.

  26 new tests across both features (analysis.py's colormap function,
  predict_widget.py's card/colouring wiring, `_rect_to_box_xyxy`'s pure
  coordinate math, and `annotate_widget.py`'s full box-to-paint round trip —
  the latter a first for this file, which had zero test coverage before).
  **Not verified:** the actual rendered look/feel on a real screen (no
  display in this sandbox).

## 2026-07-07 (even later) — requested UI polish, not a backlog item

- **Predicted/ground-truth masks now show a translucent fill under the
  outline**, instead of an outline with nothing behind it — the user
  reported outlines sometimes disappearing against a busy image and asked
  for the "filled + bordered, same colour" look common tools (QuPath's
  "Fill detections", CellProfiler's OverlayOutlines) already default to.
  napari's own Labels layer can't blend a translucent fill and a contour in
  one layer — `contour` is a 0/N *toggle*, not additive (confirmed against
  napari's own docs) — so the fix is the standard two-layer overlay: a new
  `PredictWidget._add_filled_labels` adds a low-opacity filled layer
  *underneath* the existing contour=1 outline layer, both showing the same
  label data, so a cell's fill and border always match (napari's per-label
  colour is a deterministic function of the label id, confirmed with
  `get_color()`). Applied everywhere a result mask is shown — `_show_results`,
  `_show_volume_results`, and the ground-truth overlay (which keeps its
  existing uniform-green colouring, now on both layers) — but not
  Annotate's interactive paint layer, a live editing canvas with different
  tradeoffs left for a separate look if wanted. 7 new tests exercising real
  `napari.layers.Labels`/`Image` construction (not just recorded mock
  calls): layer names/contour/opacity, matching fill/outline colour per
  cell, and no layer accumulation on repeated runs.

## 2026-07-07 (later) — user-reported bug, not a backlog item

- **Fixed the Predict panel forcing horizontal scroll when SAM2 is
  selected.** `QComboBox`/`QCheckBox` text never wraps (unlike `QLabel`), so
  a widget's on-screen width is set by its widest item's/label's text —
  three strings added for SAM2 were meaningfully longer than every sibling
  they sit next to: the engine combo's SAM2 label (51 chars vs. 37-39 for
  the other two), the "segment as z-stack" checkbox (57 chars vs. 39-44 for
  the other checkboxes), and the "propagate" tracking-mode combo item (43
  chars). Shortened all three to the same range as their siblings (moving
  the dropped detail into tooltips, which were already there); confirmed
  with `QFontMetrics` that each is now narrower than its own reference
  sibling, not just "shorter than before."

## 2026-07-07

- **SAM 2 / z-stack follow-up: closed every gap the previous entry left
  open.** Requested directly rather than picked from the backlog, after
  landing the SAM2 engine — four pieces of real, additional work:
  - `_predict_volume` now composes z-stacks with tiling (a plane large
    enough that `should_tile` recommends it goes through `_predict_tiled`
    instead of always shrinking to `resize_size`).
  - `read_volume_stack`/`has_z_stack` now keep the real Z/T axis for
    ND2/CZI/LIF too, not just TIFF/OME-TIFF (`_nd2_raw`/`_czi_raw` share
    their array+axes extraction between the channel-only and
    volume-keeping read paths; LIF falls back to its existing
    channel-only behaviour if its per-plane API doesn't look like
    expected, since there's no real `.lif` file anywhere to confirm the
    guess against).
  - `analysis.compute_measurements` now dispatches on `mask.ndim` to a
    real 3-D schema (volume, 3-D centroid, equivalent diameter, ...) —
    2-D-only regionprops properties with no 3-D equivalent (perimeter,
    circularity, eccentricity, orientation) are correctly absent rather
    than faked, and `_show_volume_results` now populates real
    measurements so "Open Measurements"/"Export CSV" work on a volume
    result exactly like the 2-D path.
  - A second, opt-in SAM2 tracking mode — **propagate** — seeds objects
    with the automatic mask generator on the first plane, then tracks
    each one across the rest of the stack with SAM2's actual video
    predictor (memory-bank propagation) instead of independent-per-plane
    detection + IoU stitching. The single most speculative piece here:
    the video predictor's exact API is this module's best-effort reading
    of the public interface, not confirmed against a real install.

  107 new tests total across both SAM2 passes (163 pre-existing → 270).
  **Not verified anywhere in this work:** actual SAM2 inference (either
  tracking mode), the guessed checkpoint/Hydra-config names, the video
  predictor's exact method signatures, real ND2/CZI/LIF files (fake-module
  round-trips only), the widget on a real screen. Full writeup in
  `docs/BACKLOG.md`.

## 2026-07-06 (night)

- **SAM 2 engine for z-stacks / time-lapse** (top open P1 backlog item —
  previously the single largest ML-pipeline gap per `docs/AUDIT_2026.md`).
  New `napari_app/engines_sam2.py` registers SAM2 as a third `EngineSpec`,
  lazy-imported and `available()`-gated exactly like Cellpose, so the app
  (and CI) are unaffected without the optional `sam2` package/checkpoint
  installed (`pip install -e ".[sam2]"`). The z-stack/time-lapse capability
  itself is a new engine-agnostic layer, not SAM2-specific: new
  `napari_app/volume_stitch.py` links independently-segmented 2-D planes
  into one consistent instance volume by adjacent-slice IoU (the same idea
  Cellpose's own `stitch3D` uses — fully pure-logic, unit-tested without any
  GPU); `napari_app/channels.py` gained a `VolumeStack`/`read_volume_stack`/
  `has_z_stack` path that keeps a TIFF/OME-TIFF's Z/T axis instead of always
  reducing it to the first plane; `predict_controller.py`'s new
  `_predict_volume`/`run_volume_prediction_async` read a stack, run *any*
  registered engine per-plane, and stitch — so this also works with Cellpose
  or CellSeg1, not only SAM2 (which remains the flagship choice, being the
  only one of the three actually trained for video/volumetric consistency).
  The Predict tab gained an off-by-default "Segment as z-stack" checkbox
  (shown only for a file that genuinely has one) and a SAM2 settings card,
  both following the existing `tiled`-toggle pattern; napari's Image/Labels
  layers are n-D natively, so results just add the volume arrays directly —
  3-D per-cell measurements are explicitly not wired up yet (`analysis.py`
  is 2-D-only), so the results card is hidden rather than showing stale
  numbers. 66 new tests (163 → 229), including a `PredictWidget`
  constructed under offscreen Qt with a mocked napari viewer (a real
  `napari.Viewer()` segfaults in this sandbox) — a first for this repo's
  test suite, and it caught a real bug pre-commit: a test file's own import
  order could flip which engine registers first and silently change the
  Predict tab's default engine. **Not verified:** any real SAM2 inference
  (no GPU/package/checkpoint here), the guessed checkpoint/Hydra-config
  filenames (overridable in the UI), or the widget on an actual screen.
  **Deliberately out of scope:** SAM2's video-predictor propagation mode
  (stronger but a different, prompt-driven workflow), ND2/CZI/LIF volumes,
  and z-stack + tiling composed together. See `docs/BACKLOG.md` for the full
  writeup.

## 2026-07-06 (evening)

- **fp16 + `torch.compile` inference** (top open P1 backlog item). Two
  off-by-default checkboxes in the Predict tab's Model settings card —
  half precision (CUDA autocast fp16) and mask-decoder `torch.compile` —
  gated CUDA-only by new `use_amp()`/`use_compile()` predicates in
  `inference_cache.py`, so they're a proven no-op on CPU/MPS rather than an
  unverifiable behaviour change. `torch.compile` failures fall back to eager
  silently; toggling it is folded into the model-cache key so it forces a
  reload+recompile only where it actually matters (a CUDA device). Scoped to
  `inference_cache.py` (the app's one prediction choke point) after
  confirming `predict.py`'s `predict_images`/`predict_config` — the
  originally-scoped touch point — have no live caller anywhere in the app.
  11 new tests (163 total), green in the full conda env and in a throwaway
  venv with only the declared `test` dependency-group installed. **Not
  verified:** the actual CUDA speedup — no CUDA in this sandbox (MPS only)
  to benchmark on.

## 2026-07-06 (later still) — user-reported bugs, not backlog items

- **Fixed a Refine crash (`KeyError: 'deterministic'`)** — `_run_refine` built
  its config through the engine-selector-dependent `_build_config()`, so
  refining a LoRA checkpoint while the Predict tab's engine selector was on
  Cellpose-SAM produced Cellpose's short config, missing every SAM/LoRA
  training key. Refine always trains via `cellseg1_train.py` regardless of
  the selector, so it now calls `_sam_config()` directly, same as the
  interactive Annotate session already did.
- **Fixed a duplicated download icon** on "Load sample microscopy images" —
  the button already had a proper `QIcon`; the post-fetch label text also
  prepended its own "⬇" emoji on top of it.
- **Fixed leftover native step-button chrome on the µm/pixel calibration
  field** — it was the one spinbox in `predict_widget.py` missing
  `setButtonSymbols(NoButtons)`, which every sibling spinbox already sets.
- **Clarified the "Sample" switcher and GT autofill** — not a bug: GT autofill
  only fires when a matching `_gt`/`_mask` sidecar file actually exists, and
  of the 6 bundled quick samples only the synthetic `sample_phantom.png` ships
  with one (real ground truth for the others requires "Download BBBC039...").
  The "Sample" dropdown now labels entries "· has GT" so this is visible
  where the user is actually looking, instead of only in the GT card's status
  line further down.

---

## 2026-07-06 (yet even later)

- **Warn before silently downsampling a large image** (P1 backlog item).
  `tiling.should_warn_no_tiling()` flags exactly the case the backlog
  described: an image large enough that "Large image" tiling would help, but
  the toggle is off. `predict_controller.run_prediction_async` now emits a
  `[HINT]` log line for that case through its existing `on_log` callback —
  already wired to the widget's log window, so no widget changes were needed.
  Scoped to the single-prediction path (not batch/benchmark). 4 new tests;
  148 pre-existing tests unmodified (152 total).

---

## 2026-07-06 (even later)

- **Engine registry** (top open P1 backlog item). New `napari_app/engine_registry.py`
  (`EngineSpec` + `register`/`get`/`all_engines`) replaces the hardcoded
  `if engine == "cellpose": ... else: ...` dispatch in `predict_controller.py`
  and the two hardcoded entries in the Predict tab's engine combo + benchmark
  checklist. Adding a future engine (StarDist/InstanSeg/Micro-SAM/DeepCell/the
  still-open SAM2 item) is now one `register()` call instead of edits across
  three files. Config-building and per-engine settings UI stay bespoke per
  engine deliberately — see `docs/BACKLOG.md` for the exact boundary. 15 new
  tests; 133 pre-existing tests unmodified (148 total).

---

## 2026-07-06 (later)

- **Added `CLAUDE.md`** — a one-line file that imports `AGENTS.md` via `@AGENTS.md`.
  Researched how Claude Code actually loads project instructions: it reads
  `CLAUDE.md` at session start, not `AGENTS.md` — confirmed directly, since
  this repo's own `AGENTS.md` was never auto-loaded before this, only read
  because the kickoff prompt explicitly said to. This closes that gap for
  good without giving up `AGENTS.md`'s cross-tool portability.
- **Renamed `streamlit_storage/` → `data_store/`** (last open P0 backlog
  item). The dir is gitignored and only ever created locally by
  `setup_napari.sh`, so this was a path-string rename, not a file move:
  `project_root.py`'s `STORAGE_DIR` constant, `setup_napari.sh`, and
  `.gitignore` updated; every other reference already derived from
  `STORAGE_DIR` so needed no change. No behaviour change.

---

## 2026-07-06

- **`predict_widget.py` god-object split.** New `napari_app/core/predict_controller.py`
  — a Qt-free `PredictController` owns config-building and predict/batch/
  benchmark orchestration; the widget just wires UI to it. 22 new tests,
  behaviour unchanged. (PR #5)
- **CI fix:** added `nibabel` to the pure-logic test dependency group —
  `data/utils.py` needs it and CI didn't have it, which the fuller test
  coverage above finally exposed.
- **Docs overhaul** (this change): `AUDIT_2026.md` moved to `docs/`, annotated
  with dated addenda instead of silently rewritten; this changelog added;
  `docs/BACKLOG.md` reconciled against the audit and against undocumented
  work; `AGENTS.md` updated with the auto-merge git workflow and a
  before-you-start reality-check step; `docs/AGENT_KICKOFF_PROMPT.md` and a
  root `README.md` added.

## 2026-07-05

- **The "Lab" design system — a full, unplanned UI redesign** (~17 commits,
  `dd4596c`..`cd6b283`). Not tracked as a backlog task at the time; this is
  the retroactive record:
  - Navigation: top tabs → a permanent icon-only left rail (`widgets/shell.py`).
  - Component system v2: `SectionCard`/`CollapsibleCard`/`CollapsibleSection`
    (`widgets/common.py`), a custom `Combo` dropdown (`widgets/controls.py`).
  - Assistant rebuilt as a real chat surface (`widgets/chat.py`): message
    bubbles, streaming.
  - New `icons.py` (icon set) and `motion.py` (micro-animations — count-up
    counters, status-dot pulse).
  - Predict panel: hero cell-count KPI, stat chips, 2×2 result-action grid.
  - Assessment: `docs/AUDIT_2026.md` §4.4.
- **Microscopy formats**: OME-TIFF/ND2/CZI/LIF readers + auto-filled µm/pixel
  from metadata (`napari_app/channels.py`). (PR #3)
- **Packaging**: real `pyproject.toml`, `pip install -e .`, `cellseg1`
  console script, napari plugin manifest, pinned `requirements.txt`. (PR #4)
- **Real multi-channel support**: channel picker + per-channel percentile
  normalisation, replacing the old collapse-to-RGB read path.
- fix(predict): coerce 16-bit / float images to uint8 before handing them to SAM.

## 2026-07-04

- **`docs/AUDIT_2026.md` written** — the due-diligence audit this changelog
  now keeps honest.
- **Streamlit GUI removed**; shared logic moved into `napari_app/core/`.
  `AGENTS.md` and `docs/BACKLOG.md` added as the agent orientation + task
  queue.
- **Test + CI foundation**: first pytest suite (analysis/benchmark/cohort/
  advisor/tiling), GitHub Actions matrix on py3.11/3.12.
- **Tiled inference** (`napari_app/tiling.py`): native-resolution tiling with
  overlap + instance stitching for large images; wired into Predict behind an
  opt-in "Large image" toggle; per-tile progress in the UI. (PR #1)
- Cellpose-SAM zero-shot engine added alongside CellSeg1/LoRA; real sample
  data; local Assistant (heuristic diagnostics + optional Ollama chat).

## Earlier (2024-12-01 – 2026-07-03)

Project origin through the initial napari desktop app: SAM+LoRA one-shot
training pipeline (`cellseg1_train.py`, `peft/`), the original Streamlit GUI
(later removed, see above), then the napari `PredictWidget`/`TrainWidget`
rewrite, ground-truth evaluation, cohort/batch analysis, engine benchmarking,
and several rounds of UI polish predating the "Lab" design system. See
`git log` for the itemized history — not reconstructed here since it predates
this file and the backlog/audit process it supports.
