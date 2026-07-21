# Packaging Velum as a macOS `.app`

Today Studio runs **unbundled**: `run_studio.sh` launches the `python3.11`
interpreter, so the Dock tile is *python*'s window icon (drawn by Qt as-is, no
macOS icon treatment — that's why the Dock icon size was hard to get right).
Turning it into a real `.app` fixes that and gives it a proper name, icon, and
double-click launch.

There are **two** ways to package, for two different goals. Pick by what you're
doing right now.

---

## Mode 1 — Dev launcher `.app` (recommended while you're building)

A **thin `.app`** that contains only an icon + a launch script; the actual
Python code still lives in your git checkout and runs from there. This is the
mode that answers *"how do I keep adding features without rebuilding/re-
downloading the app every time?"* — because the `.app` never contains your
code, **editing the code and relaunching is the whole update**. You build the
`.app` **once**.

```
Velum.app/
└── Contents/
    ├── Info.plist          # name, bundle id, icon, min-OS, retina flag
    ├── MacOS/launch        # tiny shell script -> runs studio/app.py from the repo
    └── Resources/AppIcon.icns
```

**Update workflow (no rebuild):**

1. Edit code in the repo (vibe-code it — see the workflow section below).
2. Quit and relaunch the app from the Dock. Done — it's running your new code.

You only rebuild the `.app` if you change the **icon**, the app **name**, or the
**launcher** itself — none of which happens when you add features.

### The AI-agent prompt to build it

Paste this to an agent (Claude Code / Codex / Cursor) in the repo root:

> Build a **thin macOS `.app` launcher** for Velum. Do NOT freeze or
> bundle the Python code — the `.app` must run the live source from this git
> checkout so I can edit code and just relaunch, never rebuild.
>
> Requirements:
> - Add a script `scripts/make_app.sh` that (re)creates `dist/Velum.app`
>   with this layout: `Contents/Info.plist`, `Contents/MacOS/launch` (executable),
>   `Contents/Resources/AppIcon.icns`.
> - `Contents/MacOS/launch` is a bash script that launches the app **without
>   relying on an interactive shell's PATH** (a `.app` starts with a bare
>   environment — `conda`/`python` will NOT be on PATH). Resolve the interpreter
>   robustly in this order: `$CELLSEG1_PYTHON` if set; else the conda env python
>   at `/opt/homebrew/Caskroom/miniforge/base/envs/cellseg1/bin/python` if it
>   exists; else `conda run -n cellseg1 python`. Then `exec` it on the repo's
>   `studio/app.py` with `PYTHONPATH` set to the repo root. Hard-code the repo
>   path via a placeholder the script computes from its own location if the app
>   is kept inside the repo, otherwise from an absolute path constant near the
>   top of `make_app.sh`. Redirect stdout/stderr to `~/Library/Logs/Velum.log`.
> - `Info.plist` keys: `CFBundleName`/`CFBundleDisplayName` = "Velum",
>   `CFBundleIdentifier` = "com.velum.app", `CFBundleExecutable` = "launch",
>   `CFBundleIconFile` = "AppIcon", `CFBundlePackageType` = "APPL",
>   `LSMinimumSystemVersion` = "13.0", `NSHighResolutionCapable` = true,
>   `LSUIElement` = false.
> - Copy `docs/app_icon/AppIcon.icns` to `Contents/Resources/AppIcon.icns`.
> - After building, run `codesign --force --deep --sign - "dist/Velum.app"`
>   (ad-hoc sign) so Gatekeeper/Dock behave, and print how to launch it.
> - Verify it opens (`open "dist/Velum.app"`), confirm the log file
>   gets written, and tell me exactly what you did and did NOT verify.
> - Add a short `docs/velum/PACKAGING.md` section (or update it) with how to
>   rebuild and where the log lives. Commit on a branch.
>
> Note: the app currently needs a real display + SAM weights to fully run; you
> can't verify the GUI headlessly — just verify the bundle is well-formed and
> the launcher resolves the interpreter.

### About the icon on a real `.app`

- A plain `.icns` (what we have) is drawn **as-is** — macOS does *not* add the
  Tahoe "Liquid Glass" squircle/margin to a legacy `.icns`. So with `.icns`,
  ship a **pre-padded** icon (the macOS grid is ~**0.875** of the canvas —
  measured from system icons like App Store/Notes). Regenerate a padded icns:
  scale `Default-1024` to 0.875 centered on a transparent 1024² canvas, rebuild
  the iconset, `iconutil -c icns`.
- For the **full Tahoe glass** look, ship Apple **Icon Composer**'s `.icon`
  bundle instead and compile it with Xcode 26 / `actool` — then macOS applies
  the margin and material automatically and you don't hand-pad anything.
- We keep `studio/assets/icon.png` full-bleed for now on purpose (it's only the
  unbundled python window icon); decide the above when you build the `.app`.

---

## Mode 2 — Self-contained distributable (for shipping to other people)

When you want to hand the app to someone who does **not** have the repo/conda
env, freeze everything into a bundle with **PyInstaller**. This is wired up:

**Locally** (build on the OS you're targeting — you can't cross-build):

```bash
pip install -e . pyinstaller
bash scripts/build_bundle.sh
#   macOS  -> dist/Velum.app  +  dist/Velum.dmg
#   Linux  -> dist/Velum/     +  dist/Velum-linux-x86_64.tar.gz
```

**In CI / Releases** — [`.github/workflows/release.yml`](../../.github/workflows/release.yml)
builds both on a version tag and attaches them to the GitHub Release:

```bash
git tag v0.1.0 && git push origin v0.1.0
# -> Actions builds macOS .dmg (macos runner) + Linux .tar.gz (ubuntu runner),
#    uploads both to the v0.1.0 Release. Or run it manually from the Actions tab.
```

What's bundled: Python + torch + PyQt6 + the app + small LoRA checkpoints
(`checkpoints/`). What's **not**: the ~2.5 GB SAM ViT-H backbone — it downloads
on first use into the app's data store, keeping the artifact to ~1.5–2.5 GB
(torch dominates). Every code change needs a rebuild, so this is a *release*
step, not your daily loop (that's Mode 1).

**Honest caveats** (matters, since this can't be verified without running the
actual build on each OS):
- **PyInstaller + torch almost always needs a shake-down run or two** to catch a
  missing hidden import for a specific torch/OS combo. Expect to iterate on the
  `--collect-*` flags in `scripts/build_bundle.sh` the first time; that's normal,
  not a sign it's broken.
- The macOS `.dmg` is **ad-hoc signed**, so first launch needs right-click ▸ Open
  (or `xattr -dr com.apple.quarantine`) — a real Developer ID + notarization is a
  separate step for a "just double-click" download.
- Auto-update without re-downloading the whole app: add **Sparkle** (macOS) —
  real work, only worth it once you actually distribute regularly.

**Is it too early?** No — the app is functional (real segmentation, green tests),
so packaging works. The two real costs are size (torch) and that each OS's bundle
must be built on that OS (the CI matrix handles this). Tag a release and let CI
produce the first artifacts, then iterate.

---

## The vibe-coding update loop (Mode 1)

Goal: add features by describing them, without touching the packaging.

1. **One-time:** build the dev-launcher `.app` (Mode 1). Keep it in the Dock.
2. **Each feature:** describe it to your coding agent working in this repo. It
   edits files under `studio/` (UI/logic) — never the `.app`. Studio is
   structured so screens wire to plain, unit-tested controllers
   (`studio/*_controller.py`), so most features are "add a control + a
   controller method + a test."
3. **See it:** quit the app, relaunch from the Dock. New code, same `.app`.
   (Or `bash run_studio.sh` from a terminal for a faster edit-run loop with
   logs in the foreground.)
4. **Keep it green:** `<conda-python> -m pytest studio/tests` before you commit
   — the suite is offscreen/no-GPU and fast. The agent should add a test for
   new pure logic (see `AGENTS.md`).
5. **Verify UI offscreen:** the GUI can't be driven headlessly, but a change can
   be rendered offscreen (`QT_QPA_PLATFORM=offscreen`, `widget.grab().save(...)`)
   — ask the agent to screenshot the change and show it, don't trust tests alone.
6. **Commit** on the studio branch; push.

You never re-package for steps 2–6. The `.app` is just a launcher pointed at
this code.
