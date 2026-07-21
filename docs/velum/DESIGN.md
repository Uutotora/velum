# Design language — Velum

Derived from the Label Studio design system, retuned for a scientific
microscopy tool. The authoritative values live in
`studio/theme.py`; this doc explains the intent so choices stay
coherent as new UI is added.

## Concept — "the bench & the scope"

Two grounds, both first-class:

- a warm-neutral, airy **light "bench"** for browsing and configuring;
- a deep near-black **dark "scope"** that also honours the existing "Lab"
  instrument look and is where the image lives.

The viewer's theme toggle swaps the whole token set at runtime.

## Colour

Three roles, never blurred:

| Role | Light | Dark | Use |
|------|-------|------|-----|
| **Primary** — iris indigo (interactive only) | `#4f5bd5` | `#7d8bf0` | buttons, links, focus — "action available". Never decorative. |
| **Signal** — fluor teal (active / selected / detected) | `#0fa8a0` | `#2bd4c0` | active nav, segmentation outlines, "on" toggles. Never stands in for primary. |
| **Status** (separate family) | success `#1f9d6b` · warning `#c9821f` · danger `#d9524c` | success `#33c98c` · warning `#e0a63b` · danger `#e0655f` | outcomes only. |

Neutrals are a **chosen** cool graphite (not pure grey): text `#14161a` /
`#e9ecf1`, muted `#868d98` / `#6c7480`, borders `#e4e7ec` / `#262b34`,
surfaces white/`#15181e`. The image viewport ground is `#0a0c10` in both
themes.

Categorical/label palette (`theme.VIZ`, `demo.LABEL_COLORS`): iris, teal,
kiwi, mango, persimmon, fig … a large set so many instance labels stay
distinct. Heatmap ramp = **viridis** (`theme.viridis_rgb`), used by
"Colour cells by".

## Type

- **Figtree** for all UI (bundled at `studio/fonts/`, SIL OFL,
  registered at startup). Chain falls back to `-apple-system`.
- A mono (`SF Mono` → fallbacks) for data, IDs, metrics, logs.
- Scale in use: 26 (page title) · 19 (wordmark) · 16 · 15 · 14.5 · 13.5 · 13 ·
  12.5 · 12 · 11.5 · 11 · 10.5 · 10. Weights 400/500/600. Uppercase micro-labels
  get `letter-spacing`.

## Form

- Radii: 7 (controls) · 10 · 14 (cards) · 18; window corner 12px.
- Shadows: soft, blue-tinted (`rgba(28,42,120,α)`), low alpha — elevation, not drama.
- Spacing rhythm: 2 · 4 · 8 · 14 · 16 · 24 · 34 (page gutter).
- Motion: soft fades on navigation; hover elevation; nothing loud. Respect
  reduced-motion.

## Window chrome

Frameless + rounded, our own dark title bar (own traffic lights with
hover glyphs, centred title, theme toggle). Native move via
`startSystemMove`, resize via corner `QSizeGrip`s, shape via a rounded mask.
No native grey OS title bar.

## Component vocabulary (in `components.py`)

Chip · Badge · PillButton (primary/ghost/success/danger) · IconButton ·
SelectBox · Toggle · Slider · Stepper · SegControl · StatTile · FieldRow ·
GroupLabel · Accordion · Sidebar · SmoothScrollArea (an eased-wheel-step
QScrollArea — every screen's shared `scroll()`/`_scroll()` helper builds one
of these now, not a bare QScrollArea). Every one takes a token dict and
works in both themes. Add new atoms here, not ad hoc in screens.

## Rules for new UI

1. Style through **tokens**, never hard-coded hex (except the fixed scope/canvas darks).
2. Reuse a **component** before inventing a widget; if you invent one, it's a
   reusable atom in `components.py`.
3. Primary hue = interactive only. Signal = state. Status = outcomes. Don't mix.
4. Both themes must look deliberate — check light *and* dark.
