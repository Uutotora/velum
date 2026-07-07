"""CellSeg1 Studio — the design language (tokens + Qt stylesheet builders).

The Studio identity, derived from the Label Studio design system and retuned
for a microscopy desktop tool:

- **Two grounds.** A warm-neutral, airy *light* "bench" for browsing and
  configuring; a deep near-black *dark* "scope" that also honours the existing
  "Lab" instrument look. Both are first-class — the viewer's theme toggle
  swaps the whole token set at runtime.
- **One interactive hue** — *iris indigo* — reserved strictly for actionable
  elements (distinct from the old electric blue and from Label Studio grape).
- **One signal hue** — *fluor teal* — for the active/selected/"detected"
  state (segmentation outlines, active nav), never standing in for the accent.
- **Status** (success / warning / danger) is a separate family, never the
  accent.
- Type: **Figtree** for UI (loaded at runtime by the app), a mono for data.

Pure strings — no Qt import here, so the module stays importable under the
light CI ``test`` group and the token values are unit-testable. The app calls
:func:`build_qss` with :data:`LIGHT` or :data:`DARK` and applies the result.
"""
from __future__ import annotations

from typing import Any

# Font families. Figtree is bundled and registered with QFontDatabase at
# startup (see the app); the CSS-style fallback chain keeps things sane if the
# face fails to load. The mono is macOS-native first.
SANS = '"Figtree", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
MONO = '"SF Mono", "JetBrains Mono", ui-monospace, Menlo, monospace'

# Radii (px) — small controls → cards.
R_SM, R_MD, R_LG, R_XL = 7, 10, 14, 18


# ── Token palettes ───────────────────────────────────────────────────────────
# Every semantic token exists in both palettes with the *same key*, so widgets
# style against token names and never branch on theme. (A test enforces the
# key sets are identical.)

LIGHT: dict[str, str] = {
    "bg":            "#f4f6f8",   # app canvas behind panels
    "surface":       "#ffffff",   # card / panel surface
    "surface2":      "#eceff3",   # hover / elevated fill
    "inset":         "#f6f7f9",   # recessed field "well"
    "border":        "#e4e7ec",   # hairline border
    "border_strong": "#d4d9e0",   # emphasised border / handle

    "text":          "#14161a",   # primary ink
    "text_subtle":   "#4b515b",   # secondary ink
    "text_muted":    "#868d98",   # hints / labels / timestamps

    "primary":       "#4f5bd5",   # iris — interactive only
    "primary_hover": "#5a67e6",
    "primary_press": "#3f49bf",
    "primary_weak":  "rgba(79,91,213,0.10)",
    "primary_line":  "rgba(79,91,213,0.28)",

    "signal":        "#0fa8a0",   # fluor teal — active / selected / detected
    "signal_weak":   "rgba(15,168,160,0.14)",
    "signal_line":   "rgba(15,168,160,0.40)",

    "success":       "#1f9d6b",
    "warning":       "#c9821f",
    "danger":        "#d9524c",
    "success_weak":  "rgba(31,157,107,0.12)",
    "warning_weak":  "rgba(201,130,31,0.13)",
    "danger_weak":   "rgba(217,82,76,0.12)",

    "scope":         "#0a0c10",   # image viewport ground (dark in both themes)
}

DARK: dict[str, str] = {
    "bg":            "#0d0f13",
    "surface":       "#15181e",
    "surface2":      "#1c2027",
    "inset":         "#101318",
    "border":        "#262b34",
    "border_strong": "#333a45",

    "text":          "#e9ecf1",
    "text_subtle":   "#aab2be",
    "text_muted":    "#6c7480",

    "primary":       "#7d8bf0",
    "primary_hover": "#93a0f5",
    "primary_press": "#6774e0",
    "primary_weak":  "rgba(125,139,240,0.14)",
    "primary_line":  "rgba(125,139,240,0.34)",

    "signal":        "#2bd4c0",
    "signal_weak":   "rgba(43,212,192,0.16)",
    "signal_line":   "rgba(43,212,192,0.50)",

    "success":       "#33c98c",
    "warning":       "#e0a63b",
    "danger":        "#e0655f",
    "success_weak":  "rgba(51,201,140,0.14)",
    "warning_weak":  "rgba(224,166,59,0.14)",
    "danger_weak":   "rgba(224,101,95,0.14)",

    "scope":         "#07090c",
}

# Categorical palette for labels / data-viz (order-stable; distinct on both
# grounds). Mirrors the mockup's iris/teal/kiwi/mango/persimmon/fig set.
VIZ = ["#6d87f1", "#2bd4c0", "#6fae53", "#e0982f", "#ee6a52", "#a878cf"]

# viridis control points (RGB) for the "Colour cells by" heatmap ramp.
VIRIDIS = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]

THEMES = {"light": LIGHT, "dark": DARK}


def tokens_for(theme: str) -> dict[str, str]:
    """Return the token dict for ``"light"`` or ``"dark"`` (defaults to dark)."""
    return THEMES.get(theme, DARK)


def viridis_rgb(t: float) -> tuple[int, int, int]:
    """Sample the viridis ramp at ``t`` in [0, 1] → an (r, g, b) tuple.

    Used for per-cell heatmap colouring when "Colour cells by" is a
    morphometry rather than instance id.
    """
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    seg = t * (len(VIRIDIS) - 1)
    i = int(seg)
    if i >= len(VIRIDIS) - 1:
        return VIRIDIS[-1]
    a, b = VIRIDIS[i], VIRIDIS[i + 1]
    k = seg - i
    return (round(a[0] + (b[0] - a[0]) * k),
            round(a[1] + (b[1] - a[1]) * k),
            round(a[2] + (b[2] - a[2]) * k))


def build_qss(t: dict[str, Any]) -> str:
    """Build the root Qt stylesheet for a token palette ``t``.

    Covers the base widget vocabulary (labels, inputs, combos, spin boxes,
    checkboxes, scrollbars, tooltips, menus, progress) so any child widget
    inherits the Studio look without per-widget styling. Component-specific
    chrome (sidebar, cards, layer rows) is layered on top by those widgets.
    """
    return f"""
QWidget {{
    background: {t['bg']};
    color: {t['text']};
    font-family: {SANS};
    font-size: 13px;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: {t['bg']}; border: none; }}

QLabel {{ background: transparent; color: {t['text']}; }}

QToolTip {{
    background: {t['surface2']}; color: {t['text']};
    border: 1px solid {t['border_strong']}; border-radius: {R_SM}px;
    padding: 6px 9px; font-size: 12px;
}}

QLineEdit, QPlainTextEdit {{
    background: {t['inset']}; border: 1px solid {t['border']};
    border-radius: {R_SM}px; padding: 7px 11px; color: {t['text']};
    selection-background-color: {t['primary']}; selection-color: #ffffff;
    min-height: 30px;
}}
QLineEdit:hover {{ border-color: {t['border_strong']}; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {t['primary']}; }}

QComboBox {{
    background: {t['inset']}; border: 1px solid {t['border']};
    border-radius: {R_SM}px; padding: 7px 12px; color: {t['text']}; min-height: 30px;
}}
QComboBox:hover {{ border-color: {t['border_strong']}; }}
QComboBox:focus, QComboBox:on {{ border-color: {t['primary']}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: none; width: 0; height: 0; }}
QComboBox QAbstractItemView {{
    background: {t['surface']}; color: {t['text']};
    border: 1px solid {t['border_strong']}; border-radius: {R_MD}px;
    padding: 6px; outline: none;
}}
QComboBox QAbstractItemView::item {{ min-height: 28px; padding: 5px 12px; border-radius: 7px; }}
QComboBox QAbstractItemView::item:selected {{ background: {t['primary_weak']}; color: {t['text']}; }}

QSpinBox, QDoubleSpinBox {{
    background: {t['inset']}; border: 1px solid {t['border']};
    border-radius: {R_SM}px; padding: 7px 10px; color: {t['text']};
    min-height: 30px; font-family: {MONO}; font-size: 12px;
}}
QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {t['border_strong']}; }}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {t['primary']}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}

QCheckBox {{ color: {t['text_subtle']}; spacing: 8px; font-size: 12px; background: transparent; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {t['border_strong']}; background: {t['inset']};
}}
QCheckBox::indicator:hover {{ border-color: {t['primary']}; }}
QCheckBox::indicator:checked {{ background: {t['signal']}; border-color: {t['signal']}; }}

QProgressBar {{
    background: {t['surface2']}; border: none; border-radius: 3px;
    height: 6px; color: transparent; text-align: center;
}}
QProgressBar::chunk {{ background: {t['primary']}; border-radius: 3px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t['border_strong']}; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t['border_strong']}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QMenu {{
    background: {t['surface']}; color: {t['text']};
    border: 1px solid {t['border_strong']}; border-radius: {R_MD}px; padding: 5px;
}}
QMenu::item {{ padding: 6px 14px; border-radius: 6px; }}
QMenu::item:selected {{ background: {t['primary']}; color: #ffffff; }}
"""


def button_qss(t: dict[str, Any], kind: str = "primary") -> str:
    """Stylesheet for a button of ``kind`` — primary | ghost | success | danger."""
    if kind == "primary":
        return f"""
QPushButton {{ background: {t['primary']}; color: #fff; border: none;
    border-radius: {R_MD}px; font-weight: 600; font-size: 13px; padding: 9px 15px; }}
QPushButton:hover {{ background: {t['primary_hover']}; }}
QPushButton:pressed {{ background: {t['primary_press']}; }}
QPushButton:disabled {{ background: {t['surface2']}; color: {t['text_muted']}; }}"""
    if kind == "ghost":
        return f"""
QPushButton {{ background: {t['surface']}; color: {t['text']};
    border: 1px solid {t['border_strong']}; border-radius: {R_MD}px;
    font-weight: 600; font-size: 12.5px; padding: 8px 13px; }}
QPushButton:hover {{ border-color: {t['primary_line']}; color: {t['primary']};
    background: {t['primary_weak']}; }}
QPushButton:disabled {{ color: {t['text_muted']}; border-color: {t['border']}; }}"""
    if kind == "success":
        return f"""
QPushButton {{ background: {t['success']}; color: #04170f; border: none;
    border-radius: {R_MD}px; font-weight: 700; font-size: 13px; padding: 9px 15px; }}
QPushButton:disabled {{ background: {t['surface2']}; color: {t['text_muted']}; }}"""
    if kind == "danger":
        return f"""
QPushButton {{ background: {t['danger']}; color: #fff; border: none;
    border-radius: {R_MD}px; font-weight: 600; font-size: 13px; padding: 9px 15px; }}
QPushButton:disabled {{ background: {t['surface2']}; color: {t['text_muted']}; }}"""
    raise ValueError(f"unknown button kind: {kind!r}")
