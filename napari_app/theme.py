# CellSeg1 — "Lab" design language (graphite, dark instrument world)
# ---------------------------------------------------------------------------
# A production-grade token system for a scientific desktop tool. Calm graphite
# neutrals biased toward the accent; a single blue reserved strictly for
# interactive elements ("action available"); status colours are a separate
# family and never stand in for the accent.
#
# Backwards compatibility: every token name the widgets already import
# (BG, FG, CARD_HEADER, BORDER, TEXT, DIM, LABEL, ACCENT, SUCCESS, DANGER,
# CONSOLE, WIDGET_SS, BTN_*) is preserved — only the values and the component
# stylesheets are upgraded. New tokens are added alongside.
# ---------------------------------------------------------------------------

# ── Surfaces (five-step elevation ramp) ─────────────────────────────────────
BG_APP      = "#0e1117"   # app canvas behind panels
BG          = "#0f1319"   # panel background
FG          = "#161b23"   # card surface  (S1)  — legacy name kept
CARD_HEADER = "#1e2530"   # elevated / header / hover  (S2) — legacy name kept
INPUT       = "#12161d"   # input "well" (recessed field background)
CONSOLE     = "#0a0c11"   # console / log / data-table background

# ── Borders ─────────────────────────────────────────────────────────────────
BORDER        = "#262d38"  # hairline borders
BORDER_STRONG = "#333c49"  # emphasised borders / scrollbar handle

# ── Ink ─────────────────────────────────────────────────────────────────────
TEXT  = "#e8ecf3"   # primary text
LABEL = "#99a3b3"   # field labels, section headers
DIM   = "#667085"   # subdued text (timestamps, hints)

# ── Accent (interactive only) ───────────────────────────────────────────────
ACCENT         = "#4d8fff"
ACCENT_HOVER   = "#6aa1ff"
ACCENT_PRESSED = "#3a78e0"
ACCENT_SOFT    = "rgba(77, 143, 255, 0.12)"
ACCENT_LINE    = "rgba(77, 143, 255, 0.35)"

# ── Status (a separate family, never the accent) ────────────────────────────
SUCCESS      = "#22b47f"
WARNING      = "#e0a63b"
DANGER       = "#e0524d"
SUCCESS_SOFT = "rgba(34, 180, 127, 0.13)"
WARNING_SOFT = "rgba(224, 166, 59, 0.13)"
DANGER_SOFT  = "rgba(224, 82, 77, 0.13)"

# ── Selection / "lit" state (turquoise — nav rail active icon) ──────────────
TEAL      = "#2bd4c0"
TEAL_SOFT = "rgba(43, 212, 192, 0.15)"
TEAL_LINE = "rgba(43, 212, 192, 0.55)"

# ── Data-viz categorical palette (validated CVD-safe on the dark surface) ────
# worst adjacent ΔE 18.2 (tritan), all ≥ 3:1 contrast. Assign in fixed order.
VIZ = ["#4d8fff", "#199e70", "#c98500", "#e66767", "#9085e9"]
# single-hue sequential default = accent blue; median marker = SUCCESS.

# ── Type & form ─────────────────────────────────────────────────────────────
SANS = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif'
MONO = '"SF Mono", "Menlo", "JetBrains Mono", ui-monospace, monospace'

R_SM, R_MD, R_LG, R_XL = 6, 8, 10, 14  # radii

# ---------------------------------------------------------------------------
# Base widget stylesheet — applied at the panel root.
# ---------------------------------------------------------------------------
WIDGET_SS = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: {SANS};
    font-size: 13px;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {BG};
    border: none;
}}
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG}; border-radius: 4px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: #414b5a; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER_STRONG}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QLabel {{ background: transparent; color: {TEXT}; }}

QToolTip {{
    background: {CARD_HEADER};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 6px 9px;
    font-size: 12px;
}}

QLineEdit {{
    background: {INPUT};
    border: 1px solid {BORDER};
    border-radius: {R_SM}px;
    padding: 6px 10px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
    min-height: 30px;
}}
QLineEdit:hover {{ border-color: {BORDER_STRONG}; }}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QLineEdit[readOnly="true"] {{ color: {LABEL}; background: {BG}; }}

QComboBox {{
    background: {INPUT};
    border: 1px solid {BORDER};
    border-radius: {R_SM}px;
    padding: 6px 12px;
    color: {TEXT};
    min-height: 30px;
}}
QComboBox:hover {{ border-color: {BORDER_STRONG}; }}
QComboBox:focus, QComboBox:on {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right;
    border: none; width: 26px;
}}
QComboBox::down-arrow {{ image: none; width: 0; height: 0; border: none; }}
QComboBox QAbstractItemView {{
    background: {CARD_HEADER}; color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: {R_MD}px;
    padding: 5px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
    outline: none;
    font-size: 13px;
}}
QComboBox QAbstractItemView::item {{
    min-height: 27px; padding: 4px 10px; border-radius: 5px; color: {TEXT};
}}
QComboBox QAbstractItemView::item:hover {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QComboBox QAbstractItemView::item:selected {{ background: {ACCENT}; color: #ffffff; }}

QSpinBox, QDoubleSpinBox {{
    background: {INPUT};
    border: 1px solid {BORDER};
    border-radius: {R_SM}px;
    padding: 6px 10px;
    color: {TEXT};
    min-height: 30px;
    font-family: {MONO};
    font-size: 12px;
}}
QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {BORDER_STRONG}; }}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0px; border: none; }}

QCheckBox {{ color: {LABEL}; spacing: 8px; font-size: 12px; background: transparent; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {BORDER_STRONG}; background: {INPUT};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
    image: none;
}}

QProgressBar {{
    background: {CARD_HEADER}; border: none; border-radius: 3px;
    height: 5px; color: transparent; text-align: center;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QTextEdit {{
    background: {CONSOLE};
    border: 1px solid {BORDER};
    border-radius: {R_SM}px;
    color: {LABEL};
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
    font-family: {MONO};
    font-size: 11px;
}}

QListWidget {{
    background: {INPUT};
    border: 1px solid {BORDER};
    border-radius: {R_SM}px;
    color: {TEXT};
    font-size: 12px;
    outline: none;
}}
QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: {R_SM}px;
    margin-top: 8px;
    padding-top: 8px;
    color: {LABEL};
    font-size: 11px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 5px; color: {LABEL};
}}

QMenu {{
    background: {CARD_HEADER}; color: {TEXT};
    border: 1px solid {BORDER_STRONG}; border-radius: {R_MD}px; padding: 5px;
}}
QMenu::item {{ padding: 6px 14px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT}; color: #ffffff; }}

QPushButton::menu-indicator {{ image: none; width: 0; height: 0; }}

QToolButton {{
    background: transparent; border: none; color: {LABEL};
    text-align: left; padding: 2px 0;
    font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
}}
QToolButton:hover {{ color: {TEXT}; }}
"""

# ---------------------------------------------------------------------------
# Buttons — a clear hierarchy. Primary = accent, Success = status green,
# Secondary = ghost outline, Danger = status red. Preset + Browse are utility.
# ---------------------------------------------------------------------------
BTN_PRIMARY = f"""
QPushButton {{
    background: {ACCENT}; color: #ffffff; border: none;
    border-radius: {R_MD}px; font-size: 13px; font-weight: 600;
    padding: 10px 16px; letter-spacing: 0.2px;
}}
QPushButton:hover {{ background: {ACCENT_HOVER}; }}
QPushButton:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton:disabled {{ background: {CARD_HEADER}; color: {DIM}; }}
"""

BTN_SUCCESS = f"""
QPushButton {{
    background: {SUCCESS}; color: #04170f; border: none;
    border-radius: {R_MD}px; font-size: 13px; font-weight: 700; padding: 9px 15px;
}}
QPushButton:hover {{ background: #2bc78d; }}
QPushButton:pressed {{ background: #1a8f63; }}
QPushButton:disabled {{ background: {CARD_HEADER}; color: {DIM}; }}
"""

BTN_DANGER = f"""
QPushButton {{
    background: {DANGER}; color: #ffffff; border: none;
    border-radius: {R_MD}px; font-size: 13px; font-weight: 600; padding: 9px 15px;
}}
QPushButton:hover {{ background: #e96b66; }}
QPushButton:pressed {{ background: #c53f3a; }}
QPushButton:disabled {{ background: {CARD_HEADER}; color: {DIM}; }}
"""

BTN_SECONDARY = f"""
QPushButton {{
    background: transparent; color: {LABEL};
    border: 1px solid {BORDER_STRONG}; border-radius: {R_MD}px;
    padding: 7px 13px; font-size: 12px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {TEXT}; background: {ACCENT_SOFT}; }}
QPushButton:pressed {{ background: {ACCENT_SOFT}; }}
QPushButton:disabled {{ color: {DIM}; border-color: {BORDER}; }}
"""

BTN_PRESET = f"""
QPushButton {{
    background: {INPUT}; color: {DIM};
    border: 1px solid {BORDER}; border-radius: {R_MD}px;
    padding: 8px 6px; font-size: 11px; text-align: center;
}}
QPushButton:hover {{ border-color: {BORDER_STRONG}; color: {LABEL}; }}
QPushButton:checked {{
    background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {TEXT};
}}
QPushButton:pressed {{ background: {ACCENT_SOFT}; }}
"""

BTN_BROWSE = f"""
QPushButton {{
    background: {INPUT}; color: {LABEL};
    border: 1px solid {BORDER}; border-radius: {R_SM}px;
    font-size: 15px; padding: 0; min-height: 30px; min-width: 30px;
}}
QPushButton:hover {{ color: {TEXT}; border-color: {ACCENT}; background: {ACCENT_SOFT}; }}
"""

# ---------------------------------------------------------------------------
# Small helpers for building consistent chips / pills / badges in code.
# ---------------------------------------------------------------------------
def pill_ss(fg: str, bg: str, border: str) -> str:
    """Stylesheet for a rounded status pill / chip QLabel."""
    return (f"color:{fg}; background:{bg}; border:1px solid {border};"
            f"border-radius:999px; padding:3px 10px; font-size:11px; font-weight:600;")


def badge_ss(fg: str, bg: str) -> str:
    """Stylesheet for a compact monospaced badge QLabel."""
    return (f"color:{fg}; background:{bg}; border:none; border-radius:6px;"
            f"padding:3px 8px; font-size:11px; font-weight:600; font-family:{MONO};")
