# CellSeg1 — precision dark theme
# Cool blue-tinted grays evoke a dark microscopy room; accent blue reserved
# strictly for interactive elements so it always reads as "action available".

BG          = "#14192a"   # very dark panel background
FG          = "#1e2c44"   # card surfaces (clearly above BG)
CARD_HEADER = "#2d3f5e"   # card title bands (distinctly lighter than FG)
BORDER      = "#3a5070"   # visible borders
TEXT        = "#dce4f0"   # primary text
DIM         = "#5e6d88"   # subdued text (timestamps, hints)
LABEL       = "#8a9bbe"   # field labels and section headers
ACCENT      = "#4d8fff"   # interactive blue — buttons, focus, links ONLY
SUCCESS     = "#1d9e6e"   # vibrant biogreen (GFP-inspired)
DANGER      = "#c94f4f"   # destructive red
CONSOLE     = "#0e1220"   # console / log background

WIDGET_SS = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {BG};
    border: none;
}}
QScrollBar:vertical {{
    background: transparent; width: 7px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 3px; min-height: 28px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QLabel {{ background: transparent; color: {TEXT}; }}

QLineEdit {{
    background: {FG};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 9px;
    color: {TEXT};
    min-height: 28px;
}}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QLineEdit[readOnly="true"] {{ color: {LABEL}; }}

QComboBox {{
    background: {FG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 10px;
    color: {TEXT};
    min-height: 30px;
}}
QComboBox:hover {{ border-color: {LABEL}; }}
QComboBox:focus, QComboBox:on {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right;
    border: none; width: 22px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {LABEL};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background: {FG}; color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
    outline: none;
    font-size: 13px;
}}
QComboBox QAbstractItemView::item {{
    min-height: 26px;
    padding: 4px 10px;
    border-radius: 4px;
    color: {TEXT};
}}
QComboBox QAbstractItemView::item:hover {{
    background: rgba(77, 143, 255, 0.16);
    color: {TEXT};
}}
QComboBox QAbstractItemView::item:selected {{
    background: {ACCENT};
    color: #ffffff;
}}

QSpinBox, QDoubleSpinBox {{
    background: {FG};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 9px;
    color: {TEXT};
    min-height: 28px;
    font-family: "Menlo", "SF Mono", monospace;
    font-size: 12px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 0px; border: none;
}}

QProgressBar {{
    background: {FG}; border: none; border-radius: 2px;
    height: 3px; color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}

QTextEdit {{
    background: {CONSOLE};
    border: 1px solid {BORDER};
    border-radius: 5px;
    color: {LABEL};
    font-family: "Menlo", "SF Mono", "Courier New", monospace;
    font-size: 11px;
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 5px;
    margin-top: 6px;
    padding-top: 6px;
    color: {LABEL};
    font-size: 11px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 8px;
    padding: 0 4px; color: {LABEL};
}}

QToolButton {{
    background: transparent;
    border: none;
    color: {LABEL};
    text-align: left;
    padding: 2px 0;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
QToolButton:hover {{ color: {TEXT}; }}
"""

BTN_PRIMARY = f"""
QPushButton {{
    background: {ACCENT};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    padding: 9px 16px;
    letter-spacing: 0.3px;
}}
QPushButton:hover {{ background: #6099ff; }}
QPushButton:pressed {{ background: #3a78e0; }}
QPushButton:disabled {{ background: {FG}; color: {DIM}; }}
"""

BTN_SUCCESS = f"""
QPushButton {{
    background: {SUCCESS};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 14px;
}}
QPushButton:hover {{ background: #23b87e; }}
QPushButton:pressed {{ background: #167a55; }}
QPushButton:disabled {{ background: {FG}; color: {DIM}; }}
"""

BTN_DANGER = f"""
QPushButton {{
    background: {DANGER};
    color: #ffffff;
    border: none;
    border-radius: 5px;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 14px;
}}
QPushButton:hover {{ background: #d96060; }}
QPushButton:disabled {{ background: {FG}; color: {DIM}; }}
"""

BTN_SECONDARY = f"""
QPushButton {{
    background: transparent;
    color: {LABEL};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 12px;
    font-size: 12px;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {TEXT}; }}
QPushButton:pressed {{ background: rgba(77, 143, 255, 0.1); }}
QPushButton:disabled {{ color: {DIM}; border-color: {BORDER}; }}
"""

BTN_PRESET = f"""
QPushButton {{
    background: transparent;
    color: {DIM};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 4px;
    font-size: 11px;
    text-align: center;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {LABEL}; }}
QPushButton:checked {{
    background: rgba(77, 143, 255, 0.12);
    border-color: {ACCENT};
    color: {TEXT};
}}
QPushButton:pressed {{ background: {ACCENT}; color: #fff; border-color: {ACCENT}; }}
"""

BTN_BROWSE = f"""
QPushButton {{
    background: {FG};
    color: {LABEL};
    border: 1px solid {BORDER};
    border-radius: 5px;
    font-size: 14px;
    padding: 0;
    min-height: 28px;
    min-width:  28px;
}}
QPushButton:hover {{ color: {TEXT}; border-color: {ACCENT}; }}
"""
