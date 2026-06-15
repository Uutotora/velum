# napari dark theme — exact palette
BG       = "#262930"
FG       = "#2e333c"
BORDER   = "#3a3f48"
TEXT     = "#c8ccd2"
DIM      = "#6b7280"
ACCENT   = "#007acc"
SUCCESS  = "#2d6a2d"
DANGER   = "#7a2020"
CONSOLE  = "#1a1d23"

# Global widget stylesheet — minimal, matches napari exactly
WIDGET_SS = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: -apple-system, "SF Pro Text", Arial, sans-serif;
    font-size: 12px;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {BG};
    border: none;
}}
QScrollBar:vertical {{
    background: transparent; width: 6px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 3px; min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QLabel {{ background: transparent; color: {TEXT}; }}

QLineEdit {{
    background: {FG};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 7px;
    color: {TEXT};
}}
QLineEdit:focus {{ border-color: {ACCENT}; }}

QComboBox {{
    background: {FG};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 7px;
    color: {TEXT};
}}
QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {FG}; color: {TEXT}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    outline: none;
}}

QSpinBox, QDoubleSpinBox {{
    background: {FG};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 7px;
    color: {TEXT};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {BORDER}; border: none; width: 14px; border-radius: 1px;
}}

QProgressBar {{
    background: {FG}; border: none; border-radius: 2px;
    height: 3px; color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}

QTextEdit {{
    background: {CONSOLE};
    border: 1px solid {BORDER};
    border-radius: 3px;
    color: {DIM};
    font-family: "Menlo", "SF Mono", "Courier New", monospace;
    font-size: 11px;
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 6px;
    padding-top: 6px;
    color: {DIM};
    font-size: 11px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 8px;
    padding: 0 4px; color: {DIM};
}}

QToolButton {{
    background: transparent;
    border: none;
    color: {DIM};
    text-align: left;
    padding: 2px 0;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
QToolButton:hover {{ color: {TEXT}; }}
"""

BTN_PRIMARY = f"""
QPushButton {{
    background: {ACCENT};
    color: #ffffff;
    border: none;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    padding: 7px 12px;
}}
QPushButton:hover {{ background: #1a8fd8; }}
QPushButton:pressed {{ background: #005fa3; }}
QPushButton:disabled {{ background: {FG}; color: {DIM}; }}
"""

BTN_SUCCESS = f"""
QPushButton {{
    background: {SUCCESS};
    color: #d0e8d0;
    border: none;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    padding: 7px 12px;
}}
QPushButton:hover {{ background: #357a35; }}
QPushButton:pressed {{ background: #1e4d1e; }}
QPushButton:disabled {{ background: {FG}; color: {DIM}; }}
"""

BTN_DANGER = f"""
QPushButton {{
    background: {DANGER};
    color: #e8c0c0;
    border: none;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    padding: 7px 12px;
}}
QPushButton:hover {{ background: #9e2828; }}
QPushButton:disabled {{ background: {FG}; color: {DIM}; }}
"""

BTN_SECONDARY = f"""
QPushButton {{
    background: {FG};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 5px 10px;
    font-size: 11px;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: #ffffff; }}
"""

BTN_PRESET = f"""
QPushButton {{
    background: transparent;
    color: {DIM};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 6px;
    font-size: 11px;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {TEXT}; }}
QPushButton:pressed {{ background: {ACCENT}; color: #fff; border-color: {ACCENT}; }}
"""

BTN_BROWSE = f"""
QPushButton {{
    background: {FG};
    color: {DIM};
    border: 1px solid {BORDER};
    border-radius: 3px;
    font-size: 12px;
    padding: 0;
}}
QPushButton:hover {{ color: {TEXT}; border-color: {ACCENT}; }}
"""
