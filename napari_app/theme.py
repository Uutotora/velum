# napari dark theme palette
BG      = "#262930"
FG      = "#2e333c"
BORDER  = "#3a3f48"
TEXT    = "#d0d4da"
DIM     = "#6b7280"
ACCENT  = "#007acc"
SUCCESS = "#2d6a2d"
DANGER  = "#7a2020"
CONSOLE = "#1a1d23"

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
    border-radius: 4px;
    padding: 5px 8px;
    color: {TEXT};
    min-height: 26px;
}}
QLineEdit:focus {{ border-color: {ACCENT}; }}

QComboBox {{
    background: {FG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    color: {TEXT};
    min-height: 26px;
}}
QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {FG}; color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    outline: none;
    font-size: 13px;
}}

QSpinBox, QDoubleSpinBox {{
    background: {FG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    color: {TEXT};
    min-height: 26px;
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
    border-radius: 4px;
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
}}
QToolButton:hover {{ color: {TEXT}; }}
"""

BTN_PRIMARY = f"""
QPushButton {{
    background: {ACCENT};
    color: #ffffff;
    border: none;
    border-radius: 5px;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 14px;
}}
QPushButton:hover {{ background: #1a8fd8; }}
QPushButton:pressed {{ background: #005fa3; }}
QPushButton:disabled {{ background: {FG}; color: {DIM}; }}
"""

BTN_SUCCESS = f"""
QPushButton {{
    background: {SUCCESS};
    color: #d4ead4;
    border: none;
    border-radius: 5px;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 14px;
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
    border-radius: 5px;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 14px;
}}
QPushButton:hover {{ background: #9e2828; }}
QPushButton:disabled {{ background: {FG}; color: {DIM}; }}
"""

BTN_SECONDARY = f"""
QPushButton {{
    background: {FG};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 12px;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: #ffffff; }}
QPushButton:disabled {{ color: {DIM}; }}
"""

BTN_PRESET = f"""
QPushButton {{
    background: transparent;
    color: {DIM};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 4px;
    font-size: 11px;
    text-align: center;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {TEXT}; }}
QPushButton:checked {{
    background: rgba(0, 122, 204, 0.14);
    border-color: {ACCENT};
    color: {TEXT};
}}
QPushButton:pressed {{ background: {ACCENT}; color: #fff; border-color: {ACCENT}; }}
"""

BTN_BROWSE = f"""
QPushButton {{
    background: {FG};
    color: {DIM};
    border: 1px solid {BORDER};
    border-radius: 4px;
    font-size: 13px;
    padding: 0;
    min-height: 28px;
    min-width:  28px;
}}
QPushButton:hover {{ color: {TEXT}; border-color: {ACCENT}; }}
"""
