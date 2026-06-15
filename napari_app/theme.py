# napari dark theme palette
BG       = "#262930"   # panel background
FG       = "#414851"   # widget background / dividers
PRIMARY  = "#5a626c"   # inactive elements
SECONDARY= "#868e93"   # placeholder text
TEXT     = "#f0f1f2"   # main text
ACCENT   = "#007acc"   # blue accent (napari current)
GREEN    = "#2ea043"   # success / run
RED      = "#c0392b"   # stop / danger
CONSOLE  = "#121212"   # log background
DIM      = "#a0a8b0"   # secondary text

WIDGET_SS = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
}}
QScrollArea {{ border: none; background: {BG}; }}
QScrollBar:vertical {{
    background: {BG}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {FG}; border-radius: 4px; min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QLineEdit {{
    background: {FG};
    border: 1px solid {PRIMARY};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT};
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QLineEdit::placeholder {{ color: {SECONDARY}; }}

QComboBox {{
    background: {FG};
    border: 1px solid {PRIMARY};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT};
}}
QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {FG}; color: {TEXT};
    selection-background-color: {ACCENT};
    border: 1px solid {PRIMARY};
}}

QSpinBox, QDoubleSpinBox {{
    background: {FG};
    border: 1px solid {PRIMARY};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {PRIMARY}; border: none; width: 16px; border-radius: 2px;
}}

QProgressBar {{
    background: {FG};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 4px;
}}

QGroupBox {{
    border: 1px solid {FG};
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 8px;
    color: {DIM};
    font-size: 11px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {DIM};
}}

QTextEdit {{
    background: {CONSOLE};
    border: 1px solid {FG};
    border-radius: 4px;
    color: {DIM};
    font-family: "Menlo", "Monaco", "Courier New", monospace;
    font-size: 11px;
}}

QToolButton {{
    background: transparent;
    border: none;
    color: {TEXT};
    text-align: left;
    padding: 4px 2px;
    font-weight: 600;
    font-size: 12px;
}}
QToolButton:hover {{ color: {ACCENT}; }}
"""

BTN_PRIMARY = f"""
    QPushButton {{
        background: {ACCENT};
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: 600;
        font-size: 13px;
        padding: 8px;
    }}
    QPushButton:hover {{ background: #1a8cd8; }}
    QPushButton:pressed {{ background: #005fa3; }}
    QPushButton:disabled {{ background: {FG}; color: {SECONDARY}; }}
"""

BTN_SUCCESS = f"""
    QPushButton {{
        background: {GREEN};
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: 600;
        font-size: 13px;
        padding: 8px;
    }}
    QPushButton:hover {{ background: #3ab455; }}
    QPushButton:pressed {{ background: #1e7e34; }}
    QPushButton:disabled {{ background: {FG}; color: {SECONDARY}; }}
"""

BTN_DANGER = f"""
    QPushButton {{
        background: {RED};
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: 600;
        padding: 8px;
    }}
    QPushButton:hover {{ background: #e74c3c; }}
    QPushButton:disabled {{ background: {FG}; color: {SECONDARY}; }}
"""

BTN_SECONDARY = f"""
    QPushButton {{
        background: {FG};
        color: {TEXT};
        border: 1px solid {PRIMARY};
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 11px;
    }}
    QPushButton:hover {{ background: {PRIMARY}; border-color: {ACCENT}; }}
    QPushButton:pressed {{ background: {ACCENT}; }}
"""

BTN_PRESET = f"""
    QPushButton {{
        background: transparent;
        color: {ACCENT};
        border: 1px solid {ACCENT};
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 500;
    }}
    QPushButton:hover {{ background: {ACCENT}; color: white; }}
    QPushButton:pressed {{ background: #005fa3; color: white; }}
"""

BTN_BROWSE = f"""
    QPushButton {{
        background: {FG};
        color: {SECONDARY};
        border: 1px solid {PRIMARY};
        border-radius: 4px;
        font-size: 13px;
        padding: 0;
    }}
    QPushButton:hover {{ color: {TEXT}; background: {PRIMARY}; }}
"""
