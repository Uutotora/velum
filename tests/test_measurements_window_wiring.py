"""Wiring test: MeasurementsWindow's row_selected signal.

QuPath-style "select a cell in the table -> it's highlighted on the image"
needs the table to tell the outside world which cell_id was (de)selected,
without knowing anything about napari layers itself — that's row_selected,
consumed by PredictWidget._on_measurement_row_selected (see
test_predict_labels_display_wiring.py for that side of the wiring).

Skipped in the lightweight CI image (no PyQt6).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
mw_mod = pytest.importorskip("napari_app.widgets.measurements_window")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    w = mw_mod.MeasurementsWindow()
    yield w
    w.close()


def _result(ids):
    """A minimal but real-shaped measurement result: cell_id first column,
    one numeric column after it (matches every real schema's convention)."""
    columns = [("cell_id", "Cell", "id"), ("area", "Area", "px²")]
    rows = [[i, float(i * 10)] for i in ids]
    summary = {"area": {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}}
    return {"columns": columns, "rows": rows, "summary": summary, "n_cells": len(ids)}


def test_selecting_a_row_emits_its_cell_id(window):
    window.set_result(_result([1, 2, 3]))
    received = []
    window.row_selected.connect(received.append)

    window._table.selectRow(1)   # row index 1 -> cell_id 2 (see _result)
    assert received == [2]


def test_clearing_selection_emits_minus_one(window):
    window.set_result(_result([1, 2, 3]))
    received = []
    window.row_selected.connect(received.append)

    window._table.selectRow(0)
    window._table.clearSelection()
    assert received == [1, -1]


def test_row_selected_cell_id_survives_sorting(window):
    """Column 0 is read back from the clicked table item, not indexed out of
    result["rows"] by row number — sorting the table (a one-click, built-in
    QTableWidget feature) reorders visible rows independently of that list,
    so indexing by row number would silently pick the wrong cell.
    """
    window.set_result(_result([1, 2, 3]))
    window._table.sortByColumn(1, Qt.SortOrder.DescendingOrder)

    received = []
    window.row_selected.connect(received.append)
    window._table.selectRow(0)   # visually first row after a descending sort -> cell_id 3
    assert received == [3]


def test_set_result_replaces_the_table_rather_than_appending(window):
    window.set_result(_result([1, 2]))
    window.set_result(_result([9]))   # a fresh result, not an addition to the old one
    assert window._table.rowCount() == 1
    assert window._table.item(0, 0).data(Qt.ItemDataRole.DisplayRole) == 9
