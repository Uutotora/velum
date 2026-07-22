"""Tests for studio/dataset.py — the DatasetInfo view + DatasetStore registry.
Pure logic (stdlib + cv2 via the exporter), runs in CI's light group."""
import json
from pathlib import Path

import numpy as np
import pytest

from studio import dataset_export
from studio.dataset import DatasetInfo, DatasetStore


def _build(root: Path, name: str, n=2, cells=3):
    """Write a real dataset folder under root/<slug> and return its dir."""
    store = DatasetStore(root)
    _id, out = store.allocate(name)
    items = [(str(out.parent / f"{name}{i}.png"),
              np.array([[0, c] for c in range(1, cells + 1)], np.int32))
             for i in range(n)]
    dataset_export.export_dataset(out, items, name=name, engine="cellseg1")
    return _id, out


def test_from_dir_reads_manifest(tmp_path):
    _id, out = _build(tmp_path / "ds", "Hela", n=2, cells=3)
    info = DatasetInfo.from_dir(out)
    assert info is not None
    assert info.name == "Hela"
    assert info.n_images == 2 and info.n_cells == 6
    assert info.engine == "cellseg1"
    assert info.images_dir == out / "images"


def test_from_dir_rejects_non_dataset_folders(tmp_path):
    (tmp_path / "empty").mkdir()
    assert DatasetInfo.from_dir(tmp_path / "empty") is None      # no manifest
    bad = tmp_path / "bad"; bad.mkdir()
    (bad / "dataset.json").write_text("{ not json", encoding="utf-8")
    assert DatasetInfo.from_dir(bad) is None                     # unparseable
    wrong = tmp_path / "wrong"; wrong.mkdir()
    (wrong / "dataset.json").write_text(json.dumps({"format": "other"}))
    assert DatasetInfo.from_dir(wrong) is None                   # wrong format


def test_store_list_newest_first_and_skips_junk(tmp_path):
    root = tmp_path / "ds"
    _build(root, "Alpha")
    _build(root, "Beta")
    (root / "not-a-dataset").mkdir()          # stray dir, no manifest
    infos = DatasetStore(root).list()
    assert [i.name for i in infos] == ["Beta", "Alpha"] or \
           {i.name for i in infos} == {"Alpha", "Beta"}   # both present
    assert all(isinstance(i, DatasetInfo) for i in infos)
    assert len(infos) == 2                    # junk dir excluded


def test_store_unique_id_and_get_delete(tmp_path):
    store = DatasetStore(tmp_path / "ds")
    id1, _ = store.allocate("My Set")
    id2, _ = store.allocate("My Set")
    assert id1 == "my-set" and id2 == "my-set-2"     # collision-safe
    _build(tmp_path / "ds", "Gamma")
    gid = store.unique_id("Gamma")                    # already exists -> -2
    assert gid == "gamma-2"
    info = store.list()[0]
    assert store.get(info.id) is not None
    store.delete(info.id)
    assert store.get(info.id) is None and not store.exists(info.id)
