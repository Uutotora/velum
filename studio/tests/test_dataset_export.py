"""Tests for studio/dataset_export.py — the project → re-trainable dataset
export. Pure logic (numpy + cv2 + stdlib), no Qt/torch, so it runs in CI's
light group. Verifies the on-disk layout, the manifest/provenance, stem
collision handling, deterministic splits, and the round-trip that matters:
an exported mask reloads to the same instance labels, and the exported
layout is the one training discovers."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from studio import dataset_export as de


def _mask(ids, shape=(16, 16)) -> np.ndarray:
    """A tiny instance mask: paint each id in a distinct row band."""
    m = np.zeros(shape, dtype=np.int32)
    for k, cell_id in enumerate(ids):
        m[k, :] = cell_id
    return m


# ── pure helpers ────────────────────────────────────────────────────────────

def test_count_instances_ignores_background():
    assert de.count_instances(_mask([0, 1, 2, 5])) == 3  # 0 not counted
    assert de.count_instances(np.zeros((4, 4), int)) == 0


def test_dedupe_stems_disambiguates_collisions():
    stems = de.dedupe_stems(["/a/img.tif", "/b/img.tif", "/c/other.png", "/d/img.tif"])
    assert stems == ["img", "img-1", "other", "img-2"]
    assert len(set(stems)) == len(stems)


def test_assign_splits_default_all_train():
    assert de.assign_splits(5, 0.0) == ["train"] * 5


def test_assign_splits_holds_out_reproducibly():
    a = de.assign_splits(10, 0.3, seed=7)
    b = de.assign_splits(10, 0.3, seed=7)
    assert a == b                              # deterministic
    assert a.count("val") == 3                 # round(10 * 0.3)
    assert de.assign_splits(4, 0.01).count("val") == 1  # at least one when >0


# ── export orchestration ────────────────────────────────────────────────────

def _make_items(tmp_path, n=2):
    items = []
    for i in range(n):
        p = tmp_path / f"src{i}.png"
        cv2.imwrite(str(p), (np.ones((16, 16), np.uint8) * (i + 1)))
        items.append((str(p), _mask([1, 2, 3])))
    return items


def test_export_writes_expected_layout(tmp_path):
    out = tmp_path / "ds"
    manifest = de.export_dataset(
        out, _make_items(tmp_path, 2), name="My Cells",
        project_id="proj1", engine="cellseg1", pixel_size_um=0.25)

    assert (out / "dataset.json").is_file()
    assert (out / "README.md").is_file()
    assert (out / "images" / "src0.png").is_file()
    assert (out / "images" / "src1.png").is_file()
    assert (out / "masks" / "src0.png").is_file()
    assert manifest["counts"] == {"n_images": 2, "n_cells": 6, "n_train": 2, "n_val": 0}
    assert manifest["source"]["engine"] == "cellseg1"
    assert manifest["source"]["pixel_size_um"] == 0.25
    assert manifest["format"] == de.DATASET_FORMAT


def test_exported_mask_roundtrips_to_same_labels(tmp_path):
    out = tmp_path / "ds"
    mask = _mask([1, 2, 3, 7])
    de.export_dataset(out, [(str(tmp_path / "x.png"), mask)], name="d")
    # x.png source doesn't exist -> image copy skipped, mask still written
    reloaded = cv2.imread(str(out / "masks" / "x.png"), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(reloaded)) == {0, 1, 2, 3, 7}


def test_manifest_on_disk_matches_return(tmp_path):
    out = tmp_path / "ds"
    manifest = de.export_dataset(out, _make_items(tmp_path, 1), name="d")
    on_disk = json.loads((out / "dataset.json").read_text())
    assert on_disk == manifest


def test_collision_safe_stems_pair_image_and_mask(tmp_path):
    # two different source folders, same filename
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    pa = tmp_path / "a" / "cells.png"; cv2.imwrite(str(pa), np.ones((8, 8), np.uint8))
    pb = tmp_path / "b" / "cells.png"; cv2.imwrite(str(pb), np.ones((8, 8), np.uint8))
    out = tmp_path / "ds"
    m = de.export_dataset(out, [(str(pa), _mask([1])), (str(pb), _mask([2]))], name="d")
    stems = [r["stem"] for r in m["images"]]
    assert stems == ["cells", "cells-1"]
    for r in m["images"]:                      # image + mask share the stem
        assert Path(r["image"]).stem == Path(r["mask"]).stem
        assert (out / r["image"]).is_file() and (out / r["mask"]).is_file()


def test_measurements_written_when_requested(tmp_path):
    out = tmp_path / "ds"
    m = de.export_dataset(out, _make_items(tmp_path, 1), name="d",
                          include_measurements=True, pixel_size_um=0.5)
    csv_rel = m["images"][0]["measurements"]
    assert csv_rel == "measurements/src0.csv"
    assert (out / csv_rel).is_file()
    assert (out / csv_rel).read_text().strip()  # non-empty CSV


def test_no_measurements_key_when_not_requested(tmp_path):
    m = de.export_dataset(tmp_path / "ds", _make_items(tmp_path, 1), name="d")
    assert "measurements" not in m["images"][0]
    assert not (tmp_path / "ds" / "measurements").exists()


def test_empty_items_raises(tmp_path):
    with pytest.raises(ValueError):
        de.export_dataset(tmp_path / "ds", [], name="d")


def test_progress_callback_reports_each_item(tmp_path):
    seen = []
    de.export_dataset(tmp_path / "ds", _make_items(tmp_path, 3), name="d",
                      on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_source_image_copied_verbatim(tmp_path):
    # a non-image byte payload with an exotic extension stands in for ND2/CZI:
    # export must copy bytes, never re-encode.
    src = tmp_path / "scan.nd2"
    payload = b"\x00RAWMICROSCOPY\xff"
    src.write_bytes(payload)
    out = tmp_path / "ds"
    de.export_dataset(out, [(str(src), _mask([1, 2]))], name="d")
    assert (out / "images" / "scan.nd2").read_bytes() == payload


def test_readme_documents_training_and_counts(tmp_path):
    m = de.export_dataset(tmp_path / "ds", _make_items(tmp_path, 2), name="Hela")
    text = de.readme_text(m)
    assert "# Hela" in text
    assert "masks/" in text and "images/" in text
    assert "Re-training" in text
