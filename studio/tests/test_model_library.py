"""Tests for studio/model_library.py + model_library_controller.py — the Model
Library catalog, install-status logic, download (network faked), and the
import/remove round-trip. Pure logic, light CI group (no Qt/torch/network)."""
import json
from pathlib import Path

import pytest

from studio import model_library as ml
from studio.model_library import CatalogModel, LibraryRoots
from studio.model_library_controller import ModelLibraryController


@pytest.fixture
def roots(tmp_path):
    return LibraryRoots(storage_dir=tmp_path / "store",
                        checkpoints_dir=tmp_path / "checkpoints")


# ── catalog integrity ───────────────────────────────────────────────────────
def test_catalog_ids_unique_and_families_known():
    ids = [m.id for m in ml.catalog()]
    assert len(ids) == len(set(ids)), "duplicate catalog ids"
    for m in ml.catalog():
        assert m.family in ml.FAMILY_LABELS
        assert m.engine in ("cellseg1", "cellpose", "sam2")
        # every downloadable entry has a URL + filename; every built-in has neither
        if m.builtin:
            assert not m.url and not m.filename
        else:
            assert m.url and m.filename


def test_size_label_formats():
    assert ml.get_catalog_model("sam2_tiny").size_label == "149 MB"
    assert ml.get_catalog_model("sam_vit_h").size_label == "2.5 GB"
    assert ml.get_catalog_model("cp_cpsam").size_label == "—"


def test_get_catalog_model_missing():
    assert ml.get_catalog_model("nope") is None


# ── dest paths land where the engines look ──────────────────────────────────
def test_dest_paths_match_engine_conventions(roots):
    assert ml.dest_path(ml.get_catalog_model("sam_vit_b"), roots) == \
        roots.storage_dir / "sam_backbone" / "sam_vit_b_01ec64.pth"
    assert ml.dest_path(ml.get_catalog_model("sam2_large"), roots) == \
        roots.storage_dir / "sam2_checkpoints" / "sam2.1_hiera_large.pt"
    # built-in Cellpose entries have no file we own
    assert ml.dest_path(ml.get_catalog_model("cp_cpsam"), roots) is None


# ── install status ──────────────────────────────────────────────────────────
def test_is_installed_downloadable(roots):
    m = ml.get_catalog_model("sam2_tiny")
    assert not ml.is_installed(m, roots)
    p = ml.dest_path(m, roots)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"weights")
    assert ml.is_installed(m, roots)
    assert ml.installed_size_mb(m, roots) == 0  # 7 bytes rounds to 0 MB


def test_is_installed_empty_file_is_not_installed(roots):
    m = ml.get_catalog_model("sam2_tiny")
    p = ml.dest_path(m, roots)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    assert not ml.is_installed(m, roots)


def test_builtin_installed_tracks_cellpose(roots):
    m = ml.get_catalog_model("cp_cpsam")
    assert not ml.is_installed(m, roots, cellpose_available=False)
    assert ml.is_installed(m, roots, cellpose_available=True)


# ── download (network faked) ────────────────────────────────────────────────
def test_download_streams_and_renames(roots, monkeypatch):
    m = ml.get_catalog_model("sam2_tiny")
    seen = []

    def fake_retrieve(url, dest, on_progress=None):
        assert url == m.url
        Path(dest).write_bytes(b"x" * 2048)
        if on_progress:
            on_progress(1024, 2048)
            on_progress(2048, 2048)

    monkeypatch.setattr(ml, "_urlretrieve", fake_retrieve)
    out = ml.download(m, roots, on_progress=lambda d, t: seen.append((d, t)))
    assert out == ml.dest_path(m, roots)
    assert out.exists() and out.read_bytes() == b"x" * 2048
    assert not out.with_suffix(out.suffix + ".part").exists()  # temp cleaned up
    assert seen == [(1024, 2048), (2048, 2048)]


def test_download_failure_leaves_no_partial(roots, monkeypatch):
    m = ml.get_catalog_model("sam2_tiny")

    def boom(url, dest, on_progress=None):
        Path(dest).write_bytes(b"half")
        raise ml.DownloadError("offline")

    monkeypatch.setattr(ml, "_urlretrieve", boom)
    with pytest.raises(ml.DownloadError):
        ml.download(m, roots)
    assert not ml.dest_path(m, roots).exists()
    assert not ml.dest_path(m, roots).with_suffix(".pt.part").exists()


def test_download_builtin_rejected(roots):
    with pytest.raises(ValueError):
        ml.download(ml.get_catalog_model("cp_cpsam"), roots)


# ── import your own ─────────────────────────────────────────────────────────
def test_import_sam_backbone_renamed_to_canonical(roots, tmp_path):
    src = tmp_path / "my_sam_vit_l_weights.pth"
    src.write_bytes(b"w")
    dest = ml.import_checkpoint(src, ml.FAMILY_SAM_BACKBONE, roots)
    assert dest == roots.storage_dir / "sam_backbone" / "sam_vit_l_0b3195.pth"
    assert dest.exists()


def test_import_lora_writes_sidecar(roots, tmp_path):
    src = tmp_path / "my_lora.pth"
    src.write_bytes(b"w")
    dest = ml.import_checkpoint(src, ml.FAMILY_LORA, roots, name="apples_v1")
    assert dest == roots.storage_dir / "loras" / "apples_v1.pth"
    sidecar = dest.with_suffix(".json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["imported"] is True
    # and it now shows up as a local model via the controller
    ctrl = ModelLibraryController(storage_dir=roots.storage_dir,
                                  checkpoints_dir=roots.checkpoints_dir)
    names = [lm.name for lm in ctrl.local_models()]
    assert "apples_v1" in names


def test_import_lora_copies_existing_sidecar(roots, tmp_path):
    src = tmp_path / "trained.pth"
    src.write_bytes(b"w")
    src.with_suffix(".json").write_text(json.dumps({"vit_name": "vit_b",
                                                    "image_encoder_lora_rank": 16,
                                                    "train_id": [1]}))
    dest = ml.import_checkpoint(src, ml.FAMILY_LORA, roots)
    meta = json.loads(dest.with_suffix(".json").read_text())
    assert meta["vit_name"] == "vit_b" and "imported" not in meta


def test_import_missing_file(roots, tmp_path):
    with pytest.raises(ValueError):
        ml.import_checkpoint(tmp_path / "ghost.pth", ml.FAMILY_LORA, roots)


# ── remove ──────────────────────────────────────────────────────────────────
def test_remove_deletes_file_and_sidecar(roots):
    m = ml.get_catalog_model("sam2_tiny")
    p = ml.dest_path(m, roots)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"w")
    assert ml.remove(m, roots) is True
    assert not p.exists()
    assert ml.remove(m, roots) is False  # already gone


# ── controller ──────────────────────────────────────────────────────────────
def test_controller_catalog_entries_filter(roots):
    ctrl = ModelLibraryController(storage_dir=roots.storage_dir,
                                  checkpoints_dir=roots.checkpoints_dir)
    sam2 = ctrl.catalog_entries(family=ml.FAMILY_SAM2)
    assert sam2 and all(e.model.family == ml.FAMILY_SAM2 for e in sam2)
    assert len(ctrl.catalog_entries()) == len(ml.catalog())


def test_controller_families_in_catalog_order(roots):
    ctrl = ModelLibraryController(storage_dir=roots.storage_dir,
                                  checkpoints_dir=roots.checkpoints_dir)
    keys = [k for k, _ in ctrl.families()]
    assert keys[:2] == [ml.FAMILY_SAM_BACKBONE, ml.FAMILY_SAM2]


def test_controller_local_models_include_checkpoints(roots):
    roots.checkpoints_dir.mkdir(parents=True)
    (roots.checkpoints_dir / "cyto_special.pth").write_bytes(b"w")
    ctrl = ModelLibraryController(storage_dir=roots.storage_dir,
                                  checkpoints_dir=roots.checkpoints_dir)
    locals_ = ctrl.local_models()
    assert any(lm.name == "cyto_special" and lm.family == ml.FAMILY_CELLPOSE
               for lm in locals_)


def test_controller_download_async_reports_done(roots, monkeypatch):
    ctrl = ModelLibraryController(storage_dir=roots.storage_dir,
                                  checkpoints_dir=roots.checkpoints_dir)

    def fake_retrieve(url, dest, on_progress=None):
        Path(dest).write_bytes(b"w" * 10)

    monkeypatch.setattr(ml, "_urlretrieve", fake_retrieve)
    done = []
    t = ctrl.download_async("sam2_tiny", on_done=lambda p: done.append(p))
    t.join(timeout=5)
    assert done and done[0].exists()


def test_controller_download_async_reports_error(roots, monkeypatch):
    ctrl = ModelLibraryController(storage_dir=roots.storage_dir,
                                  checkpoints_dir=roots.checkpoints_dir)

    def boom(url, dest, on_progress=None):
        raise ml.DownloadError("no network")

    monkeypatch.setattr(ml, "_urlretrieve", boom)
    errs = []
    t = ctrl.download_async("sam2_tiny", on_error=errs.append)
    t.join(timeout=5)
    assert errs == ["no network"]
