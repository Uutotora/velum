"""Pure-logic guards on the packaging metadata (pyproject.toml + napari.yaml).

These catch the ways packaging silently rots: a py-module dropped from the
flat layout, an entry point pointing at a moved file, or a napari manifest that
references a function that no longer exists. Uses only stdlib ``tomllib``
(py3.11+) so it runs in the light CI test group without torch/napari/pyyaml.
"""
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"


def _load():
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def test_pyproject_exists_and_names_project():
    data = _load()
    assert data["project"]["name"] == "cellseg1"
    assert data["project"]["version"]


def test_core_runtime_deps_declared():
    deps = _load()["project"]["dependencies"]
    names = {d.split(">")[0].split("=")[0].split("<")[0].strip().lower() for d in deps}
    for required in {"torch", "napari", "pyqt6", "cellpose", "numpy"}:
        assert required in names, f"missing runtime dep: {required}"


def test_test_group_is_light():
    """The CI test group must not drag in torch/napari (kept fast)."""
    group = _load()["dependency-groups"]["test"]
    text = " ".join(group).lower()
    assert "pytest" in text
    assert "torch" not in text and "napari" not in text


def test_console_script_entry_point_resolves():
    scripts = _load()["project"]["scripts"]
    target = scripts["cellseg1"]
    assert target == "napari_app.main:main"
    module_path = REPO / "napari_app" / "main.py"
    assert module_path.exists()
    assert "def main(" in module_path.read_text()


def test_napari_manifest_entry_point_and_file():
    eps = _load()["project"]["entry-points"]["napari.manifest"]
    # e.g. "napari_app:napari.yaml"
    pkg, _, manifest_name = eps["cellseg1"].partition(":")
    manifest = REPO / pkg / manifest_name
    assert manifest.exists(), f"manifest missing: {manifest}"
    text = manifest.read_text()
    assert "name: cellseg1" in text
    # the python_name the manifest points at must exist as a real function
    assert "napari_app._npe2:make_predict_widget" in text
    npe2_mod = REPO / "napari_app" / "_npe2.py"
    assert npe2_mod.exists()
    assert "def make_predict_widget(" in npe2_mod.read_text()


def test_declared_py_modules_all_exist():
    """Every top-level module we ship must be a real file at the repo root."""
    py_modules = _load()["tool"]["setuptools"]["py-modules"]
    assert py_modules  # non-empty
    for mod in py_modules:
        assert (REPO / f"{mod}.py").exists(), f"py-module not found: {mod}.py"


def test_namespace_packages_included_in_find():
    """peft/ and data/ have no __init__.py — packaging must find namespaces."""
    find = _load()["tool"]["setuptools"]["packages"]["find"]
    assert find["namespaces"] is True
    include = " ".join(find["include"])
    assert "peft*" in include and "data*" in include
