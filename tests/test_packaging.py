"""Pure-logic guards on the packaging metadata (pyproject.toml).

These catch the ways packaging silently rots: a py-module dropped from the flat
layout, or a console entry point pointing at a moved file. Uses only stdlib
``tomllib`` (py3.11+) so it runs in the light CI test group without
torch/PyQt6/pyyaml.
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
    for required in {"torch", "pyqt6", "cellpose", "numpy"}:
        assert required in names, f"missing runtime dep: {required}"
    # napari was removed when Studio dropped the napari plugin — it must NOT
    # come back as a runtime dep (Studio has its own Qt canvas + layer model).
    assert "napari" not in names, "napari must not be a runtime dependency"


def test_test_group_is_light():
    """The CI test group must not drag in torch/napari (kept fast)."""
    group = _load()["dependency-groups"]["test"]
    text = " ".join(group).lower()
    assert "pytest" in text
    assert "torch" not in text and "napari" not in text


def test_console_script_entry_points_launch_studio():
    scripts = _load()["project"]["scripts"]
    # Both launchers start the Studio app; the module + main() must exist.
    for name in ("cellseg1", "cellseg1-studio"):
        assert scripts[name] == "studio.app:main", f"{name} must launch studio.app:main"
    module_path = REPO / "studio" / "app.py"
    assert module_path.exists()
    assert "def main(" in module_path.read_text()


def test_no_napari_plugin_manifest():
    """Studio is a standalone app, not a napari plugin — there must be no
    napari.manifest entry point and no napari_app package left behind."""
    data = _load()
    entry_points = data["project"].get("entry-points", {})
    assert "napari.manifest" not in entry_points
    assert not (REPO / "napari_app").exists(), "napari_app/ must be fully removed"


def test_ml_core_package_is_shipped():
    """The engine-agnostic ML core Studio imports must be a packaged module."""
    include = " ".join(_load()["tool"]["setuptools"]["packages"]["find"]["include"])
    assert "velum_core*" in include
    assert (REPO / "velum_core" / "predict_controller.py").exists()
    assert (REPO / "velum_core" / "engine_registry.py").exists()


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
