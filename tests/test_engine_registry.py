"""Unit tests for velum_core.engine_registry.

Covers the registry mechanism in isolation (register/get/all_engines,
EngineSpec label defaulting) plus a regression check that importing
velum_core.engines — the module owning the two built-in engines — registers
them with the exact label strings the rest of the app (predict_widget's
combo/benchmark UI, predict_controller.ENGINE_LABELS) depends on. Pure
Python, no Qt/torch/cellpose needed — runs in the lightweight CI job too.
"""
import pytest

from velum_core import engine_registry
from velum_core.engine_registry import EngineSpec


@pytest.fixture
def isolated_registry(monkeypatch):
    """A throwaway empty registry so register()/get() tests don't depend on
    (or clobber) the real cellseg1/cellpose entries other tests rely on."""
    monkeypatch.setattr(engine_registry, "_registry", {})
    return engine_registry


def _fake_predict(image, config):
    return image


# ── Registry mechanism ────────────────────────────────────────────────────────

def test_register_and_get_roundtrip(isolated_registry):
    spec = EngineSpec(key="fake", label="Fake Engine", predict=_fake_predict)
    isolated_registry.register(spec)
    assert isolated_registry.get("fake") is spec


def test_get_unknown_key_raises(isolated_registry):
    with pytest.raises(ValueError, match="nope"):
        isolated_registry.get("nope")


def test_is_registered(isolated_registry):
    assert not isolated_registry.is_registered("fake")
    isolated_registry.register(EngineSpec(key="fake", label="Fake", predict=_fake_predict))
    assert isolated_registry.is_registered("fake")


def test_register_replaces_existing_key(isolated_registry):
    first = EngineSpec(key="fake", label="First", predict=_fake_predict)
    second = EngineSpec(key="fake", label="Second", predict=_fake_predict)
    isolated_registry.register(first)
    isolated_registry.register(second)
    assert isolated_registry.get("fake") is second


def test_all_engines_preserves_registration_order(isolated_registry):
    isolated_registry.register(EngineSpec(key="a", label="A", predict=_fake_predict))
    isolated_registry.register(EngineSpec(key="b", label="B", predict=_fake_predict))
    isolated_registry.register(EngineSpec(key="c", label="C", predict=_fake_predict))
    assert [s.key for s in isolated_registry.all_engines()] == ["a", "b", "c"]


def test_predict_is_called_through(isolated_registry):
    calls = []

    def _predict(image, config):
        calls.append((image, config))
        return "mask"

    isolated_registry.register(EngineSpec(key="fake", label="Fake", predict=_predict))
    result = isolated_registry.get("fake").predict("img", {"engine": "fake"})
    assert result == "mask"
    assert calls == [("img", {"engine": "fake"})]


# ── EngineSpec defaults ────────────────────────────────────────────────────────

def test_bench_and_result_label_default_to_label():
    spec = EngineSpec(key="x", label="X Engine", predict=_fake_predict)
    assert spec.bench_label == "X Engine"
    assert spec.result_label == "X Engine"


def test_explicit_bench_and_result_label_are_kept():
    spec = EngineSpec(key="x", label="X Engine", predict=_fake_predict,
                       bench_label="X (bench)", result_label="X")
    assert spec.bench_label == "X (bench)"
    assert spec.result_label == "X"


def test_available_defaults_to_true():
    assert EngineSpec(key="x", label="X", predict=_fake_predict).available() is True


def test_status_line_defaults_to_none():
    assert EngineSpec(key="x", label="X", predict=_fake_predict).status_line is None


# ── Built-in engines (velum_core.engines) ─────────────────────────────────────

def test_builtin_engines_are_registered():
    import velum_core.engines  # noqa: F401 — triggers registration
    assert engine_registry.is_registered("cellseg1")
    assert engine_registry.is_registered("cellpose")


def test_builtin_cellseg1_labels():
    import velum_core.engines  # noqa: F401
    spec = engine_registry.get("cellseg1")
    assert spec.label == "CellSeg1 · LoRA (one-shot, fine-tuned)"
    assert spec.bench_label == "CellSeg1 · LoRA (current checkpoint)"
    assert spec.result_label == "CellSeg1 (LoRA)"
    assert spec.available() is True


def test_builtin_cellpose_labels():
    import velum_core.engines  # noqa: F401
    spec = engine_registry.get("cellpose")
    assert spec.label == "Cellpose-SAM (zero-shot, generalist)"
    assert spec.bench_label == "Cellpose-SAM (zero-shot)"
    assert spec.result_label == "Cellpose-SAM"


def test_builtin_cellseg1_status_line_reports_cache_status():
    import velum_core.engines  # noqa: F401
    spec = engine_registry.get("cellseg1")
    assert spec.status_line is not None
    assert "model:" in spec.status_line()


def test_builtin_cellpose_status_line_is_unset():
    import velum_core.engines  # noqa: F401
    assert engine_registry.get("cellpose").status_line is None
