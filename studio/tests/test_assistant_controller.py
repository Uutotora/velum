"""Pure-logic tests for studio/assistant_controller.py — no Qt, runs in the
light CI test group. Network calls (Ollama / Custom API) are always
monkeypatched or stubbed via a fake urllib.request.urlopen; nothing here
touches a real server.
"""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from studio import assistant_controller as ac


# ── AssistantSettings / AssistantSettingsStore ────────────────────────────────

def test_settings_default_backend_is_offline():
    s = ac.AssistantSettings()
    assert s.backend == "offline"
    assert s.ollama_model == "" and s.custom_base_url == "" and s.custom_model == ""


def test_settings_round_trip_through_dict():
    s = ac.AssistantSettings(backend="ollama", ollama_model="qwen2.5:7b")
    back = ac.AssistantSettings.from_dict(s.to_dict())
    assert back == s


def test_settings_from_dict_ignores_unknown_keys_and_fills_defaults():
    s = ac.AssistantSettings.from_dict({"backend": "custom", "custom_model": "gpt-x", "bogus": 1})
    assert s.backend == "custom"
    assert s.custom_model == "gpt-x"
    assert s.ollama_model == ""


def test_settings_from_dict_rejects_unknown_backend():
    s = ac.AssistantSettings.from_dict({"backend": "telepathy"})
    assert s.backend == "offline"


def test_settings_from_dict_handles_none():
    assert ac.AssistantSettings.from_dict(None) == ac.AssistantSettings()


def test_settings_store_round_trips(tmp_path):
    store = ac.AssistantSettingsStore(tmp_path / "settings.json")
    settings = ac.AssistantSettings(backend="custom", custom_base_url="http://localhost:1234/v1",
                                    custom_api_key="sk-x", custom_model="local-model")
    store.save(settings)
    loaded = ac.AssistantSettingsStore(tmp_path / "settings.json").load()
    assert loaded == settings


def test_settings_store_missing_file_returns_defaults(tmp_path):
    store = ac.AssistantSettingsStore(tmp_path / "nope.json")
    assert store.load() == ac.AssistantSettings()


def test_settings_store_corrupt_file_returns_defaults(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{not json")
    assert ac.AssistantSettingsStore(p).load() == ac.AssistantSettings()


def test_default_settings_path_uses_shared_storage_dir():
    assert ac.default_settings_path().name == "studio_assistant_settings.json"


# ── AssistantController construction / persistence ────────────────────────────

def test_controller_persists_settings_across_instances(tmp_path):
    c1 = ac.AssistantController(storage_dir=tmp_path)
    c1.settings.backend = "ollama"
    c1.settings.ollama_model = "llama3.2:3b"
    c1.save_settings()

    c2 = ac.AssistantController(storage_dir=tmp_path)
    assert c2.settings.backend == "ollama"
    assert c2.settings.ollama_model == "llama3.2:3b"


def test_controller_accepts_an_explicit_settings_store(tmp_path):
    store = ac.AssistantSettingsStore(tmp_path / "custom_name.json")
    c = ac.AssistantController(settings_store=store)
    assert c.settings_store is store


# ── backend_ready ──────────────────────────────────────────────────────────────

def test_backend_ready_offline_is_always_false(tmp_path):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "offline"
    assert c.backend_ready() is False


def test_backend_ready_ollama_needs_a_model(tmp_path):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "ollama"
    assert c.backend_ready() is False
    c.settings.ollama_model = "qwen2.5:7b"
    assert c.backend_ready() is True


def test_backend_ready_custom_needs_base_url_and_model(tmp_path):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "custom"
    assert c.backend_ready() is False
    c.settings.custom_base_url = "http://localhost:1234/v1"
    assert c.backend_ready() is False   # model still missing
    c.settings.custom_model = "local-model"
    assert c.backend_ready() is True


# ── diagnostics passthrough (reuses the real velum_core.advisor engine) ───────

def test_diagnose_without_mask_prompts_to_predict(tmp_path):
    c = ac.AssistantController(storage_dir=tmp_path)
    diag = c.diagnose(np.zeros((20, 20), dtype=np.uint8), None, {})
    assert any("Run a prediction" in f.title for f in diag["findings"])


def test_findings_to_text_and_merge_changes(tmp_path):
    c = ac.AssistantController(storage_dir=tmp_path)
    img = np.full((80, 80), 128, dtype=np.uint8)
    empty_mask = np.zeros((80, 80), dtype=np.int32)
    params = {"points_per_side": 32, "pred_iou_thresh": 0.8, "stability_score_thresh": 0.6,
             "box_nms_thresh": 0.05, "min_mask_area": 20, "resize_size": 512}
    diag = c.diagnose(img, empty_mask, params)
    text = c.findings_to_text(diag)
    assert "suggested:" in text
    changes = c.merge_changes(diag)
    assert "pred_iou_thresh" in changes and changes["pred_iou_thresh"] < 0.8


def test_merge_changes_combines_every_finding():
    diag = {"findings": [
        SimpleNamespace(changes={"a": 1}),
        SimpleNamespace(changes={}),
        SimpleNamespace(changes={"b": 2}),
    ]}
    assert ac.AssistantController.merge_changes(diag) == {"a": 1, "b": 2}


def test_parse_suggestions_delegates_to_advisor(tmp_path):
    c = ac.AssistantController(storage_dir=tmp_path)
    out = c.parse_suggestions("blah\nSUGGEST: points_per_side=48\n")
    assert out == {"points_per_side": 48}


# ── Custom-API bridge — mocked urllib, no real network ────────────────────────

def test_normalize_base_url_strips_whitespace_and_trailing_slash():
    assert ac._normalize_base_url("  http://localhost:1234/v1/  ") == "http://localhost:1234/v1"
    assert ac._normalize_base_url("") == ""
    assert ac._normalize_base_url(None) == ""


def test_custom_api_headers_only_adds_auth_when_key_present():
    assert "Authorization" not in ac._custom_api_headers("")
    assert ac._custom_api_headers("sk-x")["Authorization"] == "Bearer sk-x"


class _FakeResponse:
    def __init__(self, status=200, body: bytes = b"", lines: list[bytes] | None = None):
        self.status = status
        self._body = body
        self._lines = lines or []

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_custom_api_available_empty_url_never_hits_the_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("urlopen should not be called for an empty base_url")
    monkeypatch.setattr(ac.urllib.request, "urlopen", boom)
    assert ac.custom_api_available("") is False


def test_custom_api_available_true_on_2xx(monkeypatch):
    monkeypatch.setattr(ac.urllib.request, "urlopen", lambda req, timeout=0: _FakeResponse(status=200))
    assert ac.custom_api_available("http://localhost:1234/v1") is True


def test_custom_api_available_false_on_exception(monkeypatch):
    def boom(req, timeout=0):
        raise OSError("connection refused")
    monkeypatch.setattr(ac.urllib.request, "urlopen", boom)
    assert ac.custom_api_available("http://localhost:1234/v1") is False


def test_custom_api_models_parses_data_list(monkeypatch):
    body = json.dumps({"data": [{"id": "gpt-b"}, {"id": "gpt-a"}, {"no_id": True}]}).encode()
    monkeypatch.setattr(ac.urllib.request, "urlopen", lambda req, timeout=0: _FakeResponse(body=body))
    assert ac.custom_api_models("http://localhost:1234/v1") == ["gpt-a", "gpt-b"]


def test_custom_api_models_empty_on_bad_json(monkeypatch):
    monkeypatch.setattr(ac.urllib.request, "urlopen", lambda req, timeout=0: _FakeResponse(body=b"not json"))
    assert ac.custom_api_models("http://localhost:1234/v1") == []


def _sse(*chunks: str) -> list[bytes]:
    lines = [f'data: {{"choices":[{{"delta":{{"content":{json.dumps(c)}}}}}]}}'.encode() for c in chunks]
    lines.append(b"data: [DONE]")
    return lines


def test_custom_api_chat_streams_tokens_and_returns_full_text(monkeypatch):
    monkeypatch.setattr(
        ac.urllib.request, "urlopen",
        lambda req, timeout=0: _FakeResponse(lines=_sse("Hel", "lo")))
    tokens = []
    full = ac.custom_api_chat("http://localhost:1234/v1", "", "local-model",
                              [{"role": "user", "content": "hi"}], tokens.append)
    assert tokens == ["Hel", "lo"]
    assert full == "Hello"


def test_custom_api_chat_ignores_non_data_and_malformed_lines(monkeypatch):
    lines = [b": comment", b"event: ping", b"data: not json"] + _sse("ok")
    monkeypatch.setattr(ac.urllib.request, "urlopen", lambda req, timeout=0: _FakeResponse(lines=lines))
    tokens = []
    full = ac.custom_api_chat("http://x/v1", "", "m", [], tokens.append)
    assert tokens == ["ok"]
    assert full == "ok"


def test_custom_api_chat_stops_cooperatively(monkeypatch):
    monkeypatch.setattr(
        ac.urllib.request, "urlopen",
        lambda req, timeout=0: _FakeResponse(lines=_sse("a", "b", "c")))
    tokens = []
    full = ac.custom_api_chat("http://x/v1", "", "m", [], tokens.append, stop=lambda: True)
    assert tokens == []
    assert full == ""


def test_custom_api_chat_sends_bearer_auth_header(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["headers"] = dict(req.header_items())
        return _FakeResponse(lines=_sse("x"))

    monkeypatch.setattr(ac.urllib.request, "urlopen", fake_urlopen)
    ac.custom_api_chat("http://x/v1", "sk-secret", "m", [], lambda t: None)
    assert captured["headers"].get("Authorization") == "Bearer sk-secret"


# ── refresh_status_async ────────────────────────────────────────────────────

def test_refresh_status_offline_is_synchronous_and_always_ok(tmp_path):
    c = ac.AssistantController(storage_dir=tmp_path)
    results = []
    thread = c.refresh_status_async(on_result=lambda ok, msg, models: results.append((ok, msg, models)))
    assert thread is None
    assert results == [(True, results[0][1], [])]
    assert "always available" in results[0][1]


def test_refresh_status_ollama_reports_reachable_and_models(tmp_path, monkeypatch):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "ollama"
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: True)
    monkeypatch.setattr("velum_core.advisor.ollama_models", lambda: ["a", "b"])
    results = []
    t = c.refresh_status_async(on_result=lambda ok, msg, models: results.append((ok, msg, models)))
    t.join(timeout=5)
    assert results == [(True, results[0][1], ["a", "b"])]
    assert "2 models" in results[0][1]


def test_refresh_status_ollama_unreachable(tmp_path, monkeypatch):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "ollama"
    monkeypatch.setattr("velum_core.advisor.ollama_available", lambda: False)
    results = []
    t = c.refresh_status_async(on_result=lambda ok, msg, models: results.append((ok, msg, models)))
    t.join(timeout=5)
    assert results == [(False, results[0][1], [])]


def test_refresh_status_custom_uses_controller_bridge_methods(tmp_path, monkeypatch):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "custom"
    monkeypatch.setattr(c, "custom_api_available", lambda: True)
    monkeypatch.setattr(c, "custom_api_models", lambda: ["local-model"])
    results = []
    t = c.refresh_status_async(on_result=lambda ok, msg, models: results.append((ok, msg, models)))
    t.join(timeout=5)
    assert results == [(True, "Connected", ["local-model"])]


# ── send_async dispatch ─────────────────────────────────────────────────────

def _diag_and_params():
    img = np.full((60, 60), 128, dtype=np.uint8)
    mask = np.zeros((60, 60), dtype=np.int32)
    params = {"points_per_side": 32, "pred_iou_thresh": 0.8, "stability_score_thresh": 0.6,
             "box_nms_thresh": 0.05, "min_mask_area": 20, "resize_size": 512}
    from velum_core import advisor
    return advisor.diagnose(img, mask, params), params


def test_send_async_offline_backend_answers_synchronously(tmp_path):
    c = ac.AssistantController(storage_dir=tmp_path)
    diag, params = _diag_and_params()
    done = []
    thread = c.send_async([], "why no cells?", diag, params,
                          on_token=lambda t: None, on_done=done.append,
                          on_error=lambda e: pytest.fail(e))
    assert thread is None
    assert len(done) == 1 and "No cells" in done[0]


def test_send_async_falls_back_to_offline_when_backend_not_ready(tmp_path):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "ollama"   # selected, but no model chosen
    diag, params = _diag_and_params()
    done = []
    thread = c.send_async([], "hi", diag, params, on_token=lambda t: None,
                          on_done=done.append, on_error=lambda e: pytest.fail(e))
    assert thread is None
    assert len(done) == 1


def test_send_async_ollama_streams_and_builds_system_prompt(tmp_path, monkeypatch):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "ollama"
    c.settings.ollama_model = "qwen2.5:7b"
    diag, params = _diag_and_params()

    captured = {}

    def fake_ollama_chat(model, messages, on_token, stop=None, temperature=0.2):
        captured["model"] = model
        captured["messages"] = messages
        on_token("Hi")
        on_token(" there")
        return "Hi there"

    monkeypatch.setattr("velum_core.advisor.ollama_chat", fake_ollama_chat)

    tokens, done = [], []
    thread = c.send_async([], "why merged?", diag, params, on_token=tokens.append,
                          on_done=done.append, on_error=lambda e: pytest.fail(e))
    assert thread is not None
    thread.join(timeout=5)
    assert tokens == ["Hi", " there"]
    assert done == ["Hi there"]
    assert captured["model"] == "qwen2.5:7b"
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][-1] == {"role": "user", "content": "why merged?"}


def test_send_async_ollama_agent_model_skips_system_prompt(tmp_path, monkeypatch):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "ollama"
    c.settings.ollama_model = f"{c.agent_model_name}:latest"
    diag, params = _diag_and_params()

    captured = {}

    def fake_ollama_chat(model, messages, on_token, stop=None, temperature=0.2):
        captured["messages"] = messages
        return "ok"

    monkeypatch.setattr("velum_core.advisor.ollama_chat", fake_ollama_chat)
    thread = c.send_async([], "why merged?", diag, params, on_token=lambda t: None,
                          on_done=lambda t: None, on_error=lambda e: pytest.fail(e))
    thread.join(timeout=5)
    assert all(m["role"] != "system" for m in captured["messages"])
    assert "Question: why merged?" in captured["messages"][-1]["content"]


def test_send_async_custom_backend_dispatches_to_module_bridge(tmp_path, monkeypatch):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "custom"
    c.settings.custom_base_url = "http://localhost:1234/v1"
    c.settings.custom_model = "local-model"
    c.settings.custom_api_key = "sk-x"
    diag, params = _diag_and_params()

    captured = {}

    def fake_custom_chat(base_url, api_key, model, messages, on_token, stop=None, temperature=0.2):
        captured.update(base_url=base_url, api_key=api_key, model=model, messages=messages)
        on_token("ok")
        return "ok"

    monkeypatch.setattr(ac, "custom_api_chat", fake_custom_chat)
    tokens, done = [], []
    thread = c.send_async([], "hello", diag, params, on_token=tokens.append,
                          on_done=done.append, on_error=lambda e: pytest.fail(e))
    thread.join(timeout=5)
    assert tokens == ["ok"] and done == ["ok"]
    assert captured["base_url"] == "http://localhost:1234/v1"
    assert captured["api_key"] == "sk-x"
    assert captured["model"] == "local-model"


def test_send_async_reports_errors_via_on_error(tmp_path, monkeypatch):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "ollama"
    c.settings.ollama_model = "qwen2.5:7b"
    diag, params = _diag_and_params()

    def fake_ollama_chat(*a, **k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr("velum_core.advisor.ollama_chat", fake_ollama_chat)
    errors = []
    thread = c.send_async([], "hi", diag, params, on_token=lambda t: None,
                          on_done=lambda t: pytest.fail("on_done should not fire"),
                          on_error=errors.append)
    thread.join(timeout=5)
    assert errors == ["connection reset"]


def test_chat_busy_true_during_send_then_false(tmp_path, monkeypatch):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "ollama"
    c.settings.ollama_model = "m"
    diag, params = _diag_and_params()
    release = threading.Event()

    def fake_ollama_chat(model, messages, on_token, stop=None, temperature=0.2):
        release.wait(timeout=5)
        return "done"

    monkeypatch.setattr("velum_core.advisor.ollama_chat", fake_ollama_chat)
    assert c.chat_busy() is False
    thread = c.send_async([], "hi", diag, params, on_token=lambda t: None,
                          on_done=lambda t: None, on_error=lambda e: None)
    assert c.chat_busy() is True
    release.set()
    thread.join(timeout=5)
    assert c.chat_busy() is False


def test_stop_chat_lets_a_running_backend_bail_out_early(tmp_path, monkeypatch):
    c = ac.AssistantController(storage_dir=tmp_path)
    c.settings.backend = "ollama"
    c.settings.ollama_model = "m"
    diag, params = _diag_and_params()
    started = threading.Event()

    def fake_ollama_chat(model, messages, on_token, stop=None, temperature=0.2):
        started.set()
        while not (stop and stop()):
            time.sleep(0.01)
        return "stopped early"

    monkeypatch.setattr("velum_core.advisor.ollama_chat", fake_ollama_chat)
    done = []
    thread = c.send_async([], "hi", diag, params, on_token=lambda t: None,
                          on_done=done.append, on_error=lambda e: None)
    started.wait(timeout=5)
    c.stop_chat()
    thread.join(timeout=5)
    assert done == ["stopped early"]


# ── Ollama model management (pull / create agent) ──────────────────────────────

def test_pull_ollama_model_async_reports_progress_and_completion(tmp_path, monkeypatch):
    c = ac.AssistantController(storage_dir=tmp_path)

    def fake_pull(name, on_progress):
        on_progress("downloading", 0.5)
        on_progress("done", 1.0)
        return True

    monkeypatch.setattr("velum_core.advisor.ollama_pull", fake_pull)
    progress, done = [], []
    thread = c.pull_ollama_model_async("llama3.2:3b", on_progress=lambda s, f: progress.append((s, f)),
                                       on_done=lambda n, ok: done.append((n, ok)))
    thread.join(timeout=5)
    assert progress == [("downloading", 0.5), ("done", 1.0)]
    assert done == [("llama3.2:3b", True)]


def test_model_op_busy_blocks_a_second_concurrent_pull(tmp_path, monkeypatch):
    c = ac.AssistantController(storage_dir=tmp_path)
    release = threading.Event()

    def fake_pull(name, on_progress):
        release.wait(timeout=5)
        return True

    monkeypatch.setattr("velum_core.advisor.ollama_pull", fake_pull)
    first = c.pull_ollama_model_async("a", on_progress=lambda s, f: None, on_done=lambda n, ok: None)
    assert c.model_op_busy() is True
    second = c.pull_ollama_model_async("b", on_progress=lambda s, f: None, on_done=lambda n, ok: None)
    assert second is None   # refused — a pull is already running
    release.set()
    first.join(timeout=5)
    assert c.model_op_busy() is False


def test_create_agent_async_reports_status_and_completion(tmp_path, monkeypatch):
    c = ac.AssistantController(storage_dir=tmp_path)

    def fake_create(base_model, on_status):
        on_status("baking…")
        return True

    monkeypatch.setattr("velum_core.advisor.ollama_create_agent", fake_create)
    statuses, done = [], []
    thread = c.create_agent_async("qwen2.5:7b", on_status=statuses.append, on_done=done.append)
    thread.join(timeout=5)
    assert statuses == ["baking…"]
    assert done == [True]


def test_recommended_models_and_agent_model_name_expose_the_real_advisor_data(tmp_path):
    c = ac.AssistantController(storage_dir=tmp_path)
    from velum_core import advisor
    assert c.recommended_models == advisor.RECOMMENDED_MODELS
    assert c.agent_model_name == advisor.AGENT_MODEL_NAME
