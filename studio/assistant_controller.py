"""Velum — the Assistant tab's controller.

Qt-free glue (mirrors ``train_controller.py`` / ``segment_controller.py``:
plain data in, plain callbacks out, background threads for anything that
touches the network) behind three interchangeable chat backends:

  * **"offline"** — the deterministic diagnostic engine
    (``velum_core.advisor.diagnose``). Always available, no model, no
    network call — reused read-only, imported lazily, exactly like
    ``segment_controller.py`` reuses ``predict_controller`` without ever
    modifying it.
  * **"ollama"** — a locally running Ollama server. Reuses
    ``velum_core.advisor``'s existing Ollama bridge verbatim (model
    discovery, pull, the task-specialised "bake an agent" flow, streaming
    chat) — again read-only.
  * **"custom"** — any OpenAI-compatible HTTP endpoint: local (LM Studio,
    vLLM, llama.cpp's server, text-generation-webui, …) or remote (OpenAI
    itself, OpenRouter, Groq, …), with or without an API key. This is new
    capability the classic app doesn't have, so it's Studio's own — the
    bridge functions below (``custom_api_available``/``custom_api_models``/
    ``custom_api_chat``), stdlib ``urllib`` only, no new dependency, same
    idiom as ``velum_core.advisor.ollama_chat``.

Settings (which backend, which model, the custom endpoint + key) persist to
one small JSON file under the shared storage dir — a machine-level choice,
not a per-project one, so it doesn't belong in ``ProjectSettings``.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

BACKENDS = ("offline", "ollama", "custom")
BACKEND_LABELS = {"offline": "Offline", "ollama": "Ollama", "custom": "Custom API"}


# ── Custom-API (OpenAI-compatible) bridge — Studio's own, stdlib only ────────

def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _custom_api_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def custom_api_available(base_url: str, api_key: str = "", timeout: float = 2.5) -> bool:
    """Best-effort reachability check via ``GET {base_url}/models``.

    Non-authoritative: some minimal OpenAI-compatible servers don't
    implement model listing even though ``/chat/completions`` works fine —
    callers should treat a ``False`` here as "couldn't confirm", not "chat
    will fail" (:meth:`AssistantController.backend_ready` never depends on
    this passing).
    """
    url = _normalize_base_url(base_url)
    if not url:
        return False
    req = urllib.request.Request(f"{url}/models", headers=_custom_api_headers(api_key))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def custom_api_models(base_url: str, api_key: str = "", timeout: float = 4.0) -> list[str]:
    """Model ids from ``GET {base_url}/models`` (the OpenAI list-models
    shape: ``{"data": [{"id": …}, …]}``). Empty on any failure — the model
    field always stays freely editable so an endpoint that can't list
    models is still usable by typing its id directly."""
    url = _normalize_base_url(base_url)
    if not url:
        return []
    req = urllib.request.Request(f"{url}/models", headers=_custom_api_headers(api_key))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return sorted({m["id"] for m in data.get("data", []) if "id" in m})
    except Exception:
        return []


def custom_api_chat(base_url: str, api_key: str, model: str, messages: list[dict[str, str]],
                    on_token: Callable[[str], None], stop: Callable[[], bool] | None = None,
                    temperature: float = 0.2, timeout: float = 120.0) -> str:
    """Stream a chat completion from an OpenAI-compatible ``/chat/completions``
    endpoint. Returns the full text — the Custom-API sibling of
    ``velum_core.advisor.ollama_chat``, same contract (streams tokens via
    ``on_token``, cooperatively cancellable via ``stop``, propagates
    exceptions to the caller rather than swallowing them).

    Server-Sent-Events framing (``data: {...}`` lines, terminated by
    ``data: [DONE]``) — the de facto standard this whole family of servers
    (OpenAI, LM Studio, vLLM, llama.cpp, OpenRouter, …) implements.
    """
    url = f"{_normalize_base_url(base_url)}/chat/completions"
    payload = json.dumps({
        "model": model, "messages": messages, "stream": True,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=_custom_api_headers(api_key))
    full: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if stop and stop():
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or [{}]
            chunk = (choices[0].get("delta", {}) or {}).get("content", "") or ""
            if chunk:
                full.append(chunk)
                on_token(chunk)
    return "".join(full)


# ── Settings (persisted, machine-level) ──────────────────────────────────────

@dataclass
class AssistantSettings:
    backend: str = "offline"           # offline | ollama | custom
    ollama_model: str = ""
    custom_base_url: str = ""
    custom_api_key: str = ""
    custom_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssistantSettings":
        known = {"backend", "ollama_model", "custom_base_url", "custom_api_key", "custom_model"}
        settings = cls(**{k: v for k, v in (data or {}).items() if k in known})
        if settings.backend not in BACKENDS:
            settings.backend = "offline"
        return settings


class AssistantSettingsStore:
    """A single small JSON file — global, not per-project (which model
    you've connected is a machine-level choice, not something that should
    vary opening one project vs. another). Mirrors ``ProjectStore.save``'s
    atomic (temp file + replace) write."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def load(self) -> AssistantSettings:
        try:
            return AssistantSettings.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return AssistantSettings()

    def save(self, settings: AssistantSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self.path)


def default_settings_path() -> Path:
    """The conventional path, shared with the rest of Studio's storage.

    Imported lazily so this module has no hard import-time dependency on
    ``project_root`` — mirrors ``train_controller.default_storage_dir``.
    """
    from studio.train_controller import default_storage_dir
    return default_storage_dir() / "studio_assistant_settings.json"


# ── Controller ────────────────────────────────────────────────────────────

class AssistantController:
    """Owns Assistant settings + backend dispatch — independent of Qt (see
    module docstring for the three backends)."""

    def __init__(self, storage_dir: Optional[Path | str] = None,
                 settings_store: Optional[AssistantSettingsStore] = None):
        if settings_store is not None:
            self.settings_store = settings_store
        elif storage_dir is not None:
            self.settings_store = AssistantSettingsStore(Path(storage_dir) / "studio_assistant_settings.json")
        else:
            self.settings_store = AssistantSettingsStore(default_settings_path())
        self.settings: AssistantSettings = self.settings_store.load()
        self._chat_thread: Optional[threading.Thread] = None
        self._stop_chat = threading.Event()
        self._model_thread: Optional[threading.Thread] = None

    def save_settings(self) -> None:
        self.settings_store.save(self.settings)

    # ── diagnostics (reuses velum_core.advisor's engine, read-only) ─────────
    @staticmethod
    def diagnose(image, mask, params: dict) -> dict:
        from velum_core import advisor
        return advisor.diagnose(image, mask, params)

    @staticmethod
    def findings_to_text(diag: dict) -> str:
        from velum_core import advisor
        return advisor.findings_to_text(diag)

    @staticmethod
    def merge_changes(diag: dict) -> dict[str, Any]:
        """Every finding's suggested changes, merged into one dict (later
        findings win on key collisions) — what "Apply" on a whole diagnosis
        (rather than one finding's own card) would send."""
        out: dict[str, Any] = {}
        for f in diag.get("findings", []):
            out.update(f.changes)
        return out

    @staticmethod
    def parse_suggestions(text: str) -> dict[str, Any]:
        from velum_core import advisor
        return advisor.parse_suggestions(text)

    # ── backend readiness (cheap, local — no network) ────────────────────────
    def backend_ready(self) -> bool:
        """Whether the *configured* backend has enough info to actually try
        a chat call right now. ``send_async`` falls back to the offline
        diagnostic reply whenever this is False, so a half-configured
        backend degrades gracefully instead of erroring."""
        if self.settings.backend == "ollama":
            return bool(self.settings.ollama_model.strip())
        if self.settings.backend == "custom":
            return bool(self.settings.custom_base_url.strip() and self.settings.custom_model.strip())
        return False

    # ── Ollama model management (reuses velum_core.advisor verbatim) ────────
    @staticmethod
    def ollama_available() -> bool:
        from velum_core import advisor
        return advisor.ollama_available()

    @staticmethod
    def ollama_models() -> list[str]:
        from velum_core import advisor
        return advisor.ollama_models()

    @property
    def recommended_models(self) -> list[dict]:
        from velum_core import advisor
        return advisor.RECOMMENDED_MODELS

    @property
    def agent_model_name(self) -> str:
        from velum_core import advisor
        return advisor.AGENT_MODEL_NAME

    def model_op_busy(self) -> bool:
        return self._model_thread is not None and self._model_thread.is_alive()

    def pull_ollama_model_async(self, name: str, *, on_progress: Callable[[str, float], None],
                                on_done: Callable[[str, bool], None]) -> Optional[threading.Thread]:
        if self.model_op_busy():
            return None
        from velum_core import advisor

        def run():
            ok = advisor.ollama_pull(name, on_progress)
            on_done(name, ok)

        t = threading.Thread(target=run, daemon=True)
        self._model_thread = t
        t.start()
        return t

    def create_agent_async(self, base_model: str, *, on_status: Callable[[str], None],
                           on_done: Callable[[bool], None]) -> Optional[threading.Thread]:
        if self.model_op_busy():
            return None
        from velum_core import advisor

        def run():
            ok = advisor.ollama_create_agent(base_model, on_status)
            on_done(ok)

        t = threading.Thread(target=run, daemon=True)
        self._model_thread = t
        t.start()
        return t

    # ── Custom API management (this module's own bridge) ────────────────────
    def custom_api_available(self) -> bool:
        return custom_api_available(self.settings.custom_base_url, self.settings.custom_api_key)

    def custom_api_models(self) -> list[str]:
        return custom_api_models(self.settings.custom_base_url, self.settings.custom_api_key)

    # ── live status check (whichever backend is currently selected) ─────────
    def refresh_status_async(
        self, *, on_result: Callable[[bool, str, list[str]], None]
    ) -> Optional[threading.Thread]:
        """Check the configured backend's live reachability off the UI
        thread. ``on_result(ok, message, models)`` — ``models`` is the
        freshly-listed set (Ollama or Custom API), empty for "offline" or on
        failure. Returns ``None`` (no thread — nothing to wait on) for the
        synchronous "offline" case."""
        backend = self.settings.backend
        if backend == "offline":
            on_result(True, "Built-in diagnostics — always available, no model needed.", [])
            return None

        def run():
            if backend == "ollama":
                ok = self.ollama_available()
                models = self.ollama_models() if ok else []
                n = len(models)
                msg = (f"Connected — {n} model{'s' if n != 1 else ''} on this machine" if ok
                       else "Ollama isn't reachable — install/start it from ollama.com")
            else:
                ok = self.custom_api_available()
                models = self.custom_api_models() if ok else []
                msg = ("Connected" if ok else
                       "Could not reach this endpoint (some servers don't expose a model "
                       "list — chat may still work if the URL and model id are correct)")
            on_result(ok, msg, models)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    # ── chat ─────────────────────────────────────────────────────────────────
    def chat_busy(self) -> bool:
        return self._chat_thread is not None and self._chat_thread.is_alive()

    def send_async(
        self, history: list[dict[str, str]], user_text: str, diag: dict, params: dict, *,
        on_token: Callable[[str], None], on_done: Callable[[str], None],
        on_error: Callable[[str], None],
    ) -> Optional[threading.Thread]:
        """Dispatch one chat turn to the configured backend.

        Returns the started background thread, or ``None`` when the
        "offline" backend (or an unconfigured one, via :meth:`backend_ready`)
        answered synchronously via ``on_done`` — there's no model to wait on,
        so the caller (the UI) tells the two cases apart by whether it gets a
        thread back, rather than needing its own offline/online branch.
        """
        from velum_core import advisor

        if self.settings.backend == "offline" or not self.backend_ready():
            on_done(advisor.findings_to_text(diag))
            return None

        self._stop_chat.clear()
        stop = self._stop_chat.is_set

        if self.settings.backend == "ollama":
            model = self.settings.ollama_model
            is_agent = model.startswith(advisor.AGENT_MODEL_NAME)
            if is_agent:
                # Persona is baked into the model; feed only the live context
                # (mirrors the classic AssistantWidget._send()'s same split).
                live = advisor.build_live_message(diag, params)
                messages = history + [{"role": "user", "content": f"{live}\n\nQuestion: {user_text}"}]
            else:
                system = advisor.build_context_prompt(diag, params)
                messages = [{"role": "system", "content": system}] + history + [
                    {"role": "user", "content": user_text}]

            def run():
                try:
                    full = advisor.ollama_chat(model, messages, on_token, stop=stop)
                    on_done(full)
                except Exception as e:
                    on_error(str(e))
        else:  # "custom"
            system = advisor.build_context_prompt(diag, params)
            messages = [{"role": "system", "content": system}] + history + [
                {"role": "user", "content": user_text}]

            def run():
                try:
                    full = custom_api_chat(
                        self.settings.custom_base_url, self.settings.custom_api_key,
                        self.settings.custom_model, messages, on_token, stop=stop)
                    on_done(full)
                except Exception as e:
                    on_error(str(e))

        t = threading.Thread(target=run, daemon=True)
        self._chat_thread = t
        t.start()
        return t

    def stop_chat(self) -> None:
        self._stop_chat.set()
