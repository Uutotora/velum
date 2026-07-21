"""Velum — the Assistant's controller.

Qt-free glue (mirrors ``train_controller.py`` / ``segment_controller.py``:
plain data in, plain callbacks out, background threads for anything that
touches the network) behind a set of interchangeable, *named* chat
providers. Every provider is one of three kinds:

  * **offline** — the deterministic diagnostic engine
    (``velum_core.advisor.diagnose``). Always available, no model, no
    network call — reused read-only, imported lazily, exactly like
    ``segment_controller.py`` reuses ``predict_controller`` without ever
    modifying it.
  * **ollama** — a locally running Ollama server. Reuses
    ``velum_core.advisor``'s existing Ollama bridge verbatim (model
    discovery, pull, the task-specialised "bake an agent" flow, streaming
    chat) — again read-only.
  * **openai** — any OpenAI-compatible HTTP ``/chat/completions`` endpoint.
    The named presets (OpenAI, OpenRouter, Groq, LM Studio, …) all share
    this one bridge — they differ only in a pre-filled base URL, whether a
    key is required, and the setup copy; a "Custom endpoint" preset lets the
    user type any base URL. stdlib ``urllib`` only, no new dependency, same
    idiom as ``velum_core.advisor.ollama_chat``.

The provider a user has chosen (``active``), each provider's own remembered
API key + model, and the custom endpoint URL persist to one small JSON file
under the shared storage dir — a machine-level choice, not a per-project
one, so it doesn't belong in ``ProjectSettings``. The Settings screen
(``studio/settings_screen.py``) is where all of this is configured; the
Assistant drawer is a pure chat surface that just reads the active provider.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ── Provider registry (declarative — the Settings screen renders from this) ───

@dataclass(frozen=True)
class ProviderSpec:
    """One selectable chat provider.

    ``kind`` drives dispatch: ``offline`` (diagnostics), ``ollama`` (the
    advisor bridge), or ``openai`` (the OpenAI-compatible bridge below). For
    ``openai`` providers ``base_url`` is the fixed endpoint (empty +
    ``editable_url`` for the user-typed "custom" one); ``needs_key`` gates
    whether an API key is required for :meth:`AssistantController.backend_ready`.
    Everything else is display/help copy the Settings screen renders.
    """
    id: str
    label: str
    kind: str                       # "offline" | "ollama" | "openai"
    base_url: str = ""
    needs_key: bool = False
    editable_url: bool = False
    local: bool = False             # runs on this machine (privacy badge)
    key_hint: str = ""              # API-key field placeholder
    docs_url: str = ""
    default_model: str = ""
    tagline: str = ""               # one-line description
    steps: tuple[str, ...] = ()     # numbered setup instructions
    model_hints: tuple[str, ...] = ()  # example model ids for the picker


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        "offline", "Offline diagnostics", "offline", local=True,
        tagline="Built-in, deterministic advice — no model, no network, always on.",
        steps=("Nothing to set up. Ask a question or hit Diagnose any time.",),
    ),
    ProviderSpec(
        "ollama", "Ollama", "ollama", local=True, docs_url="https://ollama.com",
        tagline="Run open models locally. Private, free, no API key.",
        steps=(
            "Install Ollama from ollama.com and launch it.",
            "It serves on http://localhost:11434 automatically — no config.",
            "Download a model below (or run `ollama pull llama3.2`), then pick it.",
        ),
    ),
    ProviderSpec(
        "openai", "OpenAI", "openai", base_url="https://api.openai.com/v1",
        needs_key=True, key_hint="sk-…", docs_url="https://platform.openai.com/api-keys",
        default_model="gpt-4o-mini",
        tagline="GPT-4o-class models via your OpenAI API key.",
        steps=(
            "Create a key at platform.openai.com/api-keys.",
            "Paste it below — it is stored only on this machine, never sent anywhere else.",
        ),
        model_hints=("gpt-4o-mini", "gpt-4o"),
    ),
    ProviderSpec(
        "openrouter", "OpenRouter", "openai", base_url="https://openrouter.ai/api/v1",
        needs_key=True, key_hint="sk-or-…", docs_url="https://openrouter.ai/keys",
        default_model="openai/gpt-4o-mini",
        tagline="One key, hundreds of models from every major lab.",
        steps=(
            "Create a key at openrouter.ai/keys.",
            "Paste it below, then pick a model (Test connection lists them).",
        ),
        model_hints=("openai/gpt-4o-mini", "anthropic/claude-3.7-sonnet",
                     "meta-llama/llama-3.1-70b-instruct"),
    ),
    ProviderSpec(
        "groq", "Groq", "openai", base_url="https://api.groq.com/openai/v1",
        needs_key=True, key_hint="gsk-…", docs_url="https://console.groq.com/keys",
        default_model="llama-3.3-70b-versatile",
        tagline="Extremely fast inference on open models.",
        steps=(
            "Create a key at console.groq.com/keys.",
            "Paste it below, then pick a model.",
        ),
        model_hints=("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
    ),
    ProviderSpec(
        "lmstudio", "LM Studio", "openai", base_url="http://localhost:1234/v1",
        local=True, docs_url="https://lmstudio.ai",
        tagline="A local OpenAI-compatible server with a friendly desktop UI.",
        steps=(
            "Install LM Studio from lmstudio.ai and download a model in-app.",
            "Open the Developer tab and Start Server.",
            "It serves on http://localhost:1234/v1 — no key needed.",
        ),
    ),
    ProviderSpec(
        "custom", "Custom endpoint", "openai", base_url="", editable_url=True,
        tagline="Any OpenAI-compatible server: vLLM, llama.cpp, a gateway…",
        steps=(
            "Enter the base URL (usually ending in /v1).",
            "Add an API key only if the server requires one, then pick a model.",
        ),
    ),
)

PROVIDER_BY_ID: dict[str, ProviderSpec] = {p.id: p for p in PROVIDERS}
PROVIDER_IDS: tuple[str, ...] = tuple(p.id for p in PROVIDERS)


def provider(provider_id: str) -> ProviderSpec:
    """The :class:`ProviderSpec` for ``provider_id``, falling back to the
    always-safe offline provider for an unknown id."""
    return PROVIDER_BY_ID.get(provider_id, PROVIDER_BY_ID["offline"])


# Back-compat aliases (the Assistant drawer / older call sites imported
# these two names). Providers are the real model now.
BACKENDS = PROVIDER_IDS
BACKEND_LABELS = {p.id: p.label for p in PROVIDERS}


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
    """Which provider is active plus each provider's remembered API key and
    model — so switching provider in Settings never loses a previously
    entered key. ``custom_base_url`` is the only per-provider *URL* the user
    can set (the "custom" provider); every other openai-kind provider's URL
    is fixed by its :class:`ProviderSpec`.
    """
    active: str = "offline"                             # a ProviderSpec id
    ollama_model: str = ""
    custom_base_url: str = ""
    keys: dict[str, str] = field(default_factory=dict)      # provider id -> API key
    models: dict[str, str] = field(default_factory=dict)    # provider id -> model id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssistantSettings":
        data = dict(data or {})
        # Migrate the pre-provider format (backend + flat custom_* fields) so
        # an existing install keeps its endpoint/key on upgrade.
        if "active" not in data and "backend" in data:
            data["active"] = data.get("backend")
            if data.get("custom_api_key"):
                data.setdefault("keys", {})["custom"] = data["custom_api_key"]
            if data.get("custom_model"):
                data.setdefault("models", {})["custom"] = data["custom_model"]
        known = {"active", "ollama_model", "custom_base_url", "keys", "models"}
        clean = {k: v for k, v in data.items() if k in known}
        clean["keys"] = {str(k): str(v) for k, v in (clean.get("keys") or {}).items()}
        clean["models"] = {str(k): str(v) for k, v in (clean.get("models") or {}).items()}
        settings = cls(**clean)
        if settings.active not in PROVIDER_IDS:
            settings.active = "offline"
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

    # ── provider resolution (any provider by id, or the active one) ──────────
    def active_provider(self) -> ProviderSpec:
        return provider(self.settings.active)

    def active_kind(self) -> str:
        return self.active_provider().kind

    def base_url_for(self, provider_id: str) -> str:
        """The base URL to talk to for ``provider_id`` — the user-typed one
        for the editable "custom" provider, the spec's fixed URL otherwise."""
        spec = provider(provider_id)
        return self.settings.custom_base_url if spec.editable_url else spec.base_url

    def key_for(self, provider_id: str) -> str:
        return self.settings.keys.get(provider_id, "")

    def model_for(self, provider_id: str) -> str:
        """The chosen model for ``provider_id``: the shared Ollama field for
        Ollama, else the per-provider remembered model (or the spec's default
        when the user hasn't picked one yet)."""
        spec = provider(provider_id)
        if spec.kind == "ollama":
            return self.settings.ollama_model
        return self.settings.models.get(spec.id, "") or spec.default_model

    def resolved_base_url(self) -> str:
        return self.base_url_for(self.settings.active)

    def resolved_key(self) -> str:
        return self.key_for(self.settings.active)

    def resolved_model(self) -> str:
        return self.model_for(self.settings.active)

    def set_active(self, provider_id: str) -> None:
        self.settings.active = provider_id if provider_id in PROVIDER_IDS else "offline"
        self.save_settings()

    def set_key(self, provider_id: str, key: str) -> None:
        self.settings.keys[provider_id] = key

    def set_model(self, provider_id: str, model: str) -> None:
        self.settings.models[provider_id] = model

    def provider_ready(self, provider_id: str) -> bool:
        """Same readiness rule as :meth:`backend_ready`, for any provider —
        the Settings screen uses it to show a per-card "configured" tick."""
        spec = provider(provider_id)
        if spec.kind == "ollama":
            return bool(self.settings.ollama_model.strip())
        if spec.kind == "openai":
            if not self.base_url_for(provider_id).strip() or not self.model_for(provider_id).strip():
                return False
            return bool(self.key_for(provider_id).strip()) if spec.needs_key else True
        return False

    def check_provider_async(
        self, provider_id: str, *, on_result: Callable[[bool, str, list[str]], None]
    ) -> Optional[threading.Thread]:
        """Live reachability check for a *specific* provider (not necessarily
        the active one), so each Settings card can Test its own endpoint.
        ``on_result(ok, message, models)`` — models empty for offline/failure.
        Returns ``None`` for the synchronous offline case."""
        spec = provider(provider_id)
        if spec.kind == "offline":
            on_result(True, "Built-in diagnostics — always available, no model needed.", [])
            return None

        def run():
            if spec.kind == "ollama":
                ok = self.ollama_available()
                models = self.ollama_models() if ok else []
                n = len(models)
                msg = (f"Connected — {n} model{'s' if n != 1 else ''} on this machine" if ok
                       else "Ollama isn't reachable — install/start it from ollama.com")
            else:
                base_url, key = self.base_url_for(provider_id), self.key_for(provider_id)
                ok = custom_api_available(base_url, key)
                models = custom_api_models(base_url, key) if ok else []
                msg = ("Connected" if ok else
                       "Could not reach this endpoint (some servers don't expose a model "
                       "list — chat may still work if the URL and model id are correct)")
            on_result(ok, msg, models)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

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
        """Whether the *active* provider has enough info to actually try a
        chat call right now. ``send_async`` falls back to the offline
        diagnostic reply whenever this is False, so a half-configured
        provider degrades gracefully instead of erroring."""
        kind = self.active_kind()
        if kind == "ollama":
            return bool(self.settings.ollama_model.strip())
        if kind == "openai":
            spec = self.active_provider()
            if not self.resolved_base_url().strip() or not self.resolved_model().strip():
                return False
            return bool(self.resolved_key().strip()) if spec.needs_key else True
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

    # ── OpenAI-compatible management (this module's own bridge) ─────────────
    # Resolve against the *active* provider, so these work for every
    # openai-kind preset (OpenAI/OpenRouter/Groq/LM Studio/custom), not just
    # the user-typed "custom" endpoint.
    def custom_api_available(self) -> bool:
        return custom_api_available(self.resolved_base_url(), self.resolved_key())

    def custom_api_models(self) -> list[str]:
        return custom_api_models(self.resolved_base_url(), self.resolved_key())

    # ── live status check (whichever backend is currently selected) ─────────
    def refresh_status_async(
        self, *, on_result: Callable[[bool, str, list[str]], None]
    ) -> Optional[threading.Thread]:
        """Check the configured backend's live reachability off the UI
        thread. ``on_result(ok, message, models)`` — ``models`` is the
        freshly-listed set (Ollama or Custom API), empty for "offline" or on
        failure. Returns ``None`` (no thread — nothing to wait on) for the
        synchronous "offline" case."""
        kind = self.active_kind()
        if kind == "offline":
            on_result(True, "Built-in diagnostics — always available, no model needed.", [])
            return None

        def run():
            if kind == "ollama":
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

        kind = self.active_kind()
        if kind == "offline" or not self.backend_ready():
            on_done(advisor.findings_to_text(diag))
            return None

        self._stop_chat.clear()
        stop = self._stop_chat.is_set

        if kind == "ollama":
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
        else:  # openai-compatible (OpenAI / OpenRouter / Groq / LM Studio / custom)
            system = advisor.build_context_prompt(diag, params)
            messages = [{"role": "system", "content": system}] + history + [
                {"role": "user", "content": user_text}]
            base_url, key, model = self.resolved_base_url(), self.resolved_key(), self.resolved_model()

            def run():
                try:
                    full = custom_api_chat(base_url, key, model, messages, on_token, stop=stop)
                    on_done(full)
                except Exception as e:
                    on_error(str(e))

        t = threading.Thread(target=run, daemon=True)
        self._chat_thread = t
        t.start()
        return t

    def stop_chat(self) -> None:
        self._stop_chat.set()
