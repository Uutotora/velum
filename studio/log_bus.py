"""Velum — the real, central log stream.

Every tab already produces real operational log lines: the reused
``velum_core.predict_controller.PredictController``'s ``on_log``
callback (segmentation runs, batch, benchmark), ``train_model``'s progress
lines, the Assistant's backend/connection events. Before this module,
each screen only skimmed those strings for a ``[ERROR]``/``[HINT]`` toast
and threw the rest away — the Logs console itself rendered a hard-coded
``demo.LOGS`` transcript with zero binding to any of it.

This is the one real sink they all feed instead: a bounded, thread-safe
ring buffer of :class:`LogRecord`, plus a stdlib :mod:`logging` handler so an
ordinary ``logging.getLogger(__name__).info(...)`` call anywhere in the
process — Studio's own controllers, the reused ML core, even a third-party
dependency — reaches it too, exactly like a real desktop app's log panel,
not just the hand-picked call sites that remember to invoke a bespoke
callback.

Qt-free (stdlib only: ``logging``/``threading``/``time``/``collections``) so
it stays importable in CI's light ``test`` group. ``studio.overlays.
LogsConsole`` is the one real (Qt) subscriber today; nothing here imports Qt.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

# These *are* the real `logging` levels (10/20/30/40/50), not a parallel
# scheme -- re-exported so callers can write `log_bus.WARNING` without also
# `import logging`, while a plain `logging.getLogger(__name__).warning(...)`
# call anywhere else in the process lands on the exact same scale via
# StudioLogHandler below.
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

_LEVELS = (DEBUG, INFO, WARNING, ERROR, CRITICAL)


def level_name(levelno: int) -> str:
    """Canonical name for ``levelno``, flooring to the nearest known level.

    A caller can log at a custom in-between level (``logger.log(25, ...)``);
    plain ``logging.getLevelName`` would just say ``"Level 25"``, which is a
    poor fit for a coloured badge in the console.
    """
    name = "DEBUG"
    for lv in _LEVELS:
        if levelno >= lv:
            name = logging.getLevelName(lv)
    return name


def short_source(source: str) -> str:
    """The console's compact source tag: ``"studio.segment"`` -> ``"segment"``.

    Real logger names stay properly dotted everywhere else (anyone piping
    Python's own ``logging`` output elsewhere still sees ``studio.segment``);
    this is purely a display trim for the Logs console's narrow source column.
    """
    return source[len("studio."):] if source.startswith("studio.") else source


@dataclass(frozen=True)
class LogRecord:
    seq: int
    ts: float           # time.time() epoch seconds
    level: int          # a stdlib logging level number
    source: str         # origin tag, e.g. "studio.segment" or a logger name
    message: str

    @property
    def level_name(self) -> str:
        return level_name(self.level)


class LogBus:
    """A bounded, thread-safe ring buffer of :class:`LogRecord`.

    Producers call :meth:`emit` (or the level convenience methods) from any
    thread — a predict/training worker, a urllib SSE thread for the
    Assistant's Custom-API backend — concurrently. Consumers subscribe with
    a plain callback (no Qt/psygnal dependency here, same convention as
    ``layer_model.LayerList``'s events); a Qt consumer like
    ``overlays.LogsConsole`` is responsible for marshalling that callback
    onto its own thread (the established ``_safe_emit_*`` + ``pyqtSignal``
    pattern used throughout Studio).
    """

    def __init__(self, maxlen: int = 4000):
        self._lock = threading.Lock()
        self._records: deque[LogRecord] = deque(maxlen=maxlen)
        self._subscribers: list[Callable[[LogRecord], None]] = []
        self._seq = 0

    def emit(self, level: int, message: str, source: str = "studio") -> LogRecord:
        with self._lock:
            self._seq += 1
            rec = LogRecord(seq=self._seq, ts=time.time(), level=level,
                             source=source, message=message)
            self._records.append(rec)
            subscribers = list(self._subscribers)
        for cb in subscribers:
            cb(rec)
        return rec

    def debug(self, message: str, source: str = "studio") -> LogRecord:
        return self.emit(DEBUG, message, source)

    def info(self, message: str, source: str = "studio") -> LogRecord:
        return self.emit(INFO, message, source)

    def warning(self, message: str, source: str = "studio") -> LogRecord:
        return self.emit(WARNING, message, source)

    def error(self, message: str, source: str = "studio") -> LogRecord:
        return self.emit(ERROR, message, source)

    def critical(self, message: str, source: str = "studio") -> LogRecord:
        return self.emit(CRITICAL, message, source)

    def subscribe(
        self, callback: Callable[[LogRecord], None]
    ) -> tuple[list[LogRecord], Callable[[], None]]:
        """Register ``callback`` for every future record.

        Returns ``(backlog, unsubscribe)`` — ``backlog`` is an atomic
        snapshot of every record already emitted, taken under the same lock
        as registration so a record can never be double-delivered or
        skipped across the join point (the race a separate ``snapshot()`` +
        ``subscribe()`` pair would have). A freshly-opened LogsConsole uses
        ``backlog`` to show history from before it existed.
        """
        with self._lock:
            backlog = list(self._records)
            self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return backlog, _unsubscribe

    def snapshot(self) -> list[LogRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


class StudioLogHandler(logging.Handler):
    """Bridges the stdlib ``logging`` module into a :class:`LogBus`.

    So an ordinary ``logging.getLogger(__name__).info(...)`` call anywhere
    in the process reaches the Logs console like a real desktop app's log
    panel — not just the hand-picked call sites that remember to invoke a
    bespoke ``on_log`` callback.
    """

    def __init__(self, bus: LogBus):
        super().__init__(level=logging.DEBUG)
        self.bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self.bus.emit(record.levelno, message, source=record.name)
        except Exception:
            self.handleError(record)


_PREFIX_LEVELS = (
    ("[ERROR]", ERROR),
    ("[WARN]", WARNING),
    ("[HINT]", INFO),
    ("[INFO]", INFO),
)


def emit_prefixed(bus: LogBus, message: str, source: str) -> LogRecord:
    """Route one of the ML core's existing ``on_log(msg)`` strings.

    ``velum_core.predict_controller``/``train_model`` (reused unmodified
    by ``segment_controller``/``train_controller``) already format lines with
    a fixed, code-authored ``[ERROR]``/``[WARN]``/``[HINT]``/``[INFO]``
    prefix convention (checked case-sensitively on purpose — this is not
    user text). This maps that convention onto the bus's real severity levels
    instead of everything defaulting to INFO, and strips the now-redundant
    bracket (the console renders its own coloured level badge per line).
    Anything without a recognised prefix (``▶``/``✓``/``■`` progress lines,
    plain text) is stored as-is at INFO.
    """
    text = message.strip()
    level = INFO
    for prefix, lv in _PREFIX_LEVELS:
        if text.startswith(prefix):
            level = lv
            text = text[len(prefix):].lstrip()
            break
    return bus.emit(level, text, source=source)


_default_bus = LogBus()


def get_log_bus() -> LogBus:
    """The one process-wide bus every controller/handler feeds and every
    LogsConsole reads by default. Tests should construct their own private
    ``LogBus()`` instead, for isolation — this singleton is just the
    real-app default.
    """
    return _default_bus


def install_handler(bus: Optional[LogBus] = None, *, logger: Optional[logging.Logger] = None,
                     studio_level: int = logging.DEBUG) -> LogBus:
    """Attach a :class:`StudioLogHandler` feeding ``bus`` to ``logger``.

    Defaults to the root logger, so every module in the process is
    captured, and is idempotent per ``(logger, bus)`` pair — safe to call
    from every ``StudioWindow.__init__`` (every construction path, real app
    or test) without ever attaching a duplicate handler.

    Also raises ``logger``'s own effective level to at least INFO if it's
    currently less verbose (Python's root logger defaults to WARNING, which
    would otherwise silently swallow every ``.info()``/``.debug()`` call in
    this process before it ever reached a handler) — but never *lowers* a
    level some other part of the app already set more verbose. Third-party
    libraries are left at whatever level they already log at; filtering
    *for display* is the Logs console's own level filter's job, not
    capture's. Studio's own ``"studio"`` logger namespace is always set to
    ``studio_level`` (DEBUG by default) regardless, so Studio's own
    ``.debug()`` breadcrumbs always reach the bus even if root stays at INFO.
    """
    # Deliberately `is None`, not `bus or get_log_bus()` -- LogBus defines
    # __len__ for the convenience of `len(bus)`, which means a freshly
    # constructed (empty) bus is falsy in Python's truthiness rules and a
    # plain `or` would silently discard a real, intentionally-passed-in
    # empty test bus in favour of the global singleton.
    bus = bus if bus is not None else get_log_bus()
    target = logger if logger is not None else logging.getLogger()
    if not any(isinstance(h, StudioLogHandler) and h.bus is bus for h in target.handlers):
        handler = StudioLogHandler(bus)
        handler.setFormatter(logging.Formatter("%(message)s"))
        target.addHandler(handler)
    if target.level == logging.NOTSET or target.level > logging.INFO:
        target.setLevel(logging.INFO)
    logging.getLogger("studio").setLevel(studio_level)
    return bus
