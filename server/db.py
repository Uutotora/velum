"""SQLite connection factory, schema, and migrations — standard library only.

Why SQLite as the *default*: it needs no server, no container, no configuration
— a working multi-user backend from a single file, which is what makes this
foundation usable before any web tier exists. Configured for concurrency, it is
genuinely adequate for a small deployment:

* **WAL journal mode** lets many readers run concurrently with one writer
  (instead of a single global lock), the single biggest throughput lever SQLite
  has.
* **``busy_timeout``** makes a writer *wait* for the lock instead of instantly
  failing with "database is locked" — i.e. requests queue briefly under a write
  burst rather than erroring, which is exactly the "nothing hangs / nothing
  falls over" behaviour we want.
* **``foreign_keys=ON``** enforces referential integrity (off by default in
  SQLite), so a delete cascades and an orphan row can't be inserted.
* **``synchronous=NORMAL``** is the safe, fast pairing with WAL.

Why it still **scales to Postgres without a rewrite**: nothing above SQL leaks
into the rest of the package. The schema uses only portable column types, the
repositories issue plain parameterised statements, and connections are handed
out one-per-thread. Moving to Postgres for 10k+ users/day is then: point the
``DATABASE_URL`` at Postgres, swap this connection factory for a pooled psycopg
one, and translate the ``?`` placeholders to ``%s`` — a driver change, not a
redesign. That boundary is the whole reason the storage layer is this thin.

Connections are **thread-local**: SQLite forbids sharing one connection across
threads, and a web server is multi-threaded, so :meth:`Database.connection`
returns a per-thread connection (which also lets an in-memory database persist
across calls within a thread — essential for the test suite).
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Bump when the schema below changes in a way that needs a migration. Stored in
# the database via ``PRAGMA user_version``; :func:`apply_schema` compares and
# upgrades. Version 0 is an empty/new database.
SCHEMA_VERSION = 1

# Default seconds a writer waits for the lock before giving up (as busy_timeout,
# in ms). 5s comfortably absorbs a write burst without a spurious "locked".
_BUSY_TIMEOUT_MS = 5000


# The schema. One statement per table, portable types only (TEXT/INTEGER), so a
# Postgres port is mechanical. Booleans are INTEGER 0/1; JSON payloads are TEXT
# (models.py decodes them). Foreign keys cascade on delete so removing an org
# tears down its projects/tasks/annotations — *except* the audit log, which
# deliberately keeps no FK to actor/org so the immutable trail survives the
# deletion of what it references.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    full_name     TEXT NOT NULL DEFAULT '',
    is_active     INTEGER NOT NULL DEFAULT 1,
    is_superuser  INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS organizations (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    slug       TEXT NOT NULL UNIQUE,
    created_by TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memberships (
    id         TEXT PRIMARY KEY,
    org_id     TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (org_id, user_id)
);
CREATE INDEX IF NOT EXISTS ix_memberships_user ON memberships(user_id);
CREATE INDEX IF NOT EXISTS ix_memberships_org  ON memberships(org_id);

CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    org_id      TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    slug        TEXT NOT NULL,
    created_by  TEXT NOT NULL REFERENCES users(id),
    description TEXT NOT NULL DEFAULT '',
    engine      TEXT NOT NULL DEFAULT 'cellseg1',
    settings    TEXT NOT NULL DEFAULT '{}',
    archived    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE (org_id, slug)
);
CREATE INDEX IF NOT EXISTS ix_projects_org ON projects(org_id);

CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    created_by  TEXT NOT NULL REFERENCES users(id),
    source      TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending',
    assignee_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tasks_project ON tasks(project_id);
CREATE INDEX IF NOT EXISTS ix_tasks_assignee ON tasks(assignee_id);
CREATE INDEX IF NOT EXISTS ix_tasks_project_status ON tasks(project_id, status);

CREATE TABLE IF NOT EXISTS annotations (
    id          TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    author_id   TEXT NOT NULL REFERENCES users(id),
    data        TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'draft',
    reviewed_by TEXT REFERENCES users(id) ON DELETE SET NULL,
    reviewed_at TEXT,
    review_note TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_annotations_task ON annotations(task_id);
CREATE INDEX IF NOT EXISTS ix_annotations_author ON annotations(author_id);

CREATE TABLE IF NOT EXISTS api_keys (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    org_id       TEXT REFERENCES organizations(id) ON DELETE CASCADE,
    name         TEXT NOT NULL DEFAULT '',
    prefix       TEXT NOT NULL,
    token_hash   TEXT NOT NULL UNIQUE,
    created_at   TEXT NOT NULL,
    last_used_at TEXT,
    expires_at   TEXT,
    revoked      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_api_keys_prefix ON api_keys(prefix);
CREATE INDEX IF NOT EXISTS ix_api_keys_user ON api_keys(user_id);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    revoked    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS audit_events (
    id          TEXT PRIMARY KEY,
    action      TEXT NOT NULL,
    actor_id    TEXT,
    org_id      TEXT,
    target_type TEXT NOT NULL DEFAULT '',
    target_id   TEXT NOT NULL DEFAULT '',
    detail      TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_audit_org ON audit_events(org_id);
CREATE INDEX IF NOT EXISTS ix_audit_actor ON audit_events(actor_id);
CREATE INDEX IF NOT EXISTS ix_audit_created ON audit_events(created_at);
"""


def configure_connection(conn: sqlite3.Connection) -> None:
    """Apply the per-connection pragmas that make SQLite safe & concurrent.

    Pragmas in SQLite are largely *per connection* (WAL mode is per database and
    sticky, but foreign_keys/busy_timeout are per connection), so every fresh
    connection must run this. ``row_factory`` gives dict-style rows that
    ``models.*.from_row`` consumes.
    """
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    # WAL + NORMAL: concurrent reads, one writer, durable. WAL is a no-op on an
    # in-memory database (it stays "memory"), which is fine for tests.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")


def connect(path: str = ":memory:", *, timeout: float = 5.0) -> sqlite3.Connection:
    """Open and configure a single SQLite connection."""
    conn = sqlite3.connect(path, timeout=timeout)
    configure_connection(conn)
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    """Create tables/indexes and stamp the schema version, idempotently.

    Uses ``PRAGMA user_version`` as the migration marker. A brand-new database
    (version 0) gets the full schema and is stamped to :data:`SCHEMA_VERSION`.
    An already-current database is left untouched. Future versions add their
    migration steps here, keyed on the stored version.
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current >= SCHEMA_VERSION:
        return
    conn.executescript(_SCHEMA)
    # executescript issues an implicit COMMIT; set the version after so the two
    # land together on the next commit.
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()


class Database:
    """A handle to one SQLite database, handing out thread-local connections.

    Construct with a file path for a real store, or the default ``":memory:"``
    for tests. Call :meth:`initialize` once to create the schema. Services do
    their reads through :meth:`connection` and their writes inside
    :meth:`transaction` (which commits on success, rolls back on any exception —
    so an action and its audit-log row are all-or-nothing).
    """

    def __init__(self, path: str | Path = ":memory:", *, timeout: float = 5.0):
        self.path = str(path)
        self.timeout = timeout
        self._local = threading.local()

    def connection(self) -> sqlite3.Connection:
        """The calling thread's connection, opened & configured on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self.path, timeout=self.timeout)
            self._local.conn = conn
        return conn

    def initialize(self) -> "Database":
        """Create the schema if absent; returns self for chaining."""
        apply_schema(self.connection())
        return self

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """A unit of work: commit on clean exit, roll back on any exception."""
        conn = self.connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        """Close the calling thread's connection (if any)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
