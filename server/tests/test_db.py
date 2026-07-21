"""Tests for server.db — schema, pragmas, transactions, concurrency."""
import threading

import pytest

from server.db import Database, SCHEMA_VERSION, apply_schema, connect


def test_schema_applied_and_versioned():
    conn = connect(":memory:")
    apply_schema(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    # all expected tables exist
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"users", "organizations", "memberships", "projects", "tasks",
            "annotations", "api_keys", "sessions", "audit_events"} <= names


def test_apply_schema_idempotent():
    conn = connect(":memory:")
    apply_schema(conn)
    apply_schema(conn)  # second call is a no-op, must not raise
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_pragmas_set():
    conn = connect(":memory:")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000


def test_wal_mode_on_file(tmp_path):
    conn = connect(str(tmp_path / "wal.db"))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_transaction_commits_and_rolls_back():
    db = Database().initialize()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO users (id, email, username, password_hash, created_at) "
            "VALUES ('u1','a@b.com','alice','h','2026-01-01T00:00:00+00:00')"
        )
    assert db.connection().execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1

    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO users (id, email, username, password_hash, created_at) "
                "VALUES ('u2','c@d.com','bob','h','2026-01-01T00:00:00+00:00')"
            )
            raise RuntimeError("boom")
    # the failed transaction's row must have rolled back
    assert db.connection().execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_foreign_key_cascade(tmp_path):
    db = Database(str(tmp_path / "fk.db")).initialize()
    with db.transaction() as conn:
        conn.execute("INSERT INTO users VALUES ('u1','a@b.com','alice','h','',1,0,'t',NULL)")
        conn.execute("INSERT INTO organizations VALUES ('o1','Org','org','u1','t')")
        conn.execute("INSERT INTO projects VALUES "
                     "('p1','o1','P','p','u1','','cellseg1','{}',0,'t','t')")
        conn.execute("INSERT INTO tasks VALUES "
                     "('t1','p1','T','u1','','pending',NULL,'t','t')")
    # deleting the org cascades to projects and tasks
    with db.transaction() as conn:
        conn.execute("DELETE FROM organizations WHERE id='o1'")
    c = db.connection()
    assert c.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_thread_local_connections_share_a_file(tmp_path):
    # Each thread gets its own connection (SQLite requires it); a file DB lets
    # them see each other's committed writes.
    db = Database(str(tmp_path / "shared.db")).initialize()
    with db.transaction() as conn:
        conn.execute("INSERT INTO users VALUES ('u1','a@b.com','alice','h','',1,0,'t',NULL)")

    seen = {}

    def worker():
        seen["count"] = db.connection().execute("SELECT COUNT(*) FROM users").fetchone()[0]

    th = threading.Thread(target=worker)
    th.start()
    th.join()
    assert seen["count"] == 1
