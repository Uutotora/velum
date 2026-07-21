"""Data access — one repository per entity, plain parameterised SQL.

Each repository wraps a live ``sqlite3.Connection`` and turns rows into the
:mod:`server.models` dataclasses (and back). This is the *only* layer that
speaks SQL; the services above it never see a cursor. Keeping it this thin is
what makes the eventual Postgres port mechanical — the statements are standard,
the only SQLite-ism is the ``?`` placeholder style.

Repositories **do not** commit — the caller's :meth:`Database.transaction`
owns the transaction boundary, so an action and its audit row commit together
or not at all. Reads return ``None``/``[]`` for "not there"; raising "not
found" as a user-facing error is the service layer's job (it has the context to
choose 404 vs 403).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from .models import (
    Annotation,
    ApiKey,
    AuditEvent,
    Membership,
    Organization,
    Project,
    Session,
    Task,
    User,
)
from .rbac import Role


# ── small helpers ────────────────────────────────────────────────────────────
def _insert(conn: sqlite3.Connection, table: str, values: dict[str, Any]) -> None:
    cols = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    conn.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
        tuple(values.values()),
    )


def _b(value: bool) -> int:
    return 1 if value else 0


def _j(value: Any) -> str:
    return json.dumps(value or {})


# ── users ────────────────────────────────────────────────────────────────────
class UserRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, user: User) -> User:
        _insert(self.conn, "users", {
            "id": user.id, "email": user.email, "username": user.username,
            "password_hash": user.password_hash, "full_name": user.full_name,
            "is_active": _b(user.is_active), "is_superuser": _b(user.is_superuser),
            "created_at": user.created_at, "last_login_at": user.last_login_at,
        })
        return user

    def get(self, user_id: str) -> Optional[User]:
        row = self.conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return User.from_row(row) if row else None

    def get_by_email(self, email: str) -> Optional[User]:
        row = self.conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return User.from_row(row) if row else None

    def get_by_username(self, username: str) -> Optional[User]:
        row = self.conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return User.from_row(row) if row else None

    def set_last_login(self, user_id: str, when: str) -> None:
        self.conn.execute("UPDATE users SET last_login_at=? WHERE id=?", (when, user_id))

    def set_password(self, user_id: str, password_hash: str) -> None:
        self.conn.execute("UPDATE users SET password_hash=? WHERE id=?", (password_hash, user_id))


# ── organizations & memberships ──────────────────────────────────────────────
class OrgRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, org: Organization) -> Organization:
        _insert(self.conn, "organizations", {
            "id": org.id, "name": org.name, "slug": org.slug,
            "created_by": org.created_by, "created_at": org.created_at,
        })
        return org

    def get(self, org_id: str) -> Optional[Organization]:
        row = self.conn.execute("SELECT * FROM organizations WHERE id=?", (org_id,)).fetchone()
        return Organization.from_row(row) if row else None

    def slug_exists(self, slug: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM organizations WHERE slug=?", (slug,)
        ).fetchone() is not None

    def list_for_user(self, user_id: str) -> list[Organization]:
        rows = self.conn.execute(
            "SELECT o.* FROM organizations o "
            "JOIN memberships m ON m.org_id = o.id "
            "WHERE m.user_id=? ORDER BY o.created_at",
            (user_id,),
        ).fetchall()
        return [Organization.from_row(r) for r in rows]

    def delete(self, org_id: str) -> None:
        self.conn.execute("DELETE FROM organizations WHERE id=?", (org_id,))


class MembershipRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, m: Membership) -> Membership:
        _insert(self.conn, "memberships", {
            "id": m.id, "org_id": m.org_id, "user_id": m.user_id,
            "role": m.role.value, "created_at": m.created_at,
        })
        return m

    def get(self, org_id: str, user_id: str) -> Optional[Membership]:
        row = self.conn.execute(
            "SELECT * FROM memberships WHERE org_id=? AND user_id=?", (org_id, user_id)
        ).fetchone()
        return Membership.from_row(row) if row else None

    def set_role(self, org_id: str, user_id: str, role: Role) -> None:
        self.conn.execute(
            "UPDATE memberships SET role=? WHERE org_id=? AND user_id=?",
            (role.value, org_id, user_id),
        )

    def delete(self, org_id: str, user_id: str) -> None:
        self.conn.execute(
            "DELETE FROM memberships WHERE org_id=? AND user_id=?", (org_id, user_id)
        )

    def list_for_org(self, org_id: str) -> list[Membership]:
        rows = self.conn.execute(
            "SELECT * FROM memberships WHERE org_id=? ORDER BY created_at", (org_id,)
        ).fetchall()
        return [Membership.from_row(r) for r in rows]

    def count_with_role(self, org_id: str, role: Role) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM memberships WHERE org_id=? AND role=?",
            (org_id, role.value),
        ).fetchone()[0]


# ── projects ─────────────────────────────────────────────────────────────────
class ProjectRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, p: Project) -> Project:
        _insert(self.conn, "projects", {
            "id": p.id, "org_id": p.org_id, "name": p.name, "slug": p.slug,
            "created_by": p.created_by, "description": p.description,
            "engine": p.engine, "settings": _j(p.settings), "archived": _b(p.archived),
            "created_at": p.created_at, "updated_at": p.updated_at,
        })
        return p

    def get(self, project_id: str) -> Optional[Project]:
        row = self.conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        return Project.from_row(row) if row else None

    def slug_exists(self, org_id: str, slug: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM projects WHERE org_id=? AND slug=?", (org_id, slug)
        ).fetchone() is not None

    def list_for_org(self, org_id: str, *, include_archived: bool = False) -> list[Project]:
        sql = "SELECT * FROM projects WHERE org_id=?"
        if not include_archived:
            sql += " AND archived=0"
        sql += " ORDER BY updated_at DESC"
        rows = self.conn.execute(sql, (org_id,)).fetchall()
        return [Project.from_row(r) for r in rows]

    def update(self, p: Project) -> Project:
        self.conn.execute(
            "UPDATE projects SET name=?, description=?, engine=?, settings=?, "
            "archived=?, updated_at=? WHERE id=?",
            (p.name, p.description, p.engine, _j(p.settings), _b(p.archived),
             p.updated_at, p.id),
        )
        return p

    def delete(self, project_id: str) -> None:
        self.conn.execute("DELETE FROM projects WHERE id=?", (project_id,))


# ── tasks ────────────────────────────────────────────────────────────────────
class TaskRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, t: Task) -> Task:
        _insert(self.conn, "tasks", {
            "id": t.id, "project_id": t.project_id, "name": t.name,
            "created_by": t.created_by, "source": t.source, "status": t.status,
            "assignee_id": t.assignee_id, "created_at": t.created_at,
            "updated_at": t.updated_at,
        })
        return t

    def get(self, task_id: str) -> Optional[Task]:
        row = self.conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return Task.from_row(row) if row else None

    def list_for_project(self, project_id: str, *, status: Optional[str] = None) -> list[Task]:
        if status is None:
            rows = self.conn.execute(
                "SELECT * FROM tasks WHERE project_id=? ORDER BY created_at", (project_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM tasks WHERE project_id=? AND status=? ORDER BY created_at",
                (project_id, status),
            ).fetchall()
        return [Task.from_row(r) for r in rows]

    def update(self, t: Task) -> Task:
        self.conn.execute(
            "UPDATE tasks SET name=?, source=?, status=?, assignee_id=?, updated_at=? "
            "WHERE id=?",
            (t.name, t.source, t.status, t.assignee_id, t.updated_at, t.id),
        )
        return t


# ── annotations ──────────────────────────────────────────────────────────────
class AnnotationRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, a: Annotation) -> Annotation:
        _insert(self.conn, "annotations", {
            "id": a.id, "task_id": a.task_id, "author_id": a.author_id,
            "data": _j(a.data), "status": a.status, "reviewed_by": a.reviewed_by,
            "reviewed_at": a.reviewed_at, "review_note": a.review_note,
            "created_at": a.created_at, "updated_at": a.updated_at,
        })
        return a

    def get(self, annotation_id: str) -> Optional[Annotation]:
        row = self.conn.execute(
            "SELECT * FROM annotations WHERE id=?", (annotation_id,)
        ).fetchone()
        return Annotation.from_row(row) if row else None

    def list_for_task(self, task_id: str) -> list[Annotation]:
        rows = self.conn.execute(
            "SELECT * FROM annotations WHERE task_id=? ORDER BY created_at", (task_id,)
        ).fetchall()
        return [Annotation.from_row(r) for r in rows]

    def update(self, a: Annotation) -> Annotation:
        self.conn.execute(
            "UPDATE annotations SET data=?, status=?, reviewed_by=?, reviewed_at=?, "
            "review_note=?, updated_at=? WHERE id=?",
            (_j(a.data), a.status, a.reviewed_by, a.reviewed_at, a.review_note,
             a.updated_at, a.id),
        )
        return a


# ── credentials: sessions & API keys ─────────────────────────────────────────
class SessionRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, s: Session) -> Session:
        _insert(self.conn, "sessions", {
            "id": s.id, "user_id": s.user_id, "token_hash": s.token_hash,
            "created_at": s.created_at, "expires_at": s.expires_at,
            "revoked": _b(s.revoked),
        })
        return s

    def get_by_hash(self, token_hash: str) -> Optional[Session]:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE token_hash=?", (token_hash,)
        ).fetchone()
        return Session.from_row(row) if row else None

    def revoke_by_hash(self, token_hash: str) -> None:
        self.conn.execute("UPDATE sessions SET revoked=1 WHERE token_hash=?", (token_hash,))

    def revoke_all_for_user(self, user_id: str) -> None:
        self.conn.execute("UPDATE sessions SET revoked=1 WHERE user_id=?", (user_id,))


class ApiKeyRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, k: ApiKey) -> ApiKey:
        _insert(self.conn, "api_keys", {
            "id": k.id, "user_id": k.user_id, "org_id": k.org_id, "name": k.name,
            "prefix": k.prefix, "token_hash": k.token_hash, "created_at": k.created_at,
            "last_used_at": k.last_used_at, "expires_at": k.expires_at,
            "revoked": _b(k.revoked),
        })
        return k

    def get(self, key_id: str) -> Optional[ApiKey]:
        row = self.conn.execute("SELECT * FROM api_keys WHERE id=?", (key_id,)).fetchone()
        return ApiKey.from_row(row) if row else None

    def get_by_hash(self, token_hash: str) -> Optional[ApiKey]:
        row = self.conn.execute(
            "SELECT * FROM api_keys WHERE token_hash=?", (token_hash,)
        ).fetchone()
        return ApiKey.from_row(row) if row else None

    def list_for_user(self, user_id: str) -> list[ApiKey]:
        rows = self.conn.execute(
            "SELECT * FROM api_keys WHERE user_id=? ORDER BY created_at", (user_id,)
        ).fetchall()
        return [ApiKey.from_row(r) for r in rows]

    def touch_last_used(self, key_id: str, when: str) -> None:
        self.conn.execute("UPDATE api_keys SET last_used_at=? WHERE id=?", (when, key_id))

    def revoke(self, key_id: str) -> None:
        self.conn.execute("UPDATE api_keys SET revoked=1 WHERE id=?", (key_id,))


# ── audit log ────────────────────────────────────────────────────────────────
class AuditRepo:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, e: AuditEvent) -> AuditEvent:
        _insert(self.conn, "audit_events", {
            "id": e.id, "action": e.action, "actor_id": e.actor_id,
            "org_id": e.org_id, "target_type": e.target_type,
            "target_id": e.target_id, "detail": _j(e.detail), "created_at": e.created_at,
        })
        return e

    def list_for_org(self, org_id: str, *, limit: int = 100) -> list[AuditEvent]:
        rows = self.conn.execute(
            "SELECT * FROM audit_events WHERE org_id=? ORDER BY created_at DESC LIMIT ?",
            (org_id, limit),
        ).fetchall()
        return [AuditEvent.from_row(r) for r in rows]
