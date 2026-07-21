"""The business API — orchestrates repositories, RBAC, and the audit log.

This is the layer application code (and, later, the HTTP handlers) actually
call. Each service method is a **unit of work**: it runs inside one
:meth:`Database.transaction`, so the state change and its audit-log entry commit
together or roll back together — the audit trail can never disagree with what
happened. Every mutating method:

1. resolves the caller's role in the relevant organization (one indexed lookup),
2. ``require``\\s the permission that gates the action (raising
   :class:`PermissionDenied` on failure — authorization is never optional),
3. performs the change through the repositories, and
4. writes an :class:`AuditEvent`.

Authentication (:class:`AuthService`) is stateless-token based: a login mints an
opaque session token whose hash is stored, so validating a request is a single
indexed lookup with no server-side session memory — which is what lets the
future web tier scale horizontally. The same pattern backs long-lived API keys.

:class:`ServerApp` is the front door: ``ServerApp.create("cellseg1.db")`` builds
the database, applies the schema, and exposes every service as an attribute.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import security, validation
from .db import Database
from .errors import AuthError, Conflict, NotFound, PermissionDenied, ValidationError
from .models import (
    Annotation,
    AnnotationStatus,
    ApiKey,
    AuditEvent,
    Membership,
    Organization,
    Project,
    Session,
    Task,
    TaskStatus,
    User,
    new_id,
    now_iso,
)
from .rbac import Perm, Role, can_assign_role, require
from .repository import (
    AnnotationRepo,
    ApiKeyRepo,
    AuditRepo,
    MembershipRepo,
    OrgRepo,
    ProjectRepo,
    SessionRepo,
    TaskRepo,
    UserRepo,
)

# Default lifetimes. Sessions expire in two weeks; API keys don't expire unless
# a caller asks. Both are overridable per call.
DEFAULT_SESSION_TTL = timedelta(days=14)

# A well-formed but unmatchable password hash, computed once, used to keep
# authentication timing roughly constant whether or not the account exists —
# so an attacker can't distinguish "no such user" from "wrong password" by
# response time (user-enumeration defence).
_DUMMY_HASH: Optional[str] = None


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = security.hash_password("timing-equalizer-not-a-real-password")
    return _DUMMY_HASH


def _iso_in(delta: timedelta) -> str:
    """An ISO-8601 UTC timestamp ``delta`` from now (same format as now_iso)."""
    return (datetime.now(timezone.utc) + delta).replace(microsecond=0).isoformat()


def _is_expired(expires_at: Optional[str]) -> bool:
    """True if an ISO-8601 UTC expiry is in the past. None never expires."""
    if not expires_at:
        return False
    return expires_at <= now_iso()  # lexicographic compare is chronological here


def _audit(
    conn: sqlite3.Connection,
    action: str,
    *,
    actor_id: Optional[str] = None,
    org_id: Optional[str] = None,
    target_type: str = "",
    target_id: str = "",
    detail: Optional[dict] = None,
) -> None:
    AuditRepo(conn).insert(AuditEvent(
        id=new_id(), action=action, actor_id=actor_id, org_id=org_id,
        target_type=target_type, target_id=target_id, detail=detail or {},
        created_at=now_iso(),
    ))


def _role_in_org(conn: sqlite3.Connection, org_id: str, user_id: str) -> Role:
    """Resolve the caller's role in an org, or refuse if they aren't a member.

    A non-member is a :class:`PermissionDenied`, not a :class:`NotFound`: we
    don't confirm the org exists to someone with no business seeing it.
    """
    m = MembershipRepo(conn).get(org_id, user_id)
    if m is None:
        raise PermissionDenied("You are not a member of this organization.")
    return m.role


# ── authentication & accounts ────────────────────────────────────────────────
class AuthService:
    def __init__(self, db: Database):
        self.db = db

    def register(
        self, email: str, username: str, password: str, *, full_name: str = ""
    ) -> User:
        """Create a new account. Raises Conflict on a duplicate email/username."""
        email = validation.validate_email(email)
        username = validation.validate_username(username)
        password = validation.validate_password(password)
        full_name = full_name.strip() if isinstance(full_name, str) else ""
        with self.db.transaction() as conn:
            users = UserRepo(conn)
            if users.get_by_email(email):
                raise Conflict("An account with this email already exists.", field="email")
            if users.get_by_username(username):
                raise Conflict("This username is taken.", field="username")
            user = User(
                id=new_id(), email=email, username=username,
                password_hash=security.hash_password(password), full_name=full_name,
            )
            try:
                users.insert(user)
            except sqlite3.IntegrityError:  # backstop for the check-then-insert race
                raise Conflict("An account with these details already exists.")
            _audit(conn, "user.register", actor_id=user.id,
                   target_type="user", target_id=user.id)
            return user

    def authenticate(self, identifier: str, password: str) -> User:
        """Verify credentials, returning the User, else raise AuthError.

        ``identifier`` may be an email or a username. On any failure the error
        is the same generic AuthError, and a dummy hash is verified when the
        account is missing so timing doesn't reveal whether it exists.
        """
        ident = (identifier or "").strip().lower()
        conn = self.db.connection()
        users = UserRepo(conn)
        user = users.get_by_email(ident) or users.get_by_username(ident)
        if user is None:
            security.verify_password(password, _dummy_hash())  # equalise timing
            raise AuthError("Incorrect email/username or password.")
        if not security.verify_password(password, user.password_hash):
            raise AuthError("Incorrect email/username or password.")
        if not user.is_active:
            raise AuthError("This account is disabled.")
        # Opportunistically upgrade an old/weak password hash after a good login.
        if security.needs_rehash(user.password_hash):
            with self.db.transaction() as w:
                UserRepo(w).set_password(user.id, security.hash_password(password))
        return user

    def login(
        self, identifier: str, password: str, *, ttl: timedelta = DEFAULT_SESSION_TTL
    ) -> tuple[User, str]:
        """Authenticate and mint a session token (returned raw, once)."""
        user = self.authenticate(identifier, password)
        token = self.create_session(user.id, ttl=ttl)
        with self.db.transaction() as conn:
            UserRepo(conn).set_last_login(user.id, now_iso())
            _audit(conn, "user.login", actor_id=user.id,
                   target_type="user", target_id=user.id)
        return user, token

    def create_session(self, user_id: str, *, ttl: timedelta = DEFAULT_SESSION_TTL) -> str:
        """Create a session for a user and return the raw bearer token."""
        token = security.generate_session_token()
        with self.db.transaction() as conn:
            SessionRepo(conn).insert(Session(
                id=new_id(), user_id=user_id,
                token_hash=security.hash_token(token),
                expires_at=_iso_in(ttl) if ttl else None,
            ))
        return token

    def resolve_session(self, token: str) -> User:
        """Validate a raw session token and return its (active) User.

        A missing, revoked, or expired token — or a disabled user — is an
        AuthError. This is the one call the future web tier runs on every
        authenticated request: a single indexed lookup, no shared session store.
        """
        if not token:
            raise AuthError("Authentication required.")
        conn = self.db.connection()
        session = SessionRepo(conn).get_by_hash(security.hash_token(token))
        if session is None or session.revoked or _is_expired(session.expires_at):
            raise AuthError("Session is invalid or has expired.")
        user = UserRepo(conn).get(session.user_id)
        if user is None or not user.is_active:
            raise AuthError("Account is unavailable.")
        return user

    def logout(self, token: str) -> None:
        """Revoke a session token (idempotent)."""
        with self.db.transaction() as conn:
            SessionRepo(conn).revoke_by_hash(security.hash_token(token))


class ApiKeyService:
    def __init__(self, db: Database):
        self.db = db

    def create(
        self, user_id: str, name: str, *, org_id: Optional[str] = None,
        ttl: Optional[timedelta] = None,
    ) -> tuple[ApiKey, str]:
        """Mint an API key for a user. Returns (stored record, raw key once)."""
        minted = security.generate_api_key()
        with self.db.transaction() as conn:
            key = ApiKeyRepo(conn).insert(ApiKey(
                id=new_id(), user_id=user_id, org_id=org_id,
                name=(name or "").strip(), prefix=minted.prefix,
                token_hash=minted.token_hash,
                expires_at=_iso_in(ttl) if ttl else None,
            ))
            _audit(conn, "apikey.create", actor_id=user_id, org_id=org_id,
                   target_type="api_key", target_id=key.id)
        return key, minted.raw

    def resolve(self, raw: str) -> User:
        """Authenticate a raw API key, returning its (active) User."""
        if not raw or security.api_key_prefix(raw) is None:
            raise AuthError("Invalid API key.")
        conn = self.db.connection()
        key = ApiKeyRepo(conn).get_by_hash(security.hash_token(raw))
        if key is None or key.revoked or _is_expired(key.expires_at):
            raise AuthError("API key is invalid or has expired.")
        user = UserRepo(conn).get(key.user_id)
        if user is None or not user.is_active:
            raise AuthError("Account is unavailable.")
        with self.db.transaction() as w:
            ApiKeyRepo(w).touch_last_used(key.id, now_iso())
        return user

    def list_for_user(self, user_id: str) -> list[ApiKey]:
        return ApiKeyRepo(self.db.connection()).list_for_user(user_id)

    def revoke(self, user_id: str, key_id: str) -> None:
        """Revoke one of the caller's own keys."""
        with self.db.transaction() as conn:
            keys = ApiKeyRepo(conn)
            key = keys.get(key_id)
            if key is None or key.user_id != user_id:
                raise NotFound("API key not found.")
            keys.revoke(key_id)
            _audit(conn, "apikey.revoke", actor_id=user_id, org_id=key.org_id,
                   target_type="api_key", target_id=key_id)


# ── organizations & members ──────────────────────────────────────────────────
class OrgService:
    def __init__(self, db: Database):
        self.db = db

    def create(self, owner_id: str, name: str) -> Organization:
        """Create an org; the caller becomes its OWNER (atomic with membership)."""
        name = validation.validate_name(name, field="name")
        with self.db.transaction() as conn:
            orgs = OrgRepo(conn)
            slug = self._unique_slug(orgs, validation.slugify(name))
            org = orgs.insert(Organization(
                id=new_id(), name=name, slug=slug, created_by=owner_id,
            ))
            MembershipRepo(conn).insert(Membership(
                id=new_id(), org_id=org.id, user_id=owner_id, role=Role.OWNER,
            ))
            _audit(conn, "org.create", actor_id=owner_id, org_id=org.id,
                   target_type="organization", target_id=org.id, detail={"name": name})
            return org

    def role_of(self, org_id: str, user_id: str) -> Optional[Role]:
        m = MembershipRepo(self.db.connection()).get(org_id, user_id)
        return m.role if m else None

    def list_for_user(self, user_id: str) -> list[Organization]:
        return OrgRepo(self.db.connection()).list_for_user(user_id)

    def list_members(self, actor_id: str, org_id: str) -> list[Membership]:
        _role_in_org(self.db.connection(), org_id, actor_id)  # membership required
        return MembershipRepo(self.db.connection()).list_for_org(org_id)

    def add_member(self, actor_id: str, org_id: str, user_id: str, role: Role) -> Membership:
        """Invite an existing user to the org with a role, RBAC-gated."""
        role = role if isinstance(role, Role) else Role.from_str(role)
        with self.db.transaction() as conn:
            actor_role = _role_in_org(conn, org_id, actor_id)
            require(actor_role, Perm.ORG_MANAGE_MEMBERS)
            if not can_assign_role(actor_role, role):
                raise PermissionDenied(f"You can't grant the '{role.value}' role.")
            if UserRepo(conn).get(user_id) is None:
                raise NotFound("No such user.")
            members = MembershipRepo(conn)
            if members.get(org_id, user_id) is not None:
                raise Conflict("That user is already a member.")
            m = members.insert(Membership(
                id=new_id(), org_id=org_id, user_id=user_id, role=role,
            ))
            _audit(conn, "org.add_member", actor_id=actor_id, org_id=org_id,
                   target_type="user", target_id=user_id, detail={"role": role.value})
            return m

    def set_role(self, actor_id: str, org_id: str, user_id: str, role: Role) -> Membership:
        """Change a member's role, guarding escalation and the last-owner rule."""
        role = role if isinstance(role, Role) else Role.from_str(role)
        with self.db.transaction() as conn:
            actor_role = _role_in_org(conn, org_id, actor_id)
            require(actor_role, Perm.ORG_MANAGE_MEMBERS)
            if not can_assign_role(actor_role, role):
                raise PermissionDenied(f"You can't grant the '{role.value}' role.")
            members = MembershipRepo(conn)
            current = members.get(org_id, user_id)
            if current is None:
                raise NotFound("That user is not a member.")
            if (current.role is Role.OWNER and role is not Role.OWNER
                    and members.count_with_role(org_id, Role.OWNER) <= 1):
                raise Conflict("An organization must keep at least one owner.")
            members.set_role(org_id, user_id, role)
            _audit(conn, "org.set_role", actor_id=actor_id, org_id=org_id,
                   target_type="user", target_id=user_id,
                   detail={"from": current.role.value, "to": role.value})
            return members.get(org_id, user_id)

    def remove_member(self, actor_id: str, org_id: str, user_id: str) -> None:
        """Remove a member, guarding the last-owner rule."""
        with self.db.transaction() as conn:
            actor_role = _role_in_org(conn, org_id, actor_id)
            require(actor_role, Perm.ORG_MANAGE_MEMBERS)
            members = MembershipRepo(conn)
            current = members.get(org_id, user_id)
            if current is None:
                raise NotFound("That user is not a member.")
            if (current.role is Role.OWNER
                    and members.count_with_role(org_id, Role.OWNER) <= 1):
                raise Conflict("An organization must keep at least one owner.")
            members.delete(org_id, user_id)
            _audit(conn, "org.remove_member", actor_id=actor_id, org_id=org_id,
                   target_type="user", target_id=user_id)

    @staticmethod
    def _unique_slug(orgs: OrgRepo, base: str) -> str:
        slug, n = base, 2
        while orgs.slug_exists(slug):
            slug = f"{base}-{n}"
            n += 1
        return slug


# ── projects ─────────────────────────────────────────────────────────────────
class ProjectService:
    def __init__(self, db: Database):
        self.db = db

    def create(
        self, actor_id: str, org_id: str, name: str, *,
        description: str = "", engine: str = "cellseg1",
        settings: Optional[dict] = None,
    ) -> Project:
        name = validation.validate_name(name, field="name")
        with self.db.transaction() as conn:
            require(_role_in_org(conn, org_id, actor_id), Perm.PROJECT_CREATE)
            projects = ProjectRepo(conn)
            slug = self._unique_slug(projects, org_id, validation.slugify(name))
            p = projects.insert(Project(
                id=new_id(), org_id=org_id, name=name, slug=slug,
                created_by=actor_id, description=description or "",
                engine=engine, settings=settings or {},
            ))
            _audit(conn, "project.create", actor_id=actor_id, org_id=org_id,
                   target_type="project", target_id=p.id, detail={"name": name})
            return p

    def get(self, actor_id: str, project_id: str) -> Project:
        conn = self.db.connection()
        p = ProjectRepo(conn).get(project_id)
        if p is None:
            raise NotFound("Project not found.")
        require(_role_in_org(conn, p.org_id, actor_id), Perm.PROJECT_VIEW)
        return p

    def list(self, actor_id: str, org_id: str, *, include_archived: bool = False) -> list[Project]:
        conn = self.db.connection()
        require(_role_in_org(conn, org_id, actor_id), Perm.PROJECT_VIEW)
        return ProjectRepo(conn).list_for_org(org_id, include_archived=include_archived)

    def update(
        self, actor_id: str, project_id: str, *,
        name: Optional[str] = None, description: Optional[str] = None,
        engine: Optional[str] = None, settings: Optional[dict] = None,
    ) -> Project:
        with self.db.transaction() as conn:
            projects = ProjectRepo(conn)
            p = projects.get(project_id)
            if p is None:
                raise NotFound("Project not found.")
            require(_role_in_org(conn, p.org_id, actor_id), Perm.PROJECT_EDIT)
            if name is not None:
                p.name = validation.validate_name(name, field="name")
            if description is not None:
                p.description = description
            if engine is not None:
                p.engine = engine
            if settings is not None:
                p.settings = settings
            p.updated_at = now_iso()
            projects.update(p)
            _audit(conn, "project.update", actor_id=actor_id, org_id=p.org_id,
                   target_type="project", target_id=p.id)
            return p

    def archive(self, actor_id: str, project_id: str, *, archived: bool = True) -> Project:
        with self.db.transaction() as conn:
            projects = ProjectRepo(conn)
            p = projects.get(project_id)
            if p is None:
                raise NotFound("Project not found.")
            require(_role_in_org(conn, p.org_id, actor_id), Perm.PROJECT_EDIT)
            p.archived = archived
            p.updated_at = now_iso()
            projects.update(p)
            _audit(conn, "project.archive", actor_id=actor_id, org_id=p.org_id,
                   target_type="project", target_id=p.id, detail={"archived": archived})
            return p

    def delete(self, actor_id: str, project_id: str) -> None:
        with self.db.transaction() as conn:
            projects = ProjectRepo(conn)
            p = projects.get(project_id)
            if p is None:
                raise NotFound("Project not found.")
            require(_role_in_org(conn, p.org_id, actor_id), Perm.PROJECT_DELETE)
            projects.delete(project_id)  # cascades to tasks + annotations
            _audit(conn, "project.delete", actor_id=actor_id, org_id=p.org_id,
                   target_type="project", target_id=project_id)

    @staticmethod
    def _unique_slug(projects: ProjectRepo, org_id: str, base: str) -> str:
        slug, n = base, 2
        while projects.slug_exists(org_id, slug):
            slug = f"{base}-{n}"
            n += 1
        return slug


# ── tasks ────────────────────────────────────────────────────────────────────
class TaskService:
    def __init__(self, db: Database):
        self.db = db

    def create(self, actor_id: str, project_id: str, name: str, *, source: str = "") -> Task:
        name = validation.validate_name(name, field="name")
        with self.db.transaction() as conn:
            p = ProjectRepo(conn).get(project_id)
            if p is None:
                raise NotFound("Project not found.")
            require(_role_in_org(conn, p.org_id, actor_id), Perm.TASK_CREATE)
            t = TaskRepo(conn).insert(Task(
                id=new_id(), project_id=project_id, name=name,
                created_by=actor_id, source=source or "",
            ))
            _audit(conn, "task.create", actor_id=actor_id, org_id=p.org_id,
                   target_type="task", target_id=t.id)
            return t

    def list(self, actor_id: str, project_id: str, *, status: Optional[str] = None) -> list[Task]:
        conn = self.db.connection()
        p = ProjectRepo(conn).get(project_id)
        if p is None:
            raise NotFound("Project not found.")
        require(_role_in_org(conn, p.org_id, actor_id), Perm.TASK_VIEW)
        return TaskRepo(conn).list_for_project(project_id, status=status)

    def assign(self, actor_id: str, task_id: str, assignee_id: Optional[str]) -> Task:
        """Assign (or, with assignee_id=None, unassign) a task to a member."""
        with self.db.transaction() as conn:
            tasks = TaskRepo(conn)
            t = tasks.get(task_id)
            if t is None:
                raise NotFound("Task not found.")
            p = ProjectRepo(conn).get(t.project_id)
            require(_role_in_org(conn, p.org_id, actor_id), Perm.TASK_ASSIGN)
            if assignee_id is not None and MembershipRepo(conn).get(p.org_id, assignee_id) is None:
                raise ValidationError("Assignee must be a member of the organization.",
                                      field="assignee_id")
            t.assignee_id = assignee_id
            t.status = TaskStatus.IN_PROGRESS if assignee_id else TaskStatus.PENDING
            t.updated_at = now_iso()
            tasks.update(t)
            _audit(conn, "task.assign", actor_id=actor_id, org_id=p.org_id,
                   target_type="task", target_id=t.id, detail={"assignee_id": assignee_id})
            return t


# ── annotations & the review workflow ────────────────────────────────────────
class AnnotationService:
    def __init__(self, db: Database):
        self.db = db

    def submit(self, actor_id: str, task_id: str, data: dict) -> Annotation:
        """Produce/submit an annotation for a task (moves the task to COMPLETED)."""
        with self.db.transaction() as conn:
            tasks = TaskRepo(conn)
            t = tasks.get(task_id)
            if t is None:
                raise NotFound("Task not found.")
            p = ProjectRepo(conn).get(t.project_id)
            require(_role_in_org(conn, p.org_id, actor_id), Perm.ANNOTATION_CREATE)
            a = AnnotationRepo(conn).insert(Annotation(
                id=new_id(), task_id=task_id, author_id=actor_id,
                data=data or {}, status=AnnotationStatus.SUBMITTED,
            ))
            t.status = TaskStatus.COMPLETED
            t.updated_at = now_iso()
            tasks.update(t)
            _audit(conn, "annotation.submit", actor_id=actor_id, org_id=p.org_id,
                   target_type="annotation", target_id=a.id)
            return a

    def review(
        self, actor_id: str, annotation_id: str, *, approve: bool, note: str = ""
    ) -> Annotation:
        """Approve or reject a submitted annotation (a distinct reviewer job).

        Approving marks the task REVIEWED; rejecting sends it back to
        IN_PROGRESS so it can be redone. Recorded in the audit log either way.
        """
        with self.db.transaction() as conn:
            anns = AnnotationRepo(conn)
            a = anns.get(annotation_id)
            if a is None:
                raise NotFound("Annotation not found.")
            t = TaskRepo(conn).get(a.task_id)
            p = ProjectRepo(conn).get(t.project_id)
            require(_role_in_org(conn, p.org_id, actor_id), Perm.ANNOTATION_REVIEW)
            a.status = AnnotationStatus.APPROVED if approve else AnnotationStatus.REJECTED
            a.reviewed_by = actor_id
            a.reviewed_at = now_iso()
            a.review_note = note or ""
            a.updated_at = a.reviewed_at
            anns.update(a)
            t.status = TaskStatus.REVIEWED if approve else TaskStatus.IN_PROGRESS
            t.updated_at = a.reviewed_at
            TaskRepo(conn).update(t)
            _audit(conn, "annotation.review", actor_id=actor_id, org_id=p.org_id,
                   target_type="annotation", target_id=a.id,
                   detail={"approved": approve})
            return a

    def list_for_task(self, actor_id: str, task_id: str) -> list[Annotation]:
        conn = self.db.connection()
        t = TaskRepo(conn).get(task_id)
        if t is None:
            raise NotFound("Task not found.")
        p = ProjectRepo(conn).get(t.project_id)
        require(_role_in_org(conn, p.org_id, actor_id), Perm.ANNOTATION_VIEW)
        return AnnotationRepo(conn).list_for_task(task_id)


# ── audit log (read) ─────────────────────────────────────────────────────────
class AuditService:
    def __init__(self, db: Database):
        self.db = db

    def list_for_org(self, actor_id: str, org_id: str, *, limit: int = 100) -> list[AuditEvent]:
        conn = self.db.connection()
        require(_role_in_org(conn, org_id, actor_id), Perm.AUDIT_VIEW)
        return AuditRepo(conn).list_for_org(org_id, limit=limit)


# ── front door ───────────────────────────────────────────────────────────────
class ServerApp:
    """Bundles the database and every service behind one object.

    ``app = ServerApp.create("cellseg1.db")`` (or ``ServerApp.create()`` for an
    in-memory instance) is the whole public entry point. The future HTTP tier
    constructs one of these per process and routes requests to its services.
    """

    def __init__(self, db: Database):
        self.db = db
        self.auth = AuthService(db)
        self.api_keys = ApiKeyService(db)
        self.orgs = OrgService(db)
        self.projects = ProjectService(db)
        self.tasks = TaskService(db)
        self.annotations = AnnotationService(db)
        self.audit = AuditService(db)

    @classmethod
    def create(cls, path: str = ":memory:") -> "ServerApp":
        """Build the database, apply the schema, and wire up the services."""
        return cls(Database(path).initialize())
