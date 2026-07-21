"""Integration tests for server.service — the full stack end to end.

Exercises AuthService/OrgService/ProjectService/TaskService/AnnotationService
against a real in-memory database, proving RBAC gating, the review workflow, the
audit trail, and the account/session/API-key lifecycle all work together.
"""
from datetime import timedelta

import pytest

from server.errors import AuthError, Conflict, NotFound, PermissionDenied, ValidationError
from server.models import AnnotationStatus, TaskStatus
from server.rbac import Role
from server.service import ServerApp


@pytest.fixture
def app():
    return ServerApp.create()  # in-memory


def register(app, email, username, password="password123", full_name=""):
    return app.auth.register(email, username, password, full_name=full_name)


def member(app, org_id, owner_id, role, *, email, username):
    """Register a fresh user and add them to the org with a role."""
    u = register(app, email, username)
    app.orgs.add_member(owner_id, org_id, u.id, role)
    return u


# ── accounts & sessions ──────────────────────────────────────────────────────
def test_register_and_duplicates(app):
    u = register(app, "Alice@Example.com ", "Alice")
    assert u.email == "alice@example.com" and u.username == "alice"
    with pytest.raises(Conflict) as ei:
        register(app, "alice@example.com", "other")
    assert ei.value.field == "email"
    with pytest.raises(Conflict) as ei:
        register(app, "new@example.com", "alice")
    assert ei.value.field == "username"


def test_register_validation(app):
    with pytest.raises(ValidationError):
        register(app, "not-an-email", "bob")
    with pytest.raises(ValidationError):
        register(app, "bob@example.com", "bob", password="short")


def test_login_session_resolve_logout(app):
    register(app, "a@b.com", "alice")
    user, token = app.auth.login("alice", "password123")
    assert app.auth.resolve_session(token).id == user.id
    # login by email works too, and stamps last_login_at
    _, token2 = app.auth.login("a@b.com", "password123")
    assert app.auth.resolve_session(token2).last_login_at is not None
    app.auth.logout(token)
    with pytest.raises(AuthError):
        app.auth.resolve_session(token)


def test_authenticate_failures(app):
    u = register(app, "a@b.com", "alice")
    with pytest.raises(AuthError):
        app.auth.authenticate("alice", "wrong-password")
    with pytest.raises(AuthError):
        app.auth.authenticate("ghost", "password123")  # no such user
    # a disabled account can't authenticate
    with app.db.transaction() as conn:
        conn.execute("UPDATE users SET is_active=0 WHERE id=?", (u.id,))
    with pytest.raises(AuthError):
        app.auth.authenticate("alice", "password123")


def test_session_expiry(app):
    u = register(app, "a@b.com", "alice")
    token = app.auth.create_session(u.id, ttl=timedelta(seconds=-1))
    with pytest.raises(AuthError):
        app.auth.resolve_session(token)


def test_api_key_lifecycle(app):
    u = register(app, "a@b.com", "alice")
    key, raw = app.api_keys.create(u.id, "CI token")
    assert raw.startswith("csk_")
    assert app.api_keys.resolve(raw).id == u.id
    assert app.api_keys.list_for_user(u.id)[0].name == "CI token"
    app.api_keys.revoke(u.id, key.id)
    with pytest.raises(AuthError):
        app.api_keys.resolve(raw)
    with pytest.raises(AuthError):
        app.api_keys.resolve("garbage")


# ── organizations & RBAC ─────────────────────────────────────────────────────
def test_org_create_makes_owner(app):
    owner = register(app, "a@b.com", "alice")
    org = app.orgs.create(owner.id, "My Lab")
    assert org.slug == "my-lab"
    assert app.orgs.role_of(org.id, owner.id) is Role.OWNER
    assert app.orgs.list_for_user(owner.id)[0].id == org.id


def test_add_member_and_escalation_guard(app):
    owner = register(app, "a@b.com", "alice")
    org = app.orgs.create(owner.id, "Lab")
    admin = member(app, org.id, owner.id, Role.ADMIN, email="ad@b.com", username="admin")
    # admin may add a manager (strictly below admin)…
    mgr = member(app, org.id, admin.id, Role.MANAGER, email="m@b.com", username="mgr")
    assert app.orgs.role_of(org.id, mgr.id) is Role.MANAGER
    # …but not an owner (escalation) or another admin (lateral).
    bob = register(app, "bob@b.com", "bob")
    with pytest.raises(PermissionDenied):
        app.orgs.add_member(admin.id, org.id, bob.id, Role.OWNER)
    with pytest.raises(PermissionDenied):
        app.orgs.add_member(admin.id, org.id, bob.id, Role.ADMIN)
    # a manager can't manage members at all
    with pytest.raises(PermissionDenied):
        app.orgs.add_member(mgr.id, org.id, bob.id, Role.VIEWER)


def test_add_member_conflicts_and_missing(app):
    owner = register(app, "a@b.com", "alice")
    org = app.orgs.create(owner.id, "Lab")
    bob = register(app, "bob@b.com", "bob")
    app.orgs.add_member(owner.id, org.id, bob.id, Role.VIEWER)
    with pytest.raises(Conflict):
        app.orgs.add_member(owner.id, org.id, bob.id, Role.VIEWER)  # already a member
    with pytest.raises(NotFound):
        app.orgs.add_member(owner.id, org.id, "no-such-user", Role.VIEWER)


def test_last_owner_protected(app):
    owner = register(app, "a@b.com", "alice")
    org = app.orgs.create(owner.id, "Lab")
    # can't demote or remove the only owner
    with pytest.raises(Conflict):
        app.orgs.set_role(owner.id, org.id, owner.id, Role.ADMIN)
    with pytest.raises(Conflict):
        app.orgs.remove_member(owner.id, org.id, owner.id)
    # promote a second owner, then the first can step down
    bob = member(app, org.id, owner.id, Role.OWNER, email="bob@b.com", username="bob")
    app.orgs.set_role(owner.id, org.id, owner.id, Role.ADMIN)
    assert app.orgs.role_of(org.id, owner.id) is Role.ADMIN


def test_non_member_is_denied(app):
    owner = register(app, "a@b.com", "alice")
    org = app.orgs.create(owner.id, "Lab")
    outsider = register(app, "out@b.com", "outsider")
    with pytest.raises(PermissionDenied):
        app.projects.list(outsider.id, org.id)


# ── projects ─────────────────────────────────────────────────────────────────
def test_project_permissions(app):
    owner = register(app, "a@b.com", "alice")
    org = app.orgs.create(owner.id, "Lab")
    viewer = member(app, org.id, owner.id, Role.VIEWER, email="v@b.com", username="viewer")
    mgr = member(app, org.id, owner.id, Role.MANAGER, email="m@b.com", username="mgr")
    # a viewer can't create; a manager can
    with pytest.raises(PermissionDenied):
        app.projects.create(viewer.id, org.id, "Nuclei")
    p = app.projects.create(mgr.id, org.id, "Nuclei", settings={"resize_size": 512})
    assert p.slug == "nuclei" and p.settings == {"resize_size": 512}
    # a viewer *can* read
    assert app.projects.get(viewer.id, p.id).id == p.id
    # unique slug within the org
    p2 = app.projects.create(mgr.id, org.id, "Nuclei")
    assert p2.slug == "nuclei-2"


def test_project_lifecycle_and_cascade(app):
    owner = register(app, "a@b.com", "alice")
    org = app.orgs.create(owner.id, "Lab")
    p = app.projects.create(owner.id, org.id, "Nuclei")
    app.projects.update(owner.id, p.id, name="Nuclei v2", settings={"clahe": True})
    got = app.projects.get(owner.id, p.id)
    assert got.name == "Nuclei v2" and got.settings == {"clahe": True}
    # archive hides it from the default listing
    app.projects.archive(owner.id, p.id)
    assert app.projects.list(owner.id, org.id) == []
    assert len(app.projects.list(owner.id, org.id, include_archived=True)) == 1
    # a task under it, then delete cascades
    app.tasks.create(owner.id, p.id, "img1")
    app.projects.delete(owner.id, p.id)
    with pytest.raises(NotFound):
        app.projects.get(owner.id, p.id)
    # only manager+ can delete
    p3 = app.projects.create(owner.id, org.id, "Other")
    viewer = member(app, org.id, owner.id, Role.VIEWER, email="v@b.com", username="viewer")
    with pytest.raises(PermissionDenied):
        app.projects.delete(viewer.id, p3.id)


# ── tasks & the review workflow ──────────────────────────────────────────────
def test_task_assignment(app):
    owner = register(app, "a@b.com", "alice")
    org = app.orgs.create(owner.id, "Lab")
    p = app.projects.create(owner.id, org.id, "Nuclei")
    ann = member(app, org.id, owner.id, Role.ANNOTATOR, email="an@b.com", username="ann")
    t = app.tasks.create(owner.id, p.id, "img1")
    assert t.status == TaskStatus.PENDING
    t = app.tasks.assign(owner.id, t.id, ann.id)
    assert t.assignee_id == ann.id and t.status == TaskStatus.IN_PROGRESS
    # can't assign to a non-member
    outsider = register(app, "out@b.com", "outsider")
    with pytest.raises(ValidationError):
        app.tasks.assign(owner.id, t.id, outsider.id)


def test_review_workflow(app):
    owner = register(app, "a@b.com", "alice")
    org = app.orgs.create(owner.id, "Lab")
    p = app.projects.create(owner.id, org.id, "Nuclei")
    annotator = member(app, org.id, owner.id, Role.ANNOTATOR, email="an@b.com", username="ann")
    reviewer = member(app, org.id, owner.id, Role.REVIEWER, email="rv@b.com", username="rev")
    t = app.tasks.create(owner.id, p.id, "img1")

    # annotator submits → task COMPLETED, annotation SUBMITTED
    a = app.annotations.submit(annotator.id, t.id, {"n_cells": 42})
    assert a.status == AnnotationStatus.SUBMITTED
    assert app.tasks.list(owner.id, p.id)[0].status == TaskStatus.COMPLETED

    # annotator can't review their own (or anyone's) work
    with pytest.raises(PermissionDenied):
        app.annotations.review(annotator.id, a.id, approve=True)

    # reviewer rejects → task back IN_PROGRESS
    a = app.annotations.review(reviewer.id, a.id, approve=False, note="split cells 3 & 4")
    assert a.status == AnnotationStatus.REJECTED and a.review_note == "split cells 3 & 4"
    assert app.tasks.list(owner.id, p.id)[0].status == TaskStatus.IN_PROGRESS

    # resubmit + approve → task REVIEWED
    a2 = app.annotations.submit(annotator.id, t.id, {"n_cells": 44})
    a2 = app.annotations.review(reviewer.id, a2.id, approve=True)
    assert a2.status == AnnotationStatus.APPROVED and a2.reviewed_by == reviewer.id
    assert app.tasks.list(owner.id, p.id)[0].status == TaskStatus.REVIEWED
    assert len(app.annotations.list_for_task(reviewer.id, t.id)) == 2


# ── audit log ────────────────────────────────────────────────────────────────
def test_audit_log_records_and_is_gated(app):
    owner = register(app, "a@b.com", "alice")
    org = app.orgs.create(owner.id, "Lab")
    app.projects.create(owner.id, org.id, "Nuclei")
    events = app.audit.list_for_org(owner.id, org.id)
    actions = [e.action for e in events]
    assert "org.create" in actions and "project.create" in actions
    # newest first
    assert events[0].created_at >= events[-1].created_at
    # an annotator (no AUDIT_VIEW) is denied
    annotator = member(app, org.id, owner.id, Role.ANNOTATOR, email="an@b.com", username="ann")
    with pytest.raises(PermissionDenied):
        app.audit.list_for_org(annotator.id, org.id)
