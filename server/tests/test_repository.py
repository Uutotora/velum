"""Tests for server.repository — direct CRUD and row round-trips.

These bypass the service layer to prove the SQL/model mapping itself is
correct; the RBAC/audit behaviour is covered in test_service.py.
"""
from server.db import Database
from server.models import (
    Annotation,
    ApiKey,
    AuditEvent,
    Membership,
    Organization,
    Project,
    Task,
    User,
    new_id,
    now_iso,
)
from server.rbac import Role
from server.repository import (
    AnnotationRepo,
    ApiKeyRepo,
    AuditRepo,
    MembershipRepo,
    OrgRepo,
    ProjectRepo,
    TaskRepo,
    UserRepo,
)


def _seed_user(conn, email="a@b.com", username="alice"):
    return UserRepo(conn).insert(User(
        id=new_id(), email=email, username=username, password_hash="h",
    ))


def test_user_crud_and_lookups():
    db = Database().initialize()
    with db.transaction() as conn:
        u = _seed_user(conn)
        users = UserRepo(conn)
        assert users.get(u.id).username == "alice"
        assert users.get_by_email("a@b.com").id == u.id
        assert users.get_by_username("alice").id == u.id
        assert users.get_by_email("missing@x.com") is None
        users.set_last_login(u.id, now_iso())
        assert users.get(u.id).last_login_at is not None


def test_project_settings_json_roundtrip():
    db = Database().initialize()
    with db.transaction() as conn:
        u = _seed_user(conn)
        org = OrgRepo(conn).insert(Organization(
            id=new_id(), name="Org", slug="org", created_by=u.id))
        projects = ProjectRepo(conn)
        p = projects.insert(Project(
            id=new_id(), org_id=org.id, name="Nuclei", slug="nuclei",
            created_by=u.id, settings={"resize_size": 1024, "clahe": True}))
        got = projects.get(p.id)
        assert got.settings == {"resize_size": 1024, "clahe": True}
        # update round-trips too
        got.settings = {"points_per_side": 64}
        got.name = "Nuclei v2"
        projects.update(got)
        assert projects.get(p.id).settings == {"points_per_side": 64}
        assert projects.get(p.id).name == "Nuclei v2"


def test_membership_role_and_counts():
    db = Database().initialize()
    with db.transaction() as conn:
        u = _seed_user(conn)
        org = OrgRepo(conn).insert(Organization(
            id=new_id(), name="Org", slug="org", created_by=u.id))
        members = MembershipRepo(conn)
        members.insert(Membership(id=new_id(), org_id=org.id, user_id=u.id, role=Role.OWNER))
        assert members.get(org.id, u.id).role is Role.OWNER
        assert members.count_with_role(org.id, Role.OWNER) == 1
        members.set_role(org.id, u.id, Role.ADMIN)
        assert members.get(org.id, u.id).role is Role.ADMIN
        assert members.count_with_role(org.id, Role.OWNER) == 0


def test_list_projects_orders_and_filters_archived():
    db = Database().initialize()
    with db.transaction() as conn:
        u = _seed_user(conn)
        org = OrgRepo(conn).insert(Organization(
            id=new_id(), name="Org", slug="org", created_by=u.id))
        projects = ProjectRepo(conn)
        projects.insert(Project(id=new_id(), org_id=org.id, name="A", slug="a",
                                created_by=u.id, updated_at="2026-01-01T00:00:00+00:00"))
        projects.insert(Project(id=new_id(), org_id=org.id, name="B", slug="b",
                                created_by=u.id, archived=True,
                                updated_at="2026-02-01T00:00:00+00:00"))
        active = projects.list_for_org(org.id)
        assert [p.slug for p in active] == ["a"]
        both = projects.list_for_org(org.id, include_archived=True)
        assert {p.slug for p in both} == {"a", "b"}
        # newest updated first
        assert both[0].slug == "b"


def test_task_and_annotation_repos():
    db = Database().initialize()
    with db.transaction() as conn:
        u = _seed_user(conn)
        org = OrgRepo(conn).insert(Organization(
            id=new_id(), name="Org", slug="org", created_by=u.id))
        p = ProjectRepo(conn).insert(Project(
            id=new_id(), org_id=org.id, name="P", slug="p", created_by=u.id))
        tasks = TaskRepo(conn)
        t = tasks.insert(Task(id=new_id(), project_id=p.id, name="img1", created_by=u.id))
        assert tasks.list_for_project(p.id)[0].id == t.id
        assert tasks.list_for_project(p.id, status="reviewed") == []
        anns = AnnotationRepo(conn)
        a = anns.insert(Annotation(id=new_id(), task_id=t.id, author_id=u.id,
                                   data={"n_cells": 7}))
        assert anns.list_for_task(t.id)[0].data == {"n_cells": 7}


def test_api_key_and_audit_repos():
    db = Database().initialize()
    with db.transaction() as conn:
        u = _seed_user(conn)
        keys = ApiKeyRepo(conn)
        k = keys.insert(ApiKey(id=new_id(), user_id=u.id, org_id=None, name="CI",
                               prefix="csk_ab12", token_hash="deadbeef"))
        assert keys.get_by_hash("deadbeef").id == k.id
        keys.revoke(k.id)
        assert keys.get(k.id).revoked is True

        audit = AuditRepo(conn)
        for i in range(3):
            audit.insert(AuditEvent(id=new_id(), action=f"a{i}", org_id="o1",
                                    created_at=f"2026-01-0{i+1}T00:00:00+00:00"))
        events = audit.list_for_org("o1", limit=2)
        assert len(events) == 2
        # newest first
        assert events[0].action == "a2"
