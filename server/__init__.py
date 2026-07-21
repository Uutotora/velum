"""CellSeg1 Server — the multi-user backend foundation.

This package is the **server-side** contour the `studio/` desktop app never
had: user accounts, organizations, role-based access control,
and the Label-Studio-shaped domain model (Organization → Project → Task →
Annotation → Review) persisted in a real database instead of per-machine JSON
files.

Design goals, in priority order:

1. **Runs with zero infrastructure.** The default store is stdlib ``sqlite3``
   in WAL mode — no server to install, no container, nothing to configure. A
   developer (or a small lab) gets a working multi-user backend from a single
   file. This is what makes the foundation usable *today*, before any web tier
   exists.
2. **Scales to a real deployment without a rewrite.** Every storage access
   goes through the thin repository layer in :mod:`server.repository`; the SQL
   is standard and the schema is Postgres-portable, so moving to Postgres +
   connection pooling for 10k+ users/day is a driver swap, not a redesign.
   Auth is **stateless token** based (:mod:`server.security`) so the eventual
   HTTP tier scales horizontally behind a load balancer with no shared session
   store.
3. **Testable without heavy dependencies.** Nothing in this package imports
   torch, napari, Qt, or any third-party package — it is pure standard library.
   That keeps its test suite in CI's light ``test`` dependency-group, exactly
   like the rest of this repo's pure-logic core.

The layers, bottom to top::

    errors.py       domain exceptions (AuthError / PermissionDenied / ...)
    security.py     password hashing (scrypt) + opaque token/API-key handling
    validation.py   email/username/password/slug validation
    rbac.py         roles + the permission matrix + can()/require()
    models.py       the entity dataclasses (User, Organization, Project, ...)
    db.py           sqlite3 connection factory (WAL) + schema DDL + migrations
    repository.py   data access — one repo per entity, plain parameterised SQL
    service.py      the business API — orchestrates repos + rbac + audit log

The HTTP surface (FastAPI) and the Studio client integration are deliberately
*not* here yet: they are the next slices on top of this tested foundation.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
