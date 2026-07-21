"""Velum — the standalone desktop application layer.

A faithful, native-Qt reproduction of the north-star mockup, with real
functionality wired back tab by tab; the plan, the fresh-agent prompt, the
changelog and the backlog all live in the repo's ``docs/velum/`` folder.

Modules (import direction is one-way, leaf → shell):

- :mod:`~studio.theme` — design tokens (light + dark) + QSS.
- :mod:`~studio.components` — the static UI kit + sidebar.
- :mod:`~studio.paint` — the nuclei canvas stand-in art.
- :mod:`~studio.demo` — static demo content for tabs not yet wired.
- :mod:`~studio.project` — the ``Project``/``ProjectStore`` data model, pure
  stdlib (real, persisted projects — the Projects tab's data layer).
- :mod:`~studio.project_controller` — the Qt-free ``ProjectController``
  (search/filter, favourites, the active project) the Home/Projects screens
  are bound to.
- :mod:`~studio.screens` / :mod:`~.workspace` / :mod:`~.extra_screens`
  — the screens. Home/Projects are wired to ``project_controller``; the rest
  still render static ``demo`` content pending their own tab.
- :mod:`~studio.new_project_dialog` — the "+ New Project" modal (name/
  description → import → engine), writing through ``project.ProjectStore``.
- :mod:`~studio.overlays` — assistant drawer, logs, ⌘K palette, toast.
- :mod:`~studio.window_chrome` — the frameless title bar.
- :mod:`~studio.app` — ``StudioWindow`` + ``main`` (the entry point).

No napari, no torch, at any shared module's top level — the ML core is reused
lazily, only inside the tab being wired (see ``docs/velum/ARCHITECTURE.md``).
"""
