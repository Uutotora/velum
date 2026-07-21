# docs/ — documentation map

Two levels of docs live here. Start with the one that matches your task.

## Project-wide (this folder)

The whole repo — engines, ML core, server, packaging, strategy.

| File | Job |
| --- | --- |
| [`BACKLOG.md`](BACKLOG.md) | **The task queue.** What's next, with acceptance criteria. Start here. |
| [`AUDIT_2026.md`](AUDIT_2026.md) | Strategic gap analysis (why it matters, scored). A point-in-time snapshot with dated addenda. |
| [`CHANGELOG.md`](CHANGELOG.md) | What actually shipped, dated. Every meaningful change gets a line. |
| [`AGENT_KICKOFF_PROMPT.md`](AGENT_KICKOFF_PROMPT.md) | Paste this to start a cold agent session on the repo. |
| [`app_icon/`](app_icon/) · [`screenshots/`](screenshots/) | Icon source art and README screenshots. |

## Velum app-specific ([`velum/`](velum/))

The desktop app (`studio/` package) — its design system, screen wiring, and
its own backlog/changelog. See [`velum/README.md`](velum/README.md) for the
reading order (OVERVIEW → DESIGN → ARCHITECTURE → BACKLOG → ROADMAP →
CHANGELOG → AGENT_PROMPT).

## Which backlog / changelog?

There are two of each on purpose:

- **`docs/BACKLOG.md` / `docs/CHANGELOG.md`** — repo-wide: engines, ML core,
  `server/`, packaging, CI, strategy.
- **`docs/velum/BACKLOG.md` / `docs/velum/CHANGELOG.md`** — the app itself:
  screens, UI, design, tab wiring.

When in doubt, a change to `studio/` goes in the Velum docs; anything else
(engines, `velum_core/`, `server/`, build) goes in the project-wide docs.
The authoritative "read this first" for agents is still the root
[`AGENTS.md`](../AGENTS.md).
