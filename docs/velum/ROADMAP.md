# Roadmap — Velum

Phases from design skeleton to shippable product. Each phase closes when its
`BACKLOG.md` items are done and logged in `CHANGELOG.md`.

### Phase 0 — Design skeleton ✅
Native, static, logic-free reproduction of the mockup. Frameless rounded
window, all screens + overlays, design tokens + UI kit. Launches on PyQt6
alone.

### Phase 1 — A usable shell ✅ (2026-07-09)
Projects are real (data model + store), the new-project flow works, and the
**Segment** workspace is wired: **our own** canvas (explicitly not embedded
napari — see `ARCHITECTURE.md`), our own evented `LayerList` driving the
Layers panel, real predict + Results reusing the classic app's ML core. You
can create a project, load images, segment, and read results — end to end,
in the new UI.

### Phase 2 — Differentiation ✅ (2026-07-20)
Models & Train and Dashboard are wired (2026-07-09); Assistant is wired
(2026-07-18) — a real chat (offline diagnostics, Ollama, or any
OpenAI-compatible Custom API) that can act on the Segment tab. Logs is
wired (2026-07-19) — a real, live stream from `studio/log_bus.py`, the
studio-wide log every tab's controllers and the app shell itself feed. The
⌘K command palette is wired (2026-07-20) — a real Spotlight-style action
registry (`studio/command_registry.py`) spanning every tab, fuzzy search,
full keyboard navigation; `⌘L` also now opens Logs. **P1 is fully done** —
every backlog item that makes Studio more than a viewer is real.

### Phase 3 — Polish & platform (current)
Live theme repaint + persistence, Guide/onboarding, Settings, native rounded
corners, and a packaged `.app`. The 1.0 finish.

### North star
Tens of thousands of microscopists using Studio daily; the reference
open-source tool for cell segmentation. Every decision is made for that.
