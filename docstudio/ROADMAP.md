# Roadmap — CellSeg1 Studio

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

### Phase 2 — Differentiation (current)
Models & Train and Dashboard are wired (2026-07-09). Still open: Assistant
(diagnostics), Logs, and the ⌘K command palette. The features that make
Studio more than a viewer.

### Phase 3 — Polish & platform
Live theme repaint + persistence, Guide/onboarding, Settings, native rounded
corners, and a packaged `.app`. The 1.0 finish.

### North star
Tens of thousands of microscopists using Studio daily; the reference
open-source tool for cell segmentation. Every decision is made for that.
