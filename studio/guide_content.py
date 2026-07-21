"""Velum — in-app documentation & guide content.

Pure content (no Qt): every article, keyboard shortcut and FAQ entry shown by
the Guide & Docs screen (``studio/guide_screen.py``). Kept separate from
``demo.py`` (skeleton stand-in data for tabs not yet wired) because this is
real, shipping content, not a placeholder — the guide screen reads it as its
only data source.

Written for the product's actual audience: microscopists and cell biologists,
not ML engineers (see the repo-root ``AGENTS.md``). Every claim here matches
what's actually wired today (see ``docs/velum/BACKLOG.md`` for what isn't yet).

Each ``Article.blocks`` entry is ``(kind, payload)``, rendered generically by
``guide_screen._render_block``:
  ("h", str)                heading within the article
  ("p", str)                paragraph — "**word**" renders bold
  ("ul", list[str])         bullet list — same inline markup
  ("callout", str)          a highlighted tip/note box
  ("steps", list[Step])     numbered, actionable steps (Getting Started)
  ("shortcuts", list[Shortcut])   a key-binding reference table
  ("faq", list[FAQItem])    collapsible question/answer pairs
  ("table", list[str], list[list[str]])   a simple headers+rows comparison
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Step:
    """One actionable step in a walkthrough (e.g. Getting Started).

    ``action`` is one of: a nav key Studio's ``navigate()`` already handles
    ("workspace" / "train" / "dashboard" / "projects" / "home"), the special
    tokens "new_project" / "open_sample", or "article:<id>" to jump to
    another guide article — see ``GuideScreen._run_action``.
    """
    title: str
    body: str
    action_label: Optional[str] = None
    action: Optional[str] = None


@dataclass
class Shortcut:
    keys: list[str]
    desc: str


@dataclass
class FAQItem:
    q: str
    a: str


@dataclass
class Article:
    id: str
    title: str
    category: str
    icon: str
    summary: str
    blocks: list[tuple] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


# Nav-group display order, top to bottom.
CATEGORIES: list[str] = [
    "Guide", "Working with projects", "Segmenting", "Training", "Analysis",
]


# The *only* real key bindings today (studio/app.py __init__) — every
# addition there should get a matching row here.
SHORTCUTS: list[Shortcut] = [
    Shortcut(["⌘K", "Ctrl+K"],
              "Open the command palette — jump to any action, project or engine without leaving the image."),
    Shortcut(["⌘T", "Ctrl+T"],
              "Open (or close) the Assistant — diagnose a result or ask a question without leaving the image."),
    Shortcut(["⌘L", "Ctrl+L"],
              "Open (or close) the Logs console — the live stream of every tab's real activity."),
    Shortcut(["Esc"],
              "Close the command palette, or any open drawer/console panel."),
]


FAQ: list[FAQItem] = [
    FAQItem(
        "Does my image data ever leave this computer?",
        "No. Studio runs entirely on this device — models, images and results "
        "all stay in the local **data_store** folder (see “This device” on "
        "Home). Nothing is uploaded anywhere, **unless** you deliberately "
        "connect the Assistant to a remote Custom API — see “The Assistant.” "
        "Even then, only numeric summaries (cell count, size stats, current "
        "settings) and your typed questions are sent — never the image or "
        "mask pixels themselves."),
    FAQItem(
        "Which engine should I use?",
        "**Cellpose-SAM** if you just want a fast, zero-shot answer on a "
        "typical assay. **SAM 2** if you're segmenting a z-stack or "
        "time-lapse. **CellSeg1 · LoRA** if you have one well-annotated "
        "image and want the whole cohort segmented in that exact style. "
        "See “Choosing an engine.”"),
    FAQItem(
        "What image formats can I import?",
        "TIFF, OME-TIFF, ND2, CZI and PNG — drag them in or use the file "
        "picker in “Import Images.” See “Importing images & supported "
        "formats.”"),
    FAQItem(
        "Do I need a GPU?",
        "No. Studio runs on CPU, Apple-silicon MPS or CUDA — whichever this "
        "device has. Home's “This device” card shows which one is active "
        "right now."),
    FAQItem(
        "Is Studio the same thing as the CellSeg1 napari plugin?",
        "Same segmentation engines, same underlying models. Studio is a "
        "newer, standalone shell around that core, built to be a complete "
        "desktop app rather than a napari dock widget — both read and write "
        "ordinary image and mask files, so you're not choosing a side."),
    FAQItem(
        "Where did my New Project images go?",
        "Nowhere — importing doesn't copy or move anything. Studio reads "
        "your files from wherever they already are and keeps every project "
        "organised under data_store. Open the project again from Home or "
        "Projects to get back to it."),
]


ARTICLES: list[Article] = [
    Article(
        id="overview", title="Welcome to Velum", category="Guide",
        icon="guide",
        summary="What Studio is, who it's for, and how the pieces fit together.",
        keywords=["introduction", "about", "start here"],
        blocks=[
            ("p", "Velum is a desktop app for **cell instance "
                  "segmentation** — built for microscopists and cell "
                  "biologists, not ML engineers. Import your images, segment "
                  "them, measure every cell, and compare results across a "
                  "whole project, without leaving one window."),
            ("h", "Everything lives in a project"),
            ("p", "A **project** bundles one image set with the engine and "
                  "settings you're using on it. Create one from Home or "
                  "Projects, then move through Segment, Models & Train and "
                  "Dashboard without re-importing anything."),
            ("ul", [
                "**Home** — quick actions and your most recent projects.",
                "**Projects** — every project you have, searchable and filterable.",
                "**Segment** — the workspace: layers, canvas and results for the active project.",
                "**Models & Train** — one-shot fine-tuning and your trained models.",
                "**Dashboard** — experiment tracking across every run.",
            ]),
            ("h", "Three engines, one workflow"),
            ("p", "Studio ships three interchangeable segmentation engines — "
                  "CellSeg1 · LoRA, Cellpose-SAM and SAM 2 — so the same "
                  "project, the same interface and the same measurements "
                  "work whichever one fits your images. See “Choosing an "
                  "engine.”"),
            ("callout", "Everything runs on this device — no image ever "
                        "leaves your computer. See the FAQ for details."),
        ],
    ),
    Article(
        id="getting-started", title="Getting started", category="Guide",
        icon="run",
        summary="Five steps from a fresh install to your first segmented cohort.",
        keywords=["quickstart", "onboarding", "first project", "new user"],
        blocks=[
            ("steps", [
                Step("Create a project",
                     "Name it, then import your images — drag them in or "
                     "use the file picker. TIFF, OME-TIFF, ND2, CZI and PNG "
                     "are all supported.",
                     action_label="New Project", action="new_project"),
                Step("No images ready yet? Open a sample",
                     "Try the whole workflow on a bundled nuclei, tissue or "
                     "mitosis dataset — nothing to import.",
                     action_label="Open a sample", action="open_sample"),
                Step("Choose an engine",
                     "CellSeg1 · LoRA, Cellpose-SAM or SAM 2 — picked when "
                     "you create the project, changeable any time. Not sure "
                     "which fits your images?",
                     action_label="Choosing an engine", action="article:engines"),
                Step("Segment, measure, refine",
                     "Open the project into the Segment workspace: layers "
                     "on the left, your image in the canvas, engine "
                     "settings and per-cell results on the right.",
                     action_label="Go to Segment", action="workspace"),
                Step("Compare runs in Dashboard",
                     "Every segmentation and training run is tracked, so "
                     "you can compare F1, cell counts and timing across a "
                     "whole project.",
                     action_label="Go to Dashboard", action="dashboard"),
            ]),
        ],
    ),
    Article(
        id="keyboard-shortcuts", title="Keyboard shortcuts", category="Guide",
        icon="target",
        summary="Every key binding Studio currently supports.",
        keywords=["hotkeys", "keys", "command palette", "cmd k"],
        blocks=[
            ("p", "Studio is quick to drive from the keyboard. This list is "
                  "short today and will grow as more of the app is wired up."),
            ("shortcuts", SHORTCUTS),
        ],
    ),
    Article(
        id="faq", title="FAQ & troubleshooting", category="Guide",
        icon="diagnose",
        summary="Common questions about data, engines, formats and hardware.",
        keywords=["help", "problems", "troubleshooting", "questions"],
        blocks=[
            ("p", "Answers to the questions we hear most from microscopists "
                  "trying Studio for the first time."),
            ("faq", FAQ),
        ],
    ),
    Article(
        id="projects", title="Projects & the New Project flow",
        category="Working with projects", icon="projects",
        summary="What a project is, and the three-step wizard that creates one.",
        keywords=["new project", "wizard", "favorites", "favourites", "search", "scope"],
        blocks=[
            ("p", "A project bundles one image set with the engine and "
                  "settings you're using on it — everything downstream "
                  "(segmentation, measurements, training, dashboards) is "
                  "scoped to it."),
            ("h", "Creating a project"),
            ("p", "“+ New Project” walks through three short steps:"),
            ("ul", [
                "**Name your project** — a name and an optional description.",
                "**Import images** — drag and drop, or pick files — TIFF, OME-TIFF, ND2, CZI, PNG.",
                "**Choose an engine** — CellSeg1 · LoRA, Cellpose-SAM or SAM 2.",
            ]),
            ("p", "Finishing the wizard creates the project and opens it "
                  "straight into the Segment workspace."),
            ("h", "Finding a project again"),
            ("p", "Home always shows your four most recent projects. "
                  "**Projects** lists all of them, with a search box, an "
                  "All / Favorites / Shared scope, an engine filter, and a "
                  "grid or list view — star a project to pin it under "
                  "Favorites."),
            ("callout", "Studio doesn't sync or share projects between "
                        "devices yet — “Shared” stays empty until that "
                        "lands."),
        ],
    ),
    Article(
        id="import-formats", title="Importing images & supported formats",
        category="Working with projects", icon="image",
        summary="Drag-and-drop, the file picker, and which formats Studio reads.",
        keywords=["tiff", "ome-tiff", "nd2", "czi", "png", "drag and drop", "file picker"],
        blocks=[
            ("p", "Bring images in from “Import Images” on Home, or the "
                  "Import step of the New Project wizard — drag a folder or "
                  "files in, or use the file picker."),
            ("h", "Supported formats"),
            ("ul", [
                "**TIFF** / **OME-TIFF** — including multi-channel images and z-stacks.",
                "**ND2** — Nikon microscope format.",
                "**CZI** — Zeiss microscope format.",
                "**PNG** — for anything already exported as a flat image.",
            ]),
            ("callout", "Importing doesn't copy or convert anything — "
                        "Studio reads your files from wherever they already "
                        "are."),
        ],
    ),
    Article(
        id="engines", title="Choosing an engine", category="Segmenting",
        icon="models",
        summary="CellSeg1 · LoRA, Cellpose-SAM and SAM 2 — what each is best at.",
        keywords=["cellseg1", "cellpose", "sam2", "sam 2", "lora", "zero-shot", "z-stack", "time-lapse"],
        blocks=[
            ("p", "Every project picks one segmentation engine — you can "
                  "change it later. All three read the same images and "
                  "produce the same kind of output: per-cell instance "
                  "masks, ready for measurement."),
            ("table", ["Engine", "Best for", "Setup"], [
                ["CellSeg1 · LoRA", "Your exact assay, at its most accurate", "One annotated image, a few minutes"],
                ["Cellpose-SAM", "A fast, zero-shot default", "None"],
                ["SAM 2", "Z-stacks & time-lapse, tracked across slices", "None"],
            ]),
            ("h", "CellSeg1 · LoRA"),
            ("p", "Fine-tunes SAM to your exact assay from a single "
                  "annotated image, then applies that fit to the rest of "
                  "the cohort — the most accurate option once trained. See "
                  "“One-shot LoRA training.”"),
            ("h", "Cellpose-SAM"),
            ("p", "A zero-shot generalist — no training, works out of the "
                  "box on most cell types. The right default when you just "
                  "want an answer quickly."),
            ("h", "SAM 2"),
            ("p", "Zero-shot like Cellpose-SAM, but built for volumes: "
                  "z-stacks and time-lapse. Instances are linked across "
                  "slices or frames automatically, so one cell keeps one ID "
                  "through the whole stack."),
            ("callout", "SAM 2 is an optional component — if it isn't "
                        "installed, Studio simply doesn't offer it; every "
                        "other engine keeps working normally."),
        ],
    ),
    Article(
        id="segment-workspace", title="The Segment workspace",
        category="Segmenting", icon="workspace",
        summary="Layers, canvas and inspector — where images become measurements.",
        keywords=["canvas", "layers", "results", "inspector", "viewer", "run"],
        blocks=[
            ("p", "Opening a project lands you in Segment — the workspace "
                  "where images become measurements."),
            ("ul", [
                "**Left — Images & Layers.** Switch between your project's "
                "image list and the layer stack — segmentation, ground "
                "truth, corrections, prompts and each image channel — and "
                "toggle any layer's visibility.",
                "**Centre — canvas.** Your image, with a live legend "
                "(detected / selected counts), a tool strip for selecting "
                "and editing instances, and a viewer bar for 2D ↔ 3D, roll, "
                "transpose, grid and reset-view.",
                "**Right — Segment & Results.** *Segment* holds the engine "
                "and threshold controls plus Run; *Results* shows cell "
                "count, size statistics and the per-cell list once a run "
                "finishes.",
            ]),
            ("callout", "The active project's name and engine follow you "
                        "here from Home or Projects — nothing to re-pick."),
        ],
    ),
    Article(
        id="assistant", title="The Assistant", category="Segmenting",
        icon="assistant",
        summary="Diagnose a result and apply fixes with one click — with an optional local or connected model.",
        keywords=["assistant", "chat", "diagnose", "ollama", "custom api",
                  "openai", "tune", "apply", "suggest"],
        blocks=[
            ("p", "The Assistant turns a segmentation result into concrete, "
                  "one-click fixes. Open it from the sidebar's **Assistant** "
                  "row, or Home's “Ask the Assistant” card.",),
            ("h", "Diagnose — always available, no model needed"),
            ("p", "Click the magnifier button next to the input box. A "
                  "deterministic, offline engine inspects the current image "
                  "and mask — no cells detected, likely over- or "
                  "under-segmentation, low contrast, small cells at a low "
                  "resolution — and posts one card per finding with an "
                  "**Apply** or **Apply & re-run** button that writes "
                  "straight into this project's settings."),
            ("h", "Chat — three backends"),
            ("p", "Type a question and the Assistant answers using whichever "
                  "backend is selected in the collapsed **Model** section "
                  "above the chat:"),
            ("ul", [
                "**Offline** — the same diagnostic engine as the magnifier "
                "button, phrased as an answer. Always on, nothing to set up.",
                "**Ollama** — a locally-installed language model "
                "(ollama.com) sees your current image/mask statistics and "
                "settings and can reason about them conversationally. "
                "Download one of the recommended models right from this "
                "panel, or “Tune for CellSeg1” to bake a version pinned to "
                "this app's domain.",
                "**Custom API** — any OpenAI-compatible server: your own "
                "hosted model, a local one like LM Studio or vLLM, or a "
                "cloud provider. Set a base URL, an optional API key, and a "
                "model id, then “Test connection.”",
            ]),
            ("p", "Whichever backend replies, a suggested parameter change "
                  "still becomes an Apply / Apply & re-run card — the chat "
                  "is a second way to reach the same one-click fixes, not a "
                  "separate feature."),
            ("callout", "Offline and Ollama never send anything off this "
                        "device. Custom API is the one exception — see the "
                        "FAQ for exactly what that sends."),
        ],
    ),
    Article(
        id="training", title="One-shot LoRA training", category="Training",
        icon="models",
        summary="Fine-tune SAM to your exact assay from a single annotated image.",
        keywords=["lora", "fine-tune", "finetune", "train", "rank", "epochs", "adapter"],
        blocks=[
            ("p", "CellSeg1's signature trick: fine-tune SAM to your exact "
                  "assay from a **single annotated image**, in minutes, on "
                  "this device — no ML background needed."),
            ("h", "What you need"),
            ("ul", [
                "One image with cells annotated (ground truth).",
                "A SAM backbone — ViT-H by default.",
                "A few minutes on this device's GPU, Apple-silicon MPS, or CPU.",
            ]),
            ("h", "What you get"),
            ("p", "A trained adapter you can apply to every other image in "
                  "the project — usually a noticeably better fit than a "
                  "zero-shot engine on assays with an unusual stain, "
                  "density or cell shape."),
            ("p", "Start a run from **Models & Train**: pick the annotated "
                  "image, backbone, LoRA rank and epoch count, then track "
                  "progress in Dashboard."),
            ("callout", "Rank is the main knob: lower (8) trains faster and "
                        "generalises more; higher (16+) fits harder assays "
                        "more tightly but needs a bit more data to avoid "
                        "overfitting."),
        ],
    ),
    Article(
        id="dashboard", title="Dashboard & experiment tracking",
        category="Analysis", icon="dashboard",
        summary="Compare runs, loss curves and F1 across every model you've trained.",
        keywords=["aim", "experiment tracking", "runs", "loss", "f1"],
        blocks=[
            ("p", "Every segmentation and training run is tracked "
                  "automatically — Dashboard is where you compare them."),
            ("ul", [
                "**Training loss** — watch a run converge.",
                "**F1 across runs** — see which model, engine or setting "
                "actually did better on held-out ground truth.",
                "**Runs table** — engine, F1, cell count, duration and "
                "when, for every run in this project.",
            ]),
            ("p", "Tracking is powered by Aim under the hood — “Open in "
                  "Aim” gives you the full, interactive view for deeper "
                  "digging."),
            ("callout", "A metric is never a substitute for looking at the "
                        "image — always spot-check a few cells before "
                        "trusting a number."),
        ],
    ),
]

ARTICLES_BY_ID: dict[str, Article] = {a.id: a for a in ARTICLES}
DEFAULT_ARTICLE_ID = "overview"
