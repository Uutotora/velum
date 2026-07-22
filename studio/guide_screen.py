"""Velum — the Guide & Docs screen.

The in-app documentation surface: a searchable article list on the left, the
selected article rendered on the right — reached from the sidebar's
"Guide & Docs" footer row and from Home's "Documentation" / "Getting started
guide" resource links (``studio/screens.py``). Content lives in
``guide_content.py``; this module only renders it.

Composed entirely from the existing atoms (``components.py``) and plain
``QLabel``/``QFrame`` — the same idiom every other screen in ``studio/``
uses — rather than a rich-text engine, so typography and colour stay on the
same tokens as the rest of the app in both themes.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QLineEdit,
    QStackedWidget,
)

from studio import icons
from studio import theme
from studio import guide_content
from studio import project_controller
from studio.components import Accordion, GroupLabel, PillButton, bare_widget as _bare, hline, label
from studio.screens import page_header, scroll

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _inline(text: str) -> str:
    """Render the tiny ``**bold**`` markup guide_content articles use."""
    return _BOLD_RE.sub(r"<b>\1</b>", text)


def _prose_label(text: str, size: float, color: str, weight: int = 400) -> QLabel:
    lb = QLabel(_inline(text))
    lb.setTextFormat(Qt.TextFormat.RichText)
    lb.setWordWrap(True)
    lb.setStyleSheet(f"color:{color}; font-size:{size}px; font-weight:{weight}; line-height:150%;")
    return lb


# ── block renderers ──────────────────────────────────────────────────────────
def _heading(t: dict, text: str) -> QWidget:
    lb = label(text, 15.5, t["text"], 600, -0.2)
    lb.setContentsMargins(0, 10, 0, 0)
    return lb


def _para(t: dict, text: str) -> QWidget:
    return _prose_label(text, 13.5, t["text_subtle"])


def _bullets(t: dict, items: list[str]) -> QWidget:
    w = _bare()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(7)
    for item in items:
        row = QHBoxLayout()
        row.setSpacing(9)
        dot = QLabel("•")
        dot.setFixedWidth(12)
        dot.setStyleSheet(f"color:{t['text_muted']}; font-size:13.5px;")
        row.addWidget(dot, alignment=Qt.AlignmentFlag.AlignTop)
        row.addWidget(_prose_label(item, 13.5, t["text_subtle"]), 1)
        v.addWidget(_bare(row))
    return w


def _callout(t: dict, text: str) -> QFrame:
    box = QFrame()
    box.setObjectName("GuideCallout")
    box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    # #GuideCallout{...}, not a bare/unqualified rule: an unqualified
    # setStyleSheet() cascades background/border/border-radius to every
    # descendant widget too (QLabel paints frame properties natively, being
    # a QFrame subclass) — invisible when the fill is opaque and identical,
    # but glaring here since primary_weak is translucent and doubles up.
    # Scoping to this object's own name is what HomeScreen._quick_card's
    # #QCard / ProjectsScreen's #PCard already do for the same reason.
    box.setStyleSheet(
        f"#GuideCallout{{background:{t['primary_weak']}; border:1px solid {t['primary_line']}; border-radius:14px;}}")
    v = QVBoxLayout(box)
    v.setContentsMargins(16, 14, 16, 14)
    v.setSpacing(6)
    head = QHBoxLayout()
    head.setSpacing(7)
    ic = QLabel()
    ic.setPixmap(icons.pixmap("spark", t["primary"], 13))
    head.addWidget(ic)
    head.addWidget(label("Note", 12.5, t["primary"], 600))
    head.addStretch(1)
    v.addLayout(head)
    v.addWidget(_prose_label(text, 12.5, t["text_subtle"]))
    return box


def _step_row(t: dict, index: int, step: guide_content.Step,
              run_action: Callable[[Optional[str]], None]) -> QFrame:
    card = QFrame()
    card.setObjectName("GuideStep")
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    # surface2 ("elevated fill"), not inset ("recessed well" — meant for
    # input fields, darker than the page background itself): at this width
    # inset read as a hole cut into the card revealing the canvas behind it,
    # not a distinct raised row.
    card.setStyleSheet(f"#GuideStep{{background:{t['surface2']}; border:1px solid {t['border']}; border-radius:14px;}}")
    row = QHBoxLayout(card)
    row.setContentsMargins(16, 14, 16, 14)
    row.setSpacing(14)

    badge = QLabel(str(index))
    badge.setFixedSize(26, 26)
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setStyleSheet(
        f"background:{t['signal_weak']}; color:{t['signal']}; border-radius:13px;"
        f"font-family:{theme.MONO}; font-size:12px; font-weight:700;")
    row.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)

    col = QVBoxLayout()
    col.setSpacing(3)
    col.addWidget(label(step.title, 13.5, t["text"], 600))
    col.addWidget(_prose_label(step.body, 12.5, t["text_muted"]))
    row.addLayout(col, 1)

    if step.action_label:
        btn = PillButton(step.action_label, t, "ghost", small=True)
        btn.clicked.connect(lambda _=False, a=step.action: run_action(a))
        row.addWidget(btn, alignment=Qt.AlignmentFlag.AlignVCenter)
    return card


def _steps(t: dict, steps: list[guide_content.Step],
           run_action: Callable[[Optional[str]], None]) -> QWidget:
    w = _bare()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(10)
    for i, step in enumerate(steps, start=1):
        v.addWidget(_step_row(t, i, step, run_action))
    return w


def _shortcuts_block(t: dict, shortcuts: list[guide_content.Shortcut]) -> QWidget:
    w = _bare()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(8)
    for sc in shortcuts:
        row = QFrame()
        row.setObjectName("GuideShortcut")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet(f"#GuideShortcut{{background:{t['surface2']}; border:1px solid {t['border']}; border-radius:10px;}}")
        h = QHBoxLayout(row)
        h.setContentsMargins(14, 11, 14, 11)
        h.setSpacing(12)
        keys = QHBoxLayout()
        keys.setSpacing(4)
        for k in sc.keys:
            pill = QLabel(k)
            pill.setStyleSheet(
                f"background:{t['surface2']}; color:{t['text']}; border:1px solid {t['border_strong']};"
                f"border-radius:6px; padding:3px 8px; font-family:{theme.MONO}; font-size:12px; font-weight:600;")
            keys.addWidget(pill)
        keys.addStretch(1)
        keys_wrap = _bare(keys)
        keys_wrap.setObjectName("GuideShortcutKeys")  # findable for the bare-widget regression test
        keys_wrap.setFixedWidth(150)
        h.addWidget(keys_wrap)
        h.addWidget(_prose_label(sc.desc, 12.5, t["text_subtle"]), 1)
        v.addWidget(row)
    return w


def _faq_block(t: dict, items: list[guide_content.FAQItem]) -> QWidget:
    w = _bare()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(8)
    for item in items:
        acc = Accordion(item.q, t, lead="diagnose", open_=False, caps=False)
        acc.add(_prose_label(item.a, 12.5, t["text_subtle"]))
        v.addWidget(acc)
    return w


def _table_block(t: dict, headers: list[str], rows: list[list[str]]) -> QFrame:
    card = QFrame()
    card.setObjectName("GuideTable")
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(f"#GuideTable{{background:{t['surface2']}; border:1px solid {t['border']}; border-radius:12px;}}")
    v = QVBoxLayout(card)
    v.setContentsMargins(16, 13, 16, 5)
    v.setSpacing(8)
    hrow = QHBoxLayout()
    for h in headers:
        hrow.addWidget(label(h.upper(), 10, t["text_muted"], 600, 0.5), 1)
    v.addLayout(hrow)
    v.addWidget(hline(t))
    for values in rows:
        r = QHBoxLayout()
        for i, val in enumerate(values):
            lb = QLabel(val)
            lb.setWordWrap(True)
            if i == 0:
                lb.setStyleSheet(f"color:{t['text']}; font-size:12.5px; font-weight:600;")
            else:
                lb.setStyleSheet(f"color:{t['text_subtle']}; font-size:12.5px;")
            r.addWidget(lb, 1)
        r.setContentsMargins(0, 9, 0, 9)
        row_wrap = _bare(r)
        row_wrap.setObjectName("GuideTableRow")  # findable for the bare-widget regression test
        v.addWidget(row_wrap)
        v.addWidget(hline(t))
    return card


def _render_block(t: dict, block: tuple, run_action: Callable[[Optional[str]], None]) -> QWidget:
    kind = block[0]
    if kind == "h":
        return _heading(t, block[1])
    if kind == "p":
        return _para(t, block[1])
    if kind == "ul":
        return _bullets(t, block[1])
    if kind == "callout":
        return _callout(t, block[1])
    if kind == "steps":
        return _steps(t, block[1], run_action)
    if kind == "shortcuts":
        return _shortcuts_block(t, block[1])
    if kind == "faq":
        return _faq_block(t, block[1])
    if kind == "table":
        return _table_block(t, block[1], block[2])
    raise ValueError(f"unknown guide block kind: {kind!r}")


def _render_article(t: dict, article: guide_content.Article,
                     run_action: Callable[[Optional[str]], None]) -> QWidget:
    body = _bare()
    v = QVBoxLayout(body)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(13)
    v.addWidget(label(article.title, 22, t["text"], 600, -0.4))
    sub = _prose_label(article.summary, 13.5, t["text_muted"])
    v.addWidget(sub)
    for blk in article.blocks:
        v.addWidget(_render_block(t, blk, run_action))
    v.addStretch(1)
    return body


# ── the screen ───────────────────────────────────────────────────────────────
class GuideScreen(QWidget):
    """Searchable article nav (left) + the selected article (right).

    Constructor mirrors ``HomeScreen``/``ProjectsScreen`` exactly (same 4
    callbacks) so ``app.py`` can wire it the same way, and so the Getting
    Started walkthrough can trigger the *same real actions* Home's quick
    cards do — no separate, parallel plumbing.
    """

    NAV_WIDTH = 264

    def __init__(self, t: dict, controller: "project_controller.ProjectController",
                 on_navigate: Callable[[str], None], on_open: Callable[[str], None],
                 on_new_project: Callable[[], None]):
        super().__init__()
        self._t = t
        self._controller = controller
        self._nav = on_navigate
        self._open = on_open
        self._new_project = on_new_project
        self._current_id = guide_content.DEFAULT_ARTICLE_ID
        self._nav_rows: dict[str, QFrame] = {}
        self._nav_labels: dict[str, QLabel] = {}
        self._article_pages: dict[str, QWidget] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        n_articles = len(guide_content.ARTICLES)
        n_topics = len({a.category for a in guide_content.ARTICLES})
        # Every other full screen (Home/Projects/Segment/Models/Dashboard) is a
        # sidebar-nav peer, nothing to "close". Guide is reached the same way
        # but is conceptually a utility panel like Assistant/Logs (which do
        # have an explicit close) — without this, there was no way back to
        # where you were except remembering to click a sidebar item yourself.
        self._close_btn = PillButton("Close", t, "ghost", "close", small=True)
        self._close_btn.clicked.connect(lambda: self._nav("home"))
        outer.addWidget(page_header(
            "Guide & Docs", f"{n_articles} articles across {n_topics} topics", t, self._close_btn))

        body = _bare()
        row = QHBoxLayout(body)
        row.setContentsMargins(34, 4, 34, 40)
        row.setSpacing(20)
        row.addWidget(self._build_nav(), 0)
        row.addWidget(self._build_content(), 1)
        outer.addWidget(body, 1)

        self.open_article(self._current_id)

    # ── nav rail ─────────────────────────────────────────────────────────────
    def _build_nav(self) -> QWidget:
        # No enclosing card here (no background/border/radius wrapper at
        # all) — a big panel-sized "card" behind content that already has
        # its own distinct rows/blocks is a redundant extra layer of boxing
        # ("looks like unstyled HTML div soup" was the exact complaint).
        # Home/Projects never wrap a whole *column* in a card either — only
        # the individual cards within it are boxed, floating directly on
        # the page canvas. Matching that here: just a plain, fixed-width
        # layout column: the search field and nav rows provide their own
        # (correct) visual structure without a container drawn around them.
        t = self._t
        panel = _bare()
        panel.setFixedWidth(self.NAV_WIDTH)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        search = QLineEdit()
        search.setPlaceholderText("Search the guide…")
        search.setClearButtonEnabled(True)
        search.addAction(icons.icon("search", t["text_muted"], 15), QLineEdit.ActionPosition.LeadingPosition)
        search.textChanged.connect(self._on_search)
        v.addWidget(search)

        inner = _bare()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(0, 4, 8, 8)
        iv.setSpacing(1)
        self._category_headers: dict[str, QLabel] = {}
        current_cat: Optional[str] = None
        for article in guide_content.ARTICLES:
            if article.category != current_cat:
                current_cat = article.category
                hdr = GroupLabel(current_cat, t)
                hdr.setContentsMargins(8, 12, 0, 4)
                self._category_headers[current_cat] = hdr
                iv.addWidget(hdr)
            row, lb = self._nav_row(article)
            self._nav_rows[article.id] = row
            self._nav_labels[article.id] = lb
            iv.addWidget(row)
        iv.addStretch(1)
        v.addWidget(scroll(inner), 1)
        return panel

    def _nav_row(self, article: guide_content.Article) -> tuple[QFrame, QLabel]:
        t = self._t
        row = QFrame()
        row.setObjectName("GuideNavRow")
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(9, 8, 9, 8)
        lay.setSpacing(9)
        ic = QLabel()
        ic.setPixmap(icons.pixmap(article.icon, t["text_muted"], 15))
        lay.addWidget(ic)
        lb = QLabel(article.title)
        lb.setWordWrap(True)
        lay.addWidget(lb, 1)
        row.mouseReleaseEvent = lambda e, aid=article.id: self.open_article(aid)
        return row, lb

    def _restyle_nav_row(self, article_id: str) -> None:
        # #GuideNavRow{...}, not a bare `QFrame{...}` type selector — a type
        # selector still cascades to every descendant (only an id/#name
        # selector stops at this widget), which painted a second rounded
        # box around the row's own label. See _callout's comment.
        t = self._t
        row = self._nav_rows[article_id]
        lb = self._nav_labels[article_id]
        if article_id == self._current_id:
            row.setStyleSheet(
                f"#GuideNavRow{{background:{t['primary_weak']}; border:1px solid {t['primary_line']}; border-radius:9px;}}")
            lb.setStyleSheet(f"color:{t['primary']}; font-size:12.5px; font-weight:600;")
        else:
            row.setStyleSheet(
                "#GuideNavRow{background:transparent; border:1px solid transparent; border-radius:9px;}"
                f"#GuideNavRow:hover{{background:{t['surface2']};}}")
            lb.setStyleSheet(f"color:{t['text_subtle']}; font-size:12.5px; font-weight:500;")

    def _on_search(self, text: str) -> None:
        q = text.strip().lower()
        visible_cats: set[str] = set()
        for article in guide_content.ARTICLES:
            haystack = " ".join([article.title, article.summary, *article.keywords]).lower()
            visible = not q or q in haystack
            self._nav_rows[article.id].setVisible(visible)
            if visible:
                visible_cats.add(article.category)
        for cat, hdr in self._category_headers.items():
            hdr.setVisible(cat in visible_cats)

    # ── content ──────────────────────────────────────────────────────────────
    def _build_content(self) -> QWidget:
        # No enclosing card here either (see _build_nav) — the article text
        # and its individual content blocks (steps/table/shortcuts/callout/
        # FAQ, each already boxed on its own) sit directly on the page
        # canvas, the same way Home's body text does.
        t = self._t
        self._content_stack = QStackedWidget()
        for article in guide_content.ARTICLES:
            inner = _render_article(t, article, self._run_action)
            wrap = _bare()
            wv = QVBoxLayout(wrap)
            wv.setContentsMargins(0, 4, 10, 0)  # top breathing room, scrollbar clearance
            wv.addWidget(inner)
            page = scroll(wrap)
            self._article_pages[article.id] = page
            self._content_stack.addWidget(page)
        return self._content_stack

    # ── navigation ───────────────────────────────────────────────────────────
    def open_article(self, article_id: str) -> None:
        """Show ``article_id`` in the content pane and highlight its nav row.

        Unknown ids are ignored (the screen just stays where it was) — a
        stale or mistyped "article:<id>" action shouldn't crash the app.
        """
        if article_id not in guide_content.ARTICLES_BY_ID:
            return
        self._current_id = article_id
        self._content_stack.setCurrentWidget(self._article_pages[article_id])
        for aid in self._nav_rows:
            self._restyle_nav_row(aid)

    def _run_action(self, action: Optional[str]) -> None:
        """Dispatch a ``Step.action`` — see ``guide_content.Step`` for the vocabulary."""
        if not action:
            return
        if action == "new_project":
            self._new_project()
        elif action.startswith("article:"):
            self.open_article(action.split(":", 1)[1])
        else:
            self._nav(action)
