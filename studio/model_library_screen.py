"""Velum — the Model Library screen.

A browsable catalog of segmentation models: filter by family, download real
weights straight into the folders the engines read from (so a downloaded model
is instantly usable in Segment), bring your own checkpoint, and see the models
you've already trained/imported. The Qt-thin view over
``ModelLibraryController`` — all catalog/download/import logic lives there and
is unit-tested without Qt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QObject, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget, QProgressBar,
    QFileDialog, QMenu,
)

from studio import model_library as ml
from studio.components import (
    Chip, PillButton, IconButton, hline, soft_shadow, bare_widget, label,
)
from studio.model_library_controller import ModelLibraryController, CatalogEntry
from studio.screens import page_header, scroll


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
        elif item.layout() is not None:
            _clear_layout(item.layout())


class _DL(QObject):
    """Marshals a background download's callbacks onto the GUI thread."""
    progress = pyqtSignal(int, int)
    done = pyqtSignal(str)
    error = pyqtSignal(str)


class ModelLibraryScreen(QWidget):
    def __init__(self, t: dict, controller: ModelLibraryController,
                 on_toast: Callable[[str, str], None],
                 on_navigate: Optional[Callable[[str], None]] = None):
        super().__init__()
        self._t = t
        self._c = controller
        self._toast = on_toast
        self._on_navigate = on_navigate
        self._family: Optional[str] = None          # active filter (None = all)
        self._cards: dict[str, QWidget] = {}         # model_id → card (for progress)
        self._action_btns: dict[str, PillButton] = {}  # model_id → its action button
        self._active: dict[str, _DL] = {}            # model_id → live download signals

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        import_btn = PillButton("Import model", t, "ghost", "download", small=True)
        import_btn.clicked.connect(self._import_menu)
        root.addWidget(page_header(
            "Model Library",
            "Find and download any segmentation model, or bring your own — "
            "everything runs locally on your machine.", t, action=import_btn))

        self._body = QVBoxLayout()
        self._body.setContentsMargins(34, 4, 34, 28)
        self._body.setSpacing(18)
        body_wrap = bare_widget(self._body)
        root.addWidget(scroll(body_wrap), 1)
        self.refresh()

    # ── public API ───────────────────────────────────────────────────────────
    def refresh(self) -> None:
        _clear_layout(self._body)
        self._cards.clear()
        self._action_btns.clear()
        self._body.addWidget(self._filter_row())
        self._body.addWidget(self._catalog_grid())
        local = self._c.local_models()
        if local:
            self._body.addWidget(self._your_models(local))
        self._body.addStretch(1)

    def open_import(self) -> None:
        """Public entry (⌘K / command palette)."""
        self._import_menu()

    # ── filter chips ─────────────────────────────────────────────────────────
    def _filter_row(self) -> QWidget:
        t = self._t
        wrap = bare_widget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._add_filter_chip(row, None, "All models")
        for key, lbl in self._c.families():
            self._add_filter_chip(row, key, lbl)
        row.addStretch(1)
        return wrap

    def _add_filter_chip(self, row, key: Optional[str], text: str) -> None:
        active = self._family == key
        chip = Chip(text, self._t, "primary" if active else "muted")
        chip.setCursor(Qt.CursorShape.PointingHandCursor)

        def _pick(_e, k=key):
            self._family = k
            self.refresh()
        chip.mousePressEvent = _pick
        row.addWidget(chip)

    # ── catalog ──────────────────────────────────────────────────────────────
    def _catalog_grid(self) -> QWidget:
        wrap = bare_widget()
        grid = QGridLayout(wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)
        entries = self._c.catalog_entries(self._family)
        for i, entry in enumerate(entries):
            card = self._card(entry)
            self._cards[entry.model.id] = card
            grid.addWidget(card, i // 2, i % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return wrap

    def _card(self, entry: CatalogEntry) -> QWidget:
        t = self._t
        m = entry.model
        card = QFrame()
        card.setObjectName("MLCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(
            f"QFrame#MLCard{{background:{t['surface']};border:1px solid {t['border']};"
            f"border-radius:14px;}}")
        soft_shadow(card, 14, 20, 3)
        v = QVBoxLayout(card)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(9)

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(Chip(m.family_label, t, "primary"))
        top.addWidget(Chip(m.domain, t, "muted"))
        top.addStretch(1)
        if entry.installed:
            top.addWidget(Chip("✓ Installed", t, "success"))
        elif not m.builtin:
            top.addWidget(label(m.size_label, 11, t["text_muted"]))
        v.addLayout(top)

        v.addWidget(label(m.name, 16, t["text"], 600))
        desc = label(m.description, 12.5, t["text_muted"])
        desc.setWordWrap(True)
        v.addWidget(desc)

        meta = QHBoxLayout()
        meta.setSpacing(14)
        meta.addWidget(label(f"License: {m.license}", 11, t["text_subtle"]))
        meta.addStretch(1)
        v.addLayout(meta)

        # per-card progress bar (hidden until a download starts)
        bar = QProgressBar()
        bar.setObjectName(f"bar_{m.id}")
        bar.setFixedHeight(6)
        bar.setTextVisible(False)
        bar.hide()
        v.addWidget(bar)
        card.setProperty("bar", bar)

        v.addWidget(hline(t))
        actions = QHBoxLayout()
        actions.setSpacing(8)
        info_btn = PillButton("Details", t, "ghost", "chevron", small=True)
        info_btn.clicked.connect(lambda _=False, url=m.homepage: QDesktopServices.openUrl(QUrl(url)))
        actions.addWidget(info_btn)
        actions.addStretch(1)
        actions.addWidget(self._primary_action(entry))
        v.addLayout(actions)
        return card

    def _primary_action(self, entry: CatalogEntry) -> QWidget:
        t = self._t
        m = entry.model
        if m.builtin:
            if entry.installed:
                return Chip("Ready to use", t, "success")
            hint = PillButton("Needs Cellpose", t, "ghost", small=True)
            hint.setEnabled(False)
            return hint
        if entry.installed:
            btn = PillButton("Reveal", t, "ghost", "folder", small=True)
            btn.clicked.connect(lambda _=False, mid=m.id: self._reveal(mid))
            return btn
        btn = PillButton("Download", t, "primary", "download", small=True)
        btn.clicked.connect(lambda _=False, mid=m.id: self._start_download(mid))
        self._action_btns[m.id] = btn
        return btn

    # ── download ─────────────────────────────────────────────────────────────
    # The download runs on a background thread; its callbacks are marshalled to
    # the GUI thread via a per-download ``_DL`` QObject's signals. The card is
    # NOT rebuilt while a download is live (that would destroy the progress bar
    # the worker's closure still writes to) — instead the one action button
    # flips Download↔Cancel in place, and a full ``refresh()`` runs only once
    # the download has finished (on done/error), after the last signal fired.
    def _start_download(self, model_id: str) -> None:
        if model_id in self._active:
            return
        card = self._cards.get(model_id)
        bar: Optional[QProgressBar] = card.property("bar") if card else None
        btn = self._action_btns.get(model_id)
        sig = _DL()
        self._active[model_id] = sig
        if bar is not None:
            bar.setRange(0, 0)  # indeterminate until the first byte arrives
            bar.show()
        if btn is not None:
            btn.setText("Cancel")
            try:
                btn.clicked.disconnect()
            except TypeError:
                pass
            btn.clicked.connect(lambda _=False: self._c.cancel_download())

        def _on_progress(done: int, total: int) -> None:
            if bar is None:
                return
            if total > 0:
                bar.setRange(0, total)
                bar.setValue(done)
            else:
                bar.setRange(0, 0)
        sig.progress.connect(_on_progress)
        sig.done.connect(lambda p, mid=model_id: self._on_done(mid, p))
        sig.error.connect(lambda e, mid=model_id: self._on_error(mid, e))

        self._c.download_async(
            model_id,
            on_progress=lambda d, tot: sig.progress.emit(d, tot),
            on_done=lambda p: sig.done.emit(str(p)),
            on_error=lambda e: sig.error.emit(e))

    def _on_done(self, model_id: str, path: str) -> None:
        self._active.pop(model_id, None)
        m = self._c.get(model_id)
        self._toast("Model ready", f"{m.name if m else model_id} downloaded — "
                    "you can select it in Segment now.")
        self.refresh()

    def _on_error(self, model_id: str, msg: str) -> None:
        self._active.pop(model_id, None)
        self._toast("Download stopped", msg)
        self.refresh()

    def _reveal(self, model_id: str) -> None:
        p = self._c.dest_path(model_id)
        if p and p.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.parent)))

    # ── your models ──────────────────────────────────────────────────────────
    def _your_models(self, local) -> QWidget:
        t = self._t
        wrap = bare_widget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        v.addWidget(label("Your models", 15, t["text"], 600))
        sub = label("Trained here or imported — usable in Segment as the "
                    "CellSeg1 “Model”, or as a Cellpose checkpoint.",
                    12, t["text_muted"])
        sub.setWordWrap(True)
        v.addWidget(sub)
        for lm in local:
            v.addWidget(self._local_row(lm))
        return wrap

    def _local_row(self, lm) -> QWidget:
        t = self._t
        row = QFrame()
        row.setObjectName("MLLocal")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet(
            f"QFrame#MLLocal{{background:{t['surface']};border:1px solid {t['border']};"
            f"border-radius:12px;}}")
        h = QHBoxLayout(row)
        h.setContentsMargins(16, 11, 14, 11)
        h.setSpacing(12)
        h.addWidget(Chip(ml.FAMILY_LABELS.get(lm.family, lm.family), t, "muted"))
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(label(lm.name, 13.5, t["text"], 600))
        meta = lm.meta + (f" · {lm.detail}" if lm.detail else "")
        col.addWidget(label(meta, 11.5, t["text_muted"]))
        h.addLayout(col, 1)
        reveal = IconButton("folder", t, tip="Reveal in file manager",
                            on_click=lambda p=lm.path: QDesktopServices.openUrl(
                                QUrl.fromLocalFile(str(Path(p).parent))))
        h.addWidget(reveal)
        return row

    # ── import your own ──────────────────────────────────────────────────────
    def _import_menu(self) -> None:
        t = self._t
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu{{background:{t['surface']}; border:1px solid {t['border_strong']};"
            f"border-radius:10px; padding:6px;}}"
            f"QMenu::item{{color:{t['text']}; padding:6px 14px; border-radius:6px;}}"
            f"QMenu::item:selected{{background:{t['surface2']};}}")
        for key, lbl in (
            (ml.FAMILY_LORA, "CellSeg1 LoRA (.pth)"),
            (ml.FAMILY_SAM_BACKBONE, "SAM backbone (.pth)"),
            (ml.FAMILY_SAM2, "SAM 2 checkpoint (.pt)"),
            (ml.FAMILY_CELLPOSE, "Cellpose checkpoint (.pth)"),
        ):
            act = menu.addAction(f"Import {lbl}")
            act.triggered.connect(lambda _=False, k=key: self._import_file(k))
        menu.exec(self.cursor().pos())

    def _import_file(self, family: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a checkpoint", "",
            "Model checkpoints (*.pth *.pt *.safetensors);;All files (*)")
        if not path:
            return
        try:
            dest = self._c.import_file(path, family)
        except Exception as e:
            self._toast("Import failed", str(e))
            return
        self._toast("Model imported",
                    f"{Path(dest).name} added to your library.")
        self.refresh()
