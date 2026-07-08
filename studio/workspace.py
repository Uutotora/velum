"""CellSeg1 Studio — the Workspace (Segment) screen (static design skeleton).

The signature screen: an adapted-napari **Layers** panel (list + full controls),
the image **canvas** (nuclei stand-in with overlays and a napari-style viewer
bar), and the **inspector** (Segment settings · Results). Still static — the
tab-by-tab wiring (real napari layers, real predict/results) is tracked in
``docstudio/BACKLOG.md`` — except the top-bar breadcrumb + engine chip, which
reflect the real "active project" shared from the Projects tab
(``set_active_project``).
"""
from __future__ import annotations

import html
from typing import Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QStackedWidget, QToolButton, QLineEdit, QSizePolicy, QScrollArea,
)

from studio import icons
from studio import theme, demo
from studio.project import ENGINE_LABELS, ENGINE_KIND, Project
from studio.paint import nuclei_pixmap, NucleiView
from studio.components import (
    Chip, Badge, PillButton, IconButton, SelectBox, Toggle, Slider, Stepper,
    SegControl, StatTile, FieldRow, GroupLabel, Accordion, hline, label,
)

LAYER_TYPE_ICON = {"labels": "layers", "shapes": "shapes", "points": "points", "image": "image"}


def _scroll(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setWidget(inner)
    return sa


class WorkspaceScreen(QWidget):
    def __init__(self, t: dict):
        super().__init__()
        self._t = t
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._topbar())
        self.set_active_project(None)

        main = QWidget()
        row = QHBoxLayout(main)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._left_panel())
        row.addWidget(self._viewport(), 1)
        row.addWidget(self._inspector())
        outer.addWidget(main, 1)

    # ── top bar ──────────────────────────────────────────────────────────────
    def _topbar(self) -> QWidget:
        t = self._t
        bar = QWidget()
        bar.setStyleSheet(f"background:{t['surface']};")
        bar.setFixedHeight(52)
        row = QHBoxLayout(bar)
        row.setContentsMargins(18, 0, 18, 0)
        row.setSpacing(12)
        self._crumb = QLabel()
        self._crumb.setStyleSheet("font-size:13px; font-weight:600;")
        row.addWidget(self._crumb)
        self._chip_row = row
        self._engine_chip = Chip("", t, "muted")
        row.addWidget(self._engine_chip)
        row.addStretch(1)
        row.addWidget(PillButton("Export", t, "ghost", "export", small=True))
        row.addWidget(PillButton("Run", t, "primary", "run", small=True))
        bottom = QFrame()
        bottom.setFixedHeight(1)
        bottom.setStyleSheet(f"background:{t['border']};")
        wrap = QWidget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(0)
        wl.addWidget(bar)
        wl.addWidget(bottom)
        return wrap

    # ── active project (shared from the Projects tab) ──────────────────────────
    def set_active_project(self, project: Optional[Project]) -> None:
        """Reflect the real active project in the breadcrumb + engine chip."""
        t = self._t
        if project is None:
            name, engine_key, engine_label = "No project selected", None, "No project"
        else:
            name, engine_key = project.name, project.engine
            engine_label = ENGINE_LABELS.get(engine_key, engine_key)
        self._crumb.setText(
            f"<span style='color:{t['text_muted']}'>Projects</span>"
            f"<span style='color:{t['border_strong']}'>&nbsp;/&nbsp;</span>"
            f"<span style='color:{t['text']}'>{html.escape(name)}</span>")
        kind = ENGINE_KIND.get(engine_key, "muted") if engine_key else "muted"
        idx = self._chip_row.indexOf(self._engine_chip)
        self._chip_row.removeWidget(self._engine_chip)
        self._engine_chip.setParent(None)
        self._engine_chip.deleteLater()
        self._engine_chip = Chip(engine_label, t, kind)
        self._chip_row.insertWidget(idx, self._engine_chip)

    # ── left: Images | Layers ────────────────────────────────────────────────
    def _left_panel(self) -> QWidget:
        t = self._t
        panel = QFrame()
        panel.setFixedWidth(240)
        panel.setStyleSheet(f"background:{t['inset']}; border-right:1px solid {t['border']};")
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        tabs = SegControl(["Images", "Layers"], t, 1, compact=True)
        tabwrap = QWidget()
        tw = QHBoxLayout(tabwrap)
        tw.setContentsMargins(10, 10, 10, 8)
        tw.addWidget(tabs)
        v.addWidget(tabwrap)

        stack = QStackedWidget()
        stack.addWidget(self._images_pane())
        stack.addWidget(self._layers_pane())
        stack.setCurrentIndex(1)
        tabs.changed.connect(stack.setCurrentIndex)
        v.addWidget(stack, 1)
        return panel

    def _images_pane(self) -> QWidget:
        t = self._t
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 2, 8, 8)
        v.setSpacing(4)
        search = QLineEdit()
        search.setPlaceholderText("Filter images…")
        search.addAction(icons.icon("diagnose", t["text_muted"], 14), QLineEdit.ActionPosition.LeadingPosition)
        v.addWidget(search)
        lst = QWidget()
        ll = QVBoxLayout(lst)
        ll.setContentsMargins(0, 4, 0, 0)
        ll.setSpacing(2)
        for i, (fn, st) in enumerate(demo.TASKS):
            ll.addWidget(self._task_row(i, fn, st))
        ll.addStretch(1)
        v.addWidget(_scroll(lst), 1)
        return w

    def _task_row(self, i: int, fn: str, st: str) -> QFrame:
        t = self._t
        sel = i == 1
        row = QFrame()
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setStyleSheet(
            (f"QFrame{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:8px;}}"
             if sel else
             f"QFrame{{background:transparent; border:1px solid transparent; border-radius:8px;}}"
             f"QFrame:hover{{background:{t['surface2']};}}"))
        lay = QHBoxLayout(row)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(10)
        thumb = QLabel()
        thumb.setFixedSize(38, 30)
        thumb.setPixmap(nuclei_pixmap(38, 30, 10 + i * 13, density=2.0, outline=st != "none"))
        thumb.setScaledContents(True)
        lay.addWidget(thumb)
        col = QVBoxLayout()
        col.setSpacing(1)
        fnl = QLabel(fn)
        fnl.setStyleSheet(f"color:{t['text']}; font-family:{theme.MONO}; font-size:12px; font-weight:600;")
        col.addWidget(fnl)
        col.addWidget(label(demo.STATUS_LABEL[st], 10.5, t["text_muted"]))
        lay.addLayout(col, 1)
        dot = QLabel()
        dcol = {"ok": t["success"], "pred": t["signal"], "none": t["border_strong"]}[st]
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{dcol}; border-radius:4px;")
        lay.addWidget(dot)
        return row

    def _layers_pane(self) -> QWidget:
        t = self._t
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # toolbar
        tb = QWidget()
        tbl = QHBoxLayout(tb)
        tbl.setContentsMargins(10, 8, 10, 6)
        tbl.setSpacing(3)
        for icon_name, tip in [("points", "New points layer"), ("shapes", "New shapes layer"),
                               ("new_labels", "New labels layer")]:
            tbl.addWidget(IconButton(icon_name, t, 30, tip))
        tbl.addStretch(1)
        tbl.addWidget(IconButton("shuffle", t, 30, "Shuffle label colours"))
        del_btn = IconButton("trash", t, 30, "Delete selected layer")
        del_btn.setStyleSheet(del_btn.styleSheet() +
                              f"QToolButton:hover{{background:{t['danger_weak']};}}")
        tbl.addWidget(del_btn)
        v.addWidget(tb)

        # list
        lst = QWidget()
        ll = QVBoxLayout(lst)
        ll.setContentsMargins(8, 0, 8, 8)
        ll.setSpacing(2)
        for i, (name, typ, count, vis) in enumerate(demo.LAYERS):
            ll.addWidget(self._layer_row(i, name, typ, count, vis))
        v.addWidget(lst)

        v.addWidget(self._layer_controls(), 1)
        return w

    def _layer_row(self, i: int, name: str, typ: str, count: str, vis: bool) -> QFrame:
        t = self._t
        sel = i == 0
        row = QFrame()
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setStyleSheet(
            (f"QFrame{{background:{t['surface']}; border:1px solid {t['border']}; border-radius:8px;}}"
             if sel else
             f"QFrame{{background:transparent; border:1px solid transparent; border-radius:8px;}}"
             f"QFrame:hover{{background:{t['surface2']};}}"))
        lay = QHBoxLayout(row)
        lay.setContentsMargins(8, 7, 8, 7)
        lay.setSpacing(9)
        eye = IconButton("eye" if vis else "eye_off", t, 22, "Toggle visibility")
        eye._on = vis
        def _flip(btn=eye):
            btn._on = not getattr(btn, "_on", True)
            btn.setIcon(icons.icon("eye" if btn._on else "eye_off",
                                   self._t["signal"] if btn._on else self._t["text_muted"], 14))
        if vis:
            eye.setIcon(icons.icon("eye", t["signal"], 14))
        eye.clicked.connect(_flip)
        lay.addWidget(eye)
        kind = demo.LAYER_TYPE_KIND.get(typ, "muted")
        colm = {"signal": t["signal"], "primary": t["primary"], "warning": t["warning"], "muted": t["text_subtle"]}
        weakm = {"signal": t["signal_weak"], "primary": t["primary_weak"], "warning": t["warning_weak"], "muted": t["surface2"]}
        ty = QLabel()
        ty.setFixedSize(24, 24)
        ty.setPixmap(icons.pixmap(LAYER_TYPE_ICON[typ], colm[kind], 14))
        ty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ty.setStyleSheet(f"background:{weakm[kind]}; border-radius:6px;")
        lay.addWidget(ty)
        nm = QLabel(name)
        nm.setStyleSheet(f"color:{t['text']}; font-size:12.5px; font-weight:600;")
        lay.addWidget(nm, 1)
        lay.addWidget(label(count, 10.5, t["text_muted"]))
        return row

    def _layer_controls(self) -> QWidget:
        t = self._t
        w = QFrame()
        w.setStyleSheet(f"background:{t['inset']}; border-top:1px solid {t['border']};")
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(11)

        title = QLabel("● Segmentation · labels")
        title.setStyleSheet(f"color:{t['text_muted']}; font-size:10.5px; font-weight:600; letter-spacing:0.6px;")
        v.addWidget(title)

        # mode tools (8)
        tools = QGridLayout()
        tools.setSpacing(3)
        mode_icons = [("target", "Pan / zoom", True), ("measure", "Transform", False),
                      ("brush", "Paint brush", False), ("eraser", "Eraser (1 or E)", False),
                      ("fill", "Fill bucket", False), ("polygon", "Polygon", False),
                      ("pick", "Pick label colour", False), ("shuffle", "Shuffle colours", False)]
        for i, (icon_name, tip, on) in enumerate(mode_icons):
            b = QToolButton()
            b.setToolTip(tip)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFixedHeight(30)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.setIcon(icons.icon(icon_name, "#fff" if on else t["text_subtle"], 15))
            b.setIconSize(QSize(15, 15))
            if on:
                b.setStyleSheet(f"QToolButton{{background:{t['primary']}; border:none; border-radius:7px;}}")
            else:
                b.setStyleSheet(
                    f"QToolButton{{background:{t['surface2']}; border:1px solid transparent; border-radius:7px;}}"
                    f"QToolButton:hover{{background:{t['surface']}; border-color:{t['border']};}}")
            tools.addWidget(b, 0, i)
        v.addLayout(tools)

        v.addWidget(FieldRow("opacity", Badge("0.70", t), t))
        v.addWidget(Slider(t, 0.70, t["signal"]))
        v.addWidget(FieldRow("blending", SelectBox("translucent", t), t))

        labelrow = QHBoxLayout()
        labelrow.setSpacing(8)
        chip = QFrame()
        chip.setFixedSize(22, 22)
        chip.setStyleSheet(f"background:#b23b1e; border:1px solid {t['border_strong']}; border-radius:5px;")
        labelrow.addWidget(chip)
        labelrow.addWidget(Stepper("1", t))
        v.addWidget(FieldRow("label", self._wrap(labelrow), t))

        v.addWidget(GroupLabel("label colours · more choices", t))
        pal = QGridLayout()
        pal.setSpacing(4)
        for i, col in enumerate(demo.LABEL_COLORS):
            sw = QFrame()
            sw.setFixedHeight(16)
            sw.setCursor(Qt.CursorShape.PointingHandCursor)
            border = f"2px solid {t['text']}" if i == 0 else f"1px solid rgba(128,128,128,0.3)"
            sw.setStyleSheet(f"background:{col}; border:{border}; border-radius:4px;")
            pal.addWidget(sw, i // 9, i % 9)
        v.addLayout(pal)

        v.addWidget(FieldRow("brush size", Badge("10", t), t))
        v.addWidget(Slider(t, 0.32))
        v.addWidget(FieldRow("colour mode", SelectBox("auto", t), t))
        v.addWidget(FieldRow("contour", Stepper("1", t), t))
        v.addWidget(FieldRow("n edit dim", Stepper("2", t), t))
        for name, on in [("contiguous", True), ("preserve labels", False), ("show selected", False)]:
            v.addWidget(self._check(name, on))
        v.addStretch(1)
        return _scroll(w)

    def _check(self, name: str, on: bool) -> QWidget:
        t = self._t
        row = QFrame()
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(9)
        box = QLabel()
        box.setFixedSize(16, 16)
        if on:
            box.setPixmap(icons.pixmap("check", "#04211f", 11))
            box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.setStyleSheet(f"background:{t['signal']}; border-radius:4px;")
        else:
            box.setStyleSheet(f"background:{t['inset']}; border:1px solid {t['border_strong']}; border-radius:4px;")
        lay.addWidget(box)
        lay.addWidget(label(name, 12, t["text_subtle"]))
        lay.addStretch(1)
        return row

    def _wrap(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        return w

    # ── centre: viewport ─────────────────────────────────────────────────────
    def _viewport(self) -> QWidget:
        t = self._t
        vp = QFrame()
        vp.setStyleSheet("background:#07090c;")
        canvas = NucleiView(seed=7, density=0.85, big=True)
        lay = QVBoxLayout(vp)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(canvas)

        # legend (top-left)
        legend = QWidget(vp)
        leg = QHBoxLayout(legend)
        leg.setContentsMargins(0, 0, 0, 0)
        leg.setSpacing(7)
        for text, col, rad in [("247 detected", t["signal"], 4), ("3 selected", "#8b9bf4", 2)]:
            c = QLabel(f"● {text}")
            c.setStyleSheet(
                f"color:#eaf7f5; background:rgba(8,12,16,0.6); border:1px solid rgba(255,255,255,0.1);"
                f"border-radius:999px; padding:3px 9px; font-size:11px; font-weight:600;")
            leg.addWidget(c)
        legend.move(14, 14)
        legend.adjustSize()

        # tools column (right)
        tools = QFrame(vp)
        tools.setStyleSheet(
            f"background:rgba(21,24,30,0.86); border:1px solid {t['border']}; border-radius:11px;")
        tl = QVBoxLayout(tools)
        tl.setContentsMargins(5, 5, 5, 5)
        tl.setSpacing(4)
        for icon_name, on in [("target", True), ("brush", False), ("points", False), ("target", False)]:
            b = IconButton(icon_name, t, 30)
            if on:
                b.setStyleSheet(f"QToolButton{{background:{t['signal_weak']}; border-radius:7px;}}")
                b.setIcon(icons.icon(icon_name, t["signal"], 16))
            tl.addWidget(b)
        self._vp_tools = tools

        # viewer bar (bottom-left): console, 2D/3D, roll, transpose, grid, home
        vbar = QFrame(vp)
        vbar.setStyleSheet(
            f"background:rgba(21,24,30,0.86); border:1px solid {t['border']}; border-radius:11px;")
        vl = QHBoxLayout(vbar)
        vl.setContentsMargins(5, 5, 5, 5)
        vl.setSpacing(3)
        for icon_name, on, tip in [("console", False, "Toggle console"), ("cube3d", True, "Toggle 2D / 3D"),
                                   ("refresh", False, "Roll dimensions"), ("shuffle", False, "Transpose"),
                                   ("grid", False, "Grid mode"), ("home", False, "Reset view")]:
            b = IconButton(icon_name, t, 30, tip)
            if on:
                b.setStyleSheet(f"QToolButton{{background:{t['signal_weak']}; border-radius:7px;}}")
                b.setIcon(icons.icon(icon_name, t["signal"], 16))
            vl.addWidget(b)
        self._vp_bar = vbar

        # status (bottom-right)
        status = QLabel("● Segmented in 3.2 s", vp)
        status.setStyleSheet(
            f"color:#dbe6ee; background:rgba(8,12,16,0.6); border:1px solid rgba(255,255,255,0.1);"
            f"border-radius:999px; padding:5px 11px; font-size:11.5px; font-weight:600;")
        self._vp_status = status

        vp._overlays = (legend, tools, vbar, status)
        vp.resizeEvent = lambda e: self._place_overlays(vp)
        return vp

    def _place_overlays(self, vp):
        legend, tools, vbar, status = vp._overlays
        w, h = vp.width(), vp.height()
        tools.adjustSize()
        vbar.adjustSize()
        status.adjustSize()
        tools.move(w - tools.width() - 14, 14)
        vbar.move(14, h - vbar.height() - 14)
        status.move(w - status.width() - 14, h - status.height() - 14)

    # ── right: inspector ─────────────────────────────────────────────────────
    def _inspector(self) -> QWidget:
        t = self._t
        panel = QFrame()
        panel.setFixedWidth(320)
        panel.setStyleSheet(f"background:{t['surface']}; border-left:1px solid {t['border']};")
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        tabs = SegControl(["Segment", "Results"], t, 0, compact=True)
        tw = QWidget()
        twl = QHBoxLayout(tw)
        twl.setContentsMargins(12, 10, 12, 4)
        twl.addWidget(tabs)
        v.addWidget(tw)

        stack = QStackedWidget()
        stack.addWidget(_scroll(self._segment_pane()))
        stack.addWidget(_scroll(self._results_pane()))
        tabs.changed.connect(stack.setCurrentIndex)
        v.addWidget(stack, 1)

        v.addWidget(self._runbar())
        return panel

    def _segment_pane(self) -> QWidget:
        t = self._t
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(14, 12, 14, 14)
        v.setSpacing(14)

        v.addWidget(GroupLabel("Engine", t))
        v.addWidget(SelectBox("CellSeg1 · SAM + LoRA", t, "models", t["primary"]))
        v.addWidget(FieldRow("Model", SelectBox("nuclei-dapi-r8", t), t))

        v.addWidget(GroupLabel("Quality preset", t))
        v.addWidget(SegControl(["Fast", "Balanced", "Accurate"], t, 1, compact=True))

        v.addWidget(hline(t))
        v.addWidget(GroupLabel("Detection thresholds", t))
        v.addWidget(FieldRow("Points / side", Stepper("32", t), t))
        v.addWidget(FieldRow("IoU threshold", Badge("0.80", t), t))
        v.addWidget(Slider(t, 0.80))
        v.addWidget(FieldRow("Stability score", Badge("0.60", t), t))
        v.addWidget(Slider(t, 0.60))
        v.addWidget(FieldRow("Min mask area", Stepper("20", t), t))

        v.addWidget(hline(t))
        v.addWidget(GroupLabel("Image", t))
        v.addWidget(FieldRow("Resize", SelectBox("512 px", t), t))
        v.addWidget(FieldRow("Pixel size", Badge("0.65 µm/px", t), t))
        v.addWidget(FieldRow("Channels", Badge("DAPI · Memb", t), t))
        v.addWidget(FieldRow("CLAHE contrast", Toggle(t, False), t))
        v.addWidget(FieldRow("Large image (tiling)", Toggle(t, False), t))

        v.addWidget(hline(t))
        v.addWidget(GroupLabel("Overlays", t))
        v.addWidget(FieldRow("Show predictions", Toggle(t, True), t))
        v.addWidget(FieldRow("Show ground truth", Toggle(t, False), t))

        eng = Accordion("Engine settings · CellSeg1", t, lead="settings", open_=False)
        eng.add(FieldRow("SAM backbone", SelectBox("ViT-H", t), t))
        eng.add(FieldRow("LoRA rank", Stepper("8", t), t))
        eng.add(FieldRow("Box NMS", Badge("0.05", t), t))
        v.addWidget(eng)
        v.addStretch(1)
        return w

    def _results_pane(self) -> QWidget:
        t = self._t
        r = demo.RESULTS
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(14, 12, 14, 14)
        v.setSpacing(14)

        hero = QHBoxLayout()
        hero.setSpacing(14)
        num = QLabel(str(r["cells"]))
        num.setStyleSheet(f"color:{t['success']}; font-family:{theme.MONO}; font-size:40px; font-weight:600; letter-spacing:-1.5px;")
        hero.addWidget(num)
        hero.addWidget(label("cells\ndetected", 13, t["text_muted"], 600))
        hero.addStretch(1)
        v.addLayout(hero)

        tiles = QHBoxLayout()
        tiles.setSpacing(9)
        tiles.addWidget(StatTile(r["median_d"], "px", "MEDIAN Ø", t))
        tiles.addWidget(StatTile(r["mean_area"], "px²", "MEAN AREA", t))
        tiles.addWidget(StatTile(r["coverage"], "%", "COVERAGE", t))
        v.addLayout(tiles)

        v.addWidget(GroupLabel("Pixel calibration", t))
        cal = SelectBox("0.65 µm/px — real-world units", t)
        v.addWidget(cal)
        hint = label("Enter your microscope's µm-per-pixel to get real-world units. 0 = pixels.", 11, t["text_muted"])
        hint.setWordWrap(True)
        v.addWidget(hint)

        btns = QGridLayout()
        btns.setSpacing(8)
        for i, (text, icon_name) in enumerate([("Save masks", "save"), ("Export CSV", "csv"),
                                               ("Refine…", "spark"), ("Measurements", "measure")]):
            btns.addWidget(PillButton(text, t, "ghost", icon_name, small=True), i // 2, i % 2)
        v.addLayout(btns)

        v.addWidget(hline(t))
        v.addWidget(GroupLabel("Display · colour cells by", t))
        v.addWidget(SelectBox("Instance ID (default)", t))
        heat = QFrame()
        hv = QVBoxLayout(heat)
        hv.setContentsMargins(0, 4, 0, 0)
        hv.setSpacing(3)
        grad = QFrame()
        grad.setFixedHeight(10)
        grad.setStyleSheet("background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                           "stop:0 #440154, stop:0.25 #3b528b, stop:0.5 #21918c, stop:0.75 #5ec962, stop:1 #fde725);"
                           "border-radius:5px;")
        hv.addWidget(grad)
        v.addWidget(heat)

        gt = Accordion("Ground truth & evaluation", t, lead="check", open_=False)
        gt.add(FieldRow("Ground-truth mask", Badge("phantom_gt.png", t), t))
        gt.add(FieldRow("Show ground truth", Toggle(t, False), t))
        for name, val in [("F1 score", r["f1"]), ("Precision", r["precision"]),
                          ("Recall", r["recall"]), ("AP @ 0.50", r["ap50"])]:
            mv = QLabel(val)
            mv.setStyleSheet(f"color:{t['success']}; font-family:{theme.MONO}; font-size:12.5px; font-weight:600;")
            gt.add(FieldRow(name, mv, t))
        v.addWidget(gt)

        batch = Accordion("Batch prediction", t, lead="batch", open_=False)
        note = label("Run the current engine & settings across all 128 images in this project, "
                     "then aggregate cohort statistics.", 11.5, t["text_muted"])
        note.setWordWrap(True)
        batch.add(note)
        batch.add(PillButton("Run batch (128 images)", t, "ghost", "run", small=True))
        v.addWidget(batch)

        bench = Accordion("Benchmark engines vs GT", t, lead="chart", open_=False)
        for name, val, ok in [("CellSeg1 · LoRA", "0.94", True), ("Cellpose-SAM", "0.86", False),
                              ("SAM 2", "0.90", False)]:
            mv = QLabel(val)
            col = t["success"] if ok else t["text_subtle"]
            mv.setStyleSheet(f"color:{col}; font-family:{theme.MONO}; font-size:12.5px; font-weight:600;")
            bench.add(FieldRow(name, mv, t))
        v.addWidget(bench)
        v.addStretch(1)
        return w

    def _runbar(self) -> QWidget:
        t = self._t
        bar = QFrame()
        bar.setStyleSheet(f"background:{t['surface']}; border-top:1px solid {t['border']};")
        v = QVBoxLayout(bar)
        v.setContentsMargins(14, 12, 14, 14)
        v.setSpacing(10)
        prog = QFrame()
        prog.setFixedHeight(6)
        prog.setStyleSheet(
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {t['primary']}, stop:1 {t['signal']}); border-radius:3px;")
        v.addWidget(prog)
        row = QHBoxLayout()
        row.setSpacing(9)
        run = PillButton("Run segmentation", t, "primary", "run")
        run.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(run, 1)
        row.addWidget(IconButton("batch", t, 38))
        v.addLayout(row)
        return bar
