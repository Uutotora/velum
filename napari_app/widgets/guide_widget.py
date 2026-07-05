"""
Guide tab — in-app documentation.

A single scrollable, theme-styled reference so users never have to leave the
app to learn a workflow. Content is intentionally task-oriented (what to click,
in what order) rather than exhaustive API docs.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextBrowser
from PyQt6.QtCore import Qt

from napari_app.theme import (
    BG, FG, BORDER, BORDER_STRONG, CARD_HEADER, TEXT, DIM, LABEL, ACCENT,
    SUCCESS, CONSOLE, MONO, SANS, WIDGET_SS,
)


_HTML = f"""
<style>
  body, p, li, ol, ul, h1, h2, h3 {{ font-family:{SANS}; }}
  h1 {{ color:{TEXT}; font-size:22px; font-weight:800; letter-spacing:-0.5px; margin:2px 0 2px 0; }}
  h2 {{ color:{ACCENT}; font-size:13px; font-weight:700; letter-spacing:1px; text-transform:uppercase;
        margin:22px 0 8px 0; border-bottom:1px solid {BORDER}; padding-bottom:5px; }}
  h3 {{ color:{TEXT}; font-size:12.5px; margin:12px 0 3px 0; }}
  p, li {{ color:{LABEL}; font-size:12.5px; line-height:1.6; }}
  b {{ color:{TEXT}; }}
  code {{ background:{CONSOLE}; color:{SUCCESS}; padding:1px 6px;
          border-radius:4px; font-family:{MONO}; font-size:11px; }}
  .kbd {{ background:{CARD_HEADER}; color:{TEXT}; border:1px solid {BORDER_STRONG};
          border-bottom-width:2px; border-radius:5px; padding:1px 7px; font-family:{MONO}; font-size:11px; }}
  .lead {{ color:{DIM}; font-size:12.5px; line-height:1.6; }}
  .tip {{ color:{SUCCESS}; }}
</style>

<h1>CellSeg1 — user guide</h1>
<p class="lead">Cell instance segmentation with a Segment Anything (SAM) backbone
fine-tuned to your data via LoRA. Everything below runs locally on your machine.</p>

<h2>Quick start (5 steps)</h2>
<ol>
  <li>Open the <b>Predict</b> tab and pick a <b>Checkpoint</b> (a bundled model,
      or one you trained).</li>
  <li>Choose an <b>Image</b> — browse, or drag &amp; drop it onto the panel.</li>
  <li>Press <span class="kbd">▶ Run Prediction</span> (<span class="kbd">Ctrl+R</span>).
      Cells appear as a Labels layer in the viewer.</li>
  <li>Open <b>Open measurements table</b> to review per-cell morphometry and export CSV.</li>
  <li>Not happy with the result? Open the <b>Assistant</b> tab and press
      <b>Diagnose</b> for one-click fixes.</li>
</ol>

<h2>Predict</h2>
<p>Runs automatic segmentation over the whole image. Key controls:</p>
<ul>
  <li><b>Checkpoint</b> — the LoRA adapter. Bundled ones cover common cell types;
      the label shows the reported mAP. Trained checkpoints appear as
      <code>[trained]</code>.</li>
  <li><b>Resize</b> — inference resolution. <b>1024</b> recovers small cells but is
      slower; <b>512</b> is a good default.</li>
  <li><b>Inference parameters</b> (collapsed) — thresholds that trade recall for
      precision. Prefer letting the <b>Assistant</b> tune these.</li>
  <li><b>Pixel size (µm/px)</b> — set it in the Results card to report areas in
      µm² and lengths in µm everywhere.</li>
  <li><b>Batch prediction</b> — process a whole folder to masks; stoppable. On
      completion it also writes cohort CSVs (per-cell + per-image) and opens a
      <b>Cohort analysis</b> window with population statistics and a pooled
      distribution histogram.</li>
  <li><b>Benchmark engines vs GT</b> — run the selected engines over a folder
      that has ground-truth masks and get a comparison table of F1 and Average
      Precision (mAP), so you can objectively pick the best method for your data.</li>
  <li><b>Evaluate vs GT</b> — score a single prediction against ground truth
      (F1 and AP at IoU 0.5/0.75/0.9). Ground truth is auto-detected from a
      <code>*_gt</code> / <code>masks/</code> sidecar and rendered as a green outline.</li>
  <li><b>Provenance</b> — saving a mask also writes a <code>.json</code> manifest
      (engine, parameters, versions, image hash) for reproducible analysis.</li>
  <li><b>Refine…</b> — after correcting the Labels layer by hand, fine-tunes the
      current checkpoint on your correction (50 epochs).</li>
</ul>
<p class="tip">Tip: <b>Predict on active napari layer</b>
(<span class="kbd">Ctrl+Shift+R</span>) segments whatever Image layer is
selected — handy for images already open in the viewer.</p>

<h2>Annotate — click to segment</h2>
<p>Interactive, human-in-the-loop segmentation using SAM point prompts. Start a
session (the image embedding is computed once), then:</p>
<ul>
  <li><b>Left-click</b> a cell → segments it as a new label.</li>
  <li><b>Shift + click</b> → adds a positive point to grow the last cell.</li>
  <li><b>Ctrl/⌘ + click</b> → adds a negative point to carve the last cell.</li>
</ul>
<p>The result is a normal, editable Labels layer. Use it to build ground truth,
then feed it straight into <b>Train</b> or <b>Refine</b>.</p>

<h2>Assistant — the local agent</h2>
<p>An offline expert that inspects your image and mask and recommends concrete
parameter changes you can apply with one click.</p>
<ul>
  <li><b>Diagnose current result</b> — reads coverage, cell-size distribution and
      merge/fragment signatures, then proposes fixes (each with
      <b>Apply</b> / <b>Apply &amp; re-run</b>).</li>
  <li><b>Local model</b> — optional natural-language chat. Install
      <a href="https://ollama.com">Ollama</a>, then either download a recommended
      model from the panel or use one you already have.</li>
  <li><b>Tune for CellSeg1</b> — bakes a task-specialised agent
      (<code>cellseg1-assistant</code>) from any base model: it pins the domain
      persona and a low temperature so advice is precise and repeatable.</li>
  <li>Chat answers may include <code>SUGGEST: param=value</code> lines — the app
      turns them into Apply buttons, so the model can drive the pipeline.</li>
</ul>
<p class="tip">Privacy: chat and models run entirely on localhost. Nothing is
sent to the cloud.</p>

<h2>Train</h2>
<p>Fine-tune a LoRA adapter on your own labelled images.</p>
<ul>
  <li><b>Presets</b> — Fast / Balanced / Best quality set sensible epochs, rank
      and resolution.</li>
  <li><b>Training data</b> — point to an images folder and a matching masks folder
      (label images where each cell is a distinct integer). Or click
      <b>Use active napari layers</b> to export what's open in the viewer.</li>
  <li><b>Effective batch size</b> = batch × gradient accumulation; the panel shows it.</li>
  <li>Watch the live loss curve; history of past runs is listed below.</li>
</ul>
<p>Masks are validated before training starts (cell counts, tiny objects), so you
catch labelling mistakes early.</p>

<h2>Getting better segmentations</h2>
<ul>
  <li><b>Missing cells?</b> Lower <code>pred_iou_thresh</code> and
      <code>stability_score_thresh</code>, raise <code>points_per_side</code>,
      predict at 1024.</li>
  <li><b>Touching cells merged?</b> Lower <code>box_nms_thresh</code>.</li>
  <li><b>Debris / fragments?</b> Raise <code>min_mask_area</code>.</li>
  <li><b>Faint, low-contrast images?</b> Enhance contrast before predicting.</li>
  <li><b>Domain very different from the bundled models?</b> Label ~10–20 images
      and <b>Train</b> a checkpoint — this is where CellSeg1 shines.</li>
</ul>

<h2>Keyboard shortcuts</h2>
<ul>
  <li><span class="kbd">Ctrl+R</span> — run prediction</li>
  <li><span class="kbd">Ctrl+Shift+R</span> — predict on active layer</li>
  <li><span class="kbd">Ctrl+T</span> — start training &nbsp;·&nbsp;
      <span class="kbd">Esc</span> — stop training</li>
</ul>

<h2>Files &amp; formats</h2>
<ul>
  <li><b>Images</b>: PNG, TIFF, JPG, BMP, NPY.</li>
  <li><b>Masks</b>: label images (uint16 PNG/TIFF or NPY), background = 0.</li>
  <li><b>Measurements</b>: CSV with one row per cell and all morphometric features.</li>
</ul>
<p class="lead">CellSeg1 — one-shot cell segmentation. See the project repository
for the method and citation.</p>
"""


class GuideWidget(QWidget):
    def __init__(self, viewer=None):
        super().__init__()
        self.setStyleSheet(WIDGET_SS)
        L = QVBoxLayout(); L.setContentsMargins(0, 0, 0, 0)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet(
            f"QTextBrowser {{ background:{BG}; border:none; padding:16px 18px; }}")
        browser.setHtml(_HTML)
        L.addWidget(browser)
        self.setLayout(L)
        self.setMinimumWidth(260)
