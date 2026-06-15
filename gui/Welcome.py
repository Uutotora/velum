import torch
import streamlit as st
import pandas as pd

from gui.pages.utils.web_utils import (
    LORA_PTH_DIR, TEST_IMAGE_DIR, PRED_MASK_DIR,
    inject_css, initialize_session_state, list_files, render_sidebar,
)
from project_root import STORAGE_DIR

st.set_page_config(page_title="CellSeg1", page_icon="🔬", layout="wide")
inject_css()
initialize_session_state()
render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## CellSeg1")
st.caption("Cell segmentation from a single annotated image · SAM + Low-Rank Adaptation")

# ── Status row ────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    _dev, _dev_ok = "CUDA GPU", True
elif torch.backends.mps.is_available():
    _dev, _dev_ok = "Apple Silicon MPS", True
else:
    _dev, _dev_ok = "CPU", False

sam_ok = (STORAGE_DIR / "sam_backbone" / "sam_vit_h_4b8939.pth").exists()
loras  = list_files(LORA_PTH_DIR, extensions=[".pth"])

c1, c2, c3 = st.columns(3)
c1.metric("Compute device",   ("✓ " if _dev_ok else "⚠ ") + _dev)
c2.metric("SAM backbone",     "Loaded" if sam_ok else "Missing — see setup")
c3.metric("LoRA checkpoints", str(len(loras)))

st.divider()

# ── Pre-trained checkpoints ───────────────────────────────────────────────────
st.markdown("### Pre-trained checkpoints")
st.caption("Select the checkpoint that best matches your cell type on the **Predict** page.")

CHECKPOINT_DATA = [
    {"Checkpoint": "cellseg_blood_117.pth",           "Cell type": "Blood cells · round, bright field",     "Dataset": "CSB",     "mAP@0.5": 0.941},
    {"Checkpoint": "cellpose_specialized_12.pth",     "Cell type": "General cells · mixed morphology",      "Dataset": "Cellpose","mAP@0.5": 0.917},
    {"Checkpoint": "dsb2018_stardist_435.pth",        "Cell type": "Cell nuclei · dark background",         "Dataset": "DSB2018", "mAP@0.5": 0.872},
    {"Checkpoint": "deepbacs_rod_brightfield_9.pth",  "Cell type": "Rod bacteria · E. coli, bright field",  "Dataset": "ECB",     "mAP@0.5": 0.860},
    {"Checkpoint": "deepbacs_rod_fluorescence_75.pth","Cell type": "Rod bacteria · B. subtilis, fluoresce.","Dataset": "BSF",     "mAP@0.5": 0.821},
]
df = pd.DataFrame(CHECKPOINT_DATA)
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "mAP@0.5": st.column_config.ProgressColumn(
            "mAP@0.5", min_value=0, max_value=1, format="%.3f"
        )
    },
)

st.divider()

# ── Getting started ───────────────────────────────────────────────────────────
st.markdown("### Getting started")
st.caption("Three steps from raw image to segmented cells.")

STEPS = [
    {
        "num": "01",
        "title": "Train",
        "subtitle": "Fine-tune SAM on your annotated image",
        "detail": (
            "Upload one microscopy image and its instance mask (each cell = unique integer ID). "
            "Pick a preset — **Fast · MPS** for quick iteration, **Balanced** for standard quality — "
            "then click **Start Training**. Takes 10–30 min on Apple Silicon."
        ),
        "page": "pages/1_Train.py",
        "link_label": "Go to Train →",
    },
    {
        "num": "02",
        "title": "Predict",
        "subtitle": "Run inference on new images",
        "detail": (
            "Select a LoRA checkpoint (pre-trained or your own from Train). "
            "Upload the images to segment and click **▶ Run**. "
            "Inference takes ~60 s per image on Apple Silicon MPS, ~5 s on NVIDIA 4090. "
            "Download results as a ZIP archive or inspect inline."
        ),
        "page": "pages/2_Predict.py",
        "link_label": "Go to Predict →",
    },
    {
        "num": "03",
        "title": "Visualize",
        "subtitle": "Inspect predictions · compare against ground truth",
        "detail": (
            "View contours overlaid on the original image, colored instance masks, "
            "and (if you have a ground-truth mask) a pixel-level error map. "
            "Each output mask is a uint16 PNG where pixel value = cell instance ID. "
            "Background = 0."
        ),
        "page": "pages/3_Visualize.py",
        "link_label": "Go to Visualize →",
    },
]

for step in STEPS:
    col_num, col_text, col_btn = st.columns([1, 9, 2], vertical_alignment="top")
    with col_num:
        st.markdown(
            f'<div style="background:#ef4444;color:white;border-radius:6px;'
            f'padding:5px 10px;font-weight:700;font-size:13px;text-align:center;'
            f'margin-top:2px;">{step["num"]}</div>',
            unsafe_allow_html=True,
        )
    with col_text:
        st.markdown(f"**{step['title']}** — {step['subtitle']}")
        st.caption(step["detail"])
    with col_btn:
        st.page_link(step["page"], label=step["link_label"], use_container_width=True)
    st.markdown("")

st.divider()

# ── Synthetic test image ───────────────────────────────────────────────────────
st.markdown("### Synthetic test image")
st.caption(
    "No microscopy images yet? Generate a synthetic image instantly to test the Predict pipeline."
)

with st.container(border=True):
    ctrl_col, preview_col = st.columns([3, 4], vertical_alignment="top")

    with ctrl_col:
        n_cells = st.slider("Number of cells", 20, 200, 80, step=10, key="synth_n_cells")
        rnd_seed = st.number_input("Random seed", 0, 9999, 42, step=1, key="synth_seed")

        if st.button("Generate synthetic cells", type="primary", use_container_width=True):
            import io
            import numpy as np
            from PIL import Image as PILImage

            _rng  = np.random.default_rng(int(rnd_seed))
            _size = 512
            _img  = np.full((_size, _size), 22, dtype=np.int32)
            _yy, _xx = np.mgrid[0:_size, 0:_size]
            cx_all = _rng.integers(30, _size - 30, n_cells)
            cy_all = _rng.integers(30, _size - 30, n_cells)
            rx_all = _rng.integers(10, 28, n_cells)
            ry_all = _rng.integers(10, 28, n_cells)
            br_all = _rng.integers(130, 230, n_cells)
            for cx, cy, rx, ry, br in zip(cx_all, cy_all, rx_all, ry_all, br_all):
                d2 = ((_xx - cx) / rx) ** 2 + ((_yy - cy) / ry) ** 2
                mask = d2 <= 1.0
                intensity = np.where(mask, (br * (1.0 - np.sqrt(np.clip(d2, 0, 1)) * 0.4)).astype(np.int32), 0)
                _img = np.where(mask, np.maximum(_img, intensity), _img)
            _img = np.clip(_img + _rng.integers(0, 12, (_size, _size)), 0, 255).astype(np.uint8)
            TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            PILImage.fromarray(_img).save(TEST_IMAGE_DIR / "synthetic_cells.png")
            st.session_state["_synth_generated"] = True
            st.rerun()

        st.caption(
            "Saves to `streamlit_storage/test_images/synthetic_cells.png` — "
            "pick it up on the **Predict** page."
        )

    with preview_col:
        _syn_path = TEST_IMAGE_DIR / "synthetic_cells.png"
        if _syn_path.exists():
            from PIL import Image as _PILImg
            _thumb = _PILImg.open(_syn_path)
            _thumb.thumbnail((280, 280))
            st.image(_thumb, caption="synthetic_cells.png · 512×512 · greyscale")
        else:
            st.caption("Preview will appear here after generation.")

st.divider()

# ── Expandable reference sections ─────────────────────────────────────────────
with st.expander("Training on your own data"):
    st.markdown("""
**Requirements**

| Item | Description |
|---|---|
| Training image | Any microscopy image — TIFF, PNG, JPG |
| Instance mask | Same resolution; each cell = unique integer ID; background = 0 |
| Annotation count | ≥ 30–50 cells recommended for stable convergence |

**Mask format:** uint16 or uint32 PNG/TIFF where pixel value = cell label.
Tools that export this format: **QuPath** (export instance labels), **Cellpose** (save masks), **ImageJ** (Analyze Particles → label map).

**Recommended settings for Apple Silicon (18 GB RAM, MPS)**

| Parameter | Value |
|---|---|
| Preset | Balanced (or Fast · MPS for quick test) |
| batch_size | 1 |
| gradient_accumulation | 32 |
| epoch_max | 300 |
| lora_rank | 4 |
| SAM type | vit_h |

Effective batch size = `1 × 32 = 32`. Training ~20–30 min on MPS.
""")

with st.expander("Parameter reference"):
    st.markdown("""
| Parameter | Page | Description |
|---|---|---|
| SAM type | Train / Predict | Model size: vit_h (best) > vit_l > vit_b. Must match between train and predict. |
| LoRA rank | Train / Predict | Adapter dimension. Higher = more capacity but slower. Default 4. |
| SAM image size | Train / Predict | Internal resolution. Default 512. Must match between train and predict. |
| Resize size | Train | Resize image before patch extraction. 0 = no resize. |
| Patch size | Train | Sliding window size (50% overlap). Should be ≤ resize size. |
| Gradient accumulation | Train | Effective batch size = batch_size × gradient_accumulation. |
| Points per side | Predict | Grid density for mask proposals. Higher = more candidates, slower. |
| IoU threshold | Predict | Min predicted mask IoU to keep. Higher = fewer false positives. |
| Stability score | Predict | Mask stability threshold. Higher = keeps only high-confidence masks. |
| NMS threshold | Predict | Non-max suppression overlap. Lower = merges more overlapping masks. |
""")

with st.expander("Troubleshooting"):
    st.markdown(f"""
**Crash / stuck prediction** — delete state files and restart:

```bash
rm -f {STORAGE_DIR}/prediction_state.json \\
      {STORAGE_DIR}/prediction_progress.json \\
      {STORAGE_DIR}/prediction_stop_flag

pkill -f "streamlit run"
cd ~/cellseg1
PYTHONPATH=/Users/u2ora/cellseg1 \\
  /opt/homebrew/Caskroom/miniconda/base/envs/cellseg1/bin/streamlit run \\
  gui/Welcome.py --server.headless true
```

**MPS float64 error** — already patched in this installation.

**"No LoRA models"** — checkpoints live in `streamlit_storage/loras/`. Verify they are present.

**Very slow inference** — expected on CPU (~5 min/image). Use Apple Silicon MPS device on the Predict page.

**SAM backbone missing** — download `sam_vit_h_4b8939.pth` from the SAM GitHub releases and place it in `streamlit_storage/sam_backbone/`.
""")
