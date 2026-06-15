import subprocess
from pathlib import Path

import streamlit as st
import yaml

from project_root import STORAGE_DIR

TRAIN_IMAGE_DIR = STORAGE_DIR / "train_images"
TRAIN_MASK_DIR = STORAGE_DIR / "train_masks"
TEST_IMAGE_DIR = STORAGE_DIR / "test_images"
TEST_MASK_DIR = STORAGE_DIR / "test_masks"
PRED_MASK_DIR = STORAGE_DIR / "predict_masks"
LORA_PTH_DIR = STORAGE_DIR / "loras"
SUPPORT_EXTENSION = [
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".nii",
    ".nii.gz",
    ".npy",
]


def delete_file(directory, filename):
    file_path = directory / filename
    try:
        if file_path.exists():
            file_path.unlink()
        else:
            st.error(f"File {filename} does not exist.")
    except Exception as e:
        st.error(f"Error deleting file {filename}: {e}")


def get_sam_model_path(sam_type):
    sam_models = {
        "vit_h": STORAGE_DIR / "sam_backbone" / "sam_vit_h_4b8939.pth",
        "vit_l": STORAGE_DIR / "sam_backbone" / "sam_vit_l_0b3195.pth",
        "vit_b": STORAGE_DIR / "sam_backbone" / "sam_vit_b_01ec64.pth",
    }
    return sam_models[sam_type]


def load_default_config():
    example_config = STORAGE_DIR.parent / "example_config.yaml"
    with open(example_config, "r") as f:
        config = yaml.safe_load(f)
    return config


DEVICE_LABELS = {
    "mps": "Apple Silicon GPU (MPS)",
    "cpu": "CPU",
}


def get_available_devices():
    import torch
    devices = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
        )
        if result.stdout.strip():
            devices.extend([info.split(",")[0].strip() for info in result.stdout.strip().split("\n")])
    except (FileNotFoundError, subprocess.CalledProcessError):
        if torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    devices.append("cpu")
    return devices


def inject_css():
    st.markdown("""
    <style>
    /* ── Metric cards — scientific accent ───── */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #2563eb;
        border-radius: 5px;
        padding: 8px 14px;
    }
    [data-testid="stMetric"] label {
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #64748b;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
    }

    /* ── Progress bar ────────────────────────── */
    .stProgress > div > div > div > div {
        border-radius: 2px;
        background: #2563eb;
    }

    /* ── Expanders ───────────────────────────── */
    details[data-testid="stExpander"] summary {
        padding: 5px 10px;
        font-size: 12.5px;
        font-weight: 600;
        color: #374151;
        letter-spacing: 0.02em;
    }

    /* ── Bordered containers ─────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e2e8f0 !important;
        border-radius: 5px !important;
    }

    /* ── Section label utility ───────────────── */
    .sec-label {
        font-size: 10.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 4px;
    }

    /* ── Sidebar ─────────────────────────────── */
    [data-testid="stSidebar"] { background: #f8fafc; }
    [data-testid="stSidebar"] hr { margin: 6px 0 !important; }

    /* ── File list row ───────────────────────── */
    .file-row {
        display: flex; align-items: center; gap: 8px;
        padding: 3px 6px; border-radius: 3px;
        background: #f1f5f9; margin-bottom: 2px;
        font-family: 'SF Mono','Menlo','Fira Code',monospace;
        font-size: 11.5px; color: #334155;
    }
    .file-row:hover { background: #e2e8f0; }

    /* ── Horizontal rules ────────────────────── */
    hr { margin: 8px 0 !important; border-color: #e2e8f0 !important; }

    /* ── Info/success/warning banners ────────── */
    [data-testid="stAlert"] { padding: 6px 12px; font-size: 13px; }

    /* ── Delete / icon buttons ───────────────── */
    button[data-testid="baseButton-secondary"] {
        padding: 1px 7px !important;
        font-size: 11px !important;
        min-height: 0 !important;
        line-height: 1.4 !important;
        color: #94a3b8 !important;
        border-color: #e2e8f0 !important;
        background: transparent !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        color: #ef4444 !important;
        border-color: #fca5a5 !important;
        background: #fff1f2 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def _fmt_filesize(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.0f} KB"
    return f"{n_bytes / 1024 ** 2:.1f} MB"


def show_compact_file_list(directory, filenames, delete_key_prefix, show_preview=False):
    """Compact one-row-per-file list with mini thumbnail, filename, size, and delete."""
    from PIL import Image as PILImage
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for fname in filenames:
        fpath = directory / fname
        is_img = Path(fname).suffix.lower() in image_exts

        # File metadata
        try:
            size_str = _fmt_filesize(fpath.stat().st_size)
        except OSError:
            size_str = ""

        img_dims = ""
        if is_img:
            try:
                _img = PILImage.open(fpath)
                w, h = _img.size
                img_dims = f" · {w}×{h}"
            except Exception:
                pass

        meta = f'<span style="color:#94a3b8;font-size:10.5px;margin-left:6px;">{size_str}{img_dims}</span>'

        if show_preview and is_img:
            c_thumb, c_name, c_prev, c_del = st.columns(
                [0.6, 9, 1.2, 0.7], gap="small", vertical_alignment="center"
            )
        else:
            c_thumb, c_name, c_del = st.columns(
                [0.6, 9, 0.7], gap="small", vertical_alignment="center"
            )

        with c_thumb:
            if is_img:
                try:
                    img = PILImage.open(fpath)
                    img.thumbnail((40, 40))
                    st.image(img, use_container_width=True)
                except Exception:
                    st.caption("—")
            else:
                st.caption("·")

        with c_name:
            st.markdown(
                f'<div class="file-row">{fname}{meta}</div>',
                unsafe_allow_html=True,
            )

        if show_preview and is_img:
            with c_prev:
                with st.popover("Preview", use_container_width=True):
                    try:
                        img = PILImage.open(fpath)
                        img.thumbnail((520, 520))
                        st.image(img, caption=fname, use_container_width=True)
                    except Exception as e:
                        st.caption(f"Cannot preview: {e}")

        with c_del:
            st.button("✕", key=f"{delete_key_prefix}_{fname}",
                      on_click=lambda f=fname: delete_file(directory, f),
                      help=f"Remove {fname}")


def show_image_thumbnails(directory, filenames, max_cols=5):
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    display = [f for f in filenames if Path(f).suffix.lower() in image_exts][:max_cols]
    if not display:
        return
    cols = st.columns(len(display))
    for col, fname in zip(cols, display):
        with col:
            try:
                from PIL import Image as PILImage
                img = PILImage.open(directory / fname)
                img.thumbnail((180, 180))
                label = fname if len(fname) <= 22 else fname[:20] + "…"
                st.image(img, caption=label, use_container_width=True)
            except Exception:
                st.caption(f"📄 {fname}")


def list_files(directory, extensions=None):
    if extensions is None:
        extensions = SUPPORT_EXTENSION
    p = Path(directory)
    if p.is_dir():
        files = [f.name for f in p.iterdir() if f.suffix.lower() in extensions and f.is_file()]
        return sorted(files)
    else:
        return []


def render_sidebar():
    import torch
    with st.sidebar:
        st.markdown("### CellSeg1")
        st.caption("Cell Segmentation · SAM + LoRA")
        st.divider()

        if torch.cuda.is_available():
            st.success("CUDA GPU")
        elif torch.backends.mps.is_available():
            st.success("Apple Silicon (MPS)")
        else:
            st.warning("CPU mode — inference will be slow")

        st.divider()
        st.markdown('<p class="sec-label">Files</p>', unsafe_allow_html=True)
        loras      = list_files(LORA_PTH_DIR, extensions=[".pth"])
        train_imgs = list_files(TRAIN_IMAGE_DIR)
        test_imgs  = list_files(TEST_IMAGE_DIR)
        preds      = list_files(PRED_MASK_DIR)

        st.caption(f"LoRA checkpoints: **{len(loras)}**")
        if train_imgs:
            st.caption(f"Training images: **{len(train_imgs)}**")
        if test_imgs:
            st.caption(f"Test images: **{len(test_imgs)}**")
        if preds:
            st.caption(f"Predicted masks: **{len(preds)}**")

        st.divider()
        st.markdown('<p class="sec-label">Workflow</p>', unsafe_allow_html=True)
        st.caption("1  **Train** — fine-tune on your image")
        st.caption("2  **Predict** — run inference")
        st.caption("3  **Visualize** — inspect results")


def initialize_session_state():
    defaults = {
        "train_start_time": None,
        "predict_start_time": None,
        # Train — advanced settings (keyed so presets can override them)
        "train_selected_sam_type": "vit_h",
        "train_sam_image_size": 512,
        "train_lora_rank": 4,
        "train_resize_size": 512,
        "train_patch_size": 256,
        "train_random_seed": 0,
        "train_base_lr": 0.003,
        "train_epoch_max": 300,
        "train_batch_size": 1,
        "train_gradient_accumulation": 32,
        # Visualize — file selection (persistent)
        "vis_sel_image": "None",
        "vis_sel_mask": "None",
        "vis_sel_pred": "None",
        "vis_figure": None,
        # Visualize — display settings
        "vis_n_cols": 3,
        "vis_sync_axes": True,
        "vis_display_mode": "contour",
        "vis_show_true_mask": True,
        "vis_show_pred_mask": True,
        "vis_show_error_map": True,
        "vis_overlay_alpha": 0.5,
        "vis_true_mask_color": "#FFFF00",
        "vis_pred_mask_color": "#64FF64",
        "vis_error_false_positive_color": "#64FF64",
        "vis_error_false_negative_color": "#FF6464",
        "vis_true_mask_line_width": 1,
        "vis_pred_mask_line_width": 1,
        "vis_true_mask_line_type": "solid",
        "vis_pred_mask_line_type": "solid",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
