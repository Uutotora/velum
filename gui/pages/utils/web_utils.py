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


def get_available_devices():
    devices = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if result.stdout.strip():
            gpu_info = result.stdout.strip().split("\n")
            gpu_list = [info.split(",")[0].strip() for info in gpu_info]
            devices.extend(gpu_list)
        else:
             st.warning("nvidia-smi ran successfully but found no GPUs. Only CPU is available.")
    except FileNotFoundError:
        st.warning("nvidia-smi command not found. Assuming no GPUs are available. Only CPU is available.")
    except subprocess.CalledProcessError as e:
        st.warning(f"nvidia-smi failed with error: {e.stderr}. Assuming no GPUs are available. Only CPU is available.")
    except Exception as e:
        st.error(f"An unexpected error occurred while checking for GPUs: {e}. Only CPU is available.")

    devices.append("cpu")
    return devices


def list_files(directory, extensions=None):
    if extensions is None:
        extensions = SUPPORT_EXTENSION
    p = Path(directory)
    if p.is_dir():
        files = [f.name for f in p.iterdir() if f.suffix.lower() in extensions and f.is_file()]
        return sorted(files)
    else:
        return []


def initialize_session_state():
    if "train_start_time" not in st.session_state:
        st.session_state["train_start_time"] = None
    if "predict_start_time" not in st.session_state:
        st.session_state["predict_start_time"] = None
    """Initialize session state variables."""
    defaults = {
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
