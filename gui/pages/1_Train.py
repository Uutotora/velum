import multiprocessing
import time
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
import psutil
import streamlit as st

from data.dataset import TrainDataset
from gui.pages.utils.train_model import train_model
from gui.pages.utils.train_state_manager import TrainingStateManager
from gui.pages.utils.web_utils import (
    DEVICE_LABELS,
    LORA_PTH_DIR,
    SUPPORT_EXTENSION,
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    delete_file,
    get_available_devices,
    get_sam_model_path,
    inject_css,
    initialize_session_state,
    list_files,
    load_default_config,
    render_sidebar,
    show_compact_file_list,
)
from project_root import STORAGE_DIR

# ── Presets ───────────────────────────────────────────────────────────────────

PRESETS = {
    "fast_mps": {
        "label": "Fast · MPS",
        "help": "150 epochs · batch 1 · grad accum 32 · rank 4 — quick iteration on Apple Silicon (~10 min)",
        "train_epoch_max": 150,
        "train_batch_size": 1,
        "train_gradient_accumulation": 32,
        "train_base_lr": 0.003,
        "train_lora_rank": 4,
        "train_selected_sam_type": "vit_h",
        "train_sam_image_size": 512,
        "train_resize_size": 512,
        "train_patch_size": 256,
        "train_random_seed": 0,
    },
    "balanced": {
        "label": "Balanced",
        "help": "300 epochs · batch 1 · grad accum 32 · rank 4 — recommended default (~20–30 min on MPS)",
        "train_epoch_max": 300,
        "train_batch_size": 1,
        "train_gradient_accumulation": 32,
        "train_base_lr": 0.003,
        "train_lora_rank": 4,
        "train_selected_sam_type": "vit_h",
        "train_sam_image_size": 512,
        "train_resize_size": 512,
        "train_patch_size": 256,
        "train_random_seed": 0,
    },
    "best_quality": {
        "label": "Best quality",
        "help": "500 epochs · rank 8 · SAM 1024 — maximum accuracy, ~2× slower and uses more memory",
        "train_epoch_max": 500,
        "train_batch_size": 1,
        "train_gradient_accumulation": 32,
        "train_base_lr": 0.001,
        "train_lora_rank": 8,
        "train_selected_sam_type": "vit_h",
        "train_sam_image_size": 1024,
        "train_resize_size": 1024,
        "train_patch_size": 512,
        "train_random_seed": 0,
    },
}


def apply_preset(preset_key: str) -> None:
    for k, v in PRESETS[preset_key].items():
        if k not in ("label", "help"):
            st.session_state[k] = v


# ── Loss chart ────────────────────────────────────────────────────────────────

def render_loss_chart(loss_history: list, epoch_max: int | None = None) -> None:
    epochs = [d["epoch"] for d in loss_history]
    losses = [d["loss"] for d in loss_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=losses,
        mode="lines",
        line=dict(color="#ef4444", width=2),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.07)",
        hovertemplate="Epoch %{x}<br>Loss %{y:.5f}<extra></extra>",
    ))
    if epoch_max:
        fig.update_xaxes(range=[1, epoch_max])
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=210,
        margin=dict(l=50, r=20, t=8, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        showlegend=False,
        xaxis=dict(gridcolor="#e2e8f0", zeroline=False),
        yaxis=dict(gridcolor="#e2e8f0", zeroline=False),
    )
    st.markdown('<p class="sec-label">Training loss</p>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if losses:
        min_l, last_l = min(losses), losses[-1]
        st.caption(
            f"Latest: **{last_l:.5f}** · Best: **{min_l:.5f}** · Epoch {epochs[-1]}"
            + (f" / {epoch_max}" if epoch_max else "")
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    initialize_session_state()
    st.set_page_config(page_title="Train — CellSeg1", layout="wide")
    inject_css()
    render_sidebar()

    st.markdown("## Train")
    st.caption("Fine-tune SAM with LoRA on a single annotated image · requires one image + instance mask")
    st.divider()

    # ── Data upload ───────────────────────────────────────────────────────────
    with st.container(border=True):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<p class="sec-label">Training image</p>', unsafe_allow_html=True)
            image_files = st.file_uploader(
                "Upload training images",
                accept_multiple_files=True,
                type=SUPPORT_EXTENSION,
                key="train_image_files",
                label_visibility="collapsed",
            )
            if image_files:
                TRAIN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                for f in image_files:
                    with open(TRAIN_IMAGE_DIR / f.name, "wb") as fp:
                        fp.write(f.getbuffer())
                st.success("Images saved.")
            existing_images = list_files(TRAIN_IMAGE_DIR)
            if existing_images:
                show_compact_file_list(TRAIN_IMAGE_DIR, existing_images, "del_train_img", show_preview=True)
            else:
                st.caption("No images loaded. Upload a microscopy image (PNG / TIF / JPG).")

        with col2:
            st.markdown('<p class="sec-label">Instance mask</p>', unsafe_allow_html=True)
            mask_files = st.file_uploader(
                "Upload training masks",
                accept_multiple_files=True,
                type=SUPPORT_EXTENSION,
                key="train_mask_files",
                label_visibility="collapsed",
            )
            if mask_files:
                TRAIN_MASK_DIR.mkdir(parents=True, exist_ok=True)
                for f in mask_files:
                    with open(TRAIN_MASK_DIR / f.name, "wb") as fp:
                        fp.write(f.getbuffer())
                st.success("Masks saved.")
            existing_masks = list_files(TRAIN_MASK_DIR)
            if existing_masks:
                show_compact_file_list(TRAIN_MASK_DIR, existing_masks, "del_train_msk", show_preview=True)
            else:
                st.caption("No masks loaded. Each pixel value should equal the cell instance ID (0 = background).")

    # ── Presets ───────────────────────────────────────────────────────────────
    st.markdown("")
    st.markdown('<p class="sec-label">Presets</p>', unsafe_allow_html=True)
    pc1, pc2, pc3, _spacer = st.columns([1, 1, 1, 3])
    with pc1:
        st.button(
            PRESETS["fast_mps"]["label"],
            on_click=apply_preset,
            args=("fast_mps",),
            use_container_width=True,
            help=PRESETS["fast_mps"]["help"],
        )
    with pc2:
        st.button(
            PRESETS["balanced"]["label"],
            on_click=apply_preset,
            args=("balanced",),
            use_container_width=True,
            help=PRESETS["balanced"]["help"],
        )
    with pc3:
        st.button(
            PRESETS["best_quality"]["label"],
            on_click=apply_preset,
            args=("best_quality",),
            use_container_width=True,
            help=PRESETS["best_quality"]["help"],
        )

    # ── Advanced settings ─────────────────────────────────────────────────────
    with st.expander("Advanced settings"):
        # Model
        st.markdown('<p class="sec-label">Model</p>', unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.selectbox(
                "SAM type",
                options=["vit_h", "vit_l", "vit_b"],
                key="train_selected_sam_type",
                help="vit_h (huge) gives the best results; vit_b (base) is fastest. Must match the type used at prediction time.",
            )
        with mc2:
            st.number_input(
                "LoRA rank",
                min_value=1, max_value=30,
                key="train_lora_rank",
                step=1,
                help="Adapter dimension — higher rank = more trainable parameters. Default 4 works well for most cases.",
            )
        with mc3:
            st.number_input(
                "SAM image size",
                min_value=64, max_value=1024,
                key="train_sam_image_size",
                step=64,
                help="Internal SAM resolution. Must match prediction time. Default 512.",
            )

        st.divider()

        # Data
        st.markdown('<p class="sec-label">Data</p>', unsafe_allow_html=True)
        dc1, dc2 = st.columns(2)
        with dc1:
            st.number_input(
                "Resize size",
                min_value=0, max_value=2048,
                key="train_resize_size",
                step=64,
                help="Resize the image before patching. 0 = no resize. Pipeline: Image → Resize → Patches → SAM size.",
            )
        with dc2:
            st.number_input(
                "Patch size",
                min_value=0, max_value=1024,
                key="train_patch_size",
                step=64,
                help="Patch size for sliding window extraction (50% overlap). Should be ≤ resize size.",
            )

        st.divider()

        # Training
        st.markdown('<p class="sec-label">Training</p>', unsafe_allow_html=True)
        tc1, tc2, tc3, tc4, tc5 = st.columns(5, vertical_alignment="bottom")
        with tc1:
            epoch_max = st.number_input(
                "Epochs",
                min_value=1, max_value=10000,
                key="train_epoch_max",
                step=50,
                help="Number of training epochs. 300 is the recommended default.",
            )
        with tc2:
            base_lr = st.number_input(
                "Learning rate",
                min_value=0.00001, max_value=0.3,
                key="train_base_lr",
                step=0.0001,
                format="%.5f",
                help="Peak learning rate for OneCycleLR scheduler.",
            )
        with tc3:
            batch_size = st.number_input(
                "Batch size",
                min_value=1, max_value=128,
                key="train_batch_size",
                step=1,
                help="Samples per gradient update step.",
            )
        with tc4:
            gradient_accumulation_step = st.number_input(
                "Grad. accumulation",
                min_value=1, max_value=128,
                key="train_gradient_accumulation",
                step=1,
                help="Accumulate gradients over N steps before updating weights.",
            )
        with tc5:
            random_seed = st.number_input(
                "Random seed",
                min_value=0, max_value=1_000_000,
                key="train_random_seed",
                step=1,
            )

        eff = batch_size * gradient_accumulation_step
        st.info(
            f"Effective batch size = {batch_size} × {gradient_accumulation_step} = **{eff}**  "
            f"— 8 GB VRAM: try 1×32  ·  12 GB: 2×16  ·  24 GB: 4×8",
            icon="ℹ️",
        )

    # ── State manager ─────────────────────────────────────────────────────────
    state_manager = TrainingStateManager(STORAGE_DIR)
    running_state = state_manager.load_training_state()
    if running_state:
        pid = running_state["process_id"]
        if not psutil.pid_exists(pid):
            state_manager.clear_training_state()
            running_state = None

    is_running = running_state is not None

    # ── Run ───────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="sec-label">Run</p>', unsafe_allow_html=True)

    device_options = get_available_devices()
    rc1, rc2, rc3, rc4 = st.columns([3, 2, 2, 3], vertical_alignment="bottom")
    with rc1:
        selected_device = st.selectbox(
            "Device",
            options=device_options,
            format_func=lambda d: DEVICE_LABELS.get(d, f"GPU {d}"),
            help="Select compute device for training.",
        )
    with rc2:
        st.button(
            "Start Training",
            key="train_start_button",
            type="primary",
            use_container_width=True,
            disabled=is_running or selected_device is None,
        )
    with rc3:
        st.button(
            "Stop",
            key="train_stop_button",
            type="secondary",
            use_container_width=True,
            disabled=not is_running,
        )

    # Running banner
    if is_running:
        st.info(
            "Training is running in the background. Safe to close the browser — progress is preserved.",
            icon="ℹ️",
        )

    # Start / stop button actions
    if st.session_state.get("train_start_button"):
        images = list_files(TRAIN_IMAGE_DIR)
        masks = list_files(TRAIN_MASK_DIR)
        if validate_inputs(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, images, masks):
            sam_type = st.session_state.get("train_selected_sam_type", "vit_h")
            config = prepare_config(
                epoch_max=st.session_state["train_epoch_max"],
                base_lr=st.session_state["train_base_lr"],
                batch_size=st.session_state["train_batch_size"],
                gradient_accumulation_step=st.session_state["train_gradient_accumulation"],
                image_dir=TRAIN_IMAGE_DIR,
                mask_dir=TRAIN_MASK_DIR,
                sam_type=sam_type,
                random_seed=st.session_state["train_random_seed"],
                sam_image_size=st.session_state["train_sam_image_size"],
                patch_size=st.session_state["train_patch_size"],
                resize_size=st.session_state["train_resize_size"],
                lora_rank=st.session_state["train_lora_rank"],
            )
            config["selected_device"] = selected_device
            config["train_id"] = list(range(len(images)))

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            lora_path = (
                STORAGE_DIR / "loras"
                / f"lora_{sam_type}_{st.session_state['train_lora_rank']}_{st.session_state['train_sam_image_size']}_{timestamp}.pth"
            )
            config["result_pth_path"] = str(lora_path)

            process = multiprocessing.Process(target=train_model, args=(config, state_manager))
            process.start()
            state_manager.save_training_state(process.pid, datetime.now())
            st.success("Training started.")
            st.rerun()
        else:
            st.error("Check inputs before starting.")

    if st.session_state.get("train_stop_button"):
        if running_state:
            state_manager.set_stop_flag()
            st.warning("Stop signal sent — training will halt after the current epoch.")
        else:
            st.info("No training is running.")

    # ── Progress + live loss chart ────────────────────────────────────────────
    if is_running:
        pid = running_state["process_id"]
        if psutil.pid_exists(pid):
            progress_data = state_manager.load_progress()
            current_epoch = progress_data.get("current_epoch", 0)
            progress_pct = progress_data.get("progress", 0)
            elapsed = datetime.now() - running_state["start_time"]

            if current_epoch > 0 and progress_pct > 0:
                time_per_epoch = elapsed / current_epoch
                remaining = epoch_max - current_epoch
                eta = timedelta(seconds=int((time_per_epoch * remaining).total_seconds()))
            else:
                eta = "—"

            st.progress(progress_pct / 100)
            m1, m2, m3 = st.columns(3)
            m1.metric("Epoch", f"{current_epoch} / {epoch_max}")
            m2.metric("Elapsed", str(elapsed).split(".")[0])
            m3.metric("ETA", str(eta))

            loss_history = state_manager.load_loss_history()
            if loss_history:
                render_loss_chart(loss_history, epoch_max=epoch_max)

            time.sleep(1)
            st.rerun()
        else:
            state_manager.clear_training_state()
            st.rerun()
    else:
        # Show loss chart from completed run
        loss_history = state_manager.load_loss_history()
        if loss_history:
            render_loss_chart(loss_history, epoch_max=epoch_max)

    # ── Output — download ─────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="sec-label">Output</p>', unsafe_allow_html=True)
    lora_files = list(reversed(list_files(LORA_PTH_DIR, extensions=[".pth"])))

    if lora_files:
        dl1, dl2, _spacer = st.columns([3, 2, 4], vertical_alignment="bottom")
        with dl1:
            selected_lora = st.selectbox(
                "Select LoRA checkpoint",
                options=lora_files,
                label_visibility="collapsed",
            )
        with dl2:
            with open(LORA_PTH_DIR / selected_lora, "rb") as f:
                st.download_button(
                    label="Download .pth",
                    data=f,
                    file_name=selected_lora,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        st.caption(f"Saved: `streamlit_storage/loras/{selected_lora}`")
    else:
        st.caption("No trained models yet. Run training to generate a LoRA checkpoint.")

    # ── Training history ──────────────────────────────────────────────────────
    st.divider()
    history = state_manager.load_history()
    with st.expander(f"Training history  ({len(history)} run{'s' if len(history) != 1 else ''})",
                     expanded=False):
        if history:
            import pandas as pd
            rows = []
            for h in history:
                ts = h.get("started_at", "")[:19].replace("T", " ")
                fl = h.get("final_loss")
                ep = h.get("epochs_run", 0)
                ep_max = h.get("epoch_max", "")
                ckpt = Path(h.get("checkpoint", "")).name
                rows.append({
                    "Date": ts,
                    "SAM": h.get("sam_type", ""),
                    "Rank": h.get("lora_rank", ""),
                    "Epochs": f"{ep} / {ep_max}",
                    "Final loss": f"{fl:.5f}" if fl is not None else "—",
                    "Status": h.get("status", ""),
                    "Checkpoint": ckpt,
                })
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Final loss": st.column_config.NumberColumn("Final loss", format="%.5f"),
                },
            )
        else:
            st.caption("No runs recorded yet. History is saved automatically when training completes.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_inputs(image_dir, mask_dir, images, masks):
    if not Path(image_dir).exists():
        st.error("Image directory does not exist.")
        return False
    if not Path(mask_dir).exists():
        st.error("Mask directory does not exist.")
        return False
    if len(images) == 0:
        st.error("No training images uploaded.")
        return False
    if len(masks) == 0:
        st.error("No training masks uploaded.")
        return False
    if len(images) != len(masks):
        st.error(f"Image / mask count mismatch: {len(images)} images vs {len(masks)} masks.")
        return False
    return True


def prepare_config(
    epoch_max, base_lr, batch_size, gradient_accumulation_step,
    image_dir, mask_dir, sam_type, random_seed,
    sam_image_size, patch_size, resize_size, lora_rank,
):
    config = load_default_config()
    config["epoch_max"] = epoch_max
    config["base_lr"] = base_lr
    config["batch_size"] = batch_size
    config["gradient_accumulation_step"] = gradient_accumulation_step
    config["train_image_dir"] = str(image_dir)
    config["train_mask_dir"] = str(mask_dir)
    config["resize_size"] = [resize_size, resize_size]
    config["data_dir"] = ""
    config["vit_name"] = sam_type
    config["seed"] = random_seed
    config["model_path"] = get_sam_model_path(sam_type)
    config["sam_image_size"] = sam_image_size
    config["patch_size"] = patch_size
    config["image_encoder_lora_rank"] = lora_rank
    config["mask_decoder_lora_rank"] = lora_rank
    return config


def load_dataset(config):
    return TrainDataset(
        image_dir=config["train_image_dir"],
        mask_dir=config["train_mask_dir"],
        resize_size=config["resize_size"],
        patch_size=config["patch_size"],
        train_id=config["train_id"],
        duplicate_data=config["duplicate_data"],
    )


if __name__ == "__main__":
    main()
