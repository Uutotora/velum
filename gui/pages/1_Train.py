import multiprocessing
import time
from datetime import datetime, timedelta
from pathlib import Path

import psutil
import streamlit as st

from data.dataset import TrainDataset
from gui.pages.utils.train_model import train_model
from gui.pages.utils.train_state_manager import TrainingStateManager
from gui.pages.utils.web_utils import (
    LORA_PTH_DIR,
    SUPPORT_EXTENSION,
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    delete_file,
    get_available_devices, # Changed from get_available_gpus
    get_sam_model_path,
    initialize_session_state,
    list_files,
    load_default_config,
)
from project_root import STORAGE_DIR


def main():
    initialize_session_state()
    st.set_page_config(page_title="Train", layout="wide")
    st.title("Train")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            image_files = st.file_uploader(
                "Upload Training Images",
                accept_multiple_files=True,
                type=SUPPORT_EXTENSION,
                key="train_image_files",
            )
        with col2:
            mask_files = st.file_uploader(
                "Upload Training Masks",
                accept_multiple_files=True,
                type=SUPPORT_EXTENSION,
                key="train_mask_files",
            )

        if image_files:
            TRAIN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            for uploaded_file in image_files:
                with open(TRAIN_IMAGE_DIR / uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success("Training images uploaded successfully.")

        if mask_files:
            TRAIN_MASK_DIR.mkdir(parents=True, exist_ok=True)
            for uploaded_file in mask_files:
                with open(TRAIN_MASK_DIR / uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success("Training masks uploaded successfully.")
        col1, col2 = st.columns(2)
        with col1:
            existing_images = list_files(TRAIN_IMAGE_DIR)
            with st.expander(f"Existing {len(existing_images)} Files:", expanded=True):
                if existing_images:
                    for img in existing_images:
                        col_img, col_del = st.columns([3, 1])
                        with col_img:
                            st.text(img)
                        with col_del:
                            delete_button = st.button(
                                "Delete",
                                key=f"delete_train_image_{img}",
                                on_click=lambda img=img: delete_file(TRAIN_IMAGE_DIR, img),
                            )
                else:
                    st.info("No training images uploaded yet.")
        with col2:
            existing_masks = list_files(TRAIN_MASK_DIR)
            with st.expander(f"Existing {len(existing_masks)} Files:", expanded=True):
                if existing_masks:
                    for msk in existing_masks:
                        col_msk, col_del = st.columns([3, 1])
                        with col_msk:
                            st.text(msk)
                        with col_del:
                            delete_button = st.button(  # noqa: F841
                                "Delete",
                                key=f"delete_train_mask_{msk}",
                                on_click=lambda msk=msk: delete_file(TRAIN_MASK_DIR, msk),
                            )
                else:
                    st.info("No training masks uploaded yet.")

    with st.expander("Advanced Settings"):
        col1, col2, col3, col4, col5 = st.columns(5, vertical_alignment="bottom")
        with col1:
            st.selectbox(
                "Select SAM Type",
                options=["vit_h", "vit_l", "vit_b"],
                key="train_selected_sam_type",
                help="huge > large > base",
            )
        with col2:
            sam_image_size = st.number_input(
                "SAM Image Size",
                min_value=64,
                max_value=1024,
                value=512,
                step=64,
                help="This is the input size of SAM.",
            )
        with col3:
            lora_rank = st.number_input(
                "LoRA Rank", min_value=1, max_value=30, value=4, step=1, help="This apply to both encoder and decoder."
            )
        with col4:
            resize_size = st.number_input(
                "Resize Size",
                min_value=0,
                max_value=1024,
                value=512,
                step=64,
                help="0 represent not resize. Image -> Resize image by Resize Size -> Slice patch by Patch Size - > Resize patch by SAM Image Size",
            )
        with col5:
            patch_size = st.number_input(
                "Patch Size",
                min_value=0,
                max_value=512,
                value=256,
                step=64,
                help="50% overlap between patches.",
            )
        col6, col7, col8, col9, col10 = st.columns(5, vertical_alignment="bottom")
        with col6:
            random_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=1000000,
                value=0,
                step=1,
            )
        with col7:
            base_lr = st.number_input(
                "Learning Rate",
                min_value=0.00001,
                max_value=0.3,
                value=0.003,
                step=0.0001,
                format="%.5f",
            )
        with col8:
            epoch_max = st.number_input(
                "Training Epochs",
                min_value=1,
                max_value=10000,
                value=300,
                step=50,
                help="In rare cases, training with large amounts of poorly labeled images for extended periods may cause model collapse. If this occurs, try reducing this value or using a different random seed.",
            )
        with col9:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=128,
                value=1,
                step=1,
                help="The effective batch size is equal to the batch size multiplied by the number of gradient accumulation steps.",
            )
        with col10:
            gradient_accumulation_step = st.number_input(
                "Gradient Accumulation", min_value=1, max_value=128, value=32, step=1
            )

    st.header("Run")

    state_manager = TrainingStateManager(STORAGE_DIR)
    running_state = state_manager.load_training_state()
    if running_state:
        pid = running_state["process_id"]
        if psutil.pid_exists(pid):
            pass
        else:
            state_manager.clear_training_state()
            st.success("Training is complete.")
            running_state = None
    if running_state:
        start_disabled = True
        stop_disabled = False
    else:
        start_disabled = False
        stop_disabled = True
    col1, col2, col3, col4, col5 = st.columns([3, 3, 3, 3, 3], gap="small", vertical_alignment="bottom")
    with col1:
        device_options = get_available_devices()
        if len(device_options) > 1:
             selected_device = st.selectbox("Select Device", options=device_options, help="Select CPU or GPU for training.")
        elif len(device_options) == 1 and device_options[0] == 'cpu':
             st.info("No GPUs detected. CPU will be used for training.")
             selected_device = 'cpu'
        else:
             st.error("Could not determine available devices. Please check system configuration.")
             selected_device = None

    with col2:
        # Disable start if no device could be selected
        start_button_disabled = start_disabled or (selected_device is None)
        start_button = st.button(
            "Start Training",
            key="train_start_button",
            help="Click to start training",
            type="primary",
            use_container_width=True,
            disabled=start_button_disabled,
        )
    with col3:
        stop_button = st.button(
            "Stop Training",
            key="train_stop_button",
            help="Click to stop training",
            type="secondary",
            use_container_width=True,
            disabled=stop_disabled,
        )
    with col4:
        lora_files = reversed(list_files(LORA_PTH_DIR, extensions=[".pth"]))
        selected_lora = st.selectbox(
            "Select LoRA file",
            options=lora_files,
            index=0 if lora_files else None,
        )
    with col5:
        if selected_lora:
            with open(LORA_PTH_DIR / selected_lora, "rb") as f:
                st.download_button(
                    label="Download",
                    data=f,
                    file_name=selected_lora,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        else:
            st.button("Download", help="No LoRA file available", disabled=True, use_container_width=True)

    if start_button:
        images = list_files(TRAIN_IMAGE_DIR)
        masks = list_files(TRAIN_MASK_DIR)
        if validate_inputs(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, images, masks):
            sam_type = st.session_state.get("train_selected_sam_type", "vit_h")
            config = prepare_config(
                epoch_max,
                base_lr,
                batch_size,
                gradient_accumulation_step,
                TRAIN_IMAGE_DIR,
                TRAIN_MASK_DIR,
                sam_type,
                random_seed,
                sam_image_size,
                patch_size,
                resize_size,
                lora_rank,
            )
            # Pass the selected device to the config
            config["selected_device"] = selected_device
            config["train_id"] = list(range(len(images)))

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            lora_path = STORAGE_DIR / "loras" / f"lora_{sam_type}_{lora_rank}_{sam_image_size}_{timestamp}.pth"
            config["result_pth_path"] = str(lora_path)

            process = multiprocessing.Process(
                target=train_model,
                args=(config, state_manager),
            )
            process.start()

            state_manager.save_training_state(process.pid, datetime.now())
            st.success("Training has started.")
            st.rerun()
        else:
            st.error("Validation failed. Please check your inputs.")

    if stop_button:
        if running_state:
            state_manager.set_stop_flag()
            st.warning("Training will be stopped shortly.")
        else:
            st.info("No training process is running.")

    if running_state:
        pid = running_state["process_id"]
        if psutil.pid_exists(pid):
            progress_data = state_manager.load_progress()
            progress_value = progress_data.get("progress", 0)
            current_epoch = progress_data.get("current_epoch", 0)

            _ = st.progress(progress_value / 100)
            elapsed_time = datetime.now() - running_state["start_time"]

            if current_epoch > 0 and progress_value > 0:
                time_per_epoch = elapsed_time / current_epoch
                remaining_epochs = epoch_max - current_epoch
                estimated_remaining_time = time_per_epoch * remaining_epochs
                estimated_remaining_time = timedelta(seconds=int(estimated_remaining_time.total_seconds()))
            else:
                estimated_remaining_time = "Calculating..."

            st.text(
                f"Training Epoch: {current_epoch}/{epoch_max} | "
                f"Elapsed Time: {str(elapsed_time).split('.')[0]} | "
                f"Estimated Remaining Time: {estimated_remaining_time}"
            )
            time.sleep(1)
            st.rerun()
        else:
            state_manager.clear_training_state()
            st.success("Training is complete.")
    else:
        st.info("No training process is running.")


def validate_inputs(image_dir, mask_dir, images, masks):
    if not Path(image_dir).exists():
        st.error("The specified image directory does not exist.")
        return False
    if not Path(mask_dir).exists():
        st.error("The specified mask directory does not exist.")
        return False
    if len(images) == 0:
        st.error("No training images uploaded.")
        return False
    if len(masks) == 0:
        st.error("No training masks uploaded.")
        return False
    if len(images) != len(masks):
        st.error("The number of training images and masks must be the same.")
        return False
    return True


def prepare_config(
    epoch_max,
    base_lr,
    batch_size,
    gradient_accumulation_step,
    image_dir,
    mask_dir,
    sam_type,
    random_seed,
    sam_image_size,
    patch_size,
    resize_size,
    lora_rank,
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
    train_dataset = TrainDataset(
        image_dir=config["train_image_dir"],
        mask_dir=config["train_mask_dir"],
        resize_size=config["resize_size"],
        patch_size=config["patch_size"],
        train_id=config["train_id"],
        duplicate_data=config["duplicate_data"],
    )
    return train_dataset


if __name__ == "__main__":
    main()
