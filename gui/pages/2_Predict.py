import io
import multiprocessing
import time
import zipfile
from datetime import datetime, timedelta

import psutil
import streamlit as st

from gui.pages.utils.predict_state_manager import PredictionStateManager
from gui.pages.utils.web_utils import (
    LORA_PTH_DIR,
    PRED_MASK_DIR,
    SUPPORT_EXTENSION,
    TEST_IMAGE_DIR,
    delete_file,
    get_available_devices,
    get_sam_model_path,
    initialize_session_state,
    list_files,
    load_default_config,
)
from project_root import STORAGE_DIR


def main():
    initialize_session_state()
    st.set_page_config(page_title="Prediction", layout="wide")
    st.title("Prediction")

    default_config = load_default_config()

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            image_files = st.file_uploader(
                "Upload Images for Prediction",
                accept_multiple_files=True,
                type=SUPPORT_EXTENSION,
                key="predict_image_files",
            )
        with col2:
            lora_file = st.file_uploader(
                "Upload LoRA Model (.pth) File",
                type=["pth"],
                key="predict_lora_file",
            )
        if image_files:
            TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            for uploaded_file in image_files:
                with open(TEST_IMAGE_DIR / uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success("Test images uploaded successfully.")

        if lora_file:
            lora_pth_path = LORA_PTH_DIR / lora_file.name
            with open(lora_pth_path, "wb") as f:
                f.write(lora_file.getbuffer())
            st.success("LoRA model uploaded successfully.")
        else:
            lora_pth_path = None

        col1, col2 = st.columns(2)
        with col1:
            existing_test_images = list_files(TEST_IMAGE_DIR)
            with st.expander(f"Existing {len(existing_test_images)} Files:", expanded=True):
                if existing_test_images:
                    for img in existing_test_images:
                        col_img, col_del = st.columns([4, 1])
                        with col_img:
                            st.text(img)
                        with col_del:
                            delete_button = st.button(  # noqa: F841
                                "Delete",
                                key=f"delete_test_image_{img}",
                                on_click=lambda img=img: delete_file(TEST_IMAGE_DIR, img),
                            )
                else:
                    st.info("No test images uploaded yet.")
        with col2:
            existing_lora_files = list_files(LORA_PTH_DIR, extensions=[".pth"])
            with st.expander(f"Existing {len(existing_lora_files)} Files:", expanded=True):
                if existing_lora_files:
                    for lora in existing_lora_files:
                        col_lora, col_del = st.columns([4, 1])
                        with col_lora:
                            st.text(lora)
                        with col_del:
                            delete_button = st.button(  # noqa: F841
                                "Delete",
                                key=f"delete_lora_file_{lora}",
                                on_click=lambda lora=lora: delete_file(LORA_PTH_DIR, lora),
                            )
                else:
                    st.info("No LoRA models uploaded yet.")

    with st.expander("Advanced Settings"):
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        with col1:
            st.selectbox(
                "Select SAM Type",
                options=["vit_h", "vit_l", "vit_b"],
                key="predict_selected_sam_type",
                help="Need to be the same as the training",
            )
        with col2:
            sam_image_size = st.number_input(
                "SAM Image Size",
                min_value=64,
                max_value=1024,
                value=512,
                step=64,
                help="Need to be the same as the training",
            )
        with col3:
            lora_rank = st.number_input(
                "LoRA Rank",
                min_value=1,
                max_value=30,
                value=4,
                step=1,
                help="Need to be the same as the training",
            )
        with col4:
            points_per_side = st.number_input(
                "Points per Side",
                min_value=0,
                max_value=64,
                value=default_config["points_per_side"],
            )
        with col5:
            crop_n_layers = st.number_input(
                "Crop Layers",
                min_value=0,
                max_value=5,
                value=default_config["crop_n_layers"],
            )
        with col6:
            iou_threshold = st.number_input(
                "IoU Threshold",
                min_value=0.0,
                max_value=0.95,
                step=0.05,
                value=default_config["pred_iou_thresh"],
                help="If many false positives, increase this. If many false negatives, decrease this.",
            )
        with col7:
            stability_threshold = st.number_input(
                "Stability Threshold",
                min_value=0.0,
                max_value=0.95,
                step=0.05,
                value=default_config["stability_score_thresh"],
                help="If many false positives, increase this. If many false negatives, decrease this.",
            )
        with col8:
            nms_thresh = st.number_input(
                "NMS Threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=default_config["box_nms_thresh"],
                help="Non-Maximum Suppression threshold for masks.",
            )
    st.header("Run")
    col1, col2, col3, col4, col5 = st.columns(5, vertical_alignment="bottom")

    available_lora_files = list_files(LORA_PTH_DIR, extensions=[".pth"])
    available_lora_files = list(reversed(available_lora_files))

    state_manager = PredictionStateManager(STORAGE_DIR)
    running_state = state_manager.load_prediction_state()
    if running_state:
        pid = running_state["process_id"]
        if psutil.pid_exists(pid):
            is_prediction_running = True
        else:
            state_manager.clear_prediction_state()
            is_prediction_running = False
            st.success("Prediction is complete.")
    else:
        is_prediction_running = False

    with col1:
        if available_lora_files:
            selected_lora = st.selectbox("Select LoRA to use", options=available_lora_files, key="selected_lora")
        else:
            selected_lora = None
            st.selectbox("Select LoRA to use", options=["No LoRA models available"], disabled=True)

    with col2:
        device_options = get_available_devices()
        if len(device_options) > 1:
            selected_device = st.selectbox(
                "Select Device", options=device_options, help="Select CPU or GPU for prediction."
            )
        elif len(device_options) == 1 and device_options[0] == "cpu":
            st.info("No GPUs detected. CPU will be used for prediction.")
            selected_device = "cpu"
        else:
            st.error("Could not determine available devices. Please check system configuration.")
            selected_device = None

    with col3:
        start_button_disabled = is_prediction_running or (selected_device is None)
        start_button = st.button(
            "Start Prediction", type="primary", use_container_width=True, disabled=start_button_disabled
        )
    with col4:
        stop_button = st.button(
            "Stop Prediction", type="secondary", use_container_width=True, disabled=not is_prediction_running
        )
    with col5:
        if PRED_MASK_DIR.exists() and any(PRED_MASK_DIR.iterdir()):
            st.download_button(
                label="Download",
                data=create_zip_file(PRED_MASK_DIR),
                file_name="predicted_masks.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.button("Download", disabled=True, use_container_width=True)

    if start_button:
        images = list_files(TEST_IMAGE_DIR)
        if validate_predict_inputs(images, available_lora_files):
            if selected_lora:
                selected_lora_path = LORA_PTH_DIR / selected_lora
            else:
                selected_lora_path = None

            sam_type = st.session_state.get("predict_selected_sam_type", "vit_b")
            config = prepare_config(
                default_config,
                str(selected_lora_path),
                points_per_side,
                crop_n_layers,
                0,
                0,
                sam_type,
                sam_image_size,
                iou_threshold,
                stability_threshold,
                lora_rank,
                nms_thresh,
            )
            config["selected_device"] = selected_device
            config["image_paths"] = [str(TEST_IMAGE_DIR / img) for img in images]

            if PRED_MASK_DIR.exists():
                for file in PRED_MASK_DIR.glob("*"):
                    file.unlink()
            else:
                PRED_MASK_DIR.mkdir(parents=True, exist_ok=True)
            config["output_dir"] = PRED_MASK_DIR

            process = multiprocessing.Process(
                target=run_prediction,
                args=(config, state_manager),
            )
            process.start()

            state_manager.save_prediction_state(process.pid, datetime.now(), len(images))
            st.success("Prediction has started.")
            st.rerun()
        else:
            st.error("Validation failed. Please check your inputs.")

    if stop_button:
        if is_prediction_running:
            state_manager.set_stop_flag()
            st.warning("Prediction will be stopped shortly.")
        else:
            st.info("No prediction process is running.")

    if is_prediction_running:
        pid = running_state["process_id"]
        if psutil.pid_exists(pid):
            progress_data = state_manager.load_progress()
            progress_value = progress_data.get("progress", 0)
            total_images = running_state.get("total_images", 1)
            current_image = min(progress_value + 1, total_images)
            progress_fraction = progress_value / total_images
            progress_percentage = progress_fraction * 100

            _ = st.progress(progress_fraction)

            elapsed_time = datetime.now() - running_state["start_time"]

            if progress_value > 0:
                time_per_image = elapsed_time / progress_value
                remaining_images = total_images - progress_value
                estimated_remaining_time = time_per_image * remaining_images
                estimated_remaining_time = timedelta(seconds=int(estimated_remaining_time.total_seconds()))
            else:
                estimated_remaining_time = "Calculating..."

            st.text(
                f"Predicting: {current_image}/{total_images} images | "
                f"Progress: {progress_percentage:.2f}% | "
                f"Elapsed Time: {str(elapsed_time).split('.')[0]} | "
                f"Estimated Remaining Time: {estimated_remaining_time}"
            )
            time.sleep(1)
            st.rerun()
        else:
            state_manager.clear_prediction_state()
            st.success("Prediction is complete.")
    else:
        st.info("No prediction process is running.")


def create_zip_file(directory):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in directory.glob("*"):
            zip_file.write(file, file.name)
    zip_buffer.seek(0)
    return zip_buffer


def validate_predict_inputs(images, available_lora_files):
    if len(images) == 0:
        st.error("No prediction images uploaded.")
        return False
    if not available_lora_files:
        st.error("Please upload a LoRA model file.")
        return False
    return True


def prepare_config(
    default_config,
    lora_pth_path,
    points_per_side,
    crop_n_layers,
    resize_width,
    resize_height,
    sam_type,
    sam_image_size,
    iou_threshold,
    stability_threshold,
    lora_rank,
    nms_thresh,
):
    config = default_config.copy()
    config.update(
        {
            "result_pth_path": lora_pth_path,
            "points_per_side": points_per_side,
            "crop_n_layers": crop_n_layers,
            "resize_size": [resize_width, resize_height] if resize_width > 0 and resize_height > 0 else None,
            "vit_name": sam_type,
            "model_path": get_sam_model_path(sam_type),
            "sam_image_size": sam_image_size,
            "iou_thresh": iou_threshold,
            "stability_score_thresh": stability_threshold,
            "image_encoder_lora_rank": lora_rank,
            "mask_decoder_lora_rank": lora_rank,
            "box_nms_thresh": nms_thresh,
            "pred_iou_thresh": nms_thresh,
        }
    )
    return config


def run_prediction(config, state_manager):
    import os

    selected_device = config.get("selected_device", "cpu")
    if selected_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("INFO: Running prediction on CPU.")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_device
        print(f"INFO: Running prediction on GPU {selected_device}.")
    from pathlib import Path

    import cv2

    from data.utils import read_image_to_numpy, resize_image
    from predict import predict_images

    images = [read_image_to_numpy(image_path) for image_path in config["image_paths"]]
    if (
        config.get("resize_size") and config["resize_size"][0] > 0 and config["resize_size"][1] > 0
    ):
        images = [resize_image(img, config["resize_size"]) for img in images]

    pred_masks = []
    save_pred = True
    try:
        for idx, img in enumerate(images):
            if state_manager.check_stop_flag():
                save_pred = False
                break

            mask = predict_images(config, [img])[0]
            pred_masks.append(mask)
            state_manager.save_progress(idx + 1)

        if save_pred:
            output_dir = config["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)

            for image_path, mask in zip(config["image_paths"], pred_masks):
                output_path = output_dir / Path(image_path).name
                cv2.imwrite(str(output_path.with_suffix(".png")), mask)
    finally:
        state_manager.clear_prediction_state()


if __name__ == "__main__":
    main()
